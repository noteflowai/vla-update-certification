"""Read-only semantic audit of completed native sides, including partial cohorts.

This analysis is independent of the producer's result validator. It never
repairs, scores an infrastructure error, or promotes a partial collection.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def descriptor(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": sha256(array.tobytes()).hexdigest()}


def initial_observation(archive, tree, prefix="observation"):
    for key, value in tree.items():
        path = prefix + "/" + key
        if set(value) == {"shape", "dtype", "sha256"}:
            assert descriptor(archive[path][0]) == value, path
        else:
            initial_observation(archive, value, path)


def audit_side(folder, context):
    attempt = json.loads((folder / "attempt.json").read_text())
    assert attempt["status"] == "completed", folder
    assert digest(folder / "episode.json") == attempt["episode_sha256"]
    episode = json.loads((folder / "episode.json").read_text())
    assert type(episode["success"]) is bool and episode["infrastructure_error"] is None
    assert episode["raw_folder"] == str(folder.resolve())
    identity, side = attempt["identity"], attempt["side"]
    assert episode["pipeline_sha256"] == context[side + "_pipeline_sha256"]
    assert episode["evaluator_sha256"] == context["evaluator_sha256"]
    for key in ("task", "init_state", "scene_seed", "policy_seed"):
        assert episode[key] == identity[key]
    for relative, expected in episode["raw_evidence"].items():
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        assert digest(folder / path) == expected, path
    reset0 = json.loads((folder / "reset-0.json").read_text())
    reset1 = json.loads((folder / "reset-1.json").read_text())
    assert reset0 == reset1 and reset1["actual_success"] is False
    assert reset1["prompt"] == reset1["bddl_prompt"] and reset1["goals"]
    # The producer's documented canonical encoding uses Python's default separators.
    encoded = json.dumps(reset1, sort_keys=True, allow_nan=False).encode()
    assert sha256(encoded).hexdigest() == episode["initial_input_sha256"]
    with np.load(folder / "input-000.npz", allow_pickle=False) as initial:
        assert descriptor(initial["simulator_state"]) == reset1["state"]
        assert reset1["state"]["sha256"] == episode["initial_state_sha256"]
        initial_observation(initial, reset1["observation"])
    transitions = [json.loads(line) for line in
                   (folder / "transitions.jsonl").read_text().splitlines() if line.strip()]
    steps = episode["steps"]
    assert 0 < steps <= 520 and len(transitions) == steps
    assert len(list(folder.glob("input-*.npz"))) == steps
    assert len(list(folder.glob("action-*.json"))) == steps
    actions = []
    for index, row in enumerate(transitions):
        assert row["step"] == index
        assert row["goal_values"] and row["success"] == row["actual_success"] == all(row["goal_values"])
        assert isinstance(row["terminated"], bool) and isinstance(row["truncated"], bool)
        if index < steps - 1:
            assert not (row["terminated"] or row["truncated"])
        action = json.loads((folder / f"action-{index:03d}.json").read_text())
        values = np.asarray(action["action"])
        assert action["step"] == index and values.shape == (1, 7) and np.isfinite(values).all()
        actions.append(values[0])
        next_input = folder / (f"input-{index+1:03d}.npz" if index < steps-1 else "terminal.npz")
        with np.load(next_input, allow_pickle=False) as following:
            assert sha256(following["simulator_state"].tobytes()).hexdigest() == row["simulator_state_sha256"]
    last = transitions[-1]
    assert last["success"] == episode["success"]
    assert last["terminated"] == episode["terminated"] and last["truncated"] == episode["truncated"]
    return {"folder": str(folder), "identity": identity, "side": side,
            "success": episode["success"], "steps": steps,
            "elapsed_seconds": episode["elapsed_seconds"],
            "episode_sha256": digest(folder / "episode.json"),
            "initial_state_sha256": episode["initial_state_sha256"],
            "initial_input_sha256": episode["initial_input_sha256"],
            "actions": np.asarray(actions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the earlier audit; use a new output")
    cohort = args.cohort.resolve()
    context = json.loads((cohort / "context.json").read_text())
    rows, attempts = [], []
    for folder in sorted((cohort / "raw").glob("episode-*")):
        attempt = json.loads((folder / "attempt.json").read_text())
        attempts.append({"folder": str(folder), "status": attempt["status"]})
        if attempt["status"] == "completed":
            rows.append(audit_side(folder, context))
    matched = []
    for identity_key in sorted({(r["identity"]["state_index"], r["identity"]["repeat"]) for r in rows}):
        sides = {r["side"]: r for r in rows if
                 (r["identity"]["state_index"], r["identity"]["repeat"]) == identity_key}
        if set(sides) == {"old", "new"}:
            old, new = sides["old"], sides["new"]
            assert old["identity"] == new["identity"]
            assert old["initial_state_sha256"] == new["initial_state_sha256"]
            assert old["initial_input_sha256"] == new["initial_input_sha256"]
            common = min(old["steps"], new["steps"])
            matched.append({"state_index": identity_key[0], "repeat": identity_key[1],
                            "old_success": old["success"], "new_success": new["success"],
                            "old_steps": old["steps"], "new_steps": new["steps"],
                            "action_prefix_exact": bool(np.array_equal(old["actions"][:common],
                                                                       new["actions"][:common])),
                            "common_prefix_max_abs_action_difference": float(np.max(np.abs(
                                old["actions"][:common] - new["actions"][:common])))})
    for row in rows:
        del row["actions"]
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only raw-evidence semantic audit of completed sides. "
                       "A partial audit does not establish native operational validity. "
                       "Action equality is descriptive, not an inference requirement.",
              "cohort": str(cohort), "context_file_sha256": digest(cohort / "context.json"),
              "audit_source_sha256": digest(__file__), "attempts": attempts,
              "completed_sides": len(rows), "matched_raw_pairs": len(matched),
              "sides": rows, "pairs": matched}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed_sides": len(rows), "matched_raw_pairs": len(matched),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
