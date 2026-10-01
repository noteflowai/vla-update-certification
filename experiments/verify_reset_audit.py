"""Verify complete reset/scoring evidence independently of the renderer process."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    summary = json.loads((folder/"summary.json").read_text())
    rows = [json.loads(line) for line in (folder/"observations.jsonl").read_text().splitlines()]
    suite = json.loads((ROOT/"protocols/closedloop-suite-001/manifest.json").read_text())
    health_path = ROOT/"runs/pi05-health-004/manifest.json"
    health = json.loads(health_path.read_text())
    expected = [{**case, "mode": mode, "scope": "main_state_reset_only"}
                for mode in ("relative", "absolute") for case in suite["states"]]
    expected += [{**case, "mode": "relative", "scope": "historical_health_checker_replay"}
                 for case in health["cases"]]
    assert len(expected) == len(rows) == 45 and manifest["cases"] == expected
    assert summary["audit_passed"] is True
    assert (summary["declared"], summary["completed"], summary["passed_rows"], summary["errors"]) == (45, 45, 45, 0)
    for source, expected_hash in manifest["sources"].items():
        path = Path(source)
        assert digest(path) == expected_hash, path
        assert digest(folder/("source-"+path.name)) == expected_hash, path
    episodes_path = ROOT/"runs/pi05-health-004/episodes.jsonl"
    old = {row["task"]: row for row in map(json.loads, episodes_path.read_text().splitlines())}
    references = {str(health_path): digest(health_path),
                  str(episodes_path): digest(episodes_path)}
    for index, (row, case) in enumerate(zip(rows, expected, strict=True)):
        assert row["index"] == index and row["case"] == case
        assert row["passed"] is True and row["error"] is None and "close_error" not in row
        snapshots = [json.loads((folder/f"reset-{index}-{repeat}.json").read_text())
                     for repeat in range(3)]
        assert all(item == snapshots[0] for item in snapshots)
        initial = row["initial"]
        assert initial == snapshots[0]
        assert initial["prompt"] == initial["bddl_prompt"] and initial["goals"]
        assert initial["actual_success"] is False
        assert initial["actual_success"] == all(initial["goal_values"])
        assert initial["controller_use_delta"] == [case["mode"] == "relative"]
        assert row["after_noop"]["actual_success"] == row["step_info"]["is_success"]
        assert row["after_noop"]["actual_success"] == all(row["after_noop"]["goal_values"])
        for camera, expected_pixels in initial["observation"]["pixels"].items():
            pixels = np.asarray(Image.open(folder/f"initial-{index}-{camera}.png"))
            assert list(pixels.shape) == expected_pixels["shape"]
            assert str(pixels.dtype) == expected_pixels["dtype"]
            assert sha256(pixels.tobytes()).hexdigest() == expected_pixels["sha256"]
        if case["scope"] == "historical_health_checker_replay":
            trace = ROOT/f"runs/pi05-health-004/trace-task{case['task']}.npz"
            assert row["historical_replay"]["trace_sha256"] == digest(trace)
            references[str(trace)] = digest(trace)
            with np.load(trace, allow_pickle=False) as data:
                assert sha256(data["simulator_states"][0].tobytes()).hexdigest() == initial["state"]["sha256"]
            replay = row["historical_replay"]
            assert replay["checker_success"] == replay["stored_success"] == old[case["task"]]["success"]
            assert replay["checker_success"] == all(replay["goal_values"])
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(), "verified": True,
        "declared_and_verified_rows": 45, "main_state_mode_rows": 40,
        "historical_checker_replay_rows": 5, "new_policy_update_episodes": 0,
        "sources": manifest["sources"], "historical_reference_hashes_at_verification": references,
        "summary_sha256": digest(folder/"summary.json"),
        "observations_sha256": digest(folder/"observations.jsonl"),
        "verifier_source_sha256": digest(Path(__file__)),
        "scope": "Frozen case identities, source archives, three exact input snapshots per case, "
                 "saved camera assets and historical checker replay. Benchmark predicates, "
                 "not independent geometry grading or new model-health evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite verification evidence")
    result = verify(args.run)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("sources", "historical_reference_hashes_at_verification")}), flush=True)
