"""Verify archived health evidence; no model inference or semantic re-grading."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pi05-health-004"
EXPECTED_CASES = [(0, 33), (2, 26), (4, 28), (6, 24), (8, 40)]
WEIGHT_SHA = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    supervisor = json.loads((RUN / "supervisor-outcome.json").read_text())
    assert digest(RUN / "summary.json") == supervisor["summary_sha256"]
    loader = json.loads((RUN / "loader-verification.json").read_text())
    rows = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines()]
    assert [(c["task"], c["init_state"]) for c in manifest["cases"]] == EXPECTED_CASES
    assert manifest["model_weight_sha256"] == WEIGHT_SHA
    assert manifest["episode_limit"] == 520
    assert manifest["loader"] == "direct_cuda_strict"
    assert loader["strict_keys_and_shapes"] and loader["all_loaded_tensors_exact_after_official_cast"]
    assert loader["tensor_count"] == 813
    assert len(loader["dtype_conversions"]) == 436
    assert len(loader["nonpersistent_buffers_restored_from_cpu_reference"]) == 6
    archived_sources = {}
    for original, expected in manifest["sources"].items():
        name = Path(original).name
        archived = RUN / name
        if not archived.exists():
            archived = RUN / "processor-assets" / name
        assert digest(archived) == expected, name
        archived_sources[str(archived.relative_to(ROOT))] = expected
    config = json.loads((RUN / "processor-assets/config.json").read_text())
    assert config["n_action_steps"] == config["chunk_size"] == 50
    assert config["num_inference_steps"] == 10
    assert len(rows) <= 5
    traces = []
    for index, row in enumerate(rows):
        expected = manifest["cases"][index]
        assert all(row[key] == value for key, value in expected.items())
        assert type(row["success"]) is bool or row["success"] is None
        path = RUN / f"trace-task{row['task']}.npz"
        with np.load(path, allow_pickle=False) as trace:
            actions = trace["actions"]
            states = trace["simulator_states"]
            count = row.get("steps", row.get("completed_steps_before_error"))
            assert count is not None and len(actions) == count <= 520
            assert actions.shape == (count, 1, 7)
            assert states.ndim == 2 and len(states) == count + 1
            assert np.isfinite(actions).all() and np.isfinite(states).all()
            assert digest_bytes(states[0]) == row["initial_state_sha256"]
            assert row["reset_state_exact"] is True
        traces.append({"task": row["task"], "init_state": row["init_state"],
                       "success": row["success"], "infrastructure_error": row["infrastructure_error"],
                       "recorded_steps": count, "trace_sha256": digest(path)})
    complete = sum(type(row["success"]) is bool for row in rows)
    successes = sum(row["success"] is True for row in rows)
    errors = sum(row["infrastructure_error"] is not None for row in rows)
    evaluable = len(rows) == complete == 5 and errors == 0
    passed = successes >= 3 if evaluable else None
    verdict = ("passed" if passed else "failed") if evaluable else "not_assessed"
    assert summary["attempted"] == len(rows)
    assert summary["completed_episodes"] == complete
    assert summary["successes"] == successes
    assert summary["infrastructure_errors"] == errors
    assert summary["health_gate_passed"] is passed
    assert summary["model_health_verdict"] == verdict
    ready = next(event["seconds_after_import"] for event in events if event["phase"] == "policy_ready")
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "archived_source_hashes_verified": archived_sources,
        "checkpoint_sha256_frozen_before_episodes": WEIGHT_SHA,
        "loader_tensors_exact_after_official_cast": 813,
        "official_dtype_casts": 436, "cpu_reference_buffers": 6,
        "predicted_chunk_size": 50, "executed_actions_per_prediction": 50,
        "flow_steps": 10, "complete_episodes": complete, "successes": successes,
        "infrastructure_errors": errors, "health_gate_passed": passed,
        "model_health_verdict": verdict, "traces": traces,
        "supervisor_exit_code": supervisor["wrapper_exit_code"],
        "clean_process_exit_verified": supervisor["wrapper_exit_code"] == 0,
        "supervisor_scope": supervisor["scope"],
        "policy_ready_seconds_after_import": ready,
        "scope": "Checkpoint-default 50-action health only. Raw traces and counts verified; "
                 "success booleans originate from the simulator, not an independent semantic checker audit. "
                 "Shared-host times are diagnostics. No main update efficacy or ten-action pipeline validation."
    }
    target = ROOT / "analysis/pi05-health-004-verified.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "complete_episodes", "successes", "infrastructure_errors", "model_health_verdict")}))


def digest_bytes(array):
    return sha256(array.tobytes()).hexdigest()


if __name__ == "__main__":
    main()
