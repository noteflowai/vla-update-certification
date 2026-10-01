"""Independently check saved reset inputs against XML, PNG and numerical assets."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from PIL import Image


def verify(folder):
    records = [json.loads((folder/f"reset-{i}.json").read_text()) for i in range(3)]
    inputs = [{key: value for key, value in row.items() if key != "reset_repair"}
              for row in records]
    arrays = []
    for index, record in enumerate(records):
        with np.load(folder/f"reset-{index}.npz", allow_pickle=False) as saved:
            arrays.append({key: saved[key].copy() for key in (
                "simulator_state", "model_body_pos", "model_body_quat")})
        assert sha256(arrays[-1]["simulator_state"].tobytes()).hexdigest() == record["state"]["sha256"]
        assert sha256((folder/f"reset-{index}.xml").read_bytes()).hexdigest() == record["scene_xml_sha256"]
        for camera, expected in record["observation"]["pixels"].items():
            pixels = np.asarray(Image.open(folder/f"reset-{index}-{camera}.png"))
            assert list(pixels.shape) == expected["shape"]
            assert str(pixels.dtype) == expected["dtype"]
            assert sha256(pixels.tobytes()).hexdigest() == expected["sha256"]
        assert record["actual_success"] is False
        assert record["actual_success"] == all(record["goal_values"])
        assert record["goals"] and record["prompt"] == record["bddl_prompt"]
        assert record["controller_use_delta"] == [True]
        repair = record["reset_repair"]
        assert repair["generated_samplers_after"] == repair["expected_generated_samplers"]
    assert all(item == inputs[0] for item in inputs)
    assert all(np.array_equal(row[key], arrays[0][key])
               for row in arrays for key in row)
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(), "resets": 3,
        "complete_inputs_exact": True, "saved_assets_match_snapshot_hashes": True,
        "model_body_poses_exact": True, "initial_checker_false": True,
        "repair_bookkeeping": [row["reset_repair"] for row in records],
        "manifest_sha256": sha256((folder/"manifest.json").read_bytes()).hexdigest(),
        "verifier_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Exact saved inputs and assets, no policy outcome. Predicate values "
                 "are the benchmark checker, not independent geometry grading."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite verification evidence")
    result = verify(args.run)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
