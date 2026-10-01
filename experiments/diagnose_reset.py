"""Persist every reset input to diagnose a failed structural audit."""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import traceback
import numpy as np
from PIL import Image
from audit_reset_scoring import (
    ROOT, benchmark, digest, LiberoEnvConfig, make_sub_env, snapshot, torch)


def differences(left, right, prefix=""):
    result = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            name = prefix+"."+key if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result += differences(left[key], right[key], name)
    elif left != right:
        result.append(prefix)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/"protocols/closedloop-suite-001/manifest.json"
    case = json.loads(source.read_text())["states"][args.case_index]
    (args.output/"manifest.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(), "case": case,
        "scope": "Focused diagnosis, no policy or effect-size evidence",
        "sources": {str(path): digest(path) for path in [
            Path(__file__), Path(inspect.getfile(snapshot)), source]},
        "resets": 3, "control_mode": "relative"}, indent=2)+"\n")
    torch.set_num_threads(2)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env = make_sub_env(suite, "libero_10", case["task"], case["init_state"],
                       LiberoEnvConfig(task="libero_10", control_mode="relative"), 520)
    records, observations, states = [], [], []
    try:
        for repeat in range(3):
            observation, _ = env.reset(seed=case["scene_seed"])
            record = snapshot(env, observation)
            records.append(record)
            observations.append(observation)
            states.append(np.asarray(env._env.get_sim_state()).copy())
            (args.output/f"reset-{repeat}.json").write_text(json.dumps(record, indent=2)+"\n")
            np.savez_compressed(args.output/f"reset-{repeat}.npz",
                                simulator_state=states[-1])
            for camera, pixels in observation["pixels"].items():
                Image.fromarray(pixels).save(args.output/f"reset-{repeat}-{camera}.png")
            print(json.dumps({"reset": repeat, "snapshot_persisted": True}), flush=True)
        pairs = []
        for index in (1, 2):
            pixels = {}
            for camera, original in observations[0]["pixels"].items():
                updated = observations[index]["pixels"][camera]
                delta = original.astype(np.int16)-updated.astype(np.int16)
                pixels[camera] = {
                    "changed_channels": int(np.count_nonzero(delta)),
                    "max_absolute_channel_difference": int(np.abs(delta).max())}
            pairs.append({
                "resets": [0, index], "different_fields": differences(records[0], records[index]),
                "state_max_absolute_difference": float(np.max(np.abs(states[0]-states[index]))),
                "pixel_differences": pixels})
        summary = {"pairs": pairs, "all_exact": all(not pair["different_fields"] for pair in pairs),
                   "scope": "Raw reset inputs; no acceptance tolerance chosen from these results."}
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary), flush=True)
    except Exception:
        (args.output/"error.txt").write_text(traceback.format_exc())
        raise
    finally:
        env.close()
    print(datetime.now(timezone.utc).isoformat(), "main_returned", flush=True)


if __name__ == "__main__":
    main()
