"""Live LIBERO wiring/reset audit, without a policy or update outcomes."""
import os
for key, value in {"MUJOCO_GL": "egl", "LIBERO_CONFIG_PATH": "/tmp/vlareg/libero_cfg",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}.items():
    os.environ.setdefault(key, value)
import argparse
import atexit
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import bddl_utils
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import get_libero_dummy_action, LiberoEnv

ROOT = Path(__file__).resolve().parents[1]
from libero_audit_core import make_sub_env
from canonical_reset import reset_without_sampler_accumulation, BDDLBaseDomain


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": sha256(value.tobytes()).hexdigest()}


def observation_hash(value):
    if isinstance(value, dict):
        return {key: observation_hash(item) for key, item in sorted(value.items())}
    return array_hash(value)


def snapshot(env, observation):
    control = env._env
    domain = control.env
    goals = domain.parsed_problem["goal_state"]
    return {
        "state": array_hash(control.get_sim_state()),
        "observation": observation_hash(observation),
        "scene_xml_sha256": sha256(domain.sim.model.get_xml().encode()).hexdigest(),
        "prompt": env.task_description,
        "bddl_prompt": control.language_instruction,
        "goals": goals,
        "goal_values": [bool(domain._eval_predicate(goal)) for goal in goals],
        "actual_success": bool(control.check_success()),
        "controller_use_delta": [bool(robot.controller.use_delta) for robot in control.robots],
    }


def normalized(text):
    return " ".join(text.lower().replace("_", " ").split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()

    def event(phase, **extra):
        row = {"at": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic()-started, "phase": phase, **extra}
        with (args.output/"events.jsonl").open("a") as file:
            file.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    def stop(signum, frame):
        raise TimeoutError("Declared outer audit budget exhausted")

    signal.signal(signal.SIGTERM, stop)
    atexit.register(lambda: event("interpreter_exit_hook"))
    source = ROOT/"protocols/closedloop-suite-001/manifest.json"
    suite_manifest = json.loads(source.read_text())
    cases = [{**state, "mode": mode, "scope": "main_state_reset_only"}
             for mode in ("relative", "absolute") for state in suite_manifest["states"]]
    health = json.loads((ROOT/"runs/pi05-health-004/manifest.json").read_text())
    cases += [{**state, "mode": "relative", "scope": "historical_health_checker_replay"}
              for state in health["cases"]]
    paths = [Path(__file__), source, Path(inspect.getfile(LiberoEnv)),
             Path(inspect.getfile(bddl_utils)), Path(inspect.getfile(make_sub_env)),
             Path(inspect.getfile(reset_without_sampler_accumulation)),
             Path(inspect.getfile(BDDLBaseDomain)),
             Path(__file__).with_name("HEALTH_AMENDMENT_006.md")]
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "scope": "Local canonical reset repair; no policy inference. Same-seed resets, inputs, checker binding and stored health final-state replay.",
        "sources": {str(path): digest(path) for path in paths},
        "packages": {key: importlib.metadata.version(key) for key in
                     ("lerobot", "mujoco", "robosuite", "gymnasium")},
        "gate": "Every declared row: stable simulator/input/scene reset, prompt/goal binding, "
                "initial checker false, correct controller, step info consistent and no error.",
        "limitations": "Uses benchmark predicates; not independent geometry grading or proof of solvability. "
                       "Recorded-state replay does not reproduce the policy's full action trajectory.",
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for path in paths:
        (args.output/("source-"+path.name)).write_bytes(path.read_bytes())
    event("manifest_frozen", cases=len(cases))
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    rows = []
    with (args.output/"observations.jsonl").open("w") as output:
        for index, case in enumerate(cases):
            env = None
            row = {"index": index, "case": case, "passed": False, "error": None}
            event("case_started", index=index, task=case["task"], mode=case["mode"])
            try:
                env = make_sub_env(suite, "libero_10", case["task"], case["init_state"],
                                   LiberoEnvConfig(task="libero_10", control_mode=case["mode"]), 520)
                inventory = suite_manifest["inventory"][case["task"]]
                for key, hash_key in (("init_file", "init_file_sha256"), ("bddl_file", "bddl_sha256")):
                    assert digest(Path(inventory[key])) == inventory[hash_key], key
                parsed = bddl_utils.robosuite_parse_problem(inventory["bddl_file"])
                initial, info = reset_without_sampler_accumulation(env, case["scene_seed"])
                first = snapshot(env, initial)
                (args.output/f"reset-{index}-0.json").write_text(json.dumps(first, indent=2)+"\n")
                if case["scope"] == "main_state_reset_only":
                    assert array_hash(env._init_states[case["init_state"]])["sha256"] == case["source_state_sha256"]
                assert normalized(first["prompt"]) == normalized(first["bddl_prompt"])
                assert first["goals"] == parsed["goal_state"] and len(first["goals"]) > 0
                assert first["actual_success"] == all(first["goal_values"])
                assert first["actual_success"] is False, "Task starts already successful"
                assert first["controller_use_delta"] == [case["mode"] == "relative"]
                for camera, pixels in initial["pixels"].items():
                    Image.fromarray(pixels).save(args.output/f"initial-{index}-{camera}.png")
                second_observation, _ = reset_without_sampler_accumulation(env, case["scene_seed"])
                second = snapshot(env, second_observation)
                (args.output/f"reset-{index}-1.json").write_text(json.dumps(second, indent=2)+"\n")
                assert first == second, "Same-seed simulator/input/checker/scene mismatch"
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                stepped, _, terminated, truncated, step_info = env.step(action)
                step_snapshot = snapshot(env, stepped)
                assert bool(step_info["is_success"]) == step_snapshot["actual_success"]
                assert step_snapshot["actual_success"] == all(step_snapshot["goal_values"])
                assert bool(terminated) == (bool(step_info["done"]) or step_snapshot["actual_success"])
                reset_observation, _ = reset_without_sampler_accumulation(env, case["scene_seed"])
                third = snapshot(env, reset_observation)
                (args.output/f"reset-{index}-2.json").write_text(json.dumps(third, indent=2)+"\n")
                assert first == third, "Episode carryover after step"
                row.update(initial=first, after_noop=step_snapshot,
                           reset_info=info, step_info=step_info)
                if case["scope"] == "historical_health_checker_replay":
                    trace_path = ROOT/f"runs/pi05-health-004/trace-task{case['task']}.npz"
                    old_rows = [json.loads(line) for line in (
                        ROOT/"runs/pi05-health-004/episodes.jsonl").read_text().splitlines()]
                    old = next(item for item in old_rows if item["task"] == case["task"])
                    with np.load(trace_path, allow_pickle=False) as trace:
                        assert array_hash(trace["simulator_states"][0]) == first["state"]
                        env._env.regenerate_obs_from_state(trace["simulator_states"][-1])
                    domain = env._env.env
                    values = [bool(domain._eval_predicate(goal)) for goal in first["goals"]]
                    actual = bool(env._env.check_success())
                    assert actual == all(values) == old["success"]
                    row["historical_replay"] = {
                        "trace_sha256": digest(trace_path), "goal_values": values,
                        "checker_success": actual, "stored_success": old["success"]}
                row["passed"] = True
            except Exception as error:
                row["error"] = type(error).__name__+": "+str(error)
                row["traceback"] = traceback.format_exc()
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        row["passed"] = False
                        row["close_error"] = type(error).__name__+": "+str(error)
            output.write(json.dumps(row, default=lambda value: value.item() if isinstance(
                value, np.generic) else str(value))+"\n")
            output.flush()
            rows.append(row)
            event("case_finished", index=index, passed=row["passed"], error=row["error"])
            if not row["passed"]:
                break
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(), "declared": len(cases),
        "completed": len(rows), "passed_rows": sum(row["passed"] for row in rows),
        "errors": sum(row["error"] is not None or "close_error" in row for row in rows),
        "audit_passed": len(rows) == len(cases) and all(row["passed"] for row in rows),
        "new_policy_update_episodes": 0, "scope": manifest["scope"],
        "limitations": manifest["limitations"]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    event("summary_persisted", **summary)
    return args.output


if __name__ == "__main__":
    folder = main()
    print(datetime.now(timezone.utc).isoformat(), "main_returned", str(folder), flush=True)
