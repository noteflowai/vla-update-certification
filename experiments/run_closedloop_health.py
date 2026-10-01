"""Five predeclared, fresh-state health checks; not policy-update efficacy."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/vlareg/libero_cfg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import atexit
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import importlib.metadata
import sys
import time
import traceback
import signal
import faulthandler
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
print(datetime.now(timezone.utc).isoformat(), "importing model and simulator libraries", flush=True)
import numpy as np
import torch
import gymnasium as gym
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from libero.libero import benchmark

VLA = Path("/home/dcvuser/work/vla-regressions")
sys.path.insert(0, str(VLA))
from perstate_eval import make_sub_env, scene_seed
print(datetime.now(timezone.utc).isoformat(), "library imports completed", flush=True)

MODELS = {
    "xvla": ("lerobot/xvla-libero", "12e8783e996944f5c97e490d37d4c145484ed70a", "absolute"),
    "pi05": ("lerobot/pi05_libero_finetuned_v044", "8e174154ef5f6c60a8da12ae99c303d8963138c1", "relative"),
}


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0])*1024


def sim_state(vector):
    return np.asarray(vector.envs[0].unwrapped._env.env.sim.get_state().flatten()).copy()


def unbatch_observation(value):
    if isinstance(value, dict):
        return {key: unbatch_observation(item) for key, item in value.items()}
    return np.asarray(value)[0]


def main():
    # If the shared host runs out of memory, prefer terminating this experiment.
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loader", choices=("official", "direct_cuda_strict"), default="official")
    parser.add_argument("--wall-budget", type=int, default=1800)
    parser.add_argument("--reset-mode", choices=("canonical", "legacy"), default="canonical")
    args = parser.parse_args()
    if args.wall_budget < 1 or (args.loader == "direct_cuda_strict" and args.family != "pi05"):
        raise ValueError("Invalid declared loader or wall budget")
    args.output.mkdir(parents=True, exist_ok=False)
    execution_started = time.monotonic()

    def event(phase, **details):
        row = {"at": datetime.now(timezone.utc).isoformat(),
               "seconds_after_import": time.monotonic()-execution_started,
               "phase": phase, **details}
        with (args.output/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
            handle.flush()
        print(json.dumps(row), flush=True)

    def terminated(signum, frame):
        raise TimeoutError("External bounded-run termination")

    signal.signal(signal.SIGTERM, terminated)
    event("initialization_started")
    atexit.register(lambda: event("interpreter_exit_hook"))
    model_id, revision, control = MODELS[args.family]
    checkpoint = (Path.home()/".cache/huggingface/hub"/
                  ("models--"+model_id.replace("/", "--"))/"snapshots"/revision)
    assert (checkpoint/"model.safetensors").is_file()
    processor_files = []
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config_path = checkpoint/name
        assert config_path.is_file(), f"Missing processor config: {name}"
        processor_files.append(config_path)
        for step in json.loads(config_path.read_text())["steps"]:
            if "state_file" in step:
                state_path = checkpoint/step["state_file"]
                assert state_path.is_file(), f"Missing processor state: {state_path.name}"
                processor_files.append(state_path)
    rng = np.random.default_rng(20260930)
    cases = [{"task": task, "init_state": int(rng.choice(np.arange(10, 50))),
              "scene_seed": scene_seed(3101, task, 0), "policy_seed": 310100+task}
             for task in (0, 2, 4, 6, 8)]
    for case in cases:
        case["scene_seed"] += case["init_state"]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fresh-state model health only; no update efficacy or latency claim",
        "family": args.family, "model_id": model_id, "revision": revision,
        "control_mode": control, "cases": cases, "episode_limit": 520,
        "health_gate": "at least 3/5 successes; zero infrastructure errors; exact reset checks",
        "loader": args.loader, "wall_budget_after_import_seconds": args.wall_budget,
        "reset_mode": args.reset_mode,
        "shared_gpu_context": "Other tasks may run; durations are not performance benchmarks.",
        "sources": {str(p): sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), VLA/"perstate_eval.py", checkpoint/"config.json",
                     *processor_files]},
        "model_weight_sha256": None,
        "torch": torch.__version__,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("lerobot", "transformers", "gymnasium", "mujoco", "robosuite")},
    }
    record = args.output/"manifest.json"
    record.write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output/"run_closedloop_health.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"perstate_eval.py").write_bytes((VLA/"perstate_eval.py").read_bytes())
    if args.reset_mode == "canonical":
        from canonical_reset_wrapper import CanonicalResetWrapper
        from libero_audit_core import snapshot
        for name in ("canonical_reset.py", "canonical_reset_wrapper.py", "libero_audit_core.py",
                     "audit_reset_scoring.py", "HEALTH_AMENDMENT_006.md"):
            helper = Path(__file__).with_name(name)
            manifest["sources"][str(helper)] = sha256(helper.read_bytes()).hexdigest()
            (args.output/helper.name).write_bytes(helper.read_bytes())
    if args.loader == "direct_cuda_strict":
        for name in ("strict_pi05_loader.py", "test_strict_loader.py", "HEALTH_AMENDMENT_004.md",
                     "HEALTH_LIFECYCLE_005.md"):
            helper = Path(__file__).with_name(name)
            manifest["sources"][str(helper)] = sha256(helper.read_bytes()).hexdigest()
            (args.output/helper.name).write_bytes(helper.read_bytes())
    implementation = Path(make_policy.__code__.co_filename)
    manifest["installed_policy_factory_sha256"] = sha256(implementation.read_bytes()).hexdigest()
    assets = args.output/"processor-assets"
    assets.mkdir()
    for path in [checkpoint/"config.json", *processor_files]:
        (assets/path.name).write_bytes(path.read_bytes())
    record.write_text(json.dumps(manifest, indent=2)+"\n")
    # Bind the actual cached bytes before any episode collection.
    print(datetime.now(timezone.utc).isoformat(), "hashing pinned model weights", flush=True)
    with (checkpoint/"model.safetensors").open("rb") as handle:
        import hashlib
        manifest["model_weight_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest["frozen_at"] = datetime.now(timezone.utc).isoformat()
    record.write_text(json.dumps(manifest, indent=2)+"\n")
    print(datetime.now(timezone.utc).isoformat(), "model hash frozen; checking memory", flush=True)
    event("weights_frozen", sha256=manifest["model_weight_sha256"])
    available_host = memory_available()
    if available_host < 18*1024**3:
        event("resource_gate_failed", host_available_bytes=available_host,
              host_required_bytes=18*1024**3, resource="host")
        raise RuntimeError("Insufficient available host memory for the bounded health check")
    free, total = torch.cuda.mem_get_info()
    event("preload_resource_gate", host_available_bytes=available_host,
          gpu_free_bytes=free, gpu_total_bytes=total, gpu_required_free_bytes=28*1024**3,
          cuda_device=torch.cuda.current_device(), gpu_name=torch.cuda.get_device_name())
    if free < 28*1024**3:
        raise RuntimeError("Insufficient GPU headroom for the bounded health check")
    torch.set_num_threads(2)
    fraction = .5 if args.loader == "direct_cuda_strict" else .4
    torch.cuda.set_per_process_memory_fraction(fraction)
    manifest["cuda_memory_fraction"] = fraction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path, cfg.device = str(checkpoint), "cuda"
    # Preserve the checkpoint's precision and action semantics, but avoid a
    # compilation benchmark on a shared GPU during this small health check.
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    manifest["compile_model_override"] = False
    record.write_text(json.dumps(manifest, indent=2)+"\n")
    env_cfg = LiberoEnvConfig(task="libero_10", control_mode=control)
    print(datetime.now(timezone.utc).isoformat(), "loading processors and policy", flush=True)
    event("processors_loading")
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    event("policy_loading")
    if args.loader == "direct_cuda_strict":
        from strict_pi05_loader import load_pi05
        policy, audit = load_pi05(cfg, env_cfg, checkpoint)
        (args.output/"loader-verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    else:
        policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval()
    event("policy_ready")
    print(datetime.now(timezone.utc).isoformat(), "policy ready", flush=True)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    results = []
    with (args.output/"episodes.jsonl").open("w") as output:
        for case in cases:
            print(datetime.now(timezone.utc).isoformat(), "starting case", case, flush=True)
            started = time.monotonic()
            vector = None
            result = {**case, "family": args.family, "success": None,
                      "infrastructure_error": None}
            actions, states = [], []
            event("case_started", task=case["task"], init_state=case["init_state"])

            def save_trace():
                np.savez_compressed(args.output/f"trace-task{case['task']}.npz",
                                    actions=np.asarray(actions), simulator_states=np.asarray(states))

            try:
                event("environment_constructing", task=case["task"])
                def construct():
                    environment = make_sub_env(
                        suite, "libero_10", case["task"], case["init_state"], env_cfg, 520)
                    return (CanonicalResetWrapper(environment)
                            if args.reset_mode == "canonical" else environment)
                vector = gym.vector.SyncVectorEnv([construct],
                    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
                event("environment_ready", task=case["task"])
                observation, _ = vector.reset(seed=[case["scene_seed"]])
                first = sim_state(vector)
                if args.reset_mode == "canonical":
                    first_inputs = snapshot(vector.envs[0].unwrapped, unbatch_observation(observation))
                    (args.output/f"reset-task{case['task']}-0.json").write_text(
                        json.dumps(first_inputs, indent=2)+"\n")
                observation, _ = vector.reset(seed=[case["scene_seed"]])
                second = sim_state(vector)
                result["initial_state_sha256"] = sha256(first.tobytes()).hexdigest()
                result["reset_state_exact"] = np.array_equal(first, second)
                if not result["reset_state_exact"]:
                    raise RuntimeError("Identical reset seeds did not restore identical simulator state")
                if args.reset_mode == "canonical":
                    second_inputs = snapshot(vector.envs[0].unwrapped, unbatch_observation(observation))
                    (args.output/f"reset-task{case['task']}-1.json").write_text(
                        json.dumps(second_inputs, indent=2)+"\n")
                    result["reset_inputs_exact"] = first_inputs == second_inputs
                    result["initial_input_sha256"] = sha256(
                        json.dumps(second_inputs, sort_keys=True).encode()).hexdigest()
                    if (not result["reset_inputs_exact"] or second_inputs["actual_success"]
                            or second_inputs["prompt"] != second_inputs["bddl_prompt"]
                            or not second_inputs["goals"]
                            or second_inputs["controller_use_delta"] != [control == "relative"]):
                        raise RuntimeError("Reset inputs, initial checker, prompt or controller audit failed")
                states = [second]
                save_trace()
                event("reset_verified", task=case["task"],
                      initial_state_sha256=result["initial_state_sha256"])
                policy.reset()
                torch.manual_seed(case["policy_seed"])
                for step in range(520):
                    if time.monotonic()-execution_started > args.wall_budget:
                        raise TimeoutError("Declared bounded-run wall budget exhausted")
                    if step % 50 == 0 and memory_available() < 3*1024**3:
                        raise RuntimeError("Host memory guard stopped the health check")
                    prepared = preprocess_observation(observation)
                    prepared["task"] = list(vector.call("task_description"))
                    prepared = pre(env_pre(prepared))
                    if step % 50 == 0:
                        event("action_requested", task=case["task"], step=step)
                    with torch.inference_mode():
                        selected = policy.select_action(prepared)
                    action = env_post({ACTION: post(selected.float())})[ACTION].to("cpu").numpy()
                    if not np.isfinite(action).all() or action.shape != (1, 7):
                        raise RuntimeError("Invalid processed action shape or nonfinite action")
                    observation, _, terminated, truncated, info = vector.step(action)
                    actions.append(action.copy())
                    states.append(sim_state(vector))
                    if step % 50 == 0:
                        save_trace()
                        event("rollout_progress", task=case["task"], completed_steps=step+1)
                    if (terminated | truncated).any():
                        result.update(success=bool(np.asarray(info["is_success"])[0]),
                                      steps=step+1, terminated=bool(terminated[0]),
                                      truncated=bool(truncated[0]))
                        break
                else:
                    result.update(success=False, steps=520, terminated=False, truncated=True)
                save_trace()
            except Exception as error:
                result["infrastructure_error"] = type(error).__name__+": "+str(error)
                result["completed_steps_before_error"] = len(actions)
                save_trace()
                (args.output/f"error-task{case['task']}.log").write_text(traceback.format_exc())
            finally:
                if vector is not None:
                    vector.close()
            result["elapsed_seconds"] = time.monotonic()-started
            output.write(json.dumps(result)+"\n")
            output.flush()
            results.append(result)
            print(json.dumps(result), flush=True)
            event("case_finished", task=case["task"], success=result["success"],
                  infrastructure_error=result["infrastructure_error"])
            if result["infrastructure_error"]:
                break
    passed = (len(results) == 5 and sum(r["success"] is True for r in results) >= 3
              and all(r["infrastructure_error"] is None for r in results))
    (args.output/"summary.json").write_text(json.dumps({
        "attempted": len(results), "successes": sum(r["success"] is True for r in results),
        "completed_episodes": sum(type(r["success"]) is bool for r in results),
        "infrastructure_errors": sum(r["infrastructure_error"] is not None for r in results),
        "health_gate_passed": passed if len(results) == 5 and
                             all(r["infrastructure_error"] is None for r in results) else None,
        "model_health_verdict": ("passed" if passed else "failed" if len(results) == 5 and
                                all(r["infrastructure_error"] is None for r in results)
                                else "not_assessed"),
        "scope": manifest["scope"]}, indent=2)+"\n")
    event("summary_persisted")
    return args.output


if __name__ == "__main__":
    try:
        result_folder = main()
        returned = {"at": datetime.now(timezone.utc).isoformat(), "phase": "main_returned"}
        with (result_folder/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(returned)+"\n")
        print(json.dumps(returned), flush=True)
    except Exception as error:
        # Initialization errors are not unsuccessful policy episodes.
        if "--output" in sys.argv:
            folder = Path(sys.argv[sys.argv.index("--output")+1])
            if folder.exists():
                episodes = folder/"episodes.jsonl"
                rows = [json.loads(line) for line in episodes.read_text().splitlines()] if episodes.exists() else []
                summary_file = folder/"summary.json"
                persisted = json.loads(summary_file.read_text()) if summary_file.exists() else None
                (folder/"execution-outcome.json").write_text(json.dumps({
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "status": ("health_completed_cleanup_error" if persisted
                               else "initialization_error" if not rows else "execution_error"),
                    "error": type(error).__name__+": "+str(error),
                    "completed_episodes": sum(type(row.get("success")) is bool for row in rows),
                    "model_health_verdict": persisted["model_health_verdict"] if persisted else "not_assessed",
                }, indent=2)+"\n")
        raise
