"""Actual pinned LeRobot/LIBERO backend, imported only after readiness gates."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

from native_collection import digest_file, object_digest, bind_state_reset, device_identity
from paired_collection import atomic_json


def save_arrays(path, arrays):
    import numpy as np
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def flatten_observation(value, prefix="observation"):
    import numpy as np
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten_observation(item, prefix + "/" + key))
        return result
    return {prefix: np.asarray(value)}


class LiberoNativeBackend:
    def __init__(self):
        self.policy = self.pre = self.post = self.env_pre = self.env_post = None
        self.verified_checkpoint = None

    def load(self, side, context, folder, batch_id):
        if self.policy is not None:
            raise RuntimeError("Release the previous policy before provisioning a side")
        if device_identity() != context["evaluator"]["device_identity"]:
            raise ValueError("Native device or driver changed; do not mix this cohort")
        # These imports/settings are the same ones exercised by the family-health run.
        import run_closedloop_health as h
        from health_supervisor import resource_probe
        from update_variants import configure_variant, round_linear_weights
        probe = resource_probe()
        atomic_json(folder / f"resource-{batch_id}-{side}-admission.json", probe)
        if not probe["allowed"]:
            raise RuntimeError("Native pipeline loading requires current resource headroom: "
                               + json.dumps(probe, sort_keys=True))
        for name, expected in context["evaluator"]["sources"].items():
            if digest_file(name) != expected:
                raise ValueError("Frozen evaluator source changed: " + name)
        for name, expected in context["evaluator"]["packages"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError("Frozen dependency version changed: " + name)
        pipeline = context[side + "_pipeline"]
        checkpoint = Path(pipeline["checkpoint"])
        for name, expected in pipeline["assets"].items():
            if digest_file(name) != expected:
                raise ValueError("Frozen processor/config asset changed: " + name)
        weight_path = checkpoint / "model.safetensors"
        stat = weight_path.stat()
        signature = (str(weight_path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns,
                     stat.st_ctime_ns, pipeline["model_weight_sha256"])
        # Hash once per unchanged process-local file signature. No cross-run digest cache.
        if signature != self.verified_checkpoint:
            if digest_file(weight_path) != pipeline["model_weight_sha256"]:
                raise ValueError("Checkpoint bytes differ from the health-validated model")
            self.verified_checkpoint = signature
        probe = resource_probe()
        atomic_json(folder / f"resource-{batch_id}-{side}-construction.json", probe)
        if not probe["allowed"]:
            raise RuntimeError("Resource headroom changed before native model construction")
        h.torch.set_num_threads(2)
        h.torch.cuda.set_per_process_memory_fraction(.5 if context["family"] == "pi05" else .4)
        h.torch.backends.cuda.matmul.allow_tf32 = False
        h.torch.backends.cudnn.allow_tf32 = False
        h.torch.use_deterministic_algorithms(True, warn_only=True)
        cfg = h.PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.pretrained_path, cfg.device = str(checkpoint), "cuda"
        cfg, variant = configure_variant(cfg, context["family"], pipeline["update"])
        if (variant["flow_steps"] != pipeline["flow_steps"]
                or cfg.n_action_steps != pipeline["n_action_steps"]
                or cfg.chunk_size != pipeline["chunk_size"]):
            raise ValueError("Actual loaded policy configuration differs from the frozen pipeline")
        self.env_cfg = h.LiberoEnvConfig(task="libero_10", control_mode=pipeline["control_mode"])
        self.pre, self.post = h.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}})
        self.env_pre, self.env_post = h.make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=cfg)
        if context["family"] == "pi05":
            from strict_pi05_loader import load_pi05
            self.policy, loader = load_pi05(cfg, self.env_cfg, checkpoint)
        else:
            self.policy = h.make_policy(cfg=cfg, env_cfg=self.env_cfg).eval()
            loader = {"loader": "official", "scope": "Official X-VLA strict-key loader; "
                      "not a claim of PI05-style per-tensor equality verification."}
        report = {"side": side, "resource_probe": probe, "variant": variant,
                  "loader": loader, "pipeline_sha256": context[side + "_pipeline_sha256"],
                  "runtime": {"device_identity": device_identity(),
                              "torch_build": h.torch.__version__, "cuda": h.torch.version.cuda,
                              "cudnn": h.torch.backends.cudnn.version(),
                              "torch_threads": h.torch.get_num_threads()}}
        if pipeline["update"] in ("w4", "w3"):
            report["rounding"] = round_linear_weights(self.policy, pipeline["update"])
        atomic_json(folder / f"pipeline-{batch_id}-{side}.json", report)
        self.context = context
        self.suite = h.benchmark.get_benchmark_dict()["libero_10"]()
        return {"pipeline_sha256": report["pipeline_sha256"],
                "flow_steps": variant["flow_steps"], "n_action_steps": cfg.n_action_steps,
                "report_path": str(folder / f"pipeline-{batch_id}-{side}.json")}

    def run_episode(self, side, identity, folder, deadline):
        import run_closedloop_health as h
        from canonical_reset_wrapper import CanonicalResetWrapper
        from libero_audit_core import snapshot
        vector = None
        episode = {key: identity[key] for key in
                   ("task", "init_state", "scene_seed", "policy_seed")}
        episode.update(pipeline_sha256=identity[side + "_pipeline_sha256"],
                       evaluator_sha256=identity["evaluator_sha256"],
                       success=None, infrastructure_error=None)
        started = time.monotonic()

        def budget_check():
            if time.monotonic() >= deadline:
                raise TimeoutError("Declared native physical budget exhausted")
            if h.memory_available() < 3 * 1024 ** 3:
                raise RuntimeError("Native host memory guard stopped collection")

        try:
            budget_check()
            def construct():
                return CanonicalResetWrapper(h.make_sub_env(
                    self.suite, "libero_10", identity["task"], identity["init_state"],
                    self.env_cfg, 520))
            vector = h.gym.vector.SyncVectorEnv(
                [construct], autoreset_mode=h.gym.vector.AutoresetMode.NEXT_STEP)
            snapshots, reset_states = [], []
            for index in (0, 1):
                budget_check()
                observation, _ = vector.reset(seed=[identity["scene_seed"]])
                reset_states.append(h.sim_state(vector))
                inputs = snapshot(vector.envs[0].unwrapped, h.unbatch_observation(observation))
                snapshots.append(inputs)
                atomic_json(folder / f"reset-{index}.json", inputs)
            episode["reset_state_exact"] = bool(h.np.array_equal(*reset_states))
            episode["initial_state_sha256"] = h.sha256(reset_states[1].tobytes()).hexdigest()
            episode["reset_inputs_exact"] = snapshots[0] == snapshots[1]
            episode["initial_input_sha256"] = object_digest(snapshots[1])
            current = snapshots[1]
            if (not episode["reset_state_exact"] or not episode["reset_inputs_exact"]
                    or current["actual_success"] or current["prompt"] != current["bddl_prompt"]
                    or not current["goals"] or current["controller_use_delta"] != [
                        self.env_cfg.control_mode == "relative"]):
                raise RuntimeError("Native reset, prompt, checker or controller audit failed")
            # Bind before prediction, not after observing a policy outcome.
            bind_state_reset(folder.parent / "state-resets", identity, episode)
            atomic_json(folder / "reset-verification.json", episode)
            self.policy.reset()
            h.torch.manual_seed(identity["policy_seed"])
            with (folder / "transitions.jsonl").open("a") as transitions:
                for step in range(520):
                    budget_check()
                    # Save the full raw observation before prediction, including both cameras.
                    save_arrays(folder / f"input-{step:03d}.npz", {
                        **flatten_observation(observation), "simulator_state": h.sim_state(vector)})
                    prepared = h.preprocess_observation(observation)
                    prepared["task"] = list(vector.call("task_description"))
                    prepared = self.pre(self.env_pre(prepared))
                    with h.torch.inference_mode():
                        selected = self.policy.select_action(prepared)
                    action = self.env_post({h.ACTION: self.post(selected.float())})[
                        h.ACTION].to("cpu").numpy()
                    if action.shape != (1, 7) or not h.np.isfinite(action).all():
                        raise RuntimeError("Invalid native processed action")
                    # Persist even a selected action whose simulator step is interrupted.
                    atomic_json(folder / f"action-{step:03d}.json",
                                {"step": step, "action": action.tolist(),
                                 "selected_at_monotonic": time.monotonic()})
                    observation, reward, terminated, truncated, info = vector.step(action)
                    domain = vector.envs[0].unwrapped._env.env
                    actual_success = bool(vector.envs[0].unwrapped._env.check_success())
                    delivered_success = bool(h.np.asarray(info["is_success"])[0])
                    if actual_success != delivered_success:
                        raise RuntimeError("Native delivered success differs from the actual checker")
                    env_terminated, env_truncated = bool(terminated[0]), bool(truncated[0])
                    collector_truncated = step == 519 and not (env_terminated or env_truncated)
                    row = {"step": step, "success": delivered_success,
                           "actual_success": actual_success,
                           "goal_values": [bool(domain._eval_predicate(goal))
                                           for goal in domain.parsed_problem["goal_state"]],
                           "terminated": env_terminated,
                           "truncated": env_truncated or collector_truncated,
                           "environment_terminated": env_terminated,
                           "environment_truncated": env_truncated,
                           "collector_truncated": collector_truncated,
                           "reward": h.np.asarray(reward).tolist(),
                           "simulator_state_sha256": h.sha256(h.sim_state(vector).tobytes()).hexdigest()}
                    transitions.write(json.dumps(row) + "\n")
                    transitions.flush()
                    os.fsync(transitions.fileno())
                    if row["terminated"] or row["truncated"]:
                        episode.update(success=delivered_success, steps=step + 1,
                                       terminated=row["terminated"], truncated=row["truncated"],
                                       environment_terminated=env_terminated,
                                       environment_truncated=env_truncated,
                                       collector_truncated=collector_truncated)
                        break
                else:
                    raise RuntimeError("Native collector horizon did not record a terminal transition")
                save_arrays(folder / "terminal.npz", {
                    **flatten_observation(observation), "simulator_state": h.sim_state(vector)})
            episode["elapsed_seconds"] = time.monotonic() - started
            return episode
        except BaseException as error:
            atomic_json(folder / "infrastructure-error.json", {
                **episode, "success": None,
                "infrastructure_error": type(error).__name__ + ": " + str(error),
                "elapsed_seconds": time.monotonic() - started})
            raise
        finally:
            if vector is not None:
                vector.close()

    def close(self):
        self.policy = self.pre = self.post = self.env_pre = self.env_post = None
        self.suite = None
        gc.collect()
        # No shared cache or unrelated process is touched.
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
