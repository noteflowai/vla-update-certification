"""Durable native batch boundary. Importing this module never loads a model."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

from paired_collection import PairCache, atomic_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "pi05": ("lerobot/pi05_libero_finetuned_v044",
             "8e174154ef5f6c60a8da12ae99c303d8963138c1", "relative"),
    "xvla": ("lerobot/xvla-libero",
             "12e8783e996944f5c97e490d37d4c145484ed70a", "absolute"),
}
UPDATES = ("baseline_reload", "w4", "w3", "steps2")


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def device_identity():
    """Bind resumed streams to one physical device/driver without importing CUDA."""
    text = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    name, uuid, driver = [part.strip() for part in text.strip().split(",")]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.split(",")[0].strip() not in ("0", uuid):
        raise ValueError("This frozen native protocol requires physical GPU zero")
    return {"gpu_name": name, "gpu_uuid_sha256": sha256(uuid.encode()).hexdigest(),
            "driver_version": driver, "logical_cuda_device": 0}


def bind_state_reset(folder, identity, episode, create=True):
    """The fixed state includes its complete reset inputs across every repeat."""
    folder = Path(folder)
    expected = {key: identity[key] for key in (
        "context_sha256", "state_index", "task", "init_state", "scene_seed")}
    expected.update({key: episode[key] for key in (
        "initial_state_sha256", "initial_input_sha256")})
    path = folder / f"state-{identity['state_index']}.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Fixed state's reset inputs changed across repeats or sides")
    elif create:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(path, expected)
    else:
        raise ValueError("Fixed state's reset-input binding is missing")


def verify_health(receipt_path, family):
    """Check complete episode evidence and lifecycle; never pool interrupted runs."""
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    execution = receipt.get("execution", {})
    if (receipt.get("family") != family or receipt.get("worker_started") is not True
            or execution.get("status") != "completed"
            or execution.get("clean_process_exit") is not True
            or type(execution.get("worker_exit_code")) is not int
            or execution.get("worker_exit_code") != 0
            or execution.get("episode_gate_passed") is not True
            or execution.get("episode_health_verdict") != "passed"):
        raise ValueError("Family health and clean worker exit are required")
    folder = Path(receipt["output"]).resolve()
    summary_path = folder / "summary.json"
    if digest_file(summary_path) != execution["summary_sha256"]:
        raise ValueError("Health summary changed")
    if json.loads((folder / "supervisor-outcome.json").read_text()) != execution:
        raise ValueError("Health lifecycle evidence differs")
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads(summary_path.read_text())
    model_id, revision, control = MODELS[family]
    if (manifest.get("family") != family or manifest.get("model_id") != model_id
            or manifest.get("revision") != revision
            or manifest.get("control_mode") != control
            or manifest.get("reset_mode") != "canonical"
            or manifest.get("episode_limit") != 520):
        raise ValueError("Health pipeline differs from the frozen native study")
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    cases = manifest["cases"]
    if len(cases) != 5 or len(rows) != 5 or len({r["task"] for r in cases}) != 5:
        raise ValueError("Health requires the complete five-state cohort")
    import numpy as np
    rng = np.random.default_rng(20260930)
    declared_cases = []
    for task in (0, 2, 4, 6, 8):
        state = int(rng.choice(np.arange(10, 50)))
        declared_cases.append({"task": task, "init_state": state,
                               "scene_seed": 3101 * 10007 + task * 101 + state,
                               "policy_seed": 310100 + task})
    if cases != declared_cases:
        raise ValueError("Health state selection differs from its fixed declaration")
    for row, case in zip(rows, cases, strict=True):
        for key in ("task", "init_state", "scene_seed", "policy_seed"):
            if row.get(key) != case[key]:
                raise ValueError("Health case or randomization differs")
        if (type(row.get("success")) is not bool
                or row.get("infrastructure_error") is not None
                or row.get("reset_state_exact") is not True
                or row.get("reset_inputs_exact") is not True
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= 520):
            raise ValueError("Incomplete or invalid health episode")
        snapshots = [json.loads((folder / f"reset-task{case['task']}-{i}.json").read_text())
                     for i in (0, 1)]
        initial = snapshots[0]
        if (snapshots[0] != snapshots[1]
                or object_digest(initial) != row.get("initial_input_sha256")
                or initial["actual_success"] is not False
                or initial["prompt"] != initial["bddl_prompt"] or not initial["goals"]
                or initial["controller_use_delta"] != [control == "relative"]):
            raise ValueError("Saved health reset inputs do not support the exact-reset claim")
        with np.load(folder / f"trace-task{case['task']}.npz", allow_pickle=False) as trace:
            actions, states = trace["actions"], trace["simulator_states"]
            if (actions.shape != (row["steps"], 1, 7)
                    or len(states) != row["steps"] + 1
                    or not np.isfinite(actions).all() or not np.isfinite(states).all()
                    or sha256(states[0].tobytes()).hexdigest() != row.get("initial_state_sha256")):
                raise ValueError("Saved health trace is incomplete or has a different initial state")
    successes = sum(row["success"] for row in rows)
    if (successes < 3 or summary.get("successes") != successes
            or summary.get("attempted") != 5 or summary.get("completed_episodes") != 5
            or summary.get("infrastructure_errors") != 0
            or summary.get("health_gate_passed") is not True
            or summary.get("model_health_verdict") != "passed"):
        raise ValueError("Health verdict is not supported by its episodes")
    for name, expected in manifest["sources"].items():
        if digest_file(name) != expected:
            raise ValueError("Health implementation/assets changed: " + name)
    factory = Path("/tmp/vlareg/.venv/lib/python3.12/site-packages/lerobot/policies/factory.py")
    if digest_file(factory) != manifest.get("installed_policy_factory_sha256"):
        raise ValueError("The health-validated policy factory changed")
    if family == "pi05":
        audit = json.loads((folder / "loader-verification.json").read_text())
        if (manifest.get("loader") != "direct_cuda_strict"
                or audit.get("strict_keys_and_shapes") is not True
                or audit.get("all_loaded_tensors_exact_after_official_cast") is not True):
            raise ValueError("Strict PI05 checkpoint loading is required")
    return {
        "receipt": str(receipt_path), "receipt_sha256": digest_file(receipt_path),
        "summary_sha256": digest_file(summary_path),
        "manifest_sha256": digest_file(folder / "manifest.json"),
        "episodes_sha256": digest_file(folder / "episodes.jsonl"),
        "model_weight_sha256": manifest["model_weight_sha256"],
        "packages": manifest["packages"],
        "gpu_name": receipt.get("gpu_name"),
    }


def freeze_context(family, update, receipt_path):
    """Freeze actual sources/assets before native outcomes; require the family gate."""
    from health_supervisor import reset_audit_probe
    if family not in MODELS or update not in UPDATES:
        raise ValueError("Undeclared family or update")
    health = verify_health(receipt_path, family)
    device = device_identity()
    if device["gpu_name"] != health["gpu_name"]:
        raise ValueError("Native device type differs from the family-health device")
    audit = reset_audit_probe(ROOT / "runs/reset-scoring-002/summary.json")
    if not audit["passed"]:
        raise ValueError("The source-matched complete reset audit is required")
    suite_path = ROOT / "protocols/closedloop-suite-001/manifest.json"
    suite = json.loads(suite_path.read_text())
    if len(suite["states"]) != 20:
        raise ValueError("The frozen twenty-state suite is required")
    for task in suite["inventory"]:
        for path_key, hash_key in (("init_file", "init_file_sha256"),
                                   ("bddl_file", "bddl_sha256")):
            if digest_file(task[path_key]) != task[hash_key]:
                raise ValueError("Benchmark asset changed")
    model_id, revision, control = MODELS[family]
    checkpoint = (Path.home() / ".cache/huggingface/hub" /
                  ("models--" + model_id.replace("/", "--")) / "snapshots" / revision)
    assets = [checkpoint / "config.json"]
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = checkpoint / name
        assets.append(path)
        for step in json.loads(path.read_text())["steps"]:
            if "state_file" in step:
                assets.append(checkpoint / step["state_file"])
    sources = [ROOT / "experiments" / name for name in (
        "native_collection.py", "native_rollout.py", "run_native_collection.py", "health_supervisor.py",
        "paired_collection.py", "closedloop_decisions.py", "admission.py",
        "joint_admission.py", "coarse_admission.py", "run_admission.py",
        "update_variants.py", "run_closedloop_health.py", "strict_pi05_loader.py",
        "canonical_reset.py", "canonical_reset_wrapper.py", "libero_audit_core.py",
        "audit_reset_scoring.py", "NATIVE_COLLECTION_003.md")]
    # Bind the installed policy and processor implementation, not just our wrapper.
    installed = Path("/tmp/vlareg/.venv/lib/python3.12/site-packages")
    sources.extend(sorted((installed / "lerobot/policies" / family).glob("*.py")))
    sources.extend(sorted((installed / "lerobot/processor").glob("*.py")))
    sources.extend(installed / relative for relative in (
        "lerobot/policies/factory.py", "lerobot/envs/libero.py",
        "lerobot/envs/factory.py", "lerobot/envs/utils.py"))
    import importlib.metadata
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions(path=[str(installed)])}
    packages = {**health["packages"],
                **{name: versions[name] for name in ("numpy", "scipy", "torch", "safetensors")}}
    for name, expected in health["packages"].items():
        if versions[name] != expected:
            raise ValueError("A health-validated dependency changed: " + name)
    evaluator = {"sources": {str(p): digest_file(p) for p in sources},
                 "packages": packages, "reset_audit": audit, "device_identity": device,
                 "episode_limit": 520, "reset": "canonical_exact_double_reset_and_cross_repeat_binding",
                 "compile_model": False, "tf32": False,
                 "deterministic_algorithms": "warn_only"}
    pipeline = {"family": family, "model_id": model_id, "revision": revision,
                "checkpoint": str(checkpoint), "control_mode": control,
                "model_weight_sha256": health["model_weight_sha256"],
                "assets": {str(p): digest_file(p) for p in assets},
                "loader": "direct_cuda_strict" if family == "pi05" else "official",
                "flow_steps": 10, "chunk_size": 50 if family == "pi05" else 30,
                "n_action_steps": 50 if family == "pi05" else 30}
    new_pipeline = {**pipeline, "update": update,
                    "flow_steps": 2 if update == "steps2" else 10,
                    "weight_bits": {"w4": 4, "w3": 3}.get(update)}
    old_pipeline = {**pipeline, "update": "baseline_reload", "weight_bits": None}
    protocol = {"suite_sha256": digest_file(suite_path), "family": family,
                "update": update, "cap_per_method": 1000, "alpha": .05 / (8 * 6),
                "delta": .02, "epsilon": .02, "health": health,
                "physical_budget_seconds": 12 * 3600,
                "cohort_seed_rule": "2026100100 + 4*family_index + update_index",
                "family_order": ["pi05", "xvla"], "update_order": list(UPDATES)}
    return {"family": family, "update": update,
            "seed": 2026100100 + 4 * ["pi05", "xvla"].index(family) + UPDATES.index(update),
            "states": suite["states"], "protocol": protocol, "evaluator": evaluator,
            "old_pipeline": old_pipeline, "new_pipeline": new_pipeline,
            "protocol_sha256": object_digest(protocol),
            "evaluator_sha256": object_digest(evaluator),
            "old_pipeline_sha256": object_digest(old_pipeline),
            "new_pipeline_sha256": object_digest(new_pipeline)}


class NativePairCache(PairCache):
    """Bind committed outcomes to retained native evidence before replay/use."""
    def __init__(self, folder, context):
        super().__init__(folder, context)
        self.verified_files = set()

    def verify_file(self, path, expected):
        stat = path.stat()
        signature = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
        if signature not in self.verified_files:
            if digest_file(path) != expected:
                raise ValueError("Native raw evidence changed: " + str(path))
            self.verified_files.add(signature)

    def validate(self, record, identity):
        result = PairCache.validate(record, identity)
        for side in ("old", "new"):
            episode = record[side]
            bind_state_reset(self.folder.parent / "raw/state-resets", identity, episode,
                             create=False)
            folder = (self.folder.parent / "raw" /
                      f"episode-{identity['state_index']}-{identity['repeat']}-{side}").resolve()
            if episode.get("raw_folder") != str(folder) or not episode.get("raw_evidence"):
                raise ValueError("Native side evidence is missing or belongs to another episode")
            attempt = json.loads((folder / "attempt.json").read_text())
            if attempt.get("status") != "completed":
                raise ValueError("Native side evidence is not committed")
            self.verify_file(folder / "episode.json", attempt["episode_sha256"])
            if json.loads((folder / "episode.json").read_text()) != episode:
                raise ValueError("Native pair differs from the stored side result")
            for relative, expected in episode["raw_evidence"].items():
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Native evidence path escapes the side directory")
                self.verify_file(folder / path, expected)
        return result


class NativeBatchProducer:
    """One policy at a time; persist every side before returning any paired batch."""
    def __init__(self, folder, context, backend, wall_budget=12 * 3600):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.context = deepcopy(context)
        self.backend = backend
        self.cache_identity = PairCache(self.folder / "identities", context)
        self.started = time.monotonic()
        self.wall_budget = wall_budget
        if not 0 < wall_budget <= 12 * 3600:
            raise ValueError("Native collection must respect the bounded feasibility phase")
        marker = self.folder / "context.json"
        if marker.exists() and json.loads(marker.read_text()) != self.context:
            raise ValueError("Native context changed")
        if not marker.exists():
            atomic_json(marker, self.context)

    def event(self, phase, **details):
        row = {"at": datetime.now(timezone.utc).isoformat(), "phase": phase, **details}
        with (self.folder / "physical-events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def produce_many(self, identities):
        expected = deepcopy(identities)
        if not expected or len({(i["state_index"], i["repeat"]) for i in expected}) != len(expected):
            raise ValueError("A nonempty unique requested batch is required")
        for identity in expected:
            if identity != self.cache_identity.identity(identity["state_index"], identity["repeat"]):
                raise ValueError("Native randomization differs from the frozen context")
            for side in ("old", "new"):
                path = self.folder / f"episode-{identity['state_index']}-{identity['repeat']}-{side}"
                if path.exists():
                    raise RuntimeError("Raw side attempt exists; reconcile before retry")
        batch_id = object_digest(expected)
        batch_path = self.folder / f"batch-{batch_id}.json"
        atomic_json(batch_path, {"status": "started", "identities": expected})
        records = [{"identity": i} for i in expected]
        try:
            for side in ("old", "new"):
                if time.monotonic() - self.started >= self.wall_budget:
                    raise TimeoutError("Declared native physical budget exhausted")
                self.event("pipeline_loading", side=side, batch=batch_id)
                try:
                    report = self.backend.load(side, deepcopy(self.context), self.folder, batch_id)
                    self.event("pipeline_ready", side=side, batch=batch_id, report=report)
                    for identity, record in zip(expected, records, strict=True):
                        if time.monotonic() - self.started >= self.wall_budget:
                            raise TimeoutError("Declared native physical budget exhausted")
                        path = self.folder / f"episode-{identity['state_index']}-{identity['repeat']}-{side}"
                        path.mkdir()
                        claim = {"status": "started", "identity": identity, "side": side,
                                 "pipeline_sha256": identity[f"{side}_pipeline_sha256"]}
                        atomic_json(path / "attempt.json", claim)
                        self.event("episode_started", side=side, identity=identity, raw_folder=str(path))
                        try:
                            episode = self.backend.run_episode(
                                side, deepcopy(identity), path,
                                self.started + self.wall_budget)
                            # Use the existing strict episode validator before accepting a side.
                            mirror = {**episode, "pipeline_sha256":
                                      identity[f"{'new' if side == 'old' else 'old'}_pipeline_sha256"]}
                            PairCache.validate({"identity": identity, side: episode,
                                                ("new" if side == "old" else "old"): mirror}, identity)
                            bind_state_reset(self.folder / "state-resets", identity, episode)
                            evidence = {str(p.relative_to(path)): digest_file(p)
                                        for p in sorted(path.rglob("*"))
                                        if p.is_file() and p.name not in ("attempt.json", "episode.json")}
                            if not evidence:
                                raise ValueError("Native episode has no durable raw evidence")
                            episode = {**episode, "raw_folder": str(path),
                                       "raw_evidence": evidence}
                            atomic_json(path / "episode.json", episode)
                            atomic_json(path / "attempt.json", {**claim, "status": "completed",
                                        "episode_sha256": digest_file(path / "episode.json")})
                        except BaseException as error:
                            atomic_json(path / "attempt.json", {**claim, "status": "error",
                                        "error": type(error).__name__ + ": " + str(error)})
                            self.event("episode_error", side=side, identity=identity,
                                       error=type(error).__name__ + ": " + str(error))
                            raise
                        record[side] = episode
                        self.event("episode_completed", side=side, identity=identity,
                                   success=episode["success"], steps=episode["steps"])
                finally:
                    self.backend.close()
                    self.event("pipeline_released", side=side, batch=batch_id)
            for record, identity in zip(records, expected, strict=True):
                PairCache.validate(record, identity)
            atomic_json(batch_path, {"status": "completed", "identities": expected,
                                    "records_sha256": object_digest(records)})
            return records
        except BaseException as error:
            atomic_json(batch_path, {"status": "error", "identities": expected,
                                    "error": type(error).__name__ + ": " + str(error)})
            self.event("batch_error", batch=batch_id,
                       error=type(error).__name__ + ": " + str(error))
            raise
