"""Native orchestration and gating controls. All episode fixtures are synthetic."""
from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from native_collection import (NativeBatchProducer, NativePairCache, MODELS, verify_health,
                               digest_file, object_digest, bind_state_reset, device_identity)
from paired_collection import PairCache, atomic_json
from run_native_collection import PhysicalBudget, read_native_summary, check_previous_launches
from health_supervisor import supervise
from native_rollout import LiberoNativeBackend
from test_paired_collection import CONTEXT, fixture


class BackendFixture:
    def __init__(self, failure=None):
        self.failure = failure
        self.active = None
        self.events = []

    def load(self, side, context, folder, batch_id):
        if self.active is not None:
            raise AssertionError("Two policies were retained")
        self.active = side
        self.events.append(("load", side))
        if self.failure == "load" and side == "new":
            raise RuntimeError("Synthetic loading interruption")
        return {"scope": "synthetic backend fixture"}

    def run_episode(self, side, identity, folder, deadline):
        self.events.append(("episode", side, identity["state_index"]))
        self.assert_active(side)
        atomic_json(folder / "raw-fixture.json", {"scope": "synthetic raw evidence"})
        if self.failure == "interrupt" and side == "new":
            raise KeyboardInterrupt("Synthetic episode interruption")
        episode = fixture(identity)[side]
        if self.failure == "reset" and side == "new":
            episode["initial_input_sha256"] = "c" * 64
        if self.failure == "mutate":
            identity["policy_seed"] += 1
            episode = fixture(identity)[side]
        return episode

    def assert_active(self, side):
        if side != self.active:
            raise AssertionError("Episode did not use its provisioned side")

    def close(self):
        self.events.append(("close", self.active))
        side, self.active = self.active, None
        if self.failure == "close" and side == "new":
            raise RuntimeError("Synthetic release failure")


def health_fixture(folder):
    """A fabricated five-row evidence bundle for validator tests, never research data."""
    folder = Path(folder)
    family = "pi05"
    model_id, revision, control = MODELS[family]
    rng = np.random.default_rng(20260930)
    cases = []
    for task in (0, 2, 4, 6, 8):
        state = int(rng.choice(np.arange(10, 50)))
        cases.append({"task": task, "init_state": state,
                      "scene_seed": 3101 * 10007 + task * 101 + state,
                      "policy_seed": 310100 + task})
    source = folder / "synthetic-source.txt"
    source.write_text("Synthetic gate control, not executed policy source.\n")
    manifest = {"scope": "synthetic fixture, not model health evidence",
                "family": family, "model_id": model_id, "revision": revision,
                "control_mode": control, "reset_mode": "canonical", "episode_limit": 520,
                "loader": "direct_cuda_strict", "cases": cases,
                "sources": {str(source): digest_file(source)},
                "installed_policy_factory_sha256": digest_file(
                    "/tmp/vlareg/.venv/lib/python3.12/site-packages/lerobot/policies/factory.py"),
                "model_weight_sha256": "a" * 64, "packages": {}}
    rows = [{**case, "success": True, "infrastructure_error": None,
             "reset_state_exact": True, "reset_inputs_exact": True, "steps": 2}
            for case in cases]
    import hashlib
    for row in rows:
        snapshot = {"actual_success": False, "prompt": "fixture", "bddl_prompt": "fixture",
                    "goals": [["fixture"]], "controller_use_delta": [True]}
        for index in (0, 1):
            atomic_json(folder / f"reset-task{row['task']}-{index}.json", snapshot)
        states = np.zeros((3, 1), dtype=np.float64)
        row["initial_state_sha256"] = hashlib.sha256(states[0].tobytes()).hexdigest()
        row["initial_input_sha256"] = object_digest(snapshot)
        np.savez_compressed(folder / f"trace-task{row['task']}.npz",
                            actions=np.zeros((2, 1, 7)), simulator_states=states)
    atomic_json(folder / "manifest.json", manifest)
    (folder / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(folder / "summary.json", {
        "attempted": 5, "successes": 5, "completed_episodes": 5,
        "infrastructure_errors": 0, "health_gate_passed": True, "model_health_verdict": "passed"})
    atomic_json(folder / "loader-verification.json", {
        "strict_keys_and_shapes": True, "all_loaded_tensors_exact_after_official_cast": True})
    execution = {"status": "completed", "clean_process_exit": True, "worker_exit_code": 0,
                 "episode_gate_passed": True, "episode_health_verdict": "passed",
                 "summary_sha256": digest_file(folder / "summary.json")}
    atomic_json(folder / "supervisor-outcome.json", execution)
    receipt = {"family": family, "worker_started": True,
               "output": str(folder), "execution": execution}
    atomic_json(folder / "receipt.json", receipt)
    return folder / "receipt.json"


class NativeTests(unittest.TestCase):
    def test_device_identity_binds_driver_and_checks_physical_device_selection(self):
        with patch("native_collection.subprocess.check_output",
                   return_value="Fixture GPU, GPU-fixture, 123.4\n"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                first = device_identity()
                self.assertEqual(first["gpu_name"], "Fixture GPU")
                self.assertNotIn("GPU-fixture", json.dumps(first))
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                with self.assertRaises(ValueError):
                    device_identity()
        with patch("native_collection.subprocess.check_output",
                   return_value="Fixture GPU, GPU-fixture, 124.0\n"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                self.assertNotEqual(first, device_identity())

    def test_native_order_one_policy_at_a_time_durable_sides_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            backend = BackendFixture()
            producer = NativeBatchProducer(root / "raw", CONTEXT, backend)
            cache = PairCache(root / "pairs", CONTEXT)
            self.assertEqual(cache.get_many([(0, 0), (1, 0)], producer.produce_many),
                             [(True, True)] * 2)
            self.assertEqual(backend.events, [
                ("load", "old"), ("episode", "old", 0), ("episode", "old", 1), ("close", "old"),
                ("load", "new"), ("episode", "new", 0), ("episode", "new", 1), ("close", "new")])
            before = list(backend.events)
            self.assertEqual(cache.get_many([(0, 0), (1, 0)], producer.produce_many),
                             [(True, True)] * 2)
            self.assertEqual(backend.events, before)
            for path in (root / "raw").glob("episode-*"):
                episode = json.loads((path / "episode.json").read_text())
                self.assertEqual(json.loads((path / "attempt.json").read_text())["status"], "completed")
                for relative, expected in episode["raw_evidence"].items():
                    self.assertEqual(digest_file(path / relative), expected)

    def test_interrupt_loading_release_or_pair_reset_failure_never_commits(self):
        for failure in ("interrupt", "load", "close", "reset"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                backend = BackendFixture(failure)
                producer = NativeBatchProducer(root / "raw", CONTEXT, backend)
                cache = PairCache(root / "pairs", CONTEXT)
                with self.assertRaises((RuntimeError, KeyboardInterrupt, ValueError)):
                    cache.get_many([(0, 0)], producer.produce_many)
                self.assertIsNone(backend.active)
                self.assertFalse(cache.path(0, 0).exists())
                self.assertTrue((root / "raw/episode-0-0-old/episode.json").exists())
                with self.assertRaises(RuntimeError):
                    cache.get_many([(0, 0)], producer.produce_many)
                events = [json.loads(line) for line in
                          (root / "raw/physical-events.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["phase"], "batch_error")

    def test_expected_identity_cannot_be_mutated_and_raw_attempt_blocks_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            producer = NativeBatchProducer(folder, CONTEXT, BackendFixture("mutate"))
            identity = producer.cache_identity.identity(0, 0)
            with self.assertRaises(ValueError):
                producer.produce_many([identity])
            self.assertEqual(identity["policy_seed"], producer.cache_identity.identity(0, 0)["policy_seed"])
            with self.assertRaises(RuntimeError):
                producer.produce_many([identity])

    def test_changed_native_raw_evidence_is_rejected_before_cached_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            producer = NativeBatchProducer(root / "raw", CONTEXT, BackendFixture())
            cache = NativePairCache(root / "pairs", CONTEXT)
            self.assertEqual(cache.get_many([(0, 0)], producer.produce_many), [(True, True)])
            (root / "raw/episode-0-0-old/raw-fixture.json").write_text("Changed raw fixture\n")
            with self.assertRaisesRegex(ValueError, "raw evidence changed"):
                cache.get_many([(0, 0)], lambda values: self.fail("Corruption triggered inference"))

    def test_same_paired_reset_is_not_enough_when_the_state_drifts_between_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            class Drifting(BackendFixture):
                def run_episode(self, side, identity, path, deadline):
                    result = super().run_episode(side, identity, path, deadline)
                    if identity["repeat"] == 1:
                        # Both sides would agree within this pair, but this is another scene.
                        result["initial_input_sha256"] = "c" * 64
                    return result
            producer = NativeBatchProducer(root / "raw", CONTEXT, Drifting())
            cache = PairCache(root / "pairs", CONTEXT)
            with self.assertRaisesRegex(ValueError, "across repeats"):
                cache.get_many([(0, 0), (0, 1)], producer.produce_many)
            self.assertEqual(list((root / "pairs").glob("pair-*")), [])

    def test_state_binding_accepts_new_seeds_but_rejects_changed_or_missing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = PairCache(root / "identities", CONTEXT)
            first, next_repeat = cache.identity(0, 0), cache.identity(0, 1)
            bind_state_reset(root / "state-resets", first, fixture(first)["old"])
            bind_state_reset(root / "state-resets", next_repeat, fixture(next_repeat)["new"])
            changed = {**fixture(next_repeat)["old"], "initial_state_sha256": "c" * 64}
            with self.assertRaises(ValueError):
                bind_state_reset(root / "state-resets", next_repeat, changed)
            (root / "state-resets/state-0.json").unlink()
            with self.assertRaises(ValueError):
                bind_state_reset(root / "state-resets", first, fixture(first)["old"], create=False)

    def test_invalid_batch_never_loads_a_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = BackendFixture()
            producer = NativeBatchProducer(folder, CONTEXT, backend)
            identity = producer.cache_identity.identity(0, 0)
            for identities in ([], [identity, identity], [{**identity, "scene_seed": 999}]):
                with self.assertRaises(ValueError):
                    producer.produce_many(identities)
            self.assertEqual(backend.events, [])

    def test_complete_fixture_health_passes_but_partial_or_altered_evidence_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = health_fixture(folder)
            self.assertEqual(verify_health(receipt, "pi05")["model_weight_sha256"], "a" * 64)
            with self.assertRaises(ValueError):
                verify_health(receipt, "xvla")
        for failure in ("exit", "summary", "partial", "selection", "source", "strict",
                        "reset", "success", "snapshot", "trace"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                receipt = health_fixture(folder)
                root = Path(folder)
                if failure == "exit":
                    value = json.loads(receipt.read_text())
                    value["execution"]["clean_process_exit"] = False
                    atomic_json(receipt, value)
                elif failure == "summary":
                    atomic_json(root / "summary.json", {"health_gate_passed": True})
                elif failure == "source":
                    (root / "synthetic-source.txt").write_text("Changed source")
                elif failure == "strict":
                    atomic_json(root / "loader-verification.json", {})
                elif failure == "snapshot":
                    atomic_json(root / "reset-task0-1.json", {})
                elif failure == "trace":
                    np.savez_compressed(root / "trace-task0.npz",
                                        actions=np.zeros((1, 1, 7)), simulator_states=np.zeros((2, 1)))
                else:
                    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
                    if failure == "partial":
                        rows.pop()
                    elif failure == "selection":
                        rows[0]["init_state"] += 1
                        manifest = json.loads((root / "manifest.json").read_text())
                        manifest["cases"][0]["init_state"] += 1
                        atomic_json(root / "manifest.json", manifest)
                    elif failure == "reset":
                        rows[0]["reset_inputs_exact"] = False
                    else:
                        rows[0]["success"] = None
                    (root / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    verify_health(receipt, "pi05")

    def test_shared_budget_reservations_include_crashes_and_reject_overspending(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = PhysicalBudget(Path(folder) / "budget.json")
            budget.reserve("first", 30000)
            with self.assertRaises(RuntimeError):
                budget.reserve("second", 14000)
            budget.finish("first", 10000)
            budget.reserve("second", 33000)
            with self.assertRaises(ValueError):
                budget.reserve("second", 1)
            with self.assertRaises(ValueError):
                budget.finish("second", 34000)
            with self.assertRaises(RuntimeError):
                budget.reserve("after-unresolved-crash", 201)
            self.assertEqual(json.loads((Path(folder) / "budget.json").read_text())[
                "reservations"]["second"]["status"], "reserved")

    def test_native_summary_reader_does_not_accept_health_or_partial_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            self.assertIsNone(read_native_summary(path))
            path.write_text('{"status":')
            self.assertIsNone(read_native_summary(path))
            atomic_json(path, {"health_gate_passed": True, "model_health_verdict": "passed"})
            self.assertIsNone(read_native_summary(path))
            atomic_json(path, {"status": "native_error", "error": "synthetic"})
            self.assertEqual(read_native_summary(path)["status"], "native_error")

    def test_typed_native_summary_uses_the_supervised_clean_exit_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "summary.json"
            command = [sys.executable, "-c",
                       "import json,pathlib; pathlib.Path(" + repr(str(path)) +
                       ").write_text(json.dumps({'status':'first_batch_completed'}))"]
            result = supervise(command, path, root / "worker.log", 10, 2,
                               summary_reader=read_native_summary)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["clean_process_exit"])
            self.assertIsNone(result["episode_gate_passed"])
            self.assertEqual(result["episode_health_verdict"], "not_assessed")

    def test_cache_only_retry_cannot_whitewash_a_failed_or_unresolved_native_launch(self):
        for failure in (None, "unclean", "running", "native_error", "changed_summary"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                check_previous_launches(root, CONTEXT)
                launch = root / "launches/fixture"
                launch.mkdir(parents=True)
                summary = {"status": "first_batch_completed", "context_sha256": object_digest(CONTEXT)}
                if failure == "native_error":
                    summary["status"] = "native_error"
                atomic_json(launch / "summary.json", summary)
                receipt = {"status": "finished", "context_sha256": object_digest(CONTEXT),
                           "execution": {"status": "completed", "worker_exit_code": 0,
                                         "clean_process_exit": failure != "unclean",
                                         "summary_sha256": digest_file(launch / "summary.json")}}
                if failure == "running":
                    receipt["status"] = "running"
                atomic_json(launch / "receipt.json", receipt)
                if failure == "changed_summary":
                    atomic_json(launch / "summary.json", {**summary, "changed": True})
                if failure is None:
                    check_previous_launches(root, CONTEXT)
                else:
                    with self.assertRaisesRegex(RuntimeError, "reconciliation"):
                        check_previous_launches(root, CONTEXT)

    def test_actual_episode_kernel_persists_inputs_actions_checker_and_terminal(self):
        self.exercise_episode_kernel(False)

    def test_actual_episode_kernel_rejects_checker_mismatch_as_infrastructure(self):
        self.exercise_episode_kernel(True)

    def exercise_episode_kernel(self, checker_mismatch):
        """Exercise the real kernel with a toy vector API, without a model or simulator."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = PairCache(root / "identities", CONTEXT).identity(0, 0)
            episode_folder = root / "raw/episode-0-0-old"
            episode_folder.mkdir(parents=True)
            class Tensor:
                def __init__(self, value):
                    self.value = value
                def float(self):
                    return self
                def to(self, device):
                    return self
                def numpy(self):
                    return self.value
            class Vector:
                def __init__(self, constructors, **kwargs):
                    self.steps = 0
                    domain = SimpleNamespace(parsed_problem={"goal_state": [["fixture"]]},
                                             _eval_predicate=lambda goal: bool(self.steps))
                    control = SimpleNamespace(env=domain, check_success=lambda: bool(self.steps))
                    environment = SimpleNamespace(_env=control)
                    environment.unwrapped = environment
                    self.envs = [environment]
                    self.closed = False
                def obs(self):
                    return {"camera": np.zeros((1, 2, 2, 3), dtype=np.uint8),
                            "state": np.zeros((1, 7), dtype=np.float32)}
                def reset(self, seed):
                    self.steps = 0
                    return self.obs(), {}
                def call(self, name):
                    return ("fixture prompt",)
                def step(self, action):
                    self.steps += 1
                    return (self.obs(), np.array([1.]), np.array([True]), np.array([False]),
                            {"is_success": np.array([not checker_mismatch])})
                def close(self):
                    self.closed = True
            vectors = []
            def create_vector(*args, **kwargs):
                vector = Vector(*args, **kwargs)
                vectors.append(vector)
                return vector
            import hashlib
            fake_health = SimpleNamespace(
                np=np, sha256=hashlib.sha256, ACTION="action",
                torch=SimpleNamespace(manual_seed=lambda seed: None, inference_mode=nullcontext),
                gym=SimpleNamespace(vector=SimpleNamespace(
                    SyncVectorEnv=create_vector, AutoresetMode=SimpleNamespace(NEXT_STEP=1))),
                make_sub_env=lambda *args: None,
                memory_available=lambda: 20 * 1024 ** 3,
                sim_state=lambda vector: np.array([vector.steps], dtype=np.float64),
                unbatch_observation=lambda value: value,
                preprocess_observation=lambda value: dict(value))
            snapshot = {"actual_success": False, "prompt": "fixture", "bddl_prompt": "fixture",
                        "goals": [["fixture"]], "controller_use_delta": [True]}
            fake_reset = SimpleNamespace(CanonicalResetWrapper=lambda value: value)
            fake_audit = SimpleNamespace(snapshot=lambda *args: deepcopy(snapshot))
            backend = LiberoNativeBackend()
            backend.suite = None
            backend.env_cfg = SimpleNamespace(control_mode="relative")
            backend.pre = backend.env_pre = backend.post = backend.env_post = lambda value: value
            backend.policy = SimpleNamespace(
                reset=lambda: None, select_action=lambda value: Tensor(np.zeros((1, 7))))
            import time
            with patch.dict(sys.modules, {"run_closedloop_health": fake_health,
                                          "canonical_reset_wrapper": fake_reset,
                                          "libero_audit_core": fake_audit}):
                if checker_mismatch:
                    with self.assertRaisesRegex(RuntimeError, "checker"):
                        backend.run_episode("old", identity, episode_folder, time.monotonic() + 10)
                    error = json.loads((episode_folder / "infrastructure-error.json").read_text())
                    self.assertIsNone(error["success"])
                else:
                    result = backend.run_episode("old", identity, episode_folder, time.monotonic() + 10)
                    self.assertTrue(result["success"])
                    self.assertEqual(result["steps"], 1)
                    self.assertTrue(result["reset_inputs_exact"])
                    for name in ("input-000.npz", "action-000.json", "terminal.npz",
                                 "reset-0.json", "reset-1.json", "transitions.jsonl"):
                        self.assertTrue((episode_folder / name).is_file())
                    with np.load(episode_folder / "input-000.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["observation/camera"].shape, (1, 2, 2, 3))
                    transition = json.loads((episode_folder / "transitions.jsonl").read_text())
                    self.assertEqual(transition["actual_success"], transition["success"])
                self.assertTrue(vectors[0].closed)


if __name__ == "__main__":
    unittest.main()
