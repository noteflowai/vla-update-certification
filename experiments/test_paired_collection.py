"""Crash/recovery and contamination controls using explicitly synthetic fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from paired_collection import PairCache, collect_methods

CONTEXT = {"seed": 2026093007, "family": "fixture", "update": "fixture",
           "scope": "synthetic engineering control, not policy evidence",
           "protocol_sha256": "c"*64, "evaluator_sha256": "d"*64,
           "old_pipeline_sha256": "e"*64, "new_pipeline_sha256": "f"*64,
           "states": [{"task": i, "init_state": 30, "scene_seed": 100+i} for i in range(2)]}


def fixture(identity):
    episode = {key: identity[key] for key in (
        "task", "init_state", "scene_seed", "policy_seed")}
    episode.update(success=True, infrastructure_error=None, reset_state_exact=True,
                   reset_inputs_exact=True, initial_input_sha256="b"*64,
                   steps=2, initial_state_sha256="a"*64)
    return {"identity": identity,
            **{side: {**episode, "pipeline_sha256": identity[f"{side}_pipeline_sha256"],
                      "evaluator_sha256": identity["evaluator_sha256"]}
               for side in ("old", "new")}}


class CollectionTests(unittest.TestCase):
    def test_interrupt_preserves_unresolved_attempt_and_never_reports_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            def interrupted(identity):
                raise KeyboardInterrupt("fixture interruption")
            with self.assertRaises(KeyboardInterrupt):
                collect_methods(folder, CONTEXT, interrupted, cap=40)
            summary = json.loads((Path(folder)/"summary.json").read_text())
            self.assertEqual(summary["status"], "producer_error")
            self.assertIn("KeyboardInterrupt", summary["error"])
            self.assertEqual(summary["committed_pair_policy_episodes"], 0)
            cache = PairCache(Path(folder)/"pairs", CONTEXT)
            with self.assertRaises(RuntimeError):
                cache.get(0, 0, fixture)

    def test_cache_is_idempotent_and_randomization_is_paired(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            calls = []
            def producer(identity):
                calls.append(identity)
                return fixture(identity)
            self.assertEqual(cache.get(0, 0, producer), (True, True))
            self.assertEqual(cache.get(0, 0, producer), (True, True))
            self.assertEqual(len(calls), 1)
            self.assertNotEqual(cache.identity(0, 0)["policy_seed"], cache.identity(0, 1)["policy_seed"])
            self.assertEqual(cache.identity(0, 0)["scene_seed"], 100)

    def test_partial_or_mismatched_pair_never_becomes_false_and_blocks_blind_retry(self):
        for failure in ("partial", "state", "inputs", "reset_inputs", "seed", "pipeline", "evaluator"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                cache = PairCache(folder, CONTEXT)
                def bad(identity):
                    record = fixture(identity)
                    if failure == "partial":
                        record["new"]["success"] = None
                    elif failure == "state":
                        record["new"]["initial_state_sha256"] = "b"*64
                    elif failure == "inputs":
                        record["new"]["initial_input_sha256"] = "c"*64
                    elif failure == "reset_inputs":
                        record["new"]["reset_inputs_exact"] = False
                    elif failure == "seed":
                        record["new"]["policy_seed"] += 1
                    elif failure == "pipeline":
                        record["new"]["pipeline_sha256"] = identity["old_pipeline_sha256"]
                    else:
                        record["new"]["evaluator_sha256"] = "a"*64
                    return record
                with self.assertRaises(ValueError):
                    cache.get(0, 0, bad)
                self.assertFalse(cache.path(0, 0).exists())
                with self.assertRaises(RuntimeError):
                    cache.get(0, 0, fixture)

    def test_resume_replays_exact_prefixes_without_recollecting(self):
        with tempfile.TemporaryDirectory() as folder:
            first = collect_methods(folder, CONTEXT, fixture, cap=40)
            def never_called(identity):
                raise AssertionError("Completed pair recollected")
            resumed = collect_methods(folder, CONTEXT, never_called, cap=40)
            self.assertEqual([engine.snapshot() for engine in first],
                             [engine.snapshot() for engine in resumed])
            summary = json.loads((Path(folder)/"summary.json").read_text())
            self.assertEqual(summary["committed_pair_policy_episodes"],
                             2*summary["committed_physical_pairs"])

    def test_changed_context_and_torn_journal_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            collect_methods(folder, CONTEXT, fixture, cap=40)
            with self.assertRaises(ValueError):
                PairCache(Path(folder)/"pairs", {**CONTEXT, "seed": 123})
            with (Path(folder)/"consumed-batches.jsonl").open("a") as output:
                output.write('{"method_index":')
            with self.assertRaises(ValueError):
                collect_methods(folder, CONTEXT, fixture, cap=40)

    def test_changed_consumed_evidence_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            collect_methods(folder, CONTEXT, fixture, cap=40)
            path = Path(folder)/"pairs/pair-0-0.json"
            record = json.loads(path.read_text())
            record["old"]["success"] = False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                collect_methods(folder, CONTEXT, fixture, cap=40)

    def test_context_alias_and_unconsumed_cache_corruption_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            context = json.loads(json.dumps(CONTEXT))
            cache = PairCache(folder, context)
            context["states"][0]["scene_seed"] = 999
            self.assertEqual(cache.identity(0, 0)["scene_seed"], 100)
            cache.get(0, 0, fixture)
            path = cache.path(0, 0)
            record = json.loads(path.read_text())
            record["new"]["success"] = False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                cache.get(0, 0, fixture)


if __name__ == "__main__":
    unittest.main()
