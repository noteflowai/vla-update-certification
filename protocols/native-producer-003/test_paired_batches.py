"""Durability and exact consumed-prefix controls; synthetic episodes only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paired_collection
from paired_collection import PairCache, collect_methods
from test_paired_collection import CONTEXT, fixture


class BatchTests(unittest.TestCase):
    def test_all_attempts_are_durable_before_one_producer_call_and_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            calls = []
            def batch(identities):
                calls.append(identities)
                for identity in identities:
                    attempt = Path(folder)/f"attempt-{identity['state_index']}-{identity['repeat']}.json"
                    self.assertEqual(json.loads(attempt.read_text())["status"], "started")
                return [fixture(identity) for identity in identities]
            keys = [(1, 0), (0, 0)]
            self.assertEqual(cache.get_many(keys, batch), [(True, True)]*2)
            self.assertEqual(cache.get_many(keys, batch), [(True, True)]*2)
            self.assertEqual(len(calls), 1)
            self.assertEqual([r["state_index"] for r in calls[0]], [1, 0])

    def test_only_missing_requested_pairs_enter_producer(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            cache.get(0, 0, fixture)
            def batch(identities):
                self.assertEqual([(i["state_index"], i["repeat"]) for i in identities], [(1, 0)])
                record = fixture(identities[0])
                record["new"]["success"] = False
                return [record]
            self.assertEqual(cache.get_many([(0, 0), (1, 0)], batch),
                             [(True, True), (True, False)])

    def test_duplicate_or_invalid_inventory_does_not_start_attempts(self):
        for keys in [[(0, 0), (0, 0)], [(0, 0), (9, 0)], [(0, 0, 0)]]:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as folder:
                cache = PairCache(folder, CONTEXT)
                with self.assertRaises(ValueError):
                    cache.get_many(keys, lambda identities: self.fail("Invalid batch ran"))
                self.assertEqual(list(Path(folder).glob("attempt-*")), [])

    def test_unresolved_member_blocks_entire_batch_before_new_work(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            def fail(identity):
                raise RuntimeError("Saved raw attempt remains unresolved")
            with self.assertRaises(RuntimeError):
                cache.get(1, 0, fail)
            with self.assertRaises(RuntimeError):
                cache.get_many([(0, 0), (1, 0)], lambda identities: self.fail("Blind retry"))
            self.assertFalse((Path(folder)/"attempt-0-0.json").exists())

    def test_short_reordered_or_invalid_batch_is_never_partially_accepted(self):
        for failure in ["short", "reordered", "reset"]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                cache = PairCache(folder, CONTEXT)
                def batch(identities):
                    records = [fixture(identity) for identity in identities]
                    if failure == "short":
                        return records[:1]
                    if failure == "reordered":
                        return records[::-1]
                    records[1]["new"]["initial_input_sha256"] = "c"*64
                    return records
                with self.assertRaises(ValueError):
                    cache.get_many([(0, 0), (1, 0)], batch)
                self.assertEqual(list(Path(folder).glob("pair-*")), [])
                for path in Path(folder).glob("attempt-*"):
                    self.assertEqual(json.loads(path.read_text())["status"], "error")

    def test_interrupted_generator_preserves_attempts_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            def batch(identities):
                yield fixture(identities[0])
                raise KeyboardInterrupt("Native collection interrupted")
            with self.assertRaises(KeyboardInterrupt):
                cache.get_many([(0, 0), (1, 0)], batch)
            self.assertEqual(list(Path(folder).glob("pair-*")), [])
            with self.assertRaises(RuntimeError):
                cache.get_many([(0, 0), (1, 0)], lambda identities: self.fail("Blind retry"))

    def test_producer_cannot_mutate_expected_randomization_in_single_or_batch_mode(self):
        for mode in ["single", "batch"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                cache = PairCache(folder, CONTEXT)
                def mutate(identity):
                    identity["scene_seed"] += 1
                    return fixture(identity)
                with self.assertRaises(ValueError):
                    if mode == "single":
                        cache.get(0, 0, mutate)
                    else:
                        cache.get_many([(0, 0)], lambda identities: [mutate(i) for i in identities])
                self.assertFalse(cache.path(0, 0).exists())
                self.assertEqual(cache.identity(0, 0)["scene_seed"], 100)

    def test_commit_failure_preserves_completed_pair_and_blocks_unresolved_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PairCache(folder, CONTEXT)
            original = paired_collection.atomic_json
            def disk_error(path, value):
                if path.name == "pair-1-0.json":
                    raise OSError("Synthetic commit failure")
                return original(path, value)
            with patch.object(paired_collection, "atomic_json", disk_error):
                with self.assertRaises(OSError):
                    cache.get_many([(0, 0), (1, 0)],
                                   lambda identities: [fixture(i) for i in identities])
            self.assertEqual(cache.get_many([(0, 0)], None), [(True, True)])
            with self.assertRaises(RuntimeError):
                cache.get_many([(0, 0), (1, 0)], lambda identities: self.fail("Blind retry"))

    def test_batch_interface_preserves_every_methods_consumed_prefix_and_resume(self):
        with tempfile.TemporaryDirectory() as single, tempfile.TemporaryDirectory() as batched:
            class Producer:
                def produce_many(self, identities):
                    return [fixture(identity) for identity in identities]
            left = collect_methods(single, CONTEXT, fixture, cap=40)
            right = collect_methods(batched, CONTEXT, Producer(), cap=40)
            self.assertEqual([e.snapshot() for e in left], [e.snapshot() for e in right])
            self.assertEqual((Path(single)/"consumed-batches.jsonl").read_text(),
                             (Path(batched)/"consumed-batches.jsonl").read_text())
            class Never:
                def produce_many(self, identities):
                    raise AssertionError("Committed outcomes recollected")
            resumed = collect_methods(batched, CONTEXT, Never(), cap=40)
            self.assertEqual([e.snapshot() for e in right], [e.snapshot() for e in resumed])


if __name__ == "__main__":
    unittest.main()
