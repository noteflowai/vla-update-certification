"""Real subprocess controls for the post-summary timeout boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from health_supervisor import supervise, reset_audit_probe, main

SUMMARY = {"model_health_verdict": "passed", "health_gate_passed": True}


class SupervisorTests(unittest.TestCase):
    def test_clean_exit_with_failed_episode_gate_is_not_model_readiness(self):
        failed = {"model_health_verdict": "failed", "health_gate_passed": False}
        result = self.run_worker("import pathlib\n"
                                 f"pathlib.Path(SUMMARY_PATH).write_text({json.dumps(failed)!r})\n")
        self.assertTrue(result["clean_process_exit"])
        self.assertFalse(result["episode_gate_passed"])
        self.assertEqual(result["episode_health_verdict"], "failed")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch("sys.argv", ["health_supervisor", "--family", "pi05",
                                   "--output", str(folder/"run"), "--receipt", str(folder/"receipt.json")]), \
                 patch("health_supervisor.resource_probe", return_value={"allowed": True}), \
                 patch("health_supervisor.reset_audit_probe", return_value={"passed": True}), \
                 patch("health_supervisor.supervise", return_value=result):
                self.assertEqual(main(), 5)

    def test_missing_reset_audit_blocks_model_start(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(reset_audit_probe(Path(folder)/"summary.json")["passed"])

    def test_partial_reset_audit_cannot_satisfy_the_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"summary.json"
            path.write_text(json.dumps({"audit_passed": True, "declared": 45,
                                       "completed": 16, "passed_rows": 16, "errors": 0}))
            (path.parent/"manifest.json").write_text('{"sources":{}}')
            result = reset_audit_probe(path)
            self.assertFalse(result["passed"])
            self.assertIn("Incomplete", result["reason"])

    def run_worker(self, source, wall=2., teardown=2.):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            worker = folder/"worker.py"
            worker.write_text(source.replace("SUMMARY_PATH", repr(str(folder/"summary.json"))))
            return supervise([sys.executable, str(worker)], folder/"summary.json",
                             folder/"worker.log", wall, teardown, terminate_grace=.2)

    def test_summary_switches_to_separate_teardown_budget(self):
        source = ("import pathlib,time\n"
                  f"pathlib.Path(SUMMARY_PATH).write_text({json.dumps(SUMMARY)!r})\n"
                  "time.sleep(2.5)\n")
        # Permit interpreter startup on a busy host, while making the clean
        # teardown longer than the original collection allowance.
        result = self.run_worker(source, wall=2., teardown=4.)
        self.assertEqual(result["worker_exit_code"], 0)
        self.assertTrue(result["clean_process_exit"])
        self.assertEqual(result["status"], "completed")

    def test_teardown_timeout_retains_episode_gate_but_not_clean_exit(self):
        source = ("import pathlib,time\n"
                  f"pathlib.Path(SUMMARY_PATH).write_text({json.dumps(SUMMARY)!r})\n"
                  "time.sleep(3)\n")
        result = self.run_worker(source, wall=1., teardown=.2)
        self.assertEqual(result["timeout_phase"], "teardown")
        self.assertTrue(result["episode_gate_passed"])
        self.assertFalse(result["clean_process_exit"])
        self.assertNotEqual(result["worker_exit_code"], 0)

    def test_no_summary_timeout_is_not_a_policy_failure(self):
        result = self.run_worker("import time\ntime.sleep(3)\n")
        self.assertEqual(result["timeout_phase"], "collection_or_initialization")
        self.assertIsNone(result["episode_gate_passed"])
        self.assertEqual(result["episode_health_verdict"], "not_assessed")

    def test_invalid_summary_does_not_start_teardown_or_claim_success(self):
        result = self.run_worker("import pathlib,time\n"
                                 "pathlib.Path(SUMMARY_PATH).write_text('{}')\n"
                                 "time.sleep(3)\n")
        self.assertEqual(result["timeout_phase"], "collection_or_initialization")
        self.assertIsNone(result["episode_gate_passed"])


if __name__ == "__main__":
    unittest.main()
