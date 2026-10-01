"""Owned subprocess controls: interrupted supervisor must not leave a worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from health_supervisor import supervise


def running(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


class LifecycleTests(unittest.TestCase):
    def driver(self, folder):
        folder = Path(folder)
        script = folder/"driver.py"
        script.write_text(
            "import json,pathlib,sys\n"
            f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\n"
            "from health_supervisor import supervise\n"
            f"root=pathlib.Path({str(folder)!r})\n"
            "result=supervise([sys.executable,'-c','import time; time.sleep(30)'],"
            "root/'summary.json',root/'worker.log',30,1,terminate_grace=.2,"
            "on_started=lambda pid:(root/'pid').write_text(str(pid)))\n"
            "(root/'result.json').write_text(json.dumps(result))\n")
        process = subprocess.Popen([sys.executable, str(script)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+5
        while not (folder/"pid").exists() and time.monotonic()<deadline:
            if process.poll() is not None:
                break
            time.sleep(.02)
        if not (folder/"pid").exists():
            process.kill()
            process.wait()
            self.fail("Fixture supervisor did not start its worker")
        return process, int((folder/"pid").read_text())

    def cleanup(self, process, pid):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if running(pid):
            os.killpg(pid, signal.SIGKILL)

    def test_term_and_interrupt_record_failure_and_reap_owned_worker(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as folder:
                process, pid = self.driver(folder)
                try:
                    self.assertTrue(running(pid))
                    os.kill(process.pid, signum)
                    process.wait(timeout=5)
                    result = json.loads((Path(folder)/"result.json").read_text())
                    self.assertEqual(result["status"], "supervisor_interrupted")
                    self.assertEqual(result["supervisor_interrupted_signal"], signum)
                    self.assertFalse(result["clean_process_exit"])
                    self.assertEqual(result["episode_health_verdict"], "not_assessed")
                    self.assertFalse(running(pid))
                finally:
                    self.cleanup(process, pid)

    def test_hard_killed_parent_cannot_leave_its_worker_running(self):
        with tempfile.TemporaryDirectory() as folder:
            process, pid = self.driver(folder)
            try:
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                deadline = time.monotonic()+5
                while running(pid) and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertFalse(running(pid))
                self.assertFalse((Path(folder)/"result.json").exists())
            finally:
                self.cleanup(process, pid)

    def test_start_callback_failure_still_cleans_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            started = []
            def fail(pid):
                started.append(pid)
                raise OSError("Synthetic receipt write error")
            with self.assertRaises(OSError):
                supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                          Path(folder)/"summary.json", Path(folder)/"worker.log",
                          30, 1, terminate_grace=.2, on_started=fail)
            self.assertEqual(len(started), 1)
            self.assertFalse(running(started[0]))

    def test_start_callback_observes_live_worker_before_terminal_result(self):
        with tempfile.TemporaryDirectory() as folder:
            seen = []
            def started(pid):
                self.assertTrue(running(pid))
                seen.append(pid)
            result = supervise([sys.executable, "-c", "import time; time.sleep(.15)"],
                               Path(folder)/"summary.json", Path(folder)/"worker.log",
                               1, 1, terminate_grace=.2, on_started=started)
            self.assertEqual(seen, [result["worker_pid"]])
            self.assertTrue(result["clean_process_exit"])
            self.assertTrue(result["parent_death_guard_enabled"])


if __name__ == "__main__":
    unittest.main()
