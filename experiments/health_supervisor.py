"""Check resources before heavy imports; bound collection and teardown separately."""
import argparse
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def reset_audit_probe(summary_path):
    """Require complete, source-matched simulator checks before model imports."""
    result = {"summary_path": str(summary_path), "passed": False}
    try:
        summary = json.loads(summary_path.read_text())
        manifest = json.loads((summary_path.parent/"manifest.json").read_text())
        if not (summary["audit_passed"] is True and summary["declared"] == 45
                and summary["completed"] == 45 and summary["passed_rows"] == 45
                and summary["errors"] == 0):
            raise ValueError("Incomplete or failed reset/scoring audit")
        required = [ROOT/"experiments/canonical_reset.py",
                    ROOT/"experiments/libero_audit_core.py",
                    ROOT/"experiments/audit_reset_scoring_fixed.py",
                    ROOT/"protocols/closedloop-suite-001/manifest.json"]
        for path in required:
            if manifest["sources"].get(str(path)) != sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Unmatched reset source: {path.name}")
        for name, expected in manifest["sources"].items():
            if sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Audited implementation or inventory manifest changed")
        result.update(passed=True, summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
                      manifest_sha256=sha256((summary_path.parent/"manifest.json").read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["reason"] = type(error).__name__+": "+str(error)
    return result


def resource_probe():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0])*1024
    result = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "host_available_bytes": available, "host_required_bytes": 18*1024**3,
              "gpu_required_free_bytes": 28*1024**3}
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--id=0", "--query-gpu=name,memory.free,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        name, free, used = [value.strip() for value in text.strip().split(",")]
        result.update(gpu_name=name, gpu_free_bytes=int(free)*1024**2,
                      gpu_used_bytes=int(used)*1024**2)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        result["gpu_probe_error"] = type(error).__name__+": "+str(error)
    result["allowed"] = (available >= result["host_required_bytes"]
                         and result.get("gpu_free_bytes", 0) >= result["gpu_required_free_bytes"])
    return result


def read_summary(path):
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
        if (summary.get("model_health_verdict") not in ("passed", "failed", "not_assessed")
                or "health_gate_passed" not in summary):
            return None
        return summary
    except (OSError, ValueError):
        return None


def parent_death_guard(parent_pid):
    """Linux worker dies if its supervising parent is hard-killed."""
    if sys.platform != "linux":
        raise RuntimeError("This GPU supervisor requires Linux parent-death protection")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong]*4
    libc.prctl.restype = ctypes.c_int

    def install():
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os._exit(125)
        # Close the race in which the parent died before prctl was installed.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
    return install


class SupervisorInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def supervise(argv, summary_path, logfile, wall_budget, teardown_budget,
              terminate_grace=10, on_started=None, summary_reader=read_summary):
    """Own one new process group. Saved data never imply a successful process exit."""
    if min(wall_budget, teardown_budget, terminate_grace) <= 0:
        raise ValueError("Budgets must be positive")
    started = time.monotonic()
    saved_at = None
    timeout_phase = None
    interrupted_signal = None
    summary = None
    with Path(logfile).open("w") as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 preexec_fn=parent_death_guard(os.getpid()))

        def stop_owned_worker():
            if child.poll() is None:
                # The worker registers this handler before printing this marker.
                if "importing model and simulator libraries" in Path(logfile).read_text():
                    try:
                        os.kill(child.pid, signal.SIGUSR1)
                        time.sleep(.1)
                    except ProcessLookupError:
                        pass
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=terminate_grace)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        def interrupt(signum, frame):
            raise SupervisorInterrupted(signum)
        for signum in previous:
            signal.signal(signum, interrupt)
        try:
            if on_started is not None:
                on_started(child.pid)
            while child.poll() is None:
                now = time.monotonic()
                if summary is None:
                    summary = summary_reader(Path(summary_path))
                    if summary is not None:
                        saved_at = now
                phase = "teardown" if saved_at is not None else "collection_or_initialization"
                deadline = saved_at+teardown_budget if saved_at is not None else started+wall_budget
                if now >= deadline:
                    timeout_phase = phase
                    stop_owned_worker()
                    break
                time.sleep(.05)
        except SupervisorInterrupted as caught:
            interrupted_signal = caught.signum
            stop_owned_worker()
        except BaseException:
            stop_owned_worker()
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        code = child.wait()
    final_summary = summary_reader(Path(summary_path))
    clean = code == 0 and timeout_phase is None and interrupted_signal is None
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "worker_pid": child.pid, "worker_exit_code": code,
        "status": "supervisor_interrupted" if interrupted_signal else
                  timeout_phase+"_timeout" if timeout_phase else
                  "completed" if clean and final_summary is not None else "worker_error",
        "timeout_phase": timeout_phase, "clean_process_exit": clean,
        "supervisor_interrupted_signal": interrupted_signal,
        "parent_death_guard_enabled": True,
        "collection_wall_budget_seconds": wall_budget,
        "teardown_budget_seconds": teardown_budget,
        "elapsed_seconds": time.monotonic()-started,
        "episode_health_verdict": final_summary.get("model_health_verdict", "not_assessed") if final_summary else "not_assessed",
        "episode_gate_passed": final_summary.get("health_gate_passed") if final_summary else None,
        "summary_sha256": sha256(Path(summary_path).read_bytes()).hexdigest() if final_summary else None,
        "scope": "Episode verdict and process lifecycle are separate. No forced successful exit."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("pi05", "xvla"), required=True)
    parser.add_argument("--loader", choices=("official", "direct_cuda_strict"), default="direct_cuda_strict")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wall-budget", type=int, default=1800)
    parser.add_argument("--teardown-budget", type=int, default=120)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--reset-audit", type=Path,
                        default=ROOT/"runs/reset-scoring-002/summary.json")
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError("Refuse to reuse an output or receipt")
    if args.loader == "direct_cuda_strict" and args.family != "pi05":
        raise ValueError("The strict direct-CUDA loader is PI05-only")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    probe = resource_probe()
    audit = reset_audit_probe(args.reset_audit)
    probe.update(resource_allowed=probe["allowed"], reset_audit=audit,
                 allowed=probe["allowed"] and audit["passed"])
    probe.update(family=args.family, output=str(args.output),
                 inspect_only=args.inspect_only, worker_started=False,
                 supervisor_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.receipt.write_text(json.dumps(probe, indent=2)+"\n")
    print(json.dumps(probe), flush=True)
    if args.inspect_only or not probe["allowed"]:
        return 0 if args.inspect_only else 3
    log = args.receipt.with_suffix(".worker.log")
    def record_started(pid):
        probe.update(worker_started=True, worker_pid=pid, execution={"status": "running"})
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(probe, indent=2)+"\n")
        temporary.replace(args.receipt)
    result = supervise([
        "nice", "-n", "10", "/tmp/vlareg/.venv/bin/python", "-u",
        str(ROOT/"experiments/run_closedloop_health.py"), "--family", args.family,
        "--loader", args.loader, "--output", str(args.output),
        "--wall-budget", str(args.wall_budget), "--reset-mode", "canonical"],
        args.output/"summary.json", log, args.wall_budget, args.teardown_budget,
        on_started=record_started)
    probe.update(worker_started=True, execution=result)
    args.receipt.write_text(json.dumps(probe, indent=2)+"\n")
    if args.output.is_dir():
        (args.output/"supervisor-outcome.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if not result["clean_process_exit"] or result["status"] != "completed":
        return 4
    return 0 if result["episode_gate_passed"] is True else 5


if __name__ == "__main__":
    raise SystemExit(main())
