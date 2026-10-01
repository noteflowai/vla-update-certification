"""Gate and supervise a real prefix batch; preparation performs no model inference."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

for key, value in {"MUJOCO_GL": "egl", "LIBERO_CONFIG_PATH": "/tmp/vlareg/libero_cfg",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}.items():
    os.environ.setdefault(key, value)

from native_collection import (MODELS, UPDATES, NativeBatchProducer, NativePairCache, freeze_context,
                               digest_file, object_digest)
from paired_collection import PairCache, atomic_json, collect_methods
from closedloop_decisions import METHODS, PrefixDecision
from health_supervisor import supervise, resource_probe

ROOT = Path(__file__).resolve().parents[1]


def read_native_summary(path):
    try:
        result = json.loads(path.read_text())
        if result.get("status") not in ("first_batch_completed", "completed", "native_error"):
            return None
        return result
    except (OSError, ValueError):
        return None


def check_previous_launches(output, context):
    """A later clean cache-only process cannot erase an earlier lifecycle failure."""
    expected = object_digest(context)
    for launch in sorted((output / "launches").glob("*")):
        try:
            receipt = json.loads((launch / "receipt.json").read_text())
            execution = receipt["execution"]
            summary_path = launch / "summary.json"
            summary = read_native_summary(summary_path)
            if (receipt.get("status") != "finished"
                    or receipt.get("context_sha256") != expected
                    or execution.get("clean_process_exit") is not True
                    or type(execution.get("worker_exit_code")) is not int
                    or execution["worker_exit_code"] != 0
                    or execution.get("status") != "completed"
                    or summary is None or summary["status"] == "native_error"
                    or summary.get("context_sha256") != expected
                    or digest_file(summary_path) != execution.get("summary_sha256")):
                raise ValueError("Incomplete, failed or changed native launch")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError("Prior native launch requires explicit reconciliation: "
                               + str(launch)) from error


class PhysicalBudget:
    """A shared wall-time reservation is a conservative bound on GPU occupation."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, operation):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = (json.loads(self.path.read_text()) if self.path.exists()
                     else {"limit_seconds": 12 * 3600, "reservations": {}})
            if state["limit_seconds"] != 12 * 3600:
                raise ValueError("The shared feasibility budget changed")
            result = operation(state)
            atomic_json(self.path, state)
            return result

    def reserve(self, key, seconds):
        def operation(state):
            if key in state["reservations"]:
                raise ValueError("Budget reservation already exists")
            charged = sum(item["charged_seconds"] for item in state["reservations"].values())
            if seconds <= 0 or charged + seconds > state["limit_seconds"]:
                raise RuntimeError("Shared twelve-hour feasibility budget exhausted")
            state["reservations"][key] = {"status": "reserved", "charged_seconds": seconds}
        return self.update(operation)

    def finish(self, key, actual_seconds):
        def operation(state):
            item = state["reservations"][key]
            if item["status"] != "reserved" or not 0 <= actual_seconds <= item["charged_seconds"]:
                raise ValueError("Invalid physical budget finalization")
            item.update(status="finalized", charged_seconds=actual_seconds)
        return self.update(operation)


def worker(args, context):
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    def interrupted(signum, frame):
        raise TimeoutError("Native worker externally interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    producer = None
    started = time.monotonic()
    summary_path = args.launch / "summary.json"
    try:
        from native_rollout import LiberoNativeBackend
        producer = NativeBatchProducer(args.output / "raw", context, LiberoNativeBackend(),
                                       wall_budget=args.wall_budget)
        if args.mode == "first_batch":
            method, allocation = METHODS[0]
            engine = PrefixDecision(method, allocation, context["seed"], states=20, cap=1000)
            requested = engine.request()
            cache = NativePairCache(args.output / "pairs", context)
            cache.get_many(requested, producer.produce_many)
            result = {"status": "first_batch_completed", "requested": requested,
                      "completed_pairs": len(requested),
                      "scope": "The first predeclared uniform batch only; no full-method "
                               "comparison, reference risk or update efficacy claim."}
        else:
            engines = collect_methods(args.output, context, producer, cap=1000,
                                      cache_class=NativePairCache)
            result = {"status": "completed", "methods": [e.snapshot() for e in engines],
                      "scope": "Prefix decisions only; no separately collected reference-risk labels."}
        atomic_json(summary_path, {**result, "context_sha256": object_digest(context),
                                  "worker_elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        atomic_json(summary_path, {"status": "native_error",
                                  "context_sha256": object_digest(context),
                                  "error": type(error).__name__ + ": " + str(error),
                                  "worker_elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        if producer is not None:
            producer.backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--update", choices=UPDATES, required=True)
    parser.add_argument("--health-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("first_batch", "full"), default="first_batch")
    parser.add_argument("--wall-budget", type=int, default=10800)
    parser.add_argument("--teardown-budget", type=int, default=120)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--launch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.wall_budget <= 0 or args.teardown_budget <= 0
            or args.wall_budget + args.teardown_budget + 15 > 12 * 3600):
        raise ValueError("Invalid feasibility/teardown budget")
    args.output = args.output.resolve()
    context = freeze_context(args.family, args.update, args.health_receipt)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "context.json"
    if path.exists() and json.loads(path.read_text()) != context:
        raise ValueError("Frozen context differs; preserve the previous cohort")
    if not path.exists():
        atomic_json(path, context)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "context_sha256": object_digest(context),
                          "output": str(args.output), "native_episodes_started": 0}))
        return 0
    if args.worker:
        if args.launch is None or not args.launch.is_dir():
            raise ValueError("Worker requires a supervising launch record")
        worker(args, context)
        return 0
    probe = resource_probe()
    if not probe["allowed"]:
        print(json.dumps({"status": "resource_blocked", "probe": probe}))
        return 3
    # Hold this file descriptor until the supervisor exits; concurrent resume must fail.
    cohort_lock = (args.output / "collector.lock").open("a")
    try:
        fcntl.flock(cohort_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        cohort_lock.close()
        raise RuntimeError("This cohort already has a supervising collector")
    check_previous_launches(args.output, context)
    launch = args.output / "launches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    launch.mkdir(parents=True, exist_ok=False)
    # One ledger for all eight family/update cohorts; unresolved runs retain their reservation.
    budget = PhysicalBudget(ROOT / "runs/native-feasibility-budget-001.json")
    reservation = str(launch)
    reserved = args.wall_budget + args.teardown_budget + 15
    budget.reserve(reservation, reserved)
    receipt = {"status": "starting", "context_sha256": object_digest(context),
               "mode": args.mode, "resource_probe": probe, "budget_reservation": reservation,
               "supervisor_sha256": digest_file(ROOT / "experiments/health_supervisor.py")}
    atomic_json(launch / "receipt.json", receipt)
    def record_started(pid):
        receipt.update(status="running", worker_pid=pid)
        atomic_json(launch / "receipt.json", receipt)
    command = ["nice", "-n", "10", "/tmp/vlareg/.venv/bin/python", "-u", str(Path(__file__)),
               "--family", args.family, "--update", args.update,
               "--health-receipt", str(args.health_receipt.resolve()),
               "--output", str(args.output), "--mode", args.mode,
               "--wall-budget", str(args.wall_budget), "--teardown-budget", str(args.teardown_budget),
               "--worker", "--launch", str(launch)]
    result = supervise(command, launch / "summary.json", launch / "worker.log",
                       args.wall_budget, args.teardown_budget, on_started=record_started,
                       summary_reader=read_native_summary)
    receipt.update(status="finished", execution=result)
    atomic_json(launch / "receipt.json", receipt)
    elapsed = result["elapsed_seconds"]
    if elapsed <= reserved:
        budget.finish(reservation, elapsed)
    # A terminal native_error summary cannot be promoted by an otherwise clean exit.
    summary = read_native_summary(launch / "summary.json")
    passed = (result["clean_process_exit"] and result["status"] == "completed"
              and summary is not None and summary["status"] != "native_error")
    print(json.dumps({"lifecycle": result, "native_collection_completed": passed,
                      "summary": summary}))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
