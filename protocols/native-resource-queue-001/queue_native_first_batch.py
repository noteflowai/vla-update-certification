"""One bounded, separately recorded retry after a zero-episode provisioning stop.

Wait outside model imports. This wrapper strengthens admission only; the frozen
native launcher still enforces its own source, resource, lifecycle and budget
gates. It never retries a failed second launch or modifies earlier evidence.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

from health_supervisor import resource_probe
from native_collection import freeze_context, object_digest
from paired_collection import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError("Use new retry and queue paths; preserve previous evidence")
    previous = args.previous.resolve()
    reconciliation = json.loads((previous / "reconciliation-001.json").read_text())
    old_launch = Path(reconciliation["original_launch"])
    assert reconciliation["episodes_started"] == 0 and reconciliation["outcomes_collected"] == 0
    assert reconciliation["worker_reaped"] is True and reconciliation["same_cohort_retry_allowed"] is False
    for name in ("receipt", "summary"):
        assert sha256((old_launch / f"{name}.json").read_bytes()).hexdigest() == reconciliation[f"{name}_sha256"]
    assert not list((previous / "raw").glob("episode-*"))
    context = json.loads((previous / "context.json").read_text())
    health = Path(context["protocol"]["health"]["receipt"])
    assert context["family"] == "pi05" and context["update"] == "baseline_reload"
    assert freeze_context("pi05", "baseline_reload", health) == context
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {"started_at": datetime.now(timezone.utc).isoformat(),
              "status": "waiting_for_resources", "previous": str(previous),
              "output": str(args.output.resolve()), "context_sha256": object_digest(context),
              "queue_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "host_minimum_bytes": 24 * 1024 ** 3,
              "gpu_minimum_bytes": 28 * 1024 ** 3,
              "required_consecutive_observations": 3, "poll_seconds": 60,
              "wait_budget_seconds": 3600, "native_launch_count": 0, "observations": []}
    streak = 0
    while time.monotonic() - started < 3600:
        probe = resource_probe()
        eligible = (probe.get("allowed") is True
                    and probe["host_available_bytes"] >= record["host_minimum_bytes"]
                    and probe["gpu_free_bytes"] >= record["gpu_minimum_bytes"])
        streak = streak + 1 if eligible else 0
        record["observations"].append({"probe": probe, "eligible": eligible, "streak": streak})
        atomic_json(args.receipt, record)
        if streak >= 3:
            break
        time.sleep(60)
    else:
        record.update(status="resource_wait_expired", finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.receipt, record)
        return 3
    # Source/assets must still be unchanged after waiting. No new seed or suite.
    assert freeze_context("pi05", "baseline_reload", health) == context
    record.update(status="native_launcher_running", native_launch_count=1)
    atomic_json(args.receipt, record)
    command = ["python3", str(Path(__file__).with_name("run_native_collection.py")),
               "--family", "pi05", "--update", "baseline_reload",
               "--health-receipt", str(health), "--output", str(args.output.resolve())]
    result = subprocess.run(command)
    record.update(status="native_launcher_finished", launcher_exit_code=result.returncode,
                  finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.receipt, record)
    if result.returncode:
        return result.returncode
    # Audit completed evidence only after the native launcher has exited successfully.
    audit_output = args.receipt.with_name(args.receipt.stem + "-raw-audit.json")
    audit = subprocess.run(["/tmp/vlareg/.venv/bin/python",
                            str(Path(__file__).with_name("audit_native_evidence.py")),
                            "--cohort", str(args.output.resolve()), "--output", str(audit_output)])
    record.update(status="finished", raw_audit_exit_code=audit.returncode,
                  raw_audit_output=str(audit_output), finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.receipt, record)
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
