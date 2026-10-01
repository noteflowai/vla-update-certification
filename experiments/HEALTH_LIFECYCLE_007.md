# Interrupted supervising process: fail closed and clean the owned worker

The 2026-10-01 PI05 health attempt reached the strict loaded-weight verification,
passed the first state's exact reset-input checks and completed that state
successfully in 331 steps. Its supervising execution session ended with code
143 near the declared 1,800-second limit, while its separately started worker
remained alive. The cause of the session signal was not established.

Only this attempt's known worker/process group was stopped after verifying its
command and PID. The worker retained one completed health episode and one
interrupted reset attempt, with a persisted `not_assessed` health summary.
Cleanup requested SIGTERM and then SIGKILL; the true worker return code was
unavailable after loss of the original parent, so no clean exit was inferred.
The recovery record and original supervisor source are preserved under
`runs/pi05-health-20261001-002/`.

The supervisor now:

- Atomically records `worker_started=true`, its PID and `execution.status=running`
  immediately after launch rather than waiting until collection finishes.
- Handles SIGTERM and SIGINT by stopping its own worker group, reaping the child,
  and returning a separate `supervisor_interrupted` lifecycle result.
- Cleans up its worker if the launch receipt callback fails.
- Uses Linux `PR_SET_PDEATHSIG=SIGKILL` with a parent-PID race check so its direct
  worker cannot continue after an unhandleable parent death.
- Restores caller signal handlers and keeps episode verdict, process exit and
  readiness separate. A passed episode summary cannot override an interrupted
  lifecycle.

The current health collector uses a synchronous environment inside the direct
worker. A future producer with separate simulator child processes must give
those children equivalent parent-death protection; the direct-worker guarantee
is not a claim about arbitrary descendant process trees.

Four new subprocess controls cover live startup callbacks, callback failure,
SIGTERM/SIGINT, and parent SIGKILL. Together with the seven existing supervisor
controls, eleven tests pass without importing a model or using the GPU.

```bash
python3 -m unittest discover -s experiments -p 'test_health_*.py'
```

This is an infrastructure correction, not a completed five-state model-health
gate or new policy-update evidence. Native collection remains gated on a complete
family health result, clean exit, matching provenance and the full reset audit.
