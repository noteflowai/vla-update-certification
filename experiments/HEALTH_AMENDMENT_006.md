# Local reset repair and bounded lifecycle, before new policy outcomes

The repeated-reset audit found identical flattened simulator states but different
scene XML and camera observations. LIBERO's generated property samplers
accumulate across hard resets, consuming additional placement RNG draws.
Fixed fixture poses reside in the model and are not restored by qstate alone.

`canonical_reset.py` verifies the installed base-file hash, the actual sampler
builder (base or exact argument-free super delegation), hard-reset semantics
and generated sampler types. It clears only those generated samplers immediately
before the normal hard reset and verifies the rebuilt count. The local Gym
wrapper preserves task access, step processing and vector semantics. No installed
package is edited. Unknown implementations fail rather than receive this repair.

New health runs default to this canonical mode. Exact checks now include camera
and robot observations, scene XML, task prompt, goal bindings/predicate values,
initial checker and controller, alongside the flattened simulator state.
Both reset snapshots are saved. A model-health check of this corrected pipeline
is still required; legacy cohort 004 cannot establish its health.

The five health cases, model revisions, precision/loading, checkpoint-default
50-action cadence, ten flow steps, 520-step horizon and >=3/5 success threshold
remain unchanged. No model update outcome informed this amendment. A ten-action
official-client configuration remains a separate protocol and health cohort.

`health_supervisor.py` checks GPU/host resources before heavy imports, owns only
its newly created process group and records the actual return code. Collection
and post-summary teardown have separate 1800/120 second budgets. Episode verdict
and clean process exit remain separate. SIGUSR1 stack capture is registered
before the import marker; timeout never becomes a false policy outcome.

The resource inspection and subprocess tests are engineering controls, not
validation of a real model run or a measured hardware speed improvement.
