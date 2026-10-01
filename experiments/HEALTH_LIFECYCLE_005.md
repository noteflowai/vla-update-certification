# Post-004 lifecycle instrumentation

September 30: cohort 004 persisted five complete episodes and its summary at
06:52:48 UTC, then the GNU timeout supervisor returned 124. Four successes
pass the predeclared episode gate. The exact post-result blocking phase is
unknown; a clean process exit is not established.

The current runner adds `summary_persisted`, `main_returned` and
`interpreter_exit_hook` events. A later exception preserves the persisted
episode verdict and records a separate cleanup error. No forced successful
exit, outcome replacement, action-cadence change or new efficacy claim.

This revision was made after 004 ended. Its archived runner is unchanged;
the new instrumentation has only been syntax-checked, not GPU-validated.
Future runs freeze the actual revised runner and this note before episodes.
Reset/prompt/checker semantics and a clean teardown still need verification
before main update collection.
