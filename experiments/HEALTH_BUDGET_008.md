# Bounded completion of the unchanged family-health cohort

The interrupted `pi05-health-20261001-002` attempt spent about 237 seconds
hashing weights and 960 seconds loading and checking them on the shared host.
It completed one health episode and was interrupted during the next reset.
Its verdict remains `not_assessed`; none of its rows are pooled into a new
five-state gate.

Before launching `pi05-health-20261001-003`, increase the supervisor and worker
collection/initialization allowance from 1,800 to 5,400 seconds. Keep the
120-second teardown allowance, source-matched 45-case reset audit, host/GPU
resource guards, fixed five health states, seeds, checkpoint, action cadence,
strict PI05 loader and success threshold unchanged. Use the corrected
`health-supervisor-003` lifecycle implementation.

This changes an infrastructure time allowance after an interrupted attempt,
not policy configuration or state selection. Preserve each attempt separately.
It is neither a measured speedup nor an expansion of the main update cohort.
A family passes only with all five completed episodes, the predeclared
three-success threshold, zero infrastructure errors, exact reset checks and
an independently recorded clean worker exit. No main update data may be
collected before these gates pass.
