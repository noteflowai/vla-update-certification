# Native producer correction after the development release

The `development-2026-10-01` paper and synthetic study are unchanged. They contain
no completed native update-efficacy result. Their release attachments and the
sources frozen with the synthetic cohorts remain immutable.

A subsequent PI05 baseline-reload control completed ten old-side episodes but
failed the resource-admission check before loading the independent new side.
No complete pairs were collected. The failed lifecycle, all raw files and the
shared budget charge are retained. It cannot be resumed by relaunching the same
cohort or treating an old-side outcome as a pair.

Independent raw auditing also found an error at the producer's 520-step horizon:
the episode summary said `truncated=true`, while the final saved transition
retained the environment's `truncated=false`. Two old-side episodes encountered
this branch. The remaining eight old-side episodes passed individual semantic
auditing, which still provides no paired update result. The original files have
not been rewritten or repaired.

The development source now records an effective collector truncation in the
last transition and in the summary together, while retaining the separate
environment termination/truncation flags and `collector_truncated` provenance.
It also writes each actual resource-admission probe before either loading guard
can reject a side. CPU controls execute the real episode-writing kernel with
synthetic vector APIs through success, environment truncation and the full
520-step collector horizon; they do not execute a policy or simulator.

These source changes require a fresh frozen context and matching health
qualification before any future native collection. They do not retrospectively
qualify the failed cohort. A repeat should first demonstrate sufficient resource
headroom across release and reload on an isolated runtime, with the same shared
physical budget and explicit failure reconciliation.
