# Closed-loop study and separate publication stages

This document is a concrete study design, not a claim that the study ran.
Do not launch tens of thousands of rollouts merely because a conservative
certificate requires them. First establish a useful bounded pilot.

## Research question

When do finite-suite regression budgets permit useful VLA update decisions
under a realistic rollout budget? Does predictable state allocation reduce
decision cost beyond a matched uniform anytime baseline, and where do terminal
fixed-horizon methods or an honest uncertain verdict perform better?

The study must first establish when a pooled-discordance sufficient gate settles
the decision without state profiling. A useful paper must report its advantages
as well as any statewise allocation benefit when the coarse gate is inconclusive.

## New data and evaluation controls

1. Freeze an exact list of 20 development-independent states before any outcome
   collection. Draw them without looking at which states regressed in the old
   pilot; record the public sampling seed and full candidate state inventory.
   The target guarantee applies to these 20 states only.
2. Use at least two independently implemented VLA families before claiming
   cross-model performance. Check actual local checkpoints and resource needs
   before choosing families. Do not list an unavailable model as evaluated.
3. Per family, declare four comparisons in advance: independently reload the
   identical pipeline, group-128 W4 weight rounding, group-128 W3 rounding as
   a severe stress control, and two instead of ten flow steps. Planned families
   are pinned X-VLA and pi0.5; no outcome-dependent family substitution.
   Label stress controls explicitly.
   Weight rounding without integer execution kernels supports an outcome study,
   not a hardware speed or memory-performance claim.
4. Freeze evaluator and model revisions, action transforms, success semantics,
   horizon and exact reset ordering. Verify initial-state hashes and no episode
   carryover. Include enough timestamps, actions and observations to distinguish
   a success-classification error from a policy failure.
   Record predicted chunk size separately from executed action count. PI05's
   pinned checkpoint-default configuration executes 50 actions per prediction;
   do not describe it as a reproduction of another ten-action client.
   The official LeRobot LIBERO evaluation example, fetched on September 30,
   explicitly overrides `n_action_steps=10`; its PI05 training examples do too.
   Health cohort 004 was already frozen at 50 before this documentation check.
   Keep that cohort intact. A switch to the documented ten-action recipe
   requires a pre-outcome protocol amendment and a separate health cohort;
   the 50-action check cannot validate that alternate pipeline.
   Pin the actual MuJoCo version and audit initial checker status, delivered
   prompt, reset observations and scene/scoring semantics.
5. Pair old/new episodes on state and fresh policy/environment randomization.
   Use a separate random stream across repeats. Record all attempts, crashes,
   timeouts and exclusions. Infrastructure failures trigger investigation;
   do not silently score or discard them to improve a method.
6. Ground-truth approximation requires a separately declared reference cohort
   and its uncertainty. Never label a five-repeat point estimate as true risk.
   Share replayed outcome streams across uniform/adaptive strategies to compare
   sample allocation on the same inputs; retain a genuine online execution
   pilot to test the operational cost of reset, switching and preflight.

## Allocation and comparison

- Keep alpha=.05, delta=.02, epsilon=.02 and uniform state weights as the
  development configuration. Any alternate decision objective must be frozen
  before its new cohort; include sensitivity analyses as explicitly exploratory.
- Primary bounded pilot: 1,000 pairs per declared update with an uncertain
  outcome allowed. Never describe this as a budget expected to certify all
  safe updates. Expand only if pilot precision and measured rollout cost make
  a useful question answerable.
- Family error: the primary comparison covers eight updates and six methods,
  with alpha_total/(8*6) per method/update. Report how this changes decisiveness.
  Alpha_total/8 covers the update family for one method only; per-update
  alpha=.05 is a different, weaker statement.
- Compare matched uniform/adaptive anytime certificates and a terminal exact
  interval with one inspection at its declared horizon.
- Include a joint-discordance region or another justified tighter bound as a
  baseline before presenting marginal Bonferroni bounds as practical.
- Include the pooled-discordance gate in `METHOD.md`, using independent
  weighted state draws per pair. Compare all six methods individually.
  A claim covering all eight updates and six methods uses alpha_total/(8*6)
  per method/update. The former alpha_total/8 design is retained as a
  separately labelled per-method-family sensitivity analysis; it does not
  validate selecting the first decision across six gates.
- Main metrics: accepts, rejects, uncertain outcomes; realized wrong decisions
  relative to the separately estimated reference risk with uncertainty;
  total policy episodes, elapsed time and switching cost; paired cost savings
  only on cases where both methods decide correctly.
- Keep censored costs at the cap. An uncertain run is not a successful
  5,000-pair decision. Show distributions and denominators, not just medians.

The operational design in `paper/DRAFT.md` declares the initial sampling seed,
excludes pilot/health cases, distinguishes virtual method consumption from
physical union collection, uses a separate fixed reference cohort, and caps
the first feasibility phase at 12 GPU-hours. Main-cohort state identities and
asset hashes are now bound to the actual inventory in
`protocols/closedloop-suite-001/manifest.json`: twenty states, two per task from
indices 20--49, excluding local pilot and health cases. This is zero-outcome
selection; per-family physical reset/controller audits remain required.
The prefix-only six-method engine is implemented in `closedloop_decisions.py`.
Its five controls pass; this is engineering readiness, not collected update
data. A physical outcome producer must still satisfy health/reset/scoring
checks and persist every attempted episode before main collection.

The durable controller is now implemented in `paired_collection.py`, with
seven synthetic-fixture engineering controls. Its pre-outcome contract is
`COLLECTION_PROTOCOL_002.md`. It verifies paired identities, blocks unresolved
producer attempts, hashes commits and replays consumed-prefix journals.
This does not implement the native inference producer or its all-attempt
cost log. The live reset audit exposed same-seed scene/camera drift on task 3
states 40 and 42 despite identical flattened simulator states. Generated property
samplers accumulated on hard resets. The local version-bound repair and exact
asset/vector checks pass on both states; the full 45-row audit passed with zero errors, exit code zero, and independent
source/snapshot/image/historical-replay verification.
Preserve all selected states. `HEALTH_AMENDMENT_006.md` documents the repair and
new exact-input/teardown checks before any main update outcome.
The model supervisor requires the completed, source-matched audit and sufficient
host/GPU headroom before heavy imports. A fresh corrected-pipeline health cohort
and clean model teardown are still required; legacy 004 is not that validation.

## Publication gates for a full archival final paper

Proceed to a full archival manuscript only after all of the following:

- Fresh closed-loop data cover at least two VLA families and declared updates.
- A valid theorem and its assumptions accompany the implementation.
- Allocation helps on genuine policy failures, or the evidence establishes a
  broadly relevant, well-quantified limitation of certification.
- Benchmarks survive reset, classifier and seed checks.
- The relation to the first nonarchival workshop paper is explicit; new data,
  decision methodology and substantive conclusions form the contribution.
- The literature search is updated near submission, including Admission,
  STEP/SAVI, compressed-VLA failure repair and benchmark implementation bugs.

Until then, the working manuscript is a reproducible development note.
RSS 2027 or TMLR are candidate venues, not promises of fit or acceptance.

### RSS Stage 1 is a separate preliminary stage

The official RSS 2027 call, rechecked September 30, explicitly permits missing
or preliminary empirical validation in Stage 1. It requires a semi-complete
paper with fully specified problem, hypotheses and methodology, up to five
content pages plus one references page. Deadline: December 4, 2026 23:59 AoE
(December 5 19:59 Singapore). No appendices or websites are reviewed at this
stage. Stage 2's final paper is due April 16, 2027 after the Extension Charter.

The full-paper evidence gates above are our research quality criteria, not an
additional requirement that RSS imposes before Stage 1. Preparing an honest
preliminary Stage 1 manuscript is reasonable before all new robot data exist.
It still needs a complete frozen new-study design, novelty review, official
template, anonymization and reference verification. Do not label the current
Markdown development draft as a submission-ready six-page paper.

The supplemental joint/marginal comparison is complete (30 repetitions of
seven synthetic scenarios, five paired methods). It improves some rejection
and improvement cases but is worse under zero observed discordance; preserve
both results in the Stage 1 narrative. Neither method is new statistical theory.

Official source: <https://roboticsconference.org/information/cfp/>.

## Skill direction

The current constructed-fault controls are solved by ordinary full-bundle
restoration. Do not spend GPU time to convert that deterministic observation
into a weak efficacy claim. Reopen an independent skill paper only if naturally
occurring, licensed failure cases survive full-bundle copying/version pinning
and healthy model tasks are demonstrably solvable.

A reopened study needs at least 12 independent tasks across multiple public
skills, two models, genuine native-client trials for any client-specific claim,
paired migration controls, raw tool transcripts, output grading independent
of completion claims, and a simple full-copy/version-pin baseline. Freeze task
selection before outcomes; avoid the old all-failure SWE-task floor.

Prospective variant primitives are implemented in `update_variants.py` and frozen
in `protocols/update-variants-002`. Seven CPU controls verify the actual family flow
field, unchanged cadence, partial group-128 weights, shared Parameters and exact
row-chunk equivalence. They do not implement independent model provisioning or
native episode collection. See `VARIANT_PROTOCOL_001.md`.

October 1 implementation update: the actual model/episode producer is now
implemented in `native_collection.py`, `native_rollout.py` and
`run_native_collection.py`, with fifteen new fixture controls and 42 passing
related checks. `NATIVE_COLLECTION_003.md` fixes its eight seed streams,
first requested ten-pair pilot, raw evidence verification, lifecycle and shared
twelve-hour main-collection reservation ledger. These additions do not yet
establish native operational validity. PI05's corrected five-state health run
completed with four successes, one valid policy failure, no infrastructure errors
and exit code zero. Its actual receipt and raw evidence passed the native entry
gate. X-VLA's same fixed health cohort completed separately with five successes,
no infrastructure errors and exit code zero; its evidence passed the same gate.
Both baseline contexts are prepared. The PI05 first ten-pair batch launched but
stopped at the second resource guard before model construction and before any
episode. All attempts and the 92.14-second wall cost are preserved. One separately
recorded same-context retry waits outside model imports for three consecutive
admission observations with at least 24 GiB host and 28 GiB GPU headroom. It keeps
all native gates, sources, seeds and budgets unchanged. No completed paired
outcome exists at this update. The earlier actual entry-point
probe rejected the then-unfinished PI05 receipt before model imports.
Cross-repeat initial simulator state and full input hashes must match a durable
state binding. Frozen GPU identity and driver must also match on each model load.
Preserved source snapshot: `protocols/native-producer-003/manifest.json`.
