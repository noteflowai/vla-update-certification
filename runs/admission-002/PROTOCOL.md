# Frozen screening protocol

This is a development feasibility study, not a held-out publication result.
Run manifests bind the protocol and executable source bytes before collection.
Never overwrite a completed run; disclose amendments and collect a new cohort.

## VLA admission

- Target: a finite, fixed suite of states with uniform weights.
- Loss: sum(weight * max(p_old - p_new - delta, 0)).
- delta = 0.02, admissible loss epsilon = 0.02, simultaneous error alpha = 0.05.
- Strategies: uniform and greatest current weighted loss-bound width.
- Statistical engine: Jeffreys beta-binomial mixture confidence sequences on
  positive and negative paired discordance indicators. Allocate alpha/(2*S).
- Accept only if the simultaneous loss upper bound <= epsilon; reject only if
  the lower bound > epsilon; otherwise return uncertain at budget exhaustion.
- Synthetic scenarios include unchanged, improved, sparse harm, diffuse harm,
  near-budget harm and heterogeneous states.
- Evaluation budget: 5,000 paired samples; each pair costs two policy episodes.
- 100 independent repetitions per scenario and strategy, paired by per-state
  random streams across strategies. Monte Carlo intervals are descriptive.
- Fixed-horizon baseline: a single end-of-budget exact binomial interval with
  Bonferroni allocation; never repeatedly inspect this baseline for stopping.
- Empirical replay of saved VLA episodes is descriptive only. Its paired identity
  checks must include task, initial state, scene seed, repeat and policy seed.
  Historical data cannot establish efficacy of a newly tuned allocation rule.
- No new GPU rollouts until the statistical feasibility screen is reviewed.

Assumptions: stable Bernoulli discordance probabilities within each state,
fresh independent randomization across repeats, predictable state allocation,
and a predeclared finite suite. Dependence between old/new outcomes in a pair
and across states is allowed by the union bound. No real-world safety claim.

## Skill execution contracts

- Start with verified, pinned public resource snapshots and deterministic oracle
  controls. Distinguish a constructed installation fault from an upstream bug.
- Controlled cases: complete bundle; missing resource; relocated bundle;
  execution dependency unavailable; repaired bundle.
- Freeze source hashes, task specifications and grading before model collection.
- Read, execute and grade are distinct. Grade actual outputs, not a model's
  completion claim or a file-load receipt.
- A relocated intact bundle is a negative control: a robust loader should
  preserve its task outcome. A contract checker must not reject it merely because
  its absolute installation path changed.
- Public-snapshot tests establish infrastructure behavior only. Do not label
  scripted or oracle execution as model efficacy.
- Model screening: use existing local checkpoints only, retain raw responses,
  per-turn tool results, model identity, token usage and failed attempts.
  Start with one available model; a second model is required before a
  cross-model claim. Do not reinterpret a missing model as a successful trial.
- Native-client claims require native-client runs. A controlled direct/MCP
  adapter is not a test of a branded client's skill loader.
- No public submission of a new paper until independent tasks and measured
  model outcomes support a contribution beyond existing static checks.

## Decision rules

VLA: continue only if calibrated certificates yield useful accept/reject decisions
and there is evidence of an allocation benefit or a substantive cost limitation.
No false-release observation is not a proof; the mathematical guarantee and its
assumptions must be supplied separately.

Skills: continue only if healthy task execution works, at least two execution
contract failure types occur, and restoration improves actual outcomes.
If full-bundle copying and version pinning suffice, publish an engineering
result and keep the research go/no-go negative.

## Amendment 1: state-order robustness (before admission-002 collection)

The first development cohort placed harmed states at the beginning of the
index order. The allocation algorithm uses no true probabilities, but its
deterministic tie breaking could benefit from that order. Preserve admission-001
and collect a separate cohort with seed 314159 and a fresh state permutation
for each repetition, shared across strategies. The algorithm receives only
outcome history, never the permutation or true discordance probabilities.
All other scenario definitions, budgets and certificate parameters remain fixed.
This is a development robustness check prompted by an observed design concern,
not a held-out confirmatory publication experiment.
