# Additional bound-comparison development cohort

Freeze this protocol and source bytes before collecting joint-comparison-001.
This supplements the completed marginal-bound cohorts; it does not overwrite them.

- 30 independent repetitions of each of the same seven synthetic scenarios.
- 20 states, fresh seed 161803 and per-repetition randomized state order.
- alpha=.05, delta=.02, epsilon=.02; uniform fixed weights; 5,000-pair cap.
- Compare marginal/uniform, marginal/adaptive, joint/uniform, joint/adaptive
  and terminal exact. Shared per-state streams within each repetition.
- Joint region: Dirichlet(1/2,1/2,1/2) mixture over positive, negative and zero
  discordance, alpha/S per state. Project the region onto q+ - q- with a
  constrained likelihood profile. This is standard mixture inference, not new theory.
- At the true categorical probabilities the mixture is an e-process; Ville
  and a union bound give simultaneous regions under the same fresh-repeat and
  predictable-allocation assumptions. Projection and monotone risk aggregation
  preserve the earlier decision guarantee.
- A joint region is a comparator, not assumed to be tighter. The full-vector
  mixture may pay a larger dimensionality penalty. Report worse outcomes as well.
- Deterministic zero-observed-discordance cost is a property of each implemented
  bound, not a universal sample complexity lower bound.
- This small development cohort informs method choice, not a publication claim
  or a cross-model robot result. Do not pool correlated strategies as independent.
