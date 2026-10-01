# Coarse sufficient-gate comparison, frozen before collection

This development amendment adds a necessary simple baseline; it preserves all
earlier cohorts. It is not a new statistical theory or a robot experiment.

- Seed 424243; 30 independent repetitions per scenario; state order permuted.
- Same seven synthetic cases plus a predeclared compensated-loss case:
  five states q+=.35,q-=0; fifteen q+=0,q-=.13. Aggregate new success improves
  by .01 while R_delta=.0825 exceeds epsilon=.02.
- S=20, equal weights, alpha=.05 per strategy/update, delta=.02, epsilon=.02,
  5,000-pair cap, inspections in groups of ten observations.
- Six methods: marginal/uniform, marginal/adaptive, joint/uniform,
  joint/adaptive, terminal exact, coarse/IID states. Shared state streams.
- Coarse draws a state independently and uniformly for every pair, using a
  separate RNG seed 104729+10000*scenario_index+repetition. It is not an
  outcome-adaptive allocation and does not use deterministic uniform cycling.
- With h=E q+ and g=E q-, Jensen gives max(h-g-delta,0)<=R_delta.
  Pointwise max(q+-q--delta,0)<=(1-delta)q+, so R_delta<=(1-delta)h.
- Apply standard Bernoulli mixture intervals to harm/gain indicators with
  alpha/2 each. Set L=max(l_h-u_g-delta,0), U=(1-delta)u_h. Use the same
  accept/reject/uncertain rule. IID state selection and fresh conditional
  outcomes are essential to the pooled-indicator inference.
- Six paired strategy rows are not six independent repetitions. No combining
  gates at full alpha or selecting the most favorable gate without new error
  accounting. Cost is paired samples, not measured runtime; state switching
  may change operational cost in the future robot study.
- This supplements method comparisons, rather than supporting a general lower
  bound on certification cost. Report when the simple gate is decisively better.
