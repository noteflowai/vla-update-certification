# Finite-suite admission certificate

This is an application of a standard beta-binomial mixture confidence sequence,
not a claim of new statistical theory.

Let a predeclared suite contain S states with fixed nonnegative weights summing
to one. For each state, collect old/new binary outcomes on paired fresh repeats.
Write D = X_old - X_new, q+ = P(D=1), q- = P(D=-1). Then the difference between
marginal success probabilities is exactly q+ - q-, regardless of dependence
between old and new outcomes within a pair.

The target is

    R_delta = sum_s w_s max(q+_s - q-_s - delta, 0).

It measures excess regression above a per-state tolerance. Improvements at
other states cannot offset this loss. It does not mean every state is safe:
with equal weights, a large loss on one state may fit a small overall budget.
Weights, delta and epsilon must therefore express the actual decision objective.

## Confidence sequence and guarantee

For a Bernoulli indicator with n observations and k successes, use

    E_n(p) = B(k+1/2, n-k+1/2)
             / [B(1/2,1/2) p^k (1-p)^(n-k)].

For each fixed p in (0,1), this is the mixture of Bernoulli likelihood-ratio
martingales against p under a Beta(1/2,1/2) prior. At p=0 or p=1 the limiting
process on possible sample paths is a nonnegative supermartingale. Ville's
inequality gives P(exists n: E_n(p) >= 1/a) <= a. Inverting E_n(p) < 1/a gives
an interval; including its boundary is conservative.

Apply this construction to positive and negative discordance at every state,
with a = alpha/(2*S). Let their simultaneous intervals be [l+,u+] and [l-,u-].
With probability at least 1-alpha, for every sampled time and every state,

    l+ - u- <= q+ - q- <= u+ - l-.

The two indicators need not be independent: the union bound does not require
independence. The positive-part function is monotone, so

    L = sum_s w_s max(l+_s - u-_s - delta, 0)
    U = sum_s w_s max(u+_s - l-_s - delta, 0)

satisfy L <= R_delta <= U simultaneously. Accept when U <= epsilon, reject
when L > epsilon, and otherwise keep sampling or return uncertain.

For one predeclared update, the probability of any wrong accept or reject at
any stopping time is at most alpha. This follows because a wrong decision
requires failure of the simultaneous coverage event. Zero observed mistakes in
a Monte Carlo experiment is not the source of this guarantee.

## Adaptive sampling assumptions

The next state is chosen using only past observed data. Each selected state
must supply a fresh repeat whose conditional discordance probabilities are its
fixed q+ and q-. Equivalently, one can imagine a fresh IID outcome stream per
state, inspected at predictable times. Optional skipping preserves validity.
Arbitrary serial dependence or changing probabilities within a state's stream
is not covered. Cross-state dependence is allowed only insofar as these
conditional per-state assumptions still hold.

The implemented heuristic selects the state with the largest current weighted
loss-interval width. It has no optimality guarantee. It does not receive true
scenario probabilities, and its deterministic tie breaking is checked in a
separate randomized-state-order cohort.

## Limits that must accompany any paper

- The guarantee concerns the declared finite suite, not unseen tasks or robots.
- Multiple policy updates need a family error allocation, such as alpha_j with
  sum_j alpha_j <= alpha_total. Reusing .05 for every searched candidate does
  not control the overall selection process.
- Historical paired records can be replayed descriptively. Complete pair keys
  do not establish stationarity, fresh repeats, correct benchmark reset behavior,
  or freedom from selection bias.
- The interval is deliberately conservative: it treats two marginal indicators
  separately and allocates error to every state. Fixed-horizon tests can be much
  cheaper, and an uncertainty verdict is a valid outcome.
- These bounds are statistical evidence about binary task success. They are
  not a certificate of physical safety.

Useful background: Ville (1939), martingale maximum inequality; Howard et al.
(2021), *Time-uniform, nonparametric, nonasymptotic confidence sequences*,
Annals of Statistics 49(2), DOI 10.1214/20-AOS1991. The specific implemented
Beta mixture and its inversion are written above so they can be independently
checked rather than inferred from a citation.

## Joint categorical comparator

Let the counts of +1, −1 and 0 be b, c, z, with n=b+c+z. Mixing the
categorical likelihood under Dirichlet(1/2,1/2,1/2) gives

    M_n = Γ(3/2)/Γ(n+3/2) × Π_{k∈{b,c,z}} Γ(k+1/2)/Γ(1/2).
    E_n(q) = M_n / (q+^b q-^c q0^z),     q0=1−q+−q-.

This is the standard mixture likelihood-ratio martingale at interior true q,
and a supermartingale on possible paths at support boundaries. Invert
E_n(q) ≤ S/alpha per state. Ville and a union bound cover all states and times.
Projecting each covered categorical region onto d=q+−q− preserves coverage;
positive-part aggregation then gives the same wrong-decision bound as above.
Adaptive sampling and family accounting require the same assumptions.

For a candidate d, profile over t=q++q− ∈ [|d|,1]:

    q+=(t+d)/2, q-=(t−d)/2, q0=1−t.
    log L_d(t) = b log((t+d)/2) + c log((t−d)/2) + z log(1−t).

The stationary roots satisfy

    n t² − [(b+c)+(b−c)d]t + [(b−c)d−z d²] = 0.

Check feasible roots and endpoints; terms with zero counts contribute zero.
The profile likelihood is log-concave, so its accepted d set is an interval.
Outward bisection endpoints yield the implemented projection. Independent
SciPy scalar optimization and a feasible-simplex grid check the implementation.

The comparator is not guaranteed to be tighter. When b=c=0 and n>0,
M_n=1/(2n+1), and its exact upper d endpoint is
1−[alpha/(S(2n+1))]^(1/n). This provides an independent check of its
zero-observed-discordance cost and shows the extra mixture-dimension penalty.
These mixture methods are established inference tools; empirical decision
cost for the stated nonlinear objective is the proposed research question.

## Pooled-discordance sufficient gate

A simple baseline can avoid simultaneous state profiling altogether. Draw each
state independently with the predeclared weights, then obtain one fresh pair.
The resulting harm/gain indicators have fixed probabilities
h=sum_s w_s q+_s and g=sum_s w_s q-_s. Jensen's inequality and a pointwise bound give

    max(h-g-delta, 0) <= R_delta <= (1-delta)*h,    0 <= delta < 1.

For the upper bound, if x=q+ and y=q-, a positive loss is x-y-delta.
Since x<=1 and y>=0, x-y-delta <= x-delta*x; a zero loss satisfies the
same nonnegative upper bound. Averaging proves the result.

Use the Bernoulli mixtures above for pooled harm/gain at alpha/2 each:
L=max(l_h-u_g-delta,0), U=(1-delta)*u_h. On their simultaneous coverage event,
these sandwich R_delta, so the same decision rule has wrong-decision probability
at most alpha. This is a sufficient gate, not an exact identification of R_delta.
It can admit low-discordance updates cheaply, but aggregate gains may leave it
uncertain despite large losses on some states.

IID state draws and fresh conditional outcomes are required. Do not reuse this
Bernoulli argument for outcome-adaptive state allocation or deterministic
cycling. Comparing gates at alpha individually does not authorize choosing the
first favorable decision at the same alpha: a combined rule needs its own
error split. State switching also creates an operational cost not measured by
synthetic paired-sample counts.

### A concrete limit of pooling

For twenty equally weighted states and delta=epsilon=.02, consider joint
old/new outcomes ordered as (both succeed, old only, new only, both fail).
A stable construction gives (.45,.05,.05,.45) at every state, so R_delta=0.
A mixed construction gives (.45,.10,0,.45) at ten states and (.45,0,.10,.45)
at the other ten, so R_delta=.5*(.10-.02)=.04.
Both constructions have exactly the same pooled joint-outcome distribution
and 50% aggregate success for each policy. Any procedure that uses only these
pooled outcomes, discarding state identities, receives identically distributed
data in both constructions, even with arbitrarily many IID pairs.
It cannot identify which side of the .02 statewise budget holds.

This explanatory example is not new statistical theory, a universal sample
complexity lower bound, or model evidence. It shows why a sufficient pooled
gate can legitimately remain uncertain and why retaining state information
can matter. Exact rational values and the standalone figure are produced by
`illustrate_aggregate_identifiability.py` and recorded separately in
`analysis/aggregate-identifiability.json`.
