"""Conservative anytime-valid finite-suite regression-loss certificates.

The beta-binomial mixture e-process is standard, not a new statistical method.
For each state, two Bernoulli processes mark paired positive/negative discordance.
Ville plus a union bound gives simultaneous coverage across all states/times.
Predictable adaptive selection does not change the per-state coverage.
"""
from functools import lru_cache
import math
import numpy as np


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_e(n, k, p):
    mixture = log_beta(k + .5, n - k + .5) - log_beta(.5, .5)
    if p == 0:
        return mixture if k == 0 else math.inf
    if p == 1:
        return mixture if k == n else math.inf
    return mixture - k * math.log(p) - (n - k) * math.log1p(-p)


@lru_cache(maxsize=200000)
def bernoulli_cs(n, k, alpha):
    if not (0 <= k <= n and 0 < alpha < 1):
        raise ValueError("Invalid Bernoulli count or error budget")
    if n == 0:
        return 0., 1.
    threshold = math.log(1 / alpha)
    center = k / n
    if k == 0:
        low = 0.
    else:
        a, b = 0., center
        for _ in range(38):
            mid = (a + b) / 2
            if log_e(n, k, mid) > threshold:
                a = mid
            else:
                b = mid
        low = a
    if k == n:
        high = 1.
    else:
        a, b = center, 1.
        for _ in range(38):
            mid = (a + b) / 2
            if log_e(n, k, mid) > threshold:
                b = mid
            else:
                a = mid
        high = b
    return low, high


class Certificate:
    def __init__(self, states, delta=.02, epsilon=.02, alpha=.05, weights=None):
        if states < 1 or not 0 <= delta < 1 or not 0 <= epsilon <= 1 or not 0 < alpha < 1:
            raise ValueError("Invalid certificate parameters")
        self.states, self.delta, self.epsilon, self.alpha = states, delta, epsilon, alpha
        self.weights = np.full(states, 1 / states) if weights is None else np.asarray(weights, float)
        if (self.weights.shape != (states,) or (self.weights < 0).any()
                or not np.isclose(self.weights.sum(), 1)):
            raise ValueError("Weights must be fixed, nonnegative and sum to one")
        self.n = np.zeros(states, dtype=int)
        self.plus = np.zeros(states, dtype=int)
        self.minus = np.zeros(states, dtype=int)
        self.low = np.full(states, -1.)
        self.high = np.full(states, 1.)
        self.indicator_alpha = alpha / (2 * states)

    def add(self, state, paired_differences):
        values = np.asarray(paired_differences)
        if state not in range(self.states) or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Invalid state or paired binary difference")
        self.n[state] += values.size
        self.plus[state] += (values == 1).sum()
        self.minus[state] += (values == -1).sum()
        lo_p, hi_p = bernoulli_cs(int(self.n[state]), int(self.plus[state]), self.indicator_alpha)
        lo_m, hi_m = bernoulli_cs(int(self.n[state]), int(self.minus[state]), self.indicator_alpha)
        self.low[state], self.high[state] = lo_p - hi_m, hi_p - lo_m

    def loss_bounds(self):
        lower = np.maximum(self.low - self.delta, 0)
        upper = np.maximum(self.high - self.delta, 0)
        return float(self.weights @ lower), float(self.weights @ upper)

    def decision(self):
        lower, upper = self.loss_bounds()
        if upper <= self.epsilon:
            return "accept"
        if lower > self.epsilon:
            return "reject"
        return "uncertain"

    def choose(self, strategy, round_index):
        if strategy == "uniform":
            return round_index % self.states
        if strategy != "adaptive":
            raise ValueError("Unknown sampling strategy")
        # Prioritize uncertainty in the stated loss, with deterministic tie breaking.
        widths = self.weights * (
            np.maximum(self.high - self.delta, 0) - np.maximum(self.low - self.delta, 0)
        )
        return int(np.argmax(widths))
