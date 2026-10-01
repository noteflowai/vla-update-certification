"""Prefix-only six-method decision engine for the frozen prospective study.

An external producer supplies requested old/new binary pairs. Cache contents,
unconsumed outcomes and reference-cohort labels are never passed to allocation.
This module runs no robot episodes and provides no empirical efficacy evidence.
"""
import numpy as np
from admission import Certificate
from joint_admission import JointCertificate
from coarse_admission import coarse_bounds
from run_admission import fixed_bounds

METHODS = (("marginal", "uniform"), ("marginal", "adaptive"),
           ("joint", "uniform"), ("joint", "adaptive"),
           ("fixed_exact", "uniform"), ("coarse", "iid_states"))


class PrefixDecision:
    def __init__(self, method, allocation, seed, states=20, cap=1000,
                 alpha=.05/(8*6), delta=.02, epsilon=.02):
        if ((method, allocation) not in METHODS or states < 1 or cap < 10*states
                or cap % (10*states) or not 0 < alpha < 1
                or not 0 <= delta < 1 or not 0 <= epsilon <= 1):
            raise ValueError("Invalid predeclared method configuration")
        self.method, self.allocation = method, allocation
        self.seed, self.states, self.cap = seed, states, cap
        self.alpha, self.delta, self.epsilon = alpha, delta, epsilon
        cls = JointCertificate if method == "joint" else Certificate
        self.cert = cls(states, alpha=alpha, delta=delta, epsilon=epsilon)
        self.rng = np.random.default_rng(seed)
        self.pending = None
        self.lower, self.upper = 0., 1-delta
        self.verdict = "uncertain"

    @property
    def pairs(self):
        return int(self.cert.n.sum())

    @property
    def finished(self):
        return self.verdict != "uncertain" or self.pairs == self.cap

    def request(self):
        if self.finished:
            raise StopIteration("Method decided or reached its predeclared cap")
        if self.pending is None:
            if self.method == "coarse":
                selected = self.rng.integers(0, self.states, 10).tolist()
            else:
                selected = [self.cert.choose(self.allocation, self.pairs//10)]*10
            next_repeat = self.cert.n.copy()
            self.pending = []
            for state in selected:
                self.pending.append((state, int(next_repeat[state])))
                next_repeat[state] += 1
        return tuple(self.pending)

    def consume(self, requested, outcomes):
        if self.pending is None or tuple(requested) != tuple(self.pending):
            raise ValueError("Results do not match this method's requested prefix")
        if len(outcomes) != len(self.pending):
            raise ValueError("Incomplete requested batch")
        # Validate the complete batch before any mutation.
        for pair in outcomes:
            if len(pair) != 2 or any(type(value) is not bool for value in pair):
                raise ValueError("Paired successes must be two strict booleans")
        differences = [int(old)-int(new) for old, new in outcomes]
        if self.method in ("marginal", "joint"):
            self.cert.add(self.pending[0][0], differences)
        else:
            # These methods need counts, not ten unused statewise CS updates.
            for (state, _), difference in zip(self.pending, differences):
                self.cert.n[state] += 1
                self.cert.plus[state] += difference == 1
                self.cert.minus[state] += difference == -1
        self.pending = None
        if self.method == "coarse":
            self.lower, self.upper = coarse_bounds(
                self.pairs, int(self.cert.plus.sum()), int(self.cert.minus.sum()),
                alpha=self.alpha, delta=self.delta)
        elif self.method == "fixed_exact":
            if self.pairs == self.cap:
                self.lower, self.upper = fixed_bounds(
                    self.cert.n, self.cert.plus, self.cert.minus, self.alpha, self.delta)
            else:
                # Terminal intervals must not be inspected repeatedly.
                self.lower, self.upper = 0., 1-self.delta
        else:
            self.lower, self.upper = self.cert.loss_bounds()
        self.verdict = ("accept" if self.upper <= self.epsilon else
                        "reject" if self.lower > self.epsilon else "uncertain")

    def snapshot(self):
        return {"method": self.method, "allocation": self.allocation, "seed": self.seed,
                "states": self.states, "cap": self.cap, "alpha": self.alpha,
                "delta": self.delta, "epsilon": self.epsilon,
                "pairs": self.pairs, "counts": self.cert.n.tolist(),
                "positive": self.cert.plus.tolist(), "negative": self.cert.minus.tolist(),
                "lower": self.lower, "upper": self.upper, "decision": self.verdict,
                "finished": self.finished,
                "scope": "Only consumed prefixes; statistical verdict is distinct from producer health."}
