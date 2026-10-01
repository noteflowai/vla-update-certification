"""Standard Dirichlet-mixture region projected onto paired success difference."""
from functools import lru_cache
import math
import numpy as np
from admission import Certificate


def log_mixture(n, plus, minus):
    counts = (plus, minus, n-plus-minus)
    return (sum(math.lgamma(k+.5)-math.lgamma(.5) for k in counts)
            + math.lgamma(1.5)-math.lgamma(n+1.5))


def profile_log_likelihood(n, plus, minus, difference):
    """Maximize likelihood over q+ - q- = d using a scalar total discordance."""
    if not -1 <= difference <= 1:
        return -math.inf
    if n == 0:
        return 0.
    zero = n-plus-minus
    # t=q++q-, q+=(t+d)/2, q-=(t-d)/2, q0=1-t.
    b = plus+minus+(plus-minus)*difference
    c = (plus-minus)*difference-zero*difference**2
    discriminant = max(b*b-4*n*c, 0.)
    roots = ((b+math.sqrt(discriminant))/(2*n),
             (b-math.sqrt(discriminant))/(2*n))
    candidates = [abs(difference), 1.]
    candidates += [t for t in roots if abs(difference) <= t <= 1]
    best = -math.inf
    for t in candidates:
        probabilities = ((t+difference)/2, (t-difference)/2, 1-t)
        likelihood = 0.
        for k, p in zip((plus, minus, zero), probabilities):
            if k:
                if p <= 0:
                    likelihood = -math.inf
                    break
                likelihood += k*math.log(p)
        best = max(best, likelihood)
    return best


@lru_cache(maxsize=200000)
def joint_cs(n, plus, minus, alpha):
    if not (0 <= plus <= n and 0 <= minus <= n-plus and 0 < alpha < 1):
        raise ValueError("Invalid categorical counts or error budget")
    if n == 0:
        return -1., 1.
    mixture = log_mixture(n, plus, minus)
    threshold = math.log(1/alpha)
    center = (plus-minus)/n
    def outside(d):
        return mixture-profile_log_likelihood(n, plus, minus, d) > threshold
    if outside(-1.):
        lo, hi = -1., center
        for _ in range(38):
            mid = (lo+hi)/2
            if outside(mid):
                lo = mid
            else:
                hi = mid
        lower = lo  # Outward endpoint.
    else:
        lower = -1.
    if outside(1.):
        lo, hi = center, 1.
        for _ in range(38):
            mid = (lo+hi)/2
            if outside(mid):
                hi = mid
            else:
                lo = mid
        upper = hi
    else:
        upper = 1.
    return lower, upper


class JointCertificate(Certificate):
    def add(self, state, paired_differences):
        values = np.asarray(paired_differences)
        if state not in range(self.states) or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Invalid state or paired binary difference")
        self.n[state] += values.size
        self.plus[state] += (values == 1).sum()
        self.minus[state] += (values == -1).sum()
        self.low[state], self.high[state] = joint_cs(
            int(self.n[state]), int(self.plus[state]), int(self.minus[state]),
            self.alpha/self.states)
