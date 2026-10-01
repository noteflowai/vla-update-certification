"""Sufficient risk bounds from IID state draws; no state profiling required."""
from admission import bernoulli_cs


def coarse_bounds(n, harm, gain, alpha=.05, delta=.02):
    if not (0 <= harm <= n and 0 <= gain <= n-harm and 0 <= delta < 1):
        raise ValueError("Invalid paired counts or tolerance")
    lp, up = bernoulli_cs(n, harm, alpha/2)
    _, um = bernoulli_cs(n, gain, alpha/2)
    # Jensen lower bound; pointwise max(q+ - q- - delta,0) <= (1-delta)q+.
    return max(lp-um-delta, 0.), (1-delta)*up
