import math
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import beta
from admission import Certificate, bernoulli_cs, log_e


class StatisticalChecks(unittest.TestCase):
    def test_mixture_matches_independent_numerical_integration(self):
        for n, k, p in [(8, 3, .2), (10, 0, .1), (10, 10, .9)]:
            likelihood = quad(
                lambda q: q ** (k - .5) * (1 - q) ** (n - k - .5) / beta(.5, .5),
                0, 1,
            )[0] / (p ** k * (1 - p) ** (n - k))
            self.assertAlmostEqual(math.exp(log_e(n, k, p)), likelihood, places=7)

    def test_interval_inversion_and_symmetry(self):
        for n, k in [(1, 0), (20, 3), (20, 20), (100, 5)]:
            lo, hi = bernoulli_cs(n, k, .01)
            mirrored = bernoulli_cs(n, n-k, .01)
            self.assertAlmostEqual(lo, 1-mirrored[1], places=9)
            self.assertAlmostEqual(hi, 1-mirrored[0], places=9)
            if lo > 0:
                self.assertAlmostEqual(log_e(n, k, lo), math.log(100), places=6)
            if hi < 1:
                self.assertAlmostEqual(log_e(n, k, hi), math.log(100), places=6)

    def test_sparse_harm_cannot_be_hidden_by_improvements(self):
        c = Certificate(2, delta=0, epsilon=.1)
        c.add(0, np.ones(500, dtype=int))
        c.add(1, -np.ones(500, dtype=int))
        self.assertEqual(c.decision(), "reject")

    def test_zero_observation_never_certifies(self):
        self.assertEqual(Certificate(20).decision(), "uncertain")

    def test_identical_pairs_can_eventually_be_admitted(self):
        c = Certificate(4)
        for state in range(4):
            c.add(state, np.zeros(1000, dtype=int))
        self.assertEqual(c.decision(), "accept")

    def test_bad_weights_and_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            Certificate(2, weights=[.1, .1])
        with self.assertRaises(ValueError):
            Certificate(2).add(0, [2])


if __name__ == "__main__":
    unittest.main()
