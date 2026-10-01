import unittest
import numpy as np
from coarse_admission import coarse_bounds


class CoarseTests(unittest.TestCase):
    def test_risk_sandwich_for_independent_random_simplex_probabilities(self):
        rng = np.random.default_rng(65537)
        for _ in range(1000):
            q = rng.dirichlet([.5, .5, .5], 20)
            w = rng.dirichlet(np.ones(20))
            delta = float(rng.random())
            actual = w @ np.maximum(q[:, 0]-q[:, 1]-delta, 0)
            lower = max(w @ (q[:, 0]-q[:, 1])-delta, 0)
            upper = (1-delta)*(w @ q[:, 0])
            self.assertLessEqual(lower, actual+1e-12)
            self.assertLessEqual(actual, upper+1e-12)

    def test_compensated_loss_is_not_certified_by_aggregate_gain(self):
        p = np.r_[np.full(5, .35), np.zeros(15)]
        m = np.r_[np.zeros(5), np.full(15, .13)]
        self.assertLess(float((p-m).mean()), 0)
        self.assertAlmostEqual(float(np.maximum(p-m-.02, 0).mean()), .0825)
        self.assertGreater((1-.02)*p.mean(), .02)

    def test_unobserved_data_are_uncertain(self):
        self.assertEqual(coarse_bounds(0, 0, 0), (0., .98))

    def test_impossible_paired_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            coarse_bounds(4, 3, 2)


if __name__ == "__main__":
    unittest.main()
