import math
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from joint_admission import JointCertificate, joint_cs, log_mixture, profile_log_likelihood


class JointRegionTests(unittest.TestCase):
    def test_dirichlet_moment_matches_rising_factorial_identity(self):
        expected = (.5*1.5*.5)/(1.5*2.5*3.5)
        self.assertAlmostEqual(math.exp(log_mixture(3, 2, 1)), expected)

    def test_zero_discordance_has_independent_closed_form(self):
        n, alpha = 100, .0025
        expected = 1-(alpha/(2*n+1))**(1/n)
        low, high = joint_cs(n, 0, 0, alpha)
        self.assertAlmostEqual(high, expected, places=10)
        self.assertAlmostEqual(low, -expected, places=10)

    def test_profile_matches_independent_scalar_optimizer(self):
        for n, plus, minus in ((10, 4, 2), (20, 1, 5), (20, 0, 4)):
            for d in (-.6, -.1, .2, .7):
                def likelihood(t):
                    ps = ((t+d)/2, (t-d)/2, 1-t)
                    if any(k and p <= 0 for k, p in zip((plus, minus, n-plus-minus), ps)):
                        return -math.inf
                    return sum(k*math.log(p) for k, p in
                               zip((plus, minus, n-plus-minus), ps) if k)
                result = minimize_scalar(lambda t: -likelihood(t),
                                         bounds=(abs(d)+1e-10, 1-1e-10), method="bounded")
                independent_max = max(-result.fun, likelihood(abs(d)), likelihood(1.))
                self.assertAlmostEqual(profile_log_likelihood(n, plus, minus, d),
                                       independent_max, places=5)

    def test_projection_contains_feasible_region_grid(self):
        n, plus, minus, alpha = 20, 4, 3, .01
        lower, upper = joint_cs(n, plus, minus, alpha)
        for p in np.linspace(.01, .97, 45):
            for m in np.linspace(.01, .99-p, 45):
                ll = plus*math.log(p)+minus*math.log(m)+(n-plus-minus)*math.log(1-p-m)
                if log_mixture(n, plus, minus)-ll <= math.log(1/alpha):
                    self.assertLessEqual(lower, p-m)
                    self.assertGreaterEqual(upper, p-m)

    def test_symmetry_mle_and_unobserved_state(self):
        a = joint_cs(20, 4, 3, .01)
        b = joint_cs(20, 3, 4, .01)
        self.assertAlmostEqual(a[0], -b[1])
        self.assertAlmostEqual(a[1], -b[0])
        self.assertLessEqual(a[0], .05)
        self.assertGreaterEqual(a[1], .05)
        self.assertEqual(JointCertificate(20).decision(), "uncertain")


if __name__ == "__main__":
    unittest.main()
