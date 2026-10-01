"""Controls for cache access, atomic ingestion, sampling and terminal inspection."""
import unittest
import numpy as np
from closedloop_decisions import PrefixDecision
from coarse_admission import coarse_bounds


class PrefixTests(unittest.TestCase):
    def test_repeated_request_does_not_advance_iid_rng(self):
        engine = PrefixDecision("coarse", "iid_states", seed=123)
        request = engine.request()
        self.assertEqual(request, engine.request())
        draws = np.random.default_rng(123).integers(0, 20, 10)
        self.assertEqual([s for s, _ in request], draws.tolist())
        prior = {}
        for state, repeat in request:
            self.assertEqual(repeat, prior.get(state, 0))
            prior[state] = repeat+1

    def test_bad_pair_and_wrong_prefix_are_atomic(self):
        engine = PrefixDecision("marginal", "uniform", seed=7)
        request = engine.request()
        bad = [(True, True)]*9+[(1, False)]
        with self.assertRaises(ValueError):
            engine.consume(request, bad)
        with self.assertRaises(ValueError):
            engine.consume(tuple(reversed(request)), [(True, True)]*10)
        self.assertEqual(engine.pairs, 0)
        self.assertEqual(engine.request(), request)

    def test_terminal_does_not_peek_and_stops_at_cap(self):
        engine = PrefixDecision("fixed_exact", "uniform", seed=7, states=2, cap=40)
        for block in range(4):
            request = engine.request()
            self.assertEqual([s for s, _ in request], [block % 2]*10)
            engine.consume(request, [(True, False)]*10)
            if block < 3:
                self.assertEqual(engine.verdict, "uncertain")
                self.assertEqual((engine.lower, engine.upper), (0., .98))
        self.assertTrue(engine.finished)
        with self.assertRaises(StopIteration):
            engine.request()

    def test_pooled_bounds_use_declared_joint_family_budget(self):
        engine = PrefixDecision("coarse", "iid_states", seed=123)
        request = engine.request()
        engine.consume(request, [(True, False)]*2+[(False, True)]*3+[(True, True)]*5)
        self.assertEqual((engine.lower, engine.upper),
                         coarse_bounds(10, 2, 3, alpha=.05/48, delta=.02))
        self.assertAlmostEqual(engine.alpha*8*6, .05)

    def test_statewise_methods_reject_unambiguous_suitewide_harm(self):
        for method in ("marginal", "joint"):
            for allocation in ("uniform", "adaptive"):
                with self.subTest(method=method, allocation=allocation):
                    engine = PrefixDecision(method, allocation, seed=7)
                    while not engine.finished:
                        request = engine.request()
                        engine.consume(request, [(True, False)]*10)
                    self.assertEqual(engine.verdict, "reject")

if __name__ == "__main__":
    unittest.main()
