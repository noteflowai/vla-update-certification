"""Exercise the actual LIBERO sampler builder with small non-rendering objects."""
from types import SimpleNamespace
import unittest
import numpy as np
from canonical_reset import BDDLBaseDomain, clear_generated_property_samplers, verified_sampler_builder


class Transparent(BDDLBaseDomain):
    def _add_placement_initializer(self):
        super()._add_placement_initializer()


class Impure(BDDLBaseDomain):
    def _add_placement_initializer(self):
        self.extra = True
        super()._add_placement_initializer()


class AlteredParent(BDDLBaseDomain):
    def _add_placement_initializer(self):
        pass


class DelegatingToAlteredParent(AlteredParent):
    def _add_placement_initializer(self):
        super()._add_placement_initializer()


def domain():
    obj = SimpleNamespace(name="drawer", object_properties={
        "articulation": {"default_close_ranges": (0., 1.)}})
    return SimpleNamespace(
        parsed_problem={"fixtures": {}, "objects": {}, "regions": {},
                        "initial_state": [["close", "drawer"]], "problem_name": "fixture"},
        objects_dict={}, fixtures_dict={}, object_property_initializers=[],
        object_states_dict={"drawer": SimpleNamespace(set_joint=lambda value: None)},
        get_object=lambda name: obj)


def fixture_draw(d):
    np.random.seed(123)
    for sampler in d.object_property_initializers:
        sampler.sample()
    return np.random.uniform(size=3)


class ResetRepairTests(unittest.TestCase):
    def test_base_and_exact_direct_super_delegation_are_verified(self):
        for cls in (BDDLBaseDomain, Transparent):
            self.assertTrue(verified_sampler_builder(
                object.__new__(cls)._add_placement_initializer))

    def test_added_task_builder_behavior_is_rejected(self):
        self.assertFalse(verified_sampler_builder(
            object.__new__(Impure)._add_placement_initializer))

    def test_super_delegation_to_an_altered_parent_is_rejected(self):
        self.assertFalse(verified_sampler_builder(
            object.__new__(DelegatingToAlteredParent)._add_placement_initializer))

    def test_vendor_rebuild_accumulates_samplers_and_changes_following_rng_draws(self):
        d = domain()
        BDDLBaseDomain._add_placement_initializer(d)
        first = fixture_draw(d)
        BDDLBaseDomain._add_placement_initializer(d)
        self.assertEqual(len(d.object_property_initializers), 2)
        self.assertFalse(np.array_equal(first, fixture_draw(d)))

    def test_regeneration_keeps_one_sampler_and_same_following_draws(self):
        d = domain()
        draws = []
        for _ in range(3):
            clear_generated_property_samplers(d)
            BDDLBaseDomain._add_placement_initializer(d)
            self.assertEqual(len(d.object_property_initializers), 1)
            draws.append(fixture_draw(d))
        self.assertTrue(all(np.array_equal(draws[0], value) for value in draws))

    def test_unknown_initializer_fails_before_any_mutation(self):
        d = domain()
        BDDLBaseDomain._add_placement_initializer(d)
        d.object_property_initializers.append(object())
        before = list(d.object_property_initializers)
        with self.assertRaises(ValueError):
            clear_generated_property_samplers(d)
        self.assertEqual(d.object_property_initializers, before)


if __name__ == "__main__":
    unittest.main()
