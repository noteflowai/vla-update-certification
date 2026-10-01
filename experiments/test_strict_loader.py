"""Checkpoint integrity controls using a tiny real torch module."""
import unittest
import torch
from strict_pi05_loader import assign_and_verify, restore_nonpersistent_buffers


class LoaderTests(unittest.TestCase):
    def test_cast_preserves_exact_expected_values(self):
        model = torch.nn.Linear(3, 2).double()
        state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                 "bias": torch.tensor([2., -3.])}
        audit = assign_and_verify(model, state)
        self.assertEqual(audit["tensor_count"], 2)
        self.assertTrue(torch.equal(model.weight, state["weight"].double()))
        self.assertEqual(len(audit["dtype_conversions"]), 2)

    def test_missing_key_fails_before_assignment(self):
        model = torch.nn.Linear(3, 2)
        before = model.weight.clone()
        with self.assertRaises(ValueError):
            assign_and_verify(model, {"weight": torch.ones(2, 3)})
        self.assertTrue(torch.equal(model.weight, before))

    def test_wrong_shape_and_nonfinite_values_fail(self):
        for weight in [torch.ones(3, 3), torch.full((2, 3), float("nan"))]:
            with self.subTest(shape=weight.shape):
                model = torch.nn.Linear(3, 2)
                with self.assertRaises(ValueError):
                    assign_and_verify(model, {"weight": weight, "bias": torch.zeros(2)})

    def test_silent_partial_copy_is_detected(self):
        class Broken(torch.nn.Linear):
            def load_state_dict(self, state, strict=True):
                return None
        model = Broken(3, 2)
        with self.assertRaises(RuntimeError):
            assign_and_verify(model, {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)})

    def test_derived_positions_and_scalar_scale_are_restored(self):
        model = torch.nn.Module()
        model.embedding_dim = 9
        model.register_buffer("position_ids", torch.full((1, 4), 99), persistent=False)
        model.register_buffer("embed_scale", torch.tensor(-1.), persistent=False)
        audit = restore_nonpersistent_buffers(model)
        self.assertTrue(torch.equal(model.position_ids, torch.arange(4).reshape(1, 4)))
        self.assertEqual(model.embed_scale.item(), 3.)
        self.assertEqual(len(audit), 2)

    def test_unknown_derived_buffer_fails_closed(self):
        model = torch.nn.Module()
        model.register_buffer("unrecognized", torch.ones(2), persistent=False)
        with self.assertRaises(ValueError):
            restore_nonpersistent_buffers(model)


if __name__ == "__main__":
    unittest.main()
