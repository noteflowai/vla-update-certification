"""Actual family-field contracts and small CPU tensor update controls."""
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from update_variants import configure_variant, round_linear_weights


def config(family):
    values = {"type": family, "chunk_size": 50 if family == "pi05" else 30,
              "n_action_steps": 50 if family == "pi05" else 30}
    values["num_inference_steps" if family == "pi05" else "num_denoising_steps"] = 10
    return SimpleNamespace(**values)


class VariantTests(unittest.TestCase):
    def test_chunked_rounding_and_hashes_are_independent_of_row_chunk(self):
        torch.manual_seed(12)
        source = nn.Linear(130, 7, bias=False)
        values, hashes = [], []
        for chunk in (1, 3, 128):
            model = nn.Linear(130, 7, bias=False)
            model.load_state_dict(source.state_dict())
            receipt = round_linear_weights(model, "w4", row_chunk=chunk)
            values.append(model.weight.detach().clone())
            hashes.append(receipt["layers"][0]["after_sha256"])
        self.assertTrue(all(torch.equal(values[0], item) for item in values))
        self.assertEqual(len(set(hashes)), 1)

    def test_flow_update_changes_the_consumed_family_field_without_aliasing(self):
        for family, field in (("pi05", "num_inference_steps"), ("xvla", "num_denoising_steps")):
            with self.subTest(family=family):
                original = config(family)
                updated, receipt = configure_variant(original, family, "steps2")
                self.assertEqual(getattr(updated, field), 2)
                self.assertEqual(getattr(original, field), 10)
                self.assertEqual(receipt["flow_parameter"], field)
                other = "num_denoising_steps" if family == "pi05" else "num_inference_steps"
                self.assertFalse(hasattr(updated, other))

    def test_missing_real_flow_field_or_changed_cadence_fails(self):
        wrong = config("pi05")
        del wrong.num_inference_steps
        wrong.num_denoising_steps = 10
        with self.assertRaises(ValueError):
            configure_variant(wrong, "pi05", "steps2")
        wrong = config("pi05")
        wrong.n_action_steps = 10
        with self.assertRaises(ValueError):
            configure_variant(wrong, "pi05", "w4")

    def test_last_short_group_does_not_requantize_the_whole_row(self):
        model = nn.Linear(130, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
            model.weight[:, 128:] = 100.
        receipt = round_linear_weights(model, "w3")
        self.assertTrue(torch.equal(model.weight[:, :128], torch.ones(1, 128)))
        self.assertTrue(torch.equal(model.weight[:, 128:], torch.full((1, 2), 100.)))
        self.assertEqual(receipt["layers"][0]["final_group_width"], 2)

    def test_zero_weights_dtype_and_bias_are_preserved(self):
        model = nn.Linear(7, 2).to(dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.zero_()
        bias = model.bias.detach().clone()
        round_linear_weights(model, "w4")
        self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.bias, bias))

    def test_shared_weight_is_rounded_once_and_reported(self):
        model = nn.ModuleList([nn.Linear(130, 2, bias=False), nn.Linear(130, 2, bias=False)])
        model[1].weight = model[0].weight
        receipt = round_linear_weights(model, "w4")
        self.assertEqual(receipt["unique_linear_weights"], 1)
        self.assertEqual(receipt["layers"][1]["shared_weight_with"], "0")

    def test_invalid_later_layer_is_rejected_before_any_mutation(self):
        model = nn.ModuleList([nn.Linear(7, 2, bias=False), nn.Linear(7, 2, bias=False)])
        before = model[0].weight.detach().clone()
        with torch.no_grad():
            model[1].weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            round_linear_weights(model, "w4")
        self.assertTrue(torch.equal(model[0].weight, before))


if __name__ == "__main__":
    unittest.main()
