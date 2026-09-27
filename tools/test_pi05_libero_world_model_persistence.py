"""Synthetic CPU contract tests only; no source/demo quality or training claim."""
from __future__ import annotations

import copy
import unittest

import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model_persistence import (
        persistence_contract, persistence_predictions, persistence_scoring_targets,
    )
    from pi05_libero_world_model_objectives import native_world_model_loss
else:
    from .pi05_libero_world_model_persistence import (
        persistence_contract, persistence_predictions, persistence_scoring_targets,
    )
    from .pi05_libero_world_model_objectives import native_world_model_loss


def fixture(b=2, t=4, h=3, d=5):
    inputs = {
        "history_visual_latent": torch.arange(b * t * 2 * d, dtype=torch.float32).reshape(b, t, 2, d) / 10,
        "history_visual_valid": torch.ones(b, t, 2, dtype=torch.bool),
        "history_state": torch.arange(b * t * 8, dtype=torch.float32).reshape(b, t, 8) / 7,
        "history_state_valid": torch.ones(b, t, 8, dtype=torch.bool),
        "task_instruction": [f" synthetic task {i} " for i in range(b)],
        "candidate_actions": torch.arange(b * h * 7, dtype=torch.float32).reshape(b, 1, h, 7) / 3,
    }
    targets = {
        "future_visual_latent": torch.arange(b * h * 2 * d, dtype=torch.float32).reshape(b, h, 2, d) / 11,
        "future_visual_valid": torch.ones(b, h, 2, dtype=torch.bool),
        "state_delta": torch.arange(b * h * 8, dtype=torch.float32).reshape(b, h, 8) / 13,
        "state_target_valid": torch.ones(b, h, 8, dtype=torch.bool),
    }
    return inputs, targets


class PersistenceTests(unittest.TestCase):
    def assert_tensors_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, equal_nan=True)

    def test_hand_computed_last_visual_and_zero_state(self):
        inputs, _ = fixture(b=1, t=2, h=3, d=1)
        inputs["history_visual_latent"] = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
        result = persistence_predictions(inputs)
        self.assertEqual(set(result), {"pred_future_visual_latent", "pred_state_delta"})
        torch.testing.assert_close(result["pred_future_visual_latent"],
                                   torch.tensor([[[[[3.0], [4.0]], [[3.0], [4.0]], [[3.0], [4.0]]]]]), atol=0, rtol=0)
        self.assertTrue(torch.equal(result["pred_state_delta"], torch.zeros(1, 1, 3, 8)))

    def test_nondefault_positive_shapes(self):
        for b, t, h, d in ((1, 1, 1, 1), (3, 2, 5, 7)):
            with self.subTest(shape=(b, t, h, d)):
                inputs, _ = fixture(b, t, h, d)
                result = persistence_predictions(inputs)
                self.assertEqual(result["pred_future_visual_latent"].shape, (b, 1, h, 2, d))
                self.assertEqual(result["pred_state_delta"].shape, (b, 1, h, 8))
                self.assertTrue(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in result.values()))

    def test_earlier_history_values_and_masks_do_not_predict(self):
        inputs, _ = fixture()
        before = persistence_predictions(inputs)
        inputs["history_visual_latent"][:, :-1] = float("nan")
        inputs["history_visual_valid"][:, :-1] = False
        inputs["history_state"][:, :-1] = float("inf")
        inputs["history_state_valid"][:, :-1] = False
        self.assert_tensors_equal(before, persistence_predictions(inputs))

    def test_history_state_values_do_not_predict(self):
        inputs, _ = fixture()
        before = persistence_predictions(inputs)
        inputs["history_state"].fill_(101.0)
        self.assert_tensors_equal(before, persistence_predictions(inputs))

    def test_finite_action_values_unbounded_and_ignored(self):
        inputs, _ = fixture()
        before = persistence_predictions(inputs)
        inputs["candidate_actions"].fill_(-12345.0)
        self.assert_tensors_equal(before, persistence_predictions(inputs))

    def test_arbitrary_nonempty_task_text_ignored(self):
        inputs, _ = fixture()
        before = persistence_predictions(inputs)
        inputs["task_instruction"] = ["  unrelated task  ", "not a native registry member"]
        self.assert_tensors_equal(before, persistence_predictions(inputs))
        self.assertEqual(inputs["task_instruction"][0], "  unrelated task  ")

    def test_strict_input_keys_reject_targets_metadata_and_oracle(self):
        for key in ("targets", "metadata", "task_id", "exact_contact", "future_visual_latent"):
            inputs, _ = fixture()
            inputs[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                persistence_predictions(inputs)
        inputs, _ = fixture()
        del inputs["task_instruction"]
        with self.assertRaises(ValueError):
            persistence_predictions(inputs)

    def test_k2_rejected(self):
        inputs, _ = fixture()
        inputs["candidate_actions"] = inputs["candidate_actions"].expand(-1, 2, -1, -1).clone()
        with self.assertRaises(ValueError):
            persistence_predictions(inputs)

    def test_input_dtypes_strict(self):
        for key in ("history_visual_latent", "history_state", "candidate_actions", "history_visual_valid", "history_state_valid"):
            inputs, _ = fixture()
            inputs[key] = inputs[key].to(torch.float64)
            with self.subTest(key=key), self.assertRaises(ValueError):
                persistence_predictions(inputs)

    def test_wrong_native_dimensions_and_empty_axes_rejected(self):
        cases = {
            "history_state": (2, 4, 32), "candidate_actions": (2, 1, 3, 9),
            "history_visual_latent": (2, 4, 3, 5), "history_visual_valid": (2, 4, 2, 1),
        }
        for key, shape in cases.items():
            inputs, _ = fixture()
            inputs[key] = torch.zeros(shape, dtype=inputs[key].dtype)
            with self.subTest(key=key), self.assertRaises(ValueError):
                persistence_predictions(inputs)
        for dims in ((0, 4, 3, 5), (2, 0, 3, 5), (2, 4, 0, 5), (2, 4, 3, 0)):
            inputs, _ = fixture(*dims)
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                persistence_predictions(inputs)

    def test_nonempty_per_row_task_list_required(self):
        for tasks in (("a", "b"), ["a"], ["a", " "], ["a", 9]):
            inputs, _ = fixture()
            inputs["task_instruction"] = tasks
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                persistence_predictions(inputs)

    def test_declared_valid_history_and_all_actions_must_be_finite(self):
        for key in ("history_visual_latent", "history_state", "candidate_actions"):
            inputs, _ = fixture()
            inputs[key].flatten()[0] = float("nan")
            with self.subTest(key=key), self.assertRaises(ValueError):
                persistence_predictions(inputs)
        inputs, _ = fixture()
        inputs["candidate_actions"].flatten()[-1] = float("inf")
        with self.assertRaises(ValueError):
            persistence_predictions(inputs)

    def test_masked_latest_poison_cleared_without_older_fallback(self):
        inputs, _ = fixture()
        inputs["history_visual_valid"][0, -1, 1] = False
        inputs["history_visual_latent"][0, -1, 1] = float("nan")
        inputs["history_state_valid"][1, -1, 4] = False
        inputs["history_state"][1, -1, 4] = float("inf")
        result = persistence_predictions(inputs)
        self.assertTrue(bool(torch.isfinite(result["pred_future_visual_latent"]).all()))
        self.assertTrue(torch.equal(result["pred_future_visual_latent"][0, 0, :, 1], torch.zeros(3, 5)))
        self.assertTrue(bool((inputs["history_visual_latent"][0, -2, 1] != 0).all()))

    def test_prediction_owned_storage_and_independent_horizons(self):
        inputs, _ = fixture()
        source_before = inputs["history_visual_latent"].clone()
        result = persistence_predictions(inputs)
        second_horizon = result["pred_future_visual_latent"][:, :, 1].clone()
        result["pred_future_visual_latent"][:, :, 0].fill_(-100)
        torch.testing.assert_close(inputs["history_visual_latent"], source_before, atol=0, rtol=0)
        torch.testing.assert_close(result["pred_future_visual_latent"][:, :, 1], second_horizon, atol=0, rtol=0)

    def test_no_gradient_and_no_input_mutation(self):
        inputs, _ = fixture()
        for key in ("history_visual_latent", "history_state", "candidate_actions"):
            inputs[key].requires_grad_(True)
        before = copy.deepcopy(inputs)
        result = persistence_predictions(inputs)
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in result.values()))
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                self.assertIsNone(value.grad)
            else:
                self.assertEqual(value, before[key])

    def test_common_support_hand_computed_masks(self):
        inputs, targets = fixture(b=1, h=2)
        inputs["history_visual_valid"][0, -1] = torch.tensor([True, False])
        targets["future_visual_valid"][0] = torch.tensor([[False, True], [True, True]])
        inputs["history_state_valid"][0, -1, 3] = False
        targets["state_target_valid"][0, 0, 2] = False
        scored = persistence_scoring_targets(inputs, targets)
        self.assertTrue(torch.equal(scored["future_visual_valid"], torch.tensor([[[False, False], [True, False]]])))
        expected = targets["state_target_valid"].clone()
        expected[:, :, 3] = False
        self.assertTrue(torch.equal(scored["state_target_valid"], expected))
        self.assertTrue(bool((scored["future_visual_latent"][~scored["future_visual_valid"]] == 0).all()))
        self.assertTrue(bool((scored["state_delta"][~expected] == 0).all()))

    def test_invalid_target_poison_cleared(self):
        inputs, targets = fixture()
        targets["future_visual_valid"][0, 0, 0] = False
        targets["future_visual_latent"][0, 0, 0] = float("nan")
        targets["state_target_valid"][1, 1, 1] = False
        targets["state_delta"][1, 1, 1] = float("inf")
        scored = persistence_scoring_targets(inputs, targets)
        self.assertTrue(bool(torch.isfinite(scored["future_visual_latent"]).all()))
        self.assertEqual(scored["state_delta"][1, 1, 1].item(), 0.0)

    def test_original_valid_target_poison_not_hidden_by_common_mask(self):
        for value_key, input_mask, index in (
            ("future_visual_latent", "history_visual_valid", (0, 0, 0)),
            ("state_delta", "history_state_valid", (0, 0, 0)),
        ):
            inputs, targets = fixture()
            inputs[input_mask][0, -1, 0] = False
            targets[value_key][index] = float("nan")
            with self.subTest(key=value_key), self.assertRaises(ValueError):
                persistence_scoring_targets(inputs, targets)

    def test_all_valid_targets_exact_and_owned_copy(self):
        inputs, targets = fixture()
        before = copy.deepcopy(targets)
        scored = persistence_scoring_targets(inputs, targets)
        self.assert_tensors_equal(scored, targets)
        for key in targets:
            self.assertNotEqual(scored[key].data_ptr(), targets[key].data_ptr())
            scored[key].fill_(False if scored[key].dtype == torch.bool else 99.0)
        self.assert_tensors_equal(before, targets)

    def test_empty_common_support_returned_without_loss_computation(self):
        inputs, targets = fixture()
        inputs["history_visual_valid"][:, -1] = False
        inputs["history_state_valid"][:, -1] = False
        scored = persistence_scoring_targets(inputs, targets)
        self.assertTrue(all(bool((value == 0).all()) for value in scored.values()))
        with self.assertRaisesRegex(ValueError, "no observed support"):
            native_world_model_loss(persistence_predictions(inputs), scored)

    def test_target_keys_shapes_types_and_gradients_rejected(self):
        for key, value in (
            ("metadata", {}), ("state_delta", torch.zeros(2, 1, 3, 8)),
            ("state_target_valid", torch.ones(2, 3, 8)),
            ("future_visual_latent", torch.zeros(2, 3, 2, 5, dtype=torch.float64)),
            ("state_delta", torch.zeros(2, 3, 8, requires_grad=True)),
        ):
            inputs, targets = fixture()
            targets[key] = value
            with self.subTest(key=key, shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                persistence_scoring_targets(inputs, targets)

    def test_future_target_values_do_not_change_prediction(self):
        inputs, targets = fixture()
        before = persistence_predictions(inputs)
        targets["future_visual_latent"].fill_(1000)
        targets["state_delta"].fill_(-1000)
        persistence_scoring_targets(inputs, targets)
        self.assert_tensors_equal(before, persistence_predictions(inputs))

    def test_contract_records_parameter_free_no_training_scope(self):
        contract = persistence_contract()
        self.assertEqual(contract["parameters"], 0)
        for key in ("training_or_optimizer_or_backward", "model_construction_or_pi05_loading",
                    "future_targets_used_for_prediction", "gradient_tracking", "older_valid_observation_fallback"):
            self.assertIs(contract[key], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
