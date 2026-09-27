"""Synthetic forward-only ablation tests; no backward, optimizer or data read."""
from __future__ import annotations

import unittest

import torch

if __package__ in {None, ""}:
    from pi05_libero_action_ablation import (
        ACTION_ABLATION_MODES, action_ablation_contract, action_ablation_inputs,
    )
    from pi05_libero_world_model import INPUT_KEYS, LiberoWorldModel, LiberoWorldModelConfig
else:
    from .pi05_libero_action_ablation import (
        ACTION_ABLATION_MODES, action_ablation_contract, action_ablation_inputs,
    )
    from .pi05_libero_world_model import INPUT_KEYS, LiberoWorldModel, LiberoWorldModelConfig


TASK = "pick up the black bowl on the wooden cabinet and place it on the plate"
REGISTRY = [{"task_id": 9, "source_task_index": 39, "task_instruction": TASK}]


def fixture() -> dict:
    generator = torch.Generator(device="cpu").manual_seed(8401)
    return {
        "history_visual_latent": torch.randn(2, 4, 2, 2048, generator=generator),
        "history_visual_valid": torch.ones(2, 4, 2, dtype=torch.bool),
        "history_state": torch.randn(2, 4, 8, generator=generator),
        "history_state_valid": torch.ones(2, 4, 8, dtype=torch.bool),
        "task_instruction": [TASK, TASK],
        "candidate_actions": torch.randn(2, 1, 3, 7, generator=generator),
    }


def snapshot(inputs: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else list(value)
            for key, value in inputs.items()}


class ActionAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.inputs = fixture()

    def assert_same(self, left, right):
        self.assertEqual(set(left), set(right))
        for key, value in left.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right[key], rtol=0, atol=0, equal_nan=True)
            else:
                self.assertEqual(value, right[key])

    def test_contract_is_fresh_and_explicit(self):
        contract = action_ablation_contract()
        self.assertEqual(contract["modes"], list(ACTION_ABLATION_MODES))
        self.assertIn("train action mean", contract["normalized_zero_action"])
        self.assertIn("not physical zero motion or hold", contract["normalized_zero_action"])
        self.assertFalse(contract["targets_or_metadata_accepted"])
        self.assertFalse(contract["arm_marker_passed_to_model"])
        self.assertFalse(contract["training_started_by_module"])
        self.assertFalse(contract["causal_or_counterfactual_claim_allowed"])
        self.assertIn("all-zero gradient", contract["zero_arm_gradient_note"])
        contract["modes"].clear()
        self.assertEqual(action_ablation_contract()["modes"], list(ACTION_ABLATION_MODES))

    def test_observed_copies_every_value_without_coercion(self):
        result = action_ablation_inputs(self.inputs, "observed_action")
        self.assertEqual(set(result), INPUT_KEYS)
        self.assert_same(result, self.inputs)
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.layout, self.inputs[key].layout)
                self.assertEqual(value.dtype, self.inputs[key].dtype)
                self.assertEqual(value.shape, self.inputs[key].shape)
                self.assertFalse(value.requires_grad)
                self.assertFalse(torch.is_inference(value))
                self.assertIsNone(value.grad_fn)
                self.assertIsNone(value._base)

    def test_zero_changes_only_candidate_actions(self):
        before = snapshot(self.inputs)
        result = action_ablation_inputs(self.inputs, "normalized_zero_action")
        expected = snapshot(self.inputs)
        expected["candidate_actions"].zero_()
        self.assert_same(result, expected)
        self.assert_same(self.inputs, before)
        self.assertEqual(torch.count_nonzero(result["candidate_actions"]).item(), 0)

    def test_storage_and_tasks_are_owned_in_both_directions(self):
        for mode in ACTION_ABLATION_MODES:
            original = fixture()
            result = action_ablation_inputs(original, mode)
            saved = snapshot(result)
            original_storage = {value.untyped_storage().data_ptr()
                                for value in original.values() if isinstance(value, torch.Tensor)}
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    self.assertNotIn(value.untyped_storage().data_ptr(), original_storage)
                    original[key].fill_(False if value.dtype == torch.bool else 17)
                else:
                    self.assertIsNot(value, original[key])
                    original[key][0] = "different task"
            self.assert_same(result, saved)
            saved_original = snapshot(original)
            for value in result.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
                else:
                    value.clear()
            self.assert_same(original, saved_original)

    def test_targets_and_metadata_cannot_enter(self):
        targets = {"future_state": torch.randn(2, 3, 8)}
        metadata = {"partition": "train"}
        saved = targets["future_state"].clone()
        with self.assertRaises(TypeError):
            action_ablation_inputs(self.inputs, "observed_action", targets=targets)
        with self.assertRaises(TypeError):
            action_ablation_inputs(self.inputs, "observed_action", metadata=metadata)
        torch.testing.assert_close(targets["future_state"], saved, rtol=0, atol=0)
        self.assertEqual(metadata, {"partition": "train"})

    def test_extra_missing_or_non_dict_inputs_rejected(self):
        for key in ("targets", "metadata", "diagnostic_targets", "ablation_mode", "sample_role"):
            changed = dict(self.inputs, **{key: "forbidden"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "six native"):
                action_ablation_inputs(changed, "observed_action")
        for key in INPUT_KEYS:
            changed = dict(self.inputs)
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                action_ablation_inputs(changed, "observed_action")
        with self.assertRaises(ValueError):
            action_ablation_inputs(list(self.inputs.items()), "observed_action")

    def test_mode_and_task_validation(self):
        for mode in (None, True, 0, [], "zero", "hold", "observed_action "):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                action_ablation_inputs(self.inputs, mode)
        for tasks in ((TASK, TASK), [TASK], [TASK, ""], [TASK, "  "], [TASK, 1]):
            changed = dict(self.inputs, task_instruction=tasks)
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                action_ablation_inputs(changed, "observed_action")

    def test_dtypes_are_never_coerced(self):
        for key, value in self.inputs.items():
            if isinstance(value, torch.Tensor):
                changed = dict(self.inputs)
                changed[key] = value.to(torch.float64 if value.dtype == torch.float32 else torch.float32)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "no coercion"):
                    action_ablation_inputs(changed, "observed_action")

    def test_fixed_shapes_empty_batch_and_multicandidate_rejected(self):
        alternatives = {
            "history_visual_latent": [(2, 5, 2, 2048), (2, 4, 2, 16), (0, 4, 2, 2048)],
            "history_visual_valid": [(2, 4, 1), (3, 4, 2)],
            "history_state": [(2, 4, 32), (2, 3, 8)],
            "history_state_valid": [(2, 4, 1)],
            "candidate_actions": [(2, 2, 3, 7), (2, 1, 4, 7), (2, 1, 3, 9), (1, 1, 3, 7)],
        }
        for key, shapes in alternatives.items():
            for shape in shapes:
                changed = dict(self.inputs)
                changed[key] = torch.zeros(shape, dtype=self.inputs[key].dtype)
                with self.subTest(key=key, shape=shape), self.assertRaises(ValueError):
                    action_ablation_inputs(changed, "normalized_zero_action")

    def test_nonfinite_actions_rejected_before_zeroing(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for mode in ACTION_ABLATION_MODES:
                changed = snapshot(self.inputs)
                changed["candidate_actions"][0, 0, 0, 0] = bad
                before = snapshot(changed)
                with self.subTest(bad=bad, mode=mode), self.assertRaisesRegex(ValueError, "before ablation"):
                    action_ablation_inputs(changed, mode)
                self.assert_same(changed, before)

    def test_nonfinite_valid_history_rejected(self):
        for key in ("history_visual_latent", "history_state"):
            changed = snapshot(self.inputs)
            changed[key].flatten()[0] = float("nan")
            for mode in ACTION_ABLATION_MODES:
                with self.subTest(key=key, mode=mode), self.assertRaisesRegex(ValueError, "valid .* coordinates"):
                    action_ablation_inputs(changed, mode)

    def test_invalid_history_preserved_with_boolean_masks(self):
        self.inputs["history_visual_valid"][0, 0, 0] = False
        self.inputs["history_visual_latent"][0, 0, 0].fill_(float("nan"))
        self.inputs["history_state_valid"][1, 3, 7] = False
        self.inputs["history_state"][1, 3, 7] = float("inf")
        model = self.model()
        for mode in ACTION_ABLATION_MODES:
            result = action_ablation_inputs(self.inputs, mode)
            for key in ("history_visual_latent", "history_visual_valid", "history_state", "history_state_valid"):
                torch.testing.assert_close(result[key], self.inputs[key], rtol=0, atol=0, equal_nan=True)
            with torch.inference_mode():
                outputs = model(**result)
            self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_gradient_inputs_rejected(self):
        for key in ("history_visual_latent", "history_state", "candidate_actions"):
            changed = snapshot(self.inputs)
            changed[key].requires_grad_(True)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "non-gradient"):
                action_ablation_inputs(changed, "observed_action")

    def test_all_invalid_history_is_not_silently_promoted(self):
        self.inputs["history_visual_valid"].zero_()
        self.inputs["history_state_valid"].zero_()
        self.inputs["history_visual_latent"].fill_(float("nan"))
        self.inputs["history_state"].fill_(float("inf"))
        for mode in ACTION_ABLATION_MODES:
            result = action_ablation_inputs(self.inputs, mode)
            self.assertFalse(result["history_visual_valid"].any())
            self.assertFalse(result["history_state_valid"].any())
            self.assertTrue(torch.isnan(result["history_visual_latent"]).all())
            self.assertTrue(torch.isinf(result["history_state"]).all())

    def test_inference_tensors_including_masks_rejected(self):
        for key, value in self.inputs.items():
            if isinstance(value, torch.Tensor):
                changed = dict(self.inputs)
                with torch.inference_mode():
                    changed[key] = value.clone()
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inference tensor"):
                    action_ablation_inputs(changed, "observed_action")

    def test_inference_context_rejected_no_grad_context_ordinary(self):
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "outside inference_mode"):
            action_ablation_inputs(self.inputs, "observed_action")
        with torch.no_grad():
            result = action_ablation_inputs(self.inputs, "observed_action")
        for value in result.values():
            if isinstance(value, torch.Tensor):
                self.assertFalse(torch.is_inference(value))

    def test_non_cpu_and_sparse_tensors_rejected_without_cuda(self):
        for replacement in (torch.empty(2, 1, 3, 7, device="meta"),
                            self.inputs["candidate_actions"].to_sparse()):
            changed = dict(self.inputs, candidate_actions=replacement)
            with self.assertRaises(ValueError):
                action_ablation_inputs(changed, "normalized_zero_action")

    def test_noncontiguous_dense_values_copied_without_alias(self):
        self.inputs["candidate_actions"] = torch.arange(84, dtype=torch.float32).reshape(2, 1, 7, 6)[..., ::2].transpose(2, 3)
        self.assertEqual(self.inputs["candidate_actions"].shape, (2, 1, 3, 7))
        self.assertFalse(self.inputs["candidate_actions"].is_contiguous())
        result = action_ablation_inputs(self.inputs, "observed_action")
        self.assert_same(result, self.inputs)
        self.assertNotEqual(result["candidate_actions"].untyped_storage().data_ptr(),
                            self.inputs["candidate_actions"].untyped_storage().data_ptr())

    @staticmethod
    def model():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260912)
            model = LiberoWorldModel(LiberoWorldModelConfig(hidden_dim=16), task_registry=REGISTRY)
        return model.eval().requires_grad_(False)

    def test_equal_initial_models_zero_arm_is_action_invariant(self):
        first, second = self.model(), self.model()
        changed = snapshot(self.inputs)
        changed["candidate_actions"].add_(11)
        one = action_ablation_inputs(self.inputs, "normalized_zero_action")
        two = action_ablation_inputs(changed, "normalized_zero_action")
        saved = {key: value.clone() for key, value in first.state_dict().items()}
        for key, value in second.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        with torch.inference_mode():
            left, right = first(**one), second(**two)
        self.assert_same(left, right)
        for model in (first, second):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_observed_arm_retains_action_difference_and_forward_can_change(self):
        first, second = self.model(), self.model()
        changed = snapshot(self.inputs)
        changed["candidate_actions"].add_(11)
        one = action_ablation_inputs(self.inputs, "observed_action")
        two = action_ablation_inputs(changed, "observed_action")
        self.assertFalse(torch.equal(one["candidate_actions"], two["candidate_actions"]))
        with torch.inference_mode():
            left, right = first(**one), second(**two)
        for key in left:
            self.assertGreater((left[key] - right[key]).abs().max().item(), 0)
        self.assertTrue(all(parameter.grad is None for model in (first, second) for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
