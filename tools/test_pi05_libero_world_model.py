"""CPU forward-only synthetic tests; no checkpoint, gradients, optimizer or quality claim."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

import numpy as np
import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model import (
        INPUT_KEYS, OUTPUT_KEYS, LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs,
    )
else:
    from .pi05_libero_world_model import (
        INPUT_KEYS, OUTPUT_KEYS, LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs,
    )


REGISTRY = [
    {"task_id": 9, "source_task_index": 39, "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"},
    {"task_id": 0, "source_task_index": 34, "task_instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate"},
]


def inputs(config, batch=2, candidates=3):
    generator = torch.Generator().manual_seed(709)
    return {
        "history_visual_latent": torch.randn(batch, config.context_len, 2, config.visual_dim, generator=generator),
        "history_visual_valid": torch.ones(batch, config.context_len, 2, dtype=torch.bool),
        "history_state": torch.randn(batch, config.context_len, 8, generator=generator),
        "history_state_valid": torch.ones(batch, config.context_len, 8, dtype=torch.bool),
        "task_instruction": [REGISTRY[index % 2]["task_instruction"] for index in range(batch)],
        "candidate_actions": torch.randn(batch, candidates, config.horizon, 7, generator=generator) * 3,
    }


def clone_inputs(batch):
    return {key: value.clone() if isinstance(value, torch.Tensor) else list(value) for key, value in batch.items()}


def numpy_items(batch):
    return [{key: (value[index].numpy().copy() if isinstance(value, torch.Tensor) else value[index])
             for key, value in batch.items()} for index in range(len(batch["task_instruction"]))]


class ConfigTests(unittest.TestCase):
    def test_native_defaults(self):
        self.assertEqual(LiberoWorldModelConfig(), LiberoWorldModelConfig(
            visual_dim=2048, hidden_dim=128, context_len=4, horizon=3,
            state_dim=8, action_dim=7, views=2, dropout=0.0))

    def test_config_is_immutable(self):
        config = LiberoWorldModelConfig()
        with self.assertRaises(FrozenInstanceError):
            config.action_dim = 9

    def test_native_dimensions_reject_guidewire_or_extra_views(self):
        for change in ({"state_dim": 32}, {"action_dim": 9}, {"views": 3}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "native LIBERO"):
                LiberoWorldModelConfig(**change)

    def test_dimension_types_and_ranges_rejected(self):
        for name in ("visual_dim", "hidden_dim", "context_len", "horizon", "state_dim", "action_dim", "views"):
            for value in (0, -1, True, 3.0):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LiberoWorldModelConfig(**{name: value})

    def test_dropout_invalid_rejected(self):
        for value in (-0.1, 1.0, float("nan"), float("inf"), True, "0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "dropout"):
                LiberoWorldModelConfig(dropout=value)

    def test_non_native_sizes_may_be_configured_without_changing_interface(self):
        config = LiberoWorldModelConfig(visual_dim=12, hidden_dim=24, context_len=5, horizon=2, dropout=.2)
        self.assertEqual((config.state_dim, config.action_dim, config.views), (8, 7, 2))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20260912)
        self.config = LiberoWorldModelConfig(visual_dim=16, hidden_dim=24)
        self.registry = deepcopy(REGISTRY)
        self.model = LiberoWorldModel(self.config, task_registry=self.registry).eval()
        self.batch = inputs(self.config)

    def run_model(self, batch=None, model=None):
        with torch.inference_mode():
            return (model or self.model)(**(self.batch if batch is None else batch))

    def assert_outputs_equal(self, left, right, atol=1e-7):
        self.assertEqual(set(left), OUTPUT_KEYS)
        self.assertEqual(set(right), OUTPUT_KEYS)
        for key in OUTPUT_KEYS:
            torch.testing.assert_close(left[key], right[key], atol=atol, rtol=0)

    def test_batched_multicandidate_shapes_dtype_finite(self):
        out = self.run_model()
        self.assertEqual(set(out), OUTPUT_KEYS)
        self.assertEqual(out["pred_future_visual_latent"].shape, (2, 3, 3, 2, 16))
        self.assertEqual(out["pred_state_delta"].shape, (2, 3, 3, 8))
        for value in out.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)

    def test_default_real_feature_width_forward(self):
        config = LiberoWorldModelConfig()
        model = LiberoWorldModel(config, task_registry=REGISTRY).eval()
        out = self.run_model(inputs(config, batch=2, candidates=2), model)
        self.assertEqual(out["pred_future_visual_latent"].shape, (2, 2, 3, 2, 2048))
        self.assertEqual(out["pred_state_delta"].shape, (2, 2, 3, 8))

    def test_candidate_permutation_equivariance(self):
        original = self.run_model()
        permutation = [2, 0, 1]
        changed = clone_inputs(self.batch)
        changed["candidate_actions"] = changed["candidate_actions"][:, permutation]
        out = self.run_model(changed)
        self.assert_outputs_equal({key: value[:, permutation] for key, value in original.items()}, out)

    def test_candidate_independence(self):
        original = self.run_model()
        changed = clone_inputs(self.batch)
        changed["candidate_actions"][:, 1] += 4
        out = self.run_model(changed)
        for key in OUTPUT_KEYS:
            torch.testing.assert_close(original[key][:, [0, 2]], out[key][:, [0, 2]], atol=0, rtol=0)
            self.assertFalse(torch.equal(original[key][:, 1], out[key][:, 1]))

    def test_identical_candidates_produce_identical_outputs(self):
        batch = clone_inputs(self.batch)
        batch["candidate_actions"][:, 2] = batch["candidate_actions"][:, 0]
        out = self.run_model(batch)
        for value in out.values():
            torch.testing.assert_close(value[:, 0], value[:, 2], atol=0, rtol=0)

    def test_batch_permutation_equivariance(self):
        original = self.run_model()
        changed = {key: value.flip(0) if isinstance(value, torch.Tensor) else value[::-1]
                   for key, value in self.batch.items()}
        out = self.run_model(changed)
        # Different CPU GEMM/GRU batch orderings can round float32 differently
        # (observed max 2.384185791015625e-7). This local numerical tolerance
        # does not permit batch mixing or weaken mask/candidate/causality tests.
        self.assert_outputs_equal({key: value.flip(0) for key, value in original.items()}, out, atol=1e-6)

    def test_future_action_changes_cannot_change_earlier_predictions(self):
        original = self.run_model()
        for step in (1, 2):
            changed = clone_inputs(self.batch)
            changed["candidate_actions"][:, :, step:] += 6
            out = self.run_model(changed)
            for key in OUTPUT_KEYS:
                torch.testing.assert_close(original[key][:, :, :step], out[key][:, :, :step], atol=0, rtol=0)
                self.assertFalse(torch.equal(original[key][:, :, step:], out[key][:, :, step:]))

    def test_task_changes_have_wiring_effect_not_semantic_claim(self):
        original = self.run_model()
        changed = clone_inputs(self.batch)
        changed["task_instruction"] = changed["task_instruction"][::-1]
        out = self.run_model(changed)
        self.assertTrue(all(not torch.equal(original[key], out[key]) for key in OUTPUT_KEYS))
        self.assertFalse(self.model.metadata()["semantic_or_open_language_grounding"])

    def test_masked_nan_inf_invariance_including_current_anchor(self):
        baseline = clone_inputs(self.batch)
        baseline["history_visual_valid"][:, [0, 3], 0] = False
        baseline["history_state_valid"][:, [1, 3], [1, 5]] = False
        baseline["history_visual_latent"][~baseline["history_visual_valid"]] = 0
        baseline["history_state"][~baseline["history_state_valid"]] = 0
        expected = self.run_model(baseline)
        for value in (float("nan"), float("inf"), -float("inf"), 1e30):
            changed = clone_inputs(baseline)
            changed["history_visual_latent"][~changed["history_visual_valid"]] = value
            changed["history_state"][~changed["history_state_valid"]] = value
            self.assert_outputs_equal(expected, self.run_model(changed), atol=0)

    def test_all_invalid_history_stays_finite_without_availability_claim(self):
        changed = clone_inputs(self.batch)
        changed["history_visual_valid"].fill_(False)
        changed["history_state_valid"].fill_(False)
        changed["history_visual_latent"].fill_(float("nan"))
        changed["history_state"].fill_(float("inf"))
        out = self.run_model(changed)
        self.assertTrue(all(torch.isfinite(value).all() for value in out.values()))

    def test_explicit_masks_distinguish_missing_from_observed_zero(self):
        observed = clone_inputs(self.batch)
        observed["history_visual_latent"].zero_()
        observed["history_state"].zero_()
        missing = clone_inputs(observed)
        missing["history_visual_valid"].fill_(False)
        missing["history_state_valid"].fill_(False)
        left, right = self.run_model(observed), self.run_model(missing)
        self.assertTrue(all(not torch.equal(left[key], right[key]) for key in OUTPUT_KEYS))

    def test_current_visual_anchor_and_state_delta_have_different_baselines(self):
        # A zero residual head is a wiring test, not fitting or optimization.
        with torch.no_grad():
            self.model.visual_residual_head.weight.zero_()
            self.model.visual_residual_head.bias.zero_()
            self.model.state_residual_head.weight.zero_()
            self.model.state_residual_head.bias.zero_()
        self.batch["history_visual_valid"][:, -1, 0] = False
        self.batch["history_visual_latent"][:, -1, 0] = float("nan")
        out = self.run_model()
        clean = torch.where(self.batch["history_visual_valid"][:, -1, :, None], self.batch["history_visual_latent"][:, -1], 0)
        expected = clean[:, None, None].expand(2, 3, 3, 2, 16)
        torch.testing.assert_close(out["pred_future_visual_latent"], expected, atol=0, rtol=0)
        self.assertTrue((out["pred_state_delta"] == 0).all())

    def test_native_action7_and_state8_reject_guidewire_shapes(self):
        for key, value in (("candidate_actions", torch.zeros(2, 3, 3, 9)), ("history_state", torch.zeros(2, 4, 32))):
            changed = clone_inputs(self.batch)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_model(changed)

    def test_train_normalized_actions_are_not_clipped_or_rejected(self):
        changed = clone_inputs(self.batch)
        changed["candidate_actions"].fill_(3.0)
        original = self.run_model(changed)
        changed["candidate_actions"].fill_(1.0)
        clipped = self.run_model(changed)
        self.assertTrue(all(not torch.equal(original[key], clipped[key]) for key in OUTPUT_KEYS))

    def test_nonfinite_valid_visual_state_or_any_action_rejected(self):
        for key in ("history_visual_latent", "history_state", "candidate_actions"):
            for value in (float("nan"), float("inf")):
                changed = clone_inputs(self.batch)
                changed[key].flatten()[0] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    self.run_model(changed)

    def test_mask_nonbool_and_float64_forward_rejected(self):
        for key in ("history_visual_valid", "history_state_valid", "history_state", "candidate_actions"):
            changed = clone_inputs(self.batch)
            changed[key] = changed[key].double()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_model(changed)

    def test_unknown_task_and_wrong_task_batch_rejected(self):
        for tasks in (["invented task"] * 2, [REGISTRY[0]["task_instruction"]], tuple(self.batch["task_instruction"]),
                      [REGISTRY[0]["task_instruction"] + " ", REGISTRY[1]["task_instruction"]]):
            changed = clone_inputs(self.batch)
            changed["task_instruction"] = tasks
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, "exact known"):
                self.run_model(changed)

    def test_forward_unknown_keys_and_positional_inputs_rejected(self):
        for key in ("targets", "metadata", "contact_truth"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                self.model(**self.batch, **{key: {}})
        with self.assertRaises(TypeError):
            self.model(self.batch)

    def test_registry_is_copied_sorted_and_metadata_is_not_mutable_state(self):
        metadata = self.model.metadata()
        self.assertEqual([row["task_id"] for row in metadata["task_registry_encoding"]], [0, 9])
        before = self.run_model()
        self.registry[0]["task_instruction"] = "modified outside model"
        metadata["task_registry_encoding"][0]["task_instruction"] = "modified metadata copy"
        self.assert_outputs_equal(before, self.run_model(), atol=0)
        self.assertEqual(self.model.metadata()["task_registry_encoding"][0]["task_instruction"], REGISTRY[1]["task_instruction"])
        encoding = self.model.metadata()["task_registry_encoding"]
        digest = hashlib.sha256(json.dumps(encoding, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(self.model.metadata()["task_registry_encoding_sha256"], digest)

    def test_registry_source_order_does_not_change_encoding_or_initialized_forward(self):
        torch.manual_seed(20260912)
        other = LiberoWorldModel(self.config, task_registry=list(reversed(REGISTRY))).eval()
        self.assertEqual(self.model.metadata(), other.metadata())
        self.assert_outputs_equal(self.run_model(), self.run_model(model=other), atol=0)

    def test_registry_duplicates_and_unknown_fields_rejected(self):
        for registry in ([REGISTRY[0], REGISTRY[0]], [{**REGISTRY[0], "reward": 1}], []):
            with self.subTest(registry=registry), self.assertRaises(ValueError):
                LiberoWorldModel(self.config, task_registry=registry)

    def test_no_caller_mutation_parameter_change_or_gradient(self):
        before_inputs = clone_inputs(self.batch)
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        first, second = self.run_model(), self.run_model()
        self.assert_outputs_equal(first, second, atol=0)
        for key, value in self.batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before_inputs[key], atol=0, rtol=0)
            else:
                self.assertEqual(value, before_inputs[key])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))


class CollationTests(unittest.TestCase):
    def setUp(self):
        self.config = LiberoWorldModelConfig(visual_dim=16, hidden_dim=24)
        self.batch = inputs(self.config)
        self.items = numpy_items(self.batch)

    def test_six_input_keys_owned_storage_float32_bool_exact_text(self):
        snapshots = deepcopy(self.items)
        batch = collate_window_inputs(self.items)
        self.assertEqual(set(batch), INPUT_KEYS)
        for key, expected in self.batch.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(batch[key], expected, atol=0, rtol=0)
            else:
                self.assertEqual(batch[key], expected)
        batch["history_state"].zero_()
        batch["history_state_valid"].fill_(False)
        batch["task_instruction"][0] = "changed owned list"
        for original, snapshot in zip(self.items, snapshots):
            for key in original:
                if isinstance(original[key], np.ndarray):
                    self.assertTrue(np.array_equal(original[key], snapshot[key]))
                else:
                    self.assertEqual(original[key], snapshot[key])

    def test_targets_metadata_or_unknown_fields_rejected(self):
        for key in ("targets", "metadata", "estimated_contact_flag"):
            items = deepcopy(self.items)
            items[0][key] = {}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "six native input keys"):
                collate_window_inputs(items)

    def test_whole_window_item_and_empty_list_rejected(self):
        for value in ([], [{"inputs": self.items[0], "targets": {}, "metadata": {}}], tuple(self.items)):
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                collate_window_inputs(value)

    def test_native_shapes_required_before_forward(self):
        for key, value in (("candidate_actions", np.zeros((3, 3, 9), np.float32)),
                           ("history_state", np.zeros((4, 32), np.float32)),
                           ("history_visual_latent", np.zeros((4, 3, 16), np.float32))):
            items = deepcopy(self.items)
            for item in items:
                item[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "native window input shapes"):
                collate_window_inputs(items)

    def test_nonbool_masks_and_mismatched_batch_shapes_rejected(self):
        items = deepcopy(self.items)
        items[0]["history_visual_valid"] = items[0]["history_visual_valid"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "boolean NumPy mask"):
            collate_window_inputs(items)
        items = deepcopy(self.items)
        items[0]["candidate_actions"] = items[0]["candidate_actions"][:1]
        with self.assertRaisesRegex(ValueError, "match across"):
            collate_window_inputs(items)

    def test_float64_copies_to_float32_without_normalization(self):
        for item in self.items:
            item["candidate_actions"] = np.full_like(item["candidate_actions"], 5, dtype=np.float64)
        batch = collate_window_inputs(self.items)
        self.assertEqual(batch["candidate_actions"].dtype, torch.float32)
        self.assertTrue((batch["candidate_actions"] == 5).all())

    def test_masked_nonfinite_permitted_observed_nonfinite_rejected(self):
        self.items[0]["history_visual_valid"][3, 0] = False
        self.items[0]["history_visual_latent"][3, 0] = np.nan
        batch = collate_window_inputs(self.items)
        self.assertTrue(torch.isnan(batch["history_visual_latent"][0, 3, 0]).all())
        self.items[0]["history_visual_valid"][3, 0] = True
        with self.assertRaisesRegex(ValueError, "valid collated visual"):
            collate_window_inputs(self.items)

    def test_nonfinite_action_and_float32_overflow_rejected(self):
        for value in (float("nan"), float("inf"), 1e200):
            items = deepcopy(self.items)
            items[0]["candidate_actions"] = np.full(items[0]["candidate_actions"].shape, value, dtype=np.float64)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "actions must be finite"):
                collate_window_inputs(items)


if __name__ == "__main__":
    unittest.main()
