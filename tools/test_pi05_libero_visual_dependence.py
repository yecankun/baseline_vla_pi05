"""Synthetic CPU-only contracts and native frozen inference; no real data/jobs."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_visual_dependence as diagnostic
from pi05_libero_action_ablation import ACTION_ABLATION_MODES
from pi05_libero_world_model import INPUT_KEYS, LiberoWorldModel, LiberoWorldModelConfig


REGISTRY = [{"task_id": 9, "source_task_index": 39,
             "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]


def fixture(batch=2):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(841)
        model = LiberoWorldModel(LiberoWorldModelConfig(hidden_dim=5), REGISTRY)
    model.eval().requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(842)
    inputs = {"history_visual_latent": torch.randn(batch, 4, 2, 2048, generator=generator),
              "history_visual_valid": torch.ones(batch, 4, 2, dtype=torch.bool),
              "history_state": torch.randn(batch, 4, 8, generator=generator),
              "history_state_valid": torch.ones(batch, 4, 8, dtype=torch.bool),
              "candidate_actions": torch.randn(batch, 1, 3, 7, generator=generator),
              "task_instruction": [REGISTRY[0]["task_instruction"]] * batch}
    anchor = torch.randn(2, 2048, generator=generator)
    return model, inputs, anchor


def capture_fixture(inputs, residual=0.0, state=0.0):
    """Analytic numbers solely for reductions, never claimed as model output."""
    batch = inputs["history_visual_latent"].shape[0]
    anchor = inputs["history_visual_latent"][:, -1, None, None].expand(batch, 1, 3, 2, 2048).clone()
    residual = torch.full_like(anchor, residual)
    return {"predictions": {"pred_future_visual_latent": anchor + residual,
                            "pred_state_delta": torch.full((batch, 1, 3, 8), state)},
            "residual": residual, "anchor": anchor}


def batch_slice(inputs, index):
    return {key: value[index:index + 1].clone() if isinstance(value, torch.Tensor)
            else list(value[index:index + 1]) for key, value in inputs.items()}


def capture_slice(capture, index):
    return {"predictions": {key: value[index:index + 1].clone() for key, value in capture["predictions"].items()},
            "residual": capture["residual"][index:index + 1].clone(),
            "anchor": capture["anchor"][index:index + 1].clone()}


class VisualDependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assertTensorExact(self, left, right):
        self.assertTrue(diagnostic._same_tensor(left, right))

    def test_transform_six_conditions_preserve_nonvisual_and_callers(self):
        _, raw, anchor = fixture()
        before, anchor_before = deepcopy(raw), anchor.clone()
        for arm in ACTION_ABLATION_MODES:
            for condition in diagnostic.CONDITIONS:
                result = diagnostic.transform_inputs(raw, arm, condition, anchor)
                self.assertEqual(set(result), INPUT_KEYS)
                self.assertEqual(result["task_instruction"], raw["task_instruction"])
                self.assertIsNot(result["task_instruction"], raw["task_instruction"])
                for key, value in result.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    self.assertNotEqual(value.data_ptr(), raw[key].data_ptr())
                    self.assertFalse(value.requires_grad or torch.is_inference(value))
                    if key != "history_visual_latent":
                        expected = torch.zeros_like(value) if key == "candidate_actions" and arm == "normalized_zero_action" else raw[key]
                        self.assertTensorExact(value, expected)
                visual = result["history_visual_latent"]
                expected = (raw["history_visual_latent"] if condition == "clean" else
                            raw["history_visual_latent"][:, -1:].expand_as(visual) if condition == "repeat_current" else
                            anchor[None, None].expand_as(visual))
                self.assertTensorExact(visual, expected)
                self.assertTrue(diagnostic._same_inputs(raw, before))
                self.assertTensorExact(anchor, anchor_before)

    def test_transform_outputs_own_storage_between_rows_conditions_and_anchor(self):
        _, raw, anchor = fixture()
        first = diagnostic.transform_inputs(raw, "observed_action", "fixed_train_anchor", anchor)
        second = diagnostic.transform_inputs(raw, "observed_action", "fixed_train_anchor", anchor)
        original = anchor.clone()
        first["history_visual_latent"][0, 0].add_(10)
        self.assertTensorExact(first["history_visual_latent"][1, 0], original)
        self.assertTensorExact(first["history_visual_latent"][0, 1], original)
        self.assertTensorExact(second["history_visual_latent"][0, 0], original)
        self.assertTensorExact(anchor, original)
        first["history_state"].zero_()
        self.assertFalse(torch.equal(first["history_state"], raw["history_state"]))

    def test_transform_rejects_unknown_conditions_arms_keys_and_harmful_types(self):
        _, raw, anchor = fixture()
        for condition in ("unknown", None, 1, ["clean"]):
            with self.assertRaises(ValueError):
                diagnostic.transform_inputs(raw, "observed_action", condition, anchor)
        for arm in ("unknown", None, 1):
            with self.assertRaises(ValueError):
                diagnostic.transform_inputs(raw, arm, "clean", anchor)
        for field in ("targets", "episode_index", "condition"):
            with self.assertRaises(ValueError):
                diagnostic.transform_inputs({**raw, field: 0}, "observed_action", "clean", anchor)
        for altered in ({**raw, "task_instruction": tuple(raw["task_instruction"])},
                        {**raw, "history_visual_latent": raw["history_visual_latent"].double()}):
            with self.assertRaises(ValueError):
                diagnostic.transform_inputs(altered, "observed_action", "clean", anchor)

    def test_transform_requires_all_true_masks_in_every_condition(self):
        _, raw, anchor = fixture()
        for key in ("history_visual_valid", "history_state_valid"):
            changed = deepcopy(raw)
            changed[key].flatten()[0] = False
            for condition in diagnostic.CONDITIONS:
                with self.assertRaisesRegex(ValueError, "all history masks true"):
                    diagnostic.transform_inputs(changed, "observed_action", condition, anchor)

    def test_transform_rejects_bad_anchor_and_corrupt_actions_before_zeroing(self):
        _, raw, anchor = fixture()
        invalid = [anchor.numpy(), anchor.double(), anchor[:1], anchor.clone().requires_grad_(True),
                   torch.zeros_like(anchor), torch.full_like(anchor, float("nan")),
                   torch.full_like(anchor, float("inf"))]
        with torch.inference_mode():
            invalid.append(anchor.clone())
        for value in invalid:
            with self.assertRaises(ValueError):
                diagnostic.transform_inputs(raw, "observed_action", "clean", value)
        raw["candidate_actions"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "before ablation"):
            diagnostic.transform_inputs(raw, "normalized_zero_action", "clean", anchor)

    def test_inference_context_rejected_before_constructing_owned_inputs(self):
        model, raw, anchor = fixture()
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "outside inference_mode"):
            diagnostic.transform_inputs(raw, "observed_action", "clean", anchor)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "outside inference_mode"):
            diagnostic.capture_prediction(model, raw)

    def test_capture_real_native_head_once_exact_reconstruction_owned_results(self):
        model, raw, _ = fixture()
        input_before, parameter_before = deepcopy(raw), {key: value.clone() for key, value in model.state_dict().items()}
        versions = [value._version for value in model.parameters()]
        observed = {"forward_calls": 0, "head": None, "outputs": None}

        def head_hook(_module, _args, output):
            observed["head"] = output

        def output_hook(_module, _args, output):
            observed["forward_calls"] += 1
            observed["outputs"] = output

        head_handle = model.visual_residual_head.register_forward_hook(head_hook)
        output_handle = model.register_forward_hook(output_hook)
        try:
            result = diagnostic.capture_prediction(model, raw)
            self.assertEqual(len(model.visual_residual_head._forward_hooks), 1)
        finally:
            head_handle.remove()
            output_handle.remove()
        self.assertEqual(observed["forward_calls"], 1)
        self.assertEqual(set(result), {"predictions", "residual", "anchor"})
        self.assertEqual(tuple(result["residual"].shape), (2, 1, 3, 2, 2048))
        self.assertTensorExact(result["predictions"]["pred_future_visual_latent"], result["anchor"] + result["residual"])
        self.assertTensorExact(result["residual"], observed["head"].reshape_as(result["residual"]))
        self.assertNotEqual(result["residual"].data_ptr(), observed["head"].data_ptr())
        for key, value in result["predictions"].items():
            self.assertNotEqual(value.data_ptr(), observed["outputs"][key].data_ptr())
        tensors = [result["residual"], result["anchor"], *result["predictions"].values()]
        self.assertEqual(len({value.data_ptr() for value in tensors}), len(tensors))
        self.assertTrue(all(not value.requires_grad and not torch.is_inference(value) for value in tensors))
        result["anchor"].add_(10)
        self.assertTrue(diagnostic._same_inputs(raw, input_before))
        self.assertEqual([value._version for value in model.parameters()], versions)
        self.assertTrue(all(value.grad is None and not value.requires_grad for value in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTensorExact(value, parameter_before[key])

    def test_capture_actual_residual_is_not_output_minus_anchor(self):
        model, raw, _ = fixture()
        raw["history_visual_latent"].fill_(1e8)
        model.visual_residual_head.weight.zero_()
        model.visual_residual_head.bias.fill_(0.25)
        result = diagnostic.capture_prediction(model, raw)
        self.assertTensorExact(result["residual"], torch.full_like(result["residual"], 0.25))
        self.assertEqual(int(torch.count_nonzero(result["predictions"]["pred_future_visual_latent"] - result["anchor"])), 0)

    def test_native_skip_only_model_distinguishes_repeat_from_anchor(self):
        model, raw, anchor = fixture()
        model.visual_residual_head.weight.zero_()
        model.visual_residual_head.bias.zero_()
        clean = diagnostic.transform_inputs(raw, "observed_action", "clean", anchor)
        repeated = diagnostic.transform_inputs(raw, "observed_action", "repeat_current", anchor)
        fixed = diagnostic.transform_inputs(raw, "observed_action", "fixed_train_anchor", anchor)
        clean_capture = diagnostic.capture_prediction(model, clean)
        repeated_capture = diagnostic.capture_prediction(model, repeated)
        fixed_capture = diagnostic.capture_prediction(model, fixed)
        repeat_totals, fixed_totals = diagnostic.DriftTotals(), diagnostic.DriftTotals()
        repeat_totals.update(clean_capture, repeated_capture, clean, repeated)
        fixed_totals.update(clean_capture, fixed_capture, clean, fixed)
        self.assertEqual(repeat_totals.summary()["visual_output_delta"]["exact_changed_count"], 0)
        self.assertGreater(repeat_totals.summary()["history_input_delta"]["mae"], 0)
        self.assertEqual(fixed_totals.summary()["visual_residual_delta"]["mae"], 0)
        self.assertGreater(fixed_totals.summary()["visual_output_delta"]["mae"], 0)
        self.assertEqual(fixed_totals.summary()["visual_output_delta"], fixed_totals.summary()["anchor_skip_delta"])

    def test_native_residual_can_depend_on_earlier_history_with_skip_unchanged(self):
        model, raw, anchor = fixture()
        clean = diagnostic.transform_inputs(raw, "normalized_zero_action", "clean", anchor)
        repeated = diagnostic.transform_inputs(raw, "normalized_zero_action", "repeat_current", anchor)
        clean_capture = diagnostic.capture_prediction(model, clean)
        changed_capture = diagnostic.capture_prediction(model, repeated)
        totals = diagnostic.DriftTotals()
        totals.update(clean_capture, changed_capture, clean, repeated)
        result = totals.summary()
        self.assertEqual(result["anchor_skip_delta"]["exact_changed_count"], 0)
        self.assertGreater(result["visual_residual_delta"]["exact_changed_count"], 0)
        self.assertGreater(result["state_output_delta"]["exact_changed_count"], 0)
        self.assertEqual(int(torch.count_nonzero(clean["candidate_actions"])), 0)
        self.assertEqual(int(torch.count_nonzero(repeated["candidate_actions"])), 0)

    def test_capture_rejects_nonfrozen_training_gradient_or_wrong_model(self):
        for kind in ("training", "child_training", "requires_grad", "grad", "nonfinite", "dtype", "wrong_dimensions"):
            model, raw, _ = fixture()
            if kind == "training":
                model.train()
            elif kind == "child_training":
                model.visual_residual_head.train()
            elif kind == "requires_grad":
                next(model.parameters()).requires_grad_(True)
            elif kind == "grad":
                next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
            elif kind == "nonfinite":
                next(model.parameters()).flatten()[0] = float("nan")
            elif kind == "dtype":
                model.double()
            else:
                model.config = LiberoWorldModelConfig(visual_dim=7)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                diagnostic.capture_prediction(model, raw)
            self.assertEqual(len(model.visual_residual_head._forward_hooks), 0)
        _, raw, _ = fixture()
        with self.assertRaises(ValueError):
            diagnostic.capture_prediction(torch.nn.Linear(1, 1), raw)

    def test_hook_cleanup_when_forward_raises(self):
        model, raw, _ = fixture()

        def failure(_module, _args):
            raise RuntimeError("synthetic head failure")

        existing = model.visual_residual_head.register_forward_pre_hook(failure)
        before = diagnostic._hook_state(model)
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic head failure"):
                diagnostic.capture_prediction(model, raw)
            self.assertEqual(len(model.visual_residual_head._forward_hooks), 0)
            self.assertEqual(len(model.visual_residual_head._forward_pre_hooks), 1)
            self.assertEqual(diagnostic._hook_state(model), before)
        finally:
            existing.remove()

    def test_hook_order_kwargs_and_always_called_registry_restored_on_success_and_error(self):
        for fail in (False, True):
            model, raw, _ = fixture()

            def existing_hook(_module, _args, _kwargs, _output):
                if fail:
                    raise RuntimeError("existing hook error")

            handles = [model.visual_residual_head.register_forward_hook(existing_hook, with_kwargs=True, always_call=True),
                       model.visual_residual_head.register_forward_hook(lambda *_: None),
                       model.visual_residual_head.register_forward_pre_hook(lambda *_: None, with_kwargs=True)]
            before = diagnostic._hook_state(model)
            try:
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "existing hook error"):
                        diagnostic.capture_prediction(model, raw)
                else:
                    diagnostic.capture_prediction(model, raw)
                self.assertEqual(diagnostic._hook_state(model), before)
            finally:
                for handle in handles:
                    handle.remove()

    def test_unexpected_hook_registration_during_forward_is_rejected(self):
        model, raw, _ = fixture()
        leaked = []

        def mutating_hook(_module, _args, _output):
            leaked.append(model.state_residual_head.register_forward_hook(lambda *_: None))

        existing = model.register_forward_hook(mutating_hook)
        try:
            with self.assertRaisesRegex(ValueError, "hook registries"):
                diagnostic.capture_prediction(model, raw)
            self.assertEqual(len(model.visual_residual_head._forward_hooks), 0)
        finally:
            existing.remove()
            for handle in leaked:
                handle.remove()

    def test_hook_cleanup_when_output_reconstruction_or_head_call_count_wrong(self):
        for kind in ("altered_output", "double_head", "no_head", "extra_key"):
            model, raw, _ = fixture()
            original_forward = model.forward

            def changed_forward(**inputs):
                if kind == "no_head":
                    return capture_fixture(inputs)["predictions"]
                result = original_forward(**inputs)
                if kind == "altered_output":
                    result["pred_future_visual_latent"] = result["pred_future_visual_latent"] + 1
                elif kind == "double_head":
                    model.visual_residual_head(torch.zeros(2, 1, 3, 5))
                elif kind == "extra_key":
                    result["oracle"] = torch.zeros(1)
                return result

            with patch.object(model, "forward", changed_forward), self.subTest(kind=kind), self.assertRaises(ValueError):
                diagnostic.capture_prediction(model, raw)
            self.assertEqual(len(model.visual_residual_head._forward_hooks), 0)

    def test_forward_input_or_parameter_mutation_is_rejected_and_hook_removed(self):
        for kind in ("input", "parameter", "eval_mode"):
            model, raw, _ = fixture()

            def mutation_hook(_module, _args, _output):
                if kind == "input":
                    raw["history_state"].add_(1)
                elif kind == "parameter":
                    model.visual_residual_head.weight.add_(1)
                else:
                    model.state_residual_head.train()

            existing = model.register_forward_hook(mutation_hook)
            try:
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    diagnostic.capture_prediction(model, raw)
                self.assertEqual(len(model.visual_residual_head._forward_hooks), 0)
            finally:
                existing.remove()

    def test_no_rng_change_backward_optimizer_or_feature_io(self):
        model, raw, anchor = fixture()
        rng = torch.get_rng_state().clone()
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")), \
             patch("builtins.open", side_effect=AssertionError("feature/file IO forbidden")):
            clean = diagnostic.transform_inputs(raw, "observed_action", "clean", anchor)
            changed = diagnostic.transform_inputs(raw, "observed_action", "fixed_train_anchor", anchor)
            totals = diagnostic.DriftTotals()
            totals.update(diagnostic.capture_prediction(model, clean), diagnostic.capture_prediction(model, changed), clean, changed)
            self.assertEqual(totals.summary()["windows"], 2)
        self.assertTensorExact(torch.get_rng_state(), rng)

    def test_drift_exact_scalar_quantities_and_owned_serializable_summary(self):
        _, clean, _ = fixture()
        clean["history_visual_latent"].zero_()
        changed = deepcopy(clean)
        changed["history_visual_latent"].fill_(2)
        clean_capture = capture_fixture(clean, residual=1, state=0.0)
        changed_capture = capture_fixture(changed, residual=-2, state=4.0)
        totals = diagnostic.DriftTotals()
        totals.update(clean_capture, changed_capture, clean, changed)
        result = totals.summary()
        self.assertEqual((result["windows"], result["updates"]), (2, 1))
        expected = {"state_output_delta": (48, 4), "visual_output_delta": (24576, 1),
                    "visual_residual_delta": (24576, 3), "anchor_skip_delta": (24576, 2),
                    "history_input_delta": (32768, 2)}
        for key, (count, delta) in expected.items():
            self.assertEqual(result[key], {"count": count, "sum_abs": float(count * delta),
                "sum_sq": float(count * delta * delta), "max_abs": float(delta),
                "mae": float(delta), "rmse": float(delta), "exact_changed_count": count})
        json.dumps(result, allow_nan=False)
        result["state_output_delta"]["count"] = -1
        self.assertEqual(totals.summary()["state_output_delta"]["count"], 48)

    def test_cancellation_keeps_skip_and_actual_residual_drift_separate(self):
        _, clean, _ = fixture()
        clean["history_visual_latent"].zero_()
        changed = deepcopy(clean)
        changed["history_visual_latent"].fill_(1)
        totals = diagnostic.DriftTotals()
        totals.update(capture_fixture(clean), capture_fixture(changed, residual=-1), clean, changed)
        result = totals.summary()
        self.assertEqual(result["visual_output_delta"]["mae"], 0)
        self.assertEqual(result["visual_residual_delta"]["mae"], 1)
        self.assertEqual(result["anchor_skip_delta"]["mae"], 1)
        self.assertNotIn("causal", json.dumps(result))
        self.assertNotIn("ratio", json.dumps(result))

    def test_drift_float64_promotion_before_subtraction(self):
        _, clean, _ = fixture()
        clean["history_visual_latent"].fill_(3e38)
        changed = deepcopy(clean)
        changed["history_visual_latent"].fill_(-3e38)
        totals = diagnostic.DriftTotals()
        totals.update(capture_fixture(clean, state=3e38), capture_fixture(changed, state=-3e38), clean, changed)
        result = totals.summary()
        expected = 2 * float(torch.tensor(3e38, dtype=torch.float32))
        self.assertEqual(result["state_output_delta"]["mae"], expected)
        self.assertTrue(math.isfinite(result["state_output_delta"]["sum_sq"]))
        json.dumps(result, allow_nan=False)

    def test_drift_row_order_reduction_independent_of_batch_partition(self):
        model, clean, anchor = fixture(batch=3)
        changed = diagnostic.transform_inputs(clean, "observed_action", "fixed_train_anchor", anchor)
        clean_capture, changed_capture = diagnostic.capture_prediction(model, clean), diagnostic.capture_prediction(model, changed)
        together, separate = diagnostic.DriftTotals(), diagnostic.DriftTotals()
        together.update(clean_capture, changed_capture, clean, changed)
        for index in range(3):
            separate.update(capture_slice(clean_capture, index), capture_slice(changed_capture, index),
                            batch_slice(clean, index), batch_slice(changed, index))
        one, three = together.summary(), separate.summary()
        one.pop("updates")
        three.pop("updates")
        self.assertEqual(one, three)

    def test_empty_and_identical_drift_counts_and_zero_changes(self):
        totals = diagnostic.DriftTotals()
        empty = totals.summary()
        self.assertEqual(empty["windows"], 0)
        for key in diagnostic.DRIFT_KEYS:
            self.assertEqual(empty[key]["count"], 0)
            self.assertIsNone(empty[key]["mae"])
            self.assertIsNone(empty[key]["rmse"])
        _, inputs, _ = fixture()
        capture = capture_fixture(inputs)
        totals.update(capture, capture, inputs, inputs)
        totals.update(capture, capture, inputs, inputs)
        summary = totals.summary()
        self.assertEqual((summary["windows"], summary["updates"]), (4, 2))
        for key in diagnostic.DRIFT_KEYS:
            self.assertEqual(summary[key]["exact_changed_count"], 0)
            self.assertEqual(summary[key]["mae"], 0)
        self.assertEqual(summary["state_output_delta"]["count"], 96)

    def test_drift_rejects_nonvisual_mutations_without_partial_update(self):
        _, clean, _ = fixture()
        totals = diagnostic.DriftTotals()
        clean_capture = capture_fixture(clean)
        totals.update(clean_capture, clean_capture, clean, clean)
        before = totals.summary()
        for key in ("history_state", "candidate_actions", "history_visual_valid", "history_state_valid", "task_instruction"):
            changed = deepcopy(clean)
            if key == "task_instruction":
                changed[key][0] += " changed"
            elif changed[key].dtype == torch.bool:
                changed[key].flatten()[0] = False
            else:
                changed[key].flatten()[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                totals.update(clean_capture, capture_fixture(changed), clean, changed)
            self.assertEqual(totals.summary(), before)

    def test_drift_rejects_corrupt_capture_or_unknown_metadata_atomically(self):
        _, clean, _ = fixture()
        capture = capture_fixture(clean)
        totals = diagnostic.DriftTotals()
        before = totals.summary()
        for kind in ("anchor", "residual", "state_nan", "shape", "inference", "extra_capture", "extra_prediction"):
            bad = deepcopy(capture)
            if kind in ("anchor", "residual"):
                bad[kind].flatten()[0] += 1
            elif kind == "state_nan":
                bad["predictions"]["pred_state_delta"].flatten()[0] = float("nan")
            elif kind == "shape":
                bad["residual"] = bad["residual"][:1]
            elif kind == "inference":
                with torch.inference_mode():
                    bad["residual"] = bad["residual"].clone()
            elif kind == "extra_capture":
                bad["targets"] = {}
            else:
                bad["predictions"]["metadata"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                totals.update(capture, bad, clean, clean)
            self.assertEqual(totals.summary(), before)


if __name__ == "__main__":
    unittest.main()
