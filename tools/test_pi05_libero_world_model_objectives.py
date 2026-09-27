"""Synthetic CPU arithmetic/mask/aggregation tests, never backward or optimization."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math
import unittest

import numpy as np
import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model_objectives import (
        NativeWorldModelLossConfig, NativeWorldModelMetrics, collate_window_targets,
        native_world_model_loss, objective_contract,
    )
else:
    from .pi05_libero_world_model_objectives import (
        NativeWorldModelLossConfig, NativeWorldModelMetrics, collate_window_targets,
        native_world_model_loss, objective_contract,
    )


def fixture(b=2, h=3, d=4):
    generator = torch.Generator().manual_seed(2191)
    targets = {"future_visual_latent": torch.randn(b, h, 2, d, generator=generator),
               "future_visual_valid": torch.ones(b, h, 2, dtype=torch.bool),
               "state_delta": torch.randn(b, h, 8, generator=generator),
               "state_target_valid": torch.ones(b, h, 8, dtype=torch.bool)}
    predictions = {"pred_future_visual_latent": targets["future_visual_latent"][:, None].clone() + .5,
                   "pred_state_delta": targets["state_delta"][:, None].clone() + 2}
    return predictions, targets


def numpy_targets(targets):
    return [{key: value[index].numpy().copy() for key, value in targets.items()}
            for index in range(targets["state_delta"].shape[0])]


def subbatch(predictions, targets, start, stop):
    return ({key: value[start:stop] for key, value in predictions.items()},
            {key: value[start:stop] for key, value in targets.items()})


def direct_smooth(errors, beta=1.):
    return np.where(np.abs(errors) < beta, .5 * errors ** 2 / beta, np.abs(errors) - .5 * beta)


class LossConfigTests(unittest.TestCase):
    def test_defaults_frozen_and_contract(self):
        config = NativeWorldModelLossConfig()
        self.assertEqual((config.visual_weight, config.state_weight, config.beta), (1., .25, 1.))
        with self.assertRaises(FrozenInstanceError):
            config.beta = 2
        json.dumps(objective_contract(), allow_nan=False)
        self.assertFalse(objective_contract()["optimizer_or_backward_in_module"])

    def test_bad_weights_rejected(self):
        for kwargs in ({"visual_weight": -1}, {"state_weight": -1}, {"visual_weight": 0, "state_weight": 0},
                       {"visual_weight": float("inf")}, {"state_weight": float("nan")},
                       {"visual_weight": True}, {"state_weight": "1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NativeWorldModelLossConfig(**kwargs)

    def test_beta_nonpositive_or_nonfinite_rejected(self):
        for beta in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                NativeWorldModelLossConfig(beta=beta)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_computed_smooth_l1_counts_weights_and_sums(self):
        p, t = fixture(b=1, h=1, d=2)
        t["future_visual_latent"].zero_()
        t["state_delta"].zero_()
        p["pred_future_visual_latent"][:] = torch.tensor([0., .5, 1., 2.]).reshape(1, 1, 1, 2, 2)
        p["pred_state_delta"][:] = torch.tensor([0., 1., 2., 0., 0., 0., 0., 0.]).reshape(1, 1, 1, 8)
        loss, details = native_world_model_loss(p, t)
        self.assertEqual(float(details["visual_sum"]), 2.125)
        self.assertEqual(float(details["state_sum"]), 2.)
        self.assertEqual(details["visual_count"], 4)
        self.assertEqual(details["state_count"], 8)
        self.assertEqual(float(details["visual_loss"]), .53125)
        self.assertEqual(float(details["state_loss"]), .25)
        self.assertEqual(float(loss), .59375)
        self.assertEqual(loss.dtype, torch.float32)
        for key in ("visual_sum", "state_sum"):
            self.assertEqual(details[key].dtype, torch.float64)
            self.assertFalse(details[key].requires_grad)

    def test_custom_beta_and_weights_have_exact_formula(self):
        p, t = fixture(b=1, h=1, d=2)
        t["future_visual_latent"].zero_()
        t["state_delta"].zero_()
        p["pred_future_visual_latent"].fill_(1.)
        p["pred_state_delta"].fill_(3.)
        loss, details = native_world_model_loss(p, t, NativeWorldModelLossConfig(visual_weight=.5, state_weight=2, beta=2))
        self.assertEqual(float(details["visual_loss"]), .25)
        self.assertEqual(float(details["state_loss"]), 2.)
        self.assertEqual(float(loss), 4.125)

    def test_partial_masks_normalize_by_valid_scalars_not_full_shape(self):
        p, t = fixture(b=1, h=2, d=3)
        t["future_visual_valid"].fill_(False)
        t["future_visual_valid"][0, 1, 0] = True
        t["state_target_valid"].fill_(False)
        t["state_target_valid"][0, 0, :2] = True
        _, details = native_world_model_loss(p, t)
        self.assertEqual(details["visual_count"], 3)
        self.assertEqual(details["state_count"], 2)
        self.assertAlmostEqual(float(details["visual_loss"]), .125, places=7)
        self.assertEqual(float(details["state_loss"]), 1.5)

    def test_invalid_prediction_and_target_nan_inf_removed_before_arithmetic(self):
        p, t = fixture()
        t["future_visual_valid"][:, 2, 0] = False
        t["state_target_valid"][:, 1, 5:] = False
        original = native_world_model_loss(p, t)
        for poison in (float("nan"), float("inf"), -float("inf")):
            changed_p, changed_t = deepcopy(p), deepcopy(t)
            changed_p["pred_future_visual_latent"][:, 0][~t["future_visual_valid"]] = poison
            changed_t["future_visual_latent"][~t["future_visual_valid"]] = -poison
            changed_p["pred_state_delta"][:, 0][~t["state_target_valid"]] = poison
            changed_t["state_delta"][~t["state_target_valid"]] = -poison
            loss, details = native_world_model_loss(changed_p, changed_t)
            torch.testing.assert_close(loss, original[0], atol=0, rtol=0)
            for key in ("visual_sum", "state_sum", "visual_loss", "state_loss"):
                torch.testing.assert_close(details[key], original[1][key], atol=0, rtol=0)

    def test_one_unsupported_branch_is_zero(self):
        for branch in ("visual", "state"):
            p, t = fixture()
            key = "future_visual_valid" if branch == "visual" else "state_target_valid"
            t[key].fill_(False)
            loss, details = native_world_model_loss(p, t)
            self.assertEqual(details[f"{branch}_count"], 0)
            self.assertEqual(float(details[f"{branch}_loss"]), 0)
            self.assertTrue(torch.isfinite(loss))

    def test_no_positively_weighted_support_fails_closed(self):
        p, t = fixture()
        t["future_visual_valid"].fill_(False)
        t["state_target_valid"].fill_(False)
        with self.assertRaisesRegex(ValueError, "positively weighted"):
            native_world_model_loss(p, t)
        t["state_target_valid"].fill_(True)
        with self.assertRaisesRegex(ValueError, "positively weighted"):
            native_world_model_loss(p, t, NativeWorldModelLossConfig(1, 0))
        t["future_visual_valid"].fill_(True)
        t["state_target_valid"].fill_(False)
        with self.assertRaisesRegex(ValueError, "positively weighted"):
            native_world_model_loss(p, t, NativeWorldModelLossConfig(0, 1))

    def test_observed_nonfinite_prediction_or_target_rejected(self):
        for section, key in (("p", "pred_future_visual_latent"), ("p", "pred_state_delta"),
                             ("t", "future_visual_latent"), ("t", "state_delta")):
            p, t = fixture()
            (p if section == "p" else t)[key].flatten()[0] = float("inf")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "finite"):
                native_world_model_loss(p, t)

    def test_k2_cannot_implicitly_average_or_choose_candidates(self):
        p, t = fixture()
        p = {key: value.repeat_interleave(2, dim=1) for key, value in p.items()}
        with self.assertRaisesRegex(ValueError, "exactly K=1"):
            native_world_model_loss(p, t)

    def test_shape_mismatch_rejected_without_broadcasting(self):
        p, t = fixture()
        t["state_delta"] = t["state_delta"][:, :1]
        with self.assertRaisesRegex(ValueError, "no implicit broadcasting"):
            native_world_model_loss(p, t)
        p, t = fixture()
        p["pred_state_delta"] = torch.zeros(2, 1, 3, 9)
        with self.assertRaisesRegex(ValueError, "no implicit broadcasting"):
            native_world_model_loss(p, t)

    def test_dtype_or_mask_numeric_rejected(self):
        for section, key in (("p", "pred_state_delta"), ("t", "state_delta"), ("t", "future_visual_valid")):
            p, t = fixture()
            mapping = p if section == "p" else t
            mapping[key] = mapping[key].double()
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_world_model_loss(p, t)

    def test_extra_prediction_target_or_oracle_fields_rejected(self):
        for section, key in (("p", "score"), ("t", "metadata"), ("t", "contact_truth")):
            p, t = fixture()
            (p if section == "p" else t)[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_world_model_loss(p, t)

    def test_target_gradients_rejected_but_future_prediction_autograd_retained(self):
        p, t = fixture()
        t["state_delta"].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "targets must be detached"):
            native_world_model_loss(p, t)
        t["state_delta"] = t["state_delta"].detach()
        p["pred_state_delta"].requires_grad_(True)
        loss, details = native_world_model_loss(p, t)
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(loss.grad_fn)
        self.assertFalse(details["state_sum"].requires_grad)
        self.assertIsNone(p["pred_state_delta"].grad)
        # Intentionally no backward call or optimizer in this test suite.

    def test_elementwise_and_weighted_positive_overflow_rejected(self):
        p, t = fixture()
        p["pred_state_delta"].fill_(3e38)
        t["state_delta"].fill_(-3e38)
        with self.assertRaisesRegex(ValueError, "overflow"):
            native_world_model_loss(p, t)
        p, t = fixture()
        with self.assertRaisesRegex(ValueError, "overflow"):
            native_world_model_loss(p, t, NativeWorldModelLossConfig(visual_weight=1e308))

    def test_float64_sum_avoids_artificial_reduction_overflow(self):
        p, t = fixture(b=2, h=3, d=4)
        p["pred_future_visual_latent"].fill_(1e38)
        t["future_visual_latent"].zero_()
        loss, details = native_world_model_loss(p, t, NativeWorldModelLossConfig(1, 0))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(details["visual_sum"]), float(np.finfo(np.float32).max))

    def test_uneven_batch_loss_sums_reconstruct_global_not_mean_of_batch_means(self):
        p, t = fixture(b=5)
        for row in range(5):
            p["pred_future_visual_latent"][row] = t["future_visual_latent"][row] + row
        t["future_visual_valid"][4].fill_(False)
        t["future_visual_valid"][4, 0, 0] = True
        _, full = native_world_model_loss(p, t)
        pieces = [native_world_model_loss(*subbatch(p, t, start, stop))[1] for start, stop in ((0, 2), (2, 4), (4, 5))]
        for branch in ("visual", "state"):
            self.assertEqual(sum(part[f"{branch}_count"] for part in pieces), full[f"{branch}_count"])
            self.assertEqual(sum(float(part[f"{branch}_sum"]) for part in pieces), float(full[f"{branch}_sum"]))
        naive = sum(float(part["visual_loss"]) for part in pieces) / len(pieces)
        self.assertGreater(abs(naive - float(full["visual_loss"])), .1)

    def test_no_mutation_and_inference_has_no_grad(self):
        p, t = fixture()
        snapshots = deepcopy((p, t))
        with torch.inference_mode():
            loss, details = native_world_model_loss(p, t)
        self.assertFalse(loss.requires_grad)
        for original, saved in zip((p, t), snapshots):
            for key in original:
                torch.testing.assert_close(original[key], saved[key], atol=0, rtol=0)


class MetricsTests(unittest.TestCase):
    def test_zero_errors_and_counts(self):
        p, t = fixture(b=2, h=3, d=4)
        p["pred_future_visual_latent"] = t["future_visual_latent"][:, None].clone()
        p["pred_state_delta"] = t["state_delta"][:, None].clone()
        metrics = NativeWorldModelMetrics([1.] * 8)
        metrics.update(p, t)
        summary = metrics.summary()
        self.assertEqual(summary["visual"]["aggregate"]["count"], 48)
        self.assertEqual(summary["state_normalized"]["aggregate"]["count"], 48)
        for section in ("visual", "state_normalized"):
            self.assertEqual(summary[section]["aggregate"]["mae"], 0)
            self.assertEqual(summary[section]["aggregate"]["rmse"], 0)
        self.assertNotIn("aggregate", summary["state_native"])

    def test_hand_calculated_per_horizon_view_coordinate_and_native_scale(self):
        p, t = fixture(b=1, h=2, d=2)
        t["future_visual_latent"].zero_()
        p["pred_future_visual_latent"][:] = torch.tensor([1., 1., 2., 2., 3., 3., 4., 4.]).reshape(1, 1, 2, 2, 2)
        t["state_delta"].zero_()
        p["pred_state_delta"][:] = torch.arange(1, 9).view(1, 1, 1, 8)
        t["future_visual_valid"][0, 1, 1] = False
        t["state_target_valid"][0, 1, 7] = False
        std = np.arange(1, 9, dtype=np.float64)
        metrics = NativeWorldModelMetrics(std)
        metrics.update(p, t)
        summary = metrics.summary()
        self.assertEqual(summary["visual"]["aggregate"], {"count": 6, "absolute_error_sum": 12.,
                         "squared_error_sum": 28., "mae": 2., "rmse": math.sqrt(28 / 6)})
        self.assertEqual([s["count"] for s in summary["visual"]["per_horizon"]], [4, 2])
        self.assertEqual([s["mae"] for s in summary["visual"]["per_view"]], [2., 2.])
        self.assertIsNone(summary["visual"]["per_horizon_view"][1][1]["mae"])
        self.assertEqual([s["count"] for s in summary["state_normalized"]["per_coordinate"]], [2] * 7 + [1])
        self.assertEqual([s["mae"] for s in summary["state_native"]["per_coordinate"]], list((np.arange(1, 9) ** 2).astype(float)))
        self.assertEqual(summary["state_native"]["groups"]["position"]["mae"], (1 + 4 + 9) / 3)
        self.assertEqual(summary["state_native"]["groups"]["axis_angle_coordinates"]["count"], 6)
        self.assertEqual(summary["state_native"]["groups"]["gripper"]["count"], 3)
        self.assertEqual(summary["state_native"]["per_horizon_coordinate"][0][7]["mae"], 64)
        self.assertIsNone(summary["state_native"]["per_horizon_coordinate"][1][7]["mae"])

    def test_zero_weight_branch_observed_nonfinite_remains_rejected(self):
        p, t = fixture()
        p["pred_state_delta"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "valid state prediction"):
            native_world_model_loss(p, t, NativeWorldModelLossConfig(visual_weight=1, state_weight=0))

    def test_uneven_partition_and_partial_support_identical_ordered_sums(self):
        p, t = fixture(b=7, h=3, d=5)
        generator = torch.Generator().manual_seed(919)
        p["pred_future_visual_latent"] += torch.randn(p["pred_future_visual_latent"].shape, generator=generator)
        p["pred_state_delta"] += torch.randn(p["pred_state_delta"].shape, generator=generator)
        t["future_visual_valid"][::2, 0, 0] = False
        t["state_target_valid"][1::2, 2, :5] = False
        single, many = NativeWorldModelMetrics([1.] * 8), NativeWorldModelMetrics([1.] * 8)
        single.update(p, t)
        for start, stop in ((0, 3), (3, 6), (6, 7)):
            many.update(*subbatch(p, t, start, stop))
        first, second = single.summary(), many.summary()
        self.assertEqual((first.pop("update_count"), second.pop("update_count")), (1, 3))
        self.assertEqual(first, second)

    def test_no_support_metrics_are_none_not_zero(self):
        metrics = NativeWorldModelMetrics([1.] * 8)
        summary = metrics.summary()
        self.assertEqual(summary["window_count"], 0)
        self.assertIsNone(summary["visual"]["aggregate"]["mae"])
        p, t = fixture()
        t["future_visual_valid"].fill_(False)
        t["state_target_valid"].fill_(False)
        p["pred_future_visual_latent"].fill_(float("nan"))
        t["future_visual_latent"].fill_(float("inf"))
        p["pred_state_delta"].fill_(float("inf"))
        t["state_delta"].fill_(float("nan"))
        metrics.update(p, t)
        summary = metrics.summary()
        for section in ("visual", "state_normalized"):
            self.assertEqual(summary[section]["aggregate"]["count"], 0)
            self.assertIsNone(summary[section]["aggregate"]["mae"])
            self.assertIsNone(summary[section]["aggregate"]["rmse"])
        for group in summary["state_native"]["groups"].values():
            self.assertIsNone(group["mae"])
        json.dumps(summary, allow_nan=False)

    def test_state_std_is_owned_positive_and_scaling_only(self):
        std = [2.] * 8
        metrics = NativeWorldModelMetrics(std)
        std[0] = 999
        p, t = fixture()
        metrics.update(p, t)
        summary = metrics.summary()
        self.assertEqual(summary["state_std"], [2.] * 8)
        for normalized, native in zip(summary["state_normalized"]["per_coordinate"], summary["state_native"]["per_coordinate"]):
            self.assertEqual(native["mae"], normalized["mae"] * 2)
            self.assertEqual(native["rmse"], normalized["rmse"] * 2)
        summary["state_std"][0] = 4
        self.assertEqual(metrics.summary()["state_std"][0], 2)

    def test_invalid_std_values_or_shapes_rejected(self):
        for std in ([1.] * 7, [0.] * 8, [-1.] * 8, [float("inf")] * 8, [float("nan")] * 8,
                    [True] * 8, np.ones((1, 8)), "ones", np.ones(8, dtype=bool)):
            with self.subTest(std=str(std)), self.assertRaises(ValueError):
                NativeWorldModelMetrics(std)

    def test_update_layout_change_rejected_without_accumulator_mutation(self):
        metrics = NativeWorldModelMetrics([1.] * 8)
        metrics.update(*fixture())
        before = metrics.summary()
        for h, d in ((2, 4), (3, 5)):
            with self.assertRaisesRegex(ValueError, "cannot change"):
                metrics.update(*fixture(h=h, d=d))
            self.assertEqual(before, metrics.summary())

    def test_metric_native_overflow_rejected_without_partial_update(self):
        metrics = NativeWorldModelMetrics([1e308] * 8)
        p, t = fixture()
        before = metrics.summary()
        with self.assertRaisesRegex(ValueError, "overflow"):
            metrics.update(p, t)
        self.assertEqual(before, metrics.summary())

    def test_metric_uses_detached_float64_subtraction_no_target_gradients(self):
        p, t = fixture()
        p["pred_state_delta"].requires_grad_(True)
        snapshots = {key: value.detach().clone() for key, value in p.items()}
        metrics = NativeWorldModelMetrics([1.] * 8)
        metrics.update(p, t)
        self.assertIsNone(p["pred_state_delta"].grad)
        for key in p:
            torch.testing.assert_close(p[key], snapshots[key], atol=0, rtol=0)
        t["state_delta"].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "targets must be detached"):
            metrics.update(p, t)

    def test_metric_reuses_strict_candidate_shape_and_target_allowlist(self):
        p, t = fixture()
        p["pred_future_visual_latent"] = p["pred_future_visual_latent"].repeat_interleave(2, dim=1)
        with self.assertRaisesRegex(ValueError, "exactly K=1"):
            NativeWorldModelMetrics([1.] * 8).update(p, t)
        p, t = fixture()
        t["oracle"] = 1
        with self.assertRaisesRegex(ValueError, "four native"):
            NativeWorldModelMetrics([1.] * 8).update(p, t)


class TargetCollationTests(unittest.TestCase):
    def setUp(self):
        _, self.targets = fixture()
        self.items = numpy_targets(self.targets)

    def test_exact_keys_storage_copy_float32_bool(self):
        original = deepcopy(self.items)
        self.items[0]["state_delta"] = self.items[0]["state_delta"].astype(np.float64)
        result = collate_window_targets(self.items)
        for key in self.targets:
            torch.testing.assert_close(result[key], self.targets[key], atol=0, rtol=0)
            self.assertFalse(result[key].requires_grad)
        result["state_delta"].zero_()
        result["state_target_valid"].fill_(False)
        for now, before in zip(self.items, original):
            for key in now:
                self.assertTrue(np.array_equal(now[key], before[key]))

    def test_unknown_metadata_or_whole_item_rejected(self):
        for key in ("metadata", "contact_truth", "inputs"):
            items = deepcopy(self.items)
            items[0][key] = {}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "four native"):
                collate_window_targets(items)
        with self.assertRaises(ValueError):
            collate_window_targets([{"targets": self.items[0]}])

    def test_nonbool_mask_missing_keys_and_empty_list_rejected(self):
        items = deepcopy(self.items)
        items[0]["future_visual_valid"] = items[0]["future_visual_valid"].astype(float)
        with self.assertRaisesRegex(ValueError, "boolean NumPy"):
            collate_window_targets(items)
        items = deepcopy(self.items)
        del items[0]["state_delta"]
        with self.assertRaises(ValueError):
            collate_window_targets(items)
        with self.assertRaises(ValueError):
            collate_window_targets([])

    def test_shape_mismatch_and_state9_rejected(self):
        items = deepcopy(self.items)
        items[0]["state_delta"] = items[0]["state_delta"][:1]
        with self.assertRaisesRegex(ValueError, "agree across"):
            collate_window_targets(items)
        for item in self.items:
            item["state_delta"] = np.zeros((3, 9), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "native targets require"):
            collate_window_targets(self.items)

    def test_masked_nonfinite_allowed_valid_nonfinite_or_cast_overflow_rejected(self):
        self.items[0]["future_visual_valid"][0, 0] = False
        self.items[0]["future_visual_latent"][0, 0] = np.nan
        collate_window_targets(self.items)
        self.items[0]["future_visual_valid"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "finite float32"):
            collate_window_targets(self.items)
        self.items = numpy_targets(self.targets)
        self.items[0]["state_delta"] = np.full((3, 8), 1e200, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "finite float32"):
            collate_window_targets(self.items)


if __name__ == "__main__":
    unittest.main()
