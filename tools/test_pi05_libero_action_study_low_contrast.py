"""Synthetic CPU sensitivity tests; fixtures are not trained/public results."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import pi05_libero_action_study_low_contrast as low
import run_pi05_libero_action_study_low_contrast as entrypoint
from test_pi05_libero_action_study_step0 import TinyDataset, FixturePredictor


class HistoryPredictor(FixturePredictor):
    """Fixture makes degraded-history dependence visible, with no learning."""
    def forward(self, **inputs):
        result = super().forward(**inputs)
        last = inputs["history_visual_latent"][:, -1]
        result["pred_future_visual_latent"] = last[:, None, None].expand(-1, 1, 3, 2, 2048).clone()
        return result


class LowContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.dataset = TinyDataset()
        for index, window in enumerate(self.dataset.windows):
            window.history = tuple(range(100 + index * 4, 104 + index * 4))
        self.lookup = {row: row - 100 for row in range(100, 116)}
        self.latents = np.broadcast_to(np.arange(1, 17, dtype=np.float32)[:, None, None], (16, 2, 2048)).copy()
        self.stats = {"state_std": [1.] * 8}
        self.raw = low.collate_window_inputs([item["inputs"] for item in self.dataset.items])

    def conditioned(self, arm, condition, raw=None):
        return low.condition_inputs(self.raw if raw is None else raw, self.dataset.windows, list(range(4)),
                                    self.lookup, self.latents, arm=arm, condition=condition)

    def refs(self, arm):
        groups = {}
        for index, window in enumerate(self.dataset.windows):
            groups.setdefault(window.episode_index, []).append(index)
        reference = {}
        for selected in groups.values():
            items = [self.dataset[index] for index in selected]
            raw = low.collate_window_inputs([item["inputs"] for item in items])
            targets = low.collate_window_targets([item["targets"] for item in items])
            inputs = low.condition_inputs(raw, self.dataset.windows, selected, self.lookup, self.latents,
                                         arm=arm, condition="clean")
            with torch.inference_mode():
                predictions = low.persistence_predictions(inputs) if arm == "persistence" else HistoryPredictor(action_echo=True)(**inputs)
            if arm == "persistence":
                record = {"source_input_sha256": low.base.fingerprints(raw),
                          "common_target_sha256": low.base.fingerprints(targets),
                          "predictions_sha256": {"persistence": low.base.fingerprints(predictions)}}
            else:
                record = {"input_sha256": low.base.fingerprints(inputs), "target_sha256": low.base.fingerprints(targets),
                          "prediction_sha256": low.base.fingerprints(predictions)}
            reference[tuple(selected)] = record
        return reference

    def evaluate(self, condition, arm=None, model=None, references=None, trace=None):
        arm = arm or low.base.ARMS[0]
        model = None if arm == "persistence" else (model or HistoryPredictor(action_echo=True))
        return low.evaluate_condition(model, self.dataset, self.stats, arm=arm, seed=low.base.SEEDS[0],
            condition=condition, row_lookup=self.lookup, latents=self.latents,
            references=self.refs(arm) if references is None else references, trace=trace)

    def row_fixture(self):
        offset, mapping = 1000, {}
        for eid, count in low.VALIDATION_COUNTS.items():
            mapping[eid] = np.arange(offset, offset + count, dtype=np.int64)
            offset += count
        mapping[1633] = np.arange(10, dtype=np.int64)  # explicitly excluded train rows
        rows = np.concatenate([mapping[eid] for eid in low.VALIDATION_COUNTS])
        return SimpleNamespace(indices=mapping), rows, np.zeros((524, 2, 2048), np.float32)

    def test_condition_replaces_only_history_visual_and_preserves_raw(self):
        before = low.base.fingerprints(self.raw)
        clean = self.conditioned(low.base.ARMS[0], "clean")
        noisy = self.conditioned(low.base.ARMS[0], "low_contrast")
        self.assertEqual(low.base.fingerprints(clean), before)
        self.assertEqual(low.base.fingerprints(self.raw), before)
        for key in before:
            if key != "history_visual_latent":
                self.assertEqual(low.base.fingerprints(noisy)[key], before[key])
        expected = self.latents.reshape(4, 4, 2, 2048)
        np.testing.assert_array_equal(noisy["history_visual_latent"].numpy(), expected)
        self.assertNotEqual(low.base.fingerprints(noisy)["history_visual_latent"], before["history_visual_latent"])

    def test_zero_arm_candidate_actions_zero_under_both_conditions(self):
        for condition in low.CONDITIONS:
            result = self.conditioned(low.base.ARMS[1], condition)
            self.assertEqual(int(torch.count_nonzero(result["candidate_actions"])), 0)
            self.assertTrue(bool((self.raw["candidate_actions"] != 0).all()))

    def test_ordinary_owned_tensors_and_masks_remain_intact(self):
        self.raw["history_visual_valid"][0, 0, 0] = False
        self.raw["history_state_valid"][1, 1, 1] = False
        for condition in low.CONDITIONS:
            result = self.conditioned(low.base.ARMS[0], condition)
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    self.assertFalse(torch.is_inference(value) or value.requires_grad)
                    self.assertNotEqual(value.data_ptr(), self.raw[key].data_ptr())
            self.assertTrue(torch.equal(result["history_visual_valid"], self.raw["history_visual_valid"]))
            self.assertTrue(torch.equal(result["history_state_valid"], self.raw["history_state_valid"]))
            self.assertTrue(bool((result["history_visual_latent"][0, 0, 0] == 0).all()))

    def test_targets_are_not_a_condition_input_and_cannot_be_added_to_raw(self):
        raw = dict(self.raw, targets={"fixture": True})
        with self.assertRaises(ValueError):
            self.conditioned(low.base.ARMS[0], "clean", raw)
        with self.assertRaises(TypeError):
            low.condition_inputs(self.raw, self.dataset.windows, [0, 1, 2, 3], self.lookup, self.latents,
                                 arm=low.base.ARMS[0], condition="clean", targets={})

    def test_missing_history_row_invalid_condition_arm_and_nonfinite_fail(self):
        for arm, condition in (("unknown", "clean"), (low.base.ARMS[0], "unknown")):
            with self.subTest(arm=arm, condition=condition), self.assertRaises(ValueError):
                self.conditioned(arm, condition)
        self.lookup.pop(100)
        with self.assertRaisesRegex(ValueError, "missing historical"):
            self.conditioned(low.base.ARMS[0], "low_contrast")
        self.lookup[100] = 0
        self.latents[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "shape/finiteness"):
            self.conditioned(low.base.ARMS[0], "low_contrast")

    def test_feature_row_map_exact_validation_order(self):
        pack, rows, latents = self.row_fixture()
        mapping = low.validate_feature_rows(pack, rows, latents)
        self.assertEqual(mapping, {int(row): index for index, row in enumerate(rows)})
        self.assertEqual(len(mapping), 524)
        self.assertNotIn(0, mapping)

    def test_feature_map_rejects_train_duplicate_reorder_missing_and_dtype(self):
        pack, rows, latents = self.row_fixture()
        for mode in ("train", "duplicate", "reorder", "missing", "dtype"):
            bad = rows.copy()
            if mode == "train":
                bad[0] = 0
            elif mode == "duplicate":
                bad[1] = bad[0]
            elif mode == "reorder":
                bad[[0, 1]] = bad[[1, 0]]
            elif mode == "missing":
                bad = bad[:-1]
            else:
                bad = bad.astype(np.float64)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                low.validate_feature_rows(pack, bad, latents)

    def test_feature_latents_reject_wrong_shape_dtype_and_nonfinite(self):
        pack, rows, latents = self.row_fixture()
        for mode in ("shape", "dtype", "nonfinite"):
            bad = latents[:-1] if mode == "shape" else latents.astype(np.float64) if mode == "dtype" else latents.copy()
            if mode == "nonfinite":
                bad[0, 0, 0] = np.inf
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                low.validate_feature_rows(pack, rows, bad)

    def test_clean_prediction_replay_exact_and_no_optimizer_or_backward(self):
        for arm in (*low.base.ARMS, "persistence"):
            with patch("torch.optim.AdamW", side_effect=AssertionError("no optimizer")), patch("torch.Tensor.backward", side_effect=AssertionError("no backward")):
                result = self.evaluate("clean", arm)
            self.assertEqual(result["micro"]["window_count"], 4)
            self.assertEqual(set(result["per_episode"]), {"11", "22"})

    def test_clean_reference_prediction_input_and_target_drift_fail(self):
        original = self.refs(low.base.ARMS[0])
        for key in ("prediction_sha256", "input_sha256", "target_sha256"):
            refs = deepcopy(original)
            first = next(iter(refs.values()))
            first[key][next(iter(first[key]))] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate("clean", references=refs)

    def test_persistence_clean_reference_drift_also_fails_closed(self):
        for mode in ("input", "target", "prediction"):
            refs = self.refs("persistence")
            first = next(iter(refs.values()))
            value = (first["source_input_sha256"] if mode == "input" else first["common_target_sha256"]
                     if mode == "target" else first["predictions_sha256"]["persistence"])
            value[next(iter(value))] = "0" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.evaluate("clean", "persistence", references=refs)

    def test_low_contrast_uses_clean_future_targets_and_zero_actions(self):
        targets = deepcopy([item["targets"] for item in self.dataset.items])
        for arm in low.base.ARMS:
            refs = self.refs(arm)
            model = HistoryPredictor(action_echo=True)
            trace = io.StringIO()
            noisy = self.evaluate("low_contrast", arm, model, refs, trace)
            self.assertGreater(noisy["micro"]["visual"]["aggregate"]["mae"], 0.)
            for row in map(json.loads, trace.getvalue().splitlines()):
                self.assertEqual(row["target_sha256"], refs[tuple(row["window_indices"])]["target_sha256"])
            if arm == low.base.ARMS[1]:
                self.assertTrue(all(bool((value == 0).all()) for value in model.seen))
        for original, item in zip(targets, self.dataset.items):
            for key in original:
                np.testing.assert_array_equal(original[key], item["targets"][key])

    def test_degraded_nonvisual_reference_drift_is_not_ignored(self):
        refs = self.refs(low.base.ARMS[0])
        next(iter(refs.values()))["input_sha256"]["history_state"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "nonvisual input"):
            self.evaluate("low_contrast", references=refs)

    def test_frozen_model_modes_gradient_and_wrong_persistence_use_rejected(self):
        for mode in ("train", "requires_grad", "gradient"):
            model = HistoryPredictor(action_echo=True)
            if mode == "train":
                model.train()
            elif mode == "requires_grad":
                model.marker.requires_grad_(True)
            else:
                model.marker.grad = torch.ones_like(model.marker)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "frozen eval"):
                self.evaluate("clean", model=model)
        with self.assertRaisesRegex(ValueError, "model/reference"):
            low.evaluate_condition(HistoryPredictor(), self.dataset, self.stats, arm="persistence", seed=1,
                                   condition="clean", row_lookup=self.lookup, latents=self.latents, references={})

    def test_missing_window_coverage_and_parameter_mutation_fail(self):
        self.dataset.windows.pop()
        with self.assertRaisesRegex(ValueError, "missing evaluation windows"):
            self.evaluate("clean")
        self.dataset = TinyDataset()
        for index, window in enumerate(self.dataset.windows):
            window.history = tuple(range(100 + index * 4, 104 + index * 4))
        model = HistoryPredictor(action_echo=True, mutate="parameter")
        with self.assertRaisesRegex(ValueError, "changed frozen model"):
            self.evaluate("clean", model=model)

    @staticmethod
    def summary_row(value):
        def metric(v):
            return {"state_normalized": {"aggregate": {"mae": v, "rmse": 2*v}},
                    "visual": {"aggregate": {"mae": v/10, "rmse": v/5}},
                    "masked_objective": {"total": v/20}}
        return {"episode_macro": low.base.scalar_metrics(metric(value)),
                "per_episode": {"11": metric(value-.1 if value else 0.), "22": metric(value+.1 if value else 0.)}}

    def summary_fixture(self):
        runs = {}
        for index, seed in enumerate(low.base.SEEDS):
            runs[str(seed)] = {
                low.base.ARMS[0]: {"clean": self.summary_row(index+1.), "low_contrast": self.summary_row(index+2.)},
                low.base.ARMS[1]: {"clean": self.summary_row(index+3.), "low_contrast": self.summary_row(2*index+4.)}}
        return runs, {"clean": self.summary_row(8.), "low_contrast": self.summary_row(9.)}

    def test_summary_three_seed_absolute_relative_and_descriptive_std(self):
        runs, persistence = self.summary_fixture()
        report = low.summarize(runs, persistence)
        paired = report["paired_state_macro"]["low_contrast"]
        self.assertEqual(list(paired["by_seed"].values()), [-2., -3., -4.])
        self.assertEqual(paired["mean"], -3.)
        self.assertAlmostEqual(paired["descriptive_population_std"], math.sqrt(2/3))
        changed = report["comparisons"][str(low.base.SEEDS[0])]["low_contrast_minus_clean"][low.base.ARMS[0]]["normalized_state_mae"]
        self.assertEqual(changed, {"absolute": 1., "relative_percent": 100.})
        mean = report["mean_low_contrast_minus_clean"][low.base.ARMS[0]]["normalized_state_mae"]
        self.assertEqual(mean, {"absolute": 1., "relative_percent": 50.})
        self.assertEqual(report["seed_mean_episode_macro"]["low_contrast"]["persistence"], persistence["low_contrast"]["episode_macro"])
        self.assertEqual(report["decision"], "degraded_observed_action_predictive_association_retained")
        self.assertFalse(report["robustness_certified"])

    def test_summary_rejects_missing_or_extra_seed_arm_and_condition(self):
        for mode in ("seed", "extra_seed", "arm", "condition", "persistence_condition"):
            runs, persistence = self.summary_fixture()
            first = runs[str(low.base.SEEDS[0])]
            if mode == "seed":
                runs.pop(str(low.base.SEEDS[0]))
            elif mode == "extra_seed":
                runs["extra"] = deepcopy(first)
            elif mode == "arm":
                first.pop(low.base.ARMS[0])
            elif mode == "condition":
                first[low.base.ARMS[0]].pop("clean")
            else:
                persistence.pop("clean")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                low.summarize(runs, persistence)

    def test_retained_criterion_requires_all_seeds_and_beating_persistence(self):
        runs, persistence = self.summary_fixture()
        seed = str(low.base.SEEDS[1])
        runs[seed][low.base.ARMS[0]]["low_contrast"] = deepcopy(runs[seed][low.base.ARMS[1]]["low_contrast"])
        self.assertEqual(low.summarize(runs, persistence)["decision"], "degraded_action_association_mixed_or_not_retained")
        runs, persistence = self.summary_fixture()
        persistence["low_contrast"] = self.summary_row(1.)
        report = low.summarize(runs, persistence)
        self.assertEqual(report["decision"], "degraded_action_association_mixed_or_not_retained")
        self.assertFalse(report["robustness_certified"])

    def test_persistence_zero_reference_keeps_relative_change_null(self):
        runs, persistence = self.summary_fixture()
        persistence["clean"] = self.summary_row(0.)
        report = low.summarize(runs, persistence)
        changed = report["mean_low_contrast_minus_clean"]["persistence"]["normalized_state_mae"]
        self.assertEqual(changed, {"absolute": 9., "relative_percent": None})

    def test_entrypoint_preflight_execute_and_missing_features_execute_rejected(self):
        for args in (("--stage", "preflight", "--execute"), ("--stage", "features")):
            with self.subTest(args=args), patch.dict(os.environ), patch.object(entrypoint, "validate") as preflight, patch.object(entrypoint, "feature_stage") as features, patch.object(entrypoint, "evaluate_stage") as evaluation, self.assertRaises(ValueError):
                entrypoint.main([*args, "--plan-sha256", "0"*64])
            preflight.assert_not_called()
            features.assert_not_called()
            evaluation.assert_not_called()

    def test_entrypoint_feature_report_sha_required_only_for_evaluation(self):
        for args in (("--stage", "evaluate", "--execute"),
                     ("--stage", "features", "--execute", "--feature-report-sha256", "1"*64),
                     ("--stage", "preflight", "--feature-report-sha256", "1"*64)):
            with self.subTest(args=args), patch.dict(os.environ), patch.object(entrypoint, "validate") as preflight, patch.object(entrypoint, "feature_stage") as features, patch.object(entrypoint, "evaluate_stage") as evaluation, self.assertRaises(ValueError):
                entrypoint.main([*args, "--plan-sha256", "0"*64])
            preflight.assert_not_called()
            features.assert_not_called()
            evaluation.assert_not_called()

    def test_entrypoint_guarded_preflight_never_invokes_feature_or_eval_stage(self):
        result = {"status": "fixture_preflight_only", "input_sha256": {"fixture": "1"*64}}
        output = io.StringIO()
        with patch.dict(os.environ), patch.object(entrypoint, "validate", return_value=result) as preflight, patch.object(entrypoint, "feature_stage") as features, patch.object(entrypoint, "evaluate_stage") as evaluation, redirect_stdout(output):
            entrypoint.main(["--stage", "preflight", "--plan-sha256", "0"*64])
        preflight.assert_called_once_with(entrypoint.ROOT, "0"*64)
        features.assert_not_called()
        evaluation.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
