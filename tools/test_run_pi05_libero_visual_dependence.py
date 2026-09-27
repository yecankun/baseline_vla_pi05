"""Synthetic frozen visual-dependence runner tests; never training/extraction."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_pi05_libero_visual_dependence as runner
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig
from test_pi05_libero_action_study_step0 import TinyDataset, REGISTRY


class VisualDependenceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = TinyDataset()
        for index, item in enumerate(self.dataset.items):
            for frame in range(4):
                item["inputs"]["history_visual_latent"][frame].fill(index + frame + 1.)
        self.stats = {"state_std": [1.] * 8}
        self.anchor = torch.full((2, 2048), 2.5)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(72)
            self.model = LiberoWorldModel(LiberoWorldModelConfig(hidden_dim=4), REGISTRY).eval().requires_grad_(False)

    def pack_fixture(self):
        arrays = {"episode_index": np.asarray([1530, 1312, 1312, 1633], np.int64),
                  "frame_index": np.asarray([0, 0, 1, 0], np.int64), "task_id": np.full(4, 9, np.int64),
                  "visual_valid": np.ones((4, 2), bool),
                  "visual_latent": np.arange(4 * 2 * 2048, dtype=np.float32).reshape(4, 2, 2048) / 1000}
        pack = SimpleNamespace(arrays=arrays, indices={1530: np.array([0]), 1312: np.array([1, 2]), 1633: np.array([3])},
                               episodes={eid: {"task_id": 9} for eid in (1530, 1312, 1633)})
        return pack, {"train_episode_indices": [1633, 1312], "validation_episode_indices": [1530]}

    def references(self, arm):
        grouped = {}
        for index, window in enumerate(self.dataset.windows):
            grouped.setdefault(window.episode_index, []).append(index)
        refs = {}
        for selected in grouped.values():
            raw = runner.collate_window_inputs([self.dataset[index]["inputs"] for index in selected])
            targets = runner.collate_window_targets([self.dataset[index]["targets"] for index in selected])
            inputs = runner.transform_inputs(raw, runner.base.ARMS[0] if arm == "persistence" else arm, "clean", self.anchor)
            captured = runner.persistence_capture(inputs) if arm == "persistence" else runner.capture_prediction(self.model, inputs)
            if arm == "persistence":
                record = {"source_input_sha256": runner.base.fingerprints(raw),
                          "common_target_sha256": runner.base.fingerprints(targets),
                          "predictions_sha256": {"persistence": runner.base.fingerprints(captured["predictions"])}}
            else:
                record = {"input_sha256": runner.base.fingerprints(inputs), "target_sha256": runner.base.fingerprints(targets),
                          "prediction_sha256": runner.base.fingerprints(captured["predictions"])}
            refs[tuple(selected)] = record
        return refs

    def evaluate(self, arm=None, refs=None, trace=None):
        arm = arm or runner.base.ARMS[0]
        return runner.evaluate_paired(None if arm == "persistence" else self.model, self.dataset, self.stats,
            arm=arm, seed=None if arm == "persistence" else runner.base.SEEDS[0], anchor=self.anchor,
            references=self.references(arm) if refs is None else refs, trace=trace)

    def test_anchor_is_actual_minimum_train_frame_zero_with_owned_storage(self):
        pack, split = self.pack_fixture()
        before = pack.arrays["visual_latent"].copy()
        anchor, record = runner.select_anchor(pack, split)
        np.testing.assert_array_equal(anchor.numpy(), before[1])
        self.assertEqual(record["episode_index"], 1312)
        self.assertEqual(record["frame_index"], 0)
        self.assertEqual(record["global_cache_row"], 1)
        self.assertEqual(record["pair_sha256"], runner.base.tensor_hash(anchor))
        self.assertEqual(record["view_sha256"], [runner.base.tensor_hash(anchor[i]) for i in range(2)])
        self.assertFalse(torch.is_inference(anchor) or anchor.requires_grad)
        anchor.add_(1)
        np.testing.assert_array_equal(pack.arrays["visual_latent"], before)

    def test_anchor_rejects_wrong_train_origin_validation_overlap_and_task(self):
        for mode in ("minimum", "overlap", "row_origin", "task"):
            pack, split = self.pack_fixture()
            if mode == "minimum":
                split["train_episode_indices"] = [1633]
            elif mode == "overlap":
                split["validation_episode_indices"].append(1312)
            elif mode == "row_origin":
                pack.arrays["episode_index"][1] = 1530
            else:
                pack.episodes[1530]["task_id"] = 8
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.select_anchor(pack, split)

    def test_anchor_rejects_missing_duplicate_zero_masks_dtype_and_nonfinite(self):
        for mode in ("missing_zero", "duplicate_zero", "mask", "dtype", "nonfinite"):
            pack, split = self.pack_fixture()
            if mode == "missing_zero":
                pack.arrays["frame_index"][1] = 1
            elif mode == "duplicate_zero":
                pack.arrays["frame_index"][2] = 0
            elif mode == "mask":
                pack.arrays["visual_valid"][1, 0] = False
            elif mode == "dtype":
                pack.arrays["visual_latent"] = pack.arrays["visual_latent"].astype(np.float64)
            else:
                pack.arrays["visual_latent"][1, 0, 0] = np.nan
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.select_anchor(pack, split)

    def test_real_native_capture_clean_drift_zero_and_repeat_skip_zero(self):
        before = runner.base.parameter_hash(self.model)
        with patch("torch.optim.AdamW", side_effect=AssertionError("no optimizer")), patch("torch.Tensor.backward", side_effect=AssertionError("no backward")):
            report = self.evaluate()
        self.assertEqual(set(report), set(runner.CONDITIONS))
        for key in runner.DRIFT_KEYS:
            self.assertEqual(report["clean"]["drift"]["micro"][key]["exact_changed_count"], 0)
        repeat = report["repeat_current"]["drift"]["micro"]
        self.assertEqual(repeat["anchor_skip_delta"]["exact_changed_count"], 0)
        self.assertGreater(repeat["history_input_delta"]["exact_changed_count"], 0)
        self.assertGreater(report["fixed_train_anchor"]["drift"]["micro"]["anchor_skip_delta"]["mae"], 0.)
        for condition in runner.CONDITIONS:
            self.assertEqual(report[condition]["micro"]["window_count"], 4)
            self.assertEqual(report[condition]["drift"]["micro"]["windows"], 4)
        self.assertEqual(runner.base.parameter_hash(self.model), before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.model.visual_residual_head._forward_hooks)

    def test_per_batch_clean_first_trace_and_unchanged_targets_nonvisual_inputs(self):
        refs = self.references(runner.base.ARMS[1])
        original = deepcopy(self.dataset.items)
        anchor_before = runner.base.tensor_hash(self.anchor)
        trace = io.StringIO()
        self.evaluate(runner.base.ARMS[1], refs, trace)
        rows = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([row["condition"] for row in rows], list(runner.CONDITIONS) * 2)
        self.assertEqual([row["window_indices"] for row in rows], [[0]] * 3 + [[1, 2, 3]] * 3)
        for row in rows:
            ref = refs[tuple(row["window_indices"])]
            self.assertEqual(row["target_sha256"], ref["target_sha256"])
            for key, digest in ref["input_sha256"].items():
                if key != "history_visual_latent":
                    self.assertEqual(row["input_sha256"][key], digest)
        for old, new in zip(original, self.dataset.items):
            for section in ("inputs", "targets"):
                for key, value in old[section].items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, new[section][key])
        self.assertEqual(runner.base.tensor_hash(self.anchor), anchor_before)

    def test_clean_prediction_drift_stops_before_either_intervention(self):
        refs = self.references(runner.base.ARMS[0])
        refs[(0,)]["prediction_sha256"]["pred_state_delta"] = "0" * 64
        trace = io.StringIO()
        with patch.object(runner, "transform_inputs", wraps=runner.transform_inputs) as transform, self.assertRaisesRegex(ValueError, "clean prediction replay"):
            self.evaluate(refs=refs, trace=trace)
        self.assertEqual(transform.call_count, 1)
        self.assertEqual(transform.call_args.args[2], "clean")
        self.assertEqual(trace.getvalue(), "")

    def test_reference_input_target_and_missing_batch_fail_closed(self):
        for mode in ("input", "target", "missing"):
            refs = self.references(runner.base.ARMS[0])
            if mode == "input":
                refs[(0,)]["input_sha256"]["history_state"] = "0" * 64
            elif mode == "target":
                refs[(0,)]["target_sha256"]["state_delta"] = "0" * 64
            else:
                del refs[(0,)]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.evaluate(refs=refs)

    def test_persistence_repeat_current_prediction_and_skip_are_unchanged(self):
        report = self.evaluate("persistence")
        self.assertEqual(report["clean"]["micro"], report["repeat_current"]["micro"])
        for key in ("state_output_delta", "visual_output_delta", "visual_residual_delta", "anchor_skip_delta"):
            self.assertEqual(report["repeat_current"]["drift"]["micro"][key]["exact_changed_count"], 0)
        self.assertGreater(report["fixed_train_anchor"]["drift"]["micro"]["visual_output_delta"]["mae"], 0.)
        self.assertEqual(report["fixed_train_anchor"]["drift"]["micro"]["visual_residual_delta"]["mae"], 0.)

    def test_incomplete_window_coverage_and_wrong_model_arm_are_rejected(self):
        self.dataset.windows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete validation windows"):
            self.evaluate()
        with self.assertRaisesRegex(ValueError, "model/arm mismatch"):
            runner.evaluate_paired(self.model, self.dataset, self.stats, arm="persistence", seed=None,
                anchor=self.anchor, references={})

    def summary_fixture(self):
        def row(error, drift):
            return {"episode_macro": {"normalized_state_mae": error, "visual_mae": error / 10},
                    "drift": {"episode_macro": {key: {"mae": drift, "rmse": 2 * drift} for key in runner.DRIFT_KEYS}}}
        runs = {}
        for i, seed in enumerate(runner.base.SEEDS):
            runs[str(seed)] = {arm: {
                "clean": row(float(i + 1 + offset), 0.),
                "repeat_current": row(float(2 * (i + 1) + offset), float(i + 1)),
                "fixed_train_anchor": row(float(3 * (i + 1) + offset), float(2 * (i + 1)))}
                for offset, arm in enumerate(runner.base.ARMS)}
        persistence = {"clean": row(0., 0.), "repeat_current": row(0., 0.), "fixed_train_anchor": row(3., 3.)}
        return runs, persistence

    def test_summary_paired_error_arithmetic_and_descriptive_drift_statistics(self):
        runs, persistence = self.summary_fixture()
        report = runner.summarize(runs, persistence)
        first = runner.base.ARMS[0]
        paired = report["paired_error_changes"]["repeat_current"][first]["normalized_state_mae"]
        self.assertEqual(list(paired["by_seed"].values()), [1., 2., 3.])
        self.assertEqual(paired["mean"], 2.)
        self.assertAlmostEqual(paired["descriptive_population_std"], math.sqrt(2 / 3))
        self.assertEqual(report["mean_intervention_minus_clean"]["repeat_current"][first]["normalized_state_mae"],
                         {"absolute": 2., "relative_percent": 100.})
        drift = report["seed_summary_output_drift"]["fixed_train_anchor"][first]["visual_residual_delta"]["mae"]
        self.assertEqual(drift["mean"], 4.)
        self.assertAlmostEqual(drift["descriptive_population_std"], math.sqrt(8 / 3))
        self.assertEqual(report["mean_intervention_minus_clean"]["fixed_train_anchor"]["persistence"]["normalized_state_mae"],
                         {"absolute": 3., "relative_percent": None})
        self.assertEqual(report["decision"], "report_local_visual_sensitivity_without_module_selection")
        self.assertFalse(report["robustness_certified"])

    def test_summary_refuses_incomplete_or_extra_seed_arm_condition(self):
        for mode in ("seed", "extra_seed", "arm", "condition", "persistence"):
            runs, persistence = self.summary_fixture()
            seed = str(runner.base.SEEDS[0])
            if mode == "seed":
                del runs[seed]
            elif mode == "extra_seed":
                runs["extra"] = deepcopy(runs[seed])
            elif mode == "arm":
                del runs[seed][runner.base.ARMS[0]]
            elif mode == "condition":
                del runs[seed][runner.base.ARMS[0]]["clean"]
            else:
                del persistence["clean"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.summarize(runs, persistence)

    def plan_fixture(self):
        return {"schema": "libero_visual_dependence_plan_v1", "allowed": deepcopy(runner.ALLOW),
            "layout": deepcopy(runner.LAYOUT), "anchor": deepcopy(runner.ANCHOR),
            "conditions": list(runner.CONDITIONS), "seeds": list(runner.base.SEEDS), "arms": list(runner.base.ARMS),
            "output": runner.OUT, "previous_plan_sha256": runner.OLD_PLAN_SHA,
            "evaluation_order": "per_batch_clean_replay_then_two_interventions",
            "input_sha256": {name: "1" * 64 for name in runner.NEW_CODE}}

    def test_validate_merges_pins_readonly_and_rejects_scope_or_inventory_drift(self):
        plan = self.plan_fixture()
        with patch.object(runner.base, "checked"), patch.object(runner, "read_json", return_value=plan), patch.object(runner.frozen, "validate", return_value={"input_sha256": {"old_fixture": "2" * 64}}), patch.object(runner.frozen, "rehash") as rehash:
            result = runner.validate(self.root, "3" * 64)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["input_sha256"][runner.PLAN], "3" * 64)
        self.assertEqual(result["input_sha256"]["old_fixture"], "2" * 64)
        rehash.assert_called_once_with(self.root, result["input_sha256"])
        for mode in ("training", "anchor", "order", "inventory"):
            bad = deepcopy(plan)
            if mode == "training":
                bad["allowed"]["optimizer_updates"] = True
            elif mode == "anchor":
                bad["anchor"]["episode_index"] = 1530
            elif mode == "order":
                bad["evaluation_order"] = "all_interventions_before_clean"
            else:
                bad["input_sha256"].pop(next(iter(bad["input_sha256"])))
            with self.subTest(mode=mode), patch.object(runner.base, "checked"), patch.object(runner, "read_json", return_value=bad), patch.object(runner.frozen, "validate") as previous, self.assertRaises(ValueError):
                runner.validate(self.root, "3" * 64)
            previous.assert_not_called()

    def test_cli_execution_flag_guards_and_preflight_does_not_run(self):
        for args in (("--stage", "preflight", "--execute"), ("--stage", "evaluate")):
            with self.subTest(args=args), patch.dict(os.environ), patch.object(runner, "validate") as validate, patch.object(runner, "run") as execute, self.assertRaises(ValueError):
                runner.main([*args, "--plan-sha256", "1" * 64])
            validate.assert_not_called()
            execute.assert_not_called()
        with patch.dict(os.environ), patch.object(runner, "validate", return_value={"status": "fixture_preflight", "input_sha256": {}}) as validate, patch.object(runner, "run") as execute, redirect_stdout(io.StringIO()):
            runner.main(["--stage", "preflight", "--plan-sha256", "1" * 64])
        validate.assert_called_once_with(runner.ROOT, "1" * 64)
        execute.assert_not_called()

    def test_run_requires_exact_true_and_existing_output_cannot_be_reused(self):
        for flag in (False, 1, None):
            with self.subTest(flag=flag), patch.object(runner, "validate") as validate, self.assertRaises(ValueError):
                runner.run(self.root, "1" * 64, flag)
            validate.assert_not_called()
        (self.root / runner.OUT).mkdir(parents=True)
        with patch.object(runner, "validate") as validate, self.assertRaises(ValueError):
            runner.run(self.root, "1" * 64, True)
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
