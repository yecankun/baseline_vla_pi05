"""Synthetic native-shape step0 tests; no public data, training or PI0.5 load."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_pi05_libero_action_study_step0 as step0


REGISTRY = [{"task_id": 9, "source_task_index": 39,
             "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]


class TinyDataset:
    """Unequal episode window counts deliberately distinguish macro and micro."""
    def __init__(self):
        self.windows = [SimpleNamespace(episode_index=e) for e in (11, 22, 22, 22)]
        self.items = []
        for index, window in enumerate(self.windows):
            self.items.append({"inputs": {
                "history_visual_latent": np.zeros((4, 2, 2048), np.float32),
                "history_visual_valid": np.ones((4, 2), bool),
                "history_state": np.zeros((4, 8), np.float32),
                "history_state_valid": np.ones((4, 8), bool),
                "candidate_actions": np.full((1, 3, 7), .25 + index / 10, np.float32),
                "task_instruction": REGISTRY[0]["task_instruction"]},
                "targets": {"future_visual_latent": np.zeros((3, 2, 2048), np.float32),
                    "future_visual_valid": np.ones((3, 2), bool),
                    "state_delta": np.full((3, 8), 1 if window.episode_index == 11 else 3, np.float32),
                    "state_target_valid": np.ones((3, 8), bool)},
                "metadata": {"episode_index": window.episode_index}})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class FixturePredictor(torch.nn.Module):
    """Arithmetic fixture only; real-model initialization/forward tested below."""
    def __init__(self, action_echo=False, mutate=None):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.action_echo, self.mutate, self.seen = action_echo, mutate, []
        self.eval()

    def forward(self, **inputs):
        assert set(inputs) == {"history_visual_latent", "history_visual_valid", "history_state", "history_state_valid", "candidate_actions", "task_instruction"}
        assert torch.is_inference_mode_enabled()
        self.seen.append(inputs["candidate_actions"].clone())
        if self.mutate == "input":
            inputs["candidate_actions"].add_(1)
        if self.mutate == "parameter":
            self.marker.add_(1)
        batch = len(inputs["task_instruction"])
        state = (inputs["candidate_actions"].mean(-1, keepdim=True).expand(batch, 1, 3, 8).clone()
                 if self.action_echo else torch.zeros(batch, 1, 3, 8))
        if self.mutate == "nonfinite":
            state.fill_(float("nan"))
        return {"pred_future_visual_latent": torch.zeros(batch, 1, 3, 2, 2048), "pred_state_delta": state}


@contextmanager
def forbid_training():
    with ExitStack() as stack:
        spies = [stack.enter_context(patch(target, side_effect=AssertionError("step0 must not train/refit")))
                 for target in ("torch.optim.Adam", "torch.optim.AdamW", "torch.optim.SGD",
                                "torch.Tensor.backward", "torch.autograd.backward",
                                "pi05_libero_world_model_adapter.fit_train_normalization")]
        yield spies
        for spy in spies:
            spy.assert_not_called()


class StepZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.dataset = TinyDataset()
        self.stats = {"state_std": [1.] * 8}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def pair(self, **kwargs):
        return {arm: FixturePredictor(**kwargs) for arm in step0.ARMS}

    def inputs(self):
        return step0.collate_window_inputs([item["inputs"] for item in self.dataset.items])

    def test_pair_inputs_only_action_changes_without_alias_or_source_mutation(self):
        inputs = self.inputs()
        before = step0.fingerprints(inputs)
        pair = step0.paired_inputs(inputs)
        self.assertEqual(step0.fingerprints(inputs), before)
        self.assertEqual(step0.fingerprints(pair[step0.ARMS[0]]), before)
        for key in inputs:
            if key != "candidate_actions":
                self.assertEqual(step0.fingerprints({key: pair[step0.ARMS[0]][key]}),
                                 step0.fingerprints({key: pair[step0.ARMS[1]][key]}))
            if isinstance(inputs[key], torch.Tensor):
                self.assertNotEqual(pair[step0.ARMS[0]][key].data_ptr(), inputs[key].data_ptr())
                self.assertNotEqual(pair[step0.ARMS[0]][key].data_ptr(), pair[step0.ARMS[1]][key].data_ptr())
        self.assertEqual(int(torch.count_nonzero(pair[step0.ARMS[1]]["candidate_actions"])), 0)

    def test_pair_rejects_extra_target_nonfinite_action_and_inference_context(self):
        for mode in ("target", "nan", "inference"):
            inputs = self.inputs()
            if mode == "target":
                inputs["targets"] = {}
            if mode == "nan":
                inputs["candidate_actions"][0, 0, 0, 0] = float("nan")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                if mode == "inference":
                    with torch.inference_mode():
                        step0.paired_inputs(inputs)
                else:
                    step0.paired_inputs(inputs)

    def test_real_native_pair_identical_disjoint_frozen_and_rng_restored(self):
        rng = torch.get_rng_state().clone()
        with forbid_training():
            pair = step0.initialized_pair(step0.SEEDS[0], REGISTRY)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(len({step0.parameter_hash(model) for model in pair.values()}), 1)
        for a, b in zip(pair[step0.ARMS[0]].parameters(), pair[step0.ARMS[1]].parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertFalse(a.requires_grad or b.requires_grad)
            self.assertIsNone(a.grad)
        for model in pair.values():
            self.assertTrue(all(not m.training for m in model.modules()))

    def test_initialization_rejects_unregistered_or_coerced_seed(self):
        for seed in (1, True, float(step0.SEEDS[0]), str(step0.SEEDS[0])):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                step0.initialized_pair(seed, REGISTRY)

    def test_all_three_planned_seeds_pair_match_but_cross_seed_hashes_differ(self):
        hashes = []
        with forbid_training():
            for seed in step0.SEEDS:
                pair = step0.initialized_pair(seed, REGISTRY)
                current = [step0.parameter_hash(pair[arm]) for arm in step0.ARMS]
                self.assertEqual(current[0], current[1])
                hashes.append(current[0])
        self.assertEqual(len(hashes), 3)
        self.assertEqual(len(set(hashes)), 3)

    def test_real_native_forward_only_is_finite_and_parameters_unchanged(self):
        with forbid_training():
            models = step0.initialized_pair(step0.SEEDS[0], REGISTRY)
            before = {arm: step0.parameter_hash(model) for arm, model in models.items()}
            report = step0.evaluate_pair(models, self.dataset, self.stats, batch_size=2)
        self.assertEqual(report["coverage"]["windows"], 4)
        self.assertEqual(report["parameter_sha256"], before)
        self.assertEqual({arm: step0.parameter_hash(model) for arm, model in models.items()}, before)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["backward_calls"], 0)
        self.assertTrue(report["parameters_and_versions_unchanged"])

    def test_episode_macro_not_window_weighted_micro(self):
        with forbid_training():
            report = step0.evaluate_pair(self.pair(), self.dataset, self.stats, batch_size=2)
        for metrics in report["arms"].values():
            self.assertEqual(metrics["episode_macro"]["normalized_state_mae"], 2.)
            self.assertEqual(metrics["micro"]["state_normalized"]["aggregate"]["mae"], 2.5)
            self.assertEqual(metrics["per_episode"]["11"]["masked_objective"]["state_count"], 24)
            self.assertEqual(metrics["per_episode"]["22"]["masked_objective"]["state_count"], 72)
        self.assertEqual(report["coverage"]["batches"], 3)

    def test_sufficient_statistics_and_macro_invariant_to_batch_size(self):
        for index, item in enumerate(self.dataset.items):
            item["targets"]["future_visual_latent"].fill((index + 1) * .25)
            item["targets"]["future_visual_valid"][index % 3, index % 2] = False
            item["targets"]["state_target_valid"][index % 3, index % 8] = False
            item["inputs"]["history_state_valid"][-1, (index + 2) % 8] = False
            if index % 2:
                item["inputs"]["history_visual_valid"][-1, 0] = False

        def without_update_count(value):
            if isinstance(value, dict):
                return {key: without_update_count(item) for key, item in value.items() if key != "update_count"}
            if isinstance(value, list):
                return [without_update_count(item) for item in value]
            return value

        reports = []
        with forbid_training():
            for batch_size in (1, 2, 16):
                reports.append(step0.evaluate_pair(self.pair(), self.dataset, self.stats, batch_size=batch_size))
        for report in reports[1:]:
            for arm in (*step0.ARMS, "persistence"):
                for key in ("per_episode", "micro", "episode_macro"):
                    self.assertEqual(without_update_count(report["arms"][arm][key]),
                                     without_update_count(reports[0]["arms"][arm][key]))
            self.assertEqual(report["episode_macro_changes"], reports[0]["episode_macro_changes"])
        self.assertEqual([report["coverage"]["batches"] for report in reports], [4, 3, 2])

    def test_zero_action_is_applied_during_evaluation_and_targets_stay_unchanged(self):
        before = deepcopy(self.dataset.items)
        models = self.pair(action_echo=True)
        trace = io.StringIO()
        report = step0.evaluate_pair(models, self.dataset, self.stats, batch_size=2, trace=trace)
        self.assertTrue(all(bool((v == 0).all()) for v in models[step0.ARMS[1]].seen))
        self.assertTrue(all(bool((v != 0).all()) for v in models[step0.ARMS[0]].seen))
        self.assertLess(report["episode_macro_changes"]["observed_minus_zero"]["normalized_state_mae"]["absolute"], 0)
        self.assertEqual(len(trace.getvalue().splitlines()), 3)
        for old, new in zip(before, self.dataset.items):
            for key in old["targets"]:
                np.testing.assert_array_equal(old["targets"][key], new["targets"][key])

    def test_persistence_and_both_arms_share_intersected_target_support(self):
        for item in self.dataset.items:
            item["inputs"]["history_visual_valid"][-1, 0] = False
            item["inputs"]["history_state_valid"][-1, 0] = False
        report = step0.evaluate_pair(self.pair(), self.dataset, self.stats)
        coverage = report["coverage"]
        self.assertEqual((coverage["original_state_scalars"], coverage["common_state_scalars"]), (96, 84))
        self.assertEqual((coverage["original_visual_views"], coverage["common_visual_views"]), (24, 12))
        for metrics in report["arms"].values():
            self.assertEqual(metrics["micro"]["masked_objective"]["state_count"], 84)
            self.assertEqual(metrics["micro"]["masked_objective"]["visual_count"], 12 * 2048)

    def test_fixed_full_support_gate_rejects_reduced_masks_before_forward(self):
        self.dataset.items[0]["inputs"]["history_visual_valid"][-1, 0] = False
        models = self.pair()
        with self.assertRaisesRegex(ValueError, "scoring support changed"):
            step0.evaluate_pair(models, self.dataset, self.stats, require_full_support=True)
        self.assertTrue(all(not model.seen for model in models.values()))

    def test_bad_model_state_or_pair_hash_fails_before_forward(self):
        for mode in ("training", "requires_grad", "gradient", "hash", "extra_arm"):
            models = self.pair()
            selected = models[step0.ARMS[0]]
            if mode == "training":
                selected.train()
            elif mode == "requires_grad":
                selected.marker.requires_grad_(True)
            elif mode == "gradient":
                selected.marker.grad = torch.ones_like(selected.marker)
            elif mode == "hash":
                selected.marker.add_(1)
            else:
                models["extra"] = FixturePredictor()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                step0.evaluate_pair(models, self.dataset, self.stats)
            self.assertEqual(selected.seen, [])

    def test_nonfinite_output_input_mutation_and_parameter_mutation_fail(self):
        for mode, message in (("nonfinite", "finite"), ("input", "arm inputs mutated"), ("parameter", "model changed")):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, message):
                step0.evaluate_pair(self.pair(mutate=mode), self.dataset, self.stats)

    def test_batch_limit_and_incomplete_window_inventory_fail_closed(self):
        for batch in (0, 17, True, 1.5):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                step0.evaluate_pair(self.pair(), self.dataset, self.stats, batch_size=batch)
        self.dataset.windows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete evaluation"):
            step0.evaluate_pair(self.pair(), self.dataset, self.stats)

    def test_changes_null_zero_denominator_and_empty_macro(self):
        self.assertEqual(step0.changes({"a": 1., "b": None}, {"a": 0., "b": 2.}),
                         {"a": {"absolute": 1., "relative_percent": None}, "b": {"absolute": None, "relative_percent": None}})
        with self.assertRaises(ValueError):
            step0.macro_metrics({})

    def test_checked_rejects_hash_drift_and_unsafe_paths(self):
        path = self.root / "input.json"
        path.write_text("{}", encoding="utf-8")
        digest = step0.sha256_file(path)
        self.assertEqual(step0.checked(self.root, path.name, digest), path.resolve())
        for name in ("../input.json", "./input.json", "a//b", "a\\b", str(path.resolve())):
            with self.subTest(name=name), self.assertRaises(ValueError):
                step0.checked(self.root, name, digest)
        path.write_text('{"drift":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            step0.checked(self.root, path.name, digest)

    def test_validate_rejects_wrong_path_or_scope_before_models(self):
        with self.assertRaisesRegex(ValueError, "fixed step0 plan path"):
            step0.validate(self.root, self.root / "other.json", "0"*64)
        plan = {"schema": "libero_action_study_step0_plan_v1", "authorization": {"optimization": True}}
        with patch.object(step0, "checked"), patch.object(step0, "read_json", return_value=plan), patch.object(step0, "initialized_pair") as initialized, self.assertRaisesRegex(ValueError, "scope changed"):
            step0.validate(self.root, self.root / step0.PLAN_PATH, "0"*64)
        initialized.assert_not_called()

    def test_validate_requires_complete_inventory_fixed_model_and_seeds(self):
        repo = Path(step0.__file__).resolve().parents[1]
        original = step0.read_json(repo / step0.PLAN_PATH)
        for mode in ("missing_input", "extra_input", "model", "seed"):
            plan = deepcopy(original)
            if mode == "missing_input":
                del plan["input_sha256"][next(iter(plan["input_sha256"]))]
            elif mode == "extra_input":
                plan["input_sha256"]["extra.json"] = "0"*64
            elif mode == "model":
                plan["model_config"]["hidden_dim"] += 1
            else:
                plan["seeds"][0] += 17
            with self.subTest(mode=mode), patch.object(step0, "checked"), patch.object(step0, "read_json", return_value=plan), self.assertRaises(ValueError):
                step0.validate(self.root, self.root / step0.PLAN_PATH, "0"*64)

    def test_validate_forward_only_preflight_and_runtime_version_gate(self):
        repo = Path(step0.__file__).resolve().parents[1]
        plan = step0.read_json(repo / step0.PLAN_PATH)
        study = {"future_execution_design": {"optimizer_updates_per_arm_seed": 200, "batch_size": 16}}
        cache = {"status": "passed_frozen_feature_cache_only", "output_sha256": {"fixture.npy": "1"*64}}
        for version in ("2.10.0+cu128", "different_runtime"):
            with self.subTest(version=version), patch.object(step0, "checked") as checked, patch.object(step0, "read_json", side_effect=[plan, study, cache]), patch.object(torch, "__version__", version), forbid_training():
                if version == "different_runtime":
                    with self.assertRaisesRegex(ValueError, "remote torch runtime"):
                        step0.validate(self.root, self.root / step0.PLAN_PATH, "0"*64)
                else:
                    result = step0.validate(self.root, self.root / step0.PLAN_PATH, "0"*64)
                    self.assertEqual(result["status"], "passed_preflight_no_model_or_optimizer")
                    self.assertEqual(result["optimizer_steps"], 0)
                    self.assertEqual(result["input_sha256"][step0.PACK_PATH + "/fixture.npy"], "1"*64)
                    self.assertEqual(checked.call_count, 1 + len(plan["input_sha256"]) + 1)

    def test_run_requires_explicit_true_and_fresh_output_before_validation(self):
        for flag in (False, None, 1):
            with self.subTest(flag=flag), patch.object(step0, "validate") as validate, self.assertRaises(ValueError):
                step0.run_step0(self.root, self.root / step0.PLAN_PATH, "0"*64, execute=flag)
            validate.assert_not_called()
        out = self.root / step0.OUT_PATH
        out.mkdir(parents=True)
        (out / "preserve.txt").write_text("keep", encoding="utf-8")
        with patch.object(step0, "validate") as validate, self.assertRaisesRegex(ValueError, "output exists"):
            step0.run_step0(self.root, self.root / step0.PLAN_PATH, "0"*64, execute=True)
        validate.assert_not_called()
        self.assertEqual((out / "preserve.txt").read_text(), "keep")

    def test_run_failure_preserves_preflight_and_forbids_resume(self):
        (self.root / "simulation_output").mkdir()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            with patch.object(step0, "validate", return_value={"input_sha256": {}, "fixture_only": True}), patch.object(step0, "LiberoFeaturePack", side_effect=ValueError("fixture source drift")), forbid_training(), self.assertRaisesRegex(ValueError, "fixture source drift"):
                step0.run_step0(self.root, self.root / step0.PLAN_PATH, "0"*64, execute=True)
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        out = self.root / step0.OUT_PATH
        failure = step0.read_json(out / "failure.json")
        self.assertEqual(failure["optimizer_steps"], 0)
        self.assertEqual(failure["status"], "failed_preserve_no_automatic_retry")
        self.assertTrue((out / "preflight.json").exists())
        self.assertTrue((out / "started.json").exists())
        self.assertFalse((out / "report.json").exists())
        with self.assertRaisesRegex(ValueError, "output exists"):
            step0.run_step0(self.root, self.root / step0.PLAN_PATH, "0"*64, execute=True)

    def test_cli_has_no_training_switch_and_enforces_execution_flag(self):
        with patch.object(step0, "validate") as validate, self.assertRaises(ValueError):
            step0.main(["--stage", "preflight", "--plan-sha256", "0"*64, "--execute-step0"])
        validate.assert_not_called()
        with patch.object(step0, "validate") as validate, self.assertRaises(ValueError):
            step0.main(["--stage", "step0", "--plan-sha256", "0"*64])
        validate.assert_not_called()
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            step0.main(["--stage", "train", "--plan-sha256", "0"*64])


if __name__ == "__main__":
    unittest.main()
