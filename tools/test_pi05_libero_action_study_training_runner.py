"""Synthetic runner guards; never public gradients/updates or real authorization.

Temporary positive authorization JSONs are fixtures confined to TemporaryDirectory.
The 200-step loop tests mock optimizer/update calls; they perform zero learning.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_pi05_libero_action_study_training as runner
from test_pi05_libero_action_study_step0 import TinyDataset, FixturePredictor, REGISTRY


class TrainingRunnerTests(unittest.TestCase):
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
        self.stats = {"state_std": [1.] * 8}
        self.plan_sha = "1" * 64
        self.bound = self.root / "fixture_input.json"
        self.bound.write_text('{"fixture_only":true}\n', encoding="utf-8")
        self.gradient = {"status": "passed_six_first_batch_gradient_checks_no_optimizer",
            "training_plan_sha256": self.plan_sha, "optimizer_constructed": False,
            "public_optimizer_steps": 0, "backward_calls": 6,
            "results": {str(seed): {arm: {"fixture_only": True} for arm in runner.base.ARMS} for seed in runner.base.SEEDS},
            "input_sha256": {self.bound.name: runner.sha256_file(self.bound)}}
        self.auth = {"schema": "libero_action_study_training_authorization_v1",
            "training_execution_allowed": True, "training_plan_sha256": self.plan_sha,
            "gradient_check_report_sha256": "0" * 64, "optimizer_updates_per_arm_seed": 200,
            "total_optimizer_updates": 1200, "output": runner.TRAIN_ROOT}
        self.manifest = {"batches": 200, "batch_size": 16, "seed": runner.base.SEEDS[0], "draws": []}
        for i in range(3200):
            index = i % len(self.dataset)
            self.manifest["draws"].append({"batch_index": i // 16, "draw_in_batch": i % 16,
                "window_index": index, "episode_index": self.dataset.windows[index].episode_index})

    def save_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return runner.sha256_file(path)

    def save_auth(self):
        self.auth["gradient_check_report_sha256"] = self.save_json(runner.GRAD_ROOT + "/report.json", self.gradient)
        return self.save_json(runner.AUTH, self.auth)

    def reference(self, action_echo=False):
        models = {arm: FixturePredictor(action_echo=action_echo) for arm in runner.base.ARMS}
        trace = io.StringIO()
        result = runner.base.evaluate_pair(models, self.dataset, self.stats, trace=trace,
            seed=runner.base.SEEDS[0], partition="validation", require_full_support=True)
        records = [json.loads(line) for line in trace.getvalue().splitlines()]
        refs = {(row["seed"], row["partition"], tuple(row["window_indices"])): row for row in records}
        return result, refs

    def tiny_batch(self):
        item = self.dataset[0]
        raw = runner.collate_window_inputs([item["inputs"]])
        target = runner.collate_window_targets([item["targets"]])
        return raw, target, list(range(16))

    def test_authorization_missing_flag_sha_or_file_fail_before_read(self):
        for execute, digest in ((False, "0"*64), (1, "0"*64), (True, None)):
            with self.subTest(execute=execute, digest=digest), patch.object(runner, "read_json") as reader, self.assertRaises(ValueError):
                runner.load_authorization(self.root, self.plan_sha, digest, execute)
            reader.assert_not_called()
        with self.assertRaises(ValueError):
            runner.load_authorization(self.root, self.plan_sha, "0"*64, True)

    def test_temporary_exact_positive_authorization_fixture_passes(self):
        digest = self.save_auth()
        self.assertEqual(runner.load_authorization(self.root, self.plan_sha, digest, True), self.auth)

    def test_authorization_false_missing_extra_and_budget_drift_rejected(self):
        original = deepcopy(self.auth)
        for mode in ("false", "missing", "extra", "plan", "output", "steps", "total", "coerced"):
            self.auth = deepcopy(original)
            if mode == "false":
                self.auth["training_execution_allowed"] = False
            elif mode == "missing":
                del self.auth["schema"]
            elif mode == "extra":
                self.auth["resume"] = True
            elif mode == "plan":
                self.auth["training_plan_sha256"] = "2"*64
            elif mode == "output":
                self.auth["output"] += "_replacement"
            elif mode == "steps":
                self.auth["optimizer_updates_per_arm_seed"] = 201
            elif mode == "total":
                self.auth["total_optimizer_updates"] = 1201
            else:
                self.auth["optimizer_updates_per_arm_seed"] = 200.0
            digest = self.save_auth()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.load_authorization(self.root, self.plan_sha, digest, True)

    def test_authorization_gradient_report_and_bound_input_hash_drift(self):
        digest = self.save_auth()
        (self.root / runner.AUTH).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            runner.load_authorization(self.root, self.plan_sha, digest, True)
        digest = self.save_auth()
        (self.root / runner.GRAD_ROOT / "report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            runner.load_authorization(self.root, self.plan_sha, digest, True)
        digest = self.save_auth()
        self.bound.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            runner.load_authorization(self.root, self.plan_sha, digest, True)

    def test_gradient_gate_requires_six_matching_no_optimizer_results(self):
        original = deepcopy(self.gradient)
        for mode in ("status", "plan", "optimizer", "steps", "backward", "seed", "arm"):
            self.gradient = deepcopy(original)
            if mode == "status":
                self.gradient["status"] = "failed"
            elif mode == "plan":
                self.gradient["training_plan_sha256"] = "2"*64
            elif mode == "optimizer":
                self.gradient["optimizer_constructed"] = True
            elif mode == "steps":
                self.gradient["public_optimizer_steps"] = 1
            elif mode == "backward":
                self.gradient["backward_calls"] = 5
            elif mode == "seed":
                self.gradient["results"].pop(str(runner.base.SEEDS[0]))
            else:
                self.gradient["results"][str(runner.base.SEEDS[0])].pop(runner.base.ARMS[0])
            digest = self.save_auth()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.load_authorization(self.root, self.plan_sha, digest, True)

    def test_draw_batch_all_3200_positions_exact_including_last_sixteen(self):
        # Reading windows is synthetic; no model/optimizer participates.
        seen = []
        for step in range(1, 201):
            inputs, targets, indices = runner.draw_batch(self.manifest, self.dataset, step)
            self.assertEqual(tuple(inputs["candidate_actions"].shape), (16, 1, 3, 7))
            self.assertEqual(tuple(targets["state_delta"].shape), (16, 3, 8))
            self.assertEqual(indices, [row["window_index"] for row in self.manifest["draws"][(step-1)*16:step*16]])
            seen.extend(indices)
        self.assertEqual(seen, [row["window_index"] for row in self.manifest["draws"]])
        self.assertEqual(len(seen), 3200)

    def test_draw_batch_rejects_truncation_reorder_wrong_episode_and_step(self):
        for step in (0, 201, True, 1.0):
            with self.subTest(step=step), self.assertRaises(ValueError):
                runner.draw_batch(self.manifest, self.dataset, step)
        for mode in ("short", "order", "episode", "index"):
            manifest = deepcopy(self.manifest)
            if mode == "short":
                manifest["draws"].pop()
            elif mode == "order":
                manifest["draws"][0]["draw_in_batch"] = 1
            elif mode == "episode":
                manifest["draws"][0]["episode_index"] = 999
            else:
                manifest["draws"][0]["window_index"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.draw_batch(manifest, self.dataset, 1)

    def test_initial_evaluation_reproduces_full_reference_and_statistics(self):
        reference, refs = self.reference()
        for arm in runner.base.ARMS:
            with patch.object(runner.training, "build_optimizer", side_effect=AssertionError("no optimizer")), patch("torch.Tensor.backward", side_effect=AssertionError("no backward")):
                actual = runner.evaluate_arm(FixturePredictor(), self.dataset, self.stats, arm,
                    runner.base.SEEDS[0], "validation", refs, verify_step0=True)
            runner.compare_initial(actual, reference["arms"][arm])

    def test_evaluation_rejects_input_target_and_prediction_hash_drift(self):
        _, original = self.reference()
        arm = runner.base.ARMS[0]
        for key in ("source_input_sha256", "original_target_sha256", "predictions_sha256"):
            refs = deepcopy(original)
            first = next(iter(refs.values()))
            if key == "predictions_sha256":
                first[key][arm]["pred_state_delta"] = "0"*64
            else:
                first[key][next(iter(first[key]))] = "0"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.evaluate_arm(FixturePredictor(), self.dataset, self.stats, arm,
                    runner.base.SEEDS[0], "validation", refs, verify_step0=True)

    def test_full_support_gate_cannot_be_bypassed_with_matching_input_hash(self):
        _, refs = self.reference()
        self.dataset[0]["inputs"]["history_visual_valid"][-1, 0] = False
        item = self.dataset[0]
        inputs = runner.collate_window_inputs([item["inputs"]])
        first = refs[(runner.base.SEEDS[0], "validation", (0,))]
        first["source_input_sha256"] = runner.base.fingerprints(inputs)
        first["arm_input_sha256"] = {arm: runner.base.fingerprints(runner.action_ablation_inputs(inputs, arm)) for arm in runner.base.ARMS}
        with self.assertRaisesRegex(ValueError, "scoring support changed"):
            runner.evaluate_arm(FixturePredictor(), self.dataset, self.stats, runner.base.ARMS[0],
                runner.base.SEEDS[0], "validation", refs)

    def test_final_evaluation_keeps_zero_action_and_accepts_changed_prediction(self):
        _, refs = self.reference()
        zero = FixturePredictor(action_echo=True)
        raw = FixturePredictor(action_echo=True)
        for arm, model in zip(runner.base.ARMS, (raw, zero)):
            runner.evaluate_arm(model, self.dataset, self.stats, arm, runner.base.SEEDS[0], "validation", refs, verify_step0=False)
        self.assertTrue(all(bool((actions == 0).all()) for actions in zero.seen))
        self.assertTrue(all(bool((actions != 0).all()) for actions in raw.seen))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in zero.parameters()))

    def test_compare_initial_ignores_timing_but_not_any_statistics(self):
        reference, _ = self.reference()
        expected = reference["arms"][runner.base.ARMS[0]]
        actual = deepcopy(expected)
        actual["forward_and_scoring_seconds"] = -1
        runner.compare_initial(actual, expected)
        actual["episode_macro"]["normalized_state_mae"] += .000001
        with self.assertRaises(ValueError):
            runner.compare_initial(actual, expected)

    def test_train_arm_calls_exact_steps_one_through_200_and_zero_transform(self):
        model, trace = FixturePredictor(), io.StringIO()
        def update(model, optimizer, inputs, targets, *, arm, expected_step):
            self.assertTrue(bool((inputs["candidate_actions"] == 0).all()))
            return {"step": expected_step, "fixture_no_update": True}
        sentinel = object()
        with patch.object(runner.training, "build_optimizer", return_value=sentinel) as optimizer, patch.object(runner, "draw_batch", side_effect=lambda m, d, step: self.tiny_batch()) as batches, patch.object(runner.training, "train_step", side_effect=update) as steps:
            returned, report = runner.train_arm(model, self.dataset, self.manifest, runner.base.ARMS[1], trace)
        self.assertIs(returned, sentinel)
        optimizer.assert_called_once()
        self.assertEqual([call.args[2] for call in batches.call_args_list], list(range(1, 201)))
        self.assertEqual([call.kwargs["expected_step"] for call in steps.call_args_list], list(range(1, 201)))
        self.assertEqual(report["optimizer_steps"], 200)
        self.assertEqual(report["last_step"]["step"], 200)
        self.assertEqual([json.loads(line)["step"] for line in trace.getvalue().splitlines()], list(range(1, 201)))

    def test_step200_failure_is_not_swallowed_or_reported_complete(self):
        trace = io.StringIO()
        def update(*args, **kwargs):
            if kwargs["expected_step"] == 200:
                raise ValueError("fixture last update failed")
            return {"step": kwargs["expected_step"]}
        with patch.object(runner.training, "build_optimizer", return_value=object()), patch.object(runner, "draw_batch", side_effect=lambda *args: self.tiny_batch()), patch.object(runner.training, "train_step", side_effect=update) as update_mock, self.assertRaisesRegex(ValueError, "last update failed"):
            runner.train_arm(FixturePredictor(), self.dataset, self.manifest, runner.base.ARMS[0], trace)
        self.assertEqual(update_mock.call_count, 200)
        rows = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(len(rows), 199)
        self.assertEqual(rows[-1]["step"], 199)

    def test_train_arm_counter_and_input_mutation_fail_without_success_record(self):
        for mode in ("counter", "input"):
            trace = io.StringIO()
            def update(model, optimizer, inputs, targets, **kwargs):
                if mode == "input":
                    inputs["candidate_actions"].add_(1)
                return {"step": 2 if mode == "counter" else kwargs["expected_step"]}
            with self.subTest(mode=mode), patch.object(runner.training, "build_optimizer", return_value=object()), patch.object(runner, "draw_batch", side_effect=lambda *args: self.tiny_batch()), patch.object(runner.training, "train_step", side_effect=update), self.assertRaises(ValueError):
                runner.train_arm(FixturePredictor(), self.dataset, self.manifest, runner.base.ARMS[0], trace)
            self.assertEqual(trace.getvalue(), "")

    def test_existing_outputs_reject_before_validation_or_optimizer(self):
        for relative, call in ((runner.GRAD_ROOT, lambda: runner.gradient_check(self.root, self.plan_sha, True)),
                               (runner.TRAIN_ROOT, lambda: runner.train(self.root, self.plan_sha, "2"*64, True))):
            (self.root / relative).mkdir(parents=True)
            with self.subTest(relative=relative), patch.object(runner, "validate") as validation, patch.object(runner.training, "build_optimizer") as optimizer, self.assertRaises(ValueError):
                call()
            validation.assert_not_called()
            optimizer.assert_not_called()

    def test_train_failure_after_claim_preserves_failure_and_never_retries(self):
        (self.root / "simulation_output").mkdir(exist_ok=True)
        with patch.object(runner, "validate", return_value={"stage": "train", "fixture_only": True}), patch.object(runner, "runtime_settings"), patch.object(runner, "LiberoFeaturePack", side_effect=ValueError("fixture source gate")), patch.object(runner.training, "build_optimizer") as optimizer, self.assertRaisesRegex(ValueError, "fixture source gate"):
            runner.train(self.root, self.plan_sha, "2"*64, True)
        optimizer.assert_not_called()
        out = self.root / runner.TRAIN_ROOT
        self.assertEqual(runner.read_json(out / "failure.json")["status"], "failed_preserve_no_retry")
        self.assertTrue((out / "started.json").exists())
        self.assertFalse((out / "report.json").exists())
        with patch.object(runner, "validate") as validation, self.assertRaises(ValueError):
            runner.train(self.root, self.plan_sha, "2"*64, True)
        validation.assert_not_called()

    def test_gradient_failure_preserves_no_update_claim_and_no_optimizer(self):
        (self.root / "simulation_output").mkdir(exist_ok=True)
        with patch.object(runner, "validate", return_value={"stage": "gradient_check", "fixture_only": True}), patch.object(runner, "runtime_settings"), patch.object(runner, "LiberoFeaturePack", side_effect=ValueError("fixture source gate")), patch.object(runner.training, "build_optimizer") as optimizer, self.assertRaises(ValueError):
            runner.gradient_check(self.root, self.plan_sha, True)
        optimizer.assert_not_called()
        failure = runner.read_json(self.root / runner.GRAD_ROOT / "failure.json")
        self.assertEqual(failure["public_optimizer_steps"], 0)
        self.assertFalse((self.root / runner.GRAD_ROOT / "report.json").exists())

    def test_gradient_stage_dispatches_only_six_probes_from_three_first_batches(self):
        # Probe internals have separate tests. This fixture verifies orchestration
        # only: it intentionally does not call backward or construct an optimizer.
        (self.root / "simulation_output").mkdir(exist_ok=True)
        draws = {seed: {**self.manifest, "seed": seed} for seed in runner.base.SEEDS}
        initial_hash = runner.base.parameter_hash(FixturePredictor())
        frozen = {"per_episode": {}, "episode_macro": {"normalized_state_mae": 1.},
                  "micro": {"window_count": 4, "update_count": 2}, "forward_and_scoring_seconds": .1}
        reference = {"runs": {str(seed): {part: {"parameter_sha256": {arm: initial_hash for arm in runner.base.ARMS},
                    "arms": {arm: deepcopy(frozen) for arm in runner.base.ARMS}} for part in ("train", "validation")}
                    for seed in runner.base.SEEDS}}
        validation = TinyDataset()
        for item in validation.items:
            item["targets"]["state_delta"].fill(99.)
        datasets = {"train": self.dataset, "validation": validation}
        train_targets = runner.collate_window_targets([self.dataset[d["window_index"]]["targets"]
                                                       for d in self.manifest["draws"][:16]])
        seen, events = [], []
        def initialize(seed, registry):
            models = {arm: FixturePredictor() for arm in runner.base.ARMS}
            for model in models.values():
                model.fixture_seed = seed
            return models
        def replay(model, dataset, stats, arm, seed, part, references, *, verify_step0):
            self.assertTrue(verify_step0)
            self.assertIs(dataset, datasets[part])
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            events.append((seed, arm, "evaluate", part))
            return deepcopy(frozen)
        def probe(model, inputs, targets, arm):
            self.assertTrue(model.training and all(p.requires_grad for p in model.parameters()))
            self.assertEqual(tuple(inputs["candidate_actions"].shape), (16, 1, 3, 7))
            for key in train_targets:
                self.assertTrue(torch.equal(targets[key], train_targets[key]))
            if arm == runner.base.ARMS[1]:
                self.assertTrue(bool((inputs["candidate_actions"] == 0).all()))
            seen.append(arm)
            events.append((model.fixture_seed, arm, "probe", "train_first16"))
            return {"fixture_only_no_backward": True, "backward_calls": 1, "optimizer_steps": 0}
        preflight = {"stage": "gradient_check", "input_sha256": {self.bound.name: runner.sha256_file(self.bound)}}
        with patch.object(runner, "validate", return_value=preflight), patch.object(runner, "runtime_settings"), patch.object(runner, "LiberoFeaturePack") as pack, patch.object(runner, "load_inputs", return_value=(datasets, self.stats, draws, reference, {})), patch.object(runner.base, "initialized_pair", side_effect=initialize), patch.object(runner, "evaluate_arm", side_effect=replay) as evaluations, patch.object(runner, "draw_batch", wraps=runner.draw_batch) as batches, patch.object(runner.training, "gradient_probe", side_effect=probe), patch.object(runner.training, "build_optimizer") as optimizer, patch("torch.Tensor.backward", side_effect=AssertionError("orchestration fixture has no actual backward")), redirect_stdout(io.StringIO()):
            pack.return_value.__enter__.return_value.manifest = {"task_registry": REGISTRY}
            report = runner.gradient_check(self.root, self.plan_sha, True)
        optimizer.assert_not_called()
        self.assertEqual(seen, list(runner.base.ARMS) * 3)
        self.assertEqual(evaluations.call_count, 12)
        self.assertEqual(events, [(seed, arm, operation, part) for seed in runner.base.SEEDS for arm in runner.base.ARMS
                                 for operation, part in (("evaluate", "train"), ("evaluate", "validation"), ("probe", "train_first16"))])
        self.assertEqual([call.args[2] for call in batches.call_args_list], [1, 1, 1])
        self.assertEqual(set(report["results"]), set(map(str, runner.base.SEEDS)))
        self.assertEqual(report["public_optimizer_steps"], 0)
        self.assertFalse(report["optimizer_constructed"])
        for seed in runner.base.SEEDS:
            for row in report["results"][str(seed)].values():
                self.assertEqual(row["window_indices"], [draw["window_index"] for draw in self.manifest["draws"][:16]])
                for part in ("train", "validation"):
                    replayed = row["frozen_evaluation_replay"][part]
                    self.assertEqual((replayed["windows"], replayed["batches"]), (4, 2))
                    self.assertTrue(replayed["prediction_and_statistics_match"])

    def test_gradient_replay_statistics_mismatch_blocks_probe_and_optimizer(self):
        (self.root / "simulation_output").mkdir(exist_ok=True)
        seed, arm = runner.base.SEEDS[0], runner.base.ARMS[0]
        model = FixturePredictor()
        expected = {"per_episode": {}, "episode_macro": {}, "micro": {"window_count": 4}}
        reference = {"runs": {str(seed): {"train": {
            "parameter_sha256": {arm: runner.base.parameter_hash(model)}, "arms": {arm: expected}}}}}
        wrong = {"per_episode": {}, "episode_macro": {}, "micro": {"window_count": 3},
                 "forward_and_scoring_seconds": .1}
        with patch.object(runner, "validate", return_value={"stage": "gradient_check", "input_sha256": {}}), patch.object(runner, "runtime_settings"), patch.object(runner, "LiberoFeaturePack") as pack, patch.object(runner, "load_inputs", return_value=({"train": self.dataset}, self.stats, {seed: self.manifest}, reference, {})), patch.object(runner.base, "initialized_pair", return_value={arm: model}), patch.object(runner, "evaluate_arm", return_value=wrong), patch.object(runner.training, "gradient_probe") as probe, patch.object(runner.training, "build_optimizer") as optimizer, redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "step0 statistics differ"):
            pack.return_value.__enter__.return_value.manifest = {"task_registry": REGISTRY}
            runner.gradient_check(self.root, self.plan_sha, True)
        probe.assert_not_called()
        optimizer.assert_not_called()
        failure = runner.read_json(self.root / runner.GRAD_ROOT / "failure.json")
        self.assertEqual(failure["public_optimizer_steps"], 0)
        self.assertFalse((self.root / runner.GRAD_ROOT / "report.json").exists())

    def test_rehash_detects_late_input_mutation(self):
        pins = {self.bound.name: runner.sha256_file(self.bound)}
        runner.rehash(self.root, pins)
        self.bound.write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            runner.rehash(self.root, pins)

    def test_cli_stage_flags_are_mutually_exclusive(self):
        for args in (("--stage", "train", "--execute-gradient-check"),
                     ("--stage", "gradient_check", "--execute-training"),
                     ("--stage", "gradient_check", "--authorization-sha256", "2"*64),
                     ("--stage", "train", "--execute-training", "--execute-gradient-check")):
            with self.subTest(args=args), patch.object(runner, "train") as train, patch.object(runner, "gradient_check") as probe, self.assertRaises(ValueError):
                runner.main([*args, "--plan-sha256", self.plan_sha])
            train.assert_not_called()
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
