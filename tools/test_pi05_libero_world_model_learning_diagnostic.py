"""Synthetic authorization, evaluation-mode and startup guards; no real training."""
import argparse
from contextlib import redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_pi05_libero_world_model_learning_diagnostic as runner
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig

DOCS = Path(__file__).resolve().parents[1] / "docs"
AUTH = DOCS / "libero-native-world-model-learning-authorization-v1.json"
PROTOCOL = DOCS / "libero-native-world-model-learning-protocol-v1.json"
REGISTRY = [{"task_id": 9, "source_task_index": 39, "task_instruction": "synthetic native task"}]


def argv(out):
    return ["--feature-pack", "missing", "--random-reference", "missing", "--persistence-reference", "missing",
            "--protocol", str(PROTOCOL), "--protocol-sha256", runner.PROTOCOL_SHA,
            "--authorization", str(AUTH), "--authorization-sha256", runner.AUTHORIZATION_SHA,
            "--out", str(out), "--execute-authorized-diagnostic"]


def evaluation_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(217)
        model = LiberoWorldModel(LiberoWorldModelConfig(hidden_dim=4), REGISTRY)
    inputs = {"history_visual_latent": torch.ones(2, 4, 2, 2048),
              "history_visual_valid": torch.ones(2, 4, 2, dtype=torch.bool),
              "history_state": torch.zeros(2, 4, 8), "history_state_valid": torch.ones(2, 4, 8, dtype=torch.bool),
              "task_instruction": ["synthetic native task"] * 2, "candidate_actions": torch.zeros(2, 1, 3, 7)}
    targets = {"future_visual_latent": torch.ones(2, 3, 2, 2048),
               "future_visual_valid": torch.ones(2, 3, 2, dtype=torch.bool),
               "state_delta": torch.zeros(2, 3, 8), "state_target_valid": torch.ones(2, 3, 8, dtype=torch.bool)}
    return model, {"train": [(0, inputs, targets)]}, {"state_std": [1.] * 8}


class AuthorizationTests(unittest.TestCase):
    def test_user_authorization_binds_unchanged_design_and_does_not_promote_data(self):
        auth = runner.load_authorization(AUTH, runner.AUTHORIZATION_SHA, execute=True)
        protocol = runner.load_protocol(PROTOCOL, runner.PROTOCOL_SHA)
        self.assertTrue(auth["execution_authorized"])
        self.assertFalse(protocol["future_learning_diagnostic"]["execution_authorized"])
        self.assertEqual(auth["optimizer_updates"], 200)
        self.assertFalse(auth["data_training_ready_promoted"])
        self.assertEqual(auth["design_protocol_sha256"], runner.PROTOCOL_SHA)

    def test_flag_exact_bool_and_hash_gate(self):
        for flag in (False, 1, "true", None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                runner.load_authorization(AUTH, runner.AUTHORIZATION_SHA, execute=flag)
        with self.assertRaises(ValueError):
            runner.load_authorization(AUTH, "0"*64, execute=True)
        with patch.object(runner, "sha256_file", return_value="0"*64), self.assertRaises(ValueError):
            runner.load_authorization(AUTH, runner.AUTHORIZATION_SHA, execute=True)

    def test_missing_flag_or_invalid_output_precedes_source_read(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "validate_artifacts") as source:
            bad = Path(temp) / "other"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(argv(bad)[:-1])
            with self.assertRaises(ValueError):
                runner.main(argv(bad))
            self.assertFalse(bad.exists())
            source.assert_not_called()

    def test_existing_output_not_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "run") as execute:
            out = Path(temp) / "pi05_libero_learning_diagnostic_remote_v1"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                runner.main(argv(out))
            execute.assert_not_called()

    def test_source_failure_precedes_runtime_and_optimizer(self):
        args = argparse.Namespace(authorization=AUTH, authorization_sha256=runner.AUTHORIZATION_SHA,
            execute_authorized_diagnostic=True, protocol=PROTOCOL, protocol_sha256=runner.PROTOCOL_SHA,
            out=Path("pi05_libero_learning_diagnostic_remote_v1"), feature_pack=Path("missing"))
        with patch.object(runner, "validate_artifacts", side_effect=ValueError("source failed")), \
             patch.object(runner, "runtime_gate") as runtime:
            with self.assertRaisesRegex(ValueError, "source failed"):
                runner.run(args, {})
            runtime.assert_not_called()


class EvaluationGuardTests(unittest.TestCase):
    def test_inference_and_grad_inputs_rejected_including_bool_masks(self):
        runner.ordinary_tensors({"value": torch.zeros(2), "text": ["task"]})
        with torch.inference_mode():
            mask = torch.ones(2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            runner.ordinary_tensors({"mask": mask})
        with self.assertRaises(ValueError):
            runner.ordinary_tensors({"value": torch.zeros(2, requires_grad=True)})

    def test_numeric_gate_fixed_tolerance_counts_and_keys(self):
        value = {"metric": 1., "count": 3, "axis": [0.2, None], "flag": False}
        runner.numeric_tree_close(value, deepcopy(value))
        runner.numeric_tree_close({**value, "metric": 1. + 1e-11}, value)
        for changed in ({**value, "metric": 1.1}, {**value, "count": 3.0},
                        {**value, "flag": 0}, {**value, "extra": 1}, {**value, "metric": float("nan")}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.numeric_tree_close(changed, value)

    def test_evaluation_restores_modes_requires_grad_and_parameters(self):
        model, batches, stats = evaluation_fixture()
        model.dropout.eval()  # Mixed incoming mode must also be restored.
        modes = [m.training for m in model.modules()]
        before = runner.parameter_sha(model)
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "COUNTS", {"train": 2}):
            result = runner.evaluate(model, batches, stats, Path(temp), 0)
        self.assertEqual(result["train"]["metrics"]["window_count"], 2)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(before, runner.parameter_sha(model))
        self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
        for _, inputs, targets in batches["train"]:
            runner.ordinary_tensors(inputs)
            runner.ordinary_tensors(targets)

    def test_step0_hash_mismatch_fails_even_if_shapes_finite(self):
        model, batches, stats = evaluation_fixture()
        modes = [m.training for m in model.modules()]
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "COUNTS", {"train": 2}):
            with self.assertRaisesRegex(ValueError, "step0 prediction SHA differs"):
                runner.evaluate(model, batches, stats, Path(temp), 0,
                    reference_traces={("train", 0): {"predictions": {"pred_state_delta": "0"*64}}})
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_eval_forward_uses_frozen_parameters_and_matches_frozen_reference(self):
        model, batches, stats = evaluation_fixture()
        inputs = batches["train"][0][1]
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            reference = {k: runner.tensor_sha(v) for k, v in model(**inputs).items()}
        model.train().requires_grad_(True)
        seen = []
        def check_frozen(module, args):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            self.assertTrue(all(not m.training for m in module.modules()))
            self.assertTrue(torch.is_inference_mode_enabled())
            seen.append(True)
        hook = model.register_forward_pre_hook(check_frozen)
        try:
            with tempfile.TemporaryDirectory() as temp, patch.object(runner, "COUNTS", {"train": 2}):
                runner.evaluate(model, batches, stats, Path(temp), 0,
                    reference_traces={("train", 0): {"predictions": reference}})
        finally:
            hook.remove()
        self.assertEqual(seen, [True])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(m.training for m in model.modules()))

    def test_mixed_gradient_flags_and_existing_grad_storage_restored_on_success_and_error(self):
        for fail in (False, True):
            model, batches, stats = evaluation_fixture()
            parameters = list(model.parameters())
            parameters[0].requires_grad_(False)
            parameters[1].grad = torch.ones_like(parameters[1])
            grad_storage = parameters[1].grad
            flags = [p.requires_grad for p in parameters]
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp, \
                 patch.object(runner, "COUNTS", {"train": 2}):
                if fail:
                    with self.assertRaisesRegex(ValueError, "step0 prediction SHA differs"):
                        runner.evaluate(model, batches, stats, Path(temp), 0,
                            reference_traces={("train", 0): {"predictions": {}}})
                else:
                    runner.evaluate(model, batches, stats, Path(temp), 0)
            self.assertEqual(flags, [p.requires_grad for p in parameters])
            self.assertIs(parameters[1].grad, grad_storage)
            self.assertTrue(torch.equal(grad_storage, torch.ones_like(grad_storage)))

    def test_final_evaluation_writes_audit_arrays_without_training(self):
        model, batches, stats = evaluation_fixture()
        model.eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "COUNTS", {"train": 2}):
            runner.evaluate(model, batches, stats, Path(temp), 200)
            self.assertTrue((Path(temp) / "final_train_audit.npz").is_file())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
