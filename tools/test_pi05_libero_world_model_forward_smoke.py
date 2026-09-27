"""CPU-only entrypoint guards; synthetic tensors are not model-quality data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

import smoke_pi05_libero_world_model as smoke


def report_fixture():
    return {
        "schema": "pi05_libero_feature_pair_build_v1", "status": "passed",
        "source": {"kind": "public_demonstrations", "repo_id": smoke.DEMO_REPO, "revision": smoke.DEMO_REVISION},
        "source_report_sha256": smoke.SOURCE_SHA, "plan_sha256": smoke.PLAN_SHA, "split": deepcopy(smoke.SPLIT),
        "latent_shape": [313, 2, 2048], "state_shape": [313, 8], "action_shape": [313, 7],
        "complete_episode_counts": {"1400": 140, "1402": 173}, "windows": deepcopy(smoke.COUNTS),
        "window_context_len": 4, "window_horizon": 3,
        "normalization_state_rows": 140, "normalization_action_rows": 139,
        "feature_pack_complete_for_fixed_two_episodes": True,
        "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
        "training_ready": False, "training_started": False, "optimizer_steps": 0,
        "actions_generated": 0, "simulation_steps": 0, "source_files_unchanged": True,
        "extraction_evidence": {
            "complete_rows_extracted": 313, "preprocessing_call_count": 313,
            "real_view_embedding_call_count": 626, "empty_camera_embedded": False,
            "all_modules_eval": True, "all_parameters_frozen": True,
            "all_parameter_gradients_absent": True, "parameter_versions_unchanged": True,
            "all_calls_inference_mode": True,
        },
        "checkpoint": {"model_sha256": smoke.CHECKPOINT_SHA, "load": {"loaded_parameter_fraction": 1.0}},
    }


class ReportGuards(unittest.TestCase):
    def test_report_contract_only_not_source_acceptance(self):
        smoke.check_report_contract(report_fixture())

    def test_reject_changed_source_identity(self):
        for change in ({"source_report_sha256": "0" * 64}, {"plan_sha256": "0" * 64},
                       {"source": {"kind": "evaluation_rollouts"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                smoke.check_report_contract({**report_fixture(), **change})

    def test_reject_partial_or_changed_split(self):
        for key, value in (("latent_shape", [7, 2, 2048]), ("windows", {"train": 133, "validation": 167}),
                           ("split", {"train_episode_indices": [1400, 1402], "validation_episode_indices": []}),
                           ("normalization_action_rows", 140), ("state_shape", [313, 32]),
                           ("action_shape", [313, 9])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.check_report_contract({**report_fixture(), key: value})

    def test_reject_promotion_and_nonboolean_flags(self):
        for key, value in (("family_independence_verified", True), ("training_ready", True),
                           ("training_started", 0), ("optimizer_steps", False),
                           ("checkpoint_training_overlap_unknown", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.check_report_contract({**report_fixture(), key: value})

    def test_reject_missing_or_changed_frozen_evidence(self):
        for key in report_fixture()["extraction_evidence"]:
            report = report_fixture()
            del report["extraction_evidence"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.check_report_contract(report)

    def test_reject_incomplete_checkpoint_load(self):
        report = report_fixture()
        report["checkpoint"]["load"]["loaded_parameter_fraction"] = 0.9
        with self.assertRaises(ValueError):
            smoke.check_report_contract(report)

    def test_sha_format_and_external_report_hash(self):
        for value in ("", "f" * 63, "F" * 64, 123, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke.digest_value(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "external SHA"):
                smoke.validate_artifacts(root, "0" * 64)

    def test_artifact_containment_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.json").write_text("{}", encoding="utf-8")
            self.assertEqual(smoke.contained(root, "safe.json"), root / "safe.json")
            for name in ("../report.json", "a/../safe.json", "./safe.json", "missing.json", "a\\b", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    smoke.contained(root, name)


class FakeCausalModel:
    """Small analytic fixture solely tests the checker, not the learned model."""
    def __init__(self, mode="causal"):
        self.mode = mode

    def __call__(self, *, task_instruction, candidate_actions):
        x = candidate_actions[..., 0]
        if self.mode == "future_leak":
            x = x.sum(dim=-1, keepdim=True).expand_as(x)
        elif self.mode == "candidate_mix":
            x = x.mean(dim=1, keepdim=True).expand_as(x)
        elif self.mode == "ignore_actions":
            x = torch.zeros_like(x)
        x = x.cumsum(dim=-1)
        return {"pred_future_visual_latent": x[..., None, None].expand(*x.shape, 2, 2048),
                "pred_state_delta": x[..., None].expand(*x.shape, 8)}


class ForwardGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_synthetic_candidate_wiring_checker(self):
        inputs = {"task_instruction": ["fixture a", "fixture b"], "candidate_actions": torch.zeros(2, 1, 3, 7)}
        before = smoke.input_fingerprints(inputs)
        with torch.inference_mode():
            report = smoke.candidate_wiring_probe(FakeCausalModel(), inputs)
        self.assertEqual(report["future_action_to_earlier_prediction_max_abs_difference"], 0)
        self.assertGreater(report["changed_last_action_to_final_prediction_max_abs_difference"], 0)
        self.assertEqual(before, smoke.input_fingerprints(inputs))

    def test_wiring_checker_rejects_actual_bad_dependencies(self):
        for mode in ("future_leak", "candidate_mix", "ignore_actions"):
            inputs = {"task_instruction": ["fixture"], "candidate_actions": torch.zeros(1, 1, 3, 7)}
            with self.subTest(mode=mode), torch.inference_mode(), self.assertRaises(ValueError):
                smoke.candidate_wiring_probe(FakeCausalModel(mode), inputs)

    def test_prediction_shapes_keys_dtype_grad_and_finite(self):
        outputs = FakeCausalModel()(task_instruction=["fixture"], candidate_actions=torch.zeros(1, 1, 3, 7))
        smoke.prediction_check(outputs, 1, 1)
        bad = [
            {**outputs, "score": torch.zeros(1)},
            {**outputs, "pred_state_delta": torch.zeros(1, 1, 3, 9)},
            {**outputs, "pred_state_delta": torch.zeros(1, 1, 3, 8, dtype=torch.float64)},
            {**outputs, "pred_state_delta": torch.zeros(1, 1, 3, 8, requires_grad=True)},
            {**outputs, "pred_state_delta": torch.full((1, 1, 3, 8), float("nan"))},
        ]
        for index, value in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                smoke.prediction_check(value, 1, 1)

    def test_tensor_digest_binds_shape_dtype_and_value(self):
        value = torch.zeros(2, 3)
        first = smoke.tensor_sha(value)
        self.assertNotEqual(first, smoke.tensor_sha(value.reshape(3, 2)))
        self.assertNotEqual(first, smoke.tensor_sha(value.double()))
        value[0, 0] = 1
        self.assertNotEqual(first, smoke.tensor_sha(value))


if __name__ == "__main__":
    unittest.main()
