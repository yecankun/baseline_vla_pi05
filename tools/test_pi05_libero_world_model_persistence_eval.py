"""Bounded CPU protocol, reference-match and comparison guards; no training."""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import eval_pi05_libero_world_model_persistence as runner
from test_pi05_libero_world_model_persistence import fixture
from pi05_libero_world_model_persistence import persistence_scoring_targets

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/libero-native-world-model-learning-protocol-v1.json"


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_is_diagnostic_not_authorization(self):
        p = runner.load_protocol(PROTOCOL, runner.PROTOCOL_SHA)
        self.assertEqual(p["current_execution"]["optimizer_steps"], 0)
        self.assertFalse(p["future_learning_diagnostic"]["execution_authorized"])
        self.assertEqual(p["future_learning_diagnostic"]["implementation_status"], "not_implemented")
        self.assertEqual(p["future_learning_diagnostic"]["optimizer_steps"], 200)
        self.assertFalse(p["data"]["training_ready"])

    def test_changed_protocol_or_external_hash_rejected(self):
        with self.assertRaises(ValueError):
            runner.load_protocol(PROTOCOL, "0"*64)
        with patch.object(runner, "sha256_file", return_value="0"*64):
            with self.assertRaises(ValueError):
                runner.load_protocol(PROTOCOL, runner.PROTOCOL_SHA)

    def test_sampler_sequence_and_global_rng_unchanged(self):
        p = runner.load_protocol(PROTOCOL, runner.PROTOCOL_SHA)
        before = torch.get_rng_state().clone()
        self.assertEqual(runner.sampler_plan_hash(p), p["future_learning_diagnostic"]["sampling_plan_sha256"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        p["future_learning_diagnostic"]["sampling_seed"] += 1
        with self.assertRaises(ValueError):
            runner.sampler_plan_hash(p)

    def test_bad_protocol_precedes_artifact_read_and_out_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "validate_artifacts") as gate:
            out = Path(temp) / "out"
            with self.assertRaises(ValueError):
                runner.main(["--feature-pack", "missing", "--random-reference", "missing",
                             "--protocol", str(PROTOCOL), "--protocol-sha256", "0"*64, "--out", str(out)])
            gate.assert_not_called()
            self.assertFalse(out.exists())

    def test_training_and_gpu_cli_options_absent(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(["--max-steps", "1", "--device", "cuda"])
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        names = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                 for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertFalse(names & {"LiberoWorldModel", "backward", "step", "train", "AdamW", "SGD", "from_pretrained", "select_action"})


class ComparisonTests(unittest.TestCase):
    def test_direction_and_relative_change(self):
        result = runner.change(1., 2.)
        self.assertEqual(result["absolute_change"], -1.)
        self.assertEqual(result["relative_change_percent"], -50.)
        self.assertEqual(runner.change(3., 2.)["relative_change_percent"], 50.)

    def test_zero_denominator_and_unsupported_null(self):
        self.assertIsNone(runner.change(1., 0.)["relative_change_percent"])
        self.assertEqual(runner.change(1., 0.)["absolute_change"], 1.)
        self.assertIsNone(runner.change(None, 1.)["absolute_change"])
        self.assertIsNone(runner.change(1., None)["relative_change_percent"])

    def test_nonfinite_negative_bool_errors_rejected(self):
        for value in (float("nan"), float("inf"), -1., True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.change(value, 1.)

    def test_hierarchy_kept_support_mismatch_rejected(self):
        node = {"count": 3, "absolute_error_sum": 3., "squared_error_sum": 3., "mae": 1., "rmse": 1.}
        tree = {"per_horizon": [node]}
        self.assertEqual(runner.compare_stats(tree, tree)["per_horizon"][0]["mae"]["absolute_change"], 0.)
        other = deepcopy(tree)
        other["per_horizon"][0]["count"] = 2
        with self.assertRaisesRegex(ValueError, "valid counts"):
            runner.compare_stats(tree, other)


class ExactSupportTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.targets = fixture()
        self.items = [{"metadata": {"window": i}} for i in range(2)]
        self.ref = {"batch_size": 2, "metadata": [i["metadata"] for i in self.items],
                    "inputs": runner.input_fingerprints(self.inputs),
                    "targets": {k: runner.tensor_sha(v) for k, v in self.targets.items()}}

    def test_exact_complete_support_matches(self):
        common = persistence_scoring_targets(self.inputs, self.targets)
        runner.assert_matching_batch(self.inputs, self.targets, common, self.items, self.ref)

    def test_same_count_different_mask_still_rejected(self):
        self.targets["future_visual_valid"][0, 0, 0] = False
        self.ref["targets"] = {k: runner.tensor_sha(v) for k, v in self.targets.items()}
        common = deepcopy(self.targets)
        common["future_visual_valid"][0, 0, 0] = True
        common["future_visual_valid"][0, 0, 1] = False
        self.assertEqual(int(common["future_visual_valid"].sum()), int(self.targets["future_visual_valid"].sum()))
        with self.assertRaisesRegex(ValueError, "common mask differs"):
            runner.assert_matching_batch(self.inputs, self.targets, common, self.items, self.ref)

    def test_modified_reference_input_target_or_metadata_rejected(self):
        common = persistence_scoring_targets(self.inputs, self.targets)
        for key in ("inputs", "targets", "metadata"):
            reference = deepcopy(self.ref)
            reference[key] = {} if key != "metadata" else []
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reference batch mismatch"):
                runner.assert_matching_batch(self.inputs, self.targets, common, self.items, reference)


if __name__ == "__main__":
    unittest.main()
