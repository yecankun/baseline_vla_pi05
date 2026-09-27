"""CPU-only execution guards and global objective aggregation; no optimization."""
import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_pi05_libero_world_model as runner


class ExecutionTests(unittest.TestCase):
    def test_only_explicit_zero_step_cpu(self):
        runner.validate_execution(dry_run=True, max_steps=0, device="cpu")
        for overrides in ({"dry_run": False}, {"dry_run": 1}, {"max_steps": 1},
                          {"max_steps": -1}, {"max_steps": False}, {"max_steps": 0.0},
                          {"device": "cuda"}, {"device": "cpu:0"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                runner.validate_execution(**({"dry_run": True, "max_steps": 0, "device": "cpu"} | overrides))

    def test_run_rejects_before_artifact_access(self):
        with patch.object(runner, "validate_artifacts") as gate:
            with self.assertRaises(ValueError):
                runner.run(Path("missing"), "0" * 64, Path("unused"))
            gate.assert_not_called()

    def test_source_failure_precedes_model_import(self):
        with patch.object(runner, "validate_artifacts", side_effect=ValueError("bad provenance")):
            with patch("builtins.__import__", wraps=__import__) as imports:
                with self.assertRaisesRegex(ValueError, "bad provenance"):
                    runner.run(Path("missing"), "0" * 64, Path("unused"), dry_run=True)
                self.assertFalse(any(call.args[0] in {"torch", "pi05_libero_world_model",
                                      "pi05_libero_world_model_objectives"} for call in imports.call_args_list))

    def test_cli_invalid_execution_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            base = ["--feature-pack", "missing", "--feature-report-sha256", "0" * 64, "--out", str(out)]
            for args in ([], ["--dry-run", "--max-steps", "1"],
                         ["--dry-run", "--device", "cuda"], ["--dry-run", "--max-steps", "0.0"]):
                with self.subTest(args=args), redirect_stderr(io.StringIO()):
                    with self.assertRaises((ValueError, SystemExit)):
                        runner.main(base + args)
                self.assertFalse(out.exists())

    def test_existing_output_rejected_without_run(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "run") as execute:
            with self.assertRaises(FileExistsError):
                runner.main(["--feature-pack", "missing", "--feature-report-sha256", "0" * 64,
                             "--out", temp, "--dry-run"])
            execute.assert_not_called()

    def test_failed_provenance_writes_failure_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            with patch.object(runner, "validate_artifacts", side_effect=ValueError("bad provenance")):
                with redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
                    runner.main(["--feature-pack", "missing", "--feature-report-sha256", "0" * 64,
                                 "--out", str(out), "--dry-run"])
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertFalse(status["training_started"])
            self.assertEqual(status["optimizer_steps"], 0)
            self.assertFalse((out / "report.json").exists())

    def test_no_training_or_policy_execution_calls(self):
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        forbidden = {"backward", "step", "train", "Adam", "AdamW", "SGD", "load_state_dict",
                     "from_pretrained", "select_action", "save", "candidate_wiring_probe"}
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))}
        self.assertFalse(calls & forbidden)


class TotalsTests(unittest.TestCase):
    def test_global_branches_not_batch_mean(self):
        totals = runner.ObjectiveTotals()
        totals.update({"visual_sum": 8., "visual_count": 8, "state_sum": 12., "state_count": 2})
        totals.update({"visual_sum": 12., "visual_count": 2, "state_sum": 8., "state_count": 8})
        result = totals.summary()
        self.assertEqual(result["branches"]["visual"]["mean"], 2.)
        self.assertEqual(result["branches"]["state"]["mean"], 2.)
        self.assertEqual(result["objective"], 2.5)
        self.assertNotEqual(result["objective"], ((1 + .25 * 6) + (6 + .25 * 1)) / 2)

    def test_unsupported_branch_zero(self):
        totals = runner.ObjectiveTotals()
        totals.update({"visual_sum": 0., "visual_count": 0, "state_sum": 8., "state_count": 2})
        self.assertEqual(totals.summary()["objective"], 1.)
        self.assertEqual(totals.summary()["branches"]["visual"]["mean"], 0.)

    def test_all_weighted_support_missing_rejected(self):
        totals = runner.ObjectiveTotals(visual_weight=1., state_weight=0.)
        totals.update({"visual_sum": 0., "visual_count": 0, "state_sum": 8., "state_count": 2})
        with self.assertRaises(ValueError):
            totals.summary()

    def test_invalid_statistics_transactional(self):
        totals = runner.ObjectiveTotals()
        good = {"visual_sum": 1., "visual_count": 1, "state_sum": 2., "state_count": 1}
        totals.update(good)
        before = totals.summary()
        for change in ({"state_sum": float("nan")}, {"visual_sum": float("inf")},
                       {"state_count": True}, {"state_count": -1}, {"state_count": 0},
                       {"state_sum": -1.}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                totals.update(good | change)
            self.assertEqual(totals.summary(), before)

    def test_invalid_weights(self):
        for values in ((0., 0.), (-1., 1.), (True, 1.), (float("nan"), 1.), (1., float("inf"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.ObjectiveTotals(*values)

    def test_zero_error_has_real_counts(self):
        totals = runner.ObjectiveTotals()
        totals.update({"visual_sum": 0., "visual_count": 16, "state_sum": 0., "state_count": 8})
        result = totals.summary()
        self.assertEqual(result["objective"], 0.)
        self.assertEqual(result["branches"]["visual"]["valid_scalar_count"], 16)


if __name__ == "__main__":
    unittest.main()
