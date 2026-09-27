"""Trainer contract tests; no dataset, optimizer update or environment."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_pusht_bc_act import (
    ModelRuntime, anchor_batch, binding_for, json_write, parse_args, protocol, verify_preflight,
)
from tools.pusht_bc_act_models import IMAGE, STATE


class TrainerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_explicit_training_gate(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in [["--stage", "train"], ["--stage", "train", "--model", "bc"],
                         ["--stage", "train", "--execute"], ["--execute"], ["--stop-after", "5"]]:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    parse_args(args)
        self.assertEqual(parse_args([]).stage, "preflight")
        self.assertEqual(parse_args(["--stage", "train", "--model", "act", "--execute"]).model, "act")

    def test_protocol_controls_budget_and_selection(self):
        p = protocol()
        self.assertEqual((p["steps"], p["batch_size"]), (100000, 64))
        self.assertEqual(p["preflight"]["optimizer_steps"], 0)
        self.assertTrue(p["selection"].startswith("final step 100000 only"))
        self.assertEqual(p["closed_loop_evaluation"]["seed_start"], 100000)
        self.assertEqual(p["models"]["act"]["action_horizon"], 16)
        self.assertEqual(p["models"]["bc"]["action_horizon"], 1)

    def test_anchor_sampling_same_across_models_and_resume(self):
        p = protocol()
        indices = np.arange(200, 400)
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        a = anchor_batch(indices, 5000, p)
        torch.rand(31)
        b = anchor_batch(indices, 5000, p)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.isin(a, indices).all())
        self.assertEqual(len(a), 64)
        self.assertFalse(np.array_equal(a, anchor_batch(indices, 5001, p)))
        torch.set_rng_state(before)
        anchor_batch(indices, 0, p)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_sample_step(self):
        for step in (-1, 100000, True, 1.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                anchor_batch([1, 2], step, protocol())

    def test_preflight_drift_and_wrong_stage_refused(self):
        binding = {"version": 1}
        good = {"status": "passed", "stage": "preflight", "optimizer_steps": 0,
                "binding": binding, "model_state_unchanged": True, "reload_prediction_exact": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(good))
            self.assertEqual(len(verify_preflight(path, binding)), 64)
            for key, value in [("status", "failed"), ("stage", "train"), ("optimizer_steps", 1),
                               ("binding", {}), ("model_state_unchanged", False), ("reload_prediction_exact", False)]:
                path.write_text(json.dumps({**good, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_preflight(path, binding)

    def test_runtime_binding_uses_plain_json_version_string(self):
        with patch("importlib.metadata.version", return_value="test"), patch("torch.cuda.get_device_name", return_value="test GPU"):
            binding = binding_for(SimpleNamespace(binding={"synthetic": True}), "bc", protocol())
        self.assertIs(type(binding["torch_runtime"]), str)
        from tools.pusht_bc_act_checkpoint import _binding
        self.assertEqual(_binding(binding), binding)

    def test_exclusive_json_and_atomic_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            json_write(path, {"step": 1}, exclusive=True)
            with self.assertRaises(FileExistsError):
                json_write(path, {"step": 2}, exclusive=True)
            json_write(path, {"step": 3})
            self.assertEqual(json.loads(path.read_text()), {"step": 3})
            self.assertFalse(path.with_name("status.json.tmp").exists())

    def test_real_stats_bc_forward_and_gradient_no_update(self):
        path = Path(__file__).resolve().parents[1] / "simulation_output/pusht_bc_act_data_gate_v1/normalization.json"
        stats = json.loads(path.read_text())
        runtime = ModelRuntime("bc", stats, protocol(), device="cpu")
        batch = {"observation": {IMAGE: torch.rand(2, 3, 96, 96), STATE: torch.tensor([[128., 256.], [384., 64.]])},
                 "action": torch.ones(2, 16, 2) * 256,
                 "action_is_pad": torch.zeros(2, 16, dtype=torch.bool)}
        before = {k: v.clone() for k, v in runtime.policy.state_dict().items()}
        with patch.object(runtime.optimizer, "step", side_effect=AssertionError("optimizer updates forbidden")):
            result = runtime.gradients(batch)
            self.assertGreater(result["gradient_norm_before_clip"], 0)
            pred = runtime.predict(batch)
        self.assertEqual(tuple(pred.shape), (2, 2))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in runtime.policy.state_dict().items()))
        self.assertEqual(len(runtime.optimizer.state), 0)

    def test_validation_prediction_ignores_targets(self):
        stats = {IMAGE: {"mean": [[[.485]], [[.456]], [[.406]]], "std": [[[.229]], [[.224]], [[.225]]]},
                 STATE: {"mean": [256, 256], "std": [128, 128]},
                 "action": {"mean": [256, 256], "std": [128, 128]}}
        runtime = ModelRuntime("bc", stats, protocol(), device="cpu")
        batch = {"observation": {IMAGE: torch.zeros(2, 3, 96, 96), STATE: torch.zeros(2, 2)},
                 "action": torch.zeros(2, 16, 2)}
        expected = runtime.predict(batch)
        batch["action"] = torch.full((2, 16, 2), 512.)
        self.assertTrue(torch.equal(expected, runtime.predict(batch)))

    def test_act_postprocessor_cpu_output_matches_target_device(self):
        class Pipeline:
            def __init__(self, function):
                self.function = function
            def reset(self):
                pass
            def __call__(self, batch):
                return self.function(batch)
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.name = "act"
        runtime.device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime.policy = SimpleNamespace(eval=lambda: None, reset=lambda: None,
            select_action=lambda batch: torch.zeros(2, 2, device=runtime.device))
        runtime.pre = Pipeline(lambda batch: batch)
        runtime.post = Pipeline(lambda action: action.cpu() + 256)
        batch = {"observation": {IMAGE: torch.zeros(2, 3, 96, 96, device=runtime.device),
                                  STATE: torch.zeros(2, 2, device=runtime.device)}}
        prediction = runtime.predict(batch)
        self.assertEqual(prediction.device.type, runtime.device)
        self.assertTrue(torch.equal(prediction.cpu(), torch.full((2, 2), 256.)))


if __name__ == "__main__":
    unittest.main()
