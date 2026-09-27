"""Tiny local serialization tests: no backward, optimizer steps, or dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

try:
    from . import pusht_bc_act_checkpoint as checkpoint
except ImportError:
    import pusht_bc_act_checkpoint as checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("integer_scalar", torch.tensor(3, dtype=torch.int64))
        self.register_buffer("float_scalar", torch.tensor(-0.0))
        self.register_buffer("bfloat_scalar", torch.tensor(0.5, dtype=torch.bfloat16))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.model = Tiny()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        # Populate a realistic resumable state without running optimization.
        for parameter in self.model.parameters():
            self.optimizer.state[parameter] = {
                "step": torch.tensor(7.0), "exp_avg": torch.full_like(parameter, 0.125),
                "exp_avg_sq": torch.full_like(parameter, 0.25),
            }
        self.binding = {"run": "tiny", "seed": 123, "split": [1, 2], "training": False}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, step=7, extra=None):
        return checkpoint.save_checkpoint(self.root, step, self.model, self.optimizer, self.binding, extra)

    def test_roundtrip_model_buffers_optimizer_extra_and_rng(self):
        random.seed(234)
        np.random.seed(234)
        torch.manual_seed(234)
        generator = torch.Generator().manual_seed(987)
        extra = {"sampler_generator_state": generator.get_state(), "samples_seen": 14}
        before = checkpoint.state_dict_hash(self.model)
        path = self.save(extra=extra)
        expected_random = (random.random(), np.random.rand(), torch.rand(4))
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(100)
            self.model.integer_scalar.add_(10)
        for state in self.optimizer.state.values():
            state["exp_avg"].zero_()
        random.seed(555)
        np.random.seed(555)
        torch.manual_seed(555)
        payload = checkpoint.load_checkpoint(path, self.model, self.optimizer, self.binding, "cpu")
        self.assertEqual(checkpoint.state_dict_hash(self.model), before)
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["extra"]["samples_seen"], 14)
        self.assertTrue(torch.equal(payload["extra"]["sampler_generator_state"], extra["sampler_generator_state"]))
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        self.assertTrue(torch.equal(torch.rand(4), expected_random[2]))
        for state in self.optimizer.state.values():
            self.assertEqual(state["step"].item(), 7)
            self.assertTrue(torch.equal(state["exp_avg"], torch.full_like(state["exp_avg"], 0.125)))
            self.assertTrue(torch.equal(state["exp_avg_sq"], torch.full_like(state["exp_avg_sq"], 0.25)))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_corruption_rejected_before_deserialization(self):
        path = self.save()
        data = path / "checkpoint.pt"
        with data.open("r+b") as stream:
            stream.seek(100)
            old = stream.read(1)
            stream.seek(100)
            stream.write(bytes([old[0] ^ 1]))
        with patch.object(checkpoint.torch, "load", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                checkpoint.load_checkpoint(path, self.model, self.optimizer, self.binding, "cpu")

    def test_binding_drift_rejected_before_deserialization(self):
        path = self.save()
        with patch.object(checkpoint.torch, "load", side_effect=AssertionError("must not deserialize")):
            for wrong in ({**self.binding, "seed": 124}, {**self.binding, "seed": 123.0},
                          {**self.binding, "training": 0}):
                with self.subTest(binding=wrong), self.assertRaisesRegex(ValueError, "binding"):
                    checkpoint.load_checkpoint(path, self.model, self.optimizer, wrong, "cpu")

    def test_existing_checkpoint_refused_and_unchanged(self):
        path = self.save()
        before = hashlib.sha256((path / "checkpoint.pt").read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(before, hashlib.sha256((path / "checkpoint.pt").read_bytes()).hexdigest())

    def test_existing_partial_refused_and_cannot_load(self):
        partial = self.root / "checkpoints/step_000007.partial"
        partial.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.save()
        with self.assertRaisesRegex(ValueError, "committed"):
            checkpoint.load_checkpoint(partial, self.model, self.optimizer, self.binding, "cpu")
        self.assertTrue(partial.is_dir())

    def test_failed_commit_preserves_complete_partial(self):
        with patch.object(checkpoint, "_commit_directory", side_effect=OSError("simulated interrupted commit")):
            with self.assertRaises(OSError):
                self.save()
        partial = self.root / "checkpoints/step_000007.partial"
        self.assertEqual({p.name for p in partial.iterdir()}, {"manifest.json", "checkpoint.pt"})
        with self.assertRaises(FileExistsError):
            self.save()

    def test_bad_step_and_binding(self):
        for value in (True, False, -1, 1.0, "1", None, 1000000):
            with self.subTest(step=value), self.assertRaises(ValueError):
                self.save(step=value)
        for binding in ({}, {"bad": float("nan")}, {1: "bad"}, {"bad": (1, 2)}):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                checkpoint.save_checkpoint(self.root, 0, self.model, self.optimizer, binding)
        path = self.save(step=0)
        self.assertEqual(path.name, "step_000000")

    def test_path_traversal_and_manifest_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "traversal"):
            checkpoint.save_checkpoint(self.root / ".." / "elsewhere", 7, self.model, self.optimizer, self.binding)
        path = self.save()
        with self.assertRaisesRegex(ValueError, "traversal"):
            checkpoint.load_checkpoint(path / ".." / path.name, self.model, self.optimizer, self.binding, "cpu")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["../outside.pt"] = manifest["files"].pop("checkpoint.pt")
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(checkpoint.torch, "load", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(ValueError, "paths"):
                checkpoint.load_checkpoint(path, self.model, self.optimizer, self.binding, "cpu")

    def test_hash_covers_scalar_buffers_and_dtype(self):
        original = checkpoint.state_dict_hash(self.model)
        self.model.integer_scalar.add_(1)
        self.assertNotEqual(original, checkpoint.state_dict_hash(self.model))
        self.model.integer_scalar.sub_(1)
        self.assertEqual(original, checkpoint.state_dict_hash(self.model))
        self.model.float_scalar.fill_(0.0)
        self.assertNotEqual(original, checkpoint.state_dict_hash(self.model))

    def test_model_topology_and_optimizer_mismatch(self):
        path = self.save()
        wrong = nn.Linear(2, 3)
        wrong_optimizer = torch.optim.AdamW(wrong.parameters())
        with self.assertRaisesRegex(ValueError, "parameter order/groups"):
            checkpoint.load_checkpoint(path, wrong, wrong_optimizer, self.binding, "cpu")
        with self.assertRaisesRegex(ValueError, "optimizer class"):
            checkpoint.load_checkpoint(path, self.model, torch.optim.SGD(self.model.parameters(), lr=0.1), self.binding, "cpu")
        with self.assertRaisesRegex(ValueError, "parameters"):
            checkpoint.save_checkpoint(self.root, 8, self.model, wrong_optimizer, self.binding)

    def test_optimizer_parameter_order_is_not_silently_reassigned(self):
        path = self.save()
        reversed_optimizer = torch.optim.AdamW(list(self.model.parameters())[::-1])
        with patch.object(checkpoint.torch, "load", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(ValueError, "parameter order/groups"):
                checkpoint.load_checkpoint(path, self.model, reversed_optimizer, self.binding, "cpu")

    def test_rng_schema_device_count_and_cpu_restore(self):
        state = checkpoint.capture_rng_state()
        wrong = dict(state, cuda_device_count=state["cuda_device_count"] + 1)
        with self.assertRaisesRegex(ValueError, "CUDA device count"):
            checkpoint.restore_rng_state(wrong)
        wrong = dict(state, torch_cpu=torch.ones(3, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "uint8"):
            checkpoint.restore_rng_state(wrong)
        expected = torch.rand(5)
        checkpoint.restore_rng_state(state)
        self.assertTrue(torch.equal(expected, torch.rand(5)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_all_cuda_rng_restore(self):
        saved = checkpoint.capture_rng_state()
        expected = [torch.rand(4, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
        checkpoint.restore_rng_state(saved)
        for index, value in enumerate(expected):
            self.assertTrue(torch.equal(value, torch.rand(4, device=f"cuda:{index}")))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_model_and_optimizer_roundtrip_without_optimization(self):
        path = self.save()
        model = Tiny().to("cuda")
        optimizer = torch.optim.AdamW(model.parameters())
        payload = checkpoint.load_checkpoint(path, model, optimizer, self.binding, "cuda")
        self.assertEqual(payload["step"], 7)
        self.assertEqual(checkpoint.state_dict_hash(model), checkpoint.state_dict_hash(self.model))
        for parameter, state in optimizer.state.items():
            self.assertEqual(state["exp_avg"].device, parameter.device)
            self.assertEqual(state["exp_avg_sq"].device, parameter.device)
            self.assertEqual(state["step"].device.type, "cpu")
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
