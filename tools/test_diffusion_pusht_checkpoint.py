"""Lightweight fail-closed gate tests; no LeRobot, dataset, or large weight load."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from diffusion_pusht_checkpoint import (PINS, STAT_KEYS, check_saved_config, require,
                                       sha256, split_legacy, tensor_signature, verify_files, write_json)


class CheckpointGateTests(unittest.TestCase):
    def test_pinned_resource(self):
        self.assertEqual(len(PINS), 4)
        self.assertEqual(sum(x[0] for x in PINS.values()), 1050872795)
        self.assertEqual(len(STAT_KEYS), 8)
        self.assertEqual(len(set(STAT_KEYS.values())), 6)

    def test_exact_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a"
            p.write_bytes(b"abc")
            pins = {"a": (3, hashlib.sha256(b"abc").hexdigest())}
            self.assertEqual(verify_files(d, pins)["a"]["sha256"], sha256(p))
            p.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_files(d, pins)

    def test_missing_size_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            for pins in ({"a": (0, "wrong")}, {"../a": (0, "wrong")}):
                with self.assertRaises(ValueError):
                    verify_files(d, pins)

    def test_never_overwrite_reports(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            write_json(p, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                write_json(p, {"status": "completed"})
            self.assertEqual(json.loads(p.read_text())["status"], "failed")

    def test_preserve_configuration(self):
        raw = {"device": "cuda", "crop_shape": [84, 84], "use_amp": False}
        good = {**raw, "device": "cpu", "new_library_default": None}
        check_saved_config(raw, good)
        for altered in ({**good, "crop_shape": None}, {**good, "use_amp": True}, raw):
            with self.assertRaises(ValueError):
                check_saved_config(raw, altered)

    def test_require_not_python_assert(self):
        with self.assertRaises(ValueError):
            require(False, "closed")


class TinyTensorTests(unittest.TestCase):
    def state(self):
        import torch
        state = {f"diffusion.test_{i}": torch.zeros(1) for i in range(212)}
        state["diffusion.rgb_encoder.pool.pos_grid"] = torch.zeros(9, 2)
        for key, (feature, kind) in STAT_KEYS.items():
            shape = (3, 1, 1) if feature == "observation.image" else (2,)
            state[key] = torch.full(shape, 1.0 if kind in ("max", "std") else 0.0)
        return state

    def test_valid_split(self):
        model, stats = split_legacy(self.state())
        self.assertEqual(len(model), 213)
        self.assertEqual(sum(len(v) for v in stats.values()), 6)

    def test_duplicate_disagreement(self):
        state = self.state()
        state["unnormalize_outputs.buffer_action.min"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "disagree"):
            split_legacy(state)

    def test_duplicate_signed_zero_is_byte_disagreement(self):
        state = self.state()
        state["unnormalize_outputs.buffer_action.min"][0] = -0.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            split_legacy(state)

    def test_missing_extra_and_count(self):
        for kind in ("missing", "extra", "model_count"):
            state = self.state()
            if kind == "missing":
                state.pop("normalize_inputs.buffer_observation_image.std")
            elif kind == "extra":
                state["unknown.weight"] = state["diffusion.test_0"]
            else:
                state.pop("diffusion.test_0")
            with self.assertRaises(ValueError):
                split_legacy(state)

    def test_invalid_statistics(self):
        for key, value in (("normalize_inputs.buffer_observation_image.std", 0),
                           ("normalize_inputs.buffer_observation_state.max", -1),
                           ("normalize_inputs.buffer_observation_image.mean", float("nan"))):
            state = self.state()
            state[key].fill_(value)
            with self.assertRaises(ValueError):
                split_legacy(state)

    def test_invalid_shapes(self):
        import torch
        state = self.state()
        state["diffusion.rgb_encoder.pool.pos_grid"] = torch.zeros(144, 2)
        with self.assertRaisesRegex(ValueError, "grid"):
            split_legacy(state)
        state = self.state()
        state["normalize_inputs.buffer_observation_image.std"] = torch.ones(3)
        with self.assertRaisesRegex(ValueError, "shape"):
            split_legacy(state)

    def test_dtype_and_nonfinite(self):
        import torch
        for tensor in (torch.zeros(1, dtype=torch.float64), torch.tensor([float("inf")])):
            with self.assertRaises(ValueError):
                tensor_signature(tensor)


if __name__ == "__main__":
    unittest.main()
