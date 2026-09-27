"""Synthetic arrays and stub encoder only; no LeRobot/GPU/source extraction."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_action_study_low_contrast_features as helper
from pi05_libero_world_model_adapter import ARRAY_NAMES, DEMO_REPO, DEMO_REVISION, LAYOUT, SCHEMA
from vla_benchmark_contract import VisualCorruption, apply_visual_corruption


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def fake_latent(array):
    return np.repeat(array[:, 0, 0, 0, None].astype(np.float32) / 255, 2048, axis=1)


class StubEncoder:
    calls = []
    active = 0
    fail_condition = None
    def __init__(self, policy, out):
        self.policy, self.out = policy, out
        self.count = 0
    def __enter__(self):
        if StubEncoder.active:
            raise AssertionError("TraceEncoder hooks must never nest")
        StubEncoder.active += 1
        self.stream = (self.out / "extraction_trace.jsonl").open("x", encoding="utf-8")
        return self
    def __exit__(self, *_):
        self.stream.close()
        StubEncoder.active -= 1
    def row(self, image_array, *, episode_index, frame_index, preview):
        condition = self.out.name
        if condition == self.fail_condition:
            raise ValueError("synthetic encoder failure")
        self.calls.append((condition, episode_index, frame_index, preview, image_array.copy()))
        self.count += 1
        value = fake_latent(image_array)
        self.stream.write(json.dumps({"episode_index": episode_index, "frame_index": frame_index,
                                      "latent_sha256": helper._array_sha(value)}) + "\n")
        self.stream.flush()
        if preview:
            from PIL import Image
            for view in range(2):
                for stage in ("source", "processed"):
                    Image.fromarray(image_array[view]).save(self.out / f"episode{episode_index}_frame{frame_index:04d}_view{view}_{stage}.png")
        return value
    def evidence(self, rows):
        if self.count != rows:
            raise AssertionError("stub row accounting")
        return {"complete_rows_extracted": rows, "preprocessing_call_count": rows,
                "real_view_embedding_call_count": rows * 2, "all_calls_inference_mode": True,
                "stub_only_not_real_encoder_evidence": True}


class LowContrastFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.out, self.clean_root, self.raw_root = [self.base / name for name in ("out", "cache", "raw")]
        for path in (self.out, self.clean_root, self.raw_root):
            path.mkdir()
        # Preserve12 actual IDs/partitions but use2 rows each in explicit fixtures.
        self.fixed = tuple({**e, "record_count": 2} for e in helper.FIXED_EPISODES)
        self.vals = tuple((eid, 2) for eid, _ in helper.VALIDATION_EPISODES)
        self.source = {"root": self.raw_root, "report_sha256": "a" * 64, "episodes": [],
                       "verified_files": {"fixture-only.json": "a" * 64}}
        self.plan = {"episodes": list(deepcopy(self.fixed)), "source_report_sha256": "a" * 64,
                     "expected": {"rows": 24}, "selected_split": {
                         "validation_episode_indices": [eid for eid, _ in self.vals]}}
        registry = [{"task_id": 9, "source_task_index": 39,
                     "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]
        declarations, pixels = [], []
        for position, choice in enumerate(self.fixed):
            eid = choice["episode_index"]
            images = np.zeros((2, 2, 256, 256, 3), np.uint8)
            for frame in range(2):
                images[frame, 0, :128] = [180 + position + frame, 70, 25]
                images[frame, 1, :128] = [150 + position + frame, 65, 20]
            images.flags.writeable = False
            declaration = {**deepcopy(choice), "complete_episode": True}
            self.source["episodes"].append({"declaration": declaration, "images": images})
            pixels.extend(images)
            declarations.append({"episode_index": eid, "task_id": 9, "record_count": 2,
                "complete_episode": True, "source_trajectory_id": f"source_{eid}", "leakage_group_id": f"source_group_{eid}"})
        arrays = {"episode_index": np.repeat([e["episode_index"] for e in self.fixed], 2).astype(np.int64),
                  "frame_index": np.tile([0, 1], 12).astype(np.int64), "task_id": np.full(24, 9, np.int64),
                  "state": np.zeros((24, 8), np.float32), "state_valid": np.ones((24, 8), bool),
                  "action": np.zeros((24, 7), np.float32), "visual_valid": np.ones((24, 2), bool),
                  "visual_latent": np.stack([fake_latent(array) for array in pixels])}
        self.manifest = {"schema": SCHEMA, "benchmark": "libero_spatial",
            "source": {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION, "metadata_sha256": "a" * 64},
            "extractor": {"kind": "frozen_pi05_native_multiview_v1", "checkpoint_sha256": "b" * 64,
                          "preprocessing_sha256": "c" * 64, "code_sha256": "d" * 64},
            "layout": deepcopy(LAYOUT), "fps": 10.0, "task_registry": registry, "episodes": declarations, "arrays": {}}
        self.assertEqual(set(arrays), ARRAY_NAMES)
        for name, array in arrays.items():
            path = self.clean_root / (name + ".npy")
            np.save(path, array, allow_pickle=False)
            self.manifest["arrays"][name] = {"path": path.name, "sha256": helper.sha256_file(path)}
        self._rebind_clean()
        runtime_path = self.base / "runtime.txt"
        runtime_path.write_text("synthetic runtime evidence", encoding="utf-8")
        self.runtime = {"checkpoint": "synthetic", "protocol": "synthetic", "versions": {"fixture": "only"},
                        "files_sha256": {str(runtime_path): helper.sha256_file(runtime_path)}}
        self.checkpoint = {"model_sha256": "b" * 64}
        self.policy = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        StubEncoder.calls, StubEncoder.active, StubEncoder.fail_condition = [], 0, None
        self.patches = [patch.object(helper, "FIXED_EPISODES", self.fixed), patch.object(helper, "VALIDATION_EPISODES", self.vals),
                        patch.object(helper, "CLEAN_REPORT_SHA256", self.report_sha), patch.object(helper, "CLEAN_MANIFEST_SHA256", self.manifest_sha),
                        patch.object(helper, "TraceEncoder", StubEncoder), patch.object(helper, "source_unchanged", return_value=True),
                        patch.object(helper, "runtime_snapshot", side_effect=lambda plan: deepcopy(self.runtime)),
                        patch.object(helper, "check_loaded_checkpoint", return_value=None)]
        self.mocks = [p.start() for p in self.patches]

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def _rebind_clean(self):
        for name, entry in self.manifest["arrays"].items():
            entry["sha256"] = helper.sha256_file(self.clean_root / entry["path"])
        write(self.clean_root / "manifest.json", self.manifest)
        self.manifest_sha = helper.sha256_file(self.clean_root / "manifest.json")
        report = {"status": "passed_frozen_feature_cache_only", "source_report_sha256": "a" * 64,
                  "output_sha256": {path.name: helper.sha256_file(path) for path in self.clean_root.iterdir() if path.name != "report.json"}}
        write(self.clean_root / "report.json", report)
        self.report_sha = helper.sha256_file(self.clean_root / "report.json")

    def run_helper(self, progress=None):
        return helper.extract_validation_features(self.source, self.policy, self.checkpoint, self.plan,
                                                  self.runtime, self.out, self.clean_root, progress)

    def test_exact_validation_only_serial_replay_mapping_and_frozen_no_training(self):
        write(self.out / "preflight.json", {"fixture": True})
        write(self.out / "started.json", {"fixture": True})
        progress = []
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer")):
            report = self.run_helper(lambda *args: progress.append(args))
        indices = np.load(self.out / "row_indices.npy", allow_pickle=False)
        latents = np.load(self.out / "degraded_visual_latent.npy", allow_pickle=False)
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(indices.tolist(), list(range(16, 24)))
        self.assertEqual(latents.shape, (8, 2, 2048))
        self.assertEqual(latents.dtype, np.float32)
        self.assertEqual(report["training_rows_encoded"], 0)
        self.assertEqual(report["rows_per_condition"], 8)
        self.assertEqual(report["total_encoded_rows"], 16)
        self.assertEqual(report["total_real_view_embeddings"], 32)
        self.assertEqual([call[0] for call in StubEncoder.calls], ["clean"] * 8 + ["low_contrast"] * 8)
        self.assertEqual(set(call[1] for call in StubEncoder.calls), {eid for eid, _ in self.vals})
        self.assertEqual(len(list(self.out.rglob("*.png"))), 64)
        self.assertEqual([p[3] for p in progress], ["clean"] * 4 + ["low_contrast"] * 4)
        self.assertTrue(report["clean_replay_matches_cached_latent_bytes"])
        self.assertEqual(report["source_before"], report["source_after"])
        self.assertEqual(report["runtime_before"], report["runtime_after"])
        self.assertEqual(report["rgb_viewframes_changed"], 16)
        self.assertEqual(report["latent_rows_changed"], 8)
        self.assertFalse(report["target_tensors_modified"])
        self.assertFalse(report["normalization_fitted"])
        self.assertFalse(report["feature_pack_created"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(len(list(self.out.glob("*.npy"))), 2)
        self.assertTrue(all(not module.training for module in self.policy.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.policy.parameters()))
        self.assertEqual(StubEncoder.active, 0)
        self.assertEqual(self.mocks[5].call_count, 2)  # old source_unchanged
        self.assertEqual(self.mocks[6].call_count, 2)  # old runtime_snapshot
        self.assertEqual(self.mocks[7].call_count, 2)  # old loaded-checkpoint guard
        json.dumps(report, allow_nan=False)

    def test_corruption_precedes_encoder_and_reuses_exact_wrapper_recipe(self):
        self.run_helper()
        traces = [json.loads(line) for line in (self.out / "corruption_trace.jsonl").read_text().splitlines()]
        for position, record in enumerate(traces):
            clean = StubEncoder.calls[position][4]
            actual = StubEncoder.calls[position + 8][4]
            for view, key in enumerate(helper.IMAGE_KEYS):
                expected = apply_visual_corruption(clean[view], sample_key=record["sample_key"], view_key=key,
                                                  corruption=VisualCorruption("low_contrast", 2), seed=20260911)
                self.assertTrue(np.array_equal(expected, actual[view]))
                self.assertEqual(record["views"][view]["source_rgb_sha256"], helper._array_sha(clean[view]))
                self.assertEqual(record["views"][view]["degraded_rgb_sha256"], helper._array_sha(actual[view]))
            self.assertEqual(record["original_cache_row_index"], 16 + position)

    def test_clean_replay_one_float_bit_drift_fails_without_noisy_encoding(self):
        original = StubEncoder.row
        def mismatch(owner, *args, **kwargs):
            result = original(owner, *args, **kwargs)
            result[0, 0] = np.nextafter(result[0, 0], np.float32(1))
            return result
        with patch.object(StubEncoder, "row", mismatch), self.assertRaisesRegex(ValueError, "byte-match"):
            self.run_helper()
        self.assertFalse((self.out / "low_contrast").exists())
        self.assertFalse((self.out / "degraded_visual_latent.npy").exists())
        self.assertEqual(StubEncoder.active, 0)

    def test_signed_zero_difference_also_fails_exact_replay(self):
        original = StubEncoder.row
        path = self.clean_root / "visual_latent.npy"
        values = np.load(path)
        values[16, 0, 1] = np.float32(-0.)
        np.save(path, values, allow_pickle=False)
        self._rebind_clean()
        def changed(owner, *args, **kwargs):
            result = original(owner, *args, **kwargs)
            result[0, 1] = np.float32(0.)
            return result
        with patch.object(helper, "CLEAN_REPORT_SHA256", self.report_sha), patch.object(helper, "CLEAN_MANIFEST_SHA256", self.manifest_sha), patch.object(StubEncoder, "row", changed):
            with self.assertRaisesRegex(ValueError, "byte-match"):
                self.run_helper()

    def test_constant_rgb_can_have_zero_changed_counts(self):
        for episode in self.source["episodes"]:
            array = np.zeros_like(episode["images"])
            array.flags.writeable = False
            episode["images"] = array
        np.save(self.clean_root / "visual_latent.npy", np.zeros((24, 2, 2048), np.float32), allow_pickle=False)
        self._rebind_clean()
        with patch.object(helper, "CLEAN_REPORT_SHA256", self.report_sha), patch.object(helper, "CLEAN_MANIFEST_SHA256", self.manifest_sha):
            report = self.run_helper()
        self.assertEqual(report["rgb_viewframes_changed"], 0)
        self.assertEqual(report["latent_rows_changed"], 0)

    def test_frozen_split_and_source_order_guards(self):
        self.source["episodes"] = list(reversed(self.source["episodes"]))
        with self.assertRaisesRegex(ValueError, "declaration order"):
            self.run_helper()
        self.assertFalse(StubEncoder.calls)

    def test_corruption_shape_dtype_and_source_mutation_guards(self):
        for defect in ("shape", "dtype", "mutate"):
            destination = self.base / defect
            destination.mkdir()
            def bad(image, **kwargs):
                if defect == "shape":
                    return image[:5]
                if defect == "dtype":
                    return image.astype(np.float32)
                image[:] = 0
                return image.copy()
            with self.subTest(defect=defect), patch.object(helper, "apply_visual_corruption", side_effect=bad):
                with self.assertRaisesRegex(ValueError, "shape/dtype|mutated clean"):
                    helper.extract_validation_features(self.source, self.policy, self.checkpoint, self.plan,
                                                       self.runtime, destination, self.clean_root)
            self.assertFalse((destination / "degraded_visual_latent.npy").exists())

    def test_source_or_runtime_postcheck_failure_keeps_partial_trace_not_arrays(self):
        for target in ("source", "runtime"):
            destination = self.base / target
            destination.mkdir()
            patcher = patch.object(helper, "source_unchanged", side_effect=[True, ValueError("source changed")]) if target == "source" else patch.object(helper, "runtime_snapshot", side_effect=[deepcopy(self.runtime), ValueError("runtime changed")])
            with self.subTest(target=target), patcher, self.assertRaisesRegex(ValueError, "changed"):
                helper.extract_validation_features(self.source, self.policy, self.checkpoint, self.plan,
                                                   self.runtime, destination, self.clean_root)
            self.assertTrue((destination / "corruption_trace.jsonl").exists())
            self.assertFalse((destination / "degraded_visual_latent.npy").exists())

    def test_reject_existing_artifacts_and_source_or_cache_output(self):
        (self.out / "clean").mkdir()
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.run_helper()
        for destination in (self.raw_root, self.clean_root):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "separate fresh"):
                helper.extract_validation_features(self.source, self.policy, self.checkpoint, self.plan,
                                                   self.runtime, destination, self.clean_root)

    def test_clean_cache_pin_mismatch_rejected_before_encoder(self):
        with (self.clean_root / "report.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "report bytes changed"):
            self.run_helper()
        self.assertFalse(StubEncoder.calls)

    def test_encoder_failure_closes_context_and_never_publishes_array(self):
        StubEncoder.fail_condition = "low_contrast"
        with self.assertRaisesRegex(ValueError, "synthetic encoder failure"):
            self.run_helper()
        self.assertEqual(StubEncoder.active, 0)
        self.assertTrue((self.out / "corruption_trace.jsonl").exists())
        self.assertFalse((self.out / "degraded_visual_latent.npy").exists())

    def test_policy_eval_and_parameter_version_guards(self):
        self.policy.train()
        with self.assertRaisesRegex(ValueError, "frozen/eval"):
            self.run_helper()
        self.policy.eval()
        def change(count, eid, frame, condition):
            if condition == "clean":
                with torch.no_grad():
                    next(self.policy.parameters()).add_(0)
        with self.assertRaisesRegex(ValueError, "versions changed"):
            self.run_helper(change)
        self.assertFalse((self.out / "degraded_visual_latent.npy").exists())


if __name__ == "__main__":
    unittest.main()
