"""CPU fake-policy/temporary synthetic fixtures; not public-source evidence."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Existing preparation tool is intentionally a standalone sibling-import CLI.
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import build_pi05_libero_feature_pack as builder
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization


def json_write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def make_backing(root: Path):
    paths = {}
    for episode, count in builder.EPISODE_COUNTS.items():
        path = root / f"backing_{episode}.npy"
        data = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(count, 2, 256, 256, 3))
        data[:] = 0
        for frame in range(count):
            data[frame, 0, :128, :, 0] = (frame + episode) % 255
            data[frame, 1, :, :128, 1] = (frame + episode + 50) % 255
        data.flush()
        data._mmap.close()
        paths[episode] = path
    return paths


def fixture(root: Path, backing: dict):
    # Deliberately fabricated values below exercise contracts only. None is
    # promoted to a durable source audit or checkpoint integration result.
    root.mkdir()
    plan_bytes = (TOOLS.parent / "docs" / "libero-feature-pair-plan-v2.json").read_bytes()
    (root / "plan.json").write_bytes(plan_bytes)
    report = {"schema": builder.SOURCE_SCHEMA, "status": "passed",
              "source": {"kind": "public_demonstrations", "repo_id": builder.DEMO_REPO,
                         "revision": builder.DEMO_REVISION},
              "fps": 10.0, "orientation": builder.ORIENTATION, "row_count": 313,
              "task_registry": deepcopy(builder.TASK_REGISTRY),
              "split": deepcopy(builder.FIXED_SPLIT), "plan_path": "plan.json",
              "plan_sha256": builder.PLAN_SHA256,
              "prior_source_report_sha256": builder.PRIOR_REPORT_SHA256,
              "duplicate_suffix_audit": {"status": "passed", "fixture_only": True},
              "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
              "training_ready": False, "output_sha256": {"plan.json": builder.sha256_file(root / "plan.json")},
              "episodes": []}
    for episode, count in builder.EPISODE_COUNTS.items():
        directory = root / f"episode_{episode}"
        directory.mkdir()
        os.link(backing[episode], directory / "images.npy")
        start = 237603 if episode == 1400 else 237832
        records = []
        for frame in range(count):
            action = [0.125] * 7 if episode == 1400 else [0.875] * 7
            if frame == count - 1:
                action = [-1.0] * 7
            records.append({"episode_index": episode, "frame_index": frame, "task_index": 39,
                            "task": builder.TASK_REGISTRY[0]["task_instruction"],
                            "observation.state": [frame / 16 + (100 if episode == 1402 else 0)] * 8,
                            "action": action, "timestamp": frame / 10, "index": start + frame})
        json_write(directory / "records.json", records)
        declaration = {"episode_index": episode, "task_id": 9, "source_task_index": 39,
                       "record_count": count, "complete_episode": True,
                       "images_path": f"episode_{episode}/images.npy",
                       "records_path": f"episode_{episode}/records.json",
                       "source_trajectory_id": f"{builder.DEMO_REPO}@{builder.DEMO_REVISION}/episode{episode}",
                       "leakage_group_id": f"source_episode_{episode}",
                       "metadata": {"episode_index": episode, "length": count,
                                    "dataset_from_index": start, "dataset_to_index": start + count}}
        for stem in ("images", "records"):
            name = declaration[f"{stem}_path"]
            declaration[f"{stem}_sha256"] = builder.sha256_file(root / name)
            report["output_sha256"][name] = declaration[f"{stem}_sha256"]
        report["episodes"].append(declaration)
    json_write(root / "duplicate_suffix_audit.json", report["duplicate_suffix_audit"])
    report["output_sha256"]["duplicate_suffix_audit.json"] = builder.sha256_file(root / "duplicate_suffix_audit.json")
    json_write(root / "report.json", report)
    return report


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.invalid = False

    def embed_image(self, image):
        value = (image.mean(dim=(1, 2, 3)) * self.weight)[:, None, None].expand(-1, 2, 2048).clone()
        if self.invalid:
            value[0, 0, 0] = float("nan")
        return value


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_features=OrderedDict(
            (key, SimpleNamespace(shape=shape)) for key, shape in [
                (builder.IMAGE_KEYS[0], [3, 256, 256]), (builder.IMAGE_KEYS[1], [3, 256, 256]),
                ("observation.images.empty_camera_0", [3, 224, 224])]), empty_cameras=1)
        self.model = nn.Module()
        self.model.paligemma_with_expert = FakeVision()
        self.bad_empty = False
        self.eval()

    def _preprocess_images(self, batch):
        images = [F.interpolate(batch[key], size=(224, 224), mode="bilinear", align_corners=False) * 2 - 1
                  for key in builder.IMAGE_KEYS]
        images.append(torch.full_like(images[0], -1))
        return images, [torch.tensor([True]), torch.tensor([True]), torch.tensor([self.bad_empty])]


def fake_checkpoint():
    return {"model_sha256": builder.MODEL_SHA256, "compile_model": False,
            "load": {"status": "loaded", "loaded_parameter_fraction": 1.0},
            "installed_implementation_sources": {"image_preprocess": {"sha256": "a" * 64}},
            "test_fixture_only_not_checkpoint_evidence": True}


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.shared = TemporaryDirectory()
        cls.backing = make_backing(Path(cls.shared.name))

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        self.report = fixture(self.root, self.backing)
        self.bound_digest = builder.sha256_file(self.root / "report.json")
        self.sources = []

    def tearDown(self):
        for source in self.sources:
            builder.close_source(source)
        self.temp.cleanup()

    def rebind(self):
        for episode in self.report["episodes"]:
            for stem in ("images", "records"):
                name = episode[f"{stem}_path"]
                path = self.root / name
                if path.is_file():
                    episode[f"{stem}_sha256"] = builder.sha256_file(path)
                    self.report["output_sha256"][name] = episode[f"{stem}_sha256"]
        json_write(self.root / "report.json", self.report)
        self.bound_digest = builder.sha256_file(self.root / "report.json")

    def validate(self):
        source = builder.validate_source(self.root, self.bound_digest)
        self.sources.append(source)
        return source

    def record_change(self, episode, operation):
        path = self.root / f"episode_{episode}" / "records.json"
        records = builder.read_json(path)
        operation(records)
        json_write(path, records)
        self.rebind()

    def test_complete_mmap_source_and_fixed_identity(self):
        source = self.validate()
        self.assertEqual([len(e["adapted"]) for e in source["episodes"]], [140, 173])
        self.assertTrue(all(isinstance(e["images"], np.memmap) for e in source["episodes"]))
        self.assertEqual(source["report"]["split"], builder.FIXED_SPLIT)

    def test_report_external_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "externally supplied"):
            builder.validate_source(self.root, "f" * 64)

    def test_wrong_split_rejected_before_load(self):
        self.report["split"] = {"train_episode_indices": [1402], "validation_episode_indices": [1400]}
        self.rebind()
        with self.assertRaisesRegex(ValueError, "split differs"):
            self.validate()

    def test_same_source_or_leakage_group_rejected(self):
        for key in ("source_trajectory_id", "leakage_group_id"):
            with self.subTest(key=key):
                original = self.report["episodes"][1][key]
                self.report["episodes"][1][key] = self.report["episodes"][0][key]
                self.rebind()
                with self.assertRaisesRegex(ValueError, "crosses fixed split"):
                    self.validate()
                self.report["episodes"][1][key] = original

    def test_duplicate_audit_failure_is_not_bypassed(self):
        self.report["duplicate_suffix_audit"]["status"] = "failed"
        self.rebind()
        with self.assertRaisesRegex(ValueError, "duplicate/suffix"):
            self.validate()

    def test_evaluation_rollouts_rejected(self):
        self.report["source"]["kind"] = "evaluation_rollouts"
        self.rebind()
        with self.assertRaisesRegex(ValueError, "no synthetic/rollout"):
            self.validate()

    def test_task_text_or_mapping_change_rejected(self):
        self.report["task_registry"][0]["source_task_index"] = 9
        self.rebind()
        with self.assertRaisesRegex(ValueError, "source39"):
            self.validate()

    def test_wrong_image_shape_rejected_even_with_matching_hash(self):
        path = self.root / "episode_1402" / "images.npy"
        path.unlink()  # Remove only this temporary hard link, not shared backing.
        np.save(path, np.zeros((173, 2, 3, 256, 256), dtype=np.uint8), allow_pickle=False)
        self.rebind()
        with self.assertRaisesRegex(ValueError, "complete source images"):
            self.validate()

    def test_source_record_truncation_rejected(self):
        self.record_change(1402, lambda records: records.pop())
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.validate()

    def test_false_completeness_and_changed_count_rejected(self):
        self.report["episodes"][0]["complete_episode"] = False
        self.rebind()
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.validate()
        self.report["episodes"][0]["complete_episode"] = True
        self.report["episodes"][0]["record_count"] = 139
        self.rebind()
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.validate()

    def test_unknown_oracle_or_missing_native_state_rejected(self):
        self.record_change(1400, lambda records: records[0].update({"contact_truth": 1}))
        with self.assertRaisesRegex(ValueError, "unknown/missing"):
            self.validate()
        self.record_change(1400, lambda records: records[0].pop("contact_truth"))
        self.record_change(1400, lambda records: records[0].pop("observation.state"))
        with self.assertRaisesRegex(ValueError, "unknown/missing"):
            self.validate()

    def test_timestamp_and_row_gap_rejected(self):
        self.record_change(1400, lambda records: records[1].update({"timestamp": 0.2}))
        with self.assertRaisesRegex(ValueError, "10Hz"):
            self.validate()
        self.record_change(1400, lambda records: records[1].update({"timestamp": 0.1, "frame_index": 2}))
        with self.assertRaisesRegex(ValueError, "gap/reset"):
            self.validate()

    def test_payload_hash_mismatch_rejected(self):
        path = self.root / "episode_1402" / "records.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output hash mismatch"):
            self.validate()

    def test_plan_bytes_are_pinned(self):
        (self.root / "plan.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "plan bytes changed"):
            self.validate()

    def test_native_action_no_clipping(self):
        self.record_change(1400, lambda records: records[0].update({"action": [1.01] * 7}))
        with self.assertRaisesRegex(ValueError, "no clipping"):
            self.validate()

    def test_training_and_independence_caveats_required(self):
        self.report["training_ready"] = True
        self.rebind()
        with self.assertRaisesRegex(ValueError, "boundaries"):
            self.validate()

    def test_complete_fake_build_windows_no_crossing_final_action_exclusion(self):
        source = self.validate()
        output = Path(self.temp.name) / "features"
        output.mkdir()
        policy = FakePolicy()
        report = builder.build_feature_pack(source, policy, fake_checkpoint(), output)
        self.assertEqual(report["latent_shape"], [313, 2, 2048])
        self.assertEqual(report["extraction_evidence"]["preprocessing_call_count"], 313)
        self.assertEqual(report["extraction_evidence"]["real_view_embedding_call_count"], 626)
        self.assertFalse(report["extraction_evidence"]["empty_camera_embedded"])
        self.assertFalse(report["training_ready"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(len(list(output.glob("*.png"))), 16)
        self.assertNotIn("_preprocess_images", policy.__dict__)
        self.assertNotIn("embed_image", policy.model.paligemma_with_expert.__dict__)
        trace = [json.loads(line) for line in (output / "extraction_trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(trace), 313)
        self.assertEqual([(row["episode_index"], row["frame_index"]) for row in trace],
                         [(episode, frame) for episode, count in builder.EPISODE_COUNTS.items() for frame in range(count)])
        self.assertTrue(all(row["preprocessing"]["masks"] == [[True], [True], [False]] for row in trace))
        with LiberoFeaturePack(output) as pack:
            split = pack.load_split(output / "split.json")
            norm = fit_train_normalization(pack, split)
            self.assertEqual(norm["action_record_count"], 139)
            self.assertEqual(norm["state_record_count"], 140)
            self.assertEqual(norm["action_mean"], [0.125] * 7)
            self.assertEqual(norm["state_mean"], [139 / 32] * 8)
            for name, count, episode in (("train", 134, 1400), ("validation", 167, 1402)):
                dataset = LiberoWindowDataset(pack, split, partition=name, normalization=norm)
                self.assertEqual(len(dataset), count)
                for window in dataset.windows:
                    indices = list(window.history + window.actions + window.targets)
                    self.assertTrue(np.all(pack.arrays["episode_index"][indices] == episode))
                    self.assertLess(max(window.actions), int(pack.indices[episode][-1]))
                self.assertEqual(dataset[-1]["inputs"]["candidate_actions"].shape, (1, 3, 7))
        prep = builder.read_json(output / "preparation" / "report.json")
        self.assertEqual(prep["evidence_scope"], "declared_public_feature_pack_schema_only")
        self.assertFalse(prep["training_ready"])
        with self.assertRaises(FileExistsError):
            builder.build_feature_pack(source, policy, fake_checkpoint(), output)


class TraceFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_trainable_or_training_policy_rejected(self):
        with TemporaryDirectory() as temp:
            for mutate in (lambda p: p.train(), lambda p: p.requires_grad_(True)):
                policy = FakePolicy()
                mutate(policy)
                with self.assertRaisesRegex(ValueError, "eval"):
                    builder.TraceEncoder(policy, Path(temp))

    def test_nonfinite_and_wrong_empty_mask_fail_restoring_hooks(self):
        for field in ("invalid", "bad_empty"):
            with self.subTest(field=field), TemporaryDirectory() as temp:
                policy = FakePolicy()
                if field == "invalid":
                    policy.model.paligemma_with_expert.invalid = True
                else:
                    policy.bad_empty = True
                with self.assertRaises(ValueError):
                    with builder.TraceEncoder(policy, Path(temp)) as encoder:
                        encoder.row(np.zeros((2, 256, 256, 3), dtype=np.uint8),
                                    episode_index=1400, frame_index=0, preview=False)
                self.assertNotIn("_preprocess_images", policy.__dict__)
                self.assertNotIn("embed_image", policy.model.paligemma_with_expert.__dict__)


if __name__ == "__main__":
    unittest.main()
