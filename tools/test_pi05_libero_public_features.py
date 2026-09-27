"""CPU-only negative contract tests and fake-policy extraction tests."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    import probe_pi05_libero_public_features as probe
else:
    from . import probe_pi05_libero_public_features as probe


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def image_rows():
    array = np.zeros((7, 2, 256, 256, 3), dtype=np.uint8)
    for row in range(7):
        array[row, 0, :128, :, 0] = row + 10
        array[row, 1, :, :128, 1] = row + 120
    return array


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.nan_output = False

    def embed_image(self, image):
        scalar = image.mean(dim=(1, 2, 3)) * self.weight
        value = scalar[:, None, None].expand(-1, 2, 2048).clone()
        if self.nan_output:
            value[0, 0, 0] = float("nan")
        return value


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        keys = [*probe.IMAGE_KEYS, "observation.images.empty_camera_0"]
        self.config = SimpleNamespace(image_features=OrderedDict((key, object()) for key in keys), empty_cameras=1)
        self.model = nn.Module()
        self.model.paligemma_with_expert = FakeVision()
        self.eval()

    def _preprocess_images(self, batch):
        images = [F.interpolate(batch[key], size=(224, 224), mode="bilinear", align_corners=False) * 2 - 1
                  for key in probe.IMAGE_KEYS]
        images.append(torch.full_like(images[0], -1))
        return images, [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])]


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "source").mkdir()
        payload = self.root / "source" / "metadata.json"
        payload.write_text('{"test_fixture_only":true}', encoding="utf-8")
        self.records = [{"episode_index": 1400, "frame_index": i, "task_index": 39,
                         "task": "fixture task 9", "observation.state": [0.0] * 8, "action": [0.0] * 7}
                        for i in range(7)]
        self.images = image_rows()
        self.report = {"schema": probe.SOURCE_SCHEMA, "status": "passed",
                       "source": {"kind": "public_demonstrations", "repo_id": probe.DEMO_REPO,
                                  "revision": probe.DEMO_REVISION},
                       "task_registry": [{"task_id": i, "source_task_index": source,
                                          "task_instruction": f"fixture task {i}"}
                                         for i, source in enumerate(probe.SOURCE_TASK_INDICES)],
                       "selected_episode": {"episode_index": 1400, "task_id": 9, "source_task_index": 39,
                                            "record_count": 140, "complete_episode": True},
                       "files": [{"path": "metadata.json", "local_path": "source/metadata.json",
                                  "size": payload.stat().st_size, "sha256": probe.sha256_file(payload)}],
                       "probe": {"frame_indices": list(range(7)), "row_count": 7,
                                 "images_path": "probe_images.npy", "records_path": "probe_records.json",
                                 "orientation": probe.ORIENTATION}, "output_sha256": {}}
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        np.save(self.root / "probe_images.npy", self.images, allow_pickle=False)
        save_json(self.root / "probe_records.json", self.records)
        for stem, suffix in (("images", "npy"), ("records", "json")):
            name = f"probe_{stem}.{suffix}"
            digest = probe.sha256_file(self.root / name)
            self.report["probe"][f"{stem}_sha256"] = digest
            self.report["output_sha256"][name] = digest
        save_json(self.root / "report.json", self.report)
        self.digest = probe.sha256_file(self.root / "report.json")

    def validate(self):
        return probe.validate_source(self.root, self.digest)

    def test_native_shapes_and_identity_pass(self):
        result = self.validate()
        self.assertEqual(result["images"].shape, (7, 2, 256, 256, 3))
        self.assertEqual(len(result["records"]), 7)
        self.assertEqual(result["report_sha256"], self.digest)

    def test_report_must_match_external_hash(self):
        with self.assertRaisesRegex(ValueError, "report SHA256"):
            probe.validate_source(self.root, "a" * 64)

    def test_synthetic_or_rollout_source_rejected(self):
        for kind in ("synthetic", "evaluation_rollouts"):
            with self.subTest(kind=kind):
                self.report["source"]["kind"] = kind
                self.write()
                with self.assertRaisesRegex(ValueError, "pinned public"):
                    self.validate()

    def test_source_revision_rejected(self):
        self.report["source"]["revision"] = "f" * 40
        self.write()
        with self.assertRaises(ValueError):
            self.validate()

    def test_native_mapping_cannot_be_assumed_identity(self):
        for i, row in enumerate(self.report["task_registry"]):
            row["source_task_index"] = i
        self.write()
        with self.assertRaisesRegex(ValueError, "source-to-Spatial"):
            self.validate()

    def test_all_ten_tasks_required(self):
        self.report["task_registry"].pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "ten-task"):
            self.validate()

    def test_partial_episode_declaration_rejected(self):
        self.report["selected_episode"]["complete_episode"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "complete source episode"):
            self.validate()

    def test_missing_state_or_diagnostic_input_rejected(self):
        original = deepcopy(self.records)
        del self.records[0]["observation.state"]
        self.write()
        with self.assertRaisesRegex(ValueError, "without missing"):
            self.validate()
        self.records = original
        self.records[0]["success"] = True
        self.write()
        with self.assertRaisesRegex(ValueError, "without missing"):
            self.validate()

    def test_nonconsecutive_frame_and_cross_episode_rejected(self):
        self.records[3]["frame_index"] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.validate()
        self.records[3]["frame_index"] = 3
        self.records[3]["episode_index"] = 1401
        self.write()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.validate()

    def test_task_text_mismatch_rejected(self):
        self.records[0]["task"] = "other instruction"
        self.write()
        with self.assertRaisesRegex(ValueError, "frozen Spatial mapping"):
            self.validate()

    def test_action_bound_and_nonfinite_state_rejected(self):
        self.records[0]["action"][0] = 1.1
        self.write()
        with self.assertRaisesRegex(ValueError, "no clipping"):
            self.validate()
        self.records[0]["action"][0] = 0.0
        self.records[0]["observation.state"][0] = float("nan")
        self.write()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.validate()

    def test_uint8_and_image_shape_are_not_inferred(self):
        self.images = self.images.astype(np.float32)
        self.write()
        with self.assertRaisesRegex(ValueError, "uint8"):
            self.validate()
        self.images = image_rows()[:, :, :224]
        self.write()
        with self.assertRaisesRegex(ValueError, "uint8"):
            self.validate()

    def test_extra_flip_declaration_rejected(self):
        self.report["probe"]["orientation"] = "flip_both_axes"
        self.write()
        with self.assertRaisesRegex(ValueError, "stored RGB orientation"):
            self.validate()

    def test_inventory_payload_tampering_rejected(self):
        (self.root / "source" / "metadata.json").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory hash/size mismatch"):
            self.validate()

    def test_probe_tampering_rejected(self):
        np.save(self.root / "probe_images.npy", np.ones_like(self.images), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "hash-bound"):
            self.validate()

    def test_path_traversal_and_absolute_paths_rejected(self):
        for value in ("../external.json", "/external.json", "C:/external.json", "source\\metadata.json"):
            with self.subTest(path=value):
                self.report["files"][0]["local_path"] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_duplicate_json_keys_rejected(self):
        (self.root / "report.json").write_text('{"status":"passed","status":"failed"}', encoding="utf-8")
        self.digest = probe.sha256_file(self.root / "report.json")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.validate()


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_real_view_only_preprocess_and_repeat_proof(self):
        policy = FakePolicy()
        images = image_rows()
        original = images.copy()
        latents, evidence = probe.extract_probe(policy, images)
        self.assertEqual(latents.shape, (7, 2, 2048))
        self.assertEqual(latents.dtype, np.float32)
        self.assertEqual(evidence["preprocessing_call_count"], 8)
        self.assertEqual(evidence["embed_call_count"], 16)
        self.assertTrue(evidence["repeat_first"]["bitwise_equal"])
        self.assertTrue(evidence["all_calls_inference_mode"])
        self.assertTrue(evidence["all_parameters_frozen"])
        self.assertGreater(evidence["row_variation_max_abs_from_first"], 0)
        self.assertTrue(np.array_equal(images, original))
        for call in evidence["preprocessing_calls"]:
            self.assertEqual(call["masks"], [[True], [True], [False]])
            self.assertEqual(call["processed"][2]["min"], -1)
            self.assertEqual(call["processed"][2]["max"], -1)
        expected = torch.from_numpy(images[0, 0]).permute(2, 0, 1).unsqueeze(0).float() / 255
        self.assertEqual(evidence["preprocessing_calls"][0]["inputs"][probe.IMAGE_KEYS[0]]["sha256"],
                         probe.tensor_record(expected)["sha256"])
        self.assertNotIn("_preprocess_images", policy.__dict__)
        self.assertNotIn("embed_image", policy.model.paligemma_with_expert.__dict__)

    def test_source_and_processed_preview_files(self):
        from PIL import Image
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images = image_rows()
            probe.extract_probe(FakePolicy(), images, root)
            self.assertEqual(len(list(root.glob("*.png"))), 4)
            with Image.open(root / "source_frame000_view0_stored_rgb.png") as picture:
                source = np.asarray(picture).copy()
            with Image.open(root / "processed_frame000_view0_policy_rgb.png") as processed:
                self.assertEqual(processed.size, (224, 224))
            self.assertTrue(np.array_equal(source, images[0, 0]))

    def test_training_or_unfrozen_parameters_rejected(self):
        policy = FakePolicy()
        policy.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            probe.extract_probe(policy, image_rows())
        policy.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            probe.extract_probe(policy, image_rows())

    def test_nonfinite_features_fail_and_restore_hooks(self):
        policy = FakePolicy()
        policy.model.paligemma_with_expert.nan_output = True
        with self.assertRaises(ValueError):
            probe.extract_probe(policy, image_rows())
        self.assertNotIn("_preprocess_images", policy.__dict__)
        self.assertNotIn("embed_image", policy.model.paligemma_with_expert.__dict__)

    def test_preexisting_gradients_rejected(self):
        policy = FakePolicy()
        policy.model.paligemma_with_expert.weight.grad = torch.ones(())
        with self.assertRaisesRegex(ValueError, "without parameter gradients"):
            probe.extract_probe(policy, image_rows())


class CheckpointConfigTests(unittest.TestCase):
    def setUp(self):
        # Actual pinned configuration layout, not the earlier two-view guess.
        image_features = OrderedDict([
            (probe.IMAGE_KEYS[0], SimpleNamespace(shape=[3, 256, 256])),
            (probe.IMAGE_KEYS[1], SimpleNamespace(shape=[3, 256, 256])),
            (probe.EMPTY_IMAGE_KEY, SimpleNamespace(shape=[3, 224, 224])),
        ])
        self.config = SimpleNamespace(
            image_features=image_features,
            input_features={**image_features, "observation.state": SimpleNamespace(shape=[8])},
            output_features={"action": SimpleNamespace(shape=[7])},
            image_resolution=[224, 224], empty_cameras=1, dtype="bfloat16",
        )

    def test_actual_pinned_two_real_one_empty_layout_passes_unchanged(self):
        before = deepcopy(self.config)
        layout = probe.validate_checkpoint_config(self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(list(layout["image_features"]), [*probe.IMAGE_KEYS, probe.EMPTY_IMAGE_KEY])
        self.assertEqual(layout["real_extractor_input_keys"], list(probe.IMAGE_KEYS))
        self.assertFalse(layout["empty_camera_embedded"])

    def test_missing_declared_empty_camera_rejected(self):
        self.config.image_features.pop(probe.EMPTY_IMAGE_KEY)
        self.config.input_features.pop(probe.EMPTY_IMAGE_KEY)
        with self.assertRaisesRegex(ValueError, "declared empty_camera_0"):
            probe.validate_checkpoint_config(self.config)

    def test_extra_camera_rejected(self):
        self.config.image_features["observation.images.extra"] = SimpleNamespace(shape=[3, 224, 224])
        with self.assertRaisesRegex(ValueError, "declared empty_camera_0"):
            probe.validate_checkpoint_config(self.config)

    def test_empty_camera_shape_rejected(self):
        self.config.image_features[probe.EMPTY_IMAGE_KEY].shape = [3, 256, 256]
        with self.assertRaisesRegex(ValueError, "image shape differs"):
            probe.validate_checkpoint_config(self.config)

    def test_real_camera_shape_rejected(self):
        self.config.image_features[probe.IMAGE_KEYS[0]].shape = [3, 224, 224]
        with self.assertRaisesRegex(ValueError, "image shape differs"):
            probe.validate_checkpoint_config(self.config)

    def test_camera_order_rejected(self):
        self.config.image_features.move_to_end(probe.IMAGE_KEYS[0])
        with self.assertRaisesRegex(ValueError, "ordered"):
            probe.validate_checkpoint_config(self.config)

    def test_empty_count_rejected(self):
        self.config.empty_cameras = 0
        with self.assertRaisesRegex(ValueError, "layout differs"):
            probe.validate_checkpoint_config(self.config)

    def test_foreign_input_key_rejected(self):
        self.config.input_features["observation.contact_truth"] = SimpleNamespace(shape=[1])
        with self.assertRaisesRegex(ValueError, "input feature keys"):
            probe.validate_checkpoint_config(self.config)

    def test_wrong_state_action_or_dtype_rejected(self):
        valid = deepcopy(self.config)
        self.config.input_features["observation.state"].shape = [32]
        with self.assertRaisesRegex(ValueError, "layout differs"):
            probe.validate_checkpoint_config(self.config)
        self.config = deepcopy(valid)
        self.config.output_features["action"].shape = [9]
        with self.assertRaisesRegex(ValueError, "layout differs"):
            probe.validate_checkpoint_config(self.config)
        self.config = deepcopy(valid)
        self.config.dtype = "float32"
        with self.assertRaisesRegex(ValueError, "layout differs"):
            probe.validate_checkpoint_config(self.config)


if __name__ == "__main__":
    unittest.main()
