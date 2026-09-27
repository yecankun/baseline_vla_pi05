"""CPU-only fake-policy tests; no LeRobot import, checkpoint, GPU, or data."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    from pi05_libero_feature_extraction import EMPTY_IMAGE_KEY, NATIVE_IMAGE_KEYS, extract_native_visual_features
else:
    from .pi05_libero_feature_extraction import EMPTY_IMAGE_KEY, NATIVE_IMAGE_KEYS, extract_native_visual_features


class _FakeVision(nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.events = events
        self.output_mode = "valid"

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.events.append(("embed", torch.is_grad_enabled(), torch.is_inference_mode_enabled(), image.clone()))
        means = image.mean(dim=(2, 3)) * self.weight
        tokens = torch.stack((means, means), dim=1)
        if self.output_mode == "nan":
            tokens[0, 0, 0] = float("nan")
        elif self.output_mode == "wrong_shape":
            tokens = tokens[:, 0]
        elif self.output_mode == "empty_dim":
            tokens = tokens[..., :0]
        return tokens


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_features=OrderedDict((key, object()) for key in NATIVE_IMAGE_KEYS), empty_cameras=0)
        self.events: list = []
        self.model = nn.Module()
        self.model.paligemma_with_expert = _FakeVision(self.events)
        self.preprocess_mode = "valid"
        self.eval()

    def _preprocess_images(self, batch: dict[str, torch.Tensor]):
        self.events.append(("preprocess", torch.is_grad_enabled(), torch.is_inference_mode_enabled(), tuple(batch)))
        images = [F.interpolate(batch[key], size=(4, 6), mode="bilinear", align_corners=False) * 2 - 1 for key in NATIVE_IMAGE_KEYS]
        masks = [torch.ones(image.shape[0], dtype=torch.bool) for image in images]
        if self.config.empty_cameras == 1:
            images.append(torch.full_like(images[0], -1))
            masks.append(torch.zeros_like(masks[0]))
        if self.preprocess_mode == "padded_view":
            images.append(torch.zeros_like(images[0]))
            masks.append(torch.zeros_like(masks[0]))
        elif self.preprocess_mode == "missing_view":
            images.pop()
        elif self.preprocess_mode == "invalid_mask":
            masks[1][0] = False
        elif self.preprocess_mode == "wrong_mask_dtype":
            masks[0] = masks[0].float()
        elif self.preprocess_mode == "wrong_mask_shape":
            masks[0] = masks[0][:, None]
        elif self.preprocess_mode == "missing_mask":
            masks.pop()
        elif self.preprocess_mode == "out_of_range":
            images[0][0, 0, 0, 0] = 3
        elif self.preprocess_mode == "nan":
            images[0][0, 0, 0, 0] = float("nan")
        elif self.preprocess_mode == "bad_result":
            return images
        elif self.preprocess_mode == "mutate_input":
            batch[NATIVE_IMAGE_KEYS[0]].zero_()
        elif self.preprocess_mode == "empty_view_valid":
            masks[2][0] = True
        elif self.preprocess_mode == "empty_view_wrong_value":
            images[2][0, 0, 0, 0] = 0
        return images, masks


def _batch() -> dict[str, torch.Tensor]:
    # Reverse dictionary order: configured native order, not batch order, wins.
    return {NATIVE_IMAGE_KEYS[1]: torch.ones(2, 3, 7, 9), NATIVE_IMAGE_KEYS[0]: torch.zeros(2, 3, 5, 8)}


class TestNativeVisualFeatures(unittest.TestCase):
    def test_preprocessing_order_range_pooling_and_frozen_execution(self) -> None:
        policy = _FakePolicy()
        batch = _batch()
        batch[NATIVE_IMAGE_KEYS[0]].requires_grad_()
        weight = policy.model.paligemma_with_expert.weight
        original_weight = weight.detach().clone()
        latent = extract_native_visual_features(policy, batch)
        self.assertEqual(tuple(latent.shape), (2, 2, 3))
        self.assertTrue(torch.equal(latent[:, 0], torch.full((2, 3), -2.0)))
        self.assertTrue(torch.equal(latent[:, 1], torch.full((2, 3), 2.0)))
        self.assertEqual([event[0] for event in policy.events], ["preprocess", "embed", "embed"])
        self.assertEqual(policy.events[0][3], NATIVE_IMAGE_KEYS)
        for event in policy.events:
            self.assertFalse(event[1])
            self.assertTrue(event[2])
        self.assertEqual(tuple(policy.events[1][3].shape), (2, 3, 4, 6))
        self.assertEqual(float(policy.events[1][3].min()), -1.0)
        self.assertEqual(float(policy.events[2][3].max()), 1.0)
        self.assertFalse(latent.requires_grad)
        self.assertIsNone(latent.grad_fn)
        self.assertIsNone(weight.grad)
        self.assertTrue(weight.requires_grad)
        self.assertTrue(torch.equal(weight, original_weight))
        self.assertTrue(batch[NATIVE_IMAGE_KEYS[0]].requires_grad)
        self.assertIsNone(batch[NATIVE_IMAGE_KEYS[0]].grad)
        self.assertFalse(any(module.training for module in policy.modules()))

    def test_input_clone_prevents_in_place_preprocess_mutation(self) -> None:
        policy = _FakePolicy()
        policy.preprocess_mode = "mutate_input"
        batch = _batch()
        batch[NATIVE_IMAGE_KEYS[0]].fill_(0.25)
        extract_native_visual_features(policy, batch)
        self.assertTrue(torch.equal(batch[NATIVE_IMAGE_KEYS[0]], torch.full((2, 3, 5, 8), 0.25)))

    def test_declared_empty_camera_is_checked_and_not_embedded(self) -> None:
        policy = _FakePolicy()
        policy.config.image_features[EMPTY_IMAGE_KEY] = object()
        policy.config.empty_cameras = 1
        latent = extract_native_visual_features(policy, _batch())
        self.assertEqual(tuple(latent.shape), (2, 2, 3))
        self.assertEqual([event[0] for event in policy.events], ["preprocess", "embed", "embed"])
        self.assertEqual(policy.events[0][3], NATIVE_IMAGE_KEYS)
        self.assertTrue(torch.equal(latent[:, 0], torch.full((2, 3), -2.0)))
        self.assertTrue(torch.equal(latent[:, 1], torch.full((2, 3), 2.0)))

    def test_declared_empty_camera_requires_false_mask_and_minus_one(self) -> None:
        for mode in ("empty_view_valid", "empty_view_wrong_value"):
            with self.subTest(mode=mode):
                policy = _FakePolicy()
                policy.config.image_features[EMPTY_IMAGE_KEY] = object()
                policy.config.empty_cameras = 1
                policy.preprocess_mode = mode
                with self.assertRaisesRegex(ValueError, "declared empty camera"):
                    extract_native_visual_features(policy, _batch())
                self.assertEqual([event[0] for event in policy.events], ["preprocess"])

    def test_empty_camera_count_must_match_config(self) -> None:
        for with_key, count in ((False, 1), (True, 0), (True, 2)):
            with self.subTest(with_key=with_key, count=count):
                policy = _FakePolicy()
                if with_key:
                    policy.config.image_features[EMPTY_IMAGE_KEY] = object()
                policy.config.empty_cameras = count
                with self.assertRaisesRegex(ValueError, "empty_cameras"):
                    extract_native_visual_features(policy, _batch())

    def test_training_policy_or_training_child_rejected_without_mode_mutation(self) -> None:
        for child_only in (False, True):
            with self.subTest(child_only=child_only):
                policy = _FakePolicy()
                module = policy.model if child_only else policy
                module.train()
                with self.assertRaisesRegex(ValueError, "eval mode"):
                    extract_native_visual_features(policy, _batch())
                self.assertTrue(module.training)
                self.assertEqual(policy.events, [])

    def test_non_module_and_non_mapping_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "torch.nn.Module"):
            extract_native_visual_features(object(), _batch())
        with self.assertRaisesRegex(ValueError, "mapping"):
            extract_native_visual_features(_FakePolicy(), list(_batch().values()))

    def test_extra_missing_and_privileged_fields_rejected(self) -> None:
        for key in ("observation.state", "task", "action", "sample_role", "diagnostic_targets", "exact_contact", "observation.images.empty_camera", EMPTY_IMAGE_KEY, "piper_event_id"):
            with self.subTest(key=key):
                batch = _batch()
                batch[key] = {"tip_pos": [0, 0, 0]}
                policy = _FakePolicy()
                with self.assertRaisesRegex(ValueError, "exactly"):
                    extract_native_visual_features(policy, batch)
                self.assertEqual(policy.events, [])
        batch = _batch()
        batch.pop(NATIVE_IMAGE_KEYS[1])
        with self.assertRaises(ValueError):
            extract_native_visual_features(_FakePolicy(), batch)

    def test_image_contract_rejections(self) -> None:
        cases = {
            "uint8": torch.zeros(2, 3, 5, 8, dtype=torch.uint8),
            "integer": torch.zeros(2, 3, 5, 8, dtype=torch.int64),
            "not_tensor": [[[0.0]]],
            "nhwc": torch.zeros(2, 5, 8, 3),
            "unbatched": torch.zeros(3, 5, 8),
            "four_channels": torch.zeros(2, 4, 5, 8),
            "empty_batch": torch.zeros(0, 3, 5, 8),
            "empty_height": torch.zeros(2, 3, 0, 8),
            "nan": torch.full((2, 3, 5, 8), float("nan")),
            "inf": torch.full((2, 3, 5, 8), float("inf")),
            "negative": torch.full((2, 3, 5, 8), -0.01),
            "over_one": torch.full((2, 3, 5, 8), 1.01),
            "byte_range_float": torch.full((2, 3, 5, 8), 255.0),
            "different_batch": torch.zeros(1, 3, 5, 8),
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                batch = _batch()
                batch[NATIVE_IMAGE_KEYS[0]] = value
                policy = _FakePolicy()
                with self.assertRaises(ValueError):
                    extract_native_visual_features(policy, batch)
                self.assertEqual(policy.events, [])

    def test_config_requires_exact_native_order_and_view_count(self) -> None:
        cases = [
            None,
            list(NATIVE_IMAGE_KEYS),
            OrderedDict((key, object()) for key in reversed(NATIVE_IMAGE_KEYS)),
            dict.fromkeys((NATIVE_IMAGE_KEYS[0], EMPTY_IMAGE_KEY, NATIVE_IMAGE_KEYS[1])),
            {NATIVE_IMAGE_KEYS[0]: object()},
            dict.fromkeys((*NATIVE_IMAGE_KEYS, "observation.images.empty_camera")),
            {"observation.images.side": object(), "observation.images.top": object()},
        ]
        for features in cases:
            with self.subTest(features=features):
                policy = _FakePolicy()
                policy.config.image_features = features
                with self.assertRaisesRegex(ValueError, "image_features"):
                    extract_native_visual_features(policy, _batch())
                self.assertEqual(policy.events, [])

    def test_preprocessing_outputs_fail_closed(self) -> None:
        for mode in ("padded_view", "missing_view", "invalid_mask", "wrong_mask_dtype", "wrong_mask_shape", "missing_mask", "out_of_range", "nan", "bad_result"):
            with self.subTest(mode=mode):
                policy = _FakePolicy()
                policy.preprocess_mode = mode
                with self.assertRaises(ValueError):
                    extract_native_visual_features(policy, _batch())
                self.assertEqual([event[0] for event in policy.events], ["preprocess"])

    def test_missing_preprocess_or_embed_api(self) -> None:
        policy = _FakePolicy()
        policy._preprocess_images = None
        with self.assertRaisesRegex(ValueError, "_preprocess_images"):
            extract_native_visual_features(policy, _batch())
        policy = _FakePolicy()
        policy.model.paligemma_with_expert.embed_image = None
        with self.assertRaisesRegex(ValueError, "embed_image"):
            extract_native_visual_features(policy, _batch())

    def test_invalid_embedding_outputs_fail_closed(self) -> None:
        for mode in ("nan", "wrong_shape", "empty_dim"):
            with self.subTest(mode=mode):
                policy = _FakePolicy()
                policy.model.paligemma_with_expert.output_mode = mode
                with self.assertRaises(ValueError):
                    extract_native_visual_features(policy, _batch())


if __name__ == "__main__":
    unittest.main()
