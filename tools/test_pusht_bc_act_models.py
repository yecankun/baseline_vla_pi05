"""CPU synthetic tests only: no LeRobot import, dataset, optimizer, or environment."""
import builtins
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tools.pusht_bc_act_models import (
    ACT_ARCHITECTURE, IMAGE, STATE, IMAGENET_MEAN, IMAGENET_STD,
    CompactPushTBC, act_batch, imagenet_normalize, smoke, synthetic_only_stats,
    validate_observation,
)


def observation():
    return {IMAGE: torch.zeros(2, 3, 96, 96), STATE: torch.zeros(2, 2)}


class TestPushTBCAndACTInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_compact_architecture(self):
        model = CompactPushTBC()
        convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertEqual([(layer.in_channels, layer.out_channels) for layer in convs], [(3, 32), (32, 64), (64, 64)])
        self.assertTrue(all(layer.kernel_size == (5, 5) and layer.stride == (2, 2) and layer.padding == (2, 2) for layer in convs))
        linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linears], [(9216, 128), (130, 256), (256, 2)])
        self.assertFalse(model.describe()["exact_senior_architecture_reproduction"])

    def test_bc_shape_mse_and_no_gradients(self):
        model = CompactPushTBC()
        batch = observation()
        target = torch.full((2, 2), 2.0)
        with torch.no_grad():
            predicted = model(batch)
            loss = model.loss(batch, target)
        self.assertEqual(tuple(predicted.shape), (2, 2))
        self.assertTrue(torch.equal(loss, ((predicted - target) ** 2).mean()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_output_is_unbounded_no_sigmoid_or_clipping(self):
        model = CompactPushTBC().eval()
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.copy_(torch.tensor([-5.0, 7.0]))
        result = model.select_action(observation())
        self.assertTrue(torch.equal(result, torch.tensor([[-5.0, 7.0], [-5.0, 7.0]])))

    def test_reset_is_stateless_and_preserves_parameters(self):
        model = CompactPushTBC().eval()
        batch = observation()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        first = model.select_action(batch)
        self.assertIsNone(model.reset())
        self.assertTrue(torch.equal(first, model.select_action(batch)))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        self.assertFalse(hasattr(model, "_queues"))
        with self.assertRaises(ValueError):
            model.train().select_action(batch)

    def test_raw_image_and_state_shape_range_validation(self):
        images = [torch.zeros(2, 96, 96, 3), torch.zeros(2, 3, 84, 84), torch.zeros(0, 3, 96, 96),
                  torch.zeros(2, 3, 96, 96, dtype=torch.uint8), torch.full((2, 3, 96, 96), -0.1),
                  torch.full((2, 3, 96, 96), 1.1), torch.full((2, 3, 96, 96), float("nan"))]
        for image in images:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                validate_observation({IMAGE: image, STATE: torch.zeros(2, 2)})
        for state in (torch.zeros(2, 3), torch.zeros(1, 2), torch.full((2, 2), float("inf")), torch.zeros(2, 2, dtype=torch.float64)):
            with self.subTest(shape=state.shape), self.assertRaises(ValueError):
                validate_observation({IMAGE: torch.zeros(2, 3, 96, 96), STATE: state})

    def test_no_extra_observations_enter_either_model_path(self):
        for key in ("reward", "coverage", "is_success", "environment_state", "task", "diagnostic_targets", "tip_pos", "action"):
            batch = {**observation(), key: 0.0}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    CompactPushTBC()(batch)
                with self.assertRaises(ValueError):
                    act_batch(batch)

    def test_normalization_shared_and_nonmutating(self):
        images = torch.rand(2, 3, 96, 96)
        original = images.clone()
        result = imagenet_normalize(images)
        expected = (images - torch.tensor(IMAGENET_MEAN)[None, :, None, None]) / torch.tensor(IMAGENET_STD)[None, :, None, None]
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(images, original))
        stats = synthetic_only_stats()
        self.assertTrue(torch.equal(stats[IMAGE]["mean"].flatten(), torch.tensor(IMAGENET_MEAN)))
        self.assertTrue(torch.equal(stats[STATE]["std"], torch.tensor([128.0, 128.0])))

    def test_bc_action_target_contract(self):
        model = CompactPushTBC()
        for target in (torch.zeros(2, 16, 2), torch.zeros(2, 2, dtype=torch.int64), torch.full((2, 2), float("nan"))):
            with self.subTest(shape=target.shape), self.assertRaises(ValueError):
                model.loss(observation(), target)

    def test_act_target_routing_keeps_targets_separate(self):
        obs = observation()
        actions = torch.zeros(2, 16, 2)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        batch = act_batch(obs, action=actions, action_is_pad=mask)
        self.assertEqual(set(batch), {IMAGE, STATE, "action", "action_is_pad"})
        self.assertEqual(set(obs), {IMAGE, STATE})
        self.assertEqual(set(act_batch(obs)), {IMAGE, STATE})
        self.assertTrue(torch.equal(batch["action_is_pad"], mask))

    def test_act_action_and_mask_shape_rejections(self):
        obs = observation()
        action = torch.zeros(2, 16, 2)
        masks = (torch.zeros(2, 16), torch.zeros(2, 8, dtype=torch.bool), torch.ones(2, 16, dtype=torch.bool))
        for mask in masks:
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                act_batch(obs, action=action, action_is_pad=mask)
        with self.assertRaises(ValueError):
            act_batch(obs, action=action)
        with self.assertRaises(ValueError):
            act_batch(obs, action=torch.zeros(2, 8, 2), action_is_pad=torch.zeros(2, 16, dtype=torch.bool))

    def test_act_full_config_constants_and_no_backbone_download(self):
        expected = {"dim_model": 512, "n_encoder_layers": 4, "n_decoder_layers": 1,
                    "use_vae": True, "latent_dim": 32, "n_vae_encoder_layers": 4,
                    "n_heads": 8, "dim_feedforward": 3200, "chunk_size": 16, "n_action_steps": 8,
                    "pretrained_backbone_weights": None, "vision_backbone": "resnet18"}
        self.assertTrue(all(ACT_ARCHITECTURE[key] == value for key, value in expected.items()))

    def test_bc_smoke_lazy_no_lerobot_rng_preserved(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith("lerobot"):
                raise AssertionError("BC smoke must not import LeRobot")
            return original_import(name, *args, **kwargs)
        before = torch.random.get_rng_state().clone()
        with patch("builtins.__import__", side_effect=guarded_import):
            report = smoke("bc", device="cpu", seed=20260913)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["normalized_action"]["shape"], [2, 2])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["backward_calls"], 0)
        self.assertEqual(report["environment_steps"], 0)
        self.assertFalse(report["benchmark_score_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
