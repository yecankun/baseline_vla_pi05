"""CPU fake-policy tests; no LeRobot, real checkpoint, dataset or optimizer."""
from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from tools import pusht_bc_act_inference as inference
from tools.pusht_bc_act_models import IMAGE, STATE


def observation(state=(2.0, 3.0)):
    return {IMAGE: torch.zeros(1, 3, 96, 96), STATE: torch.tensor([state], dtype=torch.float32)}


class Processor:
    def __init__(self, scale=1):
        self.scale, self.resets = scale, 0

    def reset(self):
        self.resets += 1

    def __call__(self, item):
        return item if isinstance(item, dict) else item * self.scale


class FakeChunk(nn.Module):
    def forward(self, batch):
        return batch[STATE][:, None] + torch.arange(16, dtype=torch.float32)[None, :, None]


class FakeACT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeChunk()
        self.config = SimpleNamespace(chunk_size=16, n_action_steps=8, temporal_ensemble_coeff=None)
        self.reset()

    def reset(self):
        self._action_queue = deque([], maxlen=8)

    def predict_action_chunk(self, batch):
        return self.model(batch)

    def select_action(self, batch):
        if not self._action_queue:
            self._action_queue.extend(self.predict_action_chunk(batch)[:, :8].transpose(0, 1))
        return self._action_queue.popleft()


class FakeBC(nn.Module):
    def reset(self):
        pass

    def select_action(self, batch):
        return batch[STATE]


def fake_adapter(name):
    stats = {STATE: {'mean': torch.zeros(2), 'std': torch.ones(2)},
             'action': {'mean': torch.zeros(2), 'std': torch.ones(2)}}
    return inference.InferencePolicy(SimpleNamespace(name=name, device='cpu', stats=stats,
        policy=FakeACT() if name == 'act' else FakeBC(),
        pre=Processor() if name == 'act' else None, post=Processor(2) if name == 'act' else None))


def preprocess(raw):
    return {IMAGE: torch.from_numpy(raw['pixels']).permute(2, 0, 1)[None].float() / 255,
            STATE: torch.from_numpy(raw['agent_pos'])[None].float()}


class TestInference(unittest.TestCase):
    def test_bc_normalization_native_unbounded_output(self):
        adapter = fake_adapter('bc')
        adapter.stats[STATE] = {'mean': torch.tensor([1., 2.]), 'std': torch.tensor([2., 4.])}
        adapter.stats['action'] = {'mean': torch.tensor([250., 250.]), 'std': torch.tensor([100., 100.])}
        action = adapter.select_action(observation((21, -38)))
        self.assertTrue(torch.equal(action, torch.tensor([[1250., -750.]])))

    def test_act_queue_consumed_despite_changed_observation(self):
        adapter = fake_adapter('act')
        first = adapter.select_action(observation())
        second = adapter.select_action(observation((100, 200)))
        self.assertTrue(torch.equal(first, torch.tensor([[4., 6.]])))
        self.assertTrue(torch.equal(second, torch.tensor([[6., 8.]])))
        self.assertEqual(adapter.last_queue_trace, {'before': 7, 'after': 6, 'chunk_generated': False})

    def test_real_forward_hook_and_direct_chunk_comparison(self):
        report = inference.verify_queue_and_reset(fake_adapter('act'), observation())
        self.assertTrue(report['passed'])
        self.assertTrue(report['direct_chunk_first8_exact'])
        self.assertEqual(report['model_forwards_per_replay'], 2)
        self.assertEqual([r['after'] for r in report['queue_trace']], [7, 6, 5, 4, 3, 2, 1, 0, 7])

    def test_external_queue_reset_fails(self):
        adapter = fake_adapter('act')
        adapter.select_action(observation())
        adapter.policy.reset()
        with self.assertRaisesRegex(ValueError, 'reset or consumed externally'):
            adapter.select_action(observation())

    def test_fresh_validation_predict_is_separate(self):
        adapter = fake_adapter('act')
        first = adapter.predict({'observation': observation()})
        second = adapter.predict({'observation': observation()})
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(adapter.action_calls, 1)

    def test_bc_reset_exact(self):
        self.assertTrue(inference.verify_queue_and_reset(fake_adapter('bc'), observation())['reset_replay_exact'])

    def test_target_and_truth_rejected(self):
        for name in ('bc', 'act'):
            for key in ('action', 'is_success', 'coverage', 'task', 'environment_state'):
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    fake_adapter(name).select_action({**observation(), key: torch.zeros(1)})

    def test_training_mode_rejected(self):
        adapter = fake_adapter('bc')
        adapter.policy.train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            adapter.select_action(observation())

    def test_native_observation_matches_float32_rgb_and_state(self):
        raw = {'pixels': np.full((96, 96, 3), 255, dtype=np.uint8), 'agent_pos': np.array([5., 7.])}
        obs = inference.observation_from_native(raw, preprocess, 'cpu')
        self.assertTrue(torch.equal(obs[IMAGE], torch.ones(1, 3, 96, 96)))
        self.assertEqual(obs[STATE].dtype, torch.float32)
        self.assertTrue(torch.equal(obs[STATE], torch.tensor([[5., 7.]])))

    def test_native_truth_extra_or_bad_range_rejected(self):
        raw = {'pixels': np.zeros((96, 96, 3), dtype=np.uint8), 'agent_pos': np.zeros(2)}
        for bad in ({**raw, 'environment_state': np.zeros(5)},
                    {**raw, 'pixels': np.zeros((384, 384, 3), dtype=np.uint8)},
                    {**raw, 'agent_pos': np.array([np.nan, 0.])}):
            with self.subTest(keys=list(bad)), self.assertRaises(ValueError):
                inference.observation_from_native(bad, preprocess, 'cpu')

    def test_processor_extra_fields_rejected(self):
        raw = {'pixels': np.zeros((96, 96, 3), dtype=np.uint8), 'agent_pos': np.zeros(2)}
        with self.assertRaisesRegex(ValueError, 'only image and state'):
            inference.observation_from_native(raw, lambda r: {**preprocess(r), 'action': 0}, 'cpu')


if __name__ == '__main__':
    unittest.main()
