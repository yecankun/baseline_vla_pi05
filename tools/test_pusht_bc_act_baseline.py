"""Protocol and fake-environment tests; no real model/env/checkpoint execution."""
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

from tools import run_pusht_bc_act_baseline as runner
from tools.test_pusht_bc_act_inference import fake_adapter, preprocess


def successful_smoke():
    return {'stage': 'smoke', 'status': 'completed', 'strict_protocol_pass': True,
            'protocol_sha256': runner.common.canonical_hash(runner.PROTOCOL),
            'checkpoint_binding': {'checkpoint': 'final'}, 'source_files': {'source': 'a' * 64},
            'validation_replay': {'passed': True}, 'queue_check': {'passed': True},
            'reset_replay_exact': True, 'model_state_unchanged': True,
            'episodes': [{'seed': 20260913, 'is_success': False, 'max_coverage': 0.1,
                          'native_action_out_of_bounds_steps': 0, 'native_action_out_of_bounds_elements': 0}]}


class FakeEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(seed=lambda s: None, low=np.zeros(2), high=np.full(2, 512.))
        self.observation_space = SimpleNamespace(seed=lambda s: None)
        self.steps = 0

    def raw(self):
        return {'pixels': np.zeros((96, 96, 3), dtype=np.uint8), 'agent_pos': np.array([50. + self.steps, 50.])}

    def reset(self, seed):
        self.steps = 0
        return self.raw(), {}

    def step(self, action):
        if self.steps >= 2:
            raise AssertionError('stepped after done')
        self.steps += 1
        # Reward deliberately contradicts success to test provenance.
        return self.raw(), 1., self.steps == 2, False, {'is_success': False, 'coverage': 0.1 * self.steps}


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_schedule_and_final_checkpoint(self):
        self.assertEqual(runner.PROTOCOL['benchmark_seeds'], list(range(100000, 100100)))
        self.assertEqual(runner.PROTOCOL['benchmark_max_steps'], 300)
        self.assertEqual(runner.PROTOCOL['smoke_seeds'], [20260913])
        self.assertFalse(runner.PROTOCOL['action_clipping'])
        self.assertIn('step100000', runner.PROTOCOL['checkpoint_selection'])

    def test_default_preflight_and_execute_gates(self):
        self.assertEqual(runner.parse_args([]).stage, 'preflight')
        for stage in ('smoke', 'benchmark'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(['--stage', stage])
            self.assertTrue(runner.parse_args(['--stage', stage, '--execute']).execute)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(['--execute'])

    def test_smoke_root_cannot_override_smoke(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(['--stage', 'smoke', '--execute', '--smoke-root', 'elsewhere'])

    def test_attempt_path_traversal_rejected(self):
        for bad in ('../x', 'x/y', 'x\\y', '', '.', 'x' * 49):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                runner.output_directory('smoke', 'bc', bad)
        self.assertNotEqual(runner.output_directory('smoke', 'bc'), runner.output_directory('smoke', 'act'))

    def verify(self, value):
        path = self.root / 'smoke.json'
        path.write_text(json.dumps(value))
        return runner.verify_smoke(path, {'checkpoint': 'final'}, {'source': 'a' * 64})

    def test_matching_smoke_accepted(self):
        self.assertEqual(self.verify(successful_smoke())['status'], 'completed')

    def test_checkpoint_or_source_drift_rejected(self):
        for field, value in (('checkpoint_binding', {'checkpoint': 'earlier'}),
                             ('source_files', {}), ('protocol_sha256', 'b' * 64)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding drift'):
                self.verify({**successful_smoke(), field: value})

    def test_unverified_load_or_queue_or_reset_rejected(self):
        for field, value in (('validation_replay', {}), ('queue_check', {}),
                             ('reset_replay_exact', False), ('model_state_unchanged', False)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'incomplete'):
                self.verify({**successful_smoke(), field: value})

    def test_failed_or_incomplete_smoke_rejected(self):
        for field, value in (('status', 'failed'), ('strict_protocol_pass', False), ('episodes', [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify({**successful_smoke(), field: value})

    def test_bounds_violations_not_promoted(self):
        value = successful_smoke()
        value['episodes'][0]['native_action_out_of_bounds_steps'] = 1
        with self.assertRaises(ValueError):
            self.verify(value)

    def test_failed_run_preserved_and_not_overwritten(self):
        output = self.root / 'failed'
        with patch.object(runner, 'source_binding', side_effect=ValueError('source drift')):
            result = runner.run('smoke', 'bc', output, {}, {})
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['environment_steps'], 0)
        self.assertTrue((output / 'report.json').is_file())
        with self.assertRaises(FileExistsError):
            runner.run('smoke', 'bc', output, {}, {})

    def fake_episode(self, env, **kwargs):
        return runner.episode(fake_adapter('bc'), env,
            {'np': np, 'torch': torch, 'preprocess_observation': preprocess},
            seed=20260913, max_steps=20, output=self.root, index=0, phase='primary', **kwargs)

    def test_stop_at_first_done_and_use_info_not_reward(self):
        env = FakeEnv()
        counter = {'environment_step_calls': 0, 'environment_steps': 0}
        value = self.fake_episode(env, execution_counter=counter)
        self.assertEqual(env.steps, 2)
        self.assertEqual(value['steps'], 2)
        self.assertFalse(value['is_success'])
        self.assertEqual(value['max_coverage'], 0.2)
        self.assertTrue(value['terminated'])
        self.assertEqual(counter, {'environment_step_calls': 2, 'environment_steps': 2})

    def test_environment_failure_preserves_actual_call_counts(self):
        env = FakeEnv()
        def broken_step(action):
            raise RuntimeError('simulated step failure')
        env.step = broken_step
        counter = {'environment_step_calls': 0, 'environment_steps': 0}
        with self.assertRaisesRegex(RuntimeError, 'step failure'):
            self.fake_episode(env, execution_counter=counter)
        self.assertEqual(counter, {'environment_step_calls': 1, 'environment_steps': 0})

    def test_full_trace_replay_exact_ignores_timing(self):
        first, second = self.fake_episode(FakeEnv()), self.fake_episode(FakeEnv())
        self.assertEqual(first['trace_sha256'], second['trace_sha256'])
        self.assertEqual(first['reset_observation_sha256'], second['reset_observation_sha256'])

    def test_changed_reference_reset_blocks_before_action(self):
        env = FakeEnv()
        with self.assertRaisesRegex(ValueError, 'reset observation differs'):
            self.fake_episode(env, reference_reset='a' * 64)
        self.assertEqual(env.steps, 0)


if __name__ == '__main__':
    unittest.main()
