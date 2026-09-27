"""CPU fake-rollout integration checks; not LIBERO success or model evidence."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_pi05_libero_low_contrast as runner
from pi05_libero_low_contrast_contract import load_protocol, summarize_pairs


ROOT = Path(__file__).resolve().parents[1]


class OperationalSpaceController:
    use_delta = True
    impedance_mode = "fixed"
    control_dim = 6
    input_min = np.array([-1.] * 6)
    input_max = np.array([1.] * 6)
    output_min = np.array([-.05] * 3 + [-.5] * 3)
    output_max = np.array([.05] * 3 + [.5] * 3)


class PandaGripper:
    dof = 1
    speed = .01


class InnerEnvironment:
    def __init__(self):
        self.robots = [types.SimpleNamespace(controller=OperationalSpaceController(), gripper=PandaGripper())]
        self.steps = []
        self.states = []

    def set_init_state(self, state):
        self.states.append(np.array(state, copy=True))

    def step(self, action):
        self.steps.append(np.array(action, copy=True))


class NativeEnvironment:
    task_id = 0
    init_states = True
    _reset_stride = 1
    _max_episode_steps = 280
    num_steps_wait = 10

    def __init__(self, bad_camera=False):
        self._env = InnerEnvironment()
        self._init_states = np.arange(100, dtype=np.float64).reshape(10, 10)
        self.init_state_id = 9  # Must be replaced by the actual B4b helper.
        self.unwrapped = self
        self.bad_camera = bad_camera
        self.reset_seeds = []

    def observation(self):
        pixels = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(1, 256, 256, 3)
        other = np.flip(pixels, axis=1).copy()
        if self.bad_camera:
            other = other[:, :255]
        return {"pixels": {"image": pixels, "image2": other}, "agent_pos": np.arange(8, dtype=np.float32)[None]}

    def reset(self, seed=None, **kwargs):
        self.reset_seeds.append(seed)
        self._env.set_init_state(self._init_states[self.init_state_id])
        self.init_state_id += 1
        for _ in range(self.num_steps_wait):
            self._env.step(np.zeros(7, dtype=np.float32))
        return self.observation(), {}

    def step(self, action):
        self._env.step(action)
        # Upstream success performs an unseeded reset. Its settling actions
        # must not be mistaken for additional policy actions by the hooks.
        observation, _ = self.reset()
        return observation, 1.0, True, False, {"success": True}


class SyncEnvironment:
    num_envs = 1

    def __init__(self, bad_camera=False):
        self.envs = [NativeEnvironment(bad_camera=bad_camera)]

    def reset(self, seed):
        return self.envs[0].reset(seed=seed[0])

    def step(self, action):
        return self.envs[0].step(action[0])


class Policy:
    def __init__(self):
        self._action_queue = ["stale action"]
        self.reset_count = 0

    def reset(self):
        self._action_queue.clear()
        self.reset_count += 1

    def select_action(self, batch):
        self._action_queue.append("next queued action")
        return np.array([[.1, .2, .3, .01, .02, .03, -1.00916]], dtype=np.float32)


def fake_official_module(*, bad_state=False):
    module = types.ModuleType("cpu_fake_official_rollout")
    module.seed_calls = []
    module.set_seed = lambda seed: module.seed_calls.append(seed)

    def preprocess(observation):
        state = observation["agent_pos"]
        if bad_state:
            state = state[:, :7]
        return {"observation.images.image": observation["pixels"]["image"].transpose(0, 3, 1, 2).astype(np.float32) / 255.,
                "observation.images.image2": observation["pixels"]["image2"].transpose(0, 3, 1, 2).astype(np.float32) / 255.,
                "observation.state": state}

    def rollout(*, env, policy, seeds, return_observations, **processors):
        policy.reset()
        observation, _ = env.reset(seed=seeds)
        batch = module.preprocess_observation(observation)
        action = policy.select_action(batch)
        _, _, terminated, truncated, info = env.step(action)
        return {"action": action[:, None, :], "done": np.array([[terminated or truncated]]),
                "success": np.array([[info["success"]]])}

    module.preprocess_observation = preprocess
    module.rollout = rollout
    return module


class EpisodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "docs/libero-spatial-low-contrast-protocol-v1.json")
        self.selected = {"task_id": 0, "seed": 1000, "init_state_index": 0}

    def call_episode(self, module, env, policy, condition, out):
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(isfinite=np.isfinite)}), \
             patch.object(runner, "rng_fingerprint", return_value="c" * 64):
            return runner.episode(module, env, policy, {}, self.selected, condition, out, self.protocol)

    def assert_restored(self, module, env, policy, original_preprocess):
        base = env.envs[0]
        self.assertNotIn("reset", vars(base))
        self.assertNotIn("step", vars(base))
        self.assertNotIn("set_init_state", vars(base._env))
        self.assertNotIn("step", vars(base._env))
        self.assertNotIn("select_action", vars(policy))
        self.assertIs(module.preprocess_observation, original_preprocess)

    def test_pair_excludes_initial_and_automatic_settling_and_restores_hooks(self):
        module, env, policy = fake_official_module(), SyncEnvironment(), Policy()
        original_preprocess = module.preprocess_observation
        reports = []
        with tempfile.TemporaryDirectory() as directory:
            for condition in ("clean", "low_contrast"):
                out = Path(directory) / condition
                report = self.call_episode(module, env, policy, condition, out)
                reports.append(report)
                self.assertEqual(report["steps"], 1)
                self.assertTrue(report["success"])
                self.assertEqual(report["action_outside_unit_bounds_count_per_dim"], [0, 0, 0, 0, 0, 0, 1])
                self.assertEqual(report["runtime_controller"]["settling_steps"], 10)
                saved_actions = np.load(out / "native_actions.npy", allow_pickle=False)
                self.assertEqual(saved_actions.shape, (1, 7))
                self.assertLess(float(saved_actions[0, 6]), -1.)
                self.assertEqual(len((out / "observations.jsonl").read_text().splitlines()), 1)
                self.assertEqual(json.loads((out / "episode.json").read_text())["steps"], 1)
                self.assert_restored(module, env, policy, original_preprocess)
            result = summarize_pairs(reports, "check")
            self.assertEqual(result["pair_count"], 1)
            self.assertEqual(result["paired_outcomes"]["both_success"], 1)
            self.assertEqual(reports[0]["initial_observation_sha256"], reports[1]["initial_observation_sha256"])
            for image in ("observation.images.image", "observation.images.image2"):
                self.assertNotEqual(reports[0]["first_policy_images"][image]["sha256"],
                                    reports[1]["first_policy_images"][image]["sha256"])
        self.assertEqual(len(env.envs[0]._env.steps), 42)  # (10 + 1 + 10) * two arms.
        self.assertEqual(env.envs[0].reset_seeds, [1000, None, 1000, None])
        self.assertEqual(policy.reset_count, 2)
        self.assertEqual(module.seed_calls, [1000, 1000])
        expected_hash = hashlib.sha256(env.envs[0]._init_states[0].tobytes()).hexdigest()
        self.assertEqual(reports[0]["init_state_sha256"], expected_hash)

    def test_raw_schema_failure_restores_all_original_methods(self):
        module, env, policy = fake_official_module(), SyncEnvironment(bad_camera=True), Policy()
        original_preprocess = module.preprocess_observation
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "failure"
            with self.assertRaisesRegex(ValueError, "BHWC256"):
                self.call_episode(module, env, policy, "clean", out)
            self.assertFalse((out / "episode.json").exists())
        self.assert_restored(module, env, policy, original_preprocess)
        self.assertEqual(len(env.envs[0]._env.steps), 10)

    def test_policy_state_failure_restores_all_original_methods(self):
        module, env, policy = fake_official_module(bad_state=True), SyncEnvironment(), Policy()
        original_preprocess = module.preprocess_observation
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "native8"):
                self.call_episode(module, env, policy, "low_contrast", Path(directory) / "failure")
        self.assert_restored(module, env, policy, original_preprocess)
        self.assertEqual(len(env.envs[0]._env.steps), 10)


if __name__ == "__main__":
    unittest.main()
