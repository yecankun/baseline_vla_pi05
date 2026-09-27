"""CPU-only lifecycle/early guard tests. No model or environment dependency."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_pi05_libero_low_contrast_v2 as runner


@dataclass
class Config:
    task_ids: list


class FreshLifecycleTests(unittest.TestCase):
    def test_each_arm_constructed_seeded_closed(self):
        events, records = [], []
        module = SimpleNamespace(set_seed=lambda seed: events.append(("seed", seed)),
                                 close_envs=lambda envs: events.append(("close", envs)))
        def factory(cfg, **kwargs):
            events.append(("make", cfg.task_ids, kwargs))
            return {"libero_spatial": {4: SimpleNamespace()}}
        selected = {"task_id": 4, "seed": 1000, "init_state_index": 0}
        with patch.object(runner.v1, "rng_fingerprint", return_value="a" * 64), patch.object(runner.b4b, "select_init_state"):
            for condition in ("clean", "low_contrast"):
                with runner.fresh_environment(module, factory, Config([0, 4]), selected, condition, None, records):
                    pass
        self.assertEqual([e[0] for e in events], ["seed", "make", "close"] * 2)
        self.assertEqual([e[1] for e in events if e[0] == "make"], [[4], [4]])
        self.assertTrue(all(r["closed"] for r in records))
        self.assertEqual([r["construction_index"] for r in records], [0, 1])

    def test_failure_closes_environment(self):
        closed, records = [], []
        module = SimpleNamespace(set_seed=lambda _: None, close_envs=closed.append)
        selected = {"task_id": 0, "seed": 1000, "init_state_index": 0}
        with patch.object(runner.v1, "rng_fingerprint", return_value="a" * 64), patch.object(runner.b4b, "select_init_state"):
            with self.assertRaisesRegex(RuntimeError, "episode failed"):
                with runner.fresh_environment(module, lambda *a, **k: {"libero_spatial": {0: SimpleNamespace()}}, Config([0]), selected, "clean", None, records):
                    raise RuntimeError("episode failed")
        self.assertEqual(len(closed), 1)
        self.assertTrue(records[0]["closed"])

    def test_wrong_factory_task_closed_and_rejected(self):
        closed = []
        module = SimpleNamespace(set_seed=lambda _: None, close_envs=closed.append)
        with patch.object(runner.v1, "rng_fingerprint", return_value="a" * 64):
            with self.assertRaisesRegex(ValueError, "wrong task"):
                with runner.fresh_environment(module, lambda *a, **k: {"libero_spatial": {9: object()}}, Config([0]), {"task_id": 0}, "clean", None, []):
                    self.fail("Invalid environment yielded")
        self.assertEqual(len(closed), 1)


class ActualStartGuardTests(unittest.TestCase):
    def objects(self):
        observation = {"x": 1}
        self.calls = []
        state = np.arange(3)
        inner = SimpleNamespace(set_init_state=lambda state: None)
        def reset(**kwargs):
            inner.set_init_state(state)
            return observation, {}
        env = SimpleNamespace(reset=reset, envs=[SimpleNamespace(unwrapped=SimpleNamespace(_env=inner))])
        policy = SimpleNamespace(select_action=lambda batch: self.calls.append(batch))
        expected = {"initial_observation_sha256": runner.v1.digest(observation), "post_reset_rng_sha256": "a" * 64,
                    "init_state_sha256": runner.hashlib.sha256(state.tobytes()).hexdigest()}
        return env, policy, expected

    def test_verified_reset_then_single_first_inference_and_restore(self):
        env, policy, expected = self.objects()
        originals = env.reset, policy.select_action
        pair = []
        with patch.object(runner.v1, "rng_fingerprint", return_value="a" * 64):
            with runner.guard_actual_start(env, policy, expected, pair, None) as evidence:
                env.reset(seed=[1000])
                policy.select_action({})
                policy.select_action({})
        self.assertEqual(evidence, {"seeded_resets": 1, "init_payloads": 1, "first_inferences": 1})
        self.assertEqual(pair, ["a" * 64])
        self.assertEqual((env.reset, policy.select_action), originals)
        self.assertEqual(len(self.calls), 2)

    def test_wrong_reset_stops_before_policy(self):
        env, policy, expected = self.objects()
        original = env.reset
        expected["initial_observation_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "start differs"):
            with runner.guard_actual_start(env, policy, expected, [], None):
                env.reset(seed=[1000])
                policy.select_action({})
        self.assertEqual(self.calls, [])
        self.assertIs(env.reset, original)

    def test_post_reset_rng_rejected(self):
        env, policy, expected = self.objects()
        with patch.object(runner.v1, "rng_fingerprint", return_value="b" * 64):
            with self.assertRaisesRegex(ValueError, "post-reset RNG"):
                with runner.guard_actual_start(env, policy, expected, [], None):
                    env.reset(seed=[1000])
        self.assertEqual(self.calls, [])

    def test_payload_mismatch_rejected_before_inference(self):
        env, policy, expected = self.objects()
        expected["init_state_sha256"] = "b" * 64
        original = env.envs[0].unwrapped._env.set_init_state
        with self.assertRaisesRegex(ValueError, "init payload differs"):
            with runner.guard_actual_start(env, policy, expected, [], None):
                env.reset(seed=[1000])
        self.assertEqual(self.calls, [])
        self.assertIs(env.envs[0].unwrapped._env.set_init_state, original)

    def test_unreset_or_mismatched_paired_rng_rejected(self):
        for do_reset in (False, True):
            env, policy, expected = self.objects()
            with patch.object(runner.v1, "rng_fingerprint", return_value="a" * 64):
                with self.assertRaises(ValueError):
                    with runner.guard_actual_start(env, policy, expected, ["b" * 64], None):
                        if do_reset:
                            env.reset(seed=[1000])
                        policy.select_action({})
            self.assertEqual(self.calls, [])


class GateTests(unittest.TestCase):
    def test_incomplete_gate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text(json.dumps({"status": "failed", "stage": "reset-check"}))
            with self.assertRaisesRegex(ValueError, "Incomplete prerequisite"):
                runner.verify_gate(root, "reset-check", {})


if __name__ == "__main__":
    unittest.main()
