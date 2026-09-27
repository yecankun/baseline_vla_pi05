"""Dependency-light protocol tests: no LeRobot, policy, checkpoint, or rollout."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.run_diffusion_pusht_baseline import (
    PROTOCOL, aggregate, canonical_hash, info_metrics, output_directory,
    parse_args, seed_plan, validate_native_action, wilson95,
    verify_migration_pin, verify_smoke_report,
    run, _seed, _validate_policy,
    verify_environment_parity, validate_env_kwargs,
)


def episode(seed, success=False, coverage=0.2, outside=0):
    return {"seed": seed, "is_success": success, "max_coverage": coverage,
            "native_action_out_of_bounds_steps": outside,
            "native_action_out_of_bounds_elements": outside}


def unused_runtime_function(*args, **kwargs):
    raise AssertionError("policy/environment must not be called in this failure test")


class TestFrozenPushTProtocol(unittest.TestCase):
    def test_frozen_plan_and_hash_are_stable(self):
        self.assertEqual(seed_plan("smoke"), [20260913])
        self.assertEqual(seed_plan("benchmark"), list(range(100000, 100100)))
        self.assertEqual(PROTOCOL["benchmark_max_steps"], 300)
        self.assertEqual(PROTOCOL["smoke_max_steps"], 20)
        self.assertEqual(PROTOCOL["num_inference_steps"], 100)
        self.assertEqual(PROTOCOL["n_action_steps"], 8)
        self.assertEqual(canonical_hash(PROTOCOL), canonical_hash(dict(reversed(list(PROTOCOL.items())))))

    def test_explicit_execution_gate(self):
        for stage in ("smoke", "benchmark"):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--stage", stage])
                self.assertTrue(parse_args(["--stage", stage, "--execute", "--migration-report-sha256", "a" * 64]).execute)
        self.assertEqual(parse_args([]).stage, "preflight")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--stage", "preflight", "--execute"])

    def test_attempt_paths_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = output_directory("smoke", root=root)
            self.assertEqual(output.name, "diffusion_pusht_smoke_v1")
            output.mkdir(parents=True, exist_ok=False)
            with self.assertRaises(FileExistsError):
                output.mkdir(parents=True, exist_ok=False)
            self.assertNotEqual(output, output_directory("smoke", "retry1", root))
        for attempt in ("../x", "a/b", "a\\b", "", ".", "x" * 49):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                output_directory("smoke", attempt)

    def test_true_info_values_not_reward(self):
        result = info_metrics({"is_success": np.bool_(False), "coverage": np.float64(0.1), "reward": 1.0}, terminal=False)
        self.assertEqual(result, {"is_success": False, "coverage": 0.1, "source": "info"})
        with self.assertRaises(ValueError):
            info_metrics({"reward": 1.0}, terminal=True)

    def test_final_info_masks_and_vector_scalar_support(self):
        info = {"is_success": np.array([False]), "coverage": np.array([0.1]),
                "final_info": np.array([{"is_success": True, "coverage": 0.96}], dtype=object),
                "_final_info": np.array([True])}
        self.assertEqual(info_metrics(info, terminal=True)["coverage"], 0.96)
        self.assertEqual(info_metrics(info, terminal=False)["coverage"], 0.1)
        info["_final_info"] = np.array([False])
        self.assertEqual(info_metrics(info, terminal=True)["coverage"], 0.1)

    def test_bad_info_fails_closed(self):
        for value in (float("nan"), float("inf"), -0.1, 1.01, "0.9", True, [0.1, 0.2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                info_metrics({"is_success": False, "coverage": value}, terminal=False)
        for success in (1, "true", [True, False]):
            with self.subTest(success=success), self.assertRaises(ValueError):
                info_metrics({"is_success": success, "coverage": 0.1}, terminal=False)
        with self.assertRaises(ValueError):
            info_metrics({"final_info": None}, terminal=True)

    def test_native_out_of_bounds_observed_not_clipped(self):
        original = np.array([[-0.25, 512.5]], dtype=np.float32)
        result, count = validate_native_action(original, [0, 0], [512, 512])
        self.assertIs(result, original)
        self.assertEqual(count, 2)
        self.assertTrue(np.array_equal(result, [[-0.25, 512.5]]))

    def test_native_action_invalid_shape_dtype_or_finite_rejected(self):
        for action in (np.zeros(2), np.zeros((2, 2)), np.ones((1, 2), dtype=int), np.array([[float("nan"), 1.0]])):
            with self.subTest(shape=action.shape), self.assertRaises(ValueError):
                validate_native_action(action, [0, 0], [512, 512])
        with self.assertRaises(ValueError):
            validate_native_action(np.ones((1, 2)), [-1, -1], [1, 1])

    def test_wilson_limits_counts(self):
        self.assertIsNone(wilson95(0, 0))
        self.assertAlmostEqual(wilson95(100, 100)[0], 0.9630065, places=6)
        self.assertAlmostEqual(wilson95(0, 100)[1], 0.0369935, places=6)
        self.assertAlmostEqual(wilson95(50, 100)[0], 1 - wilson95(50, 100)[1])
        with self.assertRaises(ValueError):
            wilson95(2, 1)

    def test_complete_and_partial_counts_no_success_required(self):
        rows = [episode(seed) for seed in seed_plan("benchmark")]
        result = aggregate(rows, "benchmark")
        self.assertTrue(result["strict_protocol_pass"])
        self.assertEqual(result["successes"], 0)
        self.assertAlmostEqual(result["mean_episode_max_coverage"], 0.2)
        self.assertFalse(aggregate(rows[:1], "benchmark")["strict_protocol_pass"])
        self.assertFalse(aggregate([], "benchmark")["seed_schedule_complete"])
        rows[0]["native_action_out_of_bounds_steps"] = 1
        self.assertFalse(aggregate(rows, "benchmark")["strict_protocol_pass"])

    def test_smoke_is_never_a_benchmark_score(self):
        result = aggregate([episode(20260913, True, 1.0)], "smoke")
        self.assertTrue(result["strict_protocol_pass"])
        self.assertFalse(result["benchmark_score_claim_allowed"])

    def test_seed_parity_no_duplicate_or_reordered_episodes(self):
        for seeds in ([100001], [100000, 100000], [100001, 100000]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                aggregate([episode(seed) for seed in seeds], "benchmark")

    def test_migration_sha_must_match_before_load(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "migration_report.json"
            path.write_text("{}", encoding="utf-8")
            expected = hashlib.sha256(b"{}").hexdigest()
            self.assertEqual(verify_migration_pin(path, expected), expected)
            for invalid in ("0" * 64, "a" * 63, "x" * 64):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    verify_migration_pin(path, invalid)

    def test_benchmark_requires_completed_same_binding_smoke(self):
        binding = {"migration_report_sha256": "a" * 64, "model_sha256": "b" * 64}
        report = {"stage": "smoke", "status": "completed", "strict_protocol_pass": True,
                  "protocol_sha256": canonical_hash(PROTOCOL), "checkpoint_binding": binding,
                  "episodes": [episode(20260913)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(verify_smoke_report(path, binding)["path"], str(path.resolve()))
            for field, bad in (("status", "failed"), ("strict_protocol_pass", False), ("episodes", []), ("checkpoint_binding", {}), ("protocol_sha256", "0" * 64)):
                with self.subTest(field=field):
                    invalid = {**report, field: bad}
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        verify_smoke_report(path, binding)

    def test_failed_run_preserves_status_report_and_refuses_retry(self):
        runtime = {name: unused_runtime_function for name in ("load_verified", "_make_environment", "preprocess_observation", "add_envs_task", "make_env_pre_post_processors")}
        runtime.update({"np": np, "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt"
            report = run("smoke", path, runtime)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["strict_protocol_pass"])
            self.assertEqual(report["episodes"], [])
            status = json.loads((path / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["exit_code"], 1)
            self.assertTrue((path / "started.json").is_file())
            self.assertTrue((path / "report.json").is_file())
            with self.assertRaises(FileExistsError):
                run("smoke", path, runtime)

    def test_episode_seed_reseeds_all_generators(self):
        calls = []
        runtime = {"np": SimpleNamespace(random=SimpleNamespace(seed=lambda seed: calls.append(("numpy", seed)))),
                   "torch": SimpleNamespace(manual_seed=lambda seed: calls.append(("torch", seed)), cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))))}
        _seed(100000, runtime)
        import random
        first = random.random()
        _seed(100000, runtime)
        self.assertEqual(random.random(), first)
        self.assertEqual(calls, [(name, 100000) for _ in range(2) for name in ("numpy", "torch", "cuda")])

    def test_checkpoint_config_preserves_none_but_effective100(self):
        config = SimpleNamespace(**{name: PROTOCOL[name] for name in ("n_obs_steps", "horizon", "n_action_steps", "noise_scheduler_type", "num_train_timesteps")}, num_inference_steps=None)
        param = SimpleNamespace(dtype="torch.float32", device=SimpleNamespace(type="cuda"), requires_grad=False)
        policy = SimpleNamespace(config=config, diffusion=SimpleNamespace(num_inference_steps=100), modules=lambda: [], parameters=lambda: [param])
        _validate_policy(policy)
        config.num_inference_steps = 100
        with self.assertRaises(ValueError):
            _validate_policy(policy)
        config.num_inference_steps = None
        for field, value in (("dtype", "torch.float16"), ("requires_grad", True), ("device", SimpleNamespace(type="cpu"))):
            original = getattr(param, field)
            setattr(param, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_policy(policy)
            setattr(param, field, original)

    def test_smoke_source_hashes_ignore_paths_but_not_hash_drift(self):
        names = ("runner", "load_verified", "_make_environment", "preprocess_observation", "add_envs_task", "make_env_pre_post_processors")
        sources = {name: {"path": "/old/" + name, "sha256": "a" * 64} for name in names}
        report = {"stage": "smoke", "status": "completed", "strict_protocol_pass": True,
                  "protocol_sha256": canonical_hash(PROTOCOL), "checkpoint_binding": {},
                  "episodes": [episode(20260913)], "source_files": sources}
        current = {name: {**item, "path": "/new/" + name} for name, item in sources.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            verify_smoke_report(path, {}, current_sources=current)
            for name in names:
                drifted = {**current, name: {"sha256": "b" * 64}}
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verify_smoke_report(path, {}, current_sources=drifted)

    def test_environment_and_runtime_parity(self):
        kwargs = {"obs_type": "pixels_agent_pos", "render_mode": "rgb_array", "visualization_width": 384,
                  "visualization_height": 384, "max_episode_steps": 300}
        validate_env_kwargs(kwargs)
        with self.assertRaises(ValueError):
            validate_env_kwargs({**kwargs, "max_episode_steps": 20})
        reference = {"environment_gym_kwargs": kwargs, "environment_step_source_sha256": "a" * 64,
                     "package_versions": {"lerobot": "0.4.4"}}
        verify_environment_parity(reference, dict(reference))
        for key in reference:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_environment_parity(reference, {**reference, key: None})


if __name__ == "__main__":
    unittest.main()
