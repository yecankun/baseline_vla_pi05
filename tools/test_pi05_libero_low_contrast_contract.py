"""CPU-only negative contract tests and literal visual-wrapper mechanics."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi05_libero_low_contrast_contract import load_protocol, schedule, summarize_pairs
from vla_benchmark_contract import VisualCorruption, apply_visual_corruption
from vla_visual_noise_wrapper import EvaluationVisualObservationWrapper


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/libero-spatial-low-contrast-protocol-v1.json"


def fixtures(stage="check"):
    rows = []
    for identity in schedule(stage):
        digest = hashlib.sha256(str(identity["task_id"]).encode()).hexdigest()
        for condition in ("clean", "low_contrast"):
            rows.append({**identity, "condition": condition, "success": True, "steps": 82,
                         "init_state_sha256": digest, "initial_observation_sha256": "b" * 64,
                         "first_inference_rng_sha256": "c" * 64})
    return rows


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def assert_invalid(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_actual_protocol_has_complete_pins(self):
        provenance = self.protocol["provenance"]
        self.assertEqual(len(provenance["checkpoint"]["sha256_by_file"]), 6)
        self.assertEqual(len(provenance["runtime_sources"]), 7)
        for filename, digest in provenance["source_files"].items():
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), digest)
        historical = provenance["historical_protocol"]
        self.assertEqual(hashlib.sha256((ROOT / historical["path"]).read_bytes()).hexdigest(), historical["sha256"])

    def test_native_saturation_and_diagnostic_overshoot_are_distinct(self):
        controller = self.protocol["controller"]
        self.assertTrue(controller["native_osc_clip_and_scale_retained"])
        self.assertTrue(controller["native_gripper_sign_and_internal_clip_retained"])
        self.assertFalse(controller["additional_wrapper_clipping_allowed"])
        self.assertFalse(controller["raw_overshoot_is_strict_failure"])

    def test_semantic_edits_fail_closed(self):
        changes = [
            ("controller", "additional_wrapper_clipping_allowed", True),
            ("controller", "raw_overshoot_is_strict_failure", True),
            ("controller", "native_osc_clip_and_scale_retained", False),
            ("corruption", "alpha", 0.7), ("corruption", "seed", 0),
            ("corruption", "image_paths", ["pixels.image"]),
            ("policy", "compile_model", False), ("policy", "optimizer_steps", 1),
            ("benchmark", "native_horizon", 20),
            ("schedule", "reuse_historical_clean_scores", True),
            ("schedule", "reuse_check_episodes_for_screen", True),
            ("pairing", "equal_first_inference_rng_required", False),
            ("claims", "real_system_validated", True),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                data = copy.deepcopy(self.protocol)
                data[section][key] = value
                self.assert_invalid(data)

    def test_missing_and_foreign_critical_keys_fail(self):
        for section in ("controller", "pairing", "provenance"):
            data = copy.deepcopy(self.protocol)
            data[section].pop(next(iter(data[section])))
            self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["extra_override"] = True
        self.assert_invalid(data)

    def test_boolean_integer_confusion_fails(self):
        data = copy.deepcopy(self.protocol)
        data["schema_version"] = True
        self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["schedule"]["full_episodes"] = 1
        self.assert_invalid(data)

    def test_missing_or_invalid_resource_pins_fail(self):
        for group in ("source_files", "runtime_sources"):
            data = copy.deepcopy(self.protocol)
            data["provenance"][group].pop(next(iter(data["provenance"][group])))
            self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["provenance"]["checkpoint"]["sha256_by_file"].pop("config.json")
        self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["provenance"]["source_files"]["tools/run_pi05_libero_b4b.py"] = "not-a-hash"
        self.assert_invalid(data)


class PairSummaryTests(unittest.TestCase):
    def test_fixed_stage_sizes_and_fresh_objects(self):
        self.assertEqual(schedule("check"), [{"task_id": 0, "seed": 1000, "init_state_index": 0}])
        self.assertEqual([item["task_id"] for item in schedule("screen")], list(range(10)))
        altered = schedule("check")
        altered[0]["seed"] = 999
        self.assertEqual(schedule("check")[0]["seed"], 1000)
        with self.assertRaises(ValueError):
            schedule("historical")

    def test_full_pair_is_order_independent(self):
        rows = fixtures("screen")
        result = summarize_pairs(rows, "screen")
        self.assertEqual(result, summarize_pairs(list(reversed(rows)), "screen"))
        self.assertEqual(result["episode_count"], 20)
        self.assertEqual(result["clean_success_count"], 10)
        self.assertEqual(result["delta_pp"], 0.0)
        self.assertEqual(result["paired_outcomes"]["both_success"], 10)
        self.assertFalse(result["statistical_significance_claim"])

    def test_four_outcomes_and_absolute_relative_changes(self):
        rows = fixtures("screen")
        rows[3]["success"] = False  # task1: clean only
        rows[4]["success"] = False  # task2: low contrast only
        rows[6]["success"] = rows[7]["success"] = False  # task3: both fail
        rows[9]["success"] = False  # task4: clean only
        result = summarize_pairs(rows, "screen")
        self.assertEqual(result["paired_outcomes"], {"both_success": 6, "clean_only": 2, "low_contrast_only": 1, "both_failure": 1})
        self.assertEqual(result["clean_success_count"], 8)
        self.assertEqual(result["low_contrast_success_count"], 7)
        self.assertEqual(result["delta_pp"], -10.0)
        self.assertEqual(result["relative_drop"], 0.125)
        self.assertEqual(result["per_task"][2]["delta_pp"], 100)

    def test_clean_zero_relative_drop_is_null_not_fabricated_denominator(self):
        rows = fixtures()
        rows[0]["success"] = False
        result = summarize_pairs(rows, "check")
        self.assertIsNone(result["relative_drop"])
        self.assertEqual(result["delta_pp"], 100)

    def test_mismatched_runtime_fingerprints_fail(self):
        for field in ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256"):
            with self.subTest(field=field):
                rows = fixtures()
                rows[1][field] = "d" * 64
                with self.assertRaisesRegex(ValueError, "pair mismatch"):
                    summarize_pairs(rows, "check")

    def test_missing_malformed_or_nonstring_runtime_fingerprints_fail(self):
        for field in ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256"):
            for value in (None, "", "A" * 64, "d" * 63, 9):
                with self.subTest(field=field, value=value):
                    rows = fixtures()
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        summarize_pairs(rows, "check")

    def test_missing_duplicate_or_foreign_episodes_fail(self):
        candidates = [fixtures()[:1], fixtures() + fixtures()[:1], [fixtures()[0], fixtures()[0]], fixtures("screen")]
        for rows in candidates:
            with self.assertRaises(ValueError):
                summarize_pairs(rows, "check")
        for field, value in (("task_id", 1), ("seed", 1001), ("init_state_index", 1), ("condition", "blur")):
            rows = fixtures()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                summarize_pairs(rows, "check")

    def test_strict_success_and_integer_types(self):
        for value in (1, 0, "true", None, np.bool_(True)):
            rows = fixtures()
            rows[0]["success"] = value
            with self.assertRaises(ValueError):
                summarize_pairs(rows, "check")
        for field in ("task_id", "seed", "init_state_index", "steps"):
            for value in (True, 1.0, "1", None):
                rows = fixtures()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    summarize_pairs(rows, "check")

    def test_native_horizon_and_nonmapping_rows(self):
        for steps in (0, -1, 281):
            rows = fixtures()
            rows[0]["steps"] = steps
            with self.assertRaises(ValueError):
                summarize_pairs(rows, "check")
        with self.assertRaises(ValueError):
            summarize_pairs([None, None], "check")


class VisualMechanismTests(unittest.TestCase):
    def test_low_contrast_literal_formula_and_no_global_numpy_rng_consumption(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        snapshot = image.copy()
        original_rng = np.random.get_state()
        try:
            np.random.seed(1000)
            before = np.random.get_state()
            first = apply_visual_corruption(image, sample_key="task0/init0/frame0", view_key="pixels.image",
                                            corruption=VisualCorruption("low_contrast", 2), seed=20260911)
            after = np.random.get_state()
            unit = image.astype(np.float32) / 255.0
            mean = unit.mean(axis=(0, 1), keepdims=True)
            expected = np.rint((mean + 0.45 * (unit - mean)) * 255.0).astype(np.uint8)
            np.testing.assert_array_equal(first, expected)
            np.testing.assert_array_equal(image, snapshot)
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[0], after[0])
            self.assertEqual(before[2:], after[2:])
            self.assertEqual(first.dtype, np.uint8)
            self.assertLess(float(first.std()), float(image.std()))
        finally:
            np.random.set_state(original_rng)

    def test_wrapper_changes_only_both_images_and_forwards_raw_action(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        observation = {"pixels": {"image": image, "image2": np.flip(image, axis=0).copy()},
                       "agent_pos": np.arange(8, dtype=np.float64)}

        class ToyEnvironment:
            def reset(self):
                return observation, {"reset_marker": "untouched"}

            def step(self, action):
                self.received = action
                return observation, 1.0, True, False, {"success": True}

        environment = ToyEnvironment()
        wrapper = EvaluationVisualObservationWrapper(environment, image_paths=("pixels.image", "pixels.image2"),
                    corruption=VisualCorruption("low_contrast", 2), seed=20260911,
                    episode_key="task0/init0", verify_determinism=True)
        result, info = wrapper.reset()
        self.assertEqual(info, {"reset_marker": "untouched"})
        self.assertIs(result["agent_pos"], observation["agent_pos"])
        for key in ("image", "image2"):
            self.assertFalse(np.array_equal(result["pixels"][key], observation["pixels"][key]))
            self.assertEqual(result["pixels"][key].shape, (256, 256, 3))
            self.assertEqual(result["pixels"][key].dtype, np.uint8)
        action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.00916]], dtype=np.float32)
        _, reward, terminated, truncated, info = wrapper.step(action)
        self.assertIs(environment.received, action)
        self.assertLess(float(environment.received[0, 6]), -1.0)
        self.assertEqual((reward, terminated, truncated, info), (1.0, True, False, {"success": True}))
        self.assertTrue(all(view["deterministic_duplicate"] for row in wrapper.frame_records for view in row["views"]))


class RunnerBoundaryHelperTests(unittest.TestCase):
    def test_digest_covers_nonimage_values_dtype_shape_and_dict_order(self):
        from run_pi05_libero_low_contrast import digest

        first = {"state": np.array([1, 2], dtype=np.float32), "task": "native instruction"}
        self.assertEqual(digest(first), digest({"task": first["task"], "state": first["state"].copy()}))
        for changed in ({"state": np.array([1, 3], dtype=np.float32), "task": first["task"]},
                        {"state": first["state"].astype(np.float64), "task": first["task"]},
                        {"state": first["state"].reshape(1, 2), "task": first["task"]},
                        {"state": first["state"], "task": "different instruction"}):
            self.assertNotEqual(digest(first), digest(changed))
        with self.assertRaises(ValueError):
            digest(np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            digest({"scalar": float("nan")})

    def test_image_boundary_rejects_state_schema_camera_and_dtype_changes(self):
        from run_pi05_libero_low_contrast import check_only_images_changed

        before = {"pixels": {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8),
                             "image2": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
                  "agent_pos": np.arange(8, dtype=np.float32)}
        valid = copy.deepcopy(before)
        valid["pixels"]["image"] += 1
        valid["pixels"]["image2"] += 2
        check_only_images_changed(before, valid)
        changed = copy.deepcopy(valid)
        changed["agent_pos"][0] = 99
        with self.assertRaisesRegex(ValueError, "non-image"):
            check_only_images_changed(before, changed)
        changed = copy.deepcopy(valid)
        changed["pixels"].pop("image2")
        with self.assertRaises(ValueError):
            check_only_images_changed(before, changed)
        for key in ("image", "image2"):
            for image in (valid["pixels"][key].astype(np.float32), valid["pixels"][key][0]):
                changed = copy.deepcopy(valid)
                changed["pixels"][key] = image
                with self.assertRaises(ValueError):
                    check_only_images_changed(before, changed)
        changed = copy.deepcopy(valid)
        changed["diagnostic_targets"] = {}
        with self.assertRaises(ValueError):
            check_only_images_changed(before, changed)

    def test_hooks_restore_exact_original_inherited_and_own_attributes(self):
        from run_pi05_libero_low_contrast import Hooks

        class Toy:
            def operation(self):
                return "original"

        obj = Toy()
        own = object()
        obj.own = own
        hooks = Hooks()
        try:
            hooks.set(obj, "operation", lambda: "first hook")
            hooks.set(obj, "operation", lambda: "second hook")
            hooks.set(obj, "own", "temporary")
            self.assertEqual(obj.operation(), "second hook")
        finally:
            hooks.close()
        self.assertEqual(obj.operation(), "original")
        self.assertNotIn("operation", vars(obj))
        self.assertIs(obj.own, own)
        self.assertEqual(hooks.saved, [])
        hooks.close()


if __name__ == "__main__":
    unittest.main()
