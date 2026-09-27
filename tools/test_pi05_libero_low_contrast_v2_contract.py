"""CPU-only synthetic fixtures; never policy or reset-performance evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pi05_libero_low_contrast_contract as v1
import pi05_libero_low_contrast_v2_contract as v2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/libero-spatial-low-contrast-protocol-v2.json"


def synthetic_rows(stage):
    rows = []
    for identity in v2.schedule(stage):
        digest = hashlib.sha256(str(identity["task_id"]).encode()).hexdigest()
        for condition in ("clean", "low_contrast"):
            row = {**identity, "condition": condition, "init_state_sha256": digest,
                   "initial_observation_sha256": "b" * 64}
            if stage == "reset-check":
                row["post_reset_rng_sha256"] = "c" * 64
            else:
                row.update(first_inference_rng_sha256="c" * 64, success=True, steps=82)
            rows.append(row)
    return rows


class DeltaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = v2.load_protocol(PROTOCOL)

    def assert_invalid(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                v2.load_protocol(path)

    def test_actual_parent_json_and_implementation_are_byte_pinned(self):
        parent = self.protocol["parent_protocol"]
        self.assertEqual(hashlib.sha256((ROOT / parent["path"]).read_bytes()).hexdigest(), parent["sha256"])
        for path, expected in self.protocol["implementation_sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)
        parent_contract = v1.load_protocol(ROOT / parent["path"])
        self.assertEqual(parent_contract["corruption"]["alpha"], .45)
        self.assertEqual(parent_contract["policy"]["optimizer_steps"], 0)
        self.assertFalse(parent_contract["controller"]["additional_wrapper_clipping_allowed"])

    def test_only_new_lifecycle_and_stage_delta_is_accepted(self):
        self.assertEqual(set(self.protocol), {"schema_version", "protocol_id", "evidence_scope", "parent_protocol",
                                            "implementation_sources", "lifecycle", "stages", "claims"})
        self.assertFalse(self.protocol["lifecycle"]["reuse_environment_between_conditions_or_tasks"])
        self.assertFalse(self.protocol["lifecycle"]["additional_state_forcing_allowed"])
        self.assertFalse(self.protocol["lifecycle"]["pair_hash_equality_may_be_relaxed"])

    def test_changed_scope_seed_stage_or_hash_is_rejected(self):
        changes = [("lifecycle", "reuse_environment_between_conditions_or_tasks", True),
                   ("lifecycle", "additional_state_forcing_allowed", True),
                   ("lifecycle", "native_num_steps_wait", 20),
                   ("lifecycle", "native_horizon", 20),
                   ("lifecycle", "seed_python_numpy_torch_before_environment_construction", 1001),
                   ("lifecycle", "seed_python_numpy_torch_before_official_rollout", 1001),
                   ("lifecycle", "pair_hash_equality_may_be_relaxed", True),
                   ("claims", "architecture_gain", True),
                   ("parent_protocol", "sha256", "d" * 64)]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                data = copy.deepcopy(self.protocol)
                data[section][key] = value
                self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["stages"]["check"]["task_ids"] = [0]
        self.assert_invalid(data)

    def test_unknown_missing_and_boolean_integer_keys_are_rejected(self):
        for section in ("lifecycle", "stages", "claims"):
            data = copy.deepcopy(self.protocol)
            data[section]["extra_override"] = True
            self.assert_invalid(data)
            data = copy.deepcopy(self.protocol)
            del data[section]
            self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["stages"]["reset-check"]["policy_loaded"] = 0
        self.assert_invalid(data)
        data = copy.deepcopy(self.protocol)
        data["schema_version"] = 2.0
        self.assert_invalid(data)

    def test_duplicate_protocol_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            content = PROTOCOL.read_text(encoding="utf-8").replace('"schema_version": 2,', '"schema_version": 2, "schema_version": 2,', 1)
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                v2.load_protocol(path)

    def test_actual_parent_or_source_drift_is_rejected_without_modifying_files(self):
        real_sha = v2._sha256
        for filename in (self.protocol["parent_protocol"]["path"], *self.protocol["implementation_sources"]):
            target = ROOT / filename
            with patch.object(v2, "_sha256", side_effect=lambda path: "0" * 64 if path == target else real_sha(path)):
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    v2.load_protocol(PROTOCOL)


class V2PolicyPairTests(unittest.TestCase):
    def test_stage_sizes_task4_regression_and_no_parent_task0_only_reuse(self):
        self.assertEqual([row["task_id"] for row in v2.schedule("check")], [0, 4])
        self.assertEqual([row["task_id"] for row in v2.schedule("screen")], list(range(10)))
        self.assertEqual([row["task_id"] for row in v2.schedule("reset-check")], list(range(10)))
        with self.assertRaises(ValueError):
            v2.schedule("v1-check")
        with self.assertRaises(ValueError):
            v2.summarize_pairs(synthetic_rows("check")[:2], "check")
        with self.assertRaises(ValueError):
            v2.summarize_pairs(synthetic_rows("reset-check"), "reset-check")

    def test_check_summary_and_order_independence(self):
        rows = synthetic_rows("check")
        rows[-1]["success"] = False
        result = v2.summarize_pairs(rows, "check")
        self.assertEqual(result, v2.summarize_pairs(list(reversed(rows)), "check"))
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["episode_count"], 4)
        self.assertEqual(result["clean_success_count"], 2)
        self.assertEqual(result["low_contrast_success_count"], 1)
        self.assertEqual(result["delta_pp"], -50)
        self.assertEqual(result["relative_drop"], .5)
        self.assertEqual(result["per_task"][1]["task_id"], 4)

    def test_screen_metric_schema_and_values_are_identical_to_parent(self):
        rows = synthetic_rows("screen")
        rows[3]["success"] = False
        rows[4]["success"] = False
        rows[6]["success"] = rows[7]["success"] = False
        rows[9]["success"] = False
        parent = v1.summarize_pairs(rows, "screen")
        current = v2.summarize_pairs(rows, "screen")
        self.assertEqual(set(parent), set(current))
        for key in parent:
            if key not in {"protocol_id", "evidence_scope"}:
                self.assertEqual(parent[key], current[key], key)

    def test_clean_zero_relative_drop_remains_null(self):
        rows = synthetic_rows("check")
        rows[0]["success"] = rows[2]["success"] = False
        result = v2.summarize_pairs(rows, "check")
        self.assertIsNone(result["relative_drop"])
        self.assertEqual(result["delta_pp"], 100)

    def test_missing_duplicate_foreign_or_bad_typed_policy_rows_rejected(self):
        for field, value in (("task_id", 1), ("seed", 1001), ("init_state_index", 1),
                             ("condition", "blur"), ("success", 1), ("steps", 0), ("steps", 281),
                             ("steps", True), ("steps", 82.0), ("task_id", False)):
            rows = synthetic_rows("check")
            rows[0][field] = value
            with self.assertRaises(ValueError):
                v2.summarize_pairs(rows, "check")
        rows = synthetic_rows("check")
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            v2.summarize_pairs(rows, "check")

    def test_all_three_policy_hash_gates_remain_strict(self):
        for field in ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256"):
            for value in (None, "bad", "A" * 64, "d" * 64):
                rows = synthetic_rows("check")
                rows[1][field] = value
                with self.assertRaises(ValueError):
                    v2.summarize_pairs(rows, "check")


class ResetPairTests(unittest.TestCase):
    def test_complete_reset_pairs_have_no_policy_score(self):
        rows = synthetic_rows("reset-check")
        result = v2.validate_reset_pairs(rows)
        self.assertEqual(result, v2.validate_reset_pairs(list(reversed(rows))))
        self.assertEqual(result["pair_count"], 10)
        self.assertEqual(result["row_count"], 20)
        self.assertFalse(result["policy_loaded"])
        self.assertFalse(result["policy_inference"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertNotIn("success_rate", result)
        self.assertNotIn("delta_pp", result)
        self.assertNotIn("episode_count", result)
        self.assertEqual(result["per_task"][-1]["task_id"], 9)

    def test_reset_missing_duplicate_foreign_or_badtyped_rows_rejected(self):
        rows = synthetic_rows("reset-check")
        for bad in (rows[:-1], rows + rows[:1], None, "not rows", [None] * 20):
            with self.assertRaises(ValueError):
                v2.validate_reset_pairs(bad)
        for field, value in (("condition", "foreign"), ("task_id", 10), ("seed", True), ("init_state_index", 1)):
            rows = synthetic_rows("reset-check")
            rows[0][field] = value
            with self.assertRaises(ValueError):
                v2.validate_reset_pairs(rows)
        rows = synthetic_rows("reset-check")
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            v2.validate_reset_pairs(rows)

    def test_all_three_reset_hashes_required_and_matched(self):
        for field in ("init_state_sha256", "initial_observation_sha256", "post_reset_rng_sha256"):
            for value in (None, "", "D" * 64, "d" * 64):
                rows = synthetic_rows("reset-check")
                rows[1][field] = value
                with self.assertRaises(ValueError):
                    v2.validate_reset_pairs(rows)

    def test_reset_rejects_policy_metrics_and_contradictory_execution_claims(self):
        for field, value in (("success", True), ("steps", 0), ("first_inference_rng_sha256", "c" * 64),
                             ("policy_loaded", True), ("policy_inference", True), ("optimizer_steps", 1),
                             ("policy_loaded", 0), ("optimizer_steps", False),
                             ("policy_steps", 1), ("policy_steps", False), ("policy_steps", 0.0)):
            rows = synthetic_rows("reset-check")
            rows[0][field] = value
            with self.assertRaises(ValueError):
                v2.validate_reset_pairs(rows)
        rows = synthetic_rows("reset-check")
        for row in rows:
            row["policy_steps"] = 0
        self.assertEqual(v2.validate_reset_pairs(rows)["row_count"], 20)


if __name__ == "__main__":
    unittest.main()
