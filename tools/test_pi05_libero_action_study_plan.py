"""Synthetic metadata only; no source payload, model or network."""
from copy import deepcopy
from pathlib import Path
import unittest
import plan_pi05_libero_action_ablation as planner

PLAN = Path(__file__).resolve().parents[1] / "docs/libero-action-ablation-study-plan-v1.json"


class MetadataPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = planner.read(PLAN)
        self.entries = [{"episode_index": i, "source_task_index": 39, "record_count": 14} for i in range(30)]

    def test_prospective_hash_and_no_execution_authorization(self):
        self.assertEqual(planner.sha(PLAN), planner.PLAN_SHA)
        self.assertFalse(self.plan["gates"]["optimization_authorized"])
        self.assertFalse(self.plan["gates"]["new_downloads_authorized"])

    def test_deterministic_complete_episode_split_order_independent(self):
        result = planner.select_split(self.entries, [], self.plan)
        self.assertEqual(result, planner.select_split(list(reversed(self.entries)), [], self.plan))
        self.assertEqual(len(result["episodes"]), 12)
        self.assertEqual(len(result["split"]["train_episode_indices"]), 8)
        self.assertEqual(len(result["split"]["validation_episode_indices"]), 4)
        self.assertFalse(set(result["split"]["train_episode_indices"]) & set(result["split"]["validation_episode_indices"]))
        self.assertFalse(result["training_ready"])

    def test_incomplete_task_coverage_never_provisionally_selects(self):
        result = planner.select_split(self.entries, [77], self.plan)
        self.assertEqual(result["episodes"], [])
        self.assertIsNone(result["split"])

    def test_old_task_and_short_episodes_excluded(self):
        entries = self.entries + [{"episode_index": i, "source_task_index": 39, "record_count": 14} for i in (1400, 1401, 1402)]
        entries += [{"episode_index": 50, "source_task_index": 38, "record_count": 14},
                    {"episode_index": 51, "source_task_index": 39, "record_count": 6}]
        result = planner.select_split(entries, [], self.plan)
        self.assertEqual(result["eligible_episode_count"], 30)
        self.assertFalse({1400, 1401, 1402, 50, 51} & {r["episode_index"] for r in result["episodes"]})

    def test_insufficient_episodes_and_duplicate_inventory(self):
        self.assertEqual(planner.select_split(self.entries[:11], [], self.plan)["episodes"], [])
        with self.assertRaises(ValueError):
            planner.select_split(self.entries + self.entries[:1], [], self.plan)

    def test_excess_row_budget_does_not_truncate_or_replace(self):
        for r in self.entries:
            r["record_count"] = 2000
        with self.assertRaisesRegex(ValueError, "no truncation"):
            planner.select_split(self.entries, [], self.plan)

    def projection(self):
        meta = [{"episode_index": 4, "length": 7, "dataset_from_index": 100, "dataset_to_index": 107},
                {"episode_index": 5, "length": 7, "dataset_from_index": 107, "dataset_to_index": 114}]
        rows = [{"episode_index": 4, "task_index": 39, "frame_index": i, "index": 100+i,
                 "timestamp": i/10} for i in range(7)]
        return meta, rows

    def test_metadata_projection_records_full_task_and_missing_ids(self):
        meta, rows = self.projection()
        entries, missing = planner.task_inventory(meta, rows, 10)
        self.assertEqual(entries, [{"episode_index": 4, "source_task_index": 39, "record_count": 7}])
        self.assertEqual(missing, [5])

    def test_projection_extra_fields_task_changes_gaps_and_partial_rejected(self):
        meta, rows = self.projection()
        for changed in (rows[:-1], [{**r, "action": [0]*7} for r in rows],
                        [{**r, "task_index": 38 if i == 2 else 39} for i, r in enumerate(rows)],
                        [{**r, "frame_index": 0} for r in rows],
                        [{**r, "timestamp": 99.} for r in rows]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                planner.task_inventory(meta, changed, 10)

    def test_metadata_indices_and_coercions_fail_closed(self):
        meta, rows = self.projection()
        for value in (True, -1, 1.5, "1"):
            with self.assertRaises(ValueError):
                planner.integral(value)
        bad = deepcopy(meta)
        bad[0]["length"] = 9
        with self.assertRaises(ValueError):
            planner.task_inventory(bad, rows, 10)


if __name__ == "__main__":
    unittest.main()
