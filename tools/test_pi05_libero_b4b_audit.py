"""Pure-CPU negative tests for the offline runtime-chain audit."""
from copy import deepcopy
import hashlib
import unittest

from audit_pi05_libero_b4b_result import audit_trace


def trace():
    rows = []
    for task in range(10):
        for index in range(10):
            seed = 1000 + index
            rows.extend([
                {"kind": "state_applied", "task_id": task, "seed": seed, "init_state_index": index,
                 "state_sha256": hashlib.sha256(f"{task}/{index}".encode()).hexdigest()},
                {"kind": "reset", "task_id": task, "seed": seed, "applied_init_state_index": index,
                 "reset_kind": "episode_start"},
                {"kind": "episode_complete", "task_id": task, "seed": seed, "init_state_index": index,
                 "success": False, "steps": 280, "action_min": -1.01, "action_max": 1.02},
            ])
    return rows


class TraceAuditTests(unittest.TestCase):
    def test_failure_scores_and_bound_violations_do_not_erase_trace_evidence(self):
        completed, stats = audit_trace(trace())
        self.assertEqual(len(completed), 100)
        self.assertEqual(sum(row["success"] for row in completed), 0)
        self.assertEqual(stats["seeded_starts"], 100)

    def test_unpaired_missing_payload(self):
        rows = trace()[1:]
        with self.assertRaises(ValueError): audit_trace(rows)

    def test_reset_not_applied_index(self):
        rows = trace()
        rows[1]["applied_init_state_index"] = 2
        with self.assertRaises(ValueError): audit_trace(rows)

    def test_fake_schedule_cannot_replace_payloads(self):
        rows = trace()
        for row in rows:
            if row["kind"] == "state_applied": row["init_state_index"] = 0
        with self.assertRaises(ValueError): audit_trace(rows)

    def test_swapped_episode_and_extra_episode(self):
        rows = trace()
        for bad in (rows[3:6] + rows[:3] + rows[6:], rows + rows[:3]):
            with self.subTest(extra=len(bad)), self.assertRaises(ValueError): audit_trace(bad)

    def test_early_or_duplicate_completion(self):
        rows = trace()
        for bad in ([rows[2]] + rows, rows[:3] + [rows[2]] + rows[3:]):
            with self.subTest(bad=bad[0]["kind"]), self.assertRaises(ValueError): audit_trace(bad)

    def test_nonfinite_actions_and_bad_success(self):
        for key, value in (("action_min", float("nan")), ("action_max", float("inf")),
                           ("steps", 281), ("success", 1)):
            rows = trace()
            rows[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit_trace(rows)

    def test_incomplete_chain(self):
        with self.assertRaises(ValueError): audit_trace(trace()[:-1])

    def test_automatic_reset_pair_is_separate_from_next_start(self):
        rows = trace()
        payload, reset = deepcopy(rows[3:5])
        payload["seed"] = reset["seed"] = None
        reset["reset_kind"] = "automatic"
        rows[2:2] = [payload, reset]
        completed, stats = audit_trace(rows)
        self.assertEqual(completed[0]["automatic_resets"], [1])
        self.assertEqual(completed[1]["init_state_index"], 1)
        self.assertEqual(stats["automatic_resets"], 1)

    def test_inconsistent_repeated_payload_hash_rejected(self):
        rows = trace()
        payload, reset = deepcopy(rows[3:5])
        payload["seed"] = reset["seed"] = None
        reset["reset_kind"] = "automatic"
        payload["state_sha256"] = "f" * 64
        rows[2:2] = [payload, reset]
        with self.assertRaises(ValueError): audit_trace(rows)


if __name__ == "__main__":
    unittest.main()
