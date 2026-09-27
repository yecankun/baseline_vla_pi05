"""CPU tests for public-source identity/mapping and complete-episode guards."""
import copy
import hashlib
import unittest

from audit_libero_public_source import task_mapping, validate_episode_rows, verified_payload


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"source_task_index": i, "task_instruction": f"fixture task {i}"} for i in range(40)]
        self.native = [{"task_id": i, "task_instruction": f"fixture task {i+30}"} for i in range(10)]
        self.mapping = task_mapping(self.source, self.native)
        self.episode = {"episode_index": 1, "length": 8, "dataset_from_index": 10}
        self.rows = [{"episode_index": 1, "frame_index": i, "index": 10+i,
                      "timestamp": i/10, "task_index": 30, "task": "fixture task 30",
                      "observation.state": [0.] * 8, "action": [0.] * 7} for i in range(8)]

    def check_rows(self):
        return validate_episode_rows(self.rows, self.episode, self.mapping, 10.)

    def test_mapping_and_complete_episode(self):
        self.assertEqual([r["source_task_index"] for r in self.mapping], list(range(30, 40)))
        self.assertEqual(len(self.check_rows()), 8)

    def test_exact_task_text(self):
        self.native[0]["task_instruction"] += " "
        with self.assertRaises(ValueError):
            task_mapping(self.source, self.native)

    def test_ambiguous_source_task(self):
        self.source[0]["task_instruction"] = "fixture task 30"
        with self.assertRaises(ValueError):
            task_mapping(self.source, self.native)

    def test_not_all_tasks(self):
        with self.assertRaises(ValueError):
            task_mapping(self.source, self.native[:-1])

    def test_no_truncated_episode(self):
        self.rows.pop()
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_frame_gap_and_duplicate(self):
        self.rows[4]["frame_index"] = 3
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_timestamp_mismatch(self):
        self.rows[4]["timestamp"] += .01
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_global_index_mismatch(self):
        self.rows[4]["index"] += 1
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_declared_state_cannot_be_imputed(self):
        del self.rows[3]["observation.state"]
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_native_action_bounds_no_clipping(self):
        self.rows[3]["action"][0] = 1.01
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_in_episode_task_switch(self):
        self.rows[3]["task_index"], self.rows[3]["task"] = 31, "fixture task 31"
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_task_text_index_mismatch(self):
        self.rows[3]["task"] = "fixture task 31"
        with self.assertRaises(ValueError):
            self.check_rows()

    def test_lfs_and_git_identity(self):
        data = b"public fixture bytes"
        sha = hashlib.sha256(data).hexdigest()
        entry = {"size": len(data), "lfs_sha256": sha, "blob_id": ""}
        self.assertEqual(verified_payload(data, entry), sha)
        bad = copy.deepcopy(entry)
        bad["lfs_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verified_payload(data, bad)
        entry["lfs_sha256"] = None
        entry["blob_id"] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        self.assertEqual(verified_payload(data, entry), sha)
        with self.assertRaises(ValueError):
            verified_payload(data + b"x", entry)

    def test_transfer_limit(self):
        with self.assertRaises(ValueError):
            verified_payload(b"", {"size": 100_000_001, "lfs_sha256": "", "blob_id": ""})


if __name__ == "__main__":
    unittest.main()
