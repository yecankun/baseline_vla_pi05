"""Evidence isolation checks for the retrospective review, without hardware."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_real10_reasoning as review


class EvidenceIsolationTests(unittest.TestCase):
    def prepare(self, out):
        with contextlib.redirect_stdout(io.StringIO()):
            review.prepare(out)

    def test_post_action_values_cannot_change_review_packets(self):
        real_read = review.read

        def contaminated_read(path):
            data = copy.deepcopy(real_read(path))
            data["termination_reason"] = "FUTURE_OUTCOME_SENTINEL"
            data["task_success_state"] = "FUTURE_OUTCOME_SENTINEL"
            for step in data.get("steps", [data]):
                for key in ("status", "elite_after_pose", "elite_target_reached", "elite_target_error_mm",
                            "piper_packets_sent", "piper_physical_execution_confirmed", "completed_timestamp",
                            "observation_age_at_dispatch_s", "elite_move_attempted"):
                    step[key] = "FUTURE_OUTCOME_SENTINEL"
            return data

        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/"normal", Path(temp)/"contaminated"
            self.prepare(a)
            with patch.object(review, "read", side_effect=contaminated_read):
                self.prepare(b)
            self.assertEqual((a/"packets.jsonl").read_bytes(), (b/"packets.jsonl").read_bytes())

    def test_history_only_references_earlier_cases_in_same_run(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"pack"
            self.prepare(out)
            packets = review.jsonl(out/"packets.jsonl")
            self.assertEqual(len(packets), 16)
            seen = {}
            for packet in packets:
                self.assertEqual(packet["prior_case_ids_in_run"], seen.get(packet["run"], []))
                self.assertIn("/before", packet["before_image"])
                self.assertNotIn("after", json.dumps(packet))
                seen.setdefault(packet["run"], []).append(packet["id"])

    def test_incomplete_or_duplicate_reviews_are_not_scored(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"pack"
            self.prepare(out)
            ids = [p["id"] for p in review.jsonl(out/"packets.jsonl")]
            for bad_ids in (ids[:-1], ids[:-1]+[ids[0]]):
                (out/"reviews.jsonl").write_text("".join(json.dumps({"id":cid})+"\n" for cid in bad_ids))
                with self.assertRaisesRegex(ValueError, "cover every case"):
                    review.score(out)
                self.assertFalse((out/"report.json").exists())


if __name__ == "__main__":
    unittest.main()
