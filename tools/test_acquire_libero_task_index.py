"""Synthetic, network-free guards for bounded LIBERO index acquisition.

No PyArrow, real source payloads, model weights or optimizer are used. All
requests are replaced before the downloader imports the requests module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import acquire_libero_task_index as target


DATA_PATH = "data/chunk-000/file-001.parquet"
INFO = {
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
}


def entry(payload=b"synthetic payload", path=DATA_PATH):
    return {"path": path, "size": len(payload), "lfs_sha256": hashlib.sha256(payload).hexdigest()}


def episode(eid=0, offset=0, count=2, video_start=0.0, data_file=1, video_file=1):
    result = {
        "episode_index": eid, "length": count,
        "dataset_from_index": offset, "dataset_to_index": offset + count,
        "data/chunk_index": 0, "data/file_index": data_file,
    }
    for view in target.VIEWS:
        result.update({
            f"videos/{view}/chunk_index": 0,
            f"videos/{view}/file_index": video_file,
            f"videos/{view}/from_timestamp": video_start,
            f"videos/{view}/to_timestamp": video_start + count / 10.0,
        })
    return result


def projection(eid=0, task=39, frame=0, index=0, timestamp=0.0):
    return dict(zip(target.COLS, (eid, task, frame, index, timestamp)))


def selection(ids=(0, 1), partitions=("train", "validation")):
    return {"episodes": [{"episode_index": eid, "partition": part}
                         for eid, part in zip(ids, partitions)]}


def video_inventory(video_file=1):
    return {DATA_PATH: entry(), **{name: entry(b"video inventory only", name) for name in (
        INFO["video_path"].format(video_key=view, chunk_index=0, file_index=video_file)
        for view in target.VIEWS)}}


class HelpersTests(unittest.TestCase):
    def test_safe_path_is_inside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(target.safe_path(root, DATA_PATH), (root / DATA_PATH).resolve())

    def test_safe_path_rejects_traversal_absolute_and_backslash(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("../escape", "data/../../escape", "data\\escape", str(Path(tmp).resolve()), None):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    target.safe_path(Path(tmp), name)

    def test_entry_requires_exact_data_allowlist(self):
        for path in ("../data/chunk-000/file-001.parquet", "data/chunk-00/file-001.parquet",
                     "data/chunk-000/file-001.parquet?x", "meta/tasks.parquet",
                     "videos/observation.images.image/chunk-000/file-001.mp4"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                target.validate_entry(entry(path=path))

    def test_entry_rejects_invalid_sizes(self):
        for size in (None, False, 0, -1, 1.0, "1", 100_000_001):
            item = entry()
            item["size"] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                target.validate_entry(item)

    def test_entry_rejects_missing_or_malformed_sha(self):
        for digest in (None, "", "a" * 63, "A" * 64, "g" * 64):
            item = entry()
            item["lfs_sha256"] = digest
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                target.validate_entry(item)

    def test_budget_returns_sorted_entries_and_exact_sum(self):
        a = entry(b"ab", "data/chunk-000/file-002.parquet")
        b = entry(b"c", DATA_PATH)
        entries, total = target.budget_entries([a["path"], b["path"]], {x["path"]: x for x in (a, b)}, 3)
        self.assertEqual([x["path"] for x in entries], [b["path"], a["path"]])
        self.assertEqual(total, 3)

    def test_budget_rejects_duplicate_missing_mismatched_and_excess(self):
        item = entry(b"ab")
        for names, inventory, limit in (
            ([DATA_PATH, DATA_PATH], {DATA_PATH: item}, 99),
            ([DATA_PATH], {}, 99),
            ([DATA_PATH], {DATA_PATH: entry(path="data/chunk-000/file-002.parquet")}, 99),
            ([DATA_PATH], {DATA_PATH: item}, 1),
        ):
            with self.subTest(names=names, inventory=inventory, limit=limit), self.assertRaises(ValueError):
                target.budget_entries(names, inventory, limit)

    def test_metadata_complete_contiguous_intervals(self):
        target.validate_metadata([episode(1, 2, video_start=.2), episode()], 2, 4)

    def test_metadata_rejects_global_gap_overlap_duplicate_and_wrong_totals(self):
        for rows, episodes, frames in (
            ([episode(), episode(1, 3, video_start=.2)], 2, 4),
            ([episode(), episode(1, 1, video_start=.2)], 2, 4),
            ([episode(), episode(0, 2, video_start=.2)], 2, 4),
            ([episode()], 2, 2), ([episode()], 1, 3),
            ([episode(count=0)], 1, 0),
        ):
            with self.subTest(rows=rows, episodes=episodes, frames=frames), self.assertRaises(ValueError):
                target.validate_metadata(rows, episodes, frames)

    def test_metadata_rejects_invalid_video_intervals(self):
        for view in target.VIEWS:
            for start, stop in ((-.1, .1), (0., .1), (0., float("nan")), (float("inf"), float("inf"))):
                row = episode()
                row[f"videos/{view}/from_timestamp"] = start
                row[f"videos/{view}/to_timestamp"] = stop
                with self.subTest(view=view, start=start, stop=stop), self.assertRaises(ValueError):
                    target.validate_metadata([row], 1, 2)

    def test_shard_rows_accept_only_metadata_declared_owner(self):
        target.validate_shard_rows(DATA_PATH, [projection()], {0: episode()}, INFO)
        with self.assertRaises(ValueError):
            target.validate_shard_rows("data/chunk-000/file-002.parquet", [projection()], {0: episode()}, INFO)

    def test_shard_rows_reject_unknown_episode_and_extra_or_missing_columns(self):
        for row in (projection(eid=1), {**projection(), "action": [1.]},
                    {k: v for k, v in projection().items() if k != "timestamp"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                target.validate_shard_rows(DATA_PATH, [row], {0: episode()}, INFO)

    def test_task_inventory_rejects_partial_and_conflicting_task_rows(self):
        for rows in ([projection()], [projection(), projection(task=38, frame=1, index=1, timestamp=.1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                target.task_inventory([episode()], rows, 10.)

    def test_selection_payload_inventory_deduplicates_videos_without_fetch(self):
        rows = [episode(), episode(1, 2, video_start=.2)]
        inventory = video_inventory()
        with patch.object(target.Downloads, "fetch", side_effect=AssertionError("no network allowed")):
            result = target.selected_payloads(selection(), rows, inventory, INFO)
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(len(result["unique_video_files"]), 2)
        self.assertEqual(result["unique_video_bytes"], sum(x["size"] for x in inventory.values()
                                                          if x["path"].startswith("videos/")))
        self.assertEqual(result["unique_data_paths"], [DATA_PATH])
        self.assertFalse(result["video_download_authorized"])
        self.assertFalse(result["source_duplicate_audit_passed"])
        self.assertFalse(result["family_independence_verified"])
        self.assertEqual(result["images_decoded"], 0)

    def test_selection_rejects_cross_partition_video_interval_overlap(self):
        rows = [episode(), episode(1, 2, video_start=.1)]
        with self.assertRaises(ValueError):
            target.selected_payloads(selection(), rows, video_inventory(), INFO)

    def test_selection_different_video_shards_do_not_overlap(self):
        rows = [episode(), episode(1, 2, video_file=2)]
        inventory = {**video_inventory(), **video_inventory(2)}
        result = target.selected_payloads(selection(), rows, inventory, INFO)
        self.assertEqual(len(result["unique_video_files"]), 4)

    def test_selection_same_partition_overlap_is_not_cross_split_failure(self):
        rows = [episode(), episode(1, 2, video_start=.1)]
        result = target.selected_payloads(selection(partitions=("train", "train")), rows, video_inventory(), INFO)
        self.assertEqual(len(result["episodes"]), 2)

    def test_large_future_video_is_inventory_only_and_requires_user_transfer(self):
        inventory = video_inventory()
        large_name = next(name for name in inventory if name.startswith("videos/"))
        inventory[large_name]["size"] = 100_000_001
        result = target.selected_payloads(selection(ids=(0,), partitions=("train",)),
                                           [episode()], inventory, INFO)
        self.assertEqual(result["video_files_requiring_user_transfer"], [large_name])
        self.assertFalse(result["video_download_authorized"])
        self.assertEqual(result["images_decoded"], 0)
        inventory[DATA_PATH]["size"] = 100_000_001
        with self.assertRaises(ValueError):
            target.selected_payloads(selection(ids=(0,), partitions=("train",)),
                                     [episode()], inventory, INFO)


class FakeTransportError(Exception):
    pass


class FakeResponse:
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield from self.chunks
        if self.error is not None:
            raise self.error


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.auth = {"source_repo_id": "synthetic/repository", "source_revision": "a" * 40,
                     "max_attempts_per_file_per_invocation": 2,
                     "max_cumulative_payload_response_bytes": 50_000_000}
        self.network = types.SimpleNamespace(RequestException=FakeTransportError, get=Mock())
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {"requests": self.network}).start()
        patch.object(target.time, "sleep", return_value=None).start()

    def logs(self, name="download_attempts.jsonl"):
        path = self.out / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_success_is_hashed_atomic_and_pinned_url(self):
        payload = b"synthetic complete payload"
        self.network.get.return_value = FakeResponse([payload[:4], payload[4:]])
        result = target.Downloads(self.out, self.auth).fetch(entry(payload))
        self.assertFalse(result["reused"])
        self.assertEqual((self.out / "source" / DATA_PATH).read_bytes(), payload)
        self.assertEqual(len(list(self.out.rglob("*.part"))), 0)
        self.network.get.assert_called_once_with(
            f"https://huggingface.co/datasets/synthetic/repository/resolve/{'a' * 40}/{DATA_PATH}",
            stream=True, timeout=(10, 30))
        self.assertEqual(self.logs()[0]["received_bytes"], len(payload))
        self.assertTrue(self.logs("budget_reservations.jsonl"))

    def test_relative_output_from_real_temporary_cwd_logs_successfully(self):
        payload = b"synthetic relative-output payload"
        self.network.get.return_value = FakeResponse([payload])
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.out)
            relative_out = Path("relative-output")
            relative_out.mkdir()
            downloader = target.Downloads(relative_out, self.auth)
            result = downloader.fetch(entry(payload))
            self.assertFalse(result["reused"])
            self.assertTrue(downloader.out.is_absolute())
            self.assertEqual(target.sha(relative_out / "source" / DATA_PATH),
                             hashlib.sha256(payload).hexdigest())
            attempts = [json.loads(line) for line in
                        (relative_out / "download_attempts.jsonl").read_text().splitlines()]
            self.assertEqual(len(attempts), 1)
            self.assertIsNone(attempts[0]["error_type"])
            self.assertEqual(attempts[0]["received_bytes"], len(payload))
            part_path = Path(attempts[0]["part_path"])
            self.assertFalse(part_path.is_absolute())
            self.assertNotIn("..", part_path.parts)
            self.assertEqual(part_path.suffix, ".part")
        finally:
            os.chdir(previous_cwd)

    def test_verified_cache_is_reused_without_network_or_reservation(self):
        payload = b"cached"
        path = self.out / "source" / DATA_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        result = target.Downloads(self.out, self.auth).fetch(entry(payload))
        self.assertTrue(result["reused"])
        self.network.get.assert_not_called()
        self.assertEqual(self.logs("budget_reservations.jsonl"), [])

    def test_corrupt_cache_refuses_replacement(self):
        path = self.out / "source" / DATA_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bad!")
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth).fetch(entry(b"good"))
        self.network.get.assert_not_called()
        self.assertEqual(path.read_bytes(), b"bad!")

    def test_hash_mismatch_keeps_partial_never_publishes_or_retries(self):
        self.network.get.return_value = FakeResponse([b"evil"])
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth).fetch(entry(b"good"))
        self.assertFalse((self.out / "source" / DATA_PATH).exists())
        self.assertEqual(self.network.get.call_count, 1)
        self.assertEqual([x.read_bytes() for x in self.out.rglob("*.part")], [b"evil"])
        self.assertEqual(self.logs()[0]["error_type"], "ValueError")

    def test_short_response_keeps_partial_never_publishes(self):
        self.network.get.return_value = FakeResponse([b"pa"])
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth).fetch(entry(b"payload"))
        self.assertFalse((self.out / "source" / DATA_PATH).exists())
        self.assertEqual([x.read_bytes() for x in self.out.rglob("*.part")], [b"pa"])

    def test_oversized_response_never_publishes(self):
        self.network.get.return_value = FakeResponse([b"way too much"])
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth).fetch(entry(b"small"))
        self.assertFalse((self.out / "source" / DATA_PATH).exists())
        self.assertEqual(self.network.get.call_count, 1)

    def test_two_transport_errors_exhaust_budgeted_retries(self):
        self.network.get.side_effect = FakeTransportError("synthetic transport failure")
        with self.assertRaises(RuntimeError):
            target.Downloads(self.out, self.auth).fetch(entry())
        self.assertEqual(self.network.get.call_count, 2)
        self.assertEqual(len(self.logs()), 2)
        self.assertEqual(len(self.logs("budget_reservations.jsonl")), 2)
        self.assertEqual(len(list(self.out.rglob("*.part"))), 2)
        self.assertFalse((self.out / "source" / DATA_PATH).exists())

    def test_partial_transport_retry_preserves_failed_part(self):
        payload = b"complete"
        self.network.get.side_effect = [FakeResponse([payload[:2]], FakeTransportError("interrupted")),
                                        FakeResponse([payload])]
        result = target.Downloads(self.out, self.auth).fetch(entry(payload))
        self.assertFalse(result["reused"])
        self.assertEqual(self.network.get.call_count, 2)
        self.assertEqual([p.read_bytes() for p in self.out.rglob("*.part")], [payload[:2]])
        self.assertEqual(sum(x["received_bytes"] for x in self.logs()), len(payload) + 2)

    def test_budget_rejects_before_any_request(self):
        self.auth["max_cumulative_payload_response_bytes"] = 3
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth).fetch(entry(b"four"))
        self.network.get.assert_not_called()
        self.assertFalse((self.out / "source" / DATA_PATH).exists())

    def test_resume_reservations_count_even_when_response_bytes_zero(self):
        self.auth["max_cumulative_payload_response_bytes"] = 8
        with self.assertRaises(ValueError):
            target.Downloads(self.out, self.auth, consumed=0, reserved=6).fetch(entry(b"four"))
        self.network.get.assert_not_called()

    def test_cancelled_job_refuses_request(self):
        downloader = target.Downloads(self.out, self.auth)
        downloader.stop.set()
        with self.assertRaises(RuntimeError):
            downloader.fetch(entry())
        self.network.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
