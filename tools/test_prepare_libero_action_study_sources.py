"""Synthetic source-download and resume guards: no network or real payloads.

Every requests import is replaced with a fake module. Execute orchestration
tests replace decoding with a Mock; they do not open Parquet, decode images,
load a policy, extract features, or run an optimizer.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import prepare_libero_action_study_sources as target
from test_acquire_libero_task_index import FakeResponse, FakeTransportError


VIDEO_PATH = "videos/observation.images.image/chunk-000/file-001.mp4"
OTHER_VIDEO_PATH = "videos/observation.images.image2/chunk-000/file-002.mp4"


def entry(payload=b"synthetic video bytes", path=VIDEO_PATH):
    return {"path": path, "size": len(payload),
            "lfs_sha256": hashlib.sha256(payload).hexdigest()}


def auth():
    return {"source_repo_id": "synthetic/repository", "source_revision": "a" * 40,
            "max_attempts_per_file_per_invocation": 2,
            "max_cumulative_payload_response_bytes": 500_000_000,
            "workers": 1, "orientation": "synthetic_no_decode"}


class SyntheticCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.auth = auth()
        self.network = types.SimpleNamespace(
            RequestException=FakeTransportError,
            get=Mock(side_effect=AssertionError("unconfigured network request")))
        network_patch = patch.dict("sys.modules", {"requests": self.network})
        network_patch.start()
        self.addCleanup(network_patch.stop)
        sleep_patch = patch.object(target.time, "sleep", return_value=None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def response(self, chunks, error=None):
        self.network.get.side_effect = None
        self.network.get.return_value = FakeResponse(chunks, error)

    def logs(self, name="download_attempts.jsonl"):
        return target.json_lines(self.out / name)

    def downloader(self, item=None, **kwargs):
        return target.VideoDownloads(self.out, self.auth, [item or entry()], **kwargs)


class VideoAllowlistTests(SyntheticCase):
    def test_accepts_only_two_native_camera_video_paths(self):
        for path in (VIDEO_PATH, OTHER_VIDEO_PATH):
            with self.subTest(path=path):
                target.video_entry(entry(path=path))

    def test_rejects_data_metadata_unknown_camera_and_noncanonical_paths(self):
        paths = ("data/chunk-000/file-001.parquet", "meta/tasks.parquet",
                 "videos/observation.images.image3/chunk-000/file-001.mp4",
                 "videos/observation.images.image/chunk-00/file-001.mp4",
                 VIDEO_PATH + "?download=true", VIDEO_PATH + "/extra",
                 "../" + VIDEO_PATH, "/" + VIDEO_PATH,
                 "videos/../" + VIDEO_PATH, VIDEO_PATH.replace("/", "\\"))
        for path in paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                target.video_entry(entry(path=path))
        self.network.get.assert_not_called()

    def test_rejects_large_or_noninteger_or_nonpositive_sizes(self):
        for size in (None, False, 0, -1, 1.0, "1", 100_000_001):
            with self.subTest(size=size), self.assertRaises(ValueError):
                target.video_entry({**entry(), "size": size})
        self.network.get.assert_not_called()

    def test_rejects_missing_or_malformed_sha(self):
        for digest in (None, "", "a" * 63, "A" * 64, "g" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                target.video_entry({**entry(), "lfs_sha256": digest})

    def test_duplicate_allowlist_paths_rejected(self):
        item = entry()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            target.VideoDownloads(self.out, self.auth, [item, item])

    def test_valid_video_outside_frozen_inventory_never_requests(self):
        with self.assertRaisesRegex(ValueError, "frozen acquisition inventory"):
            self.downloader().fetch(entry(path=OTHER_VIDEO_PATH))
        self.network.get.assert_not_called()
        self.assertEqual(self.logs("budget_reservations.jsonl"), [])

    def test_same_path_different_hash_or_size_never_requests(self):
        item = entry()
        for changed in ({**item, "size": item["size"] + 1},
                        {**item, "lfs_sha256": "0" * 64}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "frozen acquisition inventory"):
                self.downloader(item).fetch(changed)
        self.network.get.assert_not_called()


class VideoDownloadTests(SyntheticCase):
    def test_success_hashes_publishes_and_logs_pinned_url(self):
        payload = b"complete synthetic source bytes"
        item = entry(payload)
        self.response([payload[:4], payload[4:]])
        downloader = self.downloader(item)
        result = downloader.fetch(item)
        self.assertEqual(result, {**item, "reused": False})
        self.assertEqual((self.out / "source" / VIDEO_PATH).read_bytes(), payload)
        self.assertEqual(target.sha256_file(self.out / "source" / VIDEO_PATH), item["lfs_sha256"])
        self.assertEqual(downloader.bytes, len(payload))
        self.assertEqual(downloader.reserved, len(payload))
        self.network.get.assert_called_once_with(
            f"https://huggingface.co/datasets/synthetic/repository/resolve/{'a' * 40}/{VIDEO_PATH}",
            stream=True, timeout=(10, 30))
        self.assertEqual(self.logs("budget_reservations.jsonl"),
                         [{"path": VIDEO_PATH, "attempt": 1, "reserved_bytes": len(payload)}])
        self.assertEqual(len(self.logs()), 1)
        self.assertIsNone(self.logs()[0]["error_type"])
        self.assertEqual(self.logs()[0]["received_bytes"], len(payload))
        self.assertEqual(list(self.out.rglob("*.part")), [])

    def test_relative_output_success_and_safe_relative_log_path(self):
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.out)
            relative_out = Path("relative-source-output")
            relative_out.mkdir()
            payload = b"relative path regression"
            item = entry(payload)
            self.response([payload])
            downloader = target.VideoDownloads(relative_out, self.auth, [item])
            result = downloader.fetch(item)
            self.assertFalse(result["reused"])
            self.assertTrue(downloader.out.is_absolute())
            self.assertEqual(target.sha256_file(relative_out / "source" / VIDEO_PATH), item["lfs_sha256"])
            logs = target.json_lines(relative_out / "download_attempts.jsonl")
            self.assertEqual(len(logs), 1)
            self.assertIsNone(logs[0]["error_type"])
            self.assertEqual(logs[0]["received_bytes"], len(payload))
            part = Path(logs[0]["part_path"])
            self.assertFalse(part.is_absolute())
            self.assertNotIn("..", part.parts)
            self.assertEqual(part.suffix, ".part")
        finally:
            os.chdir(previous_cwd)

    def test_verified_cache_reuses_without_request_or_reservation(self):
        payload = b"verified cached bytes"
        item = entry(payload)
        path = self.out / "source" / VIDEO_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        result = self.downloader(item).fetch(item)
        self.assertEqual(result, {**item, "reused": True})
        self.network.get.assert_not_called()
        self.assertEqual(self.logs(), [])
        self.assertEqual(self.logs("budget_reservations.jsonl"), [])

    def test_bad_cache_hash_is_preserved_without_network(self):
        path = self.out / "source" / VIDEO_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(b"evil")
        item = entry(b"good")
        with self.assertRaises(ValueError):
            self.downloader(item).fetch(item)
        self.assertEqual(path.read_bytes(), b"evil")
        self.network.get.assert_not_called()

    def test_bad_cache_size_is_preserved_without_network(self):
        payload = b"bytes"
        item = {**entry(payload), "size": len(payload) + 1}
        path = self.out / "source" / VIDEO_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.downloader(item).fetch(item)
        self.assertEqual(path.read_bytes(), payload)
        self.network.get.assert_not_called()

    def test_response_bad_hash_is_not_published_and_not_retried(self):
        item = entry(b"good")
        self.response([b"evil"])
        with self.assertRaises(ValueError):
            self.downloader(item).fetch(item)
        self.assertEqual(self.network.get.call_count, 1)
        self.assertFalse((self.out / "source" / VIDEO_PATH).exists())
        self.assertEqual([p.read_bytes() for p in self.out.rglob("*.part")], [b"evil"])
        self.assertEqual(self.logs()[0]["error_type"], "ValueError")

    def test_short_response_is_not_published(self):
        item = entry(b"full payload")
        self.response([b"short"])
        with self.assertRaises(ValueError):
            self.downloader(item).fetch(item)
        self.assertEqual(self.network.get.call_count, 1)
        self.assertFalse((self.out / "source" / VIDEO_PATH).exists())
        self.assertEqual([p.read_bytes() for p in self.out.rglob("*.part")], [b"short"])

    def test_oversized_response_is_counted_but_not_published(self):
        item = entry(b"small")
        payload = b"too large response"
        self.response([payload])
        downloader = self.downloader(item)
        with self.assertRaisesRegex(ValueError, "exceeded pinned"):
            downloader.fetch(item)
        self.assertEqual(self.network.get.call_count, 1)
        self.assertFalse((self.out / "source" / VIDEO_PATH).exists())
        self.assertEqual(downloader.bytes, len(payload))
        self.assertEqual(self.logs()[0]["received_bytes"], len(payload))

    def test_two_transport_failures_keep_parts_and_full_reservations(self):
        item = entry(b"four")
        self.network.get.side_effect = FakeTransportError("synthetic interruption")
        downloader = self.downloader(item)
        with self.assertRaisesRegex(RuntimeError, "bounded video transport attempts exhausted"):
            downloader.fetch(item)
        self.assertEqual(self.network.get.call_count, 2)
        self.assertEqual(downloader.bytes, 0)
        self.assertEqual(downloader.reserved, 2 * item["size"])
        self.assertEqual(len(self.logs()), 2)
        self.assertEqual([r["attempt"] for r in self.logs()], [1, 2])
        self.assertEqual([r["received_bytes"] for r in self.logs()], [0, 0])
        self.assertEqual([r["error_type"] for r in self.logs()], ["FakeTransportError"] * 2)
        self.assertEqual(len(self.logs("budget_reservations.jsonl")), 2)
        self.assertEqual([p.read_bytes() for p in self.out.rglob("*.part")], [b"", b""])
        self.assertFalse((self.out / "source" / VIDEO_PATH).exists())

    def test_partial_transport_then_success_preserves_failed_part(self):
        payload = b"complete"
        item = entry(payload)
        self.network.get.side_effect = [FakeResponse([payload[:2]], FakeTransportError("partial")),
                                        FakeResponse([payload])]
        downloader = self.downloader(item)
        self.assertFalse(downloader.fetch(item)["reused"])
        self.assertEqual(self.network.get.call_count, 2)
        self.assertEqual(downloader.reserved, 2 * len(payload))
        self.assertEqual(downloader.bytes, len(payload) + 2)
        self.assertEqual(sum(r["received_bytes"] for r in self.logs()), len(payload) + 2)
        self.assertEqual([p.read_bytes() for p in self.out.rglob("*.part")], [payload[:2]])
        self.assertEqual((self.out / "source" / VIDEO_PATH).read_bytes(), payload)

    def test_reservation_over_budget_refuses_before_network(self):
        self.auth["max_cumulative_payload_response_bytes"] = 3
        item = entry(b"four")
        downloader = self.downloader(item)
        with self.assertRaisesRegex(ValueError, "reservation budget exceeded"):
            downloader.fetch(item)
        self.network.get.assert_not_called()
        self.assertEqual(self.logs(), [])
        self.assertEqual(self.logs("budget_reservations.jsonl"), [])
        self.assertTrue(downloader.stop.is_set())

    def test_prior_reservations_count_even_with_zero_response_bytes(self):
        self.auth["max_cumulative_payload_response_bytes"] = 8
        item = entry(b"four")
        with self.assertRaises(ValueError):
            self.downloader(item, consumed=0, reserved=6).fetch(item)
        self.network.get.assert_not_called()

    def test_cancelled_download_refuses_network(self):
        item = entry()
        downloader = self.downloader(item)
        downloader.stop.set()
        with self.assertRaises(RuntimeError):
            downloader.fetch(item)
        self.network.get.assert_not_called()


class ResumeTests(SyntheticCase):
    def fixture(self, *, state="failed", phase="download", consumed=2, reserved=4):
        identity = {"output": str(self.out), "code_sha256": {"synthetic.py": "a" * 64},
                    "input_sha256": {"synthetic_input": "b" * 64}}
        status = {"status": state, "phase": phase,
                  "cumulative_response_payload_bytes": consumed,
                  "conservative_reserved_bytes": reserved,
                  "error": "synthetic preserved failure"}
        target.dump(self.out / "identity.json", identity)
        target.dump(self.out / "status.json", status)
        (self.out / "download_attempts.jsonl").write_text(
            json.dumps({"path": VIDEO_PATH, "attempt": 1, "received_bytes": consumed}) + "\n", encoding="utf-8")
        (self.out / "budget_reservations.jsonl").write_text(
            json.dumps({"path": VIDEO_PATH, "attempt": 1, "reserved_bytes": reserved}) + "\n", encoding="utf-8")
        return identity, status

    def test_failed_download_same_identity_can_resume_and_preserves_failure(self):
        identity, status = self.fixture()
        self.assertEqual(target.check_resume(self.out, identity), (2, 4))
        self.assertEqual(target.read(self.out / "prior_failure_001.json"), status)
        self.assertEqual(target.read(self.out / "status.json"), status)
        self.assertEqual(target.check_resume(self.out, identity), (2, 4))
        self.assertEqual(target.read(self.out / "prior_failure_002.json"), status)
        self.network.get.assert_not_called()

    def test_running_completed_and_source_audit_cannot_resume(self):
        for state, phase in (("running", "download"), ("completed", "completed"),
                             ("completed_rejected", "completed"), ("failed", "source_audit")):
            with self.subTest(state=state, phase=phase):
                identity, _ = self.fixture(state=state, phase=phase)
                with self.assertRaisesRegex(ValueError, "only failed download phase"):
                    target.check_resume(self.out, identity)
        self.assertEqual(list(self.out.glob("prior_failure_*.json")), [])

    def test_resume_rejects_code_or_input_identity_drift(self):
        identity, _ = self.fixture()
        for field in ("code_sha256", "input_sha256", "output"):
            changed = copy.deepcopy(identity)
            changed[field] = "drift"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity/code/input drift"):
                target.check_resume(self.out, changed)
        self.assertEqual(list(self.out.glob("prior_failure_*.json")), [])

    def test_decode_or_final_report_artifacts_block_resume(self):
        identity, _ = self.fixture()
        for name in ("episodes", "download_report.json", "report.json"):
            path = self.out / name
            path.mkdir() if name == "episodes" else path.write_text("{}", encoding="utf-8")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "decode/finalization output exists"):
                    target.check_resume(self.out, identity)
            finally:
                path.rmdir() if path.is_dir() else path.unlink()
        self.assertEqual(list(self.out.glob("prior_failure_*.json")), [])

    def test_missing_attempt_ledger_rejects_nonzero_response_status(self):
        identity, _ = self.fixture()
        (self.out / "download_attempts.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
            target.check_resume(self.out, identity)
        self.assertEqual(list(self.out.glob("prior_failure_*.json")), [])

    def test_missing_reservation_ledger_rejects_nonzero_status(self):
        identity, _ = self.fixture()
        (self.out / "budget_reservations.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
            target.check_resume(self.out, identity)

    def test_accounting_status_mismatch_or_response_above_reserved_rejected(self):
        for received, reserved in ((3, 4), (2, 5), (5, 4)):
            identity, status = self.fixture()
            status["cumulative_response_payload_bytes"] = received
            status["conservative_reserved_bytes"] = reserved
            target.dump(self.out / "status.json", status)
            with self.subTest(received=received, reserved=reserved), self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                target.check_resume(self.out, identity)

    def test_reservations_over_global_limit_rejected(self):
        identity, _ = self.fixture(consumed=0, reserved=500_000_001)
        with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
            target.check_resume(self.out, identity)


class ExecuteDecisionTests(SyntheticCase):
    def context(self):
        out = self.out / "synthetic-execution"
        return {"auth": self.auth,
                "identity": {"output": str(out), "input_sha256": {}, "code_sha256": {}},
                "missing": [], "source_paths": {},
                "index": {"source": {"kind": "synthetic"}},
                "selection": {"split": {"train": [11], "validation": [12]}}}

    def test_existing_invocation_lock_blocks_resume_without_overwrite(self):
        context = self.context()
        out = Path(context["identity"]["output"])
        out.mkdir()
        lock = out / "active_invocation.lock"
        payload = b'{"pid": 123456, "scope": "preserve prior operator evidence"}\n'
        lock.write_bytes(payload)
        with patch.object(target, "_execute_locked") as inner:
            with self.assertRaises(FileExistsError):
                target.execute(context, {}, resume=True)
        inner.assert_not_called()
        self.network.get.assert_not_called()
        self.assertEqual(lock.read_bytes(), payload)
        self.assertFalse((out / "status.json").exists())

    def test_new_lock_exists_during_inner_execution_and_is_removed_after_success(self):
        context = self.context()
        lock = Path(context["identity"]["output"]) / "active_invocation.lock"

        def inner(*args):
            self.assertTrue(lock.is_file())
            self.assertEqual(target.read(lock)["pid"], os.getpid())
            return 0

        with patch.object(target, "_execute_locked", side_effect=inner):
            self.assertEqual(target.execute(context, {}), 0)
        self.assertFalse(lock.exists())
        self.network.get.assert_not_called()

    def test_new_lock_is_removed_after_inner_exception(self):
        context = self.context()
        lock = Path(context["identity"]["output"]) / "active_invocation.lock"
        with patch.object(target, "_execute_locked", side_effect=ValueError("synthetic failure")):
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                target.execute(context, {})
        self.assertFalse(lock.exists())
        self.network.get.assert_not_called()

    def test_declared_duplicate_pass_is_success_but_not_training_ready(self):
        context = self.context()
        duplicates = {"status": "passed_declared_duplicate_checks_only", "blocking_pair_count": 0}
        with patch.object(target, "decode_selected", return_value=([], duplicates)) as decoder, contextlib.redirect_stdout(io.StringIO()):
            result = target.execute(context, {"status": "synthetic_preflight"})
        self.assertEqual(result, 0)
        decoder.assert_called_once()
        out = Path(context["identity"]["output"])
        report = target.read(out / "report.json")
        status = target.read(out / "status.json")
        self.assertEqual(report["status"], "passed_declared_source_checks")
        self.assertTrue(report["source_integrity_and_declared_duplicate_checks_passed"])
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["report_sha256"], target.sha256_file(out / "report.json"))
        self.assertNotIn("active_invocation.lock", report["output_sha256"])
        self.assertFalse((out / "active_invocation.lock").exists())
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["formal_data_allowed"])
        self.assertFalse(report["family_independence_verified"])
        self.assertFalse(report["new_ids_replaced"])
        self.assertFalse(report["feature_pack_complete"])
        self.assertEqual(report["features_extracted"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.network.get.assert_not_called()

    def test_declared_overlap_rejection_is_final_nonresumable_result(self):
        context = self.context()
        duplicates = {"status": "rejected_source_isolation", "blocking_pair_count": 1}
        with patch.object(target, "decode_selected", return_value=([], duplicates)), contextlib.redirect_stdout(io.StringIO()):
            result = target.execute(context, {"status": "synthetic_preflight"})
        self.assertEqual(result, 3)
        out = Path(context["identity"]["output"])
        report = target.read(out / "report.json")
        self.assertEqual(report["status"], "rejected_source_overlap")
        self.assertFalse(report["source_integrity_and_declared_duplicate_checks_passed"])
        self.assertEqual(target.read(out / "status.json")["status"], "completed_rejected")
        with self.assertRaises(ValueError):
            target.check_resume(out, context["identity"])
        self.network.get.assert_not_called()

    def test_unknown_duplicate_status_fails_closed_without_final_report(self):
        context = self.context()
        duplicates = {"status": "passed", "blocking_pair_count": 0}
        with patch.object(target, "decode_selected", return_value=([], duplicates)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "unknown duplicate audit decision"):
                target.execute(context, {"status": "synthetic_preflight"})
        out = Path(context["identity"]["output"])
        self.assertFalse((out / "report.json").exists())
        status = target.read(out / "status.json")
        self.assertEqual((status["status"], status["phase"]), ("failed", "source_audit"))
        self.assertFalse(status["resume_allowed_only_if_download_phase"])
        self.network.get.assert_not_called()

    def test_contradictory_pass_and_blocking_pairs_fail_closed(self):
        context = self.context()
        duplicates = {"status": "passed_declared_duplicate_checks_only", "blocking_pair_count": 1}
        with patch.object(target, "decode_selected", return_value=([], duplicates)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "inconsistent source isolation decision"):
                target.execute(context, {"status": "synthetic_preflight"})
        self.assertFalse((Path(context["identity"]["output"]) / "report.json").exists())
        self.network.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
