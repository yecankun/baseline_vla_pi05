"""Small fabricated native rows/RGB only; no source/checkpoint/feature evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_action_study_feature_source as source_module


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = TemporaryDirectory()
        cls.fixed = tuple({**e, "record_count": 7} for e in source_module.FIXED_EPISODES)
        cls.old = tuple({**e, "record_count": 7} for e in source_module.OLD_EPISODES)
        cls.backing, cls.frame_hashes = {}, {}
        for e in cls.fixed + cls.old:
            eid = e["episode_index"]
            path = Path(cls.shared.name) / f"{eid}.npy"
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(7, 2, 256, 256, 3))
            array[:] = 0
            for frame in range(7):
                for view in range(2):
                    array[frame, view, 0, 0] = [eid // 256, eid % 256, frame + 20 * view]
            cls.frame_hashes[eid] = [[hashlib.sha256(array[i, v].tobytes()).hexdigest() for i in range(7)] for v in range(2)]
            array.flush()
            array._mmap.close()
            cls.backing[eid] = path

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        # Only explicit synthetic count overrides; production IDs/order remain.
        self.patch_fixed = patch.object(source_module, "FIXED_EPISODES", self.fixed)
        self.patch_old = patch.object(source_module, "OLD_EPISODES", self.old)
        self.patch_fixed.start()
        self.patch_old.start()
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = []
        self.plan = {"episodes": list(deepcopy(self.fixed)),
                     "selected_split": {f"{p}_episode_indices": [e["episode_index"] for e in self.fixed if e["partition"] == p]
                                        for p in ("train", "validation")},
                     "selection_sha256": "a" * 64, "study_plan_sha256": "b" * 64}
        identity = {"selection_sha256": self.plan["selection_sha256"], "input_sha256": {
            "/fixture/docs/libero-action-ablation-study-plan-v1.json": self.plan["study_plan_sha256"],
            "/fixture/simulation_output/libero_action_study_task_index_v1/selection.json": self.plan["selection_sha256"]}}
        write(self.root / "identity.json", identity)
        self.report = {"schema": source_module.SOURCE_SCHEMA, "status": "passed_declared_source_checks",
                       "source_integrity_and_declared_duplicate_checks_passed": True,
                       "source": {"kind": "public_demonstrations", "repo_id": source_module.DEMO_REPO,
                                  "revision": source_module.DEMO_REVISION}, "identity": identity,
                       "orientation": source_module.ORIENTATION, "selected_split": deepcopy(self.plan["selected_split"]),
                       "selected_new_rows": 84, "old_development_rows": 14, "decoded_selected_view_frames": 196,
                       "new_ids_replaced": False, "feature_pack_complete": False, "normalization_fitted": False,
                       "policy_loaded": False, "training_ready": False, "formal_data_allowed": False,
                       "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
                       "features_extracted": 0, "optimizer_steps": 0, "episodes": [], "output_sha256": {}}
        for choice in self.fixed + self.old:
            eid = choice["episode_index"]
            directory = self.root / "episodes" / f"episode_{eid}"
            directory.mkdir(parents=True)
            os.link(self.backing[eid], directory / "images.npy")
            rows = [{"episode_index": eid, "frame_index": i, "index": eid * 1000 + i, "timestamp": i / 10,
                     "task_index": 39, "task": source_module.TASK_REGISTRY[0]["task_instruction"],
                     "observation.state": [eid / 1000 + i / 10 + j / 100 for j in range(8)],
                     "action": [(eid % 100) / 100 + i / 1000] * 7} for i in range(7)]
            write(directory / "records.json", rows)
            metadata = {"episode_index": eid, "length": 7, "dataset_from_index": eid * 1000,
                        "dataset_to_index": eid * 1000 + 7}
            frames = []
            for view, key in enumerate(source_module.IMAGE_KEYS):
                metadata.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": eid,
                                 f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": .7})
                frames.append({"key": key, "from_timestamp": 0.0, "to_timestamp": .7,
                               "frames": [{"frame_index": i, "video_timestamp": i / 10,
                                           "decoded_rgb_sha256": self.frame_hashes[eid][view][i]} for i in range(7)]})
            write(directory / "decoded_frames.json", frames)
            self.report["episodes"].append({**deepcopy(choice), "complete_episode": True, "metadata": metadata,
                "source_trajectory_id": f"{source_module.DEMO_REPO}@{source_module.DEMO_REVISION}/episode{eid}",
                **{f"{stem}_path": f"episodes/episode_{eid}/{name}" for stem, name in
                   (("images", "images.npy"), ("records", "records.json"), ("decoded_frames", "decoded_frames.json"))}})
        self.rebind(recompute=True)

    def tearDown(self):
        for source in self.sources:
            source_module.close_source(source)
        self.temp.cleanup()
        self.patch_old.stop()
        self.patch_fixed.stop()

    def rebind(self, *, recompute=False):
        if recompute:
            sequences = {}
            for entry in self.report["episodes"]:
                rows = source_module.read_json(self.root / entry["records_path"])
                frames = source_module.read_json(self.root / entry["decoded_frames_path"])
                sequences[entry["episode_index"]] = (
                    np.asarray([r["observation.state"] for r in rows], np.float32),
                    np.asarray([r["action"] for r in rows], np.float32),
                    [frames[0]["frames"][i]["decoded_rgb_sha256"] + frames[1]["frames"][i]["decoded_rgb_sha256"] for i in range(len(rows))])
            self.report["duplicate_audit"] = source_module.audit_sequences(self.report["episodes"], sequences)
        write(self.root / "duplicate_audit.json", self.report["duplicate_audit"])
        outputs = self.report["output_sha256"]
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != "report.json":
                outputs[path.relative_to(self.root).as_posix()] = source_module.sha256_file(path)
        for entry in self.report["episodes"]:
            for stem in ("images", "records"):
                entry[f"{stem}_sha256"] = outputs[entry[f"{stem}_path"]]
        self.plan["duplicate_audit_sha256"] = outputs["duplicate_audit.json"]
        write(self.root / "report.json", self.report)
        self.plan["source_report_sha256"] = source_module.sha256_file(self.root / "report.json")

    def validate(self):
        source = source_module.validate_source(self.root, self.plan)
        self.sources.append(source)
        return source

    def mutate_rows(self, fn, entry_index=0):
        path = self.root / self.report["episodes"][entry_index]["records_path"]
        rows = source_module.read_json(path)
        fn(rows)
        write(path, rows)
        self.rebind()

    def test_fixed12_complete_readonly_inputs_and_old_exclusion(self):
        source = self.validate()
        self.assertEqual([e["declaration"]["episode_index"] for e in source["episodes"]], [e["episode_index"] for e in self.fixed])
        self.assertEqual(len(source["episodes"]), 12)
        self.assertTrue(all(not e["images"].flags.writeable for e in source["episodes"]))
        self.assertTrue(all(e["declaration"]["task_id"] == 9 and e["declaration"]["source_task_index"] == 39 for e in source["episodes"]))
        self.assertEqual(len(source["source_groups"]), 12)
        self.assertTrue(source_module.source_unchanged(source))
        self.assertTrue(any("episode_1400/images.npy" in p for p in source["verified_files"]))

    def test_external_report_pin(self):
        self.plan["source_report_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            self.validate()

    def test_fixed_plan_order_counts_extra_and_old_ids(self):
        for mutate in (lambda p: p["episodes"].reverse(),
                       lambda p: p["episodes"][0].update(record_count=8),
                       lambda p: p["episodes"][0].update(episode_index=1400),
                       lambda p: p["episodes"][0].update(sample_role="train")):
            with self.subTest(mutate=mutate):
                plan = deepcopy(self.plan)
                mutate(plan)
                with self.assertRaisesRegex(ValueError, "fixed episode"):
                    source_module.validate_source(self.root, plan)

    def test_bool_plan_id_rejected(self):
        self.plan["episodes"][0]["episode_index"] = True
        with self.assertRaises(ValueError):
            self.validate()

    def test_source_order_and_incomplete_rejected(self):
        self.report["episodes"].reverse()
        self.rebind()
        with self.assertRaisesRegex(ValueError, "declarations fixed"):
            self.validate()

    def test_source_identity_orientation_and_scope_flags(self):
        for key, value in (("schema", "wrong"), ("status", "failed"), ("orientation", "vertical_flip"),
                           ("training_ready", True), ("formal_data_allowed", True), ("policy_loaded", True),
                           ("new_ids_replaced", True), ("features_extracted", 1), ("optimizer_steps", 1),
                           ("family_independence_verified", True), ("checkpoint_training_overlap_unknown", False)):
            with self.subTest(key=key):
                previous = self.report[key]
                self.report[key] = value
                self.rebind()
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[key] = previous

    def test_raw_rows_unknown_privileged_and_missing_keys_rejected(self):
        for key in ("reward", "success", "diagnostic_targets", "observation.state_valid", "event_id"):
            with self.subTest(key=key):
                self.mutate_rows(lambda rows: rows[0].update({key: 1}))
                with self.assertRaisesRegex(ValueError, "unknown/missing"):
                    self.validate()
                self.mutate_rows(lambda rows: rows[0].pop(key))
        self.mutate_rows(lambda rows: rows[0].pop("observation.state"))
        with self.assertRaisesRegex(ValueError, "unknown/missing"):
            self.validate()

    def test_row_id_timestamp_task_and_native_action_guards(self):
        for key, value in (("frame_index", 3), ("index", 3), ("episode_index", 1400),
                           ("timestamp", .01), ("task_index", 34), ("task", "wrong instruction"),
                           ("action", [1.01] * 7), ("observation.state", [True] * 8)):
            with self.subTest(key=key):
                path = self.root / self.report["episodes"][0]["records_path"]
                original = source_module.read_json(path)
                self.mutate_rows(lambda rows: rows[0].update({key: value}))
                with self.assertRaises(ValueError):
                    self.validate()
                write(path, original)
                self.rebind()

    def test_truncated_episode_and_metadata_interval(self):
        self.mutate_rows(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.validate()

    def test_old_records_are_audited_but_not_returned(self):
        self.mutate_rows(lambda rows: rows[0].update(reward=1), entry_index=12)
        with self.assertRaisesRegex(ValueError, "unknown/missing"):
            self.validate()

    def test_mutation_after_validation_detected(self):
        source = self.validate()
        path = self.root / self.report["episodes"][-1]["records_path"]
        with path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            source_module.source_unchanged(source)

    def test_inventory_mutation_detected_before_decode(self):
        path = self.root / self.report["episodes"][0]["records_path"]
        with path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "output hash mismatch"):
            self.validate()

    def test_decoded_view_order_pts_and_rgb_rejected(self):
        path = self.root / self.report["episodes"][0]["decoded_frames_path"]
        original = source_module.read_json(path)
        for mutate in (lambda f: f.reverse(), lambda f: f[0]["frames"][0].update(video_timestamp=.04),
                       lambda f: f[0]["frames"][0].update(decoded_rgb_sha256="a" * 64)):
            with self.subTest(mutate=mutate):
                frames = deepcopy(original)
                mutate(frames)
                write(path, frames)
                self.rebind()
                with self.assertRaises(ValueError):
                    self.validate()
        write(path, original)

    def test_duplicate_signature_and_crosssplit_group_metadata_rejected(self):
        for mutate in (lambda d: d.update(window_length=6),
                       lambda d: d.update(connecting_action_count=7),
                       lambda d: d["same_partition_groups"][0].update(partition="validation"),
                       lambda d: d["pairs"].pop()):
            with self.subTest(mutate=mutate):
                original = deepcopy(self.report["duplicate_audit"])
                mutate(self.report["duplicate_audit"])
                self.rebind()
                with self.assertRaisesRegex(ValueError, "all-pair/signature/group"):
                    self.validate()
                self.report["duplicate_audit"] = original

    def test_same_partition_duplicates_share_leakage_group_without_dropping(self):
        first, second = self.report["episodes"][:2]
        rows_a = source_module.read_json(self.root / first["records_path"])
        rows_b = source_module.read_json(self.root / second["records_path"])
        for a, b in zip(rows_a, rows_b):
            b["observation.state"], b["action"] = a["observation.state"], a["action"]
        write(self.root / second["records_path"], rows_b)
        self.rebind(recompute=True)
        source = self.validate()
        self.assertEqual(source["episodes"][0]["declaration"]["leakage_group_id"], source["episodes"][1]["declaration"]["leakage_group_id"])
        self.assertEqual(len(source["episodes"]), 12)
        self.assertEqual(len(source["source_groups"]), 11)

    def test_source_pin_and_sidecar_provenance_required(self):
        self.report["identity"]["selection_sha256"] = "c" * 64
        self.rebind()
        with self.assertRaisesRegex(ValueError, "selection identity"):
            self.validate()

    def test_missing_image_inventory_rejected(self):
        del self.report["output_sha256"][self.report["episodes"][0]["images_path"]]
        write(self.root / "report.json", self.report)
        self.plan["source_report_sha256"] = source_module.sha256_file(self.root / "report.json")
        with self.assertRaisesRegex(ValueError, "hash-bound inventory"):
            self.validate()

    def test_source_path_escape_and_alias_rejected(self):
        for name in ("../outside.json", "C:/outside.json", "episodes/../identity.json", "./identity.json"):
            with self.subTest(name=name):
                self.report["output_sha256"][name] = self.report["output_sha256"]["identity.json"]
                write(self.root / "report.json", self.report)
                self.plan["source_report_sha256"] = source_module.sha256_file(self.root / "report.json")
                with self.assertRaises(ValueError):
                    self.validate()
                del self.report["output_sha256"][name]

    def test_missing_metadata_interval_rejected(self):
        self.report["episodes"][0]["metadata"]["dataset_to_index"] += 1
        self.rebind()
        with self.assertRaisesRegex(ValueError, "metadata interval"):
            self.validate()

    def test_rgb_shape_rejected_and_bad_memmap_closed(self):
        path = self.root / self.report["episodes"][0]["images_path"]
        path.unlink()  # Own temporary hardlink only; shared fixture stays intact.
        np.save(path, np.zeros((7, 2, 4, 4, 3), np.uint8), allow_pickle=False)
        self.rebind()
        opened = []
        original_load = np.load
        def track(*args, **kwargs):
            array = original_load(*args, **kwargs)
            opened.append(array)
            return array
        with patch.object(source_module.np, "load", side_effect=track):
            with self.assertRaisesRegex(ValueError, "readonly uint8 memmap"):
                self.validate()
        self.assertTrue(opened[0]._mmap.closed)

    def test_adapter_cannot_silently_change_state_or_action(self):
        original_adapt = source_module.adapt_native_record
        for key in ("state", "action"):
            def changed(*args, **kwargs):
                result = original_adapt(*args, **kwargs)
                result[key][0] += .1
                return result
            with self.subTest(key=key), patch.object(source_module, "adapt_native_record", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "changed or imputed"):
                    self.validate()

    def test_crosssplit_duplicates_block_even_if_outer_source_passed(self):
        first, validation = self.report["episodes"][0], self.report["episodes"][8]
        a = source_module.read_json(self.root / first["records_path"])
        b = source_module.read_json(self.root / validation["records_path"])
        for left, right in zip(a, b):
            right["observation.state"], right["action"] = left["observation.state"], left["action"]
        write(self.root / validation["records_path"], b)
        self.rebind(recompute=True)
        with self.assertRaisesRegex(ValueError, "source isolation gate"):
            self.validate()

    def test_new_family_links_not_silently_introduced(self):
        self.report["duplicate_audit"]["known_family_links"] = [[1633, 1674]]
        self.rebind()
        with self.assertRaisesRegex(ValueError, "new family claims"):
            self.validate()

    def test_duplicate_sidecar_external_pin(self):
        self.plan["duplicate_audit_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "duplicate audit SHA"):
            self.validate()

    def test_memmaps_closed_on_mid_validation_failure_and_normal_close_idempotent(self):
        opened = []
        original_load = np.load
        def track(*args, **kwargs):
            array = original_load(*args, **kwargs)
            opened.append(array)
            return array
        self.report["episodes"][1]["complete_episode"] = False
        self.rebind()
        with patch.object(source_module.np, "load", side_effect=track):
            with self.assertRaisesRegex(ValueError, "complete native"):
                self.validate()
        self.assertEqual(len(opened), 1)
        self.assertTrue(all(a._mmap.closed for a in opened))
        self.report["episodes"][1]["complete_episode"] = True
        self.rebind()
        source = self.validate()
        source_module.close_source(source)
        source_module.close_source(source)
        self.assertTrue(all(e["images"]._mmap.closed for e in source["episodes"]))


if __name__ == "__main__":
    unittest.main()
