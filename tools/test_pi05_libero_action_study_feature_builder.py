"""Synthetic CPU tests only: fake encoder/checkpoint is not extraction evidence.

The tiny fixture mimics the declared public-source schema so the unchanged
generic native preparation can run. It is never a real source acceptance test.
No LeRobot import, checkpoint loading, CUDA, network or optimization is needed.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import build_pi05_libero_action_study_features as builder
from pi05_libero_world_model_adapter import DEMO_REPO, DEMO_REVISION, sha256_file


class FixtureEncoder:
    calls = []
    nonfinite = False

    def __init__(self, policy, out):
        self.out = out

    def __enter__(self):
        self.log = (self.out / "extraction_trace.jsonl").open("x", encoding="utf-8")
        return self

    def __exit__(self, *_):
        self.log.close()

    def row(self, images, **metadata):
        self.calls.append((images.copy(), dict(metadata)))
        self.log.write(json.dumps(metadata) + "\n")
        value = np.nan if self.nonfinite else metadata["frame_index"] / 10.0
        return np.full((2, 2048), value, np.float32)

    def evidence(self, count):
        return {"fixture_only": True, "rows": count, "optimization_steps": 0}


class FeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "cache"
        self.out.mkdir()
        self.implementation = self.root / "fixture_implementation.py"
        self.implementation.write_text("# fake implementation, never executed\n", encoding="utf-8")
        self.plan = deepcopy(builder.read_json(builder.ROOT / builder.PLAN_PATH))
        self.plan["checkpoint"]["installed_modeling_sha256"] = sha256_file(self.implementation)
        self.plan["output_directory_name"] = "fixture_feature_cache"
        specs = [(101, "train", 8), (102, "train", 9), (201, "validation", 10)]
        self.plan["episodes"] = [{"episode_index": e, "partition": p, "record_count": n} for e, p, n in specs]
        self.plan["selected_split"] = {"train_episode_indices": [102, 101], "validation_episode_indices": [201]}
        self.plan["expected"] = {"rows": 27, "train_state_rows": 17, "train_action_rows": 15,
                                 "train_windows": 5, "validation_windows": 4, "latent_shape": [27, 2, 2048]}
        pins = self.plan["checkpoint"]
        self.checkpoint = {name: pins[name] for name in ("model_sha256", "config_sha256", "protocol_sha256", "versions")}
        self.checkpoint.update(device="cuda", compile_model=False, gradient_checkpointing=False,
            load={"status": "loaded", "loaded_parameter_fraction": 1.0,
                  "load_state_dict_missing_keys": [], "load_state_dict_unexpected_keys": []},
            installed_implementation_sources={key: {"path": str(self.implementation), "sha256": sha256_file(self.implementation)}
                for key in ("policy_class", "image_preprocess", "image_embedding")})
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        bound = self.source_root / "fixture.json"
        bound.write_text('{"fixture_only":true}\n', encoding="utf-8")
        self.source = {"root": self.source_root, "verified_files": {"fixture.json": sha256_file(bound)},
            "report_sha256": self.plan["source_report_sha256"], "episodes": [],
            "report": {"source": {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION},
                       "duplicate_audit": {"same_partition_groups": {"train": [[101], [102]], "validation": [[201]]}}}}
        for eid, partition, count in specs:
            declaration = {"episode_index": eid, "task_id": 9, "record_count": count, "complete_episode": True,
                           "source_trajectory_id": f"fixture_source_{eid}", "leakage_group_id": f"fixture_group_{eid}"}
            rows = []
            for frame in range(count):
                base = 1000.0 if partition == "validation" else (1.0 if eid == 101 else 3.0)
                rows.append({"episode_index": eid, "frame_index": frame, "task_id": 9,
                    "state": np.full(8, base + frame / 10, np.float32), "state_valid": np.ones(8, bool),
                    "action": np.full(7, .99 if frame == count-1 else (.1 if eid == 101 else -.2), np.float32)})
            self.source["episodes"].append({"declaration": declaration, "adapted": rows,
                "images": np.full((count, 2, 2, 2, 3), eid % 256, np.uint8)})
        self.runtime_file = self.root / "runtime.txt"
        self.runtime_file.write_text("fixture runtime\n", encoding="utf-8")
        self.runtime = {"files_sha256": {str(self.runtime_file): sha256_file(self.runtime_file)},
                        "checkpoint": str(self.root / "not_a_checkpoint"), "protocol": str(self.root / "not_a_protocol")}
        FixtureEncoder.calls = []
        FixtureEncoder.nonfinite = False

    def build(self):
        with patch.object(builder, "TraceEncoder", FixtureEncoder), redirect_stdout(io.StringIO()):
            return builder.build_feature_pack(self.source, object(), self.checkpoint, self.plan, self.runtime, self.out)

    def test_fixed_feature_plan_hash_and_feature_only_scope(self):
        plan = builder.load_plan()
        self.assertEqual(sha256_file(builder.ROOT / builder.PLAN_PATH), builder.PLAN_SHA256)
        self.assertTrue(plan["feature_extraction_allowed"])
        self.assertFalse(plan["optimization_allowed"])
        self.assertEqual(plan["expected"]["train_state_rows"] - plan["expected"]["train_action_rows"], 8)

    def test_loaded_checkpoint_exact_fixture_passes(self):
        builder.check_loaded_checkpoint(self.checkpoint, self.plan)

    def test_loaded_checkpoint_pin_and_runtime_drift_rejected(self):
        for name, value in (("model_sha256", "0"*64), ("config_sha256", "0"*64), ("protocol_sha256", "0"*64),
                            ("versions", {}), ("device", "cpu"), ("compile_model", True), ("gradient_checkpointing", True)):
            with self.subTest(name=name):
                candidate = deepcopy(self.checkpoint)
                candidate[name] = value
                with self.assertRaises(ValueError):
                    builder.check_loaded_checkpoint(candidate, self.plan)

    def test_checkpoint_coverage_status_missing_and_unexpected_rejected(self):
        for name, value in (("status", "partial"), ("loaded_parameter_fraction", .999),
                            ("load_state_dict_missing_keys", ["weight"]), ("load_state_dict_unexpected_keys", ["extra"])):
            candidate = deepcopy(self.checkpoint)
            candidate["load"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.check_loaded_checkpoint(candidate, self.plan)

    def test_installed_roles_declared_hash_and_file_drift_rejected(self):
        for mode in ("role", "declared_hash", "content"):
            candidate = deepcopy(self.checkpoint)
            if mode == "role":
                del candidate["installed_implementation_sources"]["image_embedding"]
            elif mode == "declared_hash":
                candidate["installed_implementation_sources"]["image_embedding"]["sha256"] = "0"*64
            else:
                self.implementation.write_text("drift\n", encoding="utf-8")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                builder.check_loaded_checkpoint(candidate, self.plan)

    def test_verify_files_detects_late_drift(self):
        builder.verify_files(self.runtime["files_sha256"])
        self.runtime_file.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound input changed"):
            builder.verify_files(self.runtime["files_sha256"])

    def test_complete_build_preserves_raw_arrays_and_source_images(self):
        snapshots = deepcopy(self.source["episodes"])
        report = self.build()
        for key in ("state", "state_valid", "action"):
            expected = np.stack([r[key] for e in snapshots for r in e["adapted"]])
            np.testing.assert_array_equal(np.load(self.out / f"{key}.npy", allow_pickle=False), expected)
        for original, after in zip(snapshots, self.source["episodes"]):
            np.testing.assert_array_equal(original["images"], after["images"])
            for before_row, after_row in zip(original["adapted"], after["adapted"]):
                for key in ("state", "action", "state_valid"):
                    np.testing.assert_array_equal(before_row[key], after_row[key])
        self.assertEqual(report["features_extracted"], 27)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["training_ready"])
        self.assertEqual(len(FixtureEncoder.calls), 27)
        self.assertEqual(sum(m["preview"] for _, m in FixtureEncoder.calls), 6)

    def test_normalization_uses_train_only_and_excludes_each_terminal_action(self):
        self.build()
        stats = builder.read_json(self.out / "preparation/normalization.json")
        train = self.source["episodes"][:2]
        states = np.stack([r["state"] for e in train for r in e["adapted"]]).astype(np.float64)
        actions = np.stack([r["action"] for e in train for r in e["adapted"][:-1]]).astype(np.float64)
        np.testing.assert_allclose(stats["state_mean"], states.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(stats["action_mean"], actions.mean(axis=0), rtol=0, atol=1e-12)
        self.assertEqual(stats["state_record_count"], 17)
        self.assertEqual(stats["action_record_count"], 15)
        self.assertEqual(stats["train_episode_indices"], [101, 102])
        self.assertTrue(stats["final_row_actions_excluded"])
        raw = np.load(self.out / "action.npy", allow_pickle=False)
        np.testing.assert_array_equal(raw[[7, 16, 26]], np.full((3, 7), .99, np.float32))

    def test_exact_windows_split_and_groups_retained(self):
        report = self.build()
        self.assertEqual(report["windows"], {"train": 5, "validation": 4})
        self.assertEqual(report["source_groups"], self.source["report"]["duplicate_audit"]["same_partition_groups"])
        self.assertFalse(report["grouped_sampler_fitted"])
        split = builder.read_json(self.out / "split.json")
        for key, value in self.plan["selected_split"].items():
            self.assertEqual(split[key], value)
        episode_ids = np.load(self.out / "episode_index.npy", allow_pickle=False)
        frame_ids = np.load(self.out / "frame_index.npy", allow_pickle=False)
        for partition, expected_count in (("train", 5), ("validation", 4)):
            windows = [json.loads(line) for line in (self.out / "preparation" / f"{partition}_windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(windows), expected_count)
            for window in windows:
                self.assertIn(window["episode_index"], split[f"{partition}_episode_indices"])
                for key in ("history", "actions", "targets"):
                    self.assertTrue(np.all(episode_ids[window[key]] == window["episode_index"]))
                self.assertTrue(np.all(frame_ids[window["actions"]] < self.source["episodes"][[101, 102, 201].index(window["episode_index"])]["declaration"]["record_count"]-1))

    def test_invalid_checkpoint_stops_before_encoder_or_output(self):
        self.checkpoint["load"]["status"] = "not_loaded"
        with patch.object(builder, "TraceEncoder") as encoder, self.assertRaises(ValueError):
            builder.build_feature_pack(self.source, object(), self.checkpoint, self.plan, self.runtime, self.out)
        encoder.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_source_selection_order_and_partial_rows_rejected_before_encoder(self):
        for mode in ("order", "partial"):
            source = deepcopy(self.source)
            if mode == "order":
                source["episodes"].reverse()
            else:
                source["episodes"][0]["adapted"].pop()
            with self.subTest(mode=mode), patch.object(builder, "TraceEncoder") as encoder, self.assertRaises(ValueError):
                builder.build_feature_pack(source, object(), self.checkpoint, self.plan, self.runtime, self.out)
            encoder.assert_not_called()

    def test_nonfinite_encoder_output_preserves_trace_without_complete_pack(self):
        FixtureEncoder.nonfinite = True
        with self.assertRaisesRegex(ValueError, "nonfinite latent"):
            self.build()
        self.assertTrue((self.out / "extraction_trace.jsonl").exists())
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertFalse((self.out / "preparation").exists())

    def test_builder_refuses_existing_arrays_manifest_or_trace(self):
        for name in ("state.npy", "manifest.json", "extraction_trace.jsonl"):
            marker = self.out / name
            marker.write_bytes(b"preserve me")
            with self.subTest(name=name), patch.object(builder, "TraceEncoder") as encoder, self.assertRaises(ValueError):
                builder.build_feature_pack(self.source, object(), self.checkpoint, self.plan, self.runtime, self.out)
            encoder.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"preserve me")
            marker.unlink()

    def test_source_drift_prevents_success_after_extraction(self):
        (self.source_root / "fixture.json").write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.build()
        self.assertTrue((self.out / "manifest.json").exists())

    def test_runtime_drift_prevents_success_after_extraction(self):
        self.runtime_file.write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound input changed"):
            self.build()

    def test_wrong_expected_window_counts_fail_closed(self):
        self.plan["expected"]["train_windows"] += 1
        with self.assertRaisesRegex(ValueError, "window counts"):
            self.build()

    def test_execute_existing_output_stops_before_policy_load(self):
        out = self.root / "simulation_output" / self.plan["output_directory_name"]
        out.mkdir(parents=True)
        with patch.object(builder, "load_policy") as loader, self.assertRaises(FileExistsError):
            builder.execute(self.plan, self.source, self.runtime, {}, repo=self.root)
        loader.assert_not_called()

    def test_execute_loader_failure_records_no_resume_and_does_not_retry(self):
        with patch.object(builder, "load_policy", side_effect=ValueError("fixture loader failure")) as loader, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            builder.execute(self.plan, self.source, self.runtime, {}, repo=self.root)
        loader.assert_called_once()
        out = self.root / "simulation_output" / self.plan["output_directory_name"]
        status = builder.read_json(out / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["optimizer_steps"], 0)
        self.assertFalse(status["resume_allowed"])
        self.assertFalse(status["training_started"])
        self.assertIn("fixture loader failure", (out / "run.log").read_text())
        self.assertFalse((out / "report.json").exists())

    def test_execute_encoder_failure_preserves_partial_trace_and_cannot_reenter(self):
        FixtureEncoder.nonfinite = True
        with patch.object(builder, "load_policy", return_value=(object(), self.checkpoint)), patch.object(builder, "TraceEncoder", FixtureEncoder), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
            builder.execute(self.plan, self.source, self.runtime, {}, repo=self.root)
        out = self.root / "simulation_output" / self.plan["output_directory_name"]
        before = (out / "extraction_trace.jsonl").read_bytes()
        with patch.object(builder, "load_policy") as loader, self.assertRaises(FileExistsError):
            builder.execute(self.plan, self.source, self.runtime, {}, repo=self.root)
        loader.assert_not_called()
        self.assertEqual((out / "extraction_trace.jsonl").read_bytes(), before)
        self.assertEqual(builder.read_json(out / "status.json")["status"], "failed")

    def test_cli_authorization_flags_reject_before_preflight(self):
        for argv in (("tool", "--stage", "run"), ("tool", "--stage", "preflight", "--execute-frozen-feature-cache")):
            with self.subTest(argv=argv), patch("sys.argv", list(argv)), patch.object(builder, "preflight") as preflight, self.assertRaises(ValueError):
                builder.main()
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
