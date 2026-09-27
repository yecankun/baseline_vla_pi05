"""Focused adapter/annotation regression checks on disposable synthetic records."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from prepare_real10_event_windows import build, read_jsonl, write_json, write_jsonl
from review_real10_event_windows import ANNOTATION_SCHEMA, ReviewStore, make_server


def fixture(root):
    episodes = ["fixture_left", "fixture_right"]
    for episode, task in zip(episodes, ("left", "right")):
        folder = root / episode
        folder.mkdir()
        (folder / "image.png").write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
        write_json(folder / "manifest.json", {"task": task, "camera_setup": {"top_duplicated_from_side": False}})
        rows = []
        for step in range(6):
            t = step * .25
            ctl = {"piper_busy": step == 2, "piper_step_after_command": int(step >= 2),
                   "piper_async_status": "running" if step == 2 else "idle",
                   "piper_async_command": int(step == 2),
                   "piper_executed_command": "feed_feeder_python_socket_async_started" if step == 2 else "hold",
                   "piper_async_start_timestamp": .54 if step >= 2 else None,
                   "piper_async_done_timestamp": .78 if step >= 3 else None,
                   "piper_async_error": None, "event_id": 999}
            rows.append({"step": step, "task": task, "timestamp": t,
                         "timestamps": {"side_image": t, "top_image": t + .01,
                                        "elite_pose": t + .03, "piper_action": t + .05},
                         "side_image": "image.png", "top_image": "image.png",
                         "state": {"elite_tcp_pose_6d": [step, 0, 0, 0, 0, 0],
                                   "controller_state": ctl, "estimated_contact_flag": True,
                                   "exact_tip": [999, 999, 999]},
                         "reference_action": {"piper_step_command": int(step == 2), "piper_burst_count": 1,
                                              "elite_tcp_delta_6d": [1 if step < 5 else 0, 0, 0, 0, 0, 0]}})
        write_jsonl(folder / "records.jsonl", rows)
    manifest = root / "pack_manifest.json"
    write_json(manifest, {"adapter_version": "real10_pi05_observable_history_v1",
                          "source_episodes": episodes, "counts": {"source_records": 12, "samples": 10},
                          "training_use": {"split_strategy": "all_10_episodes_train_no_validation"}})
    return manifest


class EventWindowTest(unittest.TestCase):
    def test_causal_history_and_separate_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture(root)
            report = build(root, manifest, root / "out", hold_windows=1)
            self.assertEqual((report["source_records"], report["factual_transitions"]), (12, 10))
            self.assertEqual(report["matched_executor_completions"], 2)
            obs = read_jsonl(root / "out/observations.jsonl")
            self.assertFalse(obs[0]["controller_history"]["valid"])
            self.assertEqual(obs[2]["controller_history"]["values"]["piper_async_command"], 0)
            self.assertEqual(obs[3]["controller_history"]["values"]["piper_async_command"], 1)
            self.assertIsNone(obs[3]["controller_history"]["values"]["piper_executed_feed"])
            self.assertIn("piper_executed_feed", obs[3]["controller_history"]["missing_fields"])
            self.assertNotIn("piper_async_error", obs[3]["controller_history"]["missing_fields"])
            self.assertIn("piper_async_error", obs[3]["controller_history"]["recorded_null_fields"])
            self.assertEqual(report["controller_missing_rows"]["piper_async_error"], 2)
            self.assertNotIn("exact_tip", json.dumps(obs))
            self.assertNotIn("estimated_contact_flag", json.dumps(obs))
            self.assertNotIn("event_id", json.dumps(obs))
            by_id = {o["frame_id"]: o for o in obs}
            wins = read_jsonl(root / "out/windows.jsonl")
            targets = {t["window_id"]: t for t in read_jsonl(root / "out/window_targets.jsonl")}
            for w in wins:
                self.assertNotIn("future_frame_ids", w)
                self.assertTrue(all(by_id[k]["observation_available_at_s"] <= w["anchor_timestamp_s"] for k in w["history_frame_ids"]))
                self.assertTrue(all(by_id[k]["observation_available_at_s"] > w["anchor_timestamp_s"] for k in targets[w["window_id"]]["future_frame_ids"]))
                self.assertEqual(w["source_split"], "train")
            self.assertTrue(all(a["anchor_visibility"]["side"] is None for a in read_jsonl(root / "out/annotation_template.jsonl")))
            # Delayed previous snapshot must be missing, not silently carried back.
            rows = read_jsonl(root / "fixture_left/records.jsonl")
            rows[1]["timestamps"]["piper_action"] = .8
            write_jsonl(root / "fixture_left/records.jsonl", rows)
            build(root, manifest, root / "delayed", hold_windows=1)
            self.assertFalse(read_jsonl(root / "delayed/observations.jsonl")[2]["controller_history"]["valid"])

    def test_http_save_resume_unknown_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, fixture(root), root / "out", hold_windows=0)
            store = ReviewStore(root / "out", root)
            server = make_server(store, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/api/index") as response:
                    self.assertEqual(len(json.load(response)["windows"]), 2)
                with urlopen(url + "/image?id=fixture_left/000002&view=side") as response:
                    self.assertTrue(response.read().startswith(b"\x89PNG"))
                payload = {"annotation_schema": ANNOTATION_SCHEMA,
                           "window_id": "fixture_left/000002", "reviewer": "synthetic_integration_check",
                           "anchor_visibility": {"side": "unobservable", "top": "ambiguous"},
                           "joint_motion_response": "uncertain", "motion_evidence": "insufficient", "notes": "fixture only"}
                request = Request(url + "/api/annotation", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)["status"], "reviewed")
                restored = ReviewStore(root / "out", root)
                self.assertEqual(restored.annotations[payload["window_id"]]["joint_motion_response"], "uncertain")
                self.assertTrue((root / "out/annotations_joint_v2.jsonl").exists())
                self.assertFalse((root / "out/annotations.jsonl").exists())
                self.assertFalse((root / "fixture_left/annotations.jsonl").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_no_future_never_accepts_positive_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, fixture(root), root / "out", future_s=.01, hold_windows=0)
            store = ReviewStore(root / "out", root)
            self.assertFalse(store.targets["fixture_left/000002"]["future_frame_ids"])
            with self.assertRaisesRegex(ValueError, "no future"):
                store.save({"annotation_schema": ANNOTATION_SCHEMA,
                            "window_id": "fixture_left/000002", "reviewer": "synthetic_check",
                            "anchor_visibility": {"side": "clear", "top": "clear"},
                            "joint_motion_response": "advance", "motion_evidence": "both"})

    def test_legacy_projection_preserves_revisions_and_never_guesses_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, fixture(root), root / "out", hold_windows=0)
            old = read_jsonl(root / "out/annotation_template.jsonl")
            for row in old:
                row.update(reviewer="fixture_reviewer", status="reviewed", saved_at_utc="fixture_time")
                row["anchor_visibility"] = {"side": "clear", "top": "unobservable"}
            old[0]["future_motion_response"] = {"side": "stationary", "top": "stationary"}
            old[1]["future_motion_response"] = {"side": "advance", "top": "stationary"}
            revision = {**old[0], "future_motion_response": {"side": "advance", "top": "advance"}}
            legacy_path = root / "out/annotations.jsonl"
            write_jsonl(legacy_path, [*old, revision])
            before = legacy_path.read_bytes()
            store = ReviewStore(root / "out", root)
            row = store.annotations[old[0]["window_id"]]
            self.assertEqual(row["joint_motion_response"], "advance")
            self.assertIsNone(row["motion_evidence"])
            self.assertEqual(row["status"], "needs_evidence")
            self.assertEqual(row["legacy_migration"]["source_line"], 3)
            self.assertIsNone(store.annotations[old[1]["window_id"]]["joint_motion_response"])
            self.assertFalse(store.path.exists())
            self.assertEqual(store.summary()["legacy_revisions"], 3)
            payload = {"annotation_schema": ANNOTATION_SCHEMA, "window_id": row["window_id"],
                       "reviewer": "fixture_reviewer", "anchor_visibility": row["anchor_visibility"],
                       "joint_motion_response": "advance", "motion_evidence": "side", "notes": "fixture"}
            saved = store.save(payload)
            self.assertEqual(saved["status"], "reviewed")  # Top occlusion does not veto Side evidence.
            self.assertEqual(saved["legacy_migration"]["source_line"], 3)
            self.assertEqual(ReviewStore(root / "out", root).annotations[row["window_id"]]["motion_evidence"], "side")
            with self.assertRaisesRegex(ValueError, "版本"):
                store.save(revision)
            with self.assertRaisesRegex(ValueError, "证据不足"):
                store.save({**payload, "motion_evidence": "insufficient"})
            self.assertEqual(len(read_jsonl(store.path)), 1)
            self.assertEqual(legacy_path.read_bytes(), before)


class ResponseBaselineTests(unittest.TestCase):
    def test_latest_v2_selection_and_sidecar_isolation(self):
        from run_real10_response_baseline import select_windows, folds_for
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            windows, annotations, targets = [], [], []
            episodes = ["fixture_a", "fixture_b", "fixture_c"]
            for i in range(6):
                episode = episodes[i // 2]
                key = f"{episode}/{i:06d}"
                provenance = {"window_id": key, "source_episode": episode,
                              "split_group": episode, "source_split": "train"}
                windows.append({**provenance, "task": "left", "anchor_frame_id": key,
                                "review_selection_reason": "not_a_feature"})
                annotations.append({**provenance, "annotation_schema": "real10_joint_response_annotation_v2",
                                    "annotation_source": "human_visual_review", "reviewer": "synthetic",
                                    "anchor_visibility": {"side": "clear", "top": "ambiguous"},
                                    "joint_motion_response": "advance" if i % 2 == 0 else "stationary",
                                    "motion_evidence": "side", "status": "reviewed"})
                targets.append({"window_id": key, "future_frame_ids": [f"{episode}/{i + 1:06d}"],
                                "exact_contact": "must_not_be_a_feature"})
            annotations += [{**annotations[0], "joint_motion_response": "stationary"},
                            {**annotations[1], "joint_motion_response": "uncertain"}]
            write_json(root / "manifest.json", {"source_episodes": episodes})
            for name, rows in (("windows", windows), ("window_targets", targets),
                               ("annotations_joint_v2", annotations)):
                write_jsonl(root / f"{name}.jsonl", rows)
            for name in ("annotations", "execution_log", "factual_transitions", "diagnostic_targets"):
                (root / f"{name}.jsonl").write_text("must not be opened", encoding="utf-8")
            selected, excluded, revision_count = select_windows(root)
            self.assertEqual(revision_count, 8)
            self.assertEqual(len(selected), 5)
            self.assertEqual(selected[0]["human_annotation"]["joint_motion_response"], "stationary")
            self.assertEqual(excluded[0]["reason"], "uncertain")
            seen = []
            for fold in folds_for(selected):
                train, test = fold["train_indices"], fold["test_indices"]
                self.assertFalse(set(train) & set(test))
                self.assertTrue(all(selected[i]["source_episode"] != fold["heldout_episode"] for i in train))
                self.assertTrue(all(selected[i]["source_episode"] == fold["heldout_episode"] for i in test))
                seen.extend(test)
            self.assertEqual(sorted(seen), list(range(5)))

    def test_train_only_fit_missing_values_and_controller_allowlist(self):
        import copy
        import numpy as np
        from run_real10_response_baseline import fit_ridge, predict_ridge, controller_features, metrics
        x = np.array([[0., np.nan], [2., 1.], [4., np.nan], [6., 3.]])
        y = np.array([0, 1, 0, 1])
        model = fit_ridge(x, y)
        np.testing.assert_allclose(model["mean"], [3., 2.])
        heldout = np.array([[1.e5, np.nan]])
        self.assertTrue(np.isfinite(predict_ridge(model, heldout)).all())
        np.testing.assert_allclose(model["mean"], [3., 2.])
        self.assertIsNone(metrics(np.array([1]), np.array([1]))["balanced_accuracy"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, fixture(root), root / "out", hold_windows=0)
            observations = read_jsonl(root / "out/observations.jsonl")[:3]
            clean, names = controller_features(observations)
            poisoned = copy.deepcopy(observations)
            for row in poisoned:
                row.update(exact_contact=1, diagnostic_targets={"truth": 999},
                           joint_motion_response="advance", motion_evidence="both")
                row["controller_history"]["values"]["event_id"] = 99999
            changed, changed_names = controller_features(poisoned)
            np.testing.assert_allclose(clean, changed, equal_nan=True)
            self.assertEqual(names, changed_names)
            self.assertNotIn("event_id", names)

    def test_fixed_clip_train_statistics_missing_bits_and_refit_center(self):
        import numpy as np
        from run_real10_response_baseline import fit_ridge, predict_ridge, _ridge_standardized_features
        x = np.column_stack((np.r_[np.zeros(39), 100.], np.tile([1., np.nan], 20)))
        y = np.tile([0, 1], 20)
        old = fit_ridge(x, y, block_sizes=[2])
        model = fit_ridge(x, y, block_sizes=[2], standardized_clip=5.)
        np.testing.assert_array_equal(model["mean"], old["mean"])
        np.testing.assert_array_equal(model["std"], old["std"])
        self.assertNotIn("standardized_clip", old)
        z = _ridge_standardized_features(x, model["mean"], model["std"], 5.)
        self.assertEqual(z[-1, 0], 5.)
        np.testing.assert_array_equal(z[:, 2:], ~np.isfinite(x))
        np.testing.assert_allclose(model["center"], np.mean(z / model["feature_scale"], axis=0))
        self.assertFalse(np.allclose(model["center"], old["center"]))
        heldout = np.array([[1.e9, np.nan]])
        np.testing.assert_array_equal(_ridge_standardized_features(heldout, model["mean"], model["std"], 5.), [[5., 0., 0., 1.]])
        expected = predict_ridge(model, heldout)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clipped.npz"
            np.savez_compressed(path, **model)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(predict_ridge(saved, heldout), expected)
        np.testing.assert_array_equal(model["mean"], old["mean"])
        np.testing.assert_array_equal(model["std"], old["std"])

    def test_response_roi_original_pixels_and_shared_block_scale(self):
        import numpy as np
        import cv2
        from run_real10_response_baseline import fit_ridge, predict_ridge
        from run_real10_response_roi_comparison import crop_and_resize, ROI
        original = np.zeros((1080, 1920, 3), dtype=np.uint8)
        original[540:1040, :1440] = [10, 50, 100]
        for view in ("side", "top"):
            left, top, right, bottom = ROI[view]
            expected = cv2.resize(original[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(crop_and_resize(original, view), expected)
        with self.assertRaises(ValueError):
            crop_and_resize(np.zeros((224, 224, 3), dtype=np.uint8), "side")
        x = np.array([[0., np.nan], [2., 1.], [4., np.nan], [6., 3.]])
        y = np.array([0, 1, 0, 1])
        old = fit_ridge(x, y)
        grouped = fit_ridge(x, y, block_sizes=[2])
        np.testing.assert_allclose(predict_ridge(old, x), predict_ridge(grouped, x))
        fused = fit_ridge(np.concatenate((x, np.arange(12).reshape(4, 3)), axis=1), y, block_sizes=[2, 3])
        np.testing.assert_allclose(fused["feature_scale"][[0, 1, 5, 6]], grouped["feature_scale"])
        self.assertEqual(fused["mean"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
