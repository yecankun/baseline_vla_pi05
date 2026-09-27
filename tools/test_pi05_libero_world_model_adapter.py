"""Synthetic-only native LIBERO schema tests; never loads demonstrations/policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pi05_libero_world_model_adapter import (
    ARRAY_NAMES, LAYOUT, SCHEMA, SPLIT_SCHEMA, LiberoFeaturePack, LiberoWindowDataset,
    adapt_native_record, fit_train_normalization, read_json, sha256_file,
)
from prepare_pi05_libero_world_model import prepare, write_json


def make_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    tasks = [{"task_id": i, "source_task_index": i + 20,
              "task_instruction": f"SYNTHETIC schema task {i}; not a LIBERO demonstration"}
             for i in range(10)]
    # 20 episodes, 9 observations each; one train and one val episode per task.
    episodes = [{"episode_index": e, "task_id": e // 2, "source_trajectory_id": f"synthetic-source-{e}",
                 "leakage_group_id": f"synthetic-group-{e}", "record_count": 9, "complete_episode": True}
                for e in range(20)]
    ep = np.repeat(np.arange(20, dtype=np.int64), 9)
    frame = np.tile(np.arange(9, dtype=np.int64), 20)
    state = np.repeat((frame + ep * 10)[:, None], 8, axis=1).astype(np.float32)
    state[:, 6] = 3.0  # observed constant coordinate
    state[:, 7] = np.nan  # no training support: masked in every frame
    mask = np.ones((180, 8), dtype=bool)
    mask[:, 7] = False
    latent = np.repeat(frame[:, None, None], 2, axis=1)
    latent = np.repeat(latent, 5, axis=2).astype(np.float32)
    latent[:, 1] += 100  # view order must remain identifiable
    arrays = {
        "episode_index": ep, "frame_index": frame, "task_id": ep // 2,
        "state": state, "state_valid": mask,
        "action": np.repeat((frame / 10)[:, None], 7, axis=1).astype(np.float32),
        "visual_latent": latent, "visual_valid": np.ones((180, 2), dtype=bool),
    }
    manifest = {
        "schema": SCHEMA, "benchmark": "libero_spatial", "layout": LAYOUT, "fps": 10.0,
        "source": {"kind": "synthetic_fixture", "repo_id": "synthetic/libero_contract_fixture",
                   "revision": "synthetic-v1", "metadata_sha256": "0" * 64},
        "extractor": {"kind": "synthetic_fixture", "checkpoint_sha256": "0" * 64,
                      "preprocessing_sha256": "0" * 64, "code_sha256": "0" * 64},
        "task_registry": tasks, "episodes": episodes, "arrays": {},
    }
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        manifest["arrays"][name] = {"path": path.name, "sha256": sha256_file(path)}
    write_json(root / "manifest.json", manifest)
    bind_split(root)
    return root


def bind_split(root: Path) -> None:
    write_json(root / "split.json", {
        "schema": SPLIT_SCHEMA, "feature_pack_manifest_sha256": sha256_file(root / "manifest.json"),
        "train_episode_indices": list(range(0, 20, 2)), "validation_episode_indices": list(range(1, 20, 2)),
    })


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = make_fixture(Path(self.temp.name) / "fixture")
        self.addCleanup(self.temp.cleanup)

    def pack(self):
        return LiberoFeaturePack(self.root, allow_synthetic_fixture=True)

    def setup_dataset(self, partition="train"):
        pack = self.pack()
        split = pack.load_split(self.root / "split.json")
        stats = fit_train_normalization(pack, split)
        return pack, split, stats, LiberoWindowDataset(pack, split, partition=partition, normalization=stats)

    def manifest_change(self, change):
        value = read_json(self.root / "manifest.json")
        change(value)
        write_json(self.root / "manifest.json", value)
        bind_split(self.root)

    def array_change(self, name, change):
        path = self.root / f"{name}.npy"
        value = np.load(path, allow_pickle=False)
        changed = change(value)
        if changed is not None:
            value = changed
        np.save(path, value, allow_pickle=False)
        self.manifest_change(lambda m: m["arrays"][name].update(sha256=sha256_file(path)))

    def record(self):
        return {"episode_index": 0, "frame_index": 3, "task_index": 29,
                "task": "SYNTHETIC schema task 9; not a LIBERO demonstration",
                "observation.state": np.arange(8, dtype=float), "action": np.zeros(7)}

    def adapt(self, record):
        return adapt_native_record(record, read_json(self.root / "manifest.json")["task_registry"])

    def test_native_task_mapping_and_dimensions(self):
        adapted = self.adapt(self.record())
        self.assertEqual(adapted["task_id"], 9)
        self.assertEqual(adapted["state"].shape, (8,))
        self.assertEqual(adapted["action"].shape, (7,))
        self.assertEqual(adapted["task_instruction"], self.record()["task"])

    def test_source_index_not_suite_index(self):
        record = self.record()
        record["task_index"] = 9
        with self.assertRaises(ValueError): self.adapt(record)

    def test_task_text_mismatch(self):
        record = self.record()
        record["task"] = "left"
        with self.assertRaises(ValueError): self.adapt(record)

    def test_missing_state_requires_mask(self):
        record = self.record()
        del record["observation.state"]
        with self.assertRaises(ValueError): self.adapt(record)
        record["observation.state_valid"] = np.zeros(8, dtype=bool)
        adapted = self.adapt(record)
        np.testing.assert_array_equal(adapted["state"], np.zeros(8))
        self.assertFalse(adapted["state_valid"].any())

    def test_invalid_state_mask_cannot_hide_valid_nan(self):
        record = self.record()
        record["observation.state"][2] = np.nan
        with self.assertRaises(ValueError): self.adapt(record)
        record["observation.state_valid"] = np.ones(8, dtype=bool)
        record["observation.state_valid"][2] = False
        self.assertEqual(self.adapt(record)["state"][2], 0)

    def test_nonboolean_mask_rejected(self):
        record = self.record()
        record["observation.state_valid"] = np.ones(8, dtype=float)
        with self.assertRaises(ValueError): self.adapt(record)

    def test_wrong_native_dimensions_and_actions(self):
        for key, bad in (("observation.state", np.zeros(32)), ("action", np.zeros(9)),
                         ("action", np.full(7, np.nan)), ("action", np.full(7, 1.01))):
            with self.subTest(key=key, shape=bad.shape):
                record = self.record()
                record[key] = bad
                with self.assertRaises(ValueError): self.adapt(record)

    def test_fractional_ids_rejected(self):
        record = self.record()
        record["episode_index"] = 1.5
        with self.assertRaises(ValueError): self.adapt(record)

    def test_policy_forbidden_fields(self):
        for key in ("diagnostic_targets", "object_state", "reward", "success", "sample_role",
                    "piper_intent_id", "exact_contact", "route_truth", "raw_event_id"):
            record = self.record()
            record[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError): self.adapt(record)

    def test_opt_in_and_eval_source_excluded(self):
        with self.assertRaises(ValueError): LiberoFeaturePack(self.root)
        self.manifest_change(lambda m: m["source"].update(kind="evaluation_rollouts"))
        with self.assertRaises(ValueError): self.pack()

    def test_contiguous_windows_and_alignment(self):
        pack, split, stats, dataset = self.setup_dataset()
        self.assertEqual(len(dataset), 30)
        item = dataset[0]
        self.assertEqual(item["metadata"]["history_indices"], [0, 1, 2, 3])
        self.assertEqual(item["metadata"]["action_indices"], [3, 4, 5])
        self.assertEqual(item["metadata"]["target_indices"], [4, 5, 6])
        np.testing.assert_array_equal(item["targets"]["future_visual_latent"][:, 0, 0], [4, 5, 6])
        np.testing.assert_array_equal(item["targets"]["future_visual_latent"][:, 1, 0], [104, 105, 106])
        np.testing.assert_allclose(item["targets"]["state_delta"][:, 0], np.arange(1, 4) / stats["state_std"][0])
        expected_action = (np.arange(3, 6) / 10 - stats["action_mean"][0]) / stats["action_std"][0]
        np.testing.assert_allclose(item["inputs"]["candidate_actions"][0, :, 0], expected_action, atol=1e-6)
        for w in dataset.windows:
            self.assertTrue(np.all(pack.arrays["episode_index"][list(w.history + w.targets)] == w.episode_index))
            self.assertEqual(len(w.history), 4)
            self.assertEqual(len(w.actions), 3)

    def test_metadata_targets_not_model_inputs(self):
        item = self.setup_dataset()[3][0]
        self.assertEqual(set(item["inputs"]), {"history_visual_latent", "history_visual_valid", "history_state",
                                              "history_state_valid", "task_instruction", "candidate_actions"})
        self.assertNotIn("task_id", item["inputs"])
        self.assertNotIn("source_task_index", item["inputs"])
        self.assertEqual(item["inputs"]["candidate_actions"].shape, (1, 3, 7))

    def test_missing_and_unsupported_coordinates(self):
        _, _, stats, dataset = self.setup_dataset()
        self.assertEqual(stats["state_std"][6], 1)
        self.assertEqual(stats["state_count"][7], 0)
        self.assertEqual(stats["state_mean"][7], 0)
        self.assertEqual(stats["state_std"][7], 1)
        self.assertFalse(dataset[0]["inputs"]["history_state_valid"][:, 7].any())
        self.assertFalse(dataset[0]["targets"]["state_target_valid"][:, 7].any())
        np.testing.assert_array_equal(dataset[0]["inputs"]["history_state"][:, 7], 0)

    def test_validation_has_no_effect_on_training_stats(self):
        stats = self.setup_dataset()[2]
        def change(values): values[9:18, :6] = 123456
        self.array_change("state", change)
        def change_action(values): values[9:18] = -0.99
        self.array_change("action", change_action)
        changed = self.setup_dataset()[2]
        for key in ("state_mean", "state_std", "state_count", "action_mean", "action_std"):
            self.assertEqual(stats[key], changed[key])

    def test_last_action_excluded_from_stats(self):
        stats = self.setup_dataset()[2]
        self.assertEqual(stats["state_record_count"], 90)
        self.assertEqual(stats["action_record_count"], 80)
        def change(values): values[8::9] = -1
        self.array_change("action", change)
        new_stats = self.setup_dataset()[2]
        self.assertEqual(stats["action_mean"], new_stats["action_mean"])

    def test_validation_only_coordinate_disabled(self):
        def change(values): values[9:18, 7] = 77
        self.array_change("state", change)
        def change_mask(values): values[9:18, 7] = True
        self.array_change("state_valid", change_mask)
        item = self.setup_dataset("validation")[3][0]
        self.assertFalse(item["inputs"]["history_state_valid"][:, 7].any())
        self.assertFalse(item["targets"]["state_target_valid"][:, 7].any())
        np.testing.assert_array_equal(item["inputs"]["history_state"][:, 7], 0)

    def test_invalid_anchor_masks_all_future_residuals(self):
        def change(values): values[3, 0] = False
        self.array_change("state_valid", change)
        item = self.setup_dataset()[3][0]
        self.assertFalse(item["targets"]["state_target_valid"][:, 0].any())
        np.testing.assert_array_equal(item["targets"]["state_delta"][:, 0], 0)

    def test_invalid_view_masks_latents(self):
        def change(values): values[2:5, 1] = False
        self.array_change("visual_valid", change)
        def change_latent(values): values[2:5, 1] = np.nan
        self.array_change("visual_latent", change_latent)
        item = self.setup_dataset()[3][0]
        np.testing.assert_array_equal(item["inputs"]["history_visual_latent"][2:4, 1], 0)
        np.testing.assert_array_equal(item["targets"]["future_visual_latent"][0, 1], 0)

    def test_future_visual_does_not_enter_inputs(self):
        before = self.setup_dataset()[3][0]["inputs"]
        def change(values): values[4:7] += 10000
        self.array_change("visual_latent", change)
        after = self.setup_dataset()[3][0]["inputs"]
        for key in before:
            if isinstance(before[key], np.ndarray): np.testing.assert_array_equal(before[key], after[key])
            else: self.assertEqual(before[key], after[key])

    def test_frame_gap_rejected(self):
        def change(values): values[4:9] += 1
        self.array_change("frame_index", change)
        with self.assertRaises(ValueError): self.pack()

    def test_duplicate_frame_reset_rejected(self):
        def change(values): values[4] = 0
        self.array_change("frame_index", change)
        with self.assertRaises(ValueError): self.pack()

    def test_task_switch_rejected(self):
        def change(values): values[4] = 1
        self.array_change("task_id", change)
        with self.assertRaises(ValueError): self.pack()

    def test_incomplete_episode_rejected(self):
        self.manifest_change(lambda m: m["episodes"][0].update(complete_episode=False))
        with self.assertRaises(ValueError): self.pack()

    def test_wrong_episode_count_rejected(self):
        self.manifest_change(lambda m: m["episodes"][0].update(record_count=8))
        with self.assertRaises(ValueError): self.pack()

    def test_tampered_hash_rejected(self):
        self.manifest_change(lambda m: m["arrays"]["state"].update(sha256="f" * 64))
        with self.assertRaises(ValueError): self.pack()

    def test_extra_arrays_rejected(self):
        self.manifest_change(lambda m: m["arrays"].update(exact_contact=m["arrays"]["state"]))
        with self.assertRaises(ValueError): self.pack()

    def test_path_escape_rejected(self):
        self.manifest_change(lambda m: m["arrays"]["state"].update(path="../outside.npy"))
        with self.assertRaises(ValueError): self.pack()

    def test_split_failures(self):
        original = read_json(self.root / "split.json")
        for kind in ("duplicate", "overlap", "unknown", "missing", "hash", "fractional"):
            split = deepcopy(original)
            if kind == "duplicate": split["train_episode_indices"].append(0)
            if kind == "overlap": split["train_episode_indices"].append(1)
            if kind == "unknown": split["train_episode_indices"][0] = 999
            if kind == "missing": split["train_episode_indices"].pop()
            if kind == "hash": split["feature_pack_manifest_sha256"] = "f" * 64
            if kind == "fractional": split["train_episode_indices"][0] = 0.5
            write_json(self.root / "split.json", split)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.pack().load_split(self.root / "split.json")

    def test_source_and_group_leakage(self):
        for key in ("source_trajectory_id", "leakage_group_id"):
            before = read_json(self.root / "manifest.json")
            self.manifest_change(lambda m: m["episodes"][1].update({key: m["episodes"][0][key]}))
            with self.subTest(key=key), self.assertRaises(ValueError): self.pack().load_split(self.root / "split.json")
            write_json(self.root / "manifest.json", before)
            bind_split(self.root)

    def test_mutated_split_and_unknown_window_ids_rejected(self):
        pack, split, _, _ = self.setup_dataset()
        split["train_episode_indices"][0] = 1
        with self.assertRaises(ValueError): fit_train_normalization(pack, split)
        with self.assertRaises(ValueError): pack.windows({999})
        with self.assertRaises(ValueError): pack.windows({0}, context_len=1.2)

    def test_normalization_is_copied_and_foreign_stats_rejected(self):
        pack, split, stats, dataset = self.setup_dataset()
        before = dataset[0]["inputs"]["history_state"].copy()
        stats["state_mean"][0] = 999
        np.testing.assert_array_equal(before, dataset[0]["inputs"]["history_state"])
        with self.assertRaises(ValueError): LiberoWindowDataset(pack, split, partition="train", normalization=stats)

    def test_prepare_zero_step_and_no_overwrite(self):
        out = Path(self.temp.name) / "prepared"
        report = prepare(self.root, self.root / "split.json", out, allow_synthetic_fixture=True)
        self.assertEqual(report["partitions"]["train"]["windows"], 30)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["training_ready"])
        with self.assertRaises(FileExistsError): prepare(self.root, self.root / "split.json", out, allow_synthetic_fixture=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        parser.error("output root already exists; choose a new attempt name")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeAdapterTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test_pi05_libero_feature_extraction"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    args.out_root.mkdir(parents=True, exist_ok=False)
    fixture = make_fixture(args.out_root / "synthetic_fixture")
    report = prepare(fixture, fixture / "split.json", args.out_root / "preparation", allow_synthetic_fixture=True)
    report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  fixture_records=180, fixture_episodes=20, actual_pi05_model_loaded=False,
                  adapter_and_feature_entrypoint_smoke=True)
    write_json(args.out_root / "report.json", report)
    print(json.dumps({"status": "passed", "tests": result.testsRun, "records": 180,
                      "episodes": 20, "windows": 60, "optimizer_steps": 0,
                      "out": str(args.out_root.resolve())}))


if __name__ == "__main__":
    main()
