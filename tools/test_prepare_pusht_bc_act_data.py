"""No network, dataset, LeRobot, environment or training in these tests."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.prepare_pusht_bc_act_data import (
    acquire, audit, canonical_hash, episode_split, safe_path, train_statistics, verify_files,
)


class DataGateTests(unittest.TestCase):
    def test_deterministic_whole_episode_split(self):
        first = episode_split(list(range(206)))
        self.assertEqual(first, episode_split(list(reversed(range(206)))))
        self.assertEqual(len(first["train_episodes"]), 186)
        self.assertEqual(len(first["val_episodes"]), 20)
        self.assertFalse(set(first["train_episodes"]) & set(first["val_episodes"]))
        self.assertEqual(set(first["train_episodes"] + first["val_episodes"]), set(range(206)))

    def test_invalid_split_rejected(self):
        for values, count in [([0, 0, 1], 1), ([0, 1], 0), ([0, 1], 2), ([0, -1], 1), ([False, 1], 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                episode_split(values, count)

    def test_train_statistics_exclude_validation(self):
        split = {"train_episodes": [0], "val_episodes": [1]}
        values = np.array([[1, 2], [3, 6], [1000, 2000]], dtype=float)
        stats, count = train_statistics(values, values * 2, [0, 0, 1], split)
        self.assertEqual(count, 2)
        self.assertEqual(stats["observation.state"], {"mean": [2.0, 4.0], "std": [1.0, 2.0]})
        self.assertEqual(stats["action"], {"mean": [4.0, 8.0], "std": [2.0, 4.0]})
        values[-1] = -1e9
        other, _ = train_statistics(values, values * 2, [0, 0, 1], split)
        self.assertEqual(stats, other)

    def test_stats_reject_overlap_unknown_nan_and_degenerate(self):
        values = [[1, 2], [3, 6], [5, 8]]
        with self.assertRaises(ValueError):
            train_statistics(values, values, [0, 0, 1], {"train_episodes": [0], "val_episodes": [0, 1]})
        with self.assertRaises(ValueError):
            train_statistics(values, values, [0, 0, 2], {"train_episodes": [0], "val_episodes": [1]})
        for bad in ([[1, 2]] * 3, [[1, 2], [float("nan"), 4], [5, 8]]):
            with self.assertRaises(ValueError):
                train_statistics(bad, values, [0, 0, 1], {"train_episodes": [0], "val_episodes": [1]})

    def test_path_escape_rejected(self):
        for value in ("../outside", "/absolute", "a/../../x", "a\\x", "C:/outside", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_path(Path("."), value)
        self.assertTrue(safe_path(Path("."), "meta/info.json").is_absolute())

    def test_source_hash_size_and_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "x"
            pin = {"files": {"x": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}}
            with self.assertRaises(ValueError):
                verify_files(root, pin)
            path.write_bytes(b"abc")
            self.assertEqual(verify_files(root, pin), pin["files"])
            path.write_bytes(b"xyz")
            with self.assertRaises(ValueError):
                verify_files(root, pin)
            path.write_bytes(b"ab")
            with self.assertRaises(ValueError):
                verify_files(root, pin)

    def test_large_download_refused_before_touching_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            with self.assertRaises(ValueError):
                acquire(root, {"files": {"large": {"size": 100_000_001, "sha256": "0" * 64}}})
            self.assertFalse(root.exists())

    def test_existing_source_not_redownloaded_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "x"
            path.write_bytes(b"abc")
            pin = {"files": {"x": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}}
            with patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
                self.assertEqual(acquire(root, pin), pin["files"])
                path.write_bytes(b"xyz")
                with self.assertRaises(ValueError):
                    acquire(root, pin)
            self.assertEqual(path.read_bytes(), b"xyz")

    def test_corrupt_source_cannot_create_audit_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            with self.assertRaises(ValueError):
                audit(root, {"files": {"missing": {"size": 1, "sha256": "0" * 64}}}, output)
            self.assertFalse(output.exists())

    def test_canonical_hash_stable_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
