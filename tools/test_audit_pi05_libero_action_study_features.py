"""Synthetic helper regressions only; not completed-public-cache evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_pi05_libero_action_study_features import (
    check_tensor_evidence, independent_normalization, independent_window_checks, lines,
)
from pi05_libero_world_model_adapter import LiberoFeaturePack, fit_train_normalization
from test_pi05_libero_world_model_adapter import make_fixture


class ReadbackHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = make_fixture(Path(self.temp.name) / "fixture")
        self.pack = LiberoFeaturePack(self.root, allow_synthetic_fixture=True)
        self.addCleanup(self.pack.close)
        self.split = self.pack.load_split(self.root / "split.json")
        self.stats = fit_train_normalization(self.pack, self.split)

    def test_independent_stats_masks_constant_and_terminal_actions(self):
        evidence = independent_normalization(self.pack, self.split, self.stats)
        self.assertEqual((evidence["state_rows"], evidence["action_rows"]), (90, 80))
        self.assertEqual(evidence["terminal_actions_excluded"], 10)
        self.assertEqual(self.stats["state_std"][-2:], [1.0, 1.0])
        self.assertEqual(self.stats["state_count"][-1], 0)
        self.assertTrue(np.allclose(self.stats["action_mean"], .35))

    def test_wrong_mean_rejected(self):
        stats = deepcopy(self.stats)
        stats["state_mean"][0] += .001
        with self.assertRaisesRegex(ValueError, "independent train normalization"):
            independent_normalization(self.pack, self.split, stats)

    def test_terminal_action_count_rejected(self):
        stats = deepcopy(self.stats)
        stats["action_record_count"] += 10
        with self.assertRaisesRegex(ValueError, "terminal-action"):
            independent_normalization(self.pack, self.split, stats)

    def test_every_window_both_partitions(self):
        for partition in ("train", "validation"):
            evidence = independent_window_checks(self.pack, self.split, self.stats, partition)
            self.assertEqual(evidence, {"windows": 30, "valid_state_target_coordinates": 630,
                                        "valid_visual_target_views": 180})

    def test_saved_windows_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, "saved window count"):
            independent_window_checks(self.pack, self.split, self.stats, "train", [])

    def test_bad_recorded_tensor_rejected(self):
        record = {"shape": [1, 3, 256, 256], "dtype": "torch.float32", "finite": True,
                  "sha256": "0" * 64, "min": 0.0, "max": 1.0}
        check_tensor_evidence(record, [1, 3, 256, 256], "torch.float32", 0, 1)
        record["max"] = 2.0
        with self.assertRaisesRegex(ValueError, "range"):
            check_tensor_evidence(record, [1, 3, 256, 256], "torch.float32", 0, 1)

    def test_jsonl_duplicate_and_nonfinite_rejected(self):
        path = Path(self.temp.name) / "bad.jsonl"
        for value in ('{"a":1,"a":2}\n', '{"a":NaN}\n', '\n'):
            path.write_text(value, encoding="utf-8")
            with self.assertRaises(ValueError):
                list(lines(path))


if __name__ == "__main__":
    unittest.main()
