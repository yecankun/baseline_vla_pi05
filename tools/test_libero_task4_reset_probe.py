"""CPU mechanics only; native reset evidence comes from the separate probe."""
import random
import unittest

import numpy as np

from probe_libero_task4_reset import compare, leaves, rng_components


class FakeTensor:
    def tolist(self):
        return [1, 2, 3]


class FakeTorch:
    class random:
        get_rng_state = staticmethod(FakeTensor)

    class cuda:
        get_rng_state_all = staticmethod(lambda: [FakeTensor()])


class ResetHelpersTest(unittest.TestCase):
    def test_leaves_copy_arrays(self):
        source = np.arange(3)
        result = dict(leaves({"raw": {"x": source}, "ignored": object()}))
        source[0] = 99
        self.assertEqual(result["raw/x"].tolist(), [0, 1, 2])
        self.assertEqual(set(result), {"raw/x"})

    def test_object_arrays_excluded(self):
        self.assertEqual(list(leaves({"opaque": np.array([object()], dtype=object)})), [])

    def test_rng_capture_does_not_advance(self):
        random.seed(13)
        np.random.seed(13)
        a = rng_components(FakeTorch)
        b = rng_components(FakeTorch)
        self.assertEqual(a, b)
        np.random.random()
        c = rng_components(FakeTorch)
        self.assertEqual([k for k in a if a[k] != c[k]], ["numpy"])

    def test_compare_detects_missing_leaf_and_rng(self):
        a = {"label": "a", "init_payload_sha256": "i", "rng_before_reset": {},
             "final_raw_sha256": "r", "stages": [{"stage": "reset", "leaves": {"x": 1},
             "rng": {"numpy": "a"}, "local_rng": {}, "raw_observation_sha256": "r"}]}
        b = {**a, "label": "b", "stages": [{**a["stages"][0], "leaves": {}, "rng": {"numpy": "b"}}]}
        result = compare(a, b)
        self.assertEqual(result["stages"][0]["changed_leaves"], ["x"])
        self.assertEqual(result["stages"][0]["rng_changed"], ["numpy"])
        self.assertTrue(result["final_raw_equal"])

    def test_stage_mismatch_rejected(self):
        a = {"stages": [{"stage": "a"}]}
        with self.assertRaisesRegex(ValueError, "Mismatched stages"):
            compare(a, {"stages": [{"stage": "b"}]})


if __name__ == "__main__":
    unittest.main()
