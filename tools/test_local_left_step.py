"""Critical rejection cases for the one-step image-directed hardware plan."""
import copy
import unittest

from execute_local_left_step_http import local_plan


class LocalLeftPlanTests(unittest.TestCase):
    def setUp(self):
        self.current = {"pose": [0, 0, 0, 0, 0, 0]}
        self.probe = {"status": "completed_three_axis_round_trips", "after": self.current,
                      "events": [{"axis": "Y", "direction": "outbound", "arrived": True,
                                  "image_shifts": {"side": {"delta_px": [33, 2], "ncc": .95},
                                                   "top": {"delta_px": [-16, 0], "ncc": .92}}}]}
        self.annotation = {"views": {v: {"tip": {"status": "visible", "xy": [800, 700]},
                                          "target": {"status": "visible", "xy": xy}}
                                     for v, xy in [("side", [438, 673]), ("top", [970, 693])]}}

    def test_plan_is_exactly_ten_mm_and_holds_height_orientation(self):
        p = local_plan(self.probe, self.annotation, self.current)
        self.assertEqual(p["target_pose_mm_rad"], [0, -10, 0, 0, 0, 0])

    def test_ambiguous_or_opposite_view_is_rejected(self):
        bad = copy.deepcopy(self.probe)
        bad["events"][0]["image_shifts"]["top"]["ncc"] = .5
        with self.assertRaises(ValueError): local_plan(bad, self.annotation, self.current)
        bad = copy.deepcopy(self.annotation)
        bad["views"]["top"]["target"]["xy"] = [600, 700]
        with self.assertRaises(ValueError): local_plan(self.probe, bad, self.current)

    def test_changed_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            local_plan(self.probe, self.annotation, {"pose": [.1, 0, 0, 0, 0, 0]})


if __name__ == "__main__":
    unittest.main()
