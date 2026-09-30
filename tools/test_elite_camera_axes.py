from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import probe_elite_camera_axes as runner


def sample():
    return {"state": 0, "mode": 2, "servo": True, "sync": True, "estop": 0,
            "precise": 1, "pose": [0.]*6, "joints": [0.]*6}


class AxisChecks(unittest.TestCase):
    def test_global_envelope_does_not_follow_moving_base(self):
        base, good = sample(), sample()
        good["pose"][0] = 10
        runner.check_sample(base, good)
        for key, index, value in (("pose", 0, 13.1), ("pose", 3, .01), ("joints", 2, 3.1)):
            bad = deepcopy(good); bad[key][index] = value
            with self.assertRaises(RuntimeError):
                runner.check_sample(base, bad)

    def test_ambiguous_ack_stops_without_retry(self):
        ec = SimpleNamespace(current_pose=[0.]*6, move_joint=Mock(return_value=(False, "error")), stop=Mock(return_value=True))
        with patch.object(runner, "robot_ready"), patch.object(runner, "snapshot", return_value=sample()):
            with self.assertRaises(RuntimeError):
                runner.move_once(ec, sample(), [10,0,0,0,0,0], [0]*6, {}, lambda: None, lambda: None)
        ec.move_joint.assert_called_once()
        ec.stop.assert_called_once()

    def test_camera_loss_after_dispatch_stops(self):
        ec = SimpleNamespace(current_pose=[0.]*6, move_joint=Mock(return_value=True), stop=Mock(return_value=True))
        camera = Mock(side_effect=[None, RuntimeError("camera lost")])
        with patch.object(runner, "robot_ready"), patch.object(runner, "snapshot", return_value=sample()):
            with self.assertRaises(RuntimeError):
                runner.move_once(ec, sample(), [10,0,0,0,0,0], [0]*6, {}, lambda: None, camera)
        ec.stop.assert_called_once()

    def test_template_known_translation(self):
        rng = np.random.default_rng(7)
        before = rng.integers(0,256,(1080,1920,3),dtype=np.uint8)
        after = np.roll(before, (4,-7), axis=(0,1))
        result = runner.image_shift(before, after, [100,100,60,60])
        self.assertEqual(result["delta_px"], [-7,4])


if __name__ == "__main__":
    unittest.main()
