"""Offline checks for a one-shot startup calibration and its stop path."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

import restore_elite_precision_http as runner


def sample(precise=0):
    return {"state": 0, "mode": 2, "servo": True, "sync": True, "estop": 0,
            "precise": precise, "pose": [0.]*6, "joints": [0.]*6}


class RestoreChecks(unittest.TestCase):
    def robot(self, reply=True):
        return SimpleNamespace(calibrate_encoder_zero=Mock(return_value=reply), stop=Mock(return_value=True))

    def test_restores_once(self):
        ec, report = self.robot(), {}
        with patch.object(runner, "snapshot", side_effect=[sample(), sample(1)]):
            runner.restore_once(ec, report, lambda: None, lambda: None)
        ec.calibrate_encoder_zero.assert_called_once_with()
        ec.stop.assert_not_called()
        self.assertEqual(report["status"], "precision_restored")

    def test_already_precise_sends_nothing(self):
        ec = self.robot()
        with patch.object(runner, "snapshot", return_value=sample(1)):
            runner.restore_once(ec, {}, lambda: None, lambda: None)
        ec.calibrate_encoder_zero.assert_not_called()
        ec.stop.assert_not_called()

    def test_bad_initial_state_sends_nothing(self):
        for key, value in (("state", 3), ("mode", 0), ("estop", 1), ("precise", None), ("servo", False)):
            ec, initial = self.robot(), sample()
            initial[key] = value
            with self.subTest(key=key), patch.object(runner, "snapshot", return_value=initial):
                with self.assertRaises(RuntimeError):
                    runner.restore_once(ec, {}, lambda: None, lambda: None)
            ec.calibrate_encoder_zero.assert_not_called()

    def test_ambiguous_reply_stops_without_retry(self):
        for reply in (False, (False, "error"), None):
            ec = self.robot(reply)
            with self.subTest(reply=reply), patch.object(runner, "snapshot", return_value=sample()):
                with self.assertRaises(RuntimeError):
                    runner.restore_once(ec, {}, lambda: None, lambda: None)
            ec.calibrate_encoder_zero.assert_called_once_with()
            ec.stop.assert_called_once_with()

    def test_bound_violation_stops(self):
        for key, index, value in (("pose", 0, 5.1), ("pose", 3, .021), ("joints", 1, 3.1)):
            ec, after = self.robot(), deepcopy(sample())
            after[key][index] = value
            with self.subTest(key=key, index=index), patch.object(runner, "snapshot", side_effect=[sample(), after]):
                with self.assertRaises(RuntimeError):
                    runner.restore_once(ec, {}, lambda: None, lambda: None)
            ec.stop.assert_called_once_with()

    def test_timeout_and_camera_loss_stop(self):
        for loss in (False, True):
            ec = self.robot()
            camera = Mock(side_effect=[None, RuntimeError("camera lost")]) if loss else lambda: None
            with self.subTest(loss=loss), patch.object(runner, "snapshot", return_value=sample()):
                with self.assertRaises((RuntimeError, TimeoutError)):
                    runner.restore_once(ec, {}, lambda: None, camera, timeout_s=15 if loss else 0)
            ec.calibrate_encoder_zero.assert_called_once_with()
            ec.stop.assert_called_once_with()

    def test_confirmed_tool_envelope(self):
        limits = {"tcp_mm": 5, "rpy_rad": np.deg2rad(5), "joint_deg": 3,
                  "rotation_rad": np.deg2rad(5), "tool_reach_mm": 500,
                  "tool_displacement_mm": 50}
        base, after = sample(), sample()
        after["pose"][4] = np.deg2rad(2)
        after["joints"][4] = 2
        runner.check_envelope(base, after, limits)
        # Coupled rotations can exceed the total angle while each RPY is <5°.
        after["pose"][3:] = [np.deg2rad(4)] * 3
        with self.assertRaises(RuntimeError):
            runner.check_envelope(base, after, limits)
        after["pose"][3:] = [0, np.deg2rad(2), 0]
        after["joints"][4] = 3.01
        with self.assertRaises(RuntimeError):
            runner.check_envelope(base, after, limits)

    def test_clear_only_confirmed_failure(self):
        for alarm in ("[0-7000-C],[0-E030-1]", "[],[],[],[],[0-7000-C]", "[0-1234-A],[0-7000-C]", "[0-1234-A]", None):
            ec = SimpleNamespace(alarm_info=alarm, motor_speed=[0]*6,
                                 clear_alarm=Mock(return_value=True))
            before = sample()
            before["state"] = 4
            with self.subTest(alarm=alarm), patch.object(runner, "snapshot", side_effect=[before, sample()]), patch.object(runner.time, "sleep"):
                if alarm and (alarm.startswith("[0-7000-C]") or alarm == "[],[],[],[],[0-7000-C]"):
                    runner.clear_calibration_failure_once(ec, {}, lambda: None, lambda: None)
                    ec.clear_alarm.assert_called_once_with()
                else:
                    with self.assertRaises(RuntimeError):
                        runner.clear_calibration_failure_once(ec, {}, lambda: None, lambda: None)
                    ec.clear_alarm.assert_not_called()

    def test_uncleared_or_moving_failure_never_calibrates(self):
        for speed, reply in ((1, True), (0, (False, "rejected"))):
            ec = SimpleNamespace(alarm_info="[0-7000-C]", motor_speed=[speed]*6,
                                 clear_alarm=Mock(return_value=reply))
            before = sample()
            before["state"] = 4
            with patch.object(runner, "snapshot", return_value=before):
                with self.assertRaises(RuntimeError):
                    runner.clear_calibration_failure_once(ec, {}, lambda: None, lambda: None)
            self.assertEqual(ec.clear_alarm.call_count, 0 if speed else 1)

    def test_j5_override_preserves_other_joints_and_original_reference(self):
        limits = {"tcp_mm": 5, "rpy_rad": np.deg2rad(5), "joint_deg": 3,
                  "joint_limits_deg": [3, 3, 3, 3, 5, 3],
                  "rotation_rad": np.deg2rad(5), "tool_reach_mm": 500,
                  "tool_displacement_mm": 50}
        base, current = sample(), sample()
        current["joints"][4] = 3.8
        current["pose"][4] = np.deg2rad(3.8)
        runner.check_envelope(base, current, limits)
        bad_other = deepcopy(current)
        bad_other["joints"][1] = 3.1
        with self.assertRaises(RuntimeError):
            runner.check_envelope(base, bad_other, limits)
        later = deepcopy(current)
        later["joints"][4] = 5.2
        # Only 1.4 deg since retry start, but 5.2 from original setup -> stop.
        ec = self.robot()
        with patch.object(runner, "snapshot", side_effect=[current, later]):
            with self.assertRaises(RuntimeError):
                runner.restore_once(ec, {}, lambda: None, lambda: None,
                                    limits=limits, envelope_base=base)
        ec.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
