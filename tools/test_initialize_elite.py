"""Offline checks for cold startup ordering and fail-closed dispatch."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from initialize_elite_http import initialize_once


class FakeEC:
    def __init__(self):
        self.state = SimpleNamespace(value=0)
        self.mode = SimpleNamespace(value=2)
        self.estop_status = 0
        self.servo_status = False
        self.sync_status = False
        self.initializing = 1
        self.brakes = [0]*6
        self.motor_speed = [0]*6
        self.motors = [10.]*6
        self.calls = []
        self.clear_reply = True

    def get_digital_io(self, name):
        return self.initializing

    def send_CMD(self, name):
        assert name == "get_servo_brake_off_status"
        return deepcopy(self.brakes)

    def get_motor_pos(self):
        return list(self.motors)

    def clear_alarm(self):
        self.calls.append("clear")
        self.brakes = [1]*6
        self.initializing = 0
        return self.clear_reply

    def sync(self):
        self.calls.append("sync")
        self.sync_status = True
        return True

    def set_servo_status(self, value):
        assert value == 1
        self.calls.append("servo")
        self.servo_status = True
        return True

    def stop(self):
        self.calls.append("stop")
        return True


class StartupTests(unittest.TestCase):
    def test_order_and_no_repeat_when_ready(self):
        ec, report = FakeEC(), {}
        initialize_once(ec, report, lambda: None, lambda: None)
        self.assertEqual(ec.calls, ["clear", "sync", "servo"])
        ec.calls.clear()
        initialize_once(ec, {}, lambda: None, lambda: None)
        self.assertEqual(ec.calls, [])

    def test_invalid_state_sends_nothing(self):
        ec = FakeEC()
        ec.state.value = 4
        with self.assertRaises(RuntimeError):
            initialize_once(ec, {}, lambda: None, lambda: None)
        self.assertEqual(ec.calls, [])

    def test_ambiguous_ack_stops_without_retry(self):
        ec = FakeEC()
        ec.clear_reply = (False, "SDK error")
        with self.assertRaises(RuntimeError):
            initialize_once(ec, {}, lambda: None, lambda: None)
        self.assertEqual(ec.calls, ["clear", "stop"])

    def test_motion_or_camera_loss_blocks_following_stage(self):
        for failure in ("motor", "camera"):
            ec = FakeEC()
            def cameras():
                if ec.calls:
                    if failure == "camera":
                        raise RuntimeError("camera lost")
                    ec.motors[0] += .3
            with self.assertRaises(RuntimeError):
                initialize_once(ec, {}, lambda: None, cameras)
            self.assertEqual(ec.calls, ["clear", "stop"])


if __name__ == "__main__":
    unittest.main()
