"""Offline checks for fixed-distance diagnostics and arm/feed ordering."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import execute_left_coordination_http as runner


def prediction(xyz=(.003, .004, 0), intent=1):
    return {"elite_tcp_delta_6d": [*xyz, 0., 0., 0.], "piper_intent_id": intent}


class FixedDistanceChecks(unittest.TestCase):
    def test_readiness_requires_precise_encoder_state_without_initializing(self):
        robot = SimpleNamespace(
            state=0, mode=2, servo_status=True, sync_status=True, estop_status=0,
            RobotState=SimpleNamespace(STOP=0), RobotMode=SimpleNamespace(REMOTE=2),
        )
        calls = []
        def precise(is_block):
            calls.append(is_block)
            return 0
        robot.get_servo_precise_position_status = precise
        with self.assertRaisesRegex(RuntimeError, "precise_position"):
            runner.robot_ready(robot)
        self.assertEqual(calls, [False])
        robot.get_servo_precise_position_status = lambda is_block: 1
        self.assertEqual(runner.robot_ready(robot)["precise_position"], "1")

    def test_unknown_encoder_state_is_not_treated_as_ready(self):
        robot = SimpleNamespace(
            state=0, mode=2, servo_status=True, sync_status=True, estop_status=0,
            RobotState=SimpleNamespace(STOP=0), RobotMode=SimpleNamespace(REMOTE=2),
        )
        for status in (None, (False, "unsupported", 1), "unknown"):
            robot.get_servo_precise_position_status = lambda is_block: status
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                runner.robot_ready(robot)

    def test_fixed_distance_preserves_direction_and_pose_units(self):
        pose = [100., 200., 300., .1, -.2, .3]
        raw = prediction()
        plan = runner.coordination_plan(raw, pose, 10.)
        np.testing.assert_allclose(plan["elite_target_tcp_pose_6d"], [106., 208., 300., .1, -.2, .3])
        self.assertEqual(plan["raw_elite_tcp_delta_6d"], raw["elite_tcp_delta_6d"])
        self.assertFalse(plan["is_unmodified_policy_action"])

    def test_existing_scaled_mode_is_preserved(self):
        small = runner.coordination_plan(prediction(), [0.]*6)
        np.testing.assert_allclose(small["guarded_elite_tcp_delta_6d"], [.12, .16, 0, 0, 0, 0])
        large = runner.coordination_plan(prediction((3., 4., 0.)), [0.]*6)
        self.assertAlmostEqual(np.linalg.norm(large["guarded_elite_tcp_delta_6d"][:3]), 3.)

    def test_invalid_or_out_of_scope_requests_are_rejected(self):
        for length in (0., -1., 10.01, float("nan"), float("inf")):
            with self.subTest(length=length), self.assertRaises(ValueError):
                runner.coordination_plan(prediction(), [0.]*6, length)
        for raw in (prediction((0.,0.,0.)), prediction((float("nan"),0.,0.)), prediction(intent=0)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner.coordination_plan(raw, [0.]*6, 10.)
        raw = prediction()
        raw["elite_tcp_delta_6d"][3] = .1
        with self.assertRaises(ValueError):
            runner.coordination_plan(raw, [0.]*6, 10.)

    def run_order_case(self, case):
        robot = SimpleNamespace(current_pose=[0.]*6)
        feeder = SimpleNamespace(count=0)
        events = []
        plan = runner.coordination_plan(prediction(), [0.]*6, 10.)
        if case == "over_ceiling":
            plan["translation_limit_mm"] = 11.
        if case == "outside_target":
            robot.current_pose[0] = -1.
        report = {"dispatch_base_pose": [0.]*6}

        def move(ec, target, joints, speed, report, save, **kw):
            self.assertEqual(speed, 1.)
            events.append("arm")
            if case == "arm_failure":
                raise RuntimeError("test arm failure")
            ec.current_pose = target.copy()

        def check_image():
            events.append("camera")
            if case == "camera_failure":
                raise RuntimeError("test camera failure")
            if case == "drift":
                robot.current_pose[0] += .2

        def feed_once(wait_response):
            self.assertTrue(wait_response)
            events.append("feed")
            feeder.count += 1
            return None  # Missing reply must never trigger another send.

        feeder.feed_once = feed_once
        with patch.object(runner, "execute_elite_move", move), patch.object(runner, "robot_ready", return_value={}):
            if case == "arm_only":
                runner.coordinate_once(robot, None, plan, [0.]*6, report, lambda: None, check_image, arm_only=True)
                self.assertEqual(events, ["arm", "camera"])
                self.assertEqual(feeder.count, 0)
                self.assertEqual(report["status"], "completed_arm_only_awaiting_physical_review")
            elif case == "success":
                runner.coordinate_once(robot, feeder, plan, [0.]*6, report, lambda: None, check_image)
                self.assertEqual(events, ["arm", "camera", "feed"])
                self.assertEqual(feeder.count, 1)
                self.assertFalse(report["feeder_ack_received"])
                self.assertIsNone(report["feeder_physical_execution_confirmed"])
            else:
                with self.assertRaises(RuntimeError):
                    runner.coordinate_once(robot, feeder, plan, [0.]*6, report, lambda: None, check_image)
                self.assertEqual(feeder.count, 0)
                if case in ("over_ceiling", "outside_target"):
                    self.assertEqual(events, [])

    def test_single_feed_after_arm_and_images(self):
        self.run_order_case("success")

    def test_arm_only_never_sends_feed(self):
        self.run_order_case("arm_only")

    def test_failures_prevent_feed(self):
        for case in ("arm_failure", "camera_failure", "drift", "over_ceiling", "outside_target"):
            with self.subTest(case=case):
                self.run_order_case(case)


if __name__ == "__main__":
    unittest.main()
