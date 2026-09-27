import time
import numpy as np
import threading
from typing import (
    Optional,
)
from piper_sdk import *

J1 = [15481, 88462, -41593, 12043, 23556, 13409]
J2 = [13766, 92511, -46641, 11275, 20526, 14526]

class PiperRobot:
    def __init__(
        self,
        j1: Optional[list] = J1,
        j2: Optional[list] = J2,
        can_name: str = "can0",
        move_spd_rate_ctrl: int = 100,
        timeout: float = 5.0,
    ):
        self.piper = C_PiperInterface_V2(can_name)
        self.piper.ConnectPort()
        time.sleep(0.1)
        self.move_spd_rate_ctrl = int(move_spd_rate_ctrl)
        self._ensure_can_control(timeout=float(timeout))
        self._enable()
        self.j1 = j1
        self.j2 = j2

    def _get_pos(self):
        joint_state = self.piper.GetArmJointMsgs().joint_state
        joint_state = tuple(getattr(joint_state, f"joint_{i + 1}") / 1e3 * 0.0174533 for i in range(6))
        gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1e6
        return joint_state + (gripper,)

    def _stop_teach_mode(self):
        self.piper.EmergencyStop(0x01)
        time.sleep(1.0)
        limit_angle = [0.1745, 0.7854, 0.2094]
        pos = self._get_pos()
        while not (
            abs(pos[1]) < limit_angle[0]
            and abs(pos[2]) < limit_angle[0]
            and pos[4] < limit_angle[1]
            and pos[4] > limit_angle[2]
        ):
            time.sleep(0.01)
            pos = self._get_pos()
        self.piper.EmergencyStop(0x02)
        time.sleep(1.0)

    def _ensure_can_control(self, timeout: float):
        if self.piper.GetArmStatus().arm_status.ctrl_mode != 1:
            self._stop_teach_mode()
        over_time = time.time() + timeout
        while self.piper.GetArmStatus().arm_status.ctrl_mode != 1:
            if over_time < time.time():
                raise RuntimeError("Piper CAN mode switch failed; check teach mode / robot state")
            self.piper.ModeCtrl(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)
            time.sleep(0.01)

    def _enable(self):
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        time.sleep(0.01)
        self.piper.GripperCtrl(0, 1000, 0x01, 0x00)
        print("INFO: Piper enabled")

    def step_forward(self, pause_time=0.8):
        time.sleep(pause_time)
        # 夹爪角度（距离）单位为0.001mm
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)
        self.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0x00)
        time.sleep(0.5)
        self.piper.JointCtrl(self.j1[0], self.j1[1], self.j1[2], self.j1[3], self.j1[4], self.j1[5])
        time.sleep(0.5)
        self.piper.GripperCtrl(0, 1000, 0x01, 0x00)
        time.sleep(0.5)
        self.piper.JointCtrl(self.j2[0], self.j2[1], self.j2[2], self.j2[3], self.j2[4], self.j2[5])
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)
        time.sleep(0.5)

    def step_backward(self, pause_time=0.8):
        time.sleep(pause_time)
        # 夹爪角度（距离）单位为0.001mm
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)
        self.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0x00)
        time.sleep(0.5)
        self.piper.JointCtrl(self.j2[0], self.j2[1], self.j2[2], self.j2[3], self.j2[4], self.j2[5])
        time.sleep(0.5)
        self.piper.GripperCtrl(0, 1000, 0x01, 0x00)
        time.sleep(0.5)
        self.piper.JointCtrl(self.j1[0], self.j1[1], self.j1[2], self.j1[3], self.j1[4], self.j1[5])
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)
        time.sleep(0.5)
