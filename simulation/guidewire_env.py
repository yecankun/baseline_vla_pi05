import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


TASK_INSTRUCTIONS = {
    "left": "Guide the wire into the left branch while avoiding vessel contact.",
    "right": "Guide the wire into the right branch while avoiding vessel contact.",
}


@dataclass
class VesselSpec:
    trunk_length: float = 5.0
    branch_length: float = 5.0
    branch_angle_deg: float = 34.0
    radius: float = 0.55
    safety_margin: float = 0.18


@dataclass
class EnvConfig:
    max_steps: int = 180
    advance_step: float = 0.08
    max_steer: float = 0.16
    severe_contact_limit: int = 12
    render_size: int = 512
    world_padding: float = 1.2
    vessel: VesselSpec = field(default_factory=VesselSpec)


class Guidewire2DEnv:
    """
    MVP 2D guidewire environment.

    The vessel is represented by a Y-shaped centerline with a constant radius.
    The guidewire is represented by a tip position, a tip heading, and a history
    polyline. Contact/tactile values are generated from distance to the nearest
    vessel centerline segment.
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.task = "left"
        self.instruction = TASK_INSTRUCTIONS[self.task]
        self.step_count = 0
        self.severe_contact_count = 0
        self.tip_pos = np.zeros(2, dtype=np.float32)
        self.tip_dir = np.array([1.0, 0.0], dtype=np.float32)
        self.insertion_length = 0.0
        self.guidewire_points: List[np.ndarray] = []
        self._build_vessel()

    def _build_vessel(self):
        v = self.config.vessel
        theta = math.radians(v.branch_angle_deg)
        origin = np.array([0.0, 0.0], dtype=np.float32)
        split = np.array([v.trunk_length, 0.0], dtype=np.float32)
        left_end = split + v.branch_length * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        right_end = split + v.branch_length * np.array([math.cos(theta), -math.sin(theta)], dtype=np.float32)
        self.centerline = [origin, split, left_end, right_end]
        self.segments = [(origin, split, "trunk"), (split, left_end, "left"), (split, right_end, "right")]
        self.targets = {"left": left_end, "right": right_end}

        points = np.stack([origin, split, left_end, right_end])
        pad = self.config.world_padding
        self.world_min = points.min(axis=0) - pad
        self.world_max = points.max(axis=0) + pad

    def reset(self, task: str = "left", noise: float = 0.04) -> Dict:
        if task not in TASK_INSTRUCTIONS:
            raise ValueError(f"Unknown task: {task}. Expected one of {list(TASK_INSTRUCTIONS)}")
        self.task = task
        self.instruction = TASK_INSTRUCTIONS[task]
        self.step_count = 0
        self.severe_contact_count = 0
        self.tip_pos = np.array([0.15, 0.0], dtype=np.float32)
        self.tip_pos += self.rng.normal(0.0, noise, size=2).astype(np.float32)
        self.tip_dir = np.array([1.0, 0.0], dtype=np.float32)
        self.insertion_length = 0.0
        self.guidewire_points = [self.tip_pos.copy()]
        return self._observation(done=False, success=False, failure_reason=None)

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        steer = np.asarray(action.get("steer", [0.0, 0.0]), dtype=np.float32)
        if steer.shape != (2,):
            raise ValueError("action['steer'] must be a 2D vector")
        steer_norm = float(np.linalg.norm(steer))
        if steer_norm > self.config.max_steer:
            steer = steer / max(steer_norm, 1e-6) * self.config.max_steer

        advance = int(action.get("advance", 1))
        retreat = int(action.get("retreat", 0))
        self.tip_dir = self.tip_dir + steer
        self.tip_dir = self.tip_dir / max(float(np.linalg.norm(self.tip_dir)), 1e-6)

        delta = np.zeros(2, dtype=np.float32)
        if advance:
            delta += self.tip_dir * self.config.advance_step
        if retreat:
            delta -= self.tip_dir * self.config.advance_step
        self.tip_pos = self.tip_pos + delta
        self.insertion_length += float(np.linalg.norm(delta))
        self.step_count += 1
        self.guidewire_points.append(self.tip_pos.copy())

        tactile = self._tactile_state(self.tip_pos)
        if tactile["contact_strength"] > 0.85:
            self.severe_contact_count += 1
        else:
            self.severe_contact_count = 0

        success = self._reached_target()
        failure_reason = self._failure_reason(tactile)
        done = success or failure_reason is not None or self.step_count >= self.config.max_steps
        obs = self._observation(done=done, success=success, failure_reason=failure_reason)
        reward = self._reward(success, failure_reason, tactile)
        info = {
            "success": success,
            "failure_reason": failure_reason,
            "nearest_branch": tactile["nearest_branch"],
        }
        return obs, reward, done, info

    def _observation(self, done: bool, success: bool, failure_reason: Optional[str]) -> Dict:
        tactile = self._tactile_state(self.tip_pos)
        target = self.targets[self.task]
        return {
            "instruction": self.instruction,
            "task": self.task,
            "step": self.step_count,
            "tip_pos": self.tip_pos.astype(float).tolist(),
            "tip_dir": self.tip_dir.astype(float).tolist(),
            "insertion_length": self.insertion_length,
            "target_pos": target.astype(float).tolist(),
            "distance_to_target": float(np.linalg.norm(target - self.tip_pos)),
            "distance_to_wall": tactile["distance_to_wall"],
            "contact_flag": tactile["contact_flag"],
            "contact_direction": tactile["contact_direction"],
            "contact_strength": tactile["contact_strength"],
            "nearest_branch": tactile["nearest_branch"],
            "done": done,
            "success": success,
            "failure_reason": failure_reason,
        }

    def _reward(self, success: bool, failure_reason: Optional[str], tactile: Dict) -> float:
        target = self.targets[self.task]
        progress_reward = -0.02 * float(np.linalg.norm(target - self.tip_pos))
        contact_penalty = -0.5 * tactile["contact_strength"]
        if success:
            return 10.0 + progress_reward
        if failure_reason is not None:
            return -10.0 + contact_penalty
        return progress_reward + contact_penalty

    def _failure_reason(self, tactile: Dict) -> Optional[str]:
        if tactile["distance_to_wall"] < -self.config.vessel.radius * 0.45:
            return "left_vessel"
        if self.severe_contact_count >= self.config.severe_contact_limit:
            return "persistent_severe_contact"
        wrong_target = self.targets["right" if self.task == "left" else "left"]
        if float(np.linalg.norm(wrong_target - self.tip_pos)) < self.config.vessel.radius * 0.7:
            return "wrong_branch"
        return None

    def _reached_target(self) -> bool:
        target = self.targets[self.task]
        return float(np.linalg.norm(target - self.tip_pos)) < self.config.vessel.radius * 0.65

    def _tactile_state(self, point: np.ndarray) -> Dict:
        nearest = None
        best_dist = float("inf")
        best_projection = None
        for start, end, name in self.segments:
            projection, t = self._project_point_to_segment(point, start, end)
            dist = float(np.linalg.norm(point - projection))
            if dist < best_dist:
                nearest = name
                best_dist = dist
                best_projection = projection

        v = self.config.vessel
        distance_to_wall = v.radius - best_dist
        raw_strength = (v.safety_margin - distance_to_wall) / max(v.safety_margin, 1e-6)
        contact_strength = float(np.clip(raw_strength, 0.0, 1.0))
        contact_flag = int(contact_strength > 0.0)
        contact_direction = "none"
        if contact_flag and best_projection is not None:
            offset = point - best_projection
            if abs(float(offset[1])) >= abs(float(offset[0])):
                contact_direction = "upper" if offset[1] > 0 else "lower"
            else:
                contact_direction = "right" if offset[0] > 0 else "left"

        return {
            "distance_to_wall": float(distance_to_wall),
            "contact_flag": contact_flag,
            "contact_direction": contact_direction,
            "contact_strength": contact_strength,
            "nearest_branch": nearest,
        }

    @staticmethod
    def _project_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, float]:
        seg = end - start
        denom = float(np.dot(seg, seg))
        if denom <= 1e-8:
            return start.copy(), 0.0
        t = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
        return start + t * seg, t

    def render(self, show_target: bool = True) -> np.ndarray:
        size = self.config.render_size
        img = np.full((size, size, 3), 245, dtype=np.uint8)

        def to_px(p):
            p = np.asarray(p, dtype=np.float32)
            norm = (p - self.world_min) / (self.world_max - self.world_min)
            x = int(np.clip(norm[0] * (size - 1), 0, size - 1))
            y = int(np.clip((1.0 - norm[1]) * (size - 1), 0, size - 1))
            return x, y

        radius_px = max(2, int(self.config.vessel.radius / (self.world_max[0] - self.world_min[0]) * size))
        for start, end, _name in self.segments:
            cv2.line(img, to_px(start), to_px(end), (210, 210, 210), radius_px * 2, cv2.LINE_AA)
        for start, end, _name in self.segments:
            cv2.line(img, to_px(start), to_px(end), (95, 95, 95), 2, cv2.LINE_AA)

        if len(self.guidewire_points) > 1:
            pts = np.array([to_px(p) for p in self.guidewire_points], dtype=np.int32)
            cv2.polylines(img, [pts], False, (35, 105, 220), 3, cv2.LINE_AA)

        cv2.circle(img, to_px(self.tip_pos), 7, (0, 0, 220), -1, cv2.LINE_AA)
        if show_target:
            cv2.circle(img, to_px(self.targets[self.task]), 9, (40, 170, 70), -1, cv2.LINE_AA)

        tactile = self._tactile_state(self.tip_pos)
        if tactile["contact_flag"]:
            cv2.putText(
                img,
                f"contact: {tactile['contact_direction']} {tactile['contact_strength']:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 220),
                2,
                cv2.LINE_AA,
            )
        return img


class CenterlineExpert:
    """Rule-based expert used to generate initial imitation episodes."""

    def __init__(self, steer_gain: float = 0.75):
        self.steer_gain = steer_gain

    def act(self, obs: Dict, env: Guidewire2DEnv) -> Dict:
        tip = np.asarray(obs["tip_pos"], dtype=np.float32)
        target = np.asarray(obs["target_pos"], dtype=np.float32)
        desired = self._desired_direction(tip, target, env)
        current = np.asarray(obs["tip_dir"], dtype=np.float32)
        steer = (desired - current) * self.steer_gain

        if obs["contact_flag"]:
            correction = self._contact_recovery(obs["contact_direction"])
            steer += correction * obs["contact_strength"]

        return {
            "steer": steer.astype(float).tolist(),
            "advance": 1,
            "retreat": 0,
        }

    def _desired_direction(self, tip: np.ndarray, target: np.ndarray, env: Guidewire2DEnv) -> np.ndarray:
        split = env.centerline[1]
        if tip[0] < split[0] - 0.15:
            waypoint = split
        else:
            waypoint = target
        direction = waypoint - tip
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return direction / norm

    @staticmethod
    def _contact_recovery(direction: str) -> np.ndarray:
        recovery = {
            "upper": np.array([0.0, -0.18], dtype=np.float32),
            "lower": np.array([0.0, 0.18], dtype=np.float32),
            "left": np.array([0.18, 0.0], dtype=np.float32),
            "right": np.array([-0.18, 0.0], dtype=np.float32),
        }
        return recovery.get(direction, np.zeros(2, dtype=np.float32))
