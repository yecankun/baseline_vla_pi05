import json
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import trimesh
from gymnasium import spaces


TASK_INSTRUCTIONS = {
    "left": "Guide the wire into the left vessel branch while avoiding wall contact.",
    "right": "Guide the wire into the right vessel branch while avoiding wall contact.",
}


@dataclass
class MeshEnvConfig:
    mesh_path: str = "utils/interface/model/0422.stl"
    render_size: int = 512
    occupancy_size: int = 512
    max_steps: int = 260
    advance_step_px: float = 2.4
    max_steer: float = 0.22
    vessel_dilate_px: int = 7
    wall_margin_px: float = 14.0
    target_radius_px: float = 18.0
    severe_contact_limit: int = 16


class MeshGuidewireEnv(gym.Env):
    """
    Gymnasium environment using a projected STL vessel mask.

    This is still task-level simulation: the guidewire is a tip, heading, and
    polyline. The vessel geometry comes from the STL projected to the XY plane.
    Contact/tactile values come from an image-space distance transform.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[MeshEnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config or MeshEnvConfig()
        self.rng = np.random.default_rng(seed)
        self._load_vessel_projection()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(self.config.render_size, self.config.render_size, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
                "instruction_id": spaces.Discrete(2),
            }
        )

        self.task = "left"
        self.instruction = TASK_INSTRUCTIONS[self.task]
        self.step_count = 0
        self.severe_contact_count = 0
        self.tip = np.zeros(2, dtype=np.float32)
        self.heading = np.array([0.0, -1.0], dtype=np.float32)
        self.guidewire_points = []
        self.last_obs_dict = {}

    def _load_vessel_projection(self):
        cfg = self.config
        mesh = trimesh.load(cfg.mesh_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        xy = verts[:, [0, 1]]
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        pad = (hi - lo).max() * 0.04
        lo -= pad
        hi += pad
        self.world_min = lo
        self.world_max = hi

        pixels = self._world_to_px_batch(xy, cfg.occupancy_size)
        mask = np.zeros((cfg.occupancy_size, cfg.occupancy_size), dtype=np.uint8)
        mask[pixels[:, 1], pixels[:, 0]] = 255
        kernel_size = cfg.vessel_dilate_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        self.vessel_mask = filled
        self.dist_inside = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        self.dist_outside = cv2.distanceTransform(255 - filled, cv2.DIST_L2, 5)

        ys, xs = np.where(filled > 0)
        low_y = np.percentile(ys, 92)
        high_y = np.percentile(ys, 8)
        entry_candidates = np.column_stack([xs[ys >= low_y], ys[ys >= low_y]])
        target_candidates = np.column_stack([xs[ys <= high_y], ys[ys <= high_y]])
        self.entry_px = entry_candidates.mean(axis=0).astype(np.float32)
        if len(target_candidates) < 2:
            self.left_target_px = np.array([cfg.occupancy_size * 0.35, cfg.occupancy_size * 0.1], dtype=np.float32)
            self.right_target_px = np.array([cfg.occupancy_size * 0.65, cfg.occupancy_size * 0.1], dtype=np.float32)
        else:
            median_x = np.median(target_candidates[:, 0])
            left_group = target_candidates[target_candidates[:, 0] <= median_x]
            right_group = target_candidates[target_candidates[:, 0] > median_x]
            self.left_target_px = left_group.mean(axis=0).astype(np.float32)
            self.right_target_px = right_group.mean(axis=0).astype(np.float32)

        self.targets = {"left": self.left_target_px, "right": self.right_target_px}
        self.paths = {
            "left": self._plan_path(self.entry_px, self.left_target_px),
            "right": self._plan_path(self.entry_px, self.right_target_px),
        }

    def _plan_path(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        start_t = self._nearest_inside_tuple(start)
        goal_t = self._nearest_inside_tuple(goal)
        height, width = self.vessel_mask.shape
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        ]

        def heuristic(a):
            return ((a[0] - goal_t[0]) ** 2 + (a[1] - goal_t[1]) ** 2) ** 0.5

        frontier = [(heuristic(start_t), 0.0, start_t)]
        came_from = {start_t: None}
        cost_so_far = {start_t: 0.0}

        while frontier:
            _priority, current_cost, current = heapq.heappop(frontier)
            if current == goal_t:
                break
            if current_cost > cost_so_far[current] + 1e-6:
                continue
            cx, cy = current
            for dx, dy, step_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if self.vessel_mask[ny, nx] == 0:
                    continue
                wall_clearance = max(float(self.dist_inside[ny, nx]), 1.0)
                center_penalty = 4.0 / wall_clearance
                new_cost = current_cost + step_cost * (1.0 + center_penalty)
                nxt = (nx, ny)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heapq.heappush(frontier, (new_cost + heuristic(nxt), new_cost, nxt))
                    came_from[nxt] = current

        if goal_t not in came_from:
            return np.stack([np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)])

        points = []
        current = goal_t
        while current is not None:
            points.append(current)
            current = came_from[current]
        points.reverse()
        path = np.array(points, dtype=np.float32)
        return path

    def _nearest_inside_tuple(self, point: np.ndarray) -> Tuple[int, int]:
        x = int(np.clip(round(float(point[0])), 0, self.config.occupancy_size - 1))
        y = int(np.clip(round(float(point[1])), 0, self.config.occupancy_size - 1))
        if self.vessel_mask[y, x] > 0:
            return x, y
        ys, xs = np.where(self.vessel_mask > 0)
        idx = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        return int(xs[idx]), int(ys[idx])

    def _world_to_px_batch(self, xy: np.ndarray, size: int) -> np.ndarray:
        norm = (xy - self.world_min) / np.maximum(self.world_max - self.world_min, 1e-6)
        px = np.column_stack([norm[:, 0] * (size - 1), (1.0 - norm[:, 1]) * (size - 1)])
        return np.clip(px.round().astype(np.int32), 0, size - 1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.task = options.get("task", "left")
        if self.task not in TASK_INSTRUCTIONS:
            raise ValueError(f"Unknown task: {self.task}")
        self.instruction = TASK_INSTRUCTIONS[self.task]
        self.step_count = 0
        self.severe_contact_count = 0
        self.tip = self.entry_px.copy()
        self.tip += self.rng.normal(0.0, 2.0, size=2).astype(np.float32)
        target = self.targets[self.task]
        self.heading = target - self.tip
        self.heading = self.heading / max(float(np.linalg.norm(self.heading)), 1e-6)
        self.guidewire_points = [self.tip.copy()]
        obs_dict = self._obs_dict(done=False, success=False, failure_reason=None)
        self.last_obs_dict = obs_dict
        return self._gym_obs(obs_dict), self._info(obs_dict)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer = action[:2]
        steer_norm = float(np.linalg.norm(steer))
        if steer_norm > 1.0:
            steer = steer / steer_norm
        advance = float(action[2]) >= 0.5

        self.heading = self.heading + steer * self.config.max_steer
        self.heading = self.heading / max(float(np.linalg.norm(self.heading)), 1e-6)
        if advance:
            self.tip = self.tip + self.heading * self.config.advance_step_px

        self.step_count += 1
        self.guidewire_points.append(self.tip.copy())
        tactile = self._tactile(self.tip)
        if tactile["contact_strength"] > 0.85:
            self.severe_contact_count += 1
        else:
            self.severe_contact_count = 0

        success = self._success()
        failure_reason = self._failure_reason(tactile)
        terminated = success or failure_reason is not None
        truncated = self.step_count >= self.config.max_steps
        obs_dict = self._obs_dict(done=terminated or truncated, success=success, failure_reason=failure_reason)
        self.last_obs_dict = obs_dict
        reward = self._reward(obs_dict, failure_reason)
        return self._gym_obs(obs_dict), reward, terminated, truncated, self._info(obs_dict)

    def _gym_obs(self, obs_dict: Dict) -> Dict:
        target = np.asarray(obs_dict["target_px"], dtype=np.float32)
        tip = np.asarray(obs_dict["tip_px"], dtype=np.float32)
        state = np.array(
            [
                tip[0] / self.config.occupancy_size,
                tip[1] / self.config.occupancy_size,
                obs_dict["heading"][0],
                obs_dict["heading"][1],
                target[0] / self.config.occupancy_size,
                target[1] / self.config.occupancy_size,
                obs_dict["distance_to_wall_px"] / self.config.wall_margin_px,
                obs_dict["contact_strength"],
                float(obs_dict["contact_flag"]),
                float(self.step_count) / self.config.max_steps,
                float(self.task == "left"),
                float(self.task == "right"),
            ],
            dtype=np.float32,
        )
        return {
            "image": self.render(),
            "state": state,
            "instruction_id": 0 if self.task == "left" else 1,
        }

    def _obs_dict(self, done: bool, success: bool, failure_reason: Optional[str]) -> Dict:
        tactile = self._tactile(self.tip)
        target = self.targets[self.task]
        return {
            "instruction": self.instruction,
            "task": self.task,
            "step": self.step_count,
            "tip_px": self.tip.astype(float).tolist(),
            "heading": self.heading.astype(float).tolist(),
            "target_px": target.astype(float).tolist(),
            "distance_to_target_px": float(np.linalg.norm(target - self.tip)),
            **tactile,
            "done": done,
            "success": success,
            "failure_reason": failure_reason,
        }

    def _info(self, obs_dict: Dict) -> Dict:
        return {
            "obs_dict": obs_dict,
            "instruction": self.instruction,
            "success": obs_dict["success"],
            "failure_reason": obs_dict["failure_reason"],
        }

    def _tactile(self, pt: np.ndarray) -> Dict:
        x = int(np.clip(round(float(pt[0])), 0, self.config.occupancy_size - 1))
        y = int(np.clip(round(float(pt[1])), 0, self.config.occupancy_size - 1))
        inside = self.vessel_mask[y, x] > 0
        signed_dist = float(self.dist_inside[y, x] if inside else -self.dist_outside[y, x])
        strength = float(np.clip((self.config.wall_margin_px - signed_dist) / self.config.wall_margin_px, 0.0, 1.0))
        direction = "none"
        if strength > 0.0:
            direction = self._contact_direction(x, y)
        return {
            "inside_vessel": bool(inside),
            "distance_to_wall_px": signed_dist,
            "contact_flag": int(strength > 0.0),
            "contact_direction": direction,
            "contact_strength": strength,
        }

    def _contact_direction(self, x: int, y: int) -> str:
        gx = self._sample_dist(x + 1, y) - self._sample_dist(x - 1, y)
        gy = self._sample_dist(x, y + 1) - self._sample_dist(x, y - 1)
        if abs(gy) >= abs(gx):
            return "lower" if gy > 0 else "upper"
        return "right" if gx > 0 else "left"

    def _sample_dist(self, x: int, y: int) -> float:
        x = int(np.clip(x, 0, self.config.occupancy_size - 1))
        y = int(np.clip(y, 0, self.config.occupancy_size - 1))
        return float(self.dist_inside[y, x] - self.dist_outside[y, x])

    def _success(self) -> bool:
        return float(np.linalg.norm(self.targets[self.task] - self.tip)) <= self.config.target_radius_px

    def _failure_reason(self, tactile: Dict) -> Optional[str]:
        if not tactile["inside_vessel"] and tactile["distance_to_wall_px"] < -self.config.wall_margin_px:
            return "left_vessel"
        if self.severe_contact_count >= self.config.severe_contact_limit:
            return "persistent_severe_contact"
        wrong = self.targets["right" if self.task == "left" else "left"]
        if float(np.linalg.norm(wrong - self.tip)) <= self.config.target_radius_px:
            return "wrong_branch"
        return None

    def _reward(self, obs_dict: Dict, failure_reason: Optional[str]) -> float:
        reward = -0.01 * obs_dict["distance_to_target_px"] - 0.35 * obs_dict["contact_strength"]
        if obs_dict["success"]:
            reward += 50.0
        if failure_reason is not None:
            reward -= 50.0
        return float(reward)

    def render(self):
        size = self.config.render_size
        base = cv2.resize(self.vessel_mask, (size, size), interpolation=cv2.INTER_AREA)
        img = np.full((size, size, 3), 248, dtype=np.uint8)
        img[base > 0] = (220, 220, 220)

        scale = size / self.config.occupancy_size
        if len(self.guidewire_points) > 1:
            pts = np.array(self.guidewire_points, dtype=np.float32) * scale
            cv2.polylines(img, [pts.astype(np.int32)], False, (40, 120, 230), 3, cv2.LINE_AA)

        planned = self.paths.get(self.task)
        if planned is not None and len(planned) > 1:
            pts = planned.astype(np.float32) * scale
            cv2.polylines(img, [pts.astype(np.int32)], False, (120, 180, 120), 1, cv2.LINE_AA)

        target = (self.targets[self.task] * scale).astype(np.int32)
        tip = (self.tip * scale).astype(np.int32)
        cv2.circle(img, tuple(target), 8, (40, 170, 70), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(tip), 7, (0, 0, 220), -1, cv2.LINE_AA)
        return img

    def save_debug_config(self, path: str):
        data = {
            "entry_px": self.entry_px.astype(float).tolist(),
            "left_target_px": self.left_target_px.astype(float).tolist(),
            "right_target_px": self.right_target_px.astype(float).tolist(),
            "left_path_len": int(len(self.paths["left"])),
            "right_path_len": int(len(self.paths["right"])),
            "world_min": self.world_min.astype(float).tolist(),
            "world_max": self.world_max.astype(float).tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


class MeshCenterlineExpert:
    """Simple proportional expert that steers toward the selected mesh target."""

    def act(self, env: MeshGuidewireEnv) -> np.ndarray:
        target = self._lookahead_target(env)
        desired = target - env.tip
        desired = desired / max(float(np.linalg.norm(desired)), 1e-6)
        steer = desired - env.heading
        tactile = env._tactile(env.tip)
        if tactile["contact_flag"]:
            steer += self._recovery(tactile["contact_direction"]) * tactile["contact_strength"]
        return np.array([steer[0], steer[1], 1.0], dtype=np.float32)

    def _lookahead_target(self, env: MeshGuidewireEnv, lookahead: int = 18) -> np.ndarray:
        path = env.paths.get(env.task)
        if path is None or len(path) == 0:
            return env.targets[env.task]
        dists = np.linalg.norm(path - env.tip, axis=1)
        nearest_idx = int(np.argmin(dists))
        target_idx = min(nearest_idx + lookahead, len(path) - 1)
        return path[target_idx]

    @staticmethod
    def _recovery(direction: str) -> np.ndarray:
        values = {
            "upper": np.array([0.0, 0.8], dtype=np.float32),
            "lower": np.array([0.0, -0.8], dtype=np.float32),
            "left": np.array([0.8, 0.0], dtype=np.float32),
            "right": np.array([-0.8, 0.0], dtype=np.float32),
        }
        return values.get(direction, np.zeros(2, dtype=np.float32))
