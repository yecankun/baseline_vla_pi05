import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
import trimesh
from gymnasium import spaces
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from simulation.mesh_guidewire_env import MeshEnvConfig, MeshGuidewireEnv


TASK_INSTRUCTIONS = {
    "left": "Guide the wire through the 3D vessel into the left branch while avoiding contact.",
    "right": "Guide the wire through the 3D vessel into the right branch while avoiding contact.",
}


@dataclass
class Mesh3DEnvConfig:
    mesh_path: str = "utils/interface/model/0422.stl"
    route_config_path: str = ""
    render_size: int = 768
    max_steps: int = 280
    advance_step: float = 0.035
    max_steer: float = 0.18
    target_radius: float = 0.16
    wall_margin: float = 0.018
    severe_contact_limit: int = 18
    path_stride: int = 4
    min_radius: float = 0.026
    max_radius: float = 0.07
    wall_clearance: float = 0.006
    centerline_knn: int = 240
    centerline_smooth_window: int = 9
    centerline_slice_half_width: float = 0.035
    centerline_slice_radius_scale: float = 2.4
    centerline_min_slice_points: int = 24
    path_search_backtrack: int = 8
    path_search_ahead: int = 24
    boundary_penetration_limit: int = 12


class MeshGuidewire3DEnv(gym.Env):
    """
    3D task-level guidewire environment derived from the STL vessel.

    The vessel is approximated as a 3D tube around planned center paths. The
    paths are generated from the STL's XY projection, then lifted back to 3D by
    estimating local Z centers from nearby mesh vertices. This keeps MVP physics
    simple while making state, action, tactile normal, and rendering genuinely 3D.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[Mesh3DEnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config or Mesh3DEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.mesh = trimesh.load(self.config.mesh_path, force="mesh")
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.bounds = np.asarray(self.mesh.bounds, dtype=np.float32)
        self.route_config = self._load_route_config(self.config.route_config_path)
        self._build_paths_from_projection()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(self.config.render_size, self.config.render_size, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
                "instruction_id": spaces.Discrete(2),
            }
        )

        self.task = "left"
        self.instruction = TASK_INSTRUCTIONS[self.task]
        self.step_count = 0
        self.severe_contact_count = 0
        self.boundary_projection_count = 0
        self.last_boundary_projection = False
        self.tip = self.entry.copy()
        self.heading = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.guidewire_points = [self.tip.copy()]

    def _build_paths_from_projection(self):
        projection_cfg = MeshEnvConfig(mesh_path=self.config.mesh_path)
        projection_env = MeshGuidewireEnv(projection_cfg)
        self.projection_env = projection_env
        self.world_min_xy = projection_env.world_min
        self.world_max_xy = projection_env.world_max

        self._vertex_tree = cKDTree(self.vertices)
        default_entry = self._lift_path_to_3d(projection_env.entry_px[None, :])[0]
        default_targets = {
            "left": self._lift_path_to_3d(projection_env.left_target_px[None, :])[0],
            "right": self._lift_path_to_3d(projection_env.right_target_px[None, :])[0],
        }

        if self.route_config:
            entry_guess, route_points = self._route_points_from_config(default_entry, default_targets)
            self.paths = self._build_configured_surface_paths(entry_guess, route_points)
        else:
            entry_guess = default_entry
            self.paths = self._build_surface_geodesic_paths(entry_guess, default_targets)
        self.radii = {task: self._estimate_radii(path) for task, path in self.paths.items()}

        self.entry = 0.5 * (self.paths["left"][0] + self.paths["right"][0])
        self.targets = {
            "left": self.paths["left"][-1],
            "right": self.paths["right"][-1],
        }
        self.scene_min = self.vertices.min(axis=0)
        self.scene_max = self.vertices.max(axis=0)

    @staticmethod
    def _load_route_config(path: str) -> Dict:
        if not path:
            return {}
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Route config not found: {path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _route_points_from_config(self, default_entry: np.ndarray, default_targets: Dict[str, np.ndarray]):
        cfg = self.route_config
        entry = np.asarray(cfg.get("entry", default_entry), dtype=np.float32)
        route_points = {}
        shared = [np.asarray(p, dtype=np.float32) for p in cfg.get("shared_waypoints", [])]
        for task in ("left", "right"):
            full_key = f"{task}_full_waypoints"
            if full_key in cfg:
                side_waypoints = [np.asarray(p, dtype=np.float32) for p in cfg.get(full_key, [])]
                points = side_waypoints
            else:
                side_waypoints = [np.asarray(p, dtype=np.float32) for p in cfg.get(f"{task}_waypoints", [])]
                points = [*shared, *side_waypoints]
            target = cfg.get(f"{task}_target", default_targets[task])
            points.append(np.asarray(target, dtype=np.float32))
            route_points[task] = points
        return entry, route_points

    def _build_configured_surface_paths(self, entry_guess: np.ndarray, route_points: Dict[str, list]) -> Dict[str, np.ndarray]:
        paths = {}
        for task, points in route_points.items():
            full_path_parts = []
            start = entry_guess
            for point in points:
                segment = self._build_surface_geodesic_paths(start, {"segment": point})["segment"]
                if full_path_parts:
                    segment = segment[1:]
                full_path_parts.append(segment)
                start = point
            path = np.vstack(full_path_parts)
            paths[task] = self._downsample_path(path, spacing=0.045)
        return paths

    def _build_surface_geodesic_paths(self, entry_guess: np.ndarray, target_guesses: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        start_idx = int(self._vertex_tree.query(entry_guess)[1])
        target_indices = {task: int(self._vertex_tree.query(point)[1]) for task, point in target_guesses.items()}

        edges = self.mesh.edges_unique
        edge_lengths = np.linalg.norm(self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1)
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([edge_lengths, edge_lengths])
        graph = coo_matrix((data, (row, col)), shape=(len(self.vertices), len(self.vertices))).tocsr()
        _dist, predecessors = dijkstra(graph, directed=False, indices=start_idx, return_predecessors=True)

        paths = {}
        for task, target_idx in target_indices.items():
            vertex_indices = self._reconstruct_vertex_path(predecessors, start_idx, target_idx)
            path = self.vertices[vertex_indices]
            path = self._downsample_path(path, spacing=0.045)
            path = self._center_surface_path(path)
            paths[task] = path.astype(np.float32)
        return paths

    @staticmethod
    def _reconstruct_vertex_path(predecessors: np.ndarray, start_idx: int, target_idx: int):
        path = [target_idx]
        current = target_idx
        while current != start_idx:
            current = int(predecessors[current])
            if current < 0:
                raise RuntimeError("Could not reconstruct a connected surface path on the STL mesh")
            path.append(current)
        path.reverse()
        return np.asarray(path, dtype=np.int64)

    @staticmethod
    def _downsample_path(path: np.ndarray, spacing: float) -> np.ndarray:
        if len(path) <= 2:
            return path
        kept = [path[0]]
        accum = 0.0
        last = path[0]
        for point in path[1:]:
            accum += float(np.linalg.norm(point - last))
            last = point
            if accum >= spacing:
                kept.append(point)
                accum = 0.0
        if not np.allclose(kept[-1], path[-1]):
            kept.append(path[-1])
        return np.asarray(kept, dtype=np.float32)

    def _center_surface_path(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path
        centers = []
        for idx, point in enumerate(path):
            tangent = self._path_tangent(path, idx)
            centers.append(self._estimate_cross_section_center(point, tangent))
        centered = np.asarray(centers, dtype=np.float32)
        return self._smooth_path(centered, self.config.centerline_smooth_window)

    def _estimate_cross_section_center(self, point: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
        delta = self.vertices - point
        axial = delta @ tangent
        radial_vec = delta - axial[:, None] * tangent
        radial = np.linalg.norm(radial_vec, axis=1)

        half_width = max(float(self.config.centerline_slice_half_width), 1e-6)
        radius_limit = max(float(self.config.max_radius) * float(self.config.centerline_slice_radius_scale), 1e-6)
        mask = (np.abs(axial) <= half_width) & (radial <= radius_limit)
        local = self.vertices[mask]

        if len(local) < int(self.config.centerline_min_slice_points):
            k = min(self.config.centerline_knn, len(self.vertices))
            _dist, idx = self._vertex_tree.query(point, k=k)
            local = self.vertices[np.atleast_1d(idx)]
            return np.median(local, axis=0).astype(np.float32)

        weights = np.exp(-0.5 * (axial[mask] / half_width) ** 2).astype(np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        return (local * weights[:, None]).sum(axis=0).astype(np.float32)

    @staticmethod
    def _smooth_path(path: np.ndarray, window: int) -> np.ndarray:
        if len(path) < 3 or window <= 1:
            return path
        if window % 2 == 0:
            window += 1
        pad = window // 2
        kernel = np.ones(window, dtype=np.float32) / float(window)
        padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.column_stack([
            np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(path.shape[1])
        ]).astype(np.float32)
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    def _px_to_world_xy(self, px: np.ndarray) -> np.ndarray:
        size = self.projection_env.config.occupancy_size
        norm_x = px[:, 0] / max(size - 1, 1)
        norm_y = 1.0 - px[:, 1] / max(size - 1, 1)
        x = self.world_min_xy[0] + norm_x * (self.world_max_xy[0] - self.world_min_xy[0])
        y = self.world_min_xy[1] + norm_y * (self.world_max_xy[1] - self.world_min_xy[1])
        return np.column_stack([x, y]).astype(np.float32)

    def _lift_path_to_3d(self, path_px: np.ndarray) -> np.ndarray:
        xy = self._px_to_world_xy(path_px)
        verts_xy = self.vertices[:, [0, 1]]
        z_values = []
        for point_xy in xy:
            delta = verts_xy - point_xy
            dist2 = np.sum(delta * delta, axis=1)
            nearest_idx = np.argpartition(dist2, kth=min(180, len(dist2) - 1))[:180]
            local = self.vertices[nearest_idx]
            z_values.append(float(np.median(local[:, 2])))
        z = np.asarray(z_values, dtype=np.float32)
        if len(z) > 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            z = np.convolve(np.pad(z, (3, 3), mode="edge"), kernel, mode="valid")
        return np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float32)

    def _estimate_radii(self, path_xyz: np.ndarray) -> np.ndarray:
        radii = []
        for idx, center in enumerate(path_xyz):
            # Use nearest local surface samples. Cross-section slicing is too
            # easily polluted by neighboring branches in this compact STL.
            distances, _idx = self._vertex_tree.query(center, k=min(260, len(self.vertices)))
            radius = float(np.percentile(np.atleast_1d(distances), 65))
            radius = float(np.clip(radius, self.config.min_radius, self.config.max_radius))
            radii.append(radius)
        radii = np.asarray(radii, dtype=np.float32)
        if len(radii) > 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            radii = np.convolve(radii, kernel, mode="same")
            radii[0] = radii[1]
            radii[-1] = radii[-2]
        return radii

    @staticmethod
    def _path_tangent(path: np.ndarray, idx: int) -> np.ndarray:
        if len(path) < 2:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if idx <= 0:
            tangent = path[1] - path[0]
        elif idx >= len(path) - 1:
            tangent = path[-1] - path[-2]
        else:
            tangent = path[idx + 1] - path[idx - 1]
        return tangent / max(float(np.linalg.norm(tangent)), 1e-6)

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
        self.boundary_projection_count = 0
        self.last_boundary_projection = False
        self.tip = self.entry.copy()
        self.tip += self.rng.normal(0.0, 0.012, size=3).astype(np.float32)
        path = self.paths[self.task]
        initial_target = path[min(5, len(path) - 1)]
        self.heading = initial_target - self.tip
        self.heading = self.heading / max(float(np.linalg.norm(self.heading)), 1e-6)
        self.path_progress_index = 0
        self.path_progress_float = 0.0
        self.guidewire_points = [self.tip.copy()]
        obs = self._obs_dict(done=False, success=False, failure_reason=None)
        return self._gym_obs(obs), self._info(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer = action[:3]
        steer_norm = float(np.linalg.norm(steer))
        if steer_norm > 1.0:
            steer = steer / steer_norm
        advance = float(action[3]) >= 0.5

        self.heading = self.heading + steer * self.config.max_steer
        self.heading = self.heading / max(float(np.linalg.norm(self.heading)), 1e-6)
        prev_tip = self.tip.copy()
        if advance:
            candidate = self.tip + self.heading * self.config.advance_step
            self.tip = self._constrain_to_vessel(candidate, update_progress=True)
            actual_delta = self.tip - prev_tip
            if float(np.linalg.norm(actual_delta)) > 1e-6:
                self.heading = actual_delta / float(np.linalg.norm(actual_delta))
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
        obs = self._obs_dict(done=terminated or truncated, success=success, failure_reason=failure_reason)
        reward = self._reward(obs, failure_reason)
        return self._gym_obs(obs), reward, terminated, truncated, self._info(obs)

    def _gym_obs(self, obs: Dict) -> Dict:
        scene_span = np.maximum(self.scene_max - self.scene_min, 1e-6)
        tip = (np.asarray(obs["tip_pos"], dtype=np.float32) - self.scene_min) / scene_span
        target = (np.asarray(obs["target_pos"], dtype=np.float32) - self.scene_min) / scene_span
        state = np.array(
            [
                *tip.tolist(),
                *obs["heading"],
                *target.tolist(),
                *obs["contact_normal"],
                obs["distance_to_wall"] / max(self.config.wall_margin, 1e-6),
                obs["contact_strength"],
                float(obs["contact_flag"]),
                float(self.step_count) / self.config.max_steps,
                float(self.task == "left"),
                float(self.task == "right"),
            ],
            dtype=np.float32,
        )
        return {"image": self.render(), "state": state, "instruction_id": 0 if self.task == "left" else 1}

    def _obs_dict(self, done: bool, success: bool, failure_reason: Optional[str]) -> Dict:
        tactile = self._tactile(self.tip)
        target = self.targets[self.task]
        return {
            "instruction": self.instruction,
            "task": self.task,
            "step": self.step_count,
            "tip_pos": self.tip.astype(float).tolist(),
            "heading": self.heading.astype(float).tolist(),
            "target_pos": target.astype(float).tolist(),
            "distance_to_target": float(np.linalg.norm(target - self.tip)),
            "path_progress": float(getattr(self, "path_progress_float", 0.0)),
            "boundary_projection_count": int(self.boundary_projection_count),
            "last_boundary_projection": bool(self.last_boundary_projection),
            **tactile,
            "done": done,
            "success": success,
            "failure_reason": failure_reason,
        }

    def _info(self, obs: Dict) -> Dict:
        return {"obs_dict": obs, "instruction": self.instruction, "success": obs["success"], "failure_reason": obs["failure_reason"]}

    def _nearest_path_state(self, point: np.ndarray, update_progress: bool = False):
        path = self.paths[self.task]
        radii = self.radii[self.task]
        if len(path) <= 1:
            return 0, path[0], float(radii[0])

        current = int(getattr(self, "path_progress_index", 0))
        start = max(0, current - self.config.path_search_backtrack)
        end = min(len(path) - 1, current + self.config.path_search_ahead)
        if end <= start:
            start = max(0, min(current, len(path) - 2))
            end = min(len(path) - 1, start + 1)

        best_dist = np.inf
        best_center = path[start]
        best_radius = float(radii[start])
        best_progress = float(start)
        for seg_idx in range(start, end):
            a = path[seg_idx]
            b = path[seg_idx + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
            center = a + ab * t
            dist = float(np.linalg.norm(point - center))
            if dist < best_dist:
                best_dist = dist
                best_center = center
                best_radius = float((1.0 - t) * radii[seg_idx] + t * radii[seg_idx + 1])
                best_progress = float(seg_idx) + t

        if update_progress:
            self.path_progress_float = max(float(getattr(self, "path_progress_float", 0.0)), best_progress)
            self.path_progress_index = int(np.floor(self.path_progress_float))

        return int(np.floor(best_progress)), best_center.astype(np.float32), best_radius

    def _constrain_to_vessel(self, point: np.ndarray, update_progress: bool = False) -> np.ndarray:
        _idx, center, radius = self._nearest_path_state(point, update_progress=update_progress)
        offset = point - center
        dist = float(np.linalg.norm(offset))
        max_dist = max(radius - self.config.wall_clearance, radius * 0.62)
        self.last_boundary_projection = False
        if dist <= max_dist:
            return point
        normal = offset / max(dist, 1e-6)
        self.boundary_projection_count += 1
        self.last_boundary_projection = True
        return center + normal * max_dist

    def _tactile(self, point: np.ndarray) -> Dict:
        idx, center, radius = self._nearest_path_state(point)
        offset = point - center
        radial_dist = float(np.linalg.norm(offset))
        normal = offset / max(radial_dist, 1e-6)
        distance_to_wall = radius - radial_dist
        strength = float(np.clip((self.config.wall_margin - distance_to_wall) / self.config.wall_margin, 0.0, 1.0))
        direction = self._contact_direction(normal) if strength > 0.0 else "none"
        return {
            "nearest_path_index": idx,
            "nearest_center": center.astype(float).tolist(),
            "local_radius": radius,
            "inside_vessel": bool(distance_to_wall >= 0.0),
            "distance_to_wall": float(distance_to_wall),
            "contact_flag": int(strength > 0.0),
            "contact_direction": direction,
            "contact_normal": normal.astype(float).tolist(),
            "contact_strength": strength,
            "last_boundary_projection": bool(self.last_boundary_projection),
        }

    @staticmethod
    def _contact_direction(normal: np.ndarray) -> str:
        axis = int(np.argmax(np.abs(normal)))
        if axis == 0:
            return "right" if normal[0] > 0 else "left"
        if axis == 1:
            return "forward" if normal[1] > 0 else "backward"
        return "upper" if normal[2] > 0 else "lower"

    def _success(self) -> bool:
        return float(np.linalg.norm(self.targets[self.task] - self.tip)) <= self.config.target_radius

    def _failure_reason(self, tactile: Dict) -> Optional[str]:
        if tactile["distance_to_wall"] < -self.config.wall_margin * 1.8:
            return "left_vessel"
        if self.boundary_projection_count >= self.config.boundary_penetration_limit:
            return "repeated_boundary_projection"
        if self.severe_contact_count >= self.config.severe_contact_limit:
            return "persistent_severe_contact"
        wrong = self.targets["right" if self.task == "left" else "left"]
        if float(np.linalg.norm(wrong - self.tip)) <= self.config.target_radius:
            return "wrong_branch"
        return None

    def _reward(self, obs: Dict, failure_reason: Optional[str]) -> float:
        reward = -obs["distance_to_target"] - 0.8 * obs["contact_strength"]
        if obs["success"]:
            reward += 30.0
        if failure_reason is not None:
            reward -= 30.0
        return float(reward)

    def render(self):
        size = self.config.render_size
        panel_w = size // 3
        img = np.full((size, size, 3), 248, dtype=np.uint8)
        panels = [
            (0, panel_w, (0, 1), "XY"),
            (panel_w, panel_w * 2, (0, 2), "XZ"),
            (panel_w * 2, size, (1, 2), "YZ"),
        ]
        for x0, x1, axes, label in panels:
            self._draw_panel(img[:, x0:x1], axes, label)
        return img

    def render_camera_pair(self, size: int = 512) -> Dict[str, np.ndarray]:
        """Render side/top camera-like images for learning data."""
        return {
            "side": self._render_projected_camera(size=size, axes=(1, 2)),
            "top": self._render_projected_camera(size=size, axes=(1, 0)),
        }

    def _render_projected_camera(self, size: int, axes) -> np.ndarray:
        img = np.full((size, size, 3), (235, 238, 232), dtype=np.uint8)
        verts_2d = self.vertices[:, axes]
        lo = verts_2d.min(axis=0)
        hi = verts_2d.max(axis=0)
        pad = (hi - lo).max() * 0.06
        lo -= pad
        hi += pad

        def to_px(points):
            pts = np.asarray(points, dtype=np.float32)[:, axes]
            norm = (pts - lo) / np.maximum(hi - lo, 1e-6)
            px = np.column_stack([norm[:, 0] * (size - 1), (1.0 - norm[:, 1]) * (size - 1)])
            return np.clip(px, 0, [size - 1, size - 1]).astype(np.int32)

        sample_step = max(1, len(self.vertices) // 65000)
        vessel_pts = to_px(self.vertices[::sample_step])
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[vessel_pts[:, 1], vessel_pts[:, 0]] = 255
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        vessel_layer = np.full_like(img, (155, 185, 205))
        img = np.where(mask[:, :, None] > 0, (img * 0.48 + vessel_layer * 0.52).astype(np.uint8), img)

        if len(self.guidewire_points) > 1:
            wire_pts = to_px(np.asarray(self.guidewire_points, dtype=np.float32))
            cv2.polylines(img, [wire_pts], False, (35, 125, 230), 3, cv2.LINE_AA)
        tip = to_px(self.tip[None, :])[0]
        cv2.circle(img, tuple(tip), 6, (0, 0, 220), -1, cv2.LINE_AA)
        return img

    def render_perspective(self, yaw_deg: float = -35.0, pitch_deg: float = 18.0, zoom: float = 1.0):
        size = self.config.render_size
        img = np.full((size, size, 3), 248, dtype=np.uint8)
        center = 0.5 * (self.scene_min + self.scene_max)
        span = float(np.max(self.scene_max - self.scene_min))
        rot = self._view_rotation(yaw_deg, pitch_deg)

        def project(points):
            pts = np.asarray(points, dtype=np.float32) - center
            pts = pts @ rot.T
            scale = size * 0.72 * zoom / max(span, 1e-6)
            px = np.column_stack([pts[:, 0] * scale + size * 0.5, -pts[:, 2] * scale + size * 0.55])
            depth = pts[:, 1]
            return px, depth

        sample_step = max(1, len(self.vertices) // 36000)
        v = self.vertices[::sample_step]
        px, depth = project(v)
        valid = (px[:, 0] >= 0) & (px[:, 0] < size) & (px[:, 1] >= 0) & (px[:, 1] < size)
        order = np.argsort(depth[valid])
        pix = px[valid][order].astype(np.int32)
        if len(pix):
            shade = np.linspace(205, 230, len(pix)).astype(np.uint8)
            img[pix[:, 1], pix[:, 0]] = np.column_stack([shade, shade, shade])
            img = cv2.dilate(img, np.ones((2, 2), dtype=np.uint8), iterations=1)

        self._draw_projected_polyline(img, project, self.paths[self.task], (95, 175, 95), 2)
        if len(self.guidewire_points) > 1:
            self._draw_projected_polyline(img, project, np.asarray(self.guidewire_points, dtype=np.float32), (35, 115, 230), 4)

        target_px, _ = project(self.targets[self.task][None, :])
        tip_px, _ = project(self.tip[None, :])
        cv2.circle(img, tuple(target_px[0].astype(np.int32)), 9, (35, 165, 65), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(tip_px[0].astype(np.int32)), 8, (0, 0, 220), -1, cv2.LINE_AA)

        tactile = self._tactile(self.tip)
        text = f"{self.task}  step {self.step_count}  contact {tactile['contact_direction']} {tactile['contact_strength']:.2f}"
        cv2.putText(img, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (55, 55, 55), 2, cv2.LINE_AA)
        return img

    @staticmethod
    def _view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rot_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rot_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        return rot_pitch @ rot_yaw

    @staticmethod
    def _draw_projected_polyline(img: np.ndarray, project_fn, points: np.ndarray, color, thickness: int):
        px, _depth = project_fn(points)
        pts = px.astype(np.int32)
        h, w = img.shape[:2]
        if len(pts) < 2:
            return
        clipped = []
        for pt in pts:
            clipped.append([int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))])
        cv2.polylines(img, [np.asarray(clipped, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def _draw_panel(self, panel: np.ndarray, axes, label: str):
        h, w = panel.shape[:2]
        verts_2d = self.vertices[:, axes]
        lo = verts_2d.min(axis=0)
        hi = verts_2d.max(axis=0)
        pad = (hi - lo).max() * 0.06
        lo -= pad
        hi += pad

        def to_px(points):
            pts = np.asarray(points, dtype=np.float32)[:, axes]
            norm = (pts - lo) / np.maximum(hi - lo, 1e-6)
            px = np.column_stack([norm[:, 0] * (w - 1), (1.0 - norm[:, 1]) * (h - 1)])
            return np.clip(px, 0, [w - 1, h - 1]).astype(np.int32)

        sample_step = max(1, len(self.vertices) // 18000)
        pts = to_px(self.vertices[::sample_step])
        panel[pts[:, 1], pts[:, 0]] = (215, 215, 215)
        panel[:] = cv2.dilate(panel, np.ones((2, 2), dtype=np.uint8), iterations=1)

        path_pts = to_px(self.paths[self.task])
        cv2.polylines(panel, [path_pts], False, (110, 180, 110), 1, cv2.LINE_AA)
        if len(self.guidewire_points) > 1:
            wire_pts = to_px(np.asarray(self.guidewire_points, dtype=np.float32))
            cv2.polylines(panel, [wire_pts], False, (40, 120, 230), 2, cv2.LINE_AA)

        target = to_px(self.targets[self.task][None, :])[0]
        tip = to_px(self.tip[None, :])[0]
        cv2.circle(panel, tuple(target), 6, (40, 170, 70), -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(tip), 5, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2, cv2.LINE_AA)

    def save_debug_config(self, path: str):
        data = {
            "entry": self.entry.astype(float).tolist(),
            "left_target": self.targets["left"].astype(float).tolist(),
            "right_target": self.targets["right"].astype(float).tolist(),
            "shared_waypoints": [],
            "left_waypoints": [],
            "right_waypoints": [],
            "left_path_len": int(len(self.paths["left"])),
            "right_path_len": int(len(self.paths["right"])),
            "bounds": self.bounds.astype(float).tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


class Mesh3DPathExpert:
    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.0):
        self.rng = np.random.default_rng(rng_seed)
        self.steer_noise = steer_noise

    def act(self, env: MeshGuidewire3DEnv) -> np.ndarray:
        target = self._lookahead_target(env)
        desired = target - env.tip
        desired = desired / max(float(np.linalg.norm(desired)), 1e-6)
        steer = desired - env.heading
        tactile = env._tactile(env.tip)
        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            steer -= normal * tactile["contact_strength"] * 0.9
        if self.steer_noise > 0:
            steer += self.rng.normal(0.0, self.steer_noise, size=3).astype(np.float32)
        return np.array([steer[0], steer[1], steer[2], 1.0], dtype=np.float32)

    def _lookahead_target(self, env: MeshGuidewire3DEnv, lookahead: int = 5) -> np.ndarray:
        path = env.paths[env.task]
        dists = np.linalg.norm(path - env.tip, axis=1)
        idx = int(np.argmin(dists))
        idx = max(idx, getattr(env, "path_progress_index", 0))
        env.path_progress_index = idx
        return path[min(idx + lookahead, len(path) - 1)]


class Mesh3DRecoveryExpert(Mesh3DPathExpert):
    """
    Expert that intentionally deviates toward a vessel wall and then recovers.

    This produces contact-rich trajectories for tactile/VLA training instead of
    only clean path-following demonstrations.
    """

    def __init__(
        self,
        rng_seed: int = 0,
        steer_noise: float = 0.025,
        push_start: int = 36,
        push_end: int = 96,
        push_strength: float = 0.85,
        side: str = "auto",
    ):
        super().__init__(rng_seed=rng_seed, steer_noise=steer_noise)
        self.push_start = push_start
        self.push_end = push_end
        self.push_strength = push_strength
        self.side = side

    def act(self, env: MeshGuidewire3DEnv) -> np.ndarray:
        target = self._lookahead_target(env, lookahead=5)
        desired = target - env.tip
        desired = desired / max(float(np.linalg.norm(desired)), 1e-6)
        steer = desired - env.heading
        tactile = env._tactile(env.tip)

        if self.push_start <= env.step_count <= self.push_end and tactile["contact_strength"] < 0.92:
            steer += self._push_direction(env) * self.push_strength

        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            steer -= normal * (1.25 + tactile["contact_strength"])

        if self.steer_noise > 0:
            steer += self.rng.normal(0.0, self.steer_noise, size=3).astype(np.float32)
        return np.array([steer[0], steer[1], steer[2], 1.0], dtype=np.float32)

    def _push_direction(self, env: MeshGuidewire3DEnv) -> np.ndarray:
        tangent = env.heading / max(float(np.linalg.norm(env.heading)), 1e-6)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        lateral = np.cross(tangent, up)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-6)
        vertical = up

        if self.side == "left":
            direction = lateral
        elif self.side == "right":
            direction = -lateral
        elif self.side == "upper":
            direction = vertical
        elif self.side == "lower":
            direction = -vertical
        else:
            direction = lateral if env.task == "left" else -lateral
        return direction.astype(np.float32)
