from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, MeshGuidewire3DEnv


@dataclass
class DualArmMesh3DEnvConfig(Mesh3DEnvConfig):
    piper_step_scale: float = 0.78
    piper_retreat_scale: float = 0.55
    elirobot_action_scale: float = 0.018
    magnet_decay: float = 0.84
    magnet_coupling: float = 0.62
    magnet_max_offset: float = 0.055
    dual_arm_contact_recovery: float = 0.9
    dual_arm_lookahead: int = 6
    dual_arm_boundary_projection_window: int = 16


class DualArmGuidewire3DEnv(MeshGuidewire3DEnv):
    """
    Dual-arm abstraction over the STL-derived 3D guidewire environment.

    Piper controls insertion / retreat along the path.
    Elirobot controls a magnetic traction pose that biases the guidewire laterally
    inside the local vessel frame.
    """

    def __init__(self, config: Optional[DualArmMesh3DEnvConfig] = None, seed: Optional[int] = None):
        self.dual_config = config or DualArmMesh3DEnvConfig()
        super().__init__(self.dual_config, seed=seed)
        self.action_space = spaces.Dict(
            {
                "piper": spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32),
                "elirobot_delta": spaces.Box(low=np.array([-1.0, -1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32),
            }
        )
        state_dim = 30
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(self.config.render_size, self.config.render_size, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
                "instruction_id": spaces.Discrete(2),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.piper_step_count = 0
        self.piper_insertion_length = 0.0
        self.elirobot_pose = getattr(self, "elirobot_pose", self.entry.copy())
        self.lateral_offset = getattr(self, "lateral_offset", np.zeros(2, dtype=np.float32))
        center, tangent, n1, n2, radius = self._local_path_frame(0.0)
        self.local_path_center = center
        self.local_path_tangent = tangent
        self.local_frame_x = n1
        self.local_frame_y = n2
        self.local_radius = radius
        obs, info = super().reset(seed=seed, options=options)
        self.path_progress_float = float(getattr(self, "path_progress_float", 0.0))
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        self.local_path_center = center
        self.local_path_tangent = tangent
        self.local_frame_x = n1
        self.local_frame_y = n2
        self.local_radius = radius
        offset = self.tip - center
        self.lateral_offset = np.array([float(np.dot(offset, n1)), float(np.dot(offset, n2))], dtype=np.float32)
        neutral_local = -0.45 * self.lateral_offset
        neutral_norm = float(np.linalg.norm(neutral_local))
        if neutral_norm > self.dual_config.magnet_max_offset and neutral_norm > 1e-6:
            neutral_local = neutral_local / neutral_norm * self.dual_config.magnet_max_offset
        self.elirobot_pose = center + n1 * neutral_local[0] + n2 * neutral_local[1]
        self.last_boundary_projection = False
        self.boundary_projection_count = 0
        self.boundary_projection_window = 0
        self._sync_robot_state(center, tangent, n1, n2, radius)
        obs = self._obs_dict(done=False, success=False, failure_reason=None)
        return self._gym_obs(obs), self._info(obs)

    def step(self, action):
        piper_cmd, elirobot_delta = self._parse_action(action)
        piper_cmd = float(np.clip(piper_cmd, -1.0, 1.0))
        elirobot_delta = np.asarray(elirobot_delta, dtype=np.float32)
        if float(np.linalg.norm(elirobot_delta)) > 1.0:
            elirobot_delta = elirobot_delta / max(float(np.linalg.norm(elirobot_delta)), 1e-6)

        self.piper_step_count += 1 if abs(piper_cmd) > 0.2 else 0
        self.piper_insertion_length = max(
            0.0,
            self.piper_insertion_length + max(piper_cmd, 0.0) * self.config.advance_step,
        )

        self.elirobot_pose = self.elirobot_pose + elirobot_delta * self.dual_config.elirobot_action_scale

        progress_delta = piper_cmd * self.dual_config.piper_step_scale
        if piper_cmd < 0:
            progress_delta *= self.dual_config.piper_retreat_scale
        self.path_progress_float = float(
            np.clip(
                self.path_progress_float + progress_delta,
                0.0,
                len(self.paths[self.task]) - 1,
            )
        )
        self.path_progress_index = int(np.floor(self.path_progress_float))

        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        self.local_path_center = center
        self.local_path_tangent = tangent
        self.local_frame_x = n1
        self.local_frame_y = n2
        self.local_radius = radius
        desired_lateral = self._project_local(self.elirobot_pose - center, n1, n2)
        self.lateral_offset = (
            self.dual_config.magnet_decay * self.lateral_offset
            + self.dual_config.magnet_coupling * desired_lateral
        )
        self.lateral_offset = self._clip_local_offset(self.lateral_offset, radius)

        prev_tip = self.tip.copy()
        self.tip = center + n1 * self.lateral_offset[0] + n2 * self.lateral_offset[1]
        actual_delta = self.tip - prev_tip
        if float(np.linalg.norm(actual_delta)) > 1e-6:
            self.heading = actual_delta / float(np.linalg.norm(actual_delta))
        else:
            self.heading = tangent

        self.step_count += 1
        self.guidewire_points.append(self.tip.copy())
        self._sync_robot_state(center, tangent, n1, n2, radius)

        tactile = self._tactile_from_local_frame(center, n1, n2, radius)
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

    def _sync_robot_state(self, center: np.ndarray, tangent: np.ndarray, n1: np.ndarray, n2: np.ndarray, radius: float):
        self.robot_state = {
            "piper_step": int(self.piper_step_count),
            "piper_insertion_length": float(self.piper_insertion_length),
            "elirobot_pose": self.elirobot_pose.astype(float).tolist(),
            "lateral_offset": self.lateral_offset.astype(float).tolist(),
            "path_center": center.astype(float).tolist(),
            "path_tangent": tangent.astype(float).tolist(),
            "frame_x": n1.astype(float).tolist(),
            "frame_y": n2.astype(float).tolist(),
            "local_radius": float(radius),
        }

    def set_path_progress(self, progress: float):
        self.path_progress_float = float(np.clip(progress, 0.0, len(self.paths[self.task]) - 1))
        self.path_progress_index = int(np.floor(self.path_progress_float))
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        self.local_path_center = center
        self.local_path_tangent = tangent
        self.local_frame_x = n1
        self.local_frame_y = n2
        self.local_radius = radius
        self.lateral_offset = np.zeros(2, dtype=np.float32)
        self.tip = center.copy()
        self.heading = tangent.copy()
        self.elirobot_pose = center.copy()
        self.guidewire_points = [self.tip.copy()]
        self.boundary_projection_window = 0
        self._sync_robot_state(center, tangent, n1, n2, radius)

    def _local_path_frame(self, progress: float):
        path = self.paths[self.task]
        radii = self.radii[self.task]
        idx = int(np.clip(np.floor(progress), 0, len(path) - 1))
        frac = float(progress - idx)
        if idx >= len(path) - 1:
            center = path[-1]
            tangent = path[-1] - path[-2]
            radius = float(radii[-1])
        else:
            center = (1.0 - frac) * path[idx] + frac * path[idx + 1]
            tangent = path[idx + 1] - path[idx]
            radius = float((1.0 - frac) * radii[idx] + frac * radii[idx + 1])
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(ref, tangent))) > 0.8:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n1 = np.cross(tangent, ref)
        if np.linalg.norm(n1) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            n1 = np.cross(tangent, ref)
        n1 = n1 / max(float(np.linalg.norm(n1)), 1e-6)
        n2 = np.cross(tangent, n1)
        n2 = n2 / max(float(np.linalg.norm(n2)), 1e-6)
        return center.astype(np.float32), tangent.astype(np.float32), n1.astype(np.float32), n2.astype(np.float32), radius

    @staticmethod
    def _project_local(vector: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
        return np.array([float(np.dot(vector, n1)), float(np.dot(vector, n2))], dtype=np.float32)

    def _clip_local_offset(self, offset: np.ndarray, radius: float) -> np.ndarray:
        limit = max(radius - self.config.wall_clearance, 0.0)
        norm = float(np.linalg.norm(offset))
        if norm <= limit:
            self.last_boundary_projection = False
            self.boundary_projection_window = 0
            return offset.astype(np.float32)
        self.boundary_projection_count += 1
        self.boundary_projection_window = int(getattr(self, "boundary_projection_window", 0)) + 1
        self.last_boundary_projection = True
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (offset / norm * limit).astype(np.float32)

    def _failure_reason(self, tactile: Dict) -> Optional[str]:
        if tactile["distance_to_wall"] < -self.config.wall_margin * 1.8:
            return "left_vessel"
        if getattr(self, "boundary_projection_window", 0) >= self.dual_config.dual_arm_boundary_projection_window:
            return "repeated_boundary_projection"
        if self.severe_contact_count >= self.config.severe_contact_limit:
            return "persistent_severe_contact"
        wrong = self.targets["right" if self.task == "left" else "left"]
        if float(np.linalg.norm(wrong - self.tip)) <= self.config.target_radius:
            return "wrong_branch"
        return None

    def _tactile_from_local_frame(self, center: np.ndarray, n1: np.ndarray, n2: np.ndarray, radius: float) -> Dict:
        offset = self.tip - center
        local = self._project_local(offset, n1, n2)
        radial_dist = float(np.linalg.norm(local))
        distance_to_wall = radius - radial_dist
        strength = float(np.clip((self.config.wall_margin - distance_to_wall) / self.config.wall_margin, 0.0, 1.0))
        normal = np.zeros(3, dtype=np.float32)
        if radial_dist > 1e-6:
            normal = (n1 * local[0] + n2 * local[1]) / radial_dist
        direction = self._contact_direction(normal) if strength > 0.0 else "none"
        return {
            "nearest_path_index": int(self.path_progress_index),
            "nearest_center": center.astype(float).tolist(),
            "local_radius": float(radius),
            "inside_vessel": bool(distance_to_wall >= 0.0),
            "distance_to_wall": float(distance_to_wall),
            "contact_flag": int(strength > 0.0),
            "contact_direction": direction,
            "contact_normal": normal.astype(float).tolist(),
            "contact_strength": strength,
            "last_boundary_projection": bool(self.last_boundary_projection),
        }

    def _obs_dict(self, done: bool, success: bool, failure_reason: Optional[str]) -> Dict:
        tactile = self._tactile_from_local_frame(
            self.local_path_center,
            self.local_frame_x,
            self.local_frame_y,
            self.local_radius,
        )
        target = self.targets[self.task]
        return {
            "instruction": self.instruction,
            "task": self.task,
            "step": self.step_count,
            "tip_pos": self.tip.astype(float).tolist(),
            "heading": self.heading.astype(float).tolist(),
            "target_pos": target.astype(float).tolist(),
            "distance_to_target": float(np.linalg.norm(target - self.tip)),
            "path_progress": float(self.path_progress_float),
            "piper_step": int(self.piper_step_count),
            "piper_insertion_length": float(self.piper_insertion_length),
            "boundary_projection_count": int(self.boundary_projection_count),
            "last_boundary_projection": bool(self.last_boundary_projection),
            "elirobot_pose": self.elirobot_pose.astype(float).tolist(),
            "lateral_offset": self.lateral_offset.astype(float).tolist(),
            "path_tangent": self.local_path_tangent.astype(float).tolist(),
            "local_radius": float(self.local_radius),
            **tactile,
            "robot_state": getattr(self, "robot_state", {}),
            "done": done,
            "success": success,
            "failure_reason": failure_reason,
        }

    def _gym_obs(self, obs: Dict) -> Dict:
        scene_span = np.maximum(self.scene_max - self.scene_min, 1e-6)
        tip = (np.asarray(obs["tip_pos"], dtype=np.float32) - self.scene_min) / scene_span
        target = (np.asarray(obs["target_pos"], dtype=np.float32) - self.scene_min) / scene_span
        elirobot = (np.asarray(obs["elirobot_pose"], dtype=np.float32) - self.scene_min) / scene_span
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
                float(obs["path_progress"]) / max(len(self.paths[self.task]) - 1, 1),
                float(obs["piper_step"]) / max(self.config.max_steps, 1),
                obs["piper_insertion_length"] / max(self.config.max_steps * self.config.advance_step, 1e-6),
                *elirobot.tolist(),
                *np.asarray(obs["lateral_offset"], dtype=np.float32).tolist(),
                *np.asarray(obs["path_tangent"], dtype=np.float32).tolist(),
                float(obs["local_radius"]),
            ],
            dtype=np.float32,
        )
        return {"image": self.render(), "state": state, "instruction_id": 0 if self.task == "left" else 1}

    def _parse_action(self, action):
        if isinstance(action, dict):
            piper = action.get("piper", 0.0)
            delta = action.get("elirobot_delta", [0.0, 0.0, 0.0])
            return piper, delta
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(arr) < 4:
            raise ValueError("Dual-arm action must have at least 4 values or be a dict")
        return float(arr[0]), arr[1:4]

    @staticmethod
    def _normalize_action_delta(delta: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        magnitude = float(np.clip(magnitude, 0.0, 1.0))
        return (delta / norm * magnitude).astype(np.float32)


class DualArmPathExpert:
    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.015, retreat_on_contact: bool = True, center_gain: float = 2.4):
        self.rng = np.random.default_rng(rng_seed)
        self.steer_noise = steer_noise
        self.retreat_on_contact = retreat_on_contact
        self.center_gain = center_gain

    def act(self, env: DualArmGuidewire3DEnv) -> Dict[str, np.ndarray]:
        center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
        tactile = env._tactile_from_local_frame(center, n1, n2, radius)
        lateral = self._local_lateral(env, center, n1, n2)
        correction = self._magnet_correction(env, center, n1, n2, radius, lateral, tactile)
        if tactile["contact_flag"]:
            correction = correction + self._contact_release(env, n1, n2, tactile)
        if self.steer_noise > 0:
            correction = correction + self.rng.normal(0.0, self.steer_noise, size=3).astype(np.float32)

        piper = self._piper_command(env, radius, lateral, tactile)

        return {
            "piper": np.array([piper], dtype=np.float32),
            "elirobot_delta": correction.astype(np.float32),
        }

    def _local_lateral(self, env: DualArmGuidewire3DEnv, center: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
        return env._project_local(env.tip - center, n1, n2)

    def _magnet_correction(
        self,
        env: DualArmGuidewire3DEnv,
        center: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        radius: float,
        lateral: np.ndarray,
        tactile: Dict,
    ) -> np.ndarray:
        lookahead_target = self._lookahead_target(env, lookahead=env.dual_config.dual_arm_lookahead)
        lead_local = env._project_local(lookahead_target - center, n1, n2)
        alpha = env.dual_config.magnet_coupling / max(1.0 - env.dual_config.magnet_decay, 1e-3)

        # Keep the tip centered in the local vessel frame, with a slight bias
        # toward the upcoming curve so the controller can anticipate turns.
        desired_tip_local = (-0.85 * lateral) + (0.12 * lead_local)
        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            contact_local = env._project_local(normal, n1, n2)
            contact_norm = float(np.linalg.norm(contact_local))
            if contact_norm > 1e-6:
                desired_tip_local = desired_tip_local - (0.55 * tactile["contact_strength"] * radius) * (
                    contact_local / contact_norm
                )

        desired_local = desired_tip_local / max(alpha, 1e-6)
        max_local = min(env.dual_config.magnet_max_offset * 0.28, radius * 0.35)
        desired_norm = float(np.linalg.norm(desired_local))
        if desired_norm > max_local and desired_norm > 1e-6:
            desired_local = desired_local / desired_norm * max_local

        desired_pose = center + n1 * desired_local[0] + n2 * desired_local[1]
        correction = desired_pose - env.elirobot_pose
        urgency = 0.55
        if tactile["contact_flag"] or tactile["distance_to_wall"] < env.config.wall_margin * 0.8:
            urgency = 1.0
        elif float(np.linalg.norm(lateral)) > radius * 0.55:
            urgency = 0.8
        return env._normalize_action_delta(correction, magnitude=urgency)

    def _contact_release(self, env: DualArmGuidewire3DEnv, n1: np.ndarray, n2: np.ndarray, tactile: Dict) -> np.ndarray:
        normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
        contact_local = env._project_local(normal, n1, n2)
        contact_norm = float(np.linalg.norm(contact_local))
        if contact_norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        release_local = -(contact_local / contact_norm) * (0.45 + 0.35 * tactile["contact_strength"])
        return (n1 * release_local[0] + n2 * release_local[1]).astype(np.float32)

    def _piper_command(self, env: DualArmGuidewire3DEnv, radius: float, lateral: np.ndarray, tactile: Dict) -> float:
        safe_limit = max(radius - env.config.wall_clearance, 0.0)
        radial = float(np.linalg.norm(lateral))
        near_wall = tactile["distance_to_wall"] < env.config.wall_margin * 0.85 or radial > safe_limit * 0.78
        if self.retreat_on_contact and tactile["contact_strength"] > 0.72:
            return -0.30
        if tactile["contact_strength"] > 0.30 or near_wall:
            return 0.0
        return 1.0

    def _lookahead_target(self, env: DualArmGuidewire3DEnv, lookahead: Optional[int] = None) -> np.ndarray:
        path = env.paths[env.task]
        lookahead = lookahead or env.dual_config.dual_arm_lookahead
        idx = int(np.clip(np.floor(env.path_progress_float), 0, len(path) - 1))
        idx = min(idx + lookahead, len(path) - 1)
        return path[idx]


class DualArmRecoveryExpert(DualArmPathExpert):
    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.02, push_strength: float = 0.9):
        super().__init__(rng_seed=rng_seed, steer_noise=steer_noise, retreat_on_contact=True, center_gain=3.0)
        self.push_strength = push_strength

    def act(self, env: DualArmGuidewire3DEnv) -> Dict[str, np.ndarray]:
        action = super().act(env)
        center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
        tactile = env._tactile_from_local_frame(center, n1, n2, radius)
        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            contact_local = env._project_local(normal, n1, n2)
            contact_norm = float(np.linalg.norm(contact_local))
            if contact_norm > 1e-6:
                release = -(n1 * contact_local[0] + n2 * contact_local[1]).astype(np.float32)
                action["elirobot_delta"] = env._normalize_action_delta(
                    action["elirobot_delta"] + release,
                    magnitude=1.0,
                )
            action["piper"] = np.array([-0.40 if tactile["contact_strength"] > 0.65 else -0.10], dtype=np.float32)
        elif tactile["distance_to_wall"] < env.config.wall_margin * 0.9:
            action["piper"] = np.array([0.0], dtype=np.float32)
        return action

    def _lateral_push(self, env: DualArmGuidewire3DEnv) -> np.ndarray:
        center, tangent, n1, n2, _radius = env._local_path_frame(env.path_progress_float)
        return (n1 if env.task == "left" else -n1).astype(np.float32)


class DualArmSuccessRecoveryExpert(DualArmRecoveryExpert):
    """
    A more conservative recovery expert that emphasizes successful recentering.

    This variant is intended to generate positive examples where the wire first
    backs off from the wall, recenters, then continues forward through the
    branch instead of repeatedly projecting onto the boundary.
    """

    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.008, push_strength: float = 0.65):
        super().__init__(rng_seed=rng_seed, steer_noise=steer_noise, push_strength=push_strength)
        self._last_progress = None
        self._stalled_steps = 0

    def act(self, env: DualArmGuidewire3DEnv) -> Dict[str, np.ndarray]:
        center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
        tactile = env._tactile_from_local_frame(center, n1, n2, radius)
        lateral = self._local_lateral(env, center, n1, n2)
        lookahead_target = self._lookahead_target(env, lookahead=max(5, env.dual_config.dual_arm_lookahead + 3))
        lead_local = env._project_local(lookahead_target - center, n1, n2)
        alpha = env.dual_config.magnet_coupling / max(1.0 - env.dual_config.magnet_decay, 1e-3)
        safe_limit = max(radius - env.config.wall_clearance, 0.0)
        radial = float(np.linalg.norm(lateral))
        projected = bool(getattr(env, "last_boundary_projection", False))
        near_wall = (
            tactile["contact_flag"]
            or tactile["distance_to_wall"] < env.config.wall_margin * 0.92
            or radial > safe_limit * 0.86
            or projected
        )
        severe = tactile["contact_strength"] > 0.64 or radial > safe_limit * 0.98 or projected

        progress = float(env.path_progress_float)
        if self._last_progress is not None and not tactile["contact_flag"] and progress <= self._last_progress + 0.08:
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0
        self._last_progress = progress

        forced_advance = self._stalled_steps >= 8 and not tactile["contact_flag"] and not projected
        if forced_advance:
            near_wall = False
            severe = False

        if severe:
            phase = "release"
            piper = -0.10
        elif near_wall:
            phase = "align"
            piper = 0.30
        else:
            phase = "advance"
            piper = 1.0

        if forced_advance:
            phase = "advance"
            piper = 1.0

        if env.path_progress_float > len(env.paths[env.task]) - 14:
            piper = 1.0

        if phase == "release":
            desired_tip_local = (-1.25 * lateral) + (0.04 * lead_local)
        elif phase == "align":
            desired_tip_local = (-1.05 * lateral) + (0.16 * lead_local)
        else:
            desired_tip_local = (-0.52 * lateral) + (0.34 * lead_local)

        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            contact_local = env._project_local(normal, n1, n2)
            contact_norm = float(np.linalg.norm(contact_local))
            if contact_norm > 1e-6:
                desired_tip_local = desired_tip_local - (0.75 * tactile["contact_strength"] * radius) * (
                    contact_local / contact_norm
                )

        desired_local = desired_tip_local / max(alpha, 1e-6)
        max_local = min(env.dual_config.magnet_max_offset * 0.26, radius * 0.32)
        desired_norm = float(np.linalg.norm(desired_local))
        if desired_norm > max_local and desired_norm > 1e-6:
            desired_local = desired_local / desired_norm * max_local

        desired_pose = center + n1 * desired_local[0] + n2 * desired_local[1]
        correction = desired_pose - env.elirobot_pose
        urgency = 1.0 if phase != "advance" else 0.9
        action = {
            "piper": np.array([piper], dtype=np.float32),
            "elirobot_delta": env._normalize_action_delta(correction, magnitude=urgency),
        }
        return action
