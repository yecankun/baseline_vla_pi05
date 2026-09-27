from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from simulation.mujoco_guided_wire_env import (
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoMagneticGuideExpert,
)


@dataclass
class TipGuidedWireConfig(MuJoCoGuidedWireConfig):
    """Tip-centric guidewire prototype.

    This keeps the vessel, cameras, robot kinematics, Piper feed semantics, and
    Elite-magnet attachment from the MuJoCo environment, but replaces the full
    polyline physics with a constrained magnetic tip and a hard-elastic visual
    tail.
    """

    tip_progress_gain: float = 0.45
    tip_retract_progress_gain: float = 0.015
    tip_magnetic_progress_gain: float = 0.020
    tip_lateral_magnetic_gain: float = 0.16
    tip_lateral_centering_gain: float = 0.06
    tip_contact_relief_gain: float = 0.42
    tip_max_lateral_radius_fraction: float = 0.82
    tip_tail_decay_segments: float = 18.0
    wire_visual_start_at_route_entry: bool = False
    wire_visual_route_smoothing_samples: int = 12
    wire_visual_route_resume_offset_decay_m: float = 0.06


class TipGuidedWireEnv(MuJoCoGuidedWireEnv):
    """MuJoCo-rendered environment with point-like magnetic guidewire tip dynamics."""

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._rebuild_visual_tail()
        self.last_tactile = self._tactile(self.tip)
        self._sync_mujoco()
        obs_dict = self._obs_dict(done=False, success=False, failure_reason=None)
        return self._gym_obs(obs_dict), self._info(obs_dict)

    def step(self, action):
        controller_request = self._controller_request_from_action(action)
        piper_cmd_override = self._piper_step_command_override(action)
        primitive_steps = max(int(getattr(self.config, "piper_primitive_steps", 1)), 1)
        piper_target, elite_target = self._parse_action(action)
        piper_feed_signal_target = self._piper_feed_signal(piper_target)
        if primitive_steps <= 1:
            piper_target = self._piper_joint_vector_from_feed(piper_feed_signal_target)
        else:
            piper_target = self._robot_joint_vector("piper")
        elite_target = self._clip_robot_joint_vector("elite", elite_target)
        requested_elite_target = elite_target.copy()

        previous_signal = float(getattr(self, "piper_feed_signal", self.piper_feed_home_signal))
        self._set_robot_joint_vector("piper", piper_target)
        self._apply_elite_target(elite_target)
        executed_elite_target = self._elite_target_smoothed.copy()
        self._refresh_robot_tool_poses()
        self.magnetic_pose = self._effective_magnetic_pose()

        current_signal = self._piper_feed_signal(self._robot_joint_vector("piper"))
        signal_limit = min(
            float(self.piper_joint_limits.get("joint7", (0.0, 0.035))[1]),
            -float(self.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]),
        )
        self.piper_feed_signal = float(np.clip(current_signal, -0.18 * signal_limit, signal_limit))
        self.piper_insertion_length = float(self.piper_feed_signal - self.piper_feed_home_signal)
        feed_unit = max(float(self.config.advance_step) * float(self.config.piper_advection_scale), 1e-8)
        piper_cmd = float(np.clip((self.piper_feed_signal - previous_signal) / feed_unit, -1.0, 1.0))
        if piper_cmd_override is not None:
            piper_cmd = piper_cmd_override
        policy_piper_cmd = self._piper_policy_command_from_action(action, piper_cmd)
        executed_piper_cmd, piper_busy = self._piper_controller_execute(policy_piper_cmd)
        if primitive_steps > 1:
            piper_target = self._piper_joint_vector_from_feed_command(executed_piper_cmd)
            self._set_robot_joint_vector("piper", piper_target)
            current_signal = self._piper_feed_signal(self._robot_joint_vector("piper"))
            self.piper_feed_signal = float(np.clip(current_signal, -0.18 * signal_limit, signal_limit))
            self.piper_insertion_length = float(self.piper_feed_signal - self.piper_feed_home_signal)
        piper_executed_command = self._piper_command_label(executed_piper_cmd)
        requested_feed = controller_request.get("piper_requested_feed")
        if requested_feed is None:
            requested_feed = float(policy_piper_cmd)
        requested_elite_joints = {
            name: float(requested_elite_target[index])
            for index, name in enumerate(self.elite_joint_names)
        }
        executed_elite_joints = {
            name: float(executed_elite_target[index])
            for index, name in enumerate(self.elite_joint_names)
        }
        self.controller_state = {
            "piper_intent": controller_request.get("piper_intent") or self._piper_command_label(float(requested_feed)),
            "piper_executed_command": piper_executed_command,
            "piper_requested_feed": float(requested_feed),
            "piper_executed_feed": float(executed_piper_cmd),
            "piper_motion_state": "idle" if piper_executed_command == "hold" else ("feeding" if piper_executed_command == "feed" else "retracting"),
            "piper_policy_command": float(policy_piper_cmd),
            "piper_policy_command_label": self._piper_command_label(policy_piper_cmd),
            "piper_primitive_steps": int(max(getattr(self.config, "piper_primitive_steps", 1), 1)),
            "piper_primitive_remaining_steps": int(getattr(self, "_piper_primitive_remaining_steps", 0)),
            "piper_step_count": int(self.step_count),
            "piper_insertion_length": float(self.piper_insertion_length),
            "piper_busy": bool(piper_busy),
            "piper_cooldown": False,
            "elite_action_type": str(controller_request.get("elite_action_type", "unknown")),
            "elite_tcp_delta_6d": controller_request.get("elite_tcp_delta_6d"),
            "elite_requested_tcp_pose_6d": controller_request.get("elite_tcp_pose_6d"),
            "elite_executed_tcp_pose_6d": getattr(self, "elite_tcp_pose_6d", self.robot_tool_pose6d("elite")).astype(float).tolist(),
            "elite_requested_joints": requested_elite_joints,
            "elite_executed_joints": executed_elite_joints,
            "elite_target_limited": bool(float(np.max(np.abs(executed_elite_target - requested_elite_target))) > 1e-7),
        }

        self._advance_tip(executed_piper_cmd)
        self._rebuild_visual_tail()
        self.step_count += 1
        self.last_tactile = self._tactile(self.tip)
        self.magnetic_pose = self._effective_magnetic_pose()

        if self.last_tactile["distance_to_wall"] < 0.0:
            self.boundary_projection_count += 1
            self.boundary_projection_window += 1
        else:
            self.boundary_projection_window = max(self.boundary_projection_window - 1, 0)
        if self.last_tactile["contact_strength"] > 0.85:
            self.severe_contact_count += 1
        else:
            self.severe_contact_count = max(self.severe_contact_count - 1, 0)

        self.guidewire_points.append(self.tip.copy())
        self.elite_points.append(self.elite_pose.copy())
        self._sync_mujoco()

        success = self._success()
        failure_reason = self._failure_reason(self.last_tactile)
        terminated = success or failure_reason is not None
        truncated = self.step_count >= self.config.max_steps
        obs = self._obs_dict(done=terminated or truncated, success=success, failure_reason=failure_reason)
        reward = self._reward(obs, failure_reason)
        return self._gym_obs(obs), reward, terminated, truncated, self._info(obs)

    def _apply_elite_target(self, elite_target: np.ndarray) -> None:
        self._elite_target_smoothed = (
            (1.0 - self.config.elite_motion_smoothing) * self._elite_target_smoothed
            + self.config.elite_motion_smoothing * elite_target
        ).astype(np.float32)
        current_elite = self._robot_joint_vector("elite")
        delta = self._elite_target_smoothed - current_elite
        if self.config.elite_joint_rate_limit > 0.0:
            limit = float(self.config.elite_joint_rate_limit)
            delta = np.clip(delta, -limit, limit)
        if self.config.elite_joint_accel_limit > 0.0:
            previous_delta = getattr(self, "_elite_previous_joint_delta", np.zeros_like(delta, dtype=np.float32))
            if previous_delta.shape != delta.shape:
                previous_delta = np.zeros_like(delta, dtype=np.float32)
            accel_limit = float(self.config.elite_joint_accel_limit)
            delta = previous_delta + np.clip(delta - previous_delta, -accel_limit, accel_limit)
        self._elite_target_smoothed = (current_elite + delta).astype(np.float32)
        self._elite_previous_joint_delta = delta.astype(np.float32)
        self._set_robot_joint_vector("elite", self._elite_target_smoothed)

    def _advance_tip(self, piper_cmd: float) -> None:
        previous_tip = self.tip.copy()
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        step_length = max(float(self.path_step_length[self.task]), 1e-6)
        magnet_vec = self.magnetic_pose - self.tip
        magnet_forward = float(np.dot(magnet_vec, tangent))
        progress_delta = max(0.0, piper_cmd) * float(self.config.tip_progress_gain)
        progress_delta -= max(0.0, -piper_cmd) * float(self.config.tip_retract_progress_gain)
        progress_delta += np.clip(magnet_forward / step_length, -0.5, 0.5) * float(self.config.tip_magnetic_progress_gain)
        self.path_progress_float = float(
            np.clip(self.path_progress_float + progress_delta, 0.0, len(self.paths[self.task]) - 1)
        )
        self.path_progress_index = int(np.floor(self.path_progress_float))

        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        current_local = np.array(
            [
                float(np.dot(previous_tip - center, n1)),
                float(np.dot(previous_tip - center, n2)),
            ],
            dtype=np.float32,
        )
        magnet_local = np.array(
            [
                float(np.dot(self.magnetic_pose - center, n1)),
                float(np.dot(self.magnetic_pose - center, n2)),
            ],
            dtype=np.float32,
        )
        lateral = (
            current_local * (1.0 - float(self.config.tip_lateral_centering_gain))
            + magnet_local * float(self.config.tip_lateral_magnetic_gain)
        ).astype(np.float32)

        tactile = self._tactile(center + n1 * lateral[0] + n2 * lateral[1])
        if tactile["contact_strength"] > 0.0:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            normal_local = np.array([float(np.dot(normal, n1)), float(np.dot(normal, n2))], dtype=np.float32)
            normal_norm = float(np.linalg.norm(normal_local))
            if normal_norm > 1e-6:
                lateral -= (
                    normal_local
                    / normal_norm
                    * float(tactile["contact_strength"])
                    * float(radius)
                    * float(self.config.tip_contact_relief_gain)
                )

        limit = max(float(radius) * float(self.config.tip_max_lateral_radius_fraction), 1e-6)
        norm = float(np.linalg.norm(lateral))
        if norm > limit:
            lateral = lateral / norm * limit
        self.tip = (center + n1 * lateral[0] + n2 * lateral[1]).astype(np.float32)
        motion = self.tip - previous_tip
        if float(np.linalg.norm(motion)) > 1e-6:
            self.heading = (motion / max(float(np.linalg.norm(motion)), 1e-6)).astype(np.float32)
        else:
            self.heading = tangent.astype(np.float32)

    def _rebuild_visual_tail(self) -> None:
        center, _tangent, n1, n2, _radius = self._local_path_frame(self.path_progress_float)
        lateral = np.array(
            [float(np.dot(self.tip - center, n1)), float(np.dot(self.tip - center, n2))],
            dtype=np.float32,
        )
        segment_count = int(self.config.wire_segments)
        if segment_count <= 1:
            self.positions[-1] = self.tip
            self.velocities[:] = 0.0
            return

        piper_tcp = self._wire_visual_piper_exit_point()
        max_progress = float(self.path_progress_float)
        via_progresses = self._wire_visual_via_progresses(max_progress=max_progress)
        pre_route_points = self._wire_visual_pre_route_points()
        route_override_points = self._wire_visual_route_override_points(max_progress=max_progress)
        dynamic_route_points = self._wire_visual_dynamic_route_points(max_progress=max_progress)
        explicit_entry_progress = (
            self._wire_visual_entry_progress(piper_tcp, max_progress=max_progress)
            if self._wire_visual_has_entry_progress()
            else 0.0
        )
        entry_point = self._wire_visual_entry_point()
        visual_start_points: list[np.ndarray] = [piper_tcp]
        if pre_route_points:
            visual_start_points.extend(pre_route_points)
        if entry_point is not None:
            visual_start_points.append(entry_point)
        if route_override_points or self._wire_visual_has_route_override_points():
            control_points = self._manual_visual_prefix_points(
                visual_start_points=visual_start_points,
                route_override_points=route_override_points,
                dynamic_route_points=dynamic_route_points,
                segment_count=segment_count,
                max_progress=max_progress,
                lateral=lateral,
            )
            self.positions[:] = self._resample_polyline(control_points, segment_count)
            self.positions[-1] = self.tip
            self._straighten_visual_tip_head()
            self.velocities[:] = 0.0
            return
        via_route_samples: list[np.ndarray] = []
        route_breaks = [explicit_entry_progress]
        route_breaks.extend(progress for progress in via_progresses if progress > explicit_entry_progress + 1e-6)
        if len(route_breaks) == 1 and explicit_entry_progress <= max_progress:
            route_breaks.append(explicit_entry_progress)
        for start_progress, end_progress in zip(route_breaks[:-1], route_breaks[1:]):
            if end_progress <= start_progress + 1e-6:
                continue
            route_distance = self._route_distance_between(start_progress, end_progress)
            sample_count = max(3, int(np.ceil(route_distance / 0.0025)) + 1)
            segment = self._sample_visual_route_span(
                start_progress=start_progress,
                end_progress=end_progress,
                count=sample_count,
                lateral=np.zeros(2, dtype=np.float32),
                decay_lateral=False,
            )
            skip_start = bool(via_route_samples) or entry_point is not None
            via_route_samples.extend(segment[1:] if skip_start else segment)
        if not via_route_samples and explicit_entry_progress <= max_progress and entry_point is None:
            via_route_samples.append(self._route_center_at(explicit_entry_progress))
        via_end_progress = route_breaks[-1] if route_breaks else explicit_entry_progress
        if via_end_progress < max_progress - 1e-6:
            route_distance = self._route_distance_between(via_end_progress, max_progress)
            sample_count = max(segment_count, int(np.ceil(route_distance / 0.0025)) + 1)
            route_samples = self._sample_visual_route_span(
                start_progress=via_end_progress,
                end_progress=max_progress,
                count=sample_count,
                lateral=lateral,
                decay_lateral=True,
            )
            if via_route_samples or entry_point is not None:
                route_samples = route_samples[1:]
        else:
            route_samples = [self.tip.copy()]
        if bool(getattr(self.config, "wire_visual_start_at_route_entry", False)) and via_route_samples:
            visual_start_points = []
        if not visual_start_points and via_route_samples:
            control_points = [*via_route_samples, *route_samples]
        else:
            control_points = [*visual_start_points, *via_route_samples, *route_samples]
        self.positions[:] = self._resample_polyline(control_points, segment_count)
        self.positions[-1] = self.tip
        self._straighten_visual_tip_head()
        self.velocities[:] = 0.0

    def _straighten_visual_tip_head(self) -> None:
        """Keep the red visual head from inheriting small tail resampling kinks."""
        if str(getattr(self.config, "wire_visual_mode", "")) != "line":
            return
        tip_segments = max(int(getattr(self.config, "wire_tip_visual_segments", 0)), 0)
        if tip_segments <= 0 or len(self.positions) < 3:
            return
        last_visible_index = len(self.positions) - 2
        start_index = max(last_visible_index - tip_segments + 1, 0)
        if start_index >= last_visible_index:
            return
        anchor = self.positions[start_index].copy()
        tip = np.asarray(self.tip, dtype=np.float32)
        direction = tip - anchor
        if float(np.linalg.norm(direction)) < 1e-8:
            return
        for index in range(start_index + 1, len(self.positions)):
            alpha = float(index - start_index) / float(len(self.positions) - 1 - start_index)
            self.positions[index] = (anchor * (1.0 - alpha) + tip * alpha).astype(np.float32)

    def _route_center_at(self, progress: float) -> np.ndarray:
        center, _tangent, _n1, _n2, _radius = self._local_path_frame(float(progress))
        return np.asarray(center, dtype=np.float32)

    def _wire_visual_piper_exit_point(self) -> np.ndarray:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        for key in (
            f"{task}_wire_visual_piper_exit_point",
            f"{task}_wire_visual_piper_tcp_point",
            "wire_visual_piper_exit_point",
            "wire_visual_piper_tcp_point",
        ):
            if key in route_config:
                item = route_config[key]
                point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
                frame = str(item.get("frame", "scene") if isinstance(item, dict) else "scene")
                return (point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)).astype(np.float32)
        return np.asarray(getattr(self, "piper_pose", self.positions[0]), dtype=np.float32)

    def _wire_visual_pre_route_points(self) -> list[np.ndarray]:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        points: list[np.ndarray] = []
        for key in (
            f"{task}_wire_visual_pre_route_points",
            f"{task}_wire_visual_entry_control_points",
            "wire_visual_pre_route_points",
            "wire_visual_entry_control_points",
        ):
            if key not in route_config:
                continue
            for item in route_config.get(key) or []:
                point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
                frame = str(item.get("frame", "scene") if isinstance(item, dict) else "scene")
                scene_point = point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)
                points.append(scene_point.astype(np.float32))
        return points

    def _wire_visual_entry_point(self) -> np.ndarray | None:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        for key in (
            f"{task}_wire_visual_entry_point",
            f"{task}_wire_visual_vessel_entry_point",
            "wire_visual_entry_point",
            "wire_visual_vessel_entry_point",
        ):
            if key not in route_config:
                continue
            item = route_config[key]
            point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
            frame = str(item.get("frame", "scene") if isinstance(item, dict) else "scene")
            return (point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)).astype(np.float32)
        return None

    def _wire_visual_route_override_points(self, max_progress: float) -> list[np.ndarray]:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        rows: list[tuple[float, np.ndarray]] = []
        for key in (
            f"{task}_wire_visual_route_points",
            f"{task}_wire_visual_centerline_points",
            "wire_visual_route_points",
            "wire_visual_centerline_points",
        ):
            if key not in route_config:
                continue
            for index, item in enumerate(route_config.get(key) or []):
                point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
                frame = str(item.get("frame", "scene") if isinstance(item, dict) else "scene")
                progress = float(item.get("progress", index) if isinstance(item, dict) else index)
                if progress > max_progress + 1e-6:
                    continue
                scene_point = point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)
                rows.append((progress, scene_point.astype(np.float32)))
        rows.sort(key=lambda row: row[0])
        return [point for _progress, point in rows]

    def _wire_visual_has_route_override_points(self) -> bool:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        return any(
            key in route_config
            for key in (
                f"{task}_wire_visual_route_points",
                f"{task}_wire_visual_centerline_points",
                "wire_visual_route_points",
                "wire_visual_centerline_points",
            )
        )

    def _wire_visual_dynamic_route_points(self, max_progress: float) -> list[np.ndarray]:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        rows: list[tuple[float, np.ndarray]] = []
        key = f"{task}_wire_visual_dynamic_route_points"
        if key not in route_config:
            key = "wire_visual_dynamic_route_points"
        if key in route_config:
            for index, item in enumerate(route_config.get(key) or []):
                point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
                frame = str(item.get("frame", "scene") if isinstance(item, dict) else "scene")
                scene_point = point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)
                order = float(item.get("progress", index) if isinstance(item, dict) else index)
                route_progress = (
                    float(item["route_progress"])
                    if isinstance(item, dict) and "route_progress" in item
                    else self._nearest_route_progress_to_point(scene_point, max_progress=None)
                )
                if route_progress > max_progress + 1e-6:
                    continue
                rows.append((order, scene_point.astype(np.float32)))
        rows.sort(key=lambda row: row[0])
        return [point for _progress, point in rows]

    def _wire_visual_entry_progress(self, piper_tcp: np.ndarray, max_progress: float) -> float:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        for key in (
            f"{task}_wire_visual_entry_progress",
            f"{task}_wire_visual_start_progress",
            "wire_visual_entry_progress",
            "wire_visual_start_progress",
        ):
            if key in route_config:
                return float(np.clip(float(route_config[key]), 0.0, max_progress))
        return 0.0

    def _wire_visual_has_entry_progress(self) -> bool:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        return any(
            key in route_config
            for key in (
                f"{task}_wire_visual_entry_progress",
                f"{task}_wire_visual_start_progress",
                "wire_visual_entry_progress",
                "wire_visual_start_progress",
            )
        )

    def _wire_visual_via_progresses(self, max_progress: float) -> list[float]:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        progresses: list[float] = []
        for key in (
            f"{task}_wire_visual_tail_progresses",
            f"{task}_wire_visual_via_progresses",
            "wire_visual_tail_progresses",
            "wire_visual_via_progresses",
        ):
            if key in route_config:
                for value in route_config.get(key) or []:
                    progresses.append(float(np.clip(float(value), 0.0, max_progress)))
        for key in (
            f"{task}_wire_visual_tail_points",
            f"{task}_wire_visual_via_points",
            "wire_visual_tail_points",
            "wire_visual_via_points",
        ):
            if key in route_config:
                for item in route_config.get(key) or []:
                    if isinstance(item, dict) and "progress" in item:
                        progresses.append(float(np.clip(float(item["progress"]), 0.0, max_progress)))
                        continue
                    point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
                    frame = str(item.get("frame", "route_raw") if isinstance(item, dict) else "route_raw")
                    scene_point = point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)
                    progresses.append(self._nearest_route_progress_to_point(scene_point, max_progress=max_progress))
        return sorted({round(float(progress), 6) for progress in progresses})

    def _wire_visual_via_points(self, max_progress: float) -> list[np.ndarray]:
        return [self._route_center_at(progress) for progress in self._wire_visual_via_progresses(max_progress=max_progress)]

    def _wire_visual_via_surface_points(self, max_progress: float) -> list[np.ndarray]:
        route_config = getattr(getattr(self, "reference_env", None), "route_config", {}) or {}
        task = str(getattr(self, "task", "left"))
        points: list[np.ndarray] = []
        for key in (
            f"{task}_wire_visual_tail_points",
            f"{task}_wire_visual_via_points",
            "wire_visual_tail_points",
            "wire_visual_via_points",
        ):
            if key in route_config:
                for item in route_config.get(key) or []:
                    if not isinstance(item, dict) or "surface_point" not in item:
                        continue
                    point = np.asarray(item["surface_point"], dtype=np.float32)
                    frame = str(item.get("surface_frame", item.get("frame", "scene")))
                    scene_point = point if frame in {"scene", "mujoco", "world"} else self._scene_point(point)
                    progress = self._nearest_route_progress_to_point(scene_point, max_progress=max_progress)
                    if progress <= max_progress:
                        points.append(scene_point.astype(np.float32))
        return points

    def _sample_visual_route_span(
        self,
        start_progress: float,
        end_progress: float,
        count: int,
        lateral: np.ndarray,
        decay_lateral: bool,
    ) -> list[np.ndarray]:
        route_points = []
        if count <= 0:
            return route_points
        start_progress = float(np.clip(start_progress, 0.0, len(self.paths[self.task]) - 1))
        end_progress = float(np.clip(end_progress, 0.0, len(self.paths[self.task]) - 1))
        for route_index in range(count):
            alpha = float(route_index) / float(max(count - 1, 1))
            progress = start_progress + (end_progress - start_progress) * alpha
            tail_center, _tail_tangent, tail_n1, tail_n2, _tail_radius = self._local_path_frame(progress)
            if decay_lateral:
                back_segments = float(count - 1 - route_index)
                decay = np.exp(-back_segments / max(float(self.config.tip_tail_decay_segments), 1e-6))
                tail_lateral = lateral * float(decay)
            else:
                tail_lateral = lateral
            route_points.append((tail_center + tail_n1 * tail_lateral[0] + tail_n2 * tail_lateral[1]).astype(np.float32))
        return route_points

    def _sample_visual_route_span_with_initial_offset(
        self,
        start_progress: float,
        end_progress: float,
        count: int,
        lateral: np.ndarray,
        initial_offset: np.ndarray,
    ) -> list[np.ndarray]:
        route_points = []
        if count <= 0:
            return route_points
        start_progress = float(np.clip(start_progress, 0.0, len(self.paths[self.task]) - 1))
        end_progress = float(np.clip(end_progress, 0.0, len(self.paths[self.task]) - 1))
        decay_distance = max(float(getattr(self.config, "wire_visual_route_resume_offset_decay_m", 0.06)), 1e-6)
        traveled = 0.0
        previous_center: np.ndarray | None = None
        for route_index in range(count):
            alpha = float(route_index) / float(max(count - 1, 1))
            progress = start_progress + (end_progress - start_progress) * alpha
            center, _tangent, n1, n2, _radius = self._local_path_frame(progress)
            if previous_center is not None:
                traveled += float(np.linalg.norm(center - previous_center))
            previous_center = center
            manual_decay = max(0.0, 1.0 - traveled / decay_distance)
            back_segments = float(count - 1 - route_index)
            tip_decay = np.exp(-back_segments / max(float(self.config.tip_tail_decay_segments), 1e-6))
            start_lateral = np.array(
                [float(np.dot(initial_offset, n1)), float(np.dot(initial_offset, n2))],
                dtype=np.float32,
            )
            tail_lateral = lateral * float(tip_decay) + start_lateral * float(manual_decay)
            route_points.append((center + n1 * tail_lateral[0] + n2 * tail_lateral[1]).astype(np.float32))
        return route_points

    def _sample_visual_route(
        self,
        start_progress: float,
        end_progress: float,
        count: int,
        lateral: np.ndarray,
    ) -> list[np.ndarray]:
        return self._sample_visual_route_span(
            start_progress=start_progress,
            end_progress=end_progress,
            count=count,
            lateral=lateral,
            decay_lateral=True,
        )

    @staticmethod
    def _resample_polyline(points: list[np.ndarray], count: int) -> np.ndarray:
        polyline = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(polyline) == 0:
            return np.zeros((count, 3), dtype=np.float32)
        if len(polyline) == 1 or count <= 1:
            return np.repeat(polyline[:1], max(count, 1), axis=0).astype(np.float32)
        segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total = float(cumulative[-1])
        if total <= 1e-8:
            return np.repeat(polyline[:1], count, axis=0).astype(np.float32)
        samples = np.linspace(0.0, total, count)
        out = np.zeros((count, 3), dtype=np.float32)
        seg_idx = 0
        for sample_index, distance in enumerate(samples):
            while seg_idx < len(segment_lengths) - 1 and cumulative[seg_idx + 1] < distance:
                seg_idx += 1
            seg_start = cumulative[seg_idx]
            seg_len = max(float(segment_lengths[seg_idx]), 1e-8)
            alpha = float((distance - seg_start) / seg_len)
            out[sample_index] = (1.0 - alpha) * polyline[seg_idx] + alpha * polyline[seg_idx + 1]
        return out

    def _smooth_manual_visual_route(self, points: list[np.ndarray]) -> list[np.ndarray]:
        polyline = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(polyline) <= 2:
            return [point.astype(np.float32) for point in polyline]
        samples_per_segment = int(max(getattr(self.config, "wire_visual_route_smoothing_samples", 12), 1))
        if samples_per_segment <= 1:
            return [point.astype(np.float32) for point in polyline]

        padded = np.vstack([polyline[:1], polyline, polyline[-1:]])
        smoothed: list[np.ndarray] = [polyline[0].astype(np.float32)]
        for index in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[index - 1], padded[index], padded[index + 1], padded[index + 2]
            d01 = max(float(np.linalg.norm(p1 - p0)), 1e-6)
            d12 = max(float(np.linalg.norm(p2 - p1)), 1e-6)
            d23 = max(float(np.linalg.norm(p3 - p2)), 1e-6)
            t0 = 0.0
            t1 = t0 + np.sqrt(d01)
            t2 = t1 + np.sqrt(d12)
            t3 = t2 + np.sqrt(d23)
            for sample_index in range(1, samples_per_segment + 1):
                t = float(t1 + (t2 - t1) * sample_index / samples_per_segment)
                a1 = p1 if t1 <= t0 + 1e-8 else ((t1 - t) / (t1 - t0)) * p0 + ((t - t0) / (t1 - t0)) * p1
                a2 = ((t2 - t) / (t2 - t1)) * p1 + ((t - t1) / (t2 - t1)) * p2
                a3 = p2 if t3 <= t2 + 1e-8 else ((t3 - t) / (t3 - t2)) * p2 + ((t - t2) / (t3 - t2)) * p3
                b1 = ((t2 - t) / (t2 - t0)) * a1 + ((t - t0) / (t2 - t0)) * a2
                b2 = ((t3 - t) / (t3 - t1)) * a2 + ((t - t1) / (t3 - t1)) * a3
                point = ((t2 - t) / (t2 - t1)) * b1 + ((t - t1) / (t2 - t1)) * b2
                smoothed.append(point.astype(np.float32))
        smoothed[-1] = polyline[-1].astype(np.float32)
        return smoothed

    def _manual_visual_prefix_points(
        self,
        *,
        visual_start_points: list[np.ndarray],
        route_override_points: list[np.ndarray],
        dynamic_route_points: list[np.ndarray],
        segment_count: int,
        max_progress: float,
        lateral: np.ndarray,
    ) -> list[np.ndarray]:
        if not route_override_points:
            return [*self._smooth_manual_visual_route(visual_start_points), self.tip.copy()]

        attach_point = np.asarray(route_override_points[-1], dtype=np.float32)
        prefix_route_points = route_override_points[:-1]
        attach_progress = self._nearest_route_progress_to_point(attach_point, max_progress=max_progress)
        if attach_progress < max_progress - 1e-6:
            route_distance = self._route_distance_between(attach_progress, max_progress)
            sample_count = max(segment_count, int(np.ceil(route_distance / 0.0025)) + 1)
            route_samples = self._sample_visual_route_span(
                start_progress=attach_progress,
                end_progress=max_progress,
                count=sample_count,
                lateral=lateral,
                decay_lateral=True,
            )
            if route_samples:
                if dynamic_route_points:
                    fixed_prefix = self._smooth_manual_visual_route([*visual_start_points, *prefix_route_points, attach_point])
                    blend_count = min(len(fixed_prefix), max(4, int(getattr(self.config, "wire_visual_route_smoothing_samples", 12))))
                    blend_controls = [point.copy() for point in fixed_prefix[-blend_count:]]
                    resume_progress = self._nearest_route_progress_to_point(dynamic_route_points[-1], max_progress=max_progress)
                    if resume_progress < max_progress - 1e-6:
                        route_distance = self._route_distance_between(resume_progress, max_progress)
                        resume_sample_count = max(segment_count, int(np.ceil(route_distance / 0.0025)) + 1)
                        resume_center = self._route_center_at(resume_progress)
                        dynamic_tail_end = self._sample_visual_route_span_with_initial_offset(
                            start_progress=resume_progress,
                            end_progress=max_progress,
                            count=resume_sample_count,
                            lateral=lateral,
                            initial_offset=np.asarray(dynamic_route_points[-1], dtype=np.float32) - resume_center,
                        )
                        if dynamic_tail_end and float(np.linalg.norm(dynamic_tail_end[0] - dynamic_route_points[-1])) < 1e-5:
                            dynamic_tail_end = dynamic_tail_end[1:]
                    else:
                        dynamic_tail_end = [self.tip.copy()]
                    blended_tail = self._smooth_manual_visual_route(
                        [*blend_controls, *dynamic_route_points, *dynamic_tail_end]
                    )
                    return [*fixed_prefix[:-blend_count], *blended_tail]
                fixed_prefix = self._smooth_manual_visual_route([*visual_start_points, *prefix_route_points, route_samples[0]])
                return [*fixed_prefix, *route_samples[1:]]
        return [*self._smooth_manual_visual_route([*visual_start_points, *prefix_route_points]), self.tip.copy()]

    def _nearest_route_progress_to_point(self, point: np.ndarray, max_progress: Optional[float] = None) -> float:
        path = np.asarray(self.paths[self.task], dtype=np.float32)
        if len(path) <= 1:
            return 0.0
        max_progress_value = float(len(path) - 1 if max_progress is None else np.clip(max_progress, 0.0, len(path) - 1))
        best_progress = 0.0
        best_dist = float("inf")
        last_segment = int(np.clip(np.ceil(max_progress_value), 1, len(path) - 1))
        for index in range(last_segment):
            start_progress = float(index)
            end_progress = min(float(index + 1), max_progress_value)
            if end_progress < start_progress:
                break
            start = path[index]
            end = path[index + 1]
            segment = end - start
            seg_len2 = float(np.dot(segment, segment))
            if seg_len2 <= 1e-12:
                candidate_progress = start_progress
                candidate = start
            else:
                upper = max(end_progress - start_progress, 0.0)
                t = float(np.clip(np.dot(point - start, segment) / seg_len2, 0.0, upper))
                candidate_progress = start_progress + t
                candidate = start + segment * t
            dist = float(np.linalg.norm(point - candidate))
            if dist < best_dist:
                best_dist = dist
                best_progress = candidate_progress
        return float(best_progress)

    def _route_distance_between(self, start_progress: float, end_progress: float) -> float:
        if end_progress <= start_progress:
            return 0.0
        return self._route_distance_between_unordered(start_progress, end_progress)

    def _route_distance_between_unordered(self, start_progress: float, end_progress: float) -> float:
        if np.isclose(start_progress, end_progress):
            return 0.0
        path = np.asarray(self.paths[self.task], dtype=np.float32)
        lower_progress = min(float(start_progress), float(end_progress))
        upper_progress = max(float(start_progress), float(end_progress))
        start_progress = float(np.clip(lower_progress, 0.0, len(path) - 1))
        end_progress = float(np.clip(upper_progress, 0.0, len(path) - 1))
        start_idx = int(np.floor(start_progress))
        end_idx = int(np.floor(end_progress))
        if start_idx == end_idx:
            p0, _tangent0, _n10, _n20, _radius0 = self._local_path_frame(start_progress)
            p1, _tangent1, _n11, _n21, _radius1 = self._local_path_frame(end_progress)
            return float(np.linalg.norm(p1 - p0))

        start_point, _tangent0, _n10, _n20, _radius0 = self._local_path_frame(start_progress)
        end_point, _tangent1, _n11, _n21, _radius1 = self._local_path_frame(end_progress)
        distance = float(np.linalg.norm(path[start_idx + 1] - start_point))
        for index in range(start_idx + 1, end_idx):
            distance += float(np.linalg.norm(path[index + 1] - path[index]))
        distance += float(np.linalg.norm(end_point - path[end_idx]))
        return distance

    def segment_min_distance_to_wall(self) -> float:
        return float(self._tactile(self.tip)["distance_to_wall"])


class TipMagneticGuideExpert(MuJoCoMagneticGuideExpert):
    """Alias expert for the tip-centric prototype."""
