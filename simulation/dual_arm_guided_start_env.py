from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from simulation.dual_arm_guidewire_3d_env import (
    DualArmGuidewire3DEnv,
    DualArmMesh3DEnvConfig,
    DualArmSuccessRecoveryExpert,
)


@dataclass
class GuidedStartDualArmEnvConfig(DualArmMesh3DEnvConfig):
    guided_start_min_fraction: float = 0.50
    guided_start_max_fraction: float = 0.70
    guided_start_lateral_noise: float = 0.006
    elite_initial_local_offset: float = 0.012
    elite_initial_forward_offset: float = 0.05
    piper_initial_insertion_fraction: float = 0.55


class GuidedStartDualArmGuidewire3DEnv(DualArmGuidewire3DEnv):
    """
    Dual-arm environment variant where the episode starts near the guided stage.

    The guidewire tip is initialized at a configurable progress fraction on the
    chosen left/right path. This intentionally skips the long straight insertion
    segment so Elite is active from the first step.
    """

    def __init__(self, config: Optional[GuidedStartDualArmEnvConfig] = None, seed: Optional[int] = None):
        self.guided_config = config or GuidedStartDualArmEnvConfig()
        super().__init__(self.guided_config, seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        options = options or {}
        obs, info = super().reset(seed=seed, options=options)

        path_len = max(len(self.paths[self.task]) - 1, 1)
        if "start_progress" in options:
            start_progress = float(options["start_progress"])
        else:
            start_fraction = float(
                options.get(
                    "start_fraction",
                    self.rng.uniform(
                        self.guided_config.guided_start_min_fraction,
                        self.guided_config.guided_start_max_fraction,
                    ),
                )
            )
            start_progress = start_fraction * path_len

        self._set_guided_start_progress(start_progress)
        obs_dict = self._obs_dict(done=False, success=False, failure_reason=None)
        return self._gym_obs(obs_dict), self._info(obs_dict)

    def _set_guided_start_progress(self, progress: float) -> None:
        self.path_progress_float = float(np.clip(progress, 0.0, len(self.paths[self.task]) - 1))
        self.path_progress_index = int(np.floor(self.path_progress_float))
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)

        noise = self.rng.normal(0.0, self.guided_config.guided_start_lateral_noise, size=2).astype(np.float32)
        max_noise = max(radius - self.config.wall_clearance, 0.0) * 0.35
        noise_norm = float(np.linalg.norm(noise))
        if noise_norm > max_noise and noise_norm > 1e-6:
            noise = noise / noise_norm * max_noise

        self.local_path_center = center
        self.local_path_tangent = tangent
        self.local_frame_x = n1
        self.local_frame_y = n2
        self.local_radius = radius
        self.lateral_offset = noise.astype(np.float32)
        self.tip = center + n1 * self.lateral_offset[0] + n2 * self.lateral_offset[1]
        self.heading = tangent.copy()
        self.guidewire_points = [self.tip.copy()]

        lookahead = self._guided_lookahead_point()
        lead_local = self._project_local(lookahead - center, n1, n2)
        lead_norm = float(np.linalg.norm(lead_local))
        if lead_norm > 1e-6:
            lead_local = lead_local / lead_norm * self.guided_config.elite_initial_local_offset
        else:
            lead_local = np.zeros(2, dtype=np.float32)

        # Tangential offset records that the magnet is physically ahead of the
        # current tip, while local offset is what affects the simplified physics.
        self.elirobot_pose = (
            center
            + tangent * self.guided_config.elite_initial_forward_offset
            + n1 * lead_local[0]
            + n2 * lead_local[1]
        ).astype(np.float32)

        self.piper_step_count = int(round(self.guided_config.piper_initial_insertion_fraction * self.config.max_steps))
        self.piper_insertion_length = float(self.piper_step_count * self.config.advance_step)
        self.step_count = 0
        self.severe_contact_count = 0
        self.boundary_projection_count = 0
        self.boundary_projection_window = 0
        self.last_boundary_projection = False
        self._sync_robot_state(center, tangent, n1, n2, radius)
        self.robot_state["phase"] = "guided_start"
        self.robot_state["start_progress"] = float(self.path_progress_float)
        self.robot_state["start_fraction"] = float(self.path_progress_float / max(len(self.paths[self.task]) - 1, 1))

    def set_path_progress(self, progress: float):
        self._set_guided_start_progress(progress)

    def _guided_lookahead_point(self) -> np.ndarray:
        path = self.paths[self.task]
        idx = int(np.clip(np.floor(self.path_progress_float), 0, len(path) - 1))
        idx = min(idx + max(self.dual_config.dual_arm_lookahead, 6), len(path) - 1)
        return path[idx]

    def _render_projected_camera(self, size: int, axes) -> np.ndarray:
        img = super()._render_projected_camera(size=size, axes=axes)
        elite_px = self._project_camera_points(self.elirobot_pose[None, :], size=size, axes=axes)[0]
        tip_px = self._project_camera_points(self.tip[None, :], size=size, axes=axes)[0]
        cv2.line(img, tuple(tip_px), tuple(elite_px), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(elite_px), 7, (35, 145, 245), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(elite_px), 9, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _project_camera_points(self, points: np.ndarray, size: int, axes) -> np.ndarray:
        verts_2d = self.vertices[:, axes]
        lo = verts_2d.min(axis=0)
        hi = verts_2d.max(axis=0)
        pad = (hi - lo).max() * 0.06
        lo -= pad
        hi += pad
        pts = np.asarray(points, dtype=np.float32)[:, axes]
        norm = (pts - lo) / np.maximum(hi - lo, 1e-6)
        px = np.column_stack([norm[:, 0] * (size - 1), (1.0 - norm[:, 1]) * (size - 1)])
        return np.clip(px, 0, [size - 1, size - 1]).astype(np.int32)


class GuidedStartDualArmExpert(DualArmSuccessRecoveryExpert):
    """Conservative expert tuned for the guided-start environment."""

    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.006, push_strength: float = 0.7):
        super().__init__(rng_seed=rng_seed, steer_noise=steer_noise, push_strength=push_strength)
