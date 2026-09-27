from __future__ import annotations

import math
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import trimesh
from gymnasium import spaces
from yourdfpy import URDF

from simulation.dual_arm_guided_start_env import GuidedStartDualArmEnvConfig, GuidedStartDualArmGuidewire3DEnv

REPO_ROOT = Path(__file__).resolve().parents[1]


TASK_INSTRUCTIONS = {
    "left": "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the left branch.",
    "right": "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the right branch.",
}


def _vec_to_str(vec: np.ndarray) -> str:
    return " ".join(f"{float(v):.8f}" for v in np.asarray(vec, dtype=np.float32).tolist())


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _quat_to_str(quat: np.ndarray) -> str:
    return " ".join(f"{float(v):.8f}" for v in np.asarray(quat, dtype=np.float32).tolist())


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _quat_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """
    Return a quaternion that rotates the local +Z axis to the given direction.
    MuJoCo expects quaternions in wxyz order.
    """
    direction = _normalize(direction)
    if float(np.linalg.norm(direction)) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -1.0 + 1e-6:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    axis = np.cross(z_axis, direction)
    axis = _normalize(axis)
    angle = math.acos(dot)
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def _parse_rgba(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    values = np.asarray([float(v) for v in text.split()], dtype=np.float32)
    if len(values) == 3:
        values = np.concatenate([values, np.array([1.0], dtype=np.float32)])
    return values


def _joint_values_to_vector(joint_names: list[str], joint_values: dict[str, float] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(joint_values, dict):
        return np.asarray([float(joint_values[name]) for name in joint_names], dtype=np.float32)
    arr = np.asarray(joint_values, dtype=np.float32).reshape(-1)
    if len(arr) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {len(arr)}")
    return arr.astype(np.float32)


def _joint_vector_to_dict(joint_names: list[str], joint_vector: np.ndarray) -> dict[str, float]:
    joint_vector = np.asarray(joint_vector, dtype=np.float32).reshape(-1)
    if len(joint_vector) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {len(joint_vector)}")
    return {name: float(joint_vector[index]) for index, name in enumerate(joint_names)}


def _project_vector_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    normal = _normalize(normal)
    return vector - float(np.dot(vector, normal)) * normal


def _rpy_to_quat_attr(rpy: tuple[float, float, float]) -> str:
    return _quat_to_str(_quat_from_rpy(*rpy))


def _to_urdf_path(text: str) -> Path:
    return (REPO_ROOT / text).resolve()


def _parse_xyz_rpy(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    xyz_vals = origin.attrib.get("xyz", "0 0 0")
    rpy_vals = origin.attrib.get("rpy", "0 0 0")
    xyz = np.asarray([float(v) for v in xyz_vals.split()], dtype=np.float32)
    rpy = np.asarray([float(v) for v in rpy_vals.split()], dtype=np.float32)
    return xyz, rpy


def _mesh_path_from_urdf(urdf_path: Path, filename: str) -> Path:
    mesh = Path(filename)
    if mesh.is_absolute():
        return mesh
    return (urdf_path.parent / mesh).resolve()


def _load_urdf_joint_info(urdf_path: Path) -> list[dict]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_info = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type == "fixed":
            continue
        name = joint.attrib["name"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz, rpy = _parse_xyz_rpy(origin)
        axis = joint.find("axis")
        axis_xyz = np.asarray([float(v) for v in axis.attrib.get("xyz", "0 0 1").split()], dtype=np.float32) if axis is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        limit = joint.find("limit")
        lower = float(limit.attrib.get("lower", "-3.1416")) if limit is not None else -3.1416
        upper = float(limit.attrib.get("upper", "3.1416")) if limit is not None else 3.1416
        joint_info.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis_xyz,
                "lower": lower,
                "upper": upper,
            }
        )
    return joint_info


def _load_urdf_link_visuals(urdf_path: Path) -> dict[str, list[dict]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    visuals: dict[str, list[dict]] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        entries = []
        for idx, visual in enumerate(link.findall("visual")):
            origin = visual.find("origin")
            xyz, rpy = _parse_xyz_rpy(origin)
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename")
            if not filename:
                continue
            scale = mesh_el.attrib.get("scale")
            rgba = None
            color_el = visual.find("material/color")
            if color_el is not None:
                rgba = color_el.attrib.get("rgba")
            entries.append(
                {
                    "name": f"{link_name}_vis{idx}",
                    "mesh": filename,
                    "xyz": xyz,
                    "rpy": rpy,
                    "scale": scale,
                    "rgba": rgba,
                }
            )
        visuals[link_name] = entries
    return visuals


def _load_urdf_joint_tree(urdf_path: Path) -> dict:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    children: dict[str, list[dict]] = {}
    child_links = set()
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin_xyz, origin_rpy = _parse_xyz_rpy(joint.find("origin"))
        axis_el = joint.find("axis")
        axis = np.asarray([float(v) for v in axis_el.attrib.get("xyz", "0 0 1").split()], dtype=np.float32) if axis_el is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        limit_el = joint.find("limit")
        lower = float(limit_el.attrib.get("lower", "-3.1416")) if limit_el is not None else -3.1416
        upper = float(limit_el.attrib.get("upper", "3.1416")) if limit_el is not None else 3.1416
        item = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": origin_xyz,
            "rpy": origin_rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }
        joints.append(item)
        children.setdefault(parent, []).append(item)
        child_links.add(child)
    links = [link.attrib["name"] for link in root.findall("link")]
    root_link = "base_link" if "base_link" in links else next((link for link in links if link not in child_links), links[0])
    return {"root_link": root_link, "joints": joints, "children": children, "links": links}


@dataclass
class MuJoCoGuidedWireConfig(GuidedStartDualArmEnvConfig):
    route_config_path: str = "simulation/routes/vessel_0422_wire_route_v1.json"
    scene_config_path: str = "simulation_output/robot_scene_mvp/scene_config.json"
    camera_config_path: str = "simulation_output/mujoco_camera_config.json"
    robot_mesh_manifest: str = "simulation_output/mujoco_robot_assets/manifest.json"
    robot_link_manifest: str = "simulation_output/mujoco_robot_link_assets/manifest.json"
    use_scene_scale: bool = True
    show_robots: bool = True
    robot_visual_mode: str = "kinematic"  # "kinematic", "static", or "none"
    show_vessel_mesh: bool = True
    show_path_tubes: bool = False
    show_tool_markers: bool = False
    render_overlay: bool = False
    route_picker_markers: bool = False
    route_picker_marker_task: str = "left"
    route_picker_marker_stride: float = 0.5
    route_picker_marker_size: float = 0.0022
    route_picker_marker_alpha: float = 0.9
    wire_visual_mode: str = "segments"  # "segments", "line", or "both"
    wire_visual_radius: float = 0.0025
    wire_visual_rgb: str = "0.06 0.56 1.0"
    wire_tip_visual_rgb: str = "0.06 0.56 1.0"
    wire_tip_visual_segments: int = 0
    wire_tip_visual_radius_scale: float = 1.0
    wire_tip_marker_radius: float = 0.0
    wire_tip_marker_alpha: float = 0.0
    wire_visual_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wire_line_width: int = 5
    wire_line_core_width: int = 1
    guidance_mode: str = "mvp_autopilot"  # "mvp_autopilot" or "physical"
    vessel_mesh_asset: str = "simulation_output/mujoco_vessel_assets/vessel_0422.obj"
    table_texture_asset: str = "simulation_output/mujoco_material_assets/table_wood_soft.png"
    guided_start_lateral_noise: float = 0.0025
    elirobot_action_scale: float = 0.024
    piper_insertion_window: float = 0.045
    path_search_backtrack: int = 12
    path_search_ahead: int = 24
    wire_segments: int = 34
    wire_spacing: float = 0.026
    wire_radius: float = 0.0075
    wire_mass: float = 0.002
    sim_substeps: int = 10
    sim_dt: float = 0.004
    spring_k: float = 220.0
    bend_k: float = 18.0
    damping: float = 1.6
    piper_force: float = 0.42
    piper_advection_scale: float = 0.44
    piper_primitive_steps: int = 1
    piper_primitive_feed_value: float = 1.0
    piper_primitive_retract_value: float = 1.0
    magnet_force: float = 0.14
    magnet_range: float = 0.23
    magnet_tip_height_offset: float = 0.12
    tip_centering_force: float = 2.20
    wire_centering_force: float = 0.72
    wire_centerline_relaxation: float = 0.10
    wire_scaffold_relaxation: float = 0.55
    wire_tip_offset_decay_segments: float = 7.0
    wire_max_turn_degrees: float = 70.0
    physical_tail_anchor_segments: int = 6
    physical_curvature_smoothing: float = 0.38
    physical_curvature_iterations: int = 4
    physical_tip_stiff_segments: int = 8
    physical_tip_curvature_smoothing: float = 0.86
    physical_tip_curvature_iterations: int = 4
    wire_max_speed: float = 1.2
    branch_centering_start_fraction: float = 0.66
    branch_centering_ramp_fraction: float = 0.12
    branch_centering_force_scale: float = 2.45
    branch_tip_centering_force_scale: float = 1.90
    branch_magnet_lateral_scale: float = 0.52
    branch_relaxation_scale: float = 2.10
    tip_path_drive_force: float = 0.042
    gravity_z: float = -9.81
    commanded_progress_gain: float = 0.17
    tip_drive_lookahead: int = 4
    elite_motion_smoothing: float = 0.52
    elite_joint_rate_limit: float = 0.0
    elite_joint_accel_limit: float = 0.0
    elite_joint_jerk_limit: float = 0.0
    wall_k: float = 260.0
    wall_damping: float = 1.8
    progress_gain: float = 0.58
    boundary_projection_limit: float = 0.94
    boundary_projection_relief: float = 2.20
    repeated_projection_limit: int = 36
    vessel_radius_scale: float = 1.20
    contact_projection_threshold: float = 4.00
    contact_relief_ratio: float = 0.60
    wall_contact_failure_margin: float = 2.8
    wall_contact_failure_window: int = 28
    render_width: int = 640
    render_height: int = 480
    camera_roi_margin: float = 0.055
    camera_distance_scale: float = 0.86
    camera_min_distance: float = 0.38
    render_domain_preset: str = "default"
    vessel_visual_stride: int = 3
    vessel_visual_radius_scale: float = 1.0
    render_observation: bool = False
    target_radius: float = 0.55


class MuJoCoGuidedWireEnv(gym.Env):
    """
    Lightweight MuJoCo MVP for the guided-start dual-arm task.

    MuJoCo is used for the scene graph, rendering, and physical object layout.
    The first MVP keeps the guidewire integration explicit in Python so we can
    iterate on vessel-wall and magnetic-force assumptions without fighting MJCF
    constraint tuning too early.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[MuJoCoGuidedWireConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config or MuJoCoGuidedWireConfig()
        self.rng = np.random.default_rng(seed)
        self.reference_env = GuidedStartDualArmGuidewire3DEnv(self.config, seed=seed)
        self.scene_config = self._load_scene_config()
        self.camera_config = self._load_camera_config()
        self.scene_scale = float(self.scene_config.get("vessel_scale", 1.0)) if self.config.use_scene_scale else 1.0
        self.raw_scene_center = 0.5 * (self.reference_env.scene_min.astype(np.float32) + self.reference_env.scene_max.astype(np.float32))
        self.wire_centerlines = {task: self._scene_point(path.astype(np.float32)) for task, path in self.reference_env.paths.items()}
        self.paths = self.wire_centerlines
        self.path_step_length = {
            task: float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            for task, path in self.paths.items()
        }
        self.radii = {task: radii.astype(np.float32) * self.scene_scale * self.config.vessel_radius_scale for task, radii in self.reference_env.radii.items()}
        self.targets = {task: self._scene_point(target.astype(np.float32)) for task, target in self.reference_env.targets.items()}
        self.scene_min = self._scene_point(self.reference_env.scene_min.astype(np.float32))
        self.scene_max = self._scene_point(self.reference_env.scene_max.astype(np.float32))
        self._scale_physics_params()
        self.elite_reference_paths = self._build_elite_reference_paths()
        self.elite_path_targets = {task: path[-1].copy() for task, path in self.elite_reference_paths.items()}
        assets_root = (REPO_ROOT / self.scene_config.get("assets_root", "robot_assets/standardized")).resolve()
        self.piper_urdf_path = (assets_root / self.scene_config.get("piper_urdf", "piper_description/urdf/piper_description.urdf")).resolve()
        self.elite_urdf_path = (assets_root / self.scene_config.get("elite_urdf", "elite_description/urdf/ec66_description.urdf")).resolve()
        self.piper_joint_info = _load_urdf_joint_info(self.piper_urdf_path)
        self.elite_joint_info = _load_urdf_joint_info(self.elite_urdf_path)
        self.piper_joint_names = [item["name"] for item in self.piper_joint_info]
        self.elite_joint_names = [item["name"] for item in self.elite_joint_info]
        self.piper_joint_limits = {item["name"]: (float(item["lower"]), float(item["upper"])) for item in self.piper_joint_info}
        self.elite_joint_limits = {item["name"]: (float(item["lower"]), float(item["upper"])) for item in self.elite_joint_info}
        self.piper_joint_lower = np.asarray([self.piper_joint_limits[name][0] for name in self.piper_joint_names], dtype=np.float32)
        self.piper_joint_upper = np.asarray([self.piper_joint_limits[name][1] for name in self.piper_joint_names], dtype=np.float32)
        self.elite_joint_lower = np.asarray([self.elite_joint_limits[name][0] for name in self.elite_joint_names], dtype=np.float32)
        self.elite_joint_upper = np.asarray([self.elite_joint_limits[name][1] for name in self.elite_joint_names], dtype=np.float32)
        self.piper_robot_model = URDF.load(str(self.piper_urdf_path))
        self.elite_robot_model = URDF.load(str(self.elite_urdf_path))
        self.piper_visuals = _load_urdf_link_visuals(self.piper_urdf_path)
        self.elite_visuals = _load_urdf_link_visuals(self.elite_urdf_path)
        self.piper_joint_values = self._default_joint_values(self.piper_joint_info, self.scene_config.get("piper_joints", {}))
        self.elite_joint_values = self._default_joint_values(self.elite_joint_info, self.scene_config.get("elite_joints", {}))
        self.piper_home_joint_vector = np.asarray([self.piper_joint_values[name] for name in self.piper_joint_names], dtype=np.float32)
        self.elite_home_joint_vector = np.asarray([self.elite_joint_values[name] for name in self.elite_joint_names], dtype=np.float32)
        self.robot_mesh_manifest = self._load_robot_mesh_manifest()
        self.vessel_mesh_asset = self._prepare_vessel_mesh_asset()
        self.table_texture_asset = self._prepare_table_texture_asset()
        self.task = "left"
        self.instruction = TASK_INSTRUCTIONS[self.task]

        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer: Optional[mujoco.Renderer] = None
        self.camera = mujoco.MjvCamera()
        self._cache_joint_addresses()
        self.joint_action_dim = len(self.piper_joint_names) + len(self.elite_joint_names)

        self.action_space = spaces.Dict(
            {
                "piper_joints": spaces.Box(low=self.piper_joint_lower, high=self.piper_joint_upper, dtype=np.float32),
                "elite_joints": spaces.Box(low=self.elite_joint_lower, high=self.elite_joint_upper, dtype=np.float32),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(self.config.render_height, self.config.render_width, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(34,), dtype=np.float32),
                "instruction_id": spaces.Discrete(2),
            }
        )

        self.positions = np.zeros((self.config.wire_segments, 3), dtype=np.float32)
        self.velocities = np.zeros_like(self.positions)
        self.elite_pose = np.zeros(3, dtype=np.float32)
        self.magnetic_pose = np.zeros(3, dtype=np.float32)
        self._elite_target_smoothed = np.zeros(3, dtype=np.float32)
        self._elite_previous_joint_delta = np.zeros(0, dtype=np.float32)
        self._elite_previous_joint_accel = np.zeros(0, dtype=np.float32)
        self.piper_pose = np.zeros(3, dtype=np.float32)
        self.tip = np.zeros(3, dtype=np.float32)
        self.heading = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.path_progress_float = 0.0
        self.path_progress_index = 0
        self.step_count = 0
        self.boundary_projection_count = 0
        self.boundary_projection_window = 0
        self._tip_projected_this_step = False
        self.severe_contact_count = 0
        self.guidewire_points = []
        self.last_tactile = {}

    def _effective_magnetic_pose(self) -> np.ndarray:
        return self.elite_pose.astype(np.float32).copy()

    def _front_up_magnetic_target(
        self,
        anchor: np.ndarray,
        forward_direction: np.ndarray,
        *,
        forward_offset: float,
        height_offset: float | None = None,
    ) -> np.ndarray:
        forward = _normalize(np.asarray(forward_direction, dtype=np.float32).reshape(3))
        if float(np.linalg.norm(forward)) < 1e-8:
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        height = float(self.config.magnet_tip_height_offset if height_offset is None else height_offset)
        return (
            np.asarray(anchor, dtype=np.float32).reshape(3)
            + forward * float(forward_offset)
            + up * max(height, 0.0)
        ).astype(np.float32)

    def _build_elite_reference_paths(self) -> dict[str, np.ndarray]:
        elite_paths: dict[str, np.ndarray] = {}
        for task, centerline in self.wire_centerlines.items():
            path = centerline.copy()
            if len(path) < 2:
                elite_paths[task] = path
                continue
            offset_path = []
            for idx, point in enumerate(path):
                tangent = self._path_tangent(path, idx)
                tangent_offset = min(float(self.radii[task][idx]) * 0.18, 0.012 * self.scene_scale)
                offset_path.append(point + tangent * tangent_offset)
            elite_paths[task] = np.asarray(offset_path, dtype=np.float32)
        return elite_paths

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

    @staticmethod
    def _default_joint_values(joint_info: list[dict], config_values: dict) -> dict[str, float]:
        result = {}
        for info in joint_info:
            value = float(config_values.get(info["name"], 0.0))
            result[info["name"]] = float(np.clip(value, info["lower"], info["upper"]))
        return result

    def _scene_point(self, point: np.ndarray) -> np.ndarray:
        return (np.asarray(point, dtype=np.float32) - self.raw_scene_center) * self.scene_scale

    def set_robot_joints(self, robot: str, values: dict[str, float]) -> None:
        target = self.piper_joint_values if robot == "piper" else self.elite_joint_values
        limits = self.piper_joint_limits if robot == "piper" else self.elite_joint_limits
        for name, value in values.items():
            if name not in target:
                continue
            lower, upper = limits.get(name, (-np.inf, np.inf))
            target[name] = float(np.clip(value, lower, upper))
        self._sync_mujoco()

    def _robot_joint_state(self, robot: str) -> tuple[list[str], dict[str, float], dict[str, tuple[float, float]], URDF, list[float], str, list[float]]:
        if robot == "piper":
            return (
                self.piper_joint_names,
                self.piper_joint_values,
                self.piper_joint_limits,
                self.piper_robot_model,
                self.scene_config.get("piper_pose", [-0.28, -0.137, -0.08, 0.0, 0.0, 0.0]),
                self.scene_config.get("piper_tool_link", "gripper_base"),
                self.scene_config.get("piper_tool_offset", [0.0, 0.0, 0.0]),
            )
        if robot == "elite":
            return (
                self.elite_joint_names,
                self.elite_joint_values,
                self.elite_joint_limits,
                self.elite_robot_model,
                self.scene_config.get("elite_pose", [-0.294, 0.133, -0.08, 0.0, 0.0, 3.141593]),
                self.scene_config.get("elite_tool_link", "flan"),
                self.scene_config.get("elite_magnet_offset", [0.0, 0.0, 0.0]),
            )
        raise ValueError(f"Unknown robot: {robot}")

    def _robot_joint_vector(self, robot: str) -> np.ndarray:
        joint_names, values, _limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        return np.asarray([float(values[name]) for name in joint_names], dtype=np.float32)

    def _robot_joint_vector_from_action(self, robot: str, value: dict[str, float] | list[float] | np.ndarray) -> np.ndarray:
        joint_names, _values, _limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        return _joint_values_to_vector(joint_names, value)

    def _robot_joint_dict(self, robot: str, joint_vector: np.ndarray) -> dict[str, float]:
        joint_names, _values, _limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        return _joint_vector_to_dict(joint_names, joint_vector)

    def _clip_robot_joint_vector(self, robot: str, joint_vector: np.ndarray) -> np.ndarray:
        joint_names, _values, limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        joint_vector = np.asarray(joint_vector, dtype=np.float32).reshape(-1).copy()
        if len(joint_vector) != len(joint_names):
            raise ValueError(f"{robot} action expected {len(joint_names)} joints, got {len(joint_vector)}")
        for index, name in enumerate(joint_names):
            lower, upper = limits.get(name, (-np.inf, np.inf))
            joint_vector[index] = float(np.clip(joint_vector[index], lower, upper))
        return joint_vector.astype(np.float32)

    def _piper_feed_signal(self, joint_vector: np.ndarray) -> float:
        joint_vector = np.asarray(joint_vector, dtype=np.float32).reshape(-1)
        if "joint7" not in self.piper_joint_names or "joint8" not in self.piper_joint_names:
            return 0.0
        idx7 = self.piper_joint_names.index("joint7")
        idx8 = self.piper_joint_names.index("joint8")
        return float(0.5 * (joint_vector[idx7] - joint_vector[idx8]))

    def _piper_joint_vector_from_feed(self, feed_signal: float) -> np.ndarray:
        joint_vector = self.piper_home_joint_vector.copy()
        if "joint7" in self.piper_joint_names:
            idx7 = self.piper_joint_names.index("joint7")
            lower, upper = self.piper_joint_limits.get("joint7", (0.0, 0.035))
            joint_vector[idx7] = float(np.clip(feed_signal, lower, upper))
        if "joint8" in self.piper_joint_names:
            idx8 = self.piper_joint_names.index("joint8")
            lower, upper = self.piper_joint_limits.get("joint8", (-0.035, 0.0))
            joint_vector[idx8] = float(np.clip(-feed_signal, lower, upper))
        return self._clip_robot_joint_vector("piper", joint_vector)

    def _piper_joint_vector_from_feed_command(self, feed_command: float) -> np.ndarray:
        feed_command = float(np.clip(feed_command, -1.0, 1.0))
        current_signal = float(getattr(self, "piper_feed_signal", self._piper_feed_signal(self.piper_home_joint_vector)))
        home_signal = float(getattr(self, "piper_feed_home_signal", self._piper_feed_signal(self.piper_home_joint_vector)))
        joint_signal_limit = min(
            float(self.piper_joint_limits.get("joint7", (0.0, 0.035))[1]),
            -float(self.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]),
        )
        insertion_limit = min(float(self.config.piper_insertion_window), max(joint_signal_limit - home_signal, 1e-6))
        min_signal = home_signal - 0.18 * joint_signal_limit
        max_signal = home_signal + insertion_limit
        feed_delta = feed_command * float(self.config.advance_step) * float(self.config.piper_advection_scale)
        return self._piper_joint_vector_from_feed(float(np.clip(current_signal + feed_delta, min_signal, max_signal)))

    def _piper_joint_vector_from_step_command(self, step_command: float) -> np.ndarray:
        step_command = float(np.clip(round(float(step_command)), -1.0, 1.0))
        return self._piper_joint_vector_from_feed_command(step_command)

    def _piper_step_command_override(self, action) -> Optional[float]:
        if not isinstance(action, dict):
            return None
        if "piper_step_command" not in action:
            return None
        if "piper_feed_command" in action:
            feed_command = float(np.asarray(action.get("piper_feed_command", 0.0), dtype=np.float32).reshape(-1)[0])
            return float(np.clip(feed_command, -1.0, 1.0))
        if "piper_feed" in action:
            feed_command = float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0])
            return float(np.clip(feed_command, -1.0, 1.0))
        step_command = float(np.asarray(action.get("piper_step_command", 0.0), dtype=np.float32).reshape(-1)[0])
        return float(np.clip(round(step_command), -1.0, 1.0))

    def _piper_policy_command_from_action(self, action, piper_cmd: float) -> float:
        if not isinstance(action, dict):
            return float(np.clip(piper_cmd, -1.0, 1.0))
        if "piper_step_command" in action:
            step_command = float(np.asarray(action.get("piper_step_command", 0.0), dtype=np.float32).reshape(-1)[0])
            return float(np.clip(round(step_command), -1.0, 1.0))
        if "piper_feed" in action:
            feed_command = float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0])
            return float(np.clip(feed_command, -1.0, 1.0))
        if "piper_feed_command" in action:
            feed_command = float(np.asarray(action.get("piper_feed_command", 0.0), dtype=np.float32).reshape(-1)[0])
            return float(np.clip(feed_command, -1.0, 1.0))
        return float(np.clip(piper_cmd, -1.0, 1.0))

    def _piper_controller_execute(self, policy_command: float) -> tuple[float, bool]:
        primitive_steps = max(int(getattr(self.config, "piper_primitive_steps", 1)), 1)
        if primitive_steps <= 1:
            return float(np.clip(policy_command, -1.0, 1.0)), False

        remaining = int(getattr(self, "_piper_primitive_remaining_steps", 0))
        active_command = float(getattr(self, "_piper_primitive_active_command", 0.0))
        policy_command = float(np.clip(policy_command, -1.0, 1.0))
        if remaining > 0:
            executed_command = active_command
            remaining -= 1
            if remaining <= 0:
                active_command = 0.0
            self._piper_primitive_remaining_steps = int(remaining)
            self._piper_primitive_active_command = float(active_command)
            return float(np.clip(executed_command, -1.0, 1.0)), remaining > 0
        if abs(policy_command) > 0.05 and remaining <= 0:
            if policy_command > 0.0:
                magnitude = abs(float(getattr(self.config, "piper_primitive_feed_value", 1.0)))
            else:
                magnitude = abs(float(getattr(self.config, "piper_primitive_retract_value", 1.0)))
            active_command = float(np.sign(policy_command) * np.clip(magnitude, 0.0, 1.0))
            remaining = primitive_steps
        executed_command = active_command if remaining > 0 else 0.0
        if remaining > 0:
            remaining -= 1
        if remaining <= 0:
            active_command = 0.0
        self._piper_primitive_remaining_steps = int(remaining)
        self._piper_primitive_active_command = float(active_command)
        return float(np.clip(executed_command, -1.0, 1.0)), remaining > 0

    @staticmethod
    def _piper_command_label(command: float) -> str:
        if float(command) > 0.05:
            return "feed"
        if float(command) < -0.05:
            return "retract"
        return "hold"

    def _controller_request_from_action(self, action) -> Dict:
        request: Dict[str, object] = {
            "piper_intent": None,
            "piper_requested_feed": None,
            "elite_action_type": "unknown",
            "elite_tcp_delta_6d": None,
            "elite_tcp_pose_6d": None,
        }
        if not isinstance(action, dict):
            return request
        if "piper_step_command" in action:
            step_command = float(np.asarray(action.get("piper_step_command", 0.0), dtype=np.float32).reshape(-1)[0])
            step_command = float(np.clip(round(step_command), -1.0, 1.0))
            request["piper_intent"] = self._piper_command_label(step_command)
        if "piper_feed" in action:
            request["piper_requested_feed"] = float(np.clip(float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0]), -1.0, 1.0))
        elif "piper_feed_command" in action:
            request["piper_requested_feed"] = float(np.clip(float(np.asarray(action.get("piper_feed_command", 0.0), dtype=np.float32).reshape(-1)[0]), -1.0, 1.0))
        if request["piper_intent"] is None and request["piper_requested_feed"] is not None:
            request["piper_intent"] = self._piper_command_label(float(request["piper_requested_feed"]))
        if "elite_tcp_delta_6d" in action:
            request["elite_action_type"] = "tcp_delta"
            request["elite_tcp_delta_6d"] = np.asarray(action["elite_tcp_delta_6d"], dtype=np.float32).reshape(6).astype(float).tolist()
        elif "elite_tcp_pose_6d" in action:
            request["elite_action_type"] = "tcp_pose"
        elif "elite_joints" in action:
            request["elite_action_type"] = "joints"
        if "elite_tcp_pose_6d" in action:
            request["elite_tcp_pose_6d"] = np.asarray(action["elite_tcp_pose_6d"], dtype=np.float32).reshape(6).astype(float).tolist()
        return request

    def _set_robot_joint_vector(self, robot: str, joint_vector: np.ndarray, sync: bool = False) -> None:
        joint_names, values, limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        for index, name in enumerate(joint_names):
            lower, upper = limits.get(name, (-np.inf, np.inf))
            values[name] = float(np.clip(float(joint_vector[index]), lower, upper))
        if sync:
            self._sync_mujoco()

    @staticmethod
    def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
        rot = np.asarray(matrix, dtype=float)[:3, :3]
        sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
        if sy > 1e-8:
            roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
            pitch = float(np.arctan2(-rot[2, 0], sy))
            yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        else:
            roll = float(np.arctan2(-rot[1, 2], rot[1, 1]))
            pitch = float(np.arctan2(-rot[2, 0], sy))
            yaw = 0.0
        return np.asarray([roll, pitch, yaw], dtype=np.float32)

    def _robot_tool_transform_from_vector(self, robot: str, joint_vector: np.ndarray) -> np.ndarray:
        joint_names, _values, limits, model, base_pose, tool_link, offset = self._robot_joint_state(robot)
        joint_cfg = {}
        for index, name in enumerate(joint_names):
            lower, upper = limits.get(name, (-np.inf, np.inf))
            joint_cfg[name] = float(np.clip(float(joint_vector[index]), lower, upper))
        model.update_cfg(joint_cfg)
        tf = np.asarray(model.get_transform(tool_link), dtype=float)
        base_tf = self._pose_matrix(base_pose)
        return base_tf @ tf @ self._translation_matrix(offset)

    def _robot_tool_world_from_vector(self, robot: str, joint_vector: np.ndarray) -> np.ndarray:
        point = self._robot_tool_transform_from_vector(robot, joint_vector)
        return point[:3, 3].astype(np.float32)

    def _robot_tool_pose6d_from_vector(self, robot: str, joint_vector: np.ndarray) -> np.ndarray:
        transform = self._robot_tool_transform_from_vector(robot, joint_vector)
        xyz_mm = transform[:3, 3].astype(np.float32) * 1000.0
        rpy = self._matrix_to_rpy(transform)
        return np.concatenate([xyz_mm, rpy]).astype(np.float32)

    def robot_tool_world(self, robot: str) -> np.ndarray:
        joint_vector = self._robot_joint_vector(robot)
        return self._robot_tool_world_from_vector(robot, joint_vector)

    def robot_tool_pose6d(self, robot: str) -> np.ndarray:
        joint_vector = self._robot_joint_vector(robot)
        return self._robot_tool_pose6d_from_vector(robot, joint_vector)

    def _solve_robot_ik(
        self,
        robot: str,
        target_world: np.ndarray,
        *,
        max_iters: int = 8,
        damping: float = 0.04,
        step_limit: float = 0.25,
        eps: float = 1e-3,
    ) -> dict[str, float]:
        joint_names, _values, limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        q = self._robot_joint_vector(robot).astype(np.float32)
        target_world = np.asarray(target_world, dtype=np.float32).reshape(3)
        if len(joint_names) == 0:
            return {}
        for _ in range(max(int(max_iters), 1)):
            current = self._robot_tool_world_from_vector(robot, q)
            error = target_world - current
            if float(np.linalg.norm(error)) < 1e-4:
                break
            jac = np.zeros((3, len(joint_names)), dtype=np.float32)
            for index, name in enumerate(joint_names):
                lower, upper = limits.get(name, (-np.inf, np.inf))
                q_perturbed = q.copy()
                q_perturbed[index] = float(np.clip(q_perturbed[index] + eps, lower, upper))
                if abs(float(q_perturbed[index] - q[index])) < 1e-8:
                    q_perturbed[index] = float(np.clip(q[index] - eps, lower, upper))
                denom = float(q_perturbed[index] - q[index])
                if abs(denom) < 1e-8:
                    continue
                perturbed = self._robot_tool_world_from_vector(robot, q_perturbed)
                jac[:, index] = (perturbed - current) / denom
            lhs = jac @ jac.T + (damping ** 2) * np.eye(3, dtype=np.float32)
            try:
                step = jac.T @ np.linalg.solve(lhs, error.astype(np.float32))
            except np.linalg.LinAlgError:
                step = jac.T @ np.linalg.lstsq(lhs, error.astype(np.float32), rcond=None)[0]
            step_norm = float(np.linalg.norm(step))
            if step_norm > step_limit:
                step = step * (step_limit / max(step_norm, 1e-6))
            q = q + step.astype(np.float32)
            for index, name in enumerate(joint_names):
                lower, upper = limits.get(name, (-np.inf, np.inf))
                q[index] = float(np.clip(q[index], lower, upper))
        return {name: float(q[index]) for index, name in enumerate(joint_names)}

    @staticmethod
    def _wrap_angle_error(values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    @staticmethod
    def _pose6d_measurement(pose6d: np.ndarray, rotation_weight: float) -> np.ndarray:
        pose6d = np.asarray(pose6d, dtype=np.float32).reshape(6)
        xyz_m = pose6d[:3] / 1000.0
        rpy = pose6d[3:] * float(rotation_weight)
        return np.concatenate([xyz_m, rpy]).astype(np.float32)

    def _pose6d_error(self, target_pose6d: np.ndarray, current_pose6d: np.ndarray, rotation_weight: float) -> np.ndarray:
        target_pose6d = np.asarray(target_pose6d, dtype=np.float32).reshape(6)
        current_pose6d = np.asarray(current_pose6d, dtype=np.float32).reshape(6)
        pos_error = (target_pose6d[:3] - current_pose6d[:3]) / 1000.0
        rot_error = self._wrap_angle_error(target_pose6d[3:] - current_pose6d[3:]) * float(rotation_weight)
        return np.concatenate([pos_error, rot_error]).astype(np.float32)

    def _solve_robot_pose6d_ik(
        self,
        robot: str,
        target_pose6d: np.ndarray,
        *,
        max_iters: int = 10,
        damping: float = 0.04,
        step_limit: float = 0.22,
        eps: float = 1e-3,
        rotation_weight: float = 0.08,
    ) -> dict[str, float]:
        joint_names, _values, limits, _model, _base_pose, _tool_link, _offset = self._robot_joint_state(robot)
        q = self._robot_joint_vector(robot).astype(np.float32)
        target_pose6d = np.asarray(target_pose6d, dtype=np.float32).reshape(6)
        if len(joint_names) == 0:
            return {}
        for _ in range(max(int(max_iters), 1)):
            current_pose = self._robot_tool_pose6d_from_vector(robot, q)
            error = self._pose6d_error(target_pose6d, current_pose, rotation_weight)
            if float(np.linalg.norm(error[:3])) < 1e-4 and float(np.linalg.norm(error[3:])) < 1e-3:
                break
            current_measure = self._pose6d_measurement(current_pose, rotation_weight)
            jac = np.zeros((6, len(joint_names)), dtype=np.float32)
            for index, name in enumerate(joint_names):
                lower, upper = limits.get(name, (-np.inf, np.inf))
                q_perturbed = q.copy()
                q_perturbed[index] = float(np.clip(q_perturbed[index] + eps, lower, upper))
                if abs(float(q_perturbed[index] - q[index])) < 1e-8:
                    q_perturbed[index] = float(np.clip(q[index] - eps, lower, upper))
                denom = float(q_perturbed[index] - q[index])
                if abs(denom) < 1e-8:
                    continue
                perturbed_pose = self._robot_tool_pose6d_from_vector(robot, q_perturbed)
                perturbed_measure = self._pose6d_measurement(perturbed_pose, rotation_weight)
                diff = perturbed_measure - current_measure
                diff[3:] = self._wrap_angle_error(perturbed_pose[3:] - current_pose[3:]) * float(rotation_weight)
                jac[:, index] = diff / denom
            lhs = jac @ jac.T + (damping ** 2) * np.eye(6, dtype=np.float32)
            try:
                step = jac.T @ np.linalg.solve(lhs, error.astype(np.float32))
            except np.linalg.LinAlgError:
                step = jac.T @ np.linalg.lstsq(lhs, error.astype(np.float32), rcond=None)[0]
            step_norm = float(np.linalg.norm(step))
            if step_norm > step_limit:
                step = step * (step_limit / max(step_norm, 1e-6))
            q = q + step.astype(np.float32)
            for index, name in enumerate(joint_names):
                lower, upper = limits.get(name, (-np.inf, np.inf))
                q[index] = float(np.clip(q[index], lower, upper))
        return {name: float(q[index]) for index, name in enumerate(joint_names)}

    def _elite_joint_vector_from_tcp_action(self, action: dict) -> np.ndarray:
        current_pose = self.robot_tool_pose6d("elite").astype(np.float32)
        if "elite_tcp_pose_6d" in action:
            target_pose = np.asarray(action["elite_tcp_pose_6d"], dtype=np.float32).reshape(6)
        elif "elite_tcp_delta_6d" in action:
            delta = np.asarray(action["elite_tcp_delta_6d"], dtype=np.float32).reshape(6)
            target_pose = current_pose.copy()
            target_pose[:3] = target_pose[:3] + delta[:3]
            target_pose[3:] = target_pose[3:] + delta[3:]
        else:
            raise ValueError("Elite TCP action requires elite_tcp_pose_6d or elite_tcp_delta_6d")
        joints = self._solve_robot_pose6d_ik(
            "elite",
            target_pose,
            max_iters=10,
            damping=0.04,
            step_limit=0.22,
            rotation_weight=0.08,
        )
        return self._robot_joint_vector_from_action("elite", joints)

    def _refresh_robot_tool_poses(self) -> None:
        self.piper_pose = self.robot_tool_world("piper")
        self.elite_pose = self.robot_tool_world("elite")
        self.piper_tcp_pose_6d = self.robot_tool_pose6d("piper")
        self.elite_tcp_pose_6d = self.robot_tool_pose6d("elite")

    @staticmethod
    def _pose_matrix(values: list[float]) -> np.ndarray:
        x, y, z, roll, pitch, yaw = [float(v) for v in values]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = rz @ ry @ rx
        mat[:3, 3] = [x, y, z]
        return mat

    @staticmethod
    def _translation_matrix(values: list[float]) -> np.ndarray:
        mat = np.eye(4, dtype=float)
        mat[:3, 3] = np.asarray(values, dtype=float)
        return mat

    def _load_scene_config(self) -> dict:
        path = (REPO_ROOT / self.config.scene_config_path).resolve()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_camera_config(self) -> dict:
        path = (REPO_ROOT / self.config.camera_config_path).resolve()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _robot_visual_color(robot_name: str, rgba: Optional[str], depth: int = 0) -> str:
        parsed = _parse_rgba(rgba)
        if parsed is not None and float(np.max(parsed[:3])) < 0.88:
            return " ".join(f"{float(v):.6f}" for v in parsed.tolist())
        if robot_name == "piper":
            base = np.array([0.61, 0.71, 0.84, 1.0], dtype=np.float32)
        else:
            base = np.array([0.86, 0.78, 0.68, 1.0], dtype=np.float32)
        shade = 0.018 * float((depth % 3) - 1)
        base[:3] = np.clip(base[:3] + shade, 0.0, 1.0)
        return " ".join(f"{float(v):.6f}" for v in base.tolist())

    def _prepare_vessel_mesh_asset(self) -> Optional[Path]:
        if not self.config.show_vessel_mesh:
            return None
        source = (REPO_ROOT / self.scene_config.get("vessel", "utils/interface/model/0422.stl")).resolve()
        if not source.exists():
            return None
        out_path = (REPO_ROOT / self.config.vessel_mesh_asset).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return out_path
        vessel = trimesh.load(source, force="mesh")
        if not isinstance(vessel, trimesh.Trimesh):
            return None
        vessel = vessel.copy()
        center = vessel.bounds.mean(axis=0)
        vessel.apply_translation(-center)
        vessel.apply_scale(float(self.scene_scale))
        vessel.export(out_path)
        return out_path

    def _prepare_table_texture_asset(self) -> Path:
        out_path = (REPO_ROOT / self.config.table_texture_asset).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return out_path

        width, height = 1024, 512
        rng = np.random.default_rng(20260531)
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        grain = (
            0.28 * np.sin((xx * 15.0 + 0.55 * np.sin(yy * 7.0)) * math.tau)
            + 0.12 * np.sin((xx * 38.0 + yy * 1.6) * math.tau)
            + 0.04 * rng.normal(size=(height, width))
        )
        broad_variation = 0.04 * np.sin((yy * 3.0 + 0.18 * np.sin(xx * 4.0)) * math.tau)
        shade = np.clip(0.58 + 0.13 * grain + broad_variation, 0.0, 1.0)
        base = np.array([171, 132, 88], dtype=np.float32)
        light = np.array([226, 198, 151], dtype=np.float32)
        rgb = base[None, None, :] * (1.0 - shade[:, :, None]) + light[None, None, :] * shade[:, :, None]

        seam_rows = np.arange(0, height, max(height // 4, 1))
        for row in seam_rows:
            rgb[max(row - 1, 0) : min(row + 1, height), :, :] *= 0.90
        rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigmaX=0.65, sigmaY=0.35)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return out_path

    def _render_domain_params(self) -> dict:
        preset = str(getattr(self.config, "render_domain_preset", "default") or "default")
        if preset == "branchs_like_v1":
            return {
                "skybox_rgb1": "0.10 0.42 0.22",
                "skybox_rgb2": "0.25 0.62 0.36",
                "table_rgba": "0.84 0.76 0.64 1",
                "table_reflectance": "0.02",
                "table_specular": "0.05",
                "wall_rgba": "0.10 0.56 0.27 1",
                "wall_reflectance": "0.01",
                "wall_specular": "0.03",
                "vessel_rgba": "0.88 0.94 0.92 0.28",
                "mount_piper_rgba": "0.78 0.78 0.74 1",
                "mount_elite_rgba": "0.76 0.76 0.72 1",
                "floor_rgba": "0.68 0.61 0.52 1",
                "table_shadow_rgba": "0.56 0.51 0.43 1",
                "headlight_ambient": "0.34 0.35 0.32",
                "headlight_diffuse": "0.36 0.37 0.34",
                "headlight_specular": "0.01 0.01 0.01",
                "key_diffuse": "0.48 0.49 0.45",
                "key_specular": "0.015 0.015 0.015",
                "fill_diffuse": "0.20 0.22 0.19",
                "fill_specular": "0.004 0.004 0.004",
                "rim_diffuse": "0.09 0.10 0.08",
                "rim_specular": "0.002 0.002 0.002",
            }
        return {
            "skybox_rgb1": "0.80 0.86 0.92",
            "skybox_rgb2": "0.97 0.98 0.99",
            "table_rgba": "0.96 0.90 0.78 1",
            "table_reflectance": "0.04",
            "table_specular": "0.10",
            "wall_rgba": "0.95 0.96 0.97 1",
            "wall_reflectance": "0.03",
            "wall_specular": "0.08",
            "vessel_rgba": "0.76 0.80 0.86 0.18",
            "mount_piper_rgba": "0.62 0.71 0.82 1",
            "mount_elite_rgba": "0.84 0.78 0.71 1",
            "floor_rgba": "0.90 0.89 0.87 1",
            "table_shadow_rgba": "0.68 0.66 0.62 1",
            "headlight_ambient": "0.42 0.42 0.42",
            "headlight_diffuse": "0.34 0.34 0.34",
            "headlight_specular": "0.03 0.03 0.03",
            "key_diffuse": "0.56 0.56 0.56",
            "key_specular": "0.03 0.03 0.03",
            "fill_diffuse": "0.24 0.24 0.24",
            "fill_specular": "0.01 0.01 0.01",
            "rim_diffuse": "0.12 0.12 0.12",
            "rim_specular": "0.005 0.005 0.005",
        }

    def _load_robot_mesh_manifest(self) -> dict:
        path = (REPO_ROOT / self.config.robot_mesh_manifest).resolve()
        if self.config.show_robots and self.config.robot_visual_mode == "static" and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"robots": [], "out_root": str((REPO_ROOT / "simulation_output/mujoco_robot_assets").resolve())}

    def _scale_physics_params(self) -> None:
        if not self.config.use_scene_scale:
            return
        s = max(float(self.scene_scale), 1e-6)
        self.config.wire_spacing *= s
        self.config.wire_radius *= s
        self.config.elirobot_action_scale *= s
        self.config.piper_insertion_window *= s
        self.config.elite_initial_local_offset *= s
        self.config.elite_initial_forward_offset *= s
        self.config.guided_start_lateral_noise *= s
        self.config.advance_step *= s
        self.config.wall_margin *= s
        self.config.wall_clearance *= s
        self.config.min_radius *= s
        self.config.max_radius *= s
        self.config.target_radius *= s
        self.config.magnet_range *= s
        self.config.magnet_tip_height_offset *= s
        self.config.vessel_radius_scale = float(self.config.vessel_radius_scale)

    def _build_xml(self) -> str:
        mesh_assets, robot_geoms = self._robot_xml_parts()
        kin_assets, kin_bodies = self._kinematic_robot_xml_parts()
        mesh_assets = "\n".join([mesh_assets, kin_assets])
        body_parts = []
        line_mode = str(self.config.wire_visual_mode) == "line"
        render_domain = self._render_domain_params()
        wire_half_length = float(self.config.wire_spacing * (0.50 if line_mode else 0.54))
        last_visible_wire_index = int(self.config.wire_segments) - 2 if line_mode else int(self.config.wire_segments) - 1
        tip_visual_segments = max(int(self.config.wire_tip_visual_segments), 0)
        tip_visual_start = max(last_visible_wire_index - tip_visual_segments + 1, 0)
        for index in range(self.config.wire_segments):
            is_tip_visual = line_mode and tip_visual_segments > 0 and tip_visual_start <= index <= last_visible_wire_index
            wire_radius = float(self.config.wire_visual_radius) if line_mode else float(self.config.wire_radius)
            if is_tip_visual:
                wire_radius *= max(float(self.config.wire_tip_visual_radius_scale), 1e-6)
            wire_alpha = 0.0 if line_mode and index == self.config.wire_segments - 1 else 0.98
            wire_rgb = str(self.config.wire_tip_visual_rgb if is_tip_visual else self.config.wire_visual_rgb)
            body_parts.append(
                f"""
                <body name="wire_{index}" pos="0 0 0">
                  <freejoint name="wire_{index}_free"/>
                  <geom name="wire_{index}_geom" type="capsule"
                        fromto="0 0 {-wire_half_length:.6f} 0 0 {wire_half_length:.6f}"
                        size="{wire_radius:.5f}"
                        mass="{self.config.wire_mass:.6f}" rgba="{wire_rgb} {wire_alpha:.2f}"/>
                </body>
                """
            )
        vessel_parts = []
        if self.config.show_path_tubes:
            for task, color in (("left", "0.45 0.70 0.48 0.12"), ("right", "0.45 0.55 0.85 0.12")):
                path = self.paths[task]
                radii = self.radii[task]
                stride = max(int(self.config.vessel_visual_stride), 1)
                for geom_index, idx in enumerate(range(0, len(path) - 1, stride)):
                    j = min(idx + stride, len(path) - 1)
                    p0 = path[idx]
                    p1 = path[j]
                    radius = float(0.5 * (radii[idx] + radii[j]) * self.config.vessel_visual_radius_scale)
                    vessel_parts.append(
                        f"""
                        <geom name="vessel_{task}_{geom_index:03d}" type="capsule"
                              fromto="{p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f} {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}"
                              size="{radius:.6f}" contype="0" conaffinity="0" rgba="{color}"/>
                        """
                    )
        if self.vessel_mesh_asset is not None and Path(self.vessel_mesh_asset).exists():
            vessel_parts.append(
                f"""
                <geom name="vessel_mesh" type="mesh" mesh="vessel_mesh"
                      contype="0" conaffinity="0" rgba="{render_domain['vessel_rgba']}"/>
                """
            )
        route_picker_parts = []
        if bool(getattr(self.config, "route_picker_markers", False)):
            marker_task = str(getattr(self.config, "route_picker_marker_task", self.task))
            if marker_task not in self.paths:
                marker_task = self.task
            max_progress = float(len(self.paths[marker_task]) - 1)
            marker_stride = max(float(getattr(self.config, "route_picker_marker_stride", 0.5)), 1e-3)
            marker_size = max(float(getattr(self.config, "route_picker_marker_size", 0.0022)), 1e-5)
            marker_alpha = float(np.clip(float(getattr(self.config, "route_picker_marker_alpha", 0.9)), 0.0, 1.0))
            marker_progresses = np.arange(0.0, max_progress + 1e-6, marker_stride, dtype=np.float32)
            if len(marker_progresses) == 0 or float(marker_progresses[-1]) < max_progress:
                marker_progresses = np.concatenate([marker_progresses, np.asarray([max_progress], dtype=np.float32)])
            marker_path = self.paths[marker_task]
            for marker_index, progress in enumerate(marker_progresses):
                base_index = int(np.clip(np.floor(float(progress)), 0, len(marker_path) - 1))
                fraction = float(progress) - float(base_index)
                if base_index >= len(marker_path) - 1:
                    center = marker_path[-1]
                else:
                    center = (1.0 - fraction) * marker_path[base_index] + fraction * marker_path[base_index + 1]
                progress_key = int(round(float(progress) * 1000.0))
                route_picker_parts.append(
                    f"""
                    <geom name="route_pick_marker_{marker_task}_p{progress_key:06d}_{marker_index:04d}"
                          type="sphere"
                          pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}"
                          size="{marker_size:.6f}"
                          contype="0" conaffinity="0"
                          rgba="1.0 0.52 0.02 {marker_alpha:.3f}"/>
                    """
                )
        tool_marker_parts = ""
        if self.config.show_tool_markers:
            tool_marker_parts = """
            <body name="elite_magnet" pos="0 0 0">
              <freejoint name="elite_magnet_free"/>
              <geom type="sphere" size="0.010" mass="0.05" contype="0" conaffinity="0" rgba="1.0 0.55 0.05 0.65"/>
            </body>
            <body name="piper_tip" pos="0 0 0">
              <freejoint name="piper_tip_free"/>
              <geom type="sphere" size="0.007" mass="0.03" contype="0" conaffinity="0" rgba="0.05 0.8 0.25 0.55"/>
            </body>
            """
        else:
            tool_marker_parts = """
            <body name="elite_magnet" pos="0 0 0">
              <freejoint name="elite_magnet_free"/>
              <geom type="sphere" size="0.001" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
            </body>
            <body name="piper_tip" pos="0 0 0">
              <freejoint name="piper_tip_free"/>
              <geom type="sphere" size="0.001" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
            </body>
            """

        center = 0.5 * (self.scene_min + self.scene_max)
        span = np.maximum(self.scene_max - self.scene_min, 1e-6)
        camera_distance = max(float(np.max(span)) * self.config.camera_distance_scale, self.config.camera_min_distance)
        camera_pos = center + np.array([0.0, -camera_distance, camera_distance * 0.45], dtype=np.float32)
        lookat = center
        gravity_z = float(self.config.gravity_z) if self.config.guidance_mode == "physical" else 0.0
        piper_base_pose = self.scene_config.get("piper_pose", [-0.28, -0.137, -0.08, 0.0, 0.0, 0.0])
        elite_base_pose = self.scene_config.get("elite_pose", [-0.294, 0.133, -0.08, 0.0, 0.0, 3.141593])
        robot_xy = np.array(
            [[float(piper_base_pose[0]), float(piper_base_pose[1])], [float(elite_base_pose[0]), float(elite_base_pose[1])]],
            dtype=np.float32,
        )
        table_margin_xy = np.array([0.13, 0.16], dtype=np.float32)
        table_xy_min = np.minimum(self.scene_min[:2], np.min(robot_xy, axis=0)) - table_margin_xy
        table_xy_max = np.maximum(self.scene_max[:2], np.max(robot_xy, axis=0)) + table_margin_xy
        table_half_xy = np.array(
            [
                max(float(0.5 * (table_xy_max[0] - table_xy_min[0])), 0.55 * self.scene_scale),
                max(float(0.5 * (table_xy_max[1] - table_xy_min[1])), 0.42 * self.scene_scale),
            ],
            dtype=np.float32,
        )
        table_half_z = max(float(span[2]) * 0.18, 0.035 * self.scene_scale)
        table_center = np.array(
            [
                float(0.5 * (table_xy_min[0] + table_xy_max[0])),
                float(0.5 * (table_xy_min[1] + table_xy_max[1])),
                float(self.scene_min[2] - table_half_z - 0.045 * self.scene_scale),
            ],
            dtype=np.float32,
        )
        table_top_z = float(table_center[2] + table_half_z)
        mount_radius = max(0.050, 0.80 * self.scene_scale)
        mount_half_z = max(0.014, 0.20 * self.scene_scale)
        mount_top_half_z = max(0.004, 0.06 * self.scene_scale)
        mount_z = table_top_z + mount_half_z
        mount_top_z = table_top_z + 2.0 * mount_half_z + mount_top_half_z
        back_wall_z = float(table_center[2] + 0.34 * self.scene_scale)
        side_wall_z = float(table_center[2] + 0.30 * self.scene_scale)
        room_pad_x = max(float(table_half_xy[0]) + 0.16, 0.78 * self.scene_scale)
        room_pad_y = max(float(table_half_xy[1]) + 0.16, 0.68 * self.scene_scale)
        xml = f"""
        <mujoco model="guided_wire_mvp">
          <compiler angle="radian" autolimits="true"/>
          <option timestep="{self.config.sim_dt:.5f}" gravity="0 0 {gravity_z:.6f}" integrator="implicit"/>
          <asset>
            <texture name="skybox" type="skybox" builtin="gradient" rgb1="{render_domain['skybox_rgb1']}" rgb2="{render_domain['skybox_rgb2']}" width="512" height="512"/>
            <texture name="table_tex" type="2d" file="{Path(self.table_texture_asset).as_posix()}"/>
            <material name="table_mat" texture="table_tex" texrepeat="2.4 1.6" reflectance="{render_domain['table_reflectance']}" specular="{render_domain['table_specular']}" shininess="0.12" rgba="{render_domain['table_rgba']}"/>
            <material name="wall_mat" rgba="{render_domain['wall_rgba']}" reflectance="{render_domain['wall_reflectance']}" specular="{render_domain['wall_specular']}" shininess="0.06"/>
            <material name="mount_piper_mat" rgba="{render_domain['mount_piper_rgba']}" reflectance="0.05" specular="0.14" shininess="0.18"/>
            <material name="mount_elite_mat" rgba="{render_domain['mount_elite_rgba']}" reflectance="0.05" specular="0.12" shininess="0.16"/>
            {f'<mesh name="vessel_mesh" file="{Path(self.vessel_mesh_asset).as_posix()}"/>' if self.vessel_mesh_asset is not None and Path(self.vessel_mesh_asset).exists() else ''}
            {mesh_assets}
          </asset>
          <visual>
            <global offwidth="{self.config.render_width}" offheight="{self.config.render_height}"/>
            <headlight ambient="{render_domain['headlight_ambient']}" diffuse="{render_domain['headlight_diffuse']}" specular="{render_domain['headlight_specular']}"/>
            <quality shadowsize="0"/>
          </visual>
          <worldbody>
            <light name="key" pos="0 -3 4" dir="0 1 -1" diffuse="{render_domain['key_diffuse']}" specular="{render_domain['key_specular']}"/>
            <light name="fill" pos="{center[0] + 1.7:.6f} {center[1] - 1.3:.6f} {center[2] + 2.7:.6f}"
                   dir="-0.7 0.4 -1" diffuse="{render_domain['fill_diffuse']}" specular="{render_domain['fill_specular']}"/>
            <light name="rim" pos="{center[0] - 1.6:.6f} {center[1] + 1.2:.6f} {center[2] + 2.1:.6f}"
                   dir="0.5 -0.2 -1" diffuse="{render_domain['rim_diffuse']}" specular="{render_domain['rim_specular']}"/>
            <geom name="room_back_wall" type="box"
                  pos="{center[0]:.6f} {table_xy_max[1] + 0.11:.6f} {back_wall_z:.6f}"
                  size="{room_pad_x:.6f} 0.012 {0.46 * self.scene_scale:.6f}"
                  contype="0" conaffinity="0" material="wall_mat"/>
            <geom name="room_side_wall" type="box"
                  pos="{table_xy_min[0] - 0.11:.6f} {center[1]:.6f} {side_wall_z:.6f}"
                  size="0.012 {room_pad_y:.6f} {0.42 * self.scene_scale:.6f}"
                  contype="0" conaffinity="0" material="wall_mat"/>
            <geom name="room_floor_panel" type="box"
                  pos="{center[0]:.6f} {center[1]:.6f} {table_center[2] - 0.16 * self.scene_scale:.6f}"
                  size="{room_pad_x:.6f} {room_pad_y:.6f} 0.012"
                  contype="0" conaffinity="0" rgba="{render_domain['floor_rgba']}"/>
            <body name="table" pos="{table_center[0]:.6f} {table_center[1]:.6f} {table_center[2]:.6f}">
              <geom name="table_top" type="box"
                    size="{table_half_xy[0]:.6f} {table_half_xy[1]:.6f} {table_half_z:.6f}"
                    contype="0" conaffinity="0" material="table_mat"/>
              <geom name="table_shadow" type="box"
                    pos="0 0 {-table_half_z * 0.9:.6f}"
                    size="{table_half_xy[0] * 0.985:.6f} {table_half_xy[1] * 0.985:.6f} {max(table_half_z * 0.35, 0.012):.6f}"
                    contype="0" conaffinity="0" rgba="{render_domain['table_shadow_rgba']}"/>
            </body>
            <body name="piper_table_mount" pos="{float(piper_base_pose[0]):.6f} {float(piper_base_pose[1]):.6f} {mount_z:.6f}">
              <geom name="piper_mount_column" type="cylinder"
                    size="{mount_radius:.6f} {mount_half_z:.6f}"
                    contype="0" conaffinity="0" material="mount_piper_mat"/>
              <geom name="piper_mount_plate" type="cylinder"
                    pos="0 0 {mount_half_z + mount_top_half_z:.6f}"
                    size="{mount_radius * 1.22:.6f} {mount_top_half_z:.6f}"
                    contype="0" conaffinity="0" material="mount_piper_mat"/>
            </body>
            <body name="elite_table_mount" pos="{float(elite_base_pose[0]):.6f} {float(elite_base_pose[1]):.6f} {mount_z:.6f}">
              <geom name="elite_mount_column" type="cylinder"
                    size="{mount_radius:.6f} {mount_half_z:.6f}"
                    contype="0" conaffinity="0" material="mount_elite_mat"/>
              <geom name="elite_mount_plate" type="cylinder"
                    pos="0 0 {mount_half_z + mount_top_half_z:.6f}"
                    size="{mount_radius * 1.22:.6f} {mount_top_half_z:.6f}"
                    contype="0" conaffinity="0" material="mount_elite_mat"/>
            </body>
            <camera name="side" pos="{camera_pos[0]:.6f} {camera_pos[1]:.6f} {camera_pos[2]:.6f}"
                    xyaxes="1 0 0 0 0 1" mode="fixed"/>
            <camera name="top" pos="{center[0]:.6f} {center[1]:.6f} {center[2] + camera_distance:.6f}"
                    xyaxes="1 0 0 0 1 0" mode="fixed"/>
            <camera name="perspective" pos="{camera_pos[0]:.6f} {camera_pos[1]:.6f} {camera_pos[2]:.6f}"
                    xyaxes="1 0 0 0 0 1" mode="fixed"/>
            <site name="scene_center" pos="{lookat[0]:.6f} {lookat[1]:.6f} {lookat[2]:.6f}" size="0.01" rgba="0 0 0 0"/>
            <body name="diagnostic_tip_marker" pos="0 0 0">
              <freejoint name="diagnostic_tip_marker_free"/>
              <geom name="diagnostic_tip_marker_geom" type="sphere" size="0.0045" mass="0.001"
                    contype="0" conaffinity="0" rgba="0.95 0.05 0.05 0"/>
            </body>
            <body name="wire_tip_visual_marker" pos="0 0 0">
              <freejoint name="wire_tip_visual_marker_free"/>
              <geom name="wire_tip_visual_marker_geom" type="sphere"
                    size="{max(float(self.config.wire_tip_marker_radius), 1e-4):.6f}" mass="0.001"
                    contype="0" conaffinity="0"
                    rgba="{self.config.wire_tip_visual_rgb} {float(np.clip(self.config.wire_tip_marker_alpha, 0.0, 1.0)):.2f}"/>
            </body>
            <body name="diagnostic_magnetic_marker" pos="0 0 0">
              <freejoint name="diagnostic_magnetic_marker_free"/>
              <geom name="diagnostic_magnetic_marker_geom" type="sphere" size="0.0065" mass="0.001"
                    contype="0" conaffinity="0" rgba="0.05 0.95 0.15 0"/>
            </body>
            <body name="diagnostic_elite_marker" pos="0 0 0">
              <freejoint name="diagnostic_elite_marker_free"/>
              <geom name="diagnostic_elite_marker_geom" type="sphere" size="0.0055" mass="0.001"
                    contype="0" conaffinity="0" rgba="1.0 0.48 0.05 0"/>
            </body>
            {tool_marker_parts}
            {''.join(body_parts)}
            {''.join(vessel_parts)}
            {''.join(route_picker_parts)}
            {robot_geoms}
            {kin_bodies}
          </worldbody>
        </mujoco>
        """
        return xml

    def _kinematic_robot_xml_parts(self) -> tuple[str, str]:
        if not self.config.show_robots or self.config.robot_visual_mode != "kinematic":
            return "", ""
        assets = []
        bodies = []
        for robot_name, urdf_path, visuals, pose, rgba in [
            ("piper", self.piper_urdf_path, self.piper_visuals, self.scene_config.get("piper_pose", [-0.28, -0.137, -0.08, 0.0, 0.0, 0.0]), "0.22 0.42 0.88 1"),
            ("elite", self.elite_urdf_path, self.elite_visuals, self.scene_config.get("elite_pose", [-0.294, 0.133, -0.08, 0.0, 0.0, 3.141593]), "0.92 0.42 0.22 1"),
        ]:
            root_tree = _load_urdf_joint_tree(urdf_path)
            asset_lines, body_xml = self._robot_body_xml(robot_name, urdf_path, root_tree, visuals, pose, rgba)
            assets.extend(asset_lines)
            bodies.append(body_xml)
        return "\n".join(assets), "\n".join(bodies)

    def _robot_body_xml(self, robot_name: str, urdf_path: Path, tree: dict, visuals: dict, base_pose: list[float], rgba: str) -> tuple[list[str], str]:
        asset_lines = []
        mesh_name_by_path: dict[str, str] = {}

        def ensure_mesh(filename: str) -> str:
            path = _mesh_path_from_urdf(urdf_path, filename)
            key = path.as_posix()
            if key not in mesh_name_by_path:
                mesh_name = f"{robot_name}_mesh_{len(mesh_name_by_path):03d}"
                mesh_name_by_path[key] = mesh_name
                asset_lines.append(f'<mesh name="{mesh_name}" file="{key}"/>')
            return mesh_name_by_path[key]

        x, y, z, roll, pitch, yaw = [float(v) for v in base_pose]
        root_link = tree["root_link"]

        def link_geoms(link_name: str, depth: int = 0) -> str:
            lines = []
            for visual in visuals.get(link_name, []):
                mesh_name = ensure_mesh(visual["mesh"])
                quat = _quat_from_rpy(*visual["rpy"].tolist())
                geom_rgba = self._robot_visual_color(robot_name, visual.get("rgba"), depth)
                scale_attr = f' scale="{visual["scale"]}"' if visual.get("scale") else ""
                lines.append(
                    f'<geom name="{robot_name}_{visual["name"]}_geom" type="mesh" mesh="{mesh_name}" '
                    f'pos="{_vec_to_str(visual["xyz"])}" quat="{_quat_to_str(quat)}"{scale_attr} '
                    f'contype="0" conaffinity="0" rgba="{geom_rgba}"/>'
                )
            return "\n".join(lines)

        def build_link(link_name: str, depth: int = 0) -> str:
            parts = [link_geoms(link_name, depth)]
            for joint in tree["children"].get(link_name, []):
                body_name = f"{robot_name}_{joint['child']}"
                joint_pos = _vec_to_str(joint["xyz"])
                joint_quat = _rpy_to_quat_attr(tuple(float(v) for v in joint["rpy"]))
                joint_lines = []
                if joint["type"] != "fixed":
                    mj_type = "slide" if joint["type"] == "prismatic" else "hinge"
                    joint_lines.append(
                        f'<joint name="{robot_name}_{joint["name"]}" type="{mj_type}" axis="{_vec_to_str(joint["axis"])}" '
                        f'range="{joint["lower"]:.8f} {joint["upper"]:.8f}" limited="true" damping="1.0"/>'
                    )
                child_xml = build_link(joint["child"], depth + 1)
                parts.append(
                    f'<body name="{body_name}" pos="{joint_pos}" quat="{joint_quat}">'
                    f'{"".join(joint_lines)}'
                    f'{child_xml}'
                    f'</body>'
                )
            return "\n".join(parts)

        body = (
            f'<body name="{robot_name}_base" pos="{x:.8f} {y:.8f} {z:.8f}" '
            f'quat="{_rpy_to_quat_attr((roll, pitch, yaw))}">'
            f'{build_link(root_link)}'
            f'</body>'
        )
        return asset_lines, body

    def _robot_xml_parts(self) -> tuple[str, str]:
        if not self.config.show_robots or not self.robot_mesh_manifest.get("robots"):
            return "", ""
        out_root = Path(self.robot_mesh_manifest.get("out_root", "")).resolve()
        assets = []
        geoms = []
        rgba_by_robot = {
            "piper": "0.22 0.42 0.88 1",
            "elite": "0.92 0.42 0.22 1",
        }
        for robot in self.robot_mesh_manifest.get("robots", []):
            robot_name = robot["name"]
            rgba = rgba_by_robot.get(robot_name, "0.7 0.7 0.7 1")
            if not robot.get("meshes"):
                continue
            for item in robot.get("meshes", []):
                mesh_name = f"robot_{item['name']}"
                if int(item.get("vertices", 0)) < 12 or int(item.get("faces", 0)) < 20:
                    continue
                mesh_path = (out_root / item["path"]).resolve()
                assets.append(f'<mesh name="{mesh_name}" file="{mesh_path.as_posix()}"/>')
                geoms.append(
                    f'<geom name="{mesh_name}_geom" type="mesh" mesh="{mesh_name}" '
                    f'contype="0" conaffinity="0" rgba="{rgba}"/>'
                )
        return "\n".join(assets), "\n".join(geoms)

    def _cache_joint_addresses(self) -> None:
        self.wire_qpos_addr = []
        for index in range(self.config.wire_segments):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"wire_{index}_free")
            self.wire_qpos_addr.append(int(self.model.jnt_qposadr[jid]))
        self.elite_qpos_addr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elite_magnet_free")])
        self.piper_qpos_addr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "piper_tip_free")])
        self.diagnostic_tip_qpos_addr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "diagnostic_tip_marker_free")]
        )
        self.wire_tip_visual_qpos_addr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wire_tip_visual_marker_free")]
        )
        self.diagnostic_magnetic_qpos_addr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "diagnostic_magnetic_marker_free")]
        )
        self.diagnostic_elite_qpos_addr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "diagnostic_elite_marker_free")]
        )
        self.robot_joint_qpos_addr = {}
        for prefix, values in (("piper", self.piper_joint_values), ("elite", self.elite_joint_values)):
            for joint_name in values:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{joint_name}")
                if jid >= 0:
                    self.robot_joint_qpos_addr[f"{prefix}_{joint_name}"] = int(self.model.jnt_qposadr[jid])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.task = options.get("task", "left")
        if self.task not in TASK_INSTRUCTIONS:
            raise ValueError(f"Unknown task: {self.task}")
        self.instruction = TASK_INSTRUCTIONS[self.task]

        path_len = max(len(self.paths[self.task]) - 1, 1)
        start_fraction = float(
            options.get(
                "start_fraction",
                self.rng.uniform(self.config.guided_start_min_fraction, self.config.guided_start_max_fraction),
            )
        )
        self.path_progress_float = float(np.clip(start_fraction * path_len, 0.0, path_len))
        self.path_progress_index = int(np.floor(self.path_progress_float))
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)

        lateral = np.zeros(2, dtype=np.float32)
        tip = center + n1 * lateral[0] + n2 * lateral[1]
        self.positions = np.zeros((self.config.wire_segments, 3), dtype=np.float32)
        for index in range(self.config.wire_segments):
            back_distance = self.config.wire_spacing * (self.config.wire_segments - 1 - index)
            self.positions[index] = self._centerline_point_behind(self.path_progress_float, back_distance) + n1 * lateral[0] + n2 * lateral[1]
        self.velocities = np.zeros_like(self.positions)
        self.tip = self.positions[-1].copy()
        self.heading = tangent.copy()
        self.elite_pose = self._front_up_magnetic_target(
            center + n1 * self.config.elite_initial_local_offset,
            tangent,
            forward_offset=float(self.config.elite_initial_forward_offset),
        )
        self.piper_pose = self.positions[0].copy()
        self.piper_feed_anchor = self.piper_pose.copy()
        self.piper_feed_axis = tangent.copy()
        self.piper_feed_home_signal = self._piper_feed_signal(self.piper_home_joint_vector)
        self.piper_feed_signal = self.piper_feed_home_signal
        self.piper_insertion_length = 0.0
        self._piper_primitive_remaining_steps = 0
        self._piper_primitive_active_command = 0.0
        self.controller_state = {
            "piper_intent": "hold",
            "piper_executed_command": "hold",
            "piper_requested_feed": 0.0,
            "piper_executed_feed": 0.0,
            "piper_motion_state": "idle",
            "piper_policy_command": 0.0,
            "piper_policy_command_label": "hold",
            "piper_primitive_steps": int(max(getattr(self.config, "piper_primitive_steps", 1), 1)),
            "piper_primitive_remaining_steps": 0,
            "piper_step_count": 0,
            "piper_insertion_length": 0.0,
            "piper_busy": False,
            "piper_cooldown": False,
            "elite_action_type": "reset",
            "elite_tcp_delta_6d": None,
            "elite_requested_tcp_pose_6d": None,
            "elite_executed_tcp_pose_6d": None,
            "elite_requested_joints": {},
            "elite_executed_joints": {},
            "elite_target_limited": False,
        }
        self.step_count = 0
        self.boundary_projection_count = 0
        self.boundary_projection_window = 0
        self._tip_projected_this_step = False
        self.severe_contact_count = 0
        self.guidewire_points = [self.tip.copy()]
        self.elite_points = [self.elite_pose.copy()]
        self.last_tactile = self._tactile(self.tip)
        self._set_robot_joint_vector("piper", self.piper_home_joint_vector)
        self.set_robot_joints("elite", self._solve_robot_ik("elite", self.elite_pose, max_iters=8, damping=0.03, step_limit=0.20))
        self._refresh_robot_tool_poses()
        self._elite_target_smoothed = self._robot_joint_vector("elite").copy()
        self._elite_previous_joint_delta = np.zeros_like(self._elite_target_smoothed, dtype=np.float32)
        self._elite_previous_joint_accel = np.zeros_like(self._elite_target_smoothed, dtype=np.float32)
        self.magnetic_pose = self._effective_magnetic_pose()
        self._sync_mujoco()
        obs = self._obs_dict(done=False, success=False, failure_reason=None)
        return self._gym_obs(obs), self._info(obs)

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
        self._elite_target_smoothed = (
            (1.0 - self.config.elite_motion_smoothing) * self._elite_target_smoothed
            + self.config.elite_motion_smoothing * elite_target
        ).astype(np.float32)
        current_elite = self._robot_joint_vector("elite")
        delta = self._elite_target_smoothed - current_elite
        if self.config.elite_joint_rate_limit > 0.0:
            delta = np.clip(delta, -float(self.config.elite_joint_rate_limit), float(self.config.elite_joint_rate_limit))
        if self.config.elite_joint_accel_limit > 0.0:
            previous_delta = getattr(self, "_elite_previous_joint_delta", np.zeros_like(delta, dtype=np.float32))
            if previous_delta.shape != delta.shape:
                previous_delta = np.zeros_like(delta, dtype=np.float32)
            desired_delta_change = delta - previous_delta
            delta_change = np.clip(desired_delta_change, -float(self.config.elite_joint_accel_limit), float(self.config.elite_joint_accel_limit))
            if self.config.elite_joint_jerk_limit > 0.0:
                previous_accel = getattr(self, "_elite_previous_joint_accel", np.zeros_like(delta_change, dtype=np.float32))
                if previous_accel.shape != delta_change.shape:
                    previous_accel = np.zeros_like(delta_change, dtype=np.float32)
                jerk = np.clip(
                    delta_change - previous_accel,
                    -float(self.config.elite_joint_jerk_limit),
                    float(self.config.elite_joint_jerk_limit),
                )
                delta_change = previous_accel + jerk
                delta_change = np.clip(delta_change, -float(self.config.elite_joint_accel_limit), float(self.config.elite_joint_accel_limit))
                self._elite_previous_joint_accel = delta_change.astype(np.float32)
            delta = previous_delta + delta_change
        elif self.config.elite_joint_jerk_limit > 0.0:
            previous_delta = getattr(self, "_elite_previous_joint_delta", np.zeros_like(delta, dtype=np.float32))
            if previous_delta.shape != delta.shape:
                previous_delta = np.zeros_like(delta, dtype=np.float32)
            previous_accel = getattr(self, "_elite_previous_joint_accel", np.zeros_like(delta, dtype=np.float32))
            if previous_accel.shape != delta.shape:
                previous_accel = np.zeros_like(delta, dtype=np.float32)
            desired_delta_change = delta - previous_delta
            jerk = np.clip(
                desired_delta_change - previous_accel,
                -float(self.config.elite_joint_jerk_limit),
                float(self.config.elite_joint_jerk_limit),
            )
            delta_change = previous_accel + jerk
            delta = previous_delta + delta_change
            self._elite_previous_joint_accel = delta_change.astype(np.float32)
        else:
            self._elite_previous_joint_accel = np.zeros_like(delta, dtype=np.float32)
        self._elite_target_smoothed = (current_elite + delta).astype(np.float32)
        self._elite_previous_joint_delta = delta.astype(np.float32)
        executed_elite_target = self._elite_target_smoothed.copy()
        self._set_robot_joint_vector("elite", self._elite_target_smoothed)
        self._refresh_robot_tool_poses()
        self.magnetic_pose = self._effective_magnetic_pose()
        current_signal = self._piper_feed_signal(self._robot_joint_vector("piper"))
        signal_limit = min(float(self.piper_joint_limits.get("joint7", (0.0, 0.035))[1]), -float(self.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]))
        self.piper_feed_signal = float(np.clip(current_signal, -0.18 * signal_limit, signal_limit))
        self.piper_insertion_length = float(self.piper_feed_signal - self.piper_feed_home_signal)
        piper_feed_distance = self.piper_feed_signal - previous_signal
        feed_unit = max(float(self.config.advance_step) * float(self.config.piper_advection_scale), 1e-8)
        piper_cmd = float(np.clip(piper_feed_distance / feed_unit, -1.0, 1.0))
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
            piper_feed_distance = self.piper_feed_signal - previous_signal
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

        self._tip_projected_this_step = False
        for _ in range(self.config.sim_substeps):
            self._integrate_wire(executed_piper_cmd)
        self.tip = self.positions[-1].copy()
        if len(self.guidewire_points) and float(np.linalg.norm(self.tip - self.guidewire_points[-1])) > 1e-6:
            delta = self.tip - self.guidewire_points[-1]
            self.heading = delta / max(float(np.linalg.norm(delta)), 1e-6)
        center, _radius = self._nearest_path_state(self.tip, update_progress=True)
        self.guidewire_points.append(self.tip.copy())
        self.elite_points.append(self.elite_pose.copy())
        self.step_count += 1
        self.last_tactile = self._tactile(self.tip)
        self.magnetic_pose = self._effective_magnetic_pose()
        if self._tip_projected_this_step and self.last_tactile["distance_to_wall"] < 0.0:
            self.boundary_projection_count += 1
            self.boundary_projection_window += 1
        else:
            self.boundary_projection_window = max(self.boundary_projection_window - 1, 0)
        if executed_piper_cmd > 0.08:
            self.path_progress_float = min(
                float(len(self.paths[self.task]) - 1),
                self.path_progress_float + float(executed_piper_cmd) * self.config.commanded_progress_gain,
            )
        if self.last_tactile["contact_strength"] > 0.85:
            self.severe_contact_count += 1
        else:
            self.severe_contact_count = max(self.severe_contact_count - 1, 0)

        self._sync_mujoco()
        success = self._success()
        failure_reason = self._failure_reason(self.last_tactile)
        terminated = success or failure_reason is not None
        truncated = self.step_count >= self.config.max_steps
        obs = self._obs_dict(done=terminated or truncated, success=success, failure_reason=failure_reason)
        reward = self._reward(obs, failure_reason)
        return self._gym_obs(obs), reward, terminated, truncated, self._info(obs)

    def _integrate_wire(self, piper_cmd: float) -> None:
        dt = float(self.config.sim_dt)
        forces = np.zeros_like(self.positions)

        # Segment length springs.
        for index in range(self.config.wire_segments - 1):
            delta = self.positions[index + 1] - self.positions[index]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-7:
                continue
            direction = delta / dist
            stretch = dist - self.config.wire_spacing
            rel_v = float(np.dot(self.velocities[index + 1] - self.velocities[index], direction))
            force = self.config.spring_k * stretch * direction + 0.25 * self.config.damping * rel_v * direction
            forces[index] += force
            forces[index + 1] -= force

        # Bending smoothness force.
        for index in range(1, self.config.wire_segments - 1):
            smooth = self.positions[index - 1] + self.positions[index + 1] - 2.0 * self.positions[index]
            forces[index] += self.config.bend_k * smooth

        center, tangent, _n1, _n2, _radius = self._local_path_frame(self.path_progress_float)
        effective_piper_cmd = max(float(piper_cmd), 0.0)
        forces[0] += tangent * (self.config.piper_force * effective_piper_cmd)

        # Magnetic attraction on the last few beads, strongest at the tip.
        for local_index, weight in zip(range(self.config.wire_segments - 5, self.config.wire_segments), np.linspace(0.25, 1.0, 5)):
            vec = self.magnetic_pose - self.positions[local_index]
            dist = float(np.linalg.norm(vec))
            if dist > 1e-6:
                falloff = math.exp(-((dist / max(self.config.magnet_range, 1e-6)) ** 2))
                forces[local_index] += (vec / dist) * self.config.magnet_force * float(weight) * falloff

        if self.config.guidance_mode != "physical":
            branch_weight = self._branch_centering_weight(self.path_progress_float)
            center_force_scale = 1.0 + branch_weight * (float(self.config.branch_centering_force_scale) - 1.0)
            tip_center_force_scale = 1.0 + branch_weight * (float(self.config.branch_tip_centering_force_scale) - 1.0)

            tip_center, tip_tangent, tip_n1, tip_n2, _tip_radius = self._local_path_frame(self.path_progress_float)
            lookahead_idx = min(
                int(np.floor(self.path_progress_float)) + max(int(self.config.tip_drive_lookahead), 1),
                len(self.wire_centerlines[self.task]) - 1,
            )
            lookahead_target = self.wire_centerlines[self.task][lookahead_idx]
            tip_offset = self.positions[-1] - tip_center
            tip_lateral = np.array([float(np.dot(tip_offset, tip_n1)), float(np.dot(tip_offset, tip_n2))], dtype=np.float32)
            forces[-1] += (-tip_n1 * tip_lateral[0] - tip_n2 * tip_lateral[1]) * self.config.tip_centering_force * tip_center_force_scale
            toward_lookahead = lookahead_target - self.positions[-1]
            toward_norm = float(np.linalg.norm(toward_lookahead))
            if toward_norm > 1e-6:
                forces[-1] += (toward_lookahead / toward_norm) * self.config.tip_path_drive_force
            else:
                forces[-1] += tip_tangent * self.config.tip_path_drive_force

            # Keep the wire body gently biased toward the vessel centerline so normal
            # guidance does not spend the whole rollout sliding along the wall.
            for index, point in enumerate(self.positions):
                nearest_center, radius = self._nearest_path_state(
                    point,
                    update_progress=False,
                    progress_hint=self._segment_progress_hint(index),
                )
                offset = point - nearest_center
                dist = float(np.linalg.norm(offset))
                if dist < 1e-7:
                    continue
                center_bias = dist / max(radius, 1e-6)
                segment_weight = 0.25 + 0.75 * (index / max(self.config.wire_segments - 1, 1))
                forces[index] += -(offset / dist) * self.config.wire_centering_force * center_force_scale * segment_weight * center_bias

        # Vessel-wall penalty along the active path.
        for index, point in enumerate(self.positions):
            nearest_center, radius = self._nearest_path_state(
                point,
                update_progress=False,
                progress_hint=self._segment_progress_hint(index),
            )
            offset = point - nearest_center
            dist = float(np.linalg.norm(offset))
            limit = max(radius - self.config.wall_clearance, radius * self.config.boundary_projection_limit)
            clearance = radius - dist
            if dist > limit:
                normal = offset / max(dist, 1e-6)
                penetration = dist - limit
                outward_v = float(np.dot(self.velocities[index], normal))
                forces[index] += -normal * (self.config.wall_k * 0.72 * penetration + self.config.wall_damping * 0.55 * max(outward_v, 0.0))
            elif clearance < self.config.wall_clearance * self.config.contact_projection_threshold:
                normal = offset / max(dist, 1e-6)
                soft_zone = max(self.config.wall_clearance * self.config.contact_projection_threshold, 1e-6)
                inward_bias = float(np.clip((soft_zone - clearance) / soft_zone, 0.0, 1.0))
                forces[index] += -normal * (self.config.wall_k * 0.28 * inward_bias)

        forces += -self.config.damping * self.velocities
        accel = forces / max(self.config.wire_mass, 1e-8)
        self.velocities = self.velocities + accel * dt
        max_speed = 1.2
        speeds = np.linalg.norm(self.velocities, axis=1)
        too_fast = speeds > max_speed
        if np.any(too_fast):
            self.velocities[too_fast] *= (max_speed / speeds[too_fast])[:, None]
        self.positions = self.positions + self.velocities * dt
        feed_delta = (
            effective_piper_cmd
            * self.config.advance_step
            * self.config.piper_advection_scale
            / max(int(self.config.sim_substeps), 1)
        )
        if abs(feed_delta) > 1e-9:
            for index, point in enumerate(self.positions):
                _center, tangent, _n1, _n2, _local_radius = self._local_path_frame(self._segment_progress_hint(index))
                self.positions[index] = point + tangent * feed_delta

        if self.config.guidance_mode != "physical":
            self._relax_wire_toward_centerline()
            self._stabilize_wire_polyline(use_centerline_scaffold=True)
        else:
            self._anchor_physical_tail()
            self._stabilize_wire_polyline(use_centerline_scaffold=False)

        # Hard clip only when penalty did not recover within one substep.
        for _ in range(2):
            if not self._project_wire_inside_vessel():
                break
        if self.config.guidance_mode == "physical":
            self._anchor_physical_tail()
            self._stabilize_wire_polyline(use_centerline_scaffold=False)

    def _sync_mujoco(self) -> None:
        for index, addr in enumerate(self.wire_qpos_addr):
            self.data.qpos[addr : addr + 3] = self._wire_segment_pose_position(index)
            self.data.qpos[addr + 3 : addr + 7] = self._wire_segment_quaternion(index)
        self.data.qpos[self.piper_qpos_addr : self.piper_qpos_addr + 3] = self.piper_pose
        self.data.qpos[self.piper_qpos_addr + 3 : self.piper_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.elite_qpos_addr : self.elite_qpos_addr + 3] = self.elite_pose
        self.data.qpos[self.elite_qpos_addr + 3 : self.elite_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.diagnostic_tip_qpos_addr : self.diagnostic_tip_qpos_addr + 3] = self.tip
        self.data.qpos[self.diagnostic_tip_qpos_addr + 3 : self.diagnostic_tip_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        wire_visual_offset = np.asarray(self.config.wire_visual_offset, dtype=np.float32).reshape(3)
        self.data.qpos[self.wire_tip_visual_qpos_addr : self.wire_tip_visual_qpos_addr + 3] = self.tip + wire_visual_offset
        self.data.qpos[self.wire_tip_visual_qpos_addr + 3 : self.wire_tip_visual_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.diagnostic_magnetic_qpos_addr : self.diagnostic_magnetic_qpos_addr + 3] = self.magnetic_pose
        self.data.qpos[self.diagnostic_magnetic_qpos_addr + 3 : self.diagnostic_magnetic_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.diagnostic_elite_qpos_addr : self.diagnostic_elite_qpos_addr + 3] = self.elite_pose
        self.data.qpos[self.diagnostic_elite_qpos_addr + 3 : self.diagnostic_elite_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        for joint_name, value in self.piper_joint_values.items():
            addr = self.robot_joint_qpos_addr.get(f"piper_{joint_name}")
            if addr is not None:
                self.data.qpos[addr] = float(value)
        for joint_name, value in self.elite_joint_values.items():
            addr = self.robot_joint_qpos_addr.get(f"elite_{joint_name}")
            if addr is not None:
                self.data.qpos[addr] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def _wire_segment_pose_position(self, index: int) -> np.ndarray:
        visual_offset = np.asarray(self.config.wire_visual_offset, dtype=np.float32).reshape(3)
        visual_positions = self._wire_visual_positions()
        if str(self.config.wire_visual_mode) == "line" and index < len(visual_positions) - 1:
            return ((visual_positions[index] + visual_positions[index + 1]) * 0.5 + visual_offset).astype(np.float32)
        source = visual_positions[index] if index < len(visual_positions) else self.positions[index]
        return (source + visual_offset).astype(np.float32)

    def _wire_segment_quaternion(self, index: int) -> np.ndarray:
        visual_positions = self._wire_visual_positions()
        if len(visual_positions) < 2:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if str(self.config.wire_visual_mode) == "line" and index < len(visual_positions) - 1:
            tangent = visual_positions[index + 1] - visual_positions[index]
            return _quat_from_z_axis(tangent)
        prev_idx = max(index - 1, 0)
        next_idx = min(index + 1, len(visual_positions) - 1)
        tangent = visual_positions[next_idx] - visual_positions[prev_idx]
        if float(np.linalg.norm(tangent)) < 1e-8:
            if index < len(visual_positions) - 1:
                tangent = visual_positions[index + 1] - visual_positions[index]
            else:
                tangent = visual_positions[index] - visual_positions[index - 1]
        return _quat_from_z_axis(tangent)

    def _wire_visual_positions(self) -> np.ndarray:
        positions = np.asarray(self.positions, dtype=np.float32)
        return positions

    def _centerline_point_behind(self, progress: float, distance: float) -> np.ndarray:
        path = self.paths[self.task]
        idx = int(np.clip(np.floor(progress), 0, len(path) - 1))
        frac = float(progress - idx)
        if idx >= len(path) - 1:
            current = path[-1].astype(np.float32)
            idx = len(path) - 1
        else:
            current = ((1.0 - frac) * path[idx] + frac * path[idx + 1]).astype(np.float32)

        remaining = float(max(distance, 0.0))
        if remaining <= 1e-8:
            return current

        if idx < len(path) - 1:
            segment_start = path[idx].astype(np.float32)
            segment_len = float(np.linalg.norm(current - segment_start))
            if segment_len > 1e-8:
                if remaining <= segment_len:
                    return current + (segment_start - current) * (remaining / segment_len)
                remaining -= segment_len
                current = segment_start

        for point_index in range(idx, 0, -1):
            previous = path[point_index - 1].astype(np.float32)
            segment_len = float(np.linalg.norm(current - previous))
            if segment_len <= 1e-8:
                current = previous
                continue
            if remaining <= segment_len:
                return current + (previous - current) * (remaining / segment_len)
            remaining -= segment_len
            current = previous

        return path[0].astype(np.float32)

    def _segment_progress_hint(self, index: int) -> float:
        tip_to_segment_distance = self.config.wire_spacing * (self.config.wire_segments - 1 - index)
        step_length = max(float(self.path_step_length[self.task]), 1e-6)
        return max(0.0, float(self.path_progress_float) - tip_to_segment_distance / step_length)

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

    def _branch_centering_weight(self, progress: float) -> float:
        path_len = max(len(self.paths[self.task]) - 1, 1)
        fraction = float(progress) / float(path_len)
        start = float(self.config.branch_centering_start_fraction)
        ramp = max(float(self.config.branch_centering_ramp_fraction), 1e-6)
        x = float(np.clip((fraction - start) / ramp, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _elite_path_frame(self, progress: float):
        path = self.elite_reference_paths[self.task]
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

    def _nearest_path_state(self, point: np.ndarray, update_progress: bool = False, progress_hint: Optional[float] = None):
        path = self.paths[self.task]
        radii = self.radii[self.task]
        search_progress = self.path_progress_float if progress_hint is None else float(progress_hint)
        current = int(np.clip(np.floor(search_progress), 0, len(path) - 1))
        start = max(0, current - self.config.path_search_backtrack)
        end = min(len(path) - 1, current + self.config.path_search_ahead)
        best_dist = np.inf
        best_center = path[current]
        best_radius = float(radii[current])
        best_progress = float(current)
        for seg_idx in range(start, max(start + 1, end)):
            a = path[seg_idx]
            b = path[min(seg_idx + 1, len(path) - 1)]
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
            center = a + ab * t
            dist = float(np.linalg.norm(point - center))
            if dist < best_dist:
                best_dist = dist
                best_center = center
                best_radius = float((1.0 - t) * radii[seg_idx] + t * radii[min(seg_idx + 1, len(path) - 1)])
                best_progress = float(seg_idx) + t
        if update_progress:
            previous = float(self.path_progress_float)
            self.path_progress_float = max(previous, previous + self.config.progress_gain * max(best_progress - previous, 0.0))
            self.path_progress_index = int(np.floor(self.path_progress_float))
        return best_center.astype(np.float32), best_radius

    def _tactile(self, point: np.ndarray) -> Dict:
        center, radius = self._nearest_path_state(point, update_progress=False)
        offset = point - center
        radial_dist = float(np.linalg.norm(offset))
        normal = offset / max(radial_dist, 1e-6)
        distance_to_wall = radius - radial_dist
        strength = float(np.clip((self.config.wall_margin - distance_to_wall) / max(self.config.wall_margin, 1e-6), 0.0, 1.0))
        return {
            "nearest_path_index": int(self.path_progress_index),
            "nearest_center": center.astype(float).tolist(),
            "local_radius": float(radius),
            "inside_vessel": bool(distance_to_wall >= 0.0),
            "distance_to_wall": float(distance_to_wall),
            "contact_flag": int(strength > 0.0),
            "contact_direction": self.reference_env._contact_direction(normal) if strength > 0 else "none",
            "contact_normal": normal.astype(float).tolist(),
            "contact_strength": strength,
            "last_boundary_projection": bool(self.boundary_projection_count > 0),
        }

    def segment_min_distance_to_wall(self) -> float:
        min_distance = float("inf")
        for index, point in enumerate(self.positions):
            center, radius = self._nearest_path_state(
                point,
                update_progress=False,
                progress_hint=self._segment_progress_hint(index),
            )
            min_distance = min(min_distance, float(radius - np.linalg.norm(point - center)))
        return min_distance

    @staticmethod
    def _clip_local_offset(offset: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(offset))
        if norm <= limit or norm < 1e-6:
            return offset.astype(np.float32)
        return (offset / norm * limit).astype(np.float32)

    def _project_wire_inside_vessel(self) -> bool:
        projected = False
        for index, point in enumerate(self.positions):
            nearest_center, radius = self._nearest_path_state(
                point,
                update_progress=False,
                progress_hint=self._segment_progress_hint(index),
            )
            offset = point - nearest_center
            dist = float(np.linalg.norm(offset))
            limit = max(radius - self.config.wall_clearance * self.config.boundary_projection_relief, radius * 0.70)
            if dist > limit:
                normal = offset / max(dist, 1e-6)
                self.positions[index] = nearest_center + normal * limit
                outward_v = max(float(np.dot(self.velocities[index], normal)), 0.0)
                self.velocities[index] -= normal * self.config.contact_relief_ratio * outward_v
                projected = True
                if index == self.config.wire_segments - 1:
                    self._tip_projected_this_step = True
        return projected

    def _relax_wire_toward_centerline(self) -> None:
        branch_weight = self._branch_centering_weight(self.path_progress_float)
        relaxation_scale = 1.0 + branch_weight * (float(self.config.branch_relaxation_scale) - 1.0)
        relaxation = float(np.clip(self.config.wire_centerline_relaxation * relaxation_scale, 0.0, 1.0))
        if relaxation <= 0.0:
            return
        for index, point in enumerate(self.positions):
            center, _radius = self._nearest_path_state(
                point,
                update_progress=False,
                progress_hint=self._segment_progress_hint(index),
            )
            offset = point - center
            segment_weight = 0.45 + 0.55 * (index / max(self.config.wire_segments - 1, 1))
            self.positions[index] = center + offset * (1.0 - relaxation * segment_weight)
            self.velocities[index] *= 1.0 - 0.35 * relaxation * segment_weight

    def _stabilize_wire_polyline(self, use_centerline_scaffold: bool = True) -> None:
        if len(self.positions) < 3:
            return
        old_positions = self.positions.copy()
        physical_tip_anchor = self.positions[-1].copy() if not use_centerline_scaffold else None
        rest = max(float(self.config.wire_spacing), 1e-6)

        # The guidewire is represented by independent MuJoCo capsules. Without
        # a monotonic path scaffold, spacing constraints can preserve a local
        # fold and render as two blue lines near the entry.
        scaffold_relaxation = float(np.clip(self.config.wire_scaffold_relaxation, 0.0, 1.0)) if use_centerline_scaffold else 0.0
        if scaffold_relaxation > 0.0:
            progress = float(self.path_progress_float)
            tip_center = self._centerline_point_behind(progress, 0.0)
            tip_offset = self.positions[-1] - tip_center
            decay_segments = max(float(self.config.wire_tip_offset_decay_segments), 1e-6)
            for index in range(self.config.wire_segments):
                distance_back = rest * float(self.config.wire_segments - 1 - index)
                center = self._centerline_point_behind(progress, distance_back)
                segment_back = float(self.config.wire_segments - 1 - index)
                offset_decay = float(np.exp(-segment_back / decay_segments))
                target = center + tip_offset * offset_decay
                segment_weight = 0.45 + 0.35 * (1.0 - index / max(self.config.wire_segments - 1, 1))
                blend = scaffold_relaxation * segment_weight
                self.positions[index] = (1.0 - blend) * self.positions[index] + blend * target

        length_iterations = 5 if not use_centerline_scaffold else 4
        for _ in range(length_iterations):
            for index in range(self.config.wire_segments - 2, -1, -1):
                delta = self.positions[index] - self.positions[index + 1]
                dist = float(np.linalg.norm(delta))
                if dist < rest * 0.35:
                    _center, tangent, _n1, _n2, _radius = self._local_path_frame(self._segment_progress_hint(index))
                    delta = -tangent
                    dist = 1.0
                self.positions[index] = self.positions[index + 1] + delta / dist * rest
            for index in range(1, self.config.wire_segments):
                delta = self.positions[index] - self.positions[index - 1]
                dist = float(np.linalg.norm(delta))
                if dist < rest * 0.35:
                    _center, tangent, _n1, _n2, _radius = self._local_path_frame(self._segment_progress_hint(index))
                    delta = tangent
                    dist = 1.0
                self.positions[index] = self.positions[index - 1] + delta / dist * rest
            if use_centerline_scaffold:
                for index in range(1, self.config.wire_segments - 1):
                    target = 0.5 * (self.positions[index - 1] + self.positions[index + 1])
                    self.positions[index] = 0.72 * self.positions[index] + 0.28 * target

        max_turn = float(np.cos(np.deg2rad(np.clip(self.config.wire_max_turn_degrees, 1.0, 179.0))))
        curvature_iterations = int(self.config.physical_curvature_iterations) if not use_centerline_scaffold else 2
        curvature_blend = float(np.clip(self.config.physical_curvature_smoothing, 0.0, 0.95)) if not use_centerline_scaffold else 1.0
        for _ in range(max(curvature_iterations, 0)):
            segments = np.diff(self.positions, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            directions = segments / np.maximum(lengths[:, None], 1e-9)
            for index in range(1, self.config.wire_segments - 1):
                if float(np.dot(directions[index - 1], directions[index])) < max_turn:
                    midpoint = 0.5 * (self.positions[index - 1] + self.positions[index + 1])
                    if use_centerline_scaffold:
                        progress = self._segment_progress_hint(index)
                        center = self._centerline_point_behind(progress, 0.0)
                        self.positions[index] = 0.45 * midpoint + 0.55 * center
                    else:
                        self.positions[index] = (1.0 - curvature_blend) * self.positions[index] + curvature_blend * midpoint
            if not use_centerline_scaffold:
                for index in range(self.config.wire_segments - 2, -1, -1):
                    delta = self.positions[index] - self.positions[index + 1]
                    dist = float(np.linalg.norm(delta))
                    if dist > 1e-8:
                        self.positions[index] = self.positions[index + 1] + delta / dist * rest
                for index in range(1, self.config.wire_segments):
                    delta = self.positions[index] - self.positions[index - 1]
                    dist = float(np.linalg.norm(delta))
                    if dist > 1e-8:
                        self.positions[index] = self.positions[index - 1] + delta / dist * rest
        if not use_centerline_scaffold:
            self._stabilize_physical_tip(rest)
            if physical_tip_anchor is not None:
                self.positions[-1] = physical_tip_anchor
                for index in range(self.config.wire_segments - 2, -1, -1):
                    delta = self.positions[index] - self.positions[index + 1]
                    dist = float(np.linalg.norm(delta))
                    if dist > 1e-8:
                        self.positions[index] = self.positions[index + 1] + delta / dist * rest

        self.velocities += (self.positions - old_positions) / max(float(self.config.sim_dt), 1e-6)
        speed = np.linalg.norm(self.velocities, axis=1)
        max_speed = max(float(self.config.wire_max_speed), 1e-6)
        too_fast = speed > max_speed
        if np.any(too_fast):
            self.velocities[too_fast] *= (max_speed / np.maximum(speed[too_fast], 1e-6))[:, None]

    def _anchor_physical_tail(self) -> None:
        count = int(np.clip(self.config.physical_tail_anchor_segments, 0, self.config.wire_segments))
        if count <= 0:
            return
        _center, tangent, n1, n2, _radius = self._local_path_frame(self._segment_progress_hint(count))
        anchor = self.positions[count].copy() if count < self.config.wire_segments else self.positions[-1].copy()
        rest = max(float(self.config.wire_spacing), 1e-6)
        for index in range(count - 1, -1, -1):
            back = float(count - index) * rest
            self.positions[index] = anchor - tangent * back
            self.velocities[index] *= 0.25

    def _stabilize_physical_tip(self, rest: float) -> None:
        count = int(np.clip(self.config.physical_tip_stiff_segments, 3, self.config.wire_segments))
        start = max(self.config.wire_segments - count, 1)
        blend = float(np.clip(self.config.physical_tip_curvature_smoothing, 0.0, 0.95))
        tip_anchor = self.positions[-1].copy()
        for _ in range(max(int(self.config.physical_tip_curvature_iterations), 0)):
            self.positions[-1] = tip_anchor
            for index in range(start, self.config.wire_segments - 1):
                midpoint = 0.5 * (self.positions[index - 1] + self.positions[index + 1])
                self.positions[index] = (1.0 - blend) * self.positions[index] + blend * midpoint
            for index in range(self.config.wire_segments - 2, start - 1, -1):
                delta = self.positions[index] - self.positions[index + 1]
                dist = float(np.linalg.norm(delta))
                if dist > 1e-8:
                    self.positions[index] = self.positions[index + 1] + delta / dist * rest
            self.positions[-1] = tip_anchor
            for index in range(start, self.config.wire_segments - 1):
                delta = self.positions[index] - self.positions[index - 1]
                dist = float(np.linalg.norm(delta))
                if dist > 1e-8:
                    self.positions[index] = self.positions[index - 1] + delta / dist * rest
            self.positions[-1] = tip_anchor

    def _success(self) -> bool:
        return float(np.linalg.norm(self.targets[self.task] - self.tip)) <= self.config.target_radius

    def _failure_reason(self, tactile: Dict) -> Optional[str]:
        if tactile["distance_to_wall"] < -self.config.wall_margin * self.config.wall_contact_failure_margin:
            return "left_vessel"
        if self.boundary_projection_window >= self.config.wall_contact_failure_window:
            if self.boundary_projection_count >= self.config.repeated_projection_limit:
                return "repeated_boundary_projection"
        if self.boundary_projection_count >= self.config.repeated_projection_limit * 2:
            return "repeated_boundary_projection"
        if self.severe_contact_count >= self.config.severe_contact_limit * 4:
            return "persistent_severe_contact"
        wrong = self.targets["right" if self.task == "left" else "left"]
        if float(np.linalg.norm(wrong - self.tip)) <= self.config.target_radius:
            return "wrong_branch"
        return None

    def _reward(self, obs: Dict, failure_reason: Optional[str]) -> float:
        reward = -obs["distance_to_target"] - 0.35 * obs["contact_strength"]
        reward += 0.015 * obs["path_progress"]
        if obs["success"]:
            reward += 30.0
        if failure_reason is not None:
            reward -= 30.0
        return float(reward)

    def _obs_dict(self, done: bool, success: bool, failure_reason: Optional[str]) -> Dict:
        tactile = self._tactile(self.tip)
        target = self.targets[self.task]
        center, tangent, n1, n2, radius = self._local_path_frame(self.path_progress_float)
        lateral = np.array([float(np.dot(self.tip - center, n1)), float(np.dot(self.tip - center, n2))], dtype=np.float32)
        return {
            "instruction": self.instruction,
            "task": self.task,
            "step": self.step_count,
            "tip_pos": self.tip.astype(float).tolist(),
            "heading": self.heading.astype(float).tolist(),
            "target_pos": target.astype(float).tolist(),
            "distance_to_target": float(np.linalg.norm(target - self.tip)),
            "path_progress": float(self.path_progress_float),
            "piper_step": int(self.step_count),
            "piper_insertion_length": float(getattr(self, "piper_insertion_length", 0.0)),
            "boundary_projection_count": int(self.boundary_projection_count),
            "boundary_projection_window": int(self.boundary_projection_window),
            "last_boundary_projection": bool(self.boundary_projection_count > 0),
            "elirobot_pose": self.elite_pose.astype(float).tolist(),
            "elite_tcp_pose_6d": getattr(self, "elite_tcp_pose_6d", self.robot_tool_pose6d("elite")).astype(float).tolist(),
            "magnetic_pose": self.magnetic_pose.astype(float).tolist(),
            "lateral_offset": lateral.astype(float).tolist(),
            "path_tangent": tangent.astype(float).tolist(),
            "local_radius": float(radius),
            "elite_reference_pose": self.elite_reference_paths[self.task][min(int(np.floor(self.path_progress_float)), len(self.elite_reference_paths[self.task]) - 1)].astype(float).tolist(),
            "robot_state": {
                "mode": "mujoco_guided_wire_mvp",
                "piper_pose": self.piper_pose.astype(float).tolist(),
                "elite_pose": self.elite_pose.astype(float).tolist(),
                "piper_tcp_pose_6d": getattr(self, "piper_tcp_pose_6d", self.robot_tool_pose6d("piper")).astype(float).tolist(),
                "elite_tcp_pose_6d": getattr(self, "elite_tcp_pose_6d", self.robot_tool_pose6d("elite")).astype(float).tolist(),
                "piper_tool_world": self.piper_pose.astype(float).tolist(),
                "elite_tool_world": self.elite_pose.astype(float).tolist(),
                "magnetic_effective_world": self.magnetic_pose.astype(float).tolist(),
                "piper_joints": {name: float(value) for name, value in self.piper_joint_values.items()},
                "elite_joints": {name: float(value) for name, value in self.elite_joint_values.items()},
                "wire_segments": int(self.config.wire_segments),
            },
            "controller_state": getattr(self, "controller_state", {}),
            **tactile,
            "done": done,
            "success": success,
            "failure_reason": failure_reason,
        }

    def _gym_obs(self, obs: Dict) -> Dict:
        scene_span = np.maximum(self.scene_max - self.scene_min, 1e-6)
        tip = (np.asarray(obs["tip_pos"], dtype=np.float32) - self.scene_min) / scene_span
        target = (np.asarray(obs["target_pos"], dtype=np.float32) - self.scene_min) / scene_span
        elite = (np.asarray(obs["elirobot_pose"], dtype=np.float32) - self.scene_min) / scene_span
        state = np.array(
            [
                *tip.tolist(),
                *obs["heading"],
                *target.tolist(),
                *obs["contact_normal"],
                obs["distance_to_wall"] / max(self.config.wall_margin, 1e-6),
                obs["contact_strength"],
                float(obs["contact_flag"]),
                float(self.step_count) / max(self.config.max_steps, 1),
                float(self.task == "left"),
                float(self.task == "right"),
                float(obs["path_progress"]) / max(len(self.wire_centerlines[self.task]) - 1, 1),
                obs["piper_insertion_length"] / max(self.config.max_steps * self.config.advance_step, 1e-6),
                *elite.tolist(),
                *np.asarray(obs["lateral_offset"], dtype=np.float32).tolist(),
                *np.asarray(obs["path_tangent"], dtype=np.float32).tolist(),
                float(obs["local_radius"]),
            ],
            dtype=np.float32,
        )
        if self.config.render_observation:
            image = self.render()
        else:
            image = np.zeros((self.config.render_height, self.config.render_width, 3), dtype=np.uint8)
        return {"image": image, "state": state, "instruction_id": 0 if self.task == "left" else 1}

    def _info(self, obs: Dict) -> Dict:
        return {"obs_dict": obs, "instruction": self.instruction, "success": obs["success"], "failure_reason": obs["failure_reason"]}

    def _parse_action(self, action):
        if isinstance(action, dict):
            if "piper_feed" in action and ("elite_tcp_delta_6d" in action or "elite_tcp_pose_6d" in action):
                piper_feed = float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0])
                return (
                    self._piper_joint_vector_from_feed_command(piper_feed),
                    self._elite_joint_vector_from_tcp_action(action),
                )
            if "piper_step_command" in action and ("elite_tcp_delta_6d" in action or "elite_tcp_pose_6d" in action):
                piper_step_command = float(np.asarray(action.get("piper_step_command", 0.0), dtype=np.float32).reshape(-1)[0])
                return (
                    self._piper_joint_vector_from_step_command(piper_step_command),
                    self._elite_joint_vector_from_tcp_action(action),
                )
            if "piper_feed" in action and "elite_joints" in action:
                piper_feed = float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0])
                return (
                    self._piper_joint_vector_from_feed_command(piper_feed),
                    self._robot_joint_vector_from_action("elite", action["elite_joints"]),
                )
            if "piper_step_command" in action and "elite_joints" in action:
                piper_step_command = float(np.asarray(action.get("piper_step_command", 0.0), dtype=np.float32).reshape(-1)[0])
                return (
                    self._piper_joint_vector_from_step_command(piper_step_command),
                    self._robot_joint_vector_from_action("elite", action["elite_joints"]),
                )
            if "piper_joints" in action and "elite_joints" in action:
                return (
                    self._robot_joint_vector_from_action("piper", action["piper_joints"]),
                    self._robot_joint_vector_from_action("elite", action["elite_joints"]),
                )
            if "piper" in action and "elirobot_delta" in action:
                # Legacy compatibility path for older datasets and rollouts.
                piper = float(np.asarray(action.get("piper", 0.0), dtype=np.float32).reshape(-1)[0])
                delta = np.asarray(action.get("elirobot_delta", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                piper = float(np.clip(piper, -1.0, 1.0))
                current_piper_pose = self.robot_tool_world("piper").astype(np.float32)
                feed_axis = self._local_path_frame(self.path_progress_float)[1]
                piper_target_pose = current_piper_pose + feed_axis * piper * float(self.config.advance_step) * float(self.config.piper_advection_scale)
                elite_target_pose = self.elite_pose + delta * float(self.config.elirobot_action_scale)
                return (
                    self._robot_joint_vector_from_action("piper", self._solve_robot_ik("piper", piper_target_pose, max_iters=6, damping=0.04, step_limit=0.18)),
                    self._robot_joint_vector_from_action("elite", self._solve_robot_ik("elite", elite_target_pose, max_iters=5, damping=0.04, step_limit=0.22)),
                )
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(arr) == self.joint_action_dim:
            piper_dim = len(self.piper_joint_names)
            return arr[:piper_dim], arr[piper_dim:]
        if len(arr) < 4:
            raise ValueError("MuJoCo guided wire action must contain joint targets or legacy piper + elite delta values")
        piper = float(np.clip(arr[0], -1.0, 1.0))
        delta = np.asarray(arr[1:4], dtype=np.float32).reshape(3)
        current_piper_pose = self.robot_tool_world("piper").astype(np.float32)
        feed_axis = self._local_path_frame(self.path_progress_float)[1]
        piper_target_pose = current_piper_pose + feed_axis * piper * float(self.config.advance_step) * float(self.config.piper_advection_scale)
        elite_target_pose = self.elite_pose + delta * float(self.config.elirobot_action_scale)
        return (
            self._robot_joint_vector_from_action("piper", self._solve_robot_ik("piper", piper_target_pose, max_iters=6, damping=0.04, step_limit=0.18)),
            self._robot_joint_vector_from_action("elite", self._solve_robot_ik("elite", elite_target_pose, max_iters=5, damping=0.04, step_limit=0.22)),
        )

    def render(self):
        return self.render_camera("perspective")

    def render_camera_pair(self, size: int = 512) -> Dict[str, np.ndarray]:
        return {
            "side": cv2.resize(self.render_camera("side"), (size, size), interpolation=cv2.INTER_AREA),
            "top": cv2.resize(self.render_camera("top"), (size, size), interpolation=cv2.INTER_AREA),
        }

    def _apply_render_domain_postprocess(self, image: np.ndarray, camera: str) -> np.ndarray:
        preset = str(getattr(self.config, "render_domain_preset", "default") or "default")
        if preset != "branchs_like_v1":
            return image

        img = image.astype(np.float32) / 255.0
        original = img.copy()
        luminance = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        img = luminance[:, :, None] + 0.84 * (img - luminance[:, :, None])
        img *= 1.02
        img[:, :, 0] *= 0.96
        img[:, :, 1] = img[:, :, 1] * 1.03 + 0.012
        img[:, :, 2] *= 0.98

        height, width = img.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx - 0.5 * width) / max(0.5 * width, 1.0)
        ny = (yy - 0.5 * height) / max(0.5 * height, 1.0)
        vignette = np.clip(1.02 - 0.08 * (nx * nx + ny * ny), 0.90, 1.02)
        img *= vignette[:, :, None]

        red_mask = (
            (original[:, :, 2] > 0.50)
            & (original[:, :, 2] > original[:, :, 1] * 1.45)
            & (original[:, :, 2] > original[:, :, 0] * 1.35)
        )
        orange_surface_mask = (
            (img[:, :, 2] > 0.22)
            & (img[:, :, 2] > img[:, :, 1] * 1.10)
            & (img[:, :, 1] > img[:, :, 0] * 1.05)
            & (~red_mask)
        )
        if np.any(orange_surface_mask):
            surface_gray = (
                0.114 * img[orange_surface_mask, 0]
                + 0.587 * img[orange_surface_mask, 1]
                + 0.299 * img[orange_surface_mask, 2]
            )
            img[orange_surface_mask, 0] = 0.72 * img[orange_surface_mask, 0] + 0.28 * surface_gray
            img[orange_surface_mask, 1] = np.maximum(img[orange_surface_mask, 1], surface_gray * 1.02)
            img[orange_surface_mask, 2] = 0.62 * img[orange_surface_mask, 2] + 0.38 * surface_gray
        if np.any(red_mask):
            img[red_mask, 2] = np.maximum(img[red_mask, 2], original[red_mask, 2] * 0.88)
            img[red_mask, 1] *= 0.78
            img[red_mask, 0] *= 0.78

        img = np.clip(img, 0.0, 1.0)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.25, sigmaY=0.25)
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def render_camera(
        self,
        camera: str = "perspective",
        hide_robot_visuals: bool = False,
        diagnostic_overlay: bool = False,
    ) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.config.render_height, width=self.config.render_width)
        hidden_geom_ids: list[int] = []
        hidden_alpha: np.ndarray | None = None
        if hide_robot_visuals:
            hidden_geom_ids = self._robot_visual_geom_ids()
            if hidden_geom_ids:
                hidden_alpha = self.model.geom_rgba[hidden_geom_ids, 3].copy()
                self.model.geom_rgba[hidden_geom_ids, 3] = 0.0
        diagnostic_geom_ids: list[int] = []
        diagnostic_alpha: np.ndarray | None = None
        if diagnostic_overlay:
            diagnostic_geom_ids = self._diagnostic_marker_geom_ids()
            if diagnostic_geom_ids:
                diagnostic_alpha = self.model.geom_rgba[diagnostic_geom_ids, 3].copy()
                self.model.geom_rgba[diagnostic_geom_ids, 3] = 0.95
        self._configure_free_camera(camera)
        try:
            self.renderer.update_scene(self.data, camera=self.camera)
            rgb = self.renderer.render()
        finally:
            if hidden_geom_ids and hidden_alpha is not None:
                self.model.geom_rgba[hidden_geom_ids, 3] = hidden_alpha
            if diagnostic_geom_ids and diagnostic_alpha is not None:
                self.model.geom_rgba[diagnostic_geom_ids, 3] = diagnostic_alpha
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = self._apply_render_domain_postprocess(bgr, camera)
        if str(self.config.wire_visual_mode) == "both":
            bgr = self._overlay_wire_line(bgr)
        if self.config.render_overlay:
            return self._overlay_wire_and_markers(bgr)
        return bgr

    def _robot_visual_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        for geom_id in range(int(self.model.ngeom)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            is_kinematic_robot = (name.startswith("piper_") or name.startswith("elite_")) and name.endswith("_geom")
            is_static_robot = name.startswith("robot_") and name.endswith("_geom")
            if is_kinematic_robot or is_static_robot:
                geom_ids.append(geom_id)
        return geom_ids

    def _diagnostic_marker_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        for name in (
            "diagnostic_tip_marker_geom",
            "diagnostic_magnetic_marker_geom",
            "diagnostic_elite_marker_geom",
        ):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                geom_ids.append(int(geom_id))
        return geom_ids

    def _overlay_wire_line(self, image: np.ndarray) -> np.ndarray:
        if len(self.positions) < 2:
            return image
        pts = self._project_world_points(np.asarray(self.positions, dtype=np.float32), image.shape[1], image.shape[0])
        if len(pts) < 2:
            return image
        overlay = image.copy()
        line_width = max(int(self.config.wire_line_width), 1)
        core_width = max(int(self.config.wire_line_core_width), 1)
        cv2.polylines(overlay, [pts], False, (205, 165, 120), line_width, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], False, (245, 238, 224), core_width, cv2.LINE_AA)
        return overlay

    def _overlay_wire_and_markers(self, image: np.ndarray, include_magnetic: bool = False) -> np.ndarray:
        if len(self.positions) < 2:
            return image
        overlay = image.copy()
        pts = self._project_world_points(np.asarray(self.positions, dtype=np.float32), image.shape[1], image.shape[0])
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, (255, 140, 25), 5, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], False, (255, 245, 235), 1, cv2.LINE_AA)
        if len(self.guidewire_points) > 1:
            trail = self._project_world_points(np.asarray(self.guidewire_points, dtype=np.float32), image.shape[1], image.shape[0])
            if len(trail) >= 2:
                cv2.polylines(overlay, [trail], False, (255, 205, 95), 1, cv2.LINE_AA)
        if len(self.elite_points) > 1:
            elite_trail = self._project_world_points(np.asarray(self.elite_points, dtype=np.float32), image.shape[1], image.shape[0])
            if len(elite_trail) >= 2:
                cv2.polylines(overlay, [elite_trail], False, (70, 180, 255), 1, cv2.LINE_AA)
        elite = self._project_world_points(self.elite_pose[None, :], image.shape[1], image.shape[0])
        magnetic = self._project_world_points(self.magnetic_pose[None, :], image.shape[1], image.shape[0])
        tip = self._project_world_points(self.tip[None, :], image.shape[1], image.shape[0])
        if len(elite):
            cv2.circle(overlay, tuple(elite[0]), 6, (40, 150, 255), -1, cv2.LINE_AA)
        if include_magnetic and len(magnetic):
            cv2.circle(overlay, tuple(magnetic[0]), 7, (40, 220, 80), -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(magnetic[0]), 10, (15, 80, 20), 2, cv2.LINE_AA)
        if len(tip):
            cv2.circle(overlay, tuple(tip[0]), 5, (0, 0, 220), -1, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.92, image, 0.08, 0.0)

    def _project_world_points(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        lookat = np.asarray(self.camera.lookat, dtype=np.float32)
        distance = max(float(self.camera.distance), 1e-6)
        az = math.radians(float(self.camera.azimuth))
        el = math.radians(float(self.camera.elevation))
        forward = np.array(
            [
                math.cos(el) * math.sin(az),
                -math.cos(el) * math.cos(az),
                math.sin(el),
            ],
            dtype=np.float32,
        )
        forward = _normalize(forward)
        cam_pos = lookat - forward * distance
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = _normalize(right)
        up = np.cross(right, forward)
        up = _normalize(up)

        rel = points - cam_pos[None, :]
        x = rel @ right
        y = rel @ up
        z = rel @ forward
        valid = z > 1e-4
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)

        focal = 0.92 * min(width, height)
        px = np.column_stack([width * 0.5 + focal * (x[valid] / z[valid]), height * 0.52 - focal * (y[valid] / z[valid])])
        px[:, 0] = np.clip(px[:, 0], 0, width - 1)
        px[:, 1] = np.clip(px[:, 1], 0, height - 1)
        return px.astype(np.int32)

    def _configure_free_camera(self, camera: str) -> None:
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        path = self.paths[self.task]
        radius = float(np.max(self.radii[self.task])) if self.task in self.radii else 0.0
        margin = max(float(self.config.camera_roi_margin), radius * 2.2)
        lo = np.min(path, axis=0) - margin
        hi = np.max(path, axis=0) + margin
        center = 0.5 * (lo + hi)
        span = np.maximum(hi - lo, 1e-6)
        distance = max(float(np.max(span)) * self.config.camera_distance_scale, self.config.camera_min_distance)
        if camera == "top":
            preset = self.camera_config.get("top", {})
            offset = np.asarray(preset.get("lookat_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.camera.lookat[:] = center + offset
            self.camera.distance = distance * float(preset.get("distance_scale", 0.98))
            self.camera.azimuth = float(preset.get("azimuth", 145.0))
            self.camera.elevation = float(preset.get("elevation", -62.0))
        elif camera == "side":
            preset = self.camera_config.get("side", {})
            offset = np.asarray(preset.get("lookat_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.camera.lookat[:] = center + offset
            self.camera.distance = distance * float(preset.get("distance_scale", 0.72))
            self.camera.azimuth = float(preset.get("azimuth", 180.0))
            self.camera.elevation = float(preset.get("elevation", -8.0))
        elif camera == "overview":
            preset = self.camera_config.get("overview", {})
            offset = np.asarray(preset.get("lookat_offset", [0.0, 0.0, 0.035]), dtype=np.float32)
            self.camera.lookat[:] = center + offset
            self.camera.distance = distance * float(preset.get("distance_scale", 1.18))
            self.camera.azimuth = float(preset.get("azimuth", 180.0))
            self.camera.elevation = float(preset.get("elevation", -58.0))
        else:
            preset = self.camera_config.get("perspective", {})
            offset = np.asarray(preset.get("lookat_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.camera.lookat[:] = center + offset
            self.camera.distance = distance * float(preset.get("distance_scale", 0.66))
            self.camera.azimuth = float(preset.get("azimuth", 145.0 if self.task == "left" else 35.0))
            self.camera.elevation = float(preset.get("elevation", -24.0))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class MuJoCoGuidedWireExpert:
    def __init__(self, rng_seed: int = 0, steer_noise: float = 0.02):
        self.rng = np.random.default_rng(rng_seed)
        self.steer_noise = steer_noise
        self.piper_feed_direction = 1.0

    def act(self, env: MuJoCoGuidedWireEnv) -> Dict[str, np.ndarray]:
        center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
        lookahead = self._lookahead(env)
        lead_local = np.array(
            [
                float(np.dot(lookahead - center, n1)),
                float(np.dot(lookahead - center, n2)),
            ],
            dtype=np.float32,
        )
        tip_local = np.array(
            [
                float(np.dot(env.tip - center, n1)),
                float(np.dot(env.tip - center, n2)),
            ],
            dtype=np.float32,
        )
        follow_gain = 0.08 if env.task == "left" else 0.10
        desired_tip_local = (-1.25 * tip_local) + (follow_gain * lead_local)
        tactile = env._tactile(env.tip)
        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            contact_local = np.array([float(np.dot(normal, n1)), float(np.dot(normal, n2))], dtype=np.float32)
            contact_norm = float(np.linalg.norm(contact_local))
            if contact_norm > 1e-6:
                desired_tip_local = desired_tip_local - (0.95 * tactile["contact_strength"] * radius) * (
                    contact_local / contact_norm
                )
            desired_tip_local = desired_tip_local - 0.30 * tip_local
        desired_local = desired_tip_local
        max_local = min(env.config.magnet_range * 0.16, radius * 0.18)
        desired_norm = float(np.linalg.norm(desired_local))
        if desired_norm > max_local and desired_norm > 1e-6:
            desired_local = desired_local / desired_norm * max_local
        tangent_offset = min(radius * 0.12, 0.010)
        desired_elite = env._front_up_magnetic_target(
            center + n1 * desired_local[0] + n2 * desired_local[1],
            tangent,
            forward_offset=tangent_offset,
        )
        if self.steer_noise > 0:
            desired_elite = desired_elite + self.rng.normal(0.0, self.steer_noise * env.config.elirobot_action_scale, size=3).astype(np.float32)

        piper_cmd = 1.0
        penetrated_wall = tactile["distance_to_wall"] < -env.config.wall_margin * 0.20
        hard_contact = tactile["contact_strength"] > 0.92
        soft_contact = tactile["distance_to_wall"] < env.config.wall_margin * 0.20 or tactile["contact_strength"] > 0.80
        if penetrated_wall or hard_contact:
            piper_cmd = 0.16
        elif soft_contact:
            piper_cmd = 0.30
        elif env.task == "left":
            piper_cmd = 0.86
        else:
            piper_cmd = 0.90

        joint_signal_limit = min(float(env.piper_joint_limits.get("joint7", (0.0, 0.035))[1]), -float(env.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]))
        insertion_limit = min(float(env.config.piper_insertion_window), max(joint_signal_limit - float(env.piper_feed_home_signal), 1e-6))
        current_insertion = float(getattr(env, "piper_insertion_length", 0.0))
        lower_band = insertion_limit * 0.55
        upper_band = insertion_limit * 0.93
        if penetrated_wall or hard_contact or current_insertion >= upper_band:
            self.piper_feed_direction = -1.0
        elif current_insertion <= lower_band:
            self.piper_feed_direction = 1.0

        feed_step = float(piper_cmd) * float(env.config.advance_step) * float(env.config.piper_advection_scale)
        feed_step = max(feed_step, insertion_limit * 0.12)
        if self.piper_feed_direction < 0.0:
            feed_step = max(feed_step * 2.2, insertion_limit * 0.24)
        desired_insertion = current_insertion + self.piper_feed_direction * feed_step
        desired_insertion = float(np.clip(desired_insertion, 0.0, insertion_limit))
        piper_joints = env._piper_joint_vector_from_feed(float(env.piper_feed_home_signal) + desired_insertion)
        elite_joints = env._solve_robot_ik("elite", desired_elite, max_iters=5, damping=0.04, step_limit=0.22)
        return {
            "piper_joints": env._robot_joint_dict("piper", piper_joints),
            "elite_joints": env._robot_joint_dict("elite", env._robot_joint_vector_from_action("elite", elite_joints)),
        }

    def _lookahead(self, env: MuJoCoGuidedWireEnv) -> np.ndarray:
        path = env.wire_centerlines[env.task]
        idx = int(np.clip(np.floor(env.path_progress_float), 0, len(path) - 1))
        ahead = 5 if env.task == "left" else 7
        idx = min(idx + max(ahead, 1), len(path) - 1)
        return path[idx]


class MuJoCoMagneticGuideExpert:
    def __init__(
        self,
        rng_seed: int = 0,
        steer_noise: float = 0.0,
        lookahead_points: int = 8,
        elite_ahead: float = 0.018,
        lateral_gain: float = 0.75,
        piper_cmd: float = 0.34,
    ):
        self.rng = np.random.default_rng(rng_seed)
        self.steer_noise = steer_noise
        self.lookahead_points = lookahead_points
        self.elite_ahead = elite_ahead
        self.lateral_gain = lateral_gain
        self.piper_cmd = piper_cmd
        self.piper_feed_direction = 1.0

    def act(self, env: MuJoCoGuidedWireEnv) -> Dict[str, np.ndarray]:
        center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
        lookahead = self._lookahead(env)
        desired_direction = lookahead - env.tip
        if float(np.linalg.norm(desired_direction)) < 1e-6:
            desired_direction = tangent.copy()
        desired_direction = desired_direction / max(float(np.linalg.norm(desired_direction)), 1e-6)

        tip_offset = env.tip - center
        tip_local = np.array([float(np.dot(tip_offset, n1)), float(np.dot(tip_offset, n2))], dtype=np.float32)
        correction = -n1 * tip_local[0] - n2 * tip_local[1]
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > 1e-6:
            correction = correction / correction_norm * min(radius * self.lateral_gain, radius * 0.55)

        desired_elite = env._front_up_magnetic_target(
            env.tip + correction,
            desired_direction,
            forward_offset=self.elite_ahead,
        )
        if self.steer_noise > 0.0:
            desired_elite = desired_elite + self.rng.normal(0.0, self.steer_noise * env.config.elirobot_action_scale, size=3).astype(np.float32)

        piper_vec = self._piper_action(env)
        elite_joints = env._solve_robot_ik("elite", desired_elite, max_iters=7, damping=0.04, step_limit=0.20)
        return {
            "piper_joints": env._robot_joint_dict("piper", piper_vec),
            "elite_joints": env._robot_joint_dict("elite", env._robot_joint_vector_from_action("elite", elite_joints)),
        }

    def _piper_action(self, env: MuJoCoGuidedWireEnv) -> np.ndarray:
        tactile = env._tactile(env.tip)
        joint_signal_limit = min(float(env.piper_joint_limits.get("joint7", (0.0, 0.035))[1]), -float(env.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]))
        insertion_limit = min(float(env.config.piper_insertion_window), max(joint_signal_limit - float(env.piper_feed_home_signal), 1e-6))
        current_insertion = float(getattr(env, "piper_insertion_length", 0.0))
        hard_contact = tactile["contact_strength"] > 0.85 or tactile["distance_to_wall"] < -env.config.wall_margin * 0.10
        soft_contact = tactile["contact_strength"] > 0.55 or tactile["distance_to_wall"] < env.config.wall_margin * 0.15
        if hard_contact or current_insertion >= insertion_limit * 0.90:
            self.piper_feed_direction = -1.0
        elif current_insertion <= insertion_limit * 0.50:
            self.piper_feed_direction = 1.0
        piper_cmd = 0.08 if hard_contact else (0.16 if soft_contact else self.piper_cmd)
        feed_step = piper_cmd * float(env.config.advance_step) * float(env.config.piper_advection_scale)
        feed_step = max(feed_step, insertion_limit * 0.06)
        if self.piper_feed_direction < 0.0:
            feed_step = max(feed_step * 1.8, insertion_limit * 0.14)
        desired_insertion = float(np.clip(current_insertion + self.piper_feed_direction * feed_step, 0.0, insertion_limit))
        return env._piper_joint_vector_from_feed(float(env.piper_feed_home_signal) + desired_insertion)

    def _lookahead(self, env: MuJoCoGuidedWireEnv) -> np.ndarray:
        path = env.wire_centerlines[env.task]
        idx = int(np.clip(np.floor(env.path_progress_float), 0, len(path) - 1))
        idx = min(idx + max(int(self.lookahead_points), 1), len(path) - 1)
        return path[idx]


class MuJoCoRoutePlanGuideExpert:
    """Open-loop route-plan expert for Route 2 formal-validity probes.

    This expert intentionally avoids exact tip position, wall/contact distance,
    and tactile feedback. It follows a pre-registered route with a scheduled
    progress variable and uses robot IK to place the Elite tool near the planned
    route point.
    """

    def __init__(
        self,
        plan_step: float = 0.32,
        elite_ahead: float = 0.018,
        piper_cmd: float = 0.34,
        piper_command_period: int = 125,
        piper_command_width: int = 1,
        route_plan_command_phase_lock: bool = False,
        piper_command_as_event: bool = False,
    ):
        self.plan_step = float(plan_step)
        self.elite_ahead = float(elite_ahead)
        self.piper_cmd = float(piper_cmd)
        self.piper_command_period = max(int(piper_command_period), 1)
        self.piper_command_width = max(int(piper_command_width), 1)
        self.route_plan_command_phase_lock = bool(route_plan_command_phase_lock)
        self.piper_command_as_event = bool(piper_command_as_event)
        self.plan_progress: Optional[float] = None
        self.piper_feed_direction = 1.0

    def act(self, env: MuJoCoGuidedWireEnv) -> Dict[str, np.ndarray]:
        if self.plan_progress is None:
            self.plan_progress = float(env.path_progress_float)
        piper_step_command = self._piper_step_command(env)
        piper_feed_command = float(np.clip(piper_step_command * self.piper_cmd, -1.0, 1.0))
        piper_execution_active = piper_step_command > 0
        if self.piper_command_as_event:
            controller_state = getattr(env, "controller_state", {}) or {}
            piper_execution_active = piper_execution_active or bool(controller_state.get("piper_busy", False))
        plan_step = self.plan_step
        if self.route_plan_command_phase_lock:
            if self.piper_command_as_event:
                primitive_steps = max(int(getattr(env.config, "piper_primitive_steps", 1)), 1)
                duty = min(max(primitive_steps / self.piper_command_period, 1e-6), 1.0)
            else:
                duty = min(max(self.piper_command_width / self.piper_command_period, 1e-6), 1.0)
            plan_step = (self.plan_step / duty) if piper_execution_active else 0.0
        self.plan_progress = float(
            np.clip(self.plan_progress + plan_step, 0.0, len(env.wire_centerlines[env.task]) - 1)
        )

        center, tangent, _n1, _n2, _radius = env._local_path_frame(self.plan_progress)
        desired_elite = env._front_up_magnetic_target(center, tangent, forward_offset=self.elite_ahead)

        piper_vec = env._piper_joint_vector_from_feed_command(piper_feed_command)
        elite_joints = env._solve_robot_ik("elite", desired_elite, max_iters=7, damping=0.04, step_limit=0.20)
        return {
            "piper_joints": env._robot_joint_dict("piper", piper_vec),
            "piper_feed_command": piper_feed_command,
            "piper_step_command": piper_step_command,
            "elite_joints": env._robot_joint_dict("elite", env._robot_joint_vector_from_action("elite", elite_joints)),
            "elite_desired_pose_3d": desired_elite.astype(float).tolist(),
            "elite_plan_progress": float(self.plan_progress),
            "elite_plan_tangent": tangent.astype(float).tolist(),
        }

    def _piper_action(self, env: MuJoCoGuidedWireEnv) -> np.ndarray:
        joint_signal_limit = min(
            float(env.piper_joint_limits.get("joint7", (0.0, 0.035))[1]),
            -float(env.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]),
        )
        insertion_limit = min(float(env.config.piper_insertion_window), max(joint_signal_limit - float(env.piper_feed_home_signal), 1e-6))
        current_insertion = float(getattr(env, "piper_insertion_length", 0.0))
        if current_insertion >= insertion_limit * 0.90:
            self.piper_feed_direction = -1.0
        elif current_insertion <= insertion_limit * 0.50:
            self.piper_feed_direction = 1.0
        feed_step = self.piper_cmd * float(env.config.advance_step) * float(env.config.piper_advection_scale)
        feed_step = max(feed_step, insertion_limit * 0.06)
        if self.piper_feed_direction < 0.0:
            feed_step = max(feed_step * 1.8, insertion_limit * 0.14)
        desired_insertion = float(np.clip(current_insertion + self.piper_feed_direction * feed_step, 0.0, insertion_limit))
        return env._piper_joint_vector_from_feed(float(env.piper_feed_home_signal) + desired_insertion)

    def _piper_step_command(self, env: MuJoCoGuidedWireEnv) -> int:
        phase = int(getattr(env, "step_count", 0)) % self.piper_command_period
        if self.piper_command_as_event:
            return 1 if phase == 0 else 0
        return 1 if phase < self.piper_command_width else 0
