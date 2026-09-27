from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from yourdfpy import URDF

from tools.visualize_dual_robot_scene import pose_matrix, translation_matrix


def _as_joint_map(names: list[str], values: Optional[Mapping[str, float] | list[float]]) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(name): float(value) for name, value in values.items()}
    if len(values) > len(names):
        raise ValueError(f"Got {len(values)} joint values for {len(names)} movable joints")
    return {name: float(value) for name, value in zip(names, values)}


def _point_from_transform(transform: np.ndarray) -> list[float]:
    return [float(x) for x in transform[:3, 3]]


@dataclass
class RobotToolKinematics:
    name: str
    urdf_path: Path
    base_pose: list[float]
    tool_link: str
    tool_offset: list[float]

    def __post_init__(self) -> None:
        self.urdf_path = Path(self.urdf_path).resolve()
        self.robot = URDF.load(str(self.urdf_path))
        self.movable_joint_names = [joint.name for joint in self.robot.robot.joints if joint.type != "fixed"]

    @property
    def base_transform(self) -> np.ndarray:
        return pose_matrix(self.base_pose)

    def joint_map(self, values: Optional[Mapping[str, float] | list[float]] = None) -> dict[str, float]:
        return _as_joint_map(self.movable_joint_names, values)

    def link_transform(self, link_name: Optional[str] = None, joints: Optional[Mapping[str, float] | list[float]] = None) -> np.ndarray:
        self.robot.update_cfg(self.joint_map(joints))
        link = link_name or self.tool_link
        return self.base_transform @ np.asarray(self.robot.get_transform(link), dtype=float)

    def tool_transform(self, joints: Optional[Mapping[str, float] | list[float]] = None) -> np.ndarray:
        return self.link_transform(self.tool_link, joints) @ translation_matrix(self.tool_offset)

    def tool_position(self, joints: Optional[Mapping[str, float] | list[float]] = None) -> list[float]:
        return _point_from_transform(self.tool_transform(joints))

    def offset_from_world_point(
        self,
        world_point: list[float],
        link_name: Optional[str] = None,
        joints: Optional[Mapping[str, float] | list[float]] = None,
    ) -> list[float]:
        link_tf = self.link_transform(link_name or self.tool_link, joints)
        point = np.ones(4, dtype=float)
        point[:3] = np.asarray(world_point, dtype=float)
        local = np.linalg.inv(link_tf) @ point
        return [float(x) for x in local[:3]]

    def summary(self, joints: Optional[Mapping[str, float] | list[float]] = None) -> dict:
        return {
            "name": self.name,
            "urdf": self.urdf_path.as_posix(),
            "base_pose": [float(x) for x in self.base_pose],
            "movable_joints": self.movable_joint_names,
            "tool_link": self.tool_link,
            "tool_offset": [float(x) for x in self.tool_offset],
            "tool_world": self.tool_position(joints),
        }


@dataclass
class DualRobotKinematics:
    piper: RobotToolKinematics
    elite: RobotToolKinematics

    @classmethod
    def from_scene_config(cls, config_path: str | Path) -> "DualRobotKinematics":
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        root = (config_path.parents[2] / config.get("assets_root", "robot_assets/standardized")).resolve()
        if not root.exists():
            root = Path(config.get("assets_root", "robot_assets/standardized")).resolve()

        piper_urdf = root / config.get("piper_urdf", "piper_description/urdf/piper_description.urdf")
        elite_urdf = root / config.get("elite_urdf", "elite_description/urdf/ec66_description.urdf")

        piper = RobotToolKinematics(
            name="piper",
            urdf_path=piper_urdf,
            base_pose=config["piper_pose"],
            tool_link=config.get("piper_tool_link", "gripper_base"),
            tool_offset=config.get("piper_tool_offset", [0.0, 0.0, 0.0]),
        )
        if "piper_tool_offset" not in config and "piper_tool_point" in config:
            piper.tool_offset = piper.offset_from_world_point(config["piper_tool_point"])

        elite = RobotToolKinematics(
            name="elite",
            urdf_path=elite_urdf,
            base_pose=config["elite_pose"],
            tool_link=config.get("elite_tool_link", "flan"),
            tool_offset=config.get("elite_magnet_offset", [0.0, 0.0, 0.0]),
        )
        return cls(piper=piper, elite=elite)

    def summary(
        self,
        piper_joints: Optional[Mapping[str, float] | list[float]] = None,
        elite_joints: Optional[Mapping[str, float] | list[float]] = None,
    ) -> dict:
        return {
            "piper": self.piper.summary(piper_joints),
            "elite": self.elite.summary(elite_joints),
        }
