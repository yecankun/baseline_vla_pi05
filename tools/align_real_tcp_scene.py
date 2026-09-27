from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_dual_robot_scene import (
    get_urdf_link_transform_with_joints,
    load_real_tcp_paths,
    load_urdf_meshes,
    load_vessel,
    make_magnet_tool,
    make_marker,
    parse_joint_config,
    pose_matrix,
    translation_matrix,
    trimesh_to_open3d,
)


DEFAULT_CONFIG = "simulation_output/robot_scene_mvp/scene_config.json"
DEFAULT_ALIGNMENT = "simulation_output/robot_scene_mvp/real_tcp_alignment.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return mat


def scale_matrix(scale: float) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[0, 0] = scale
    mat[1, 1] = scale
    mat[2, 2] = scale
    return mat


def translation(values: list[float]) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = values
    return mat


def make_polyline(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points, dtype=np.float64, copy=True)),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32, copy=True)),
    )
    geom.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return geom


def load_alignment(path: Path, defaults: dict) -> tuple[dict, bool]:
    if not path.exists():
        return defaults, False
    data = load_json(path)
    merged = {**defaults, **data}
    merged["tcp_offset"] = data.get("tcp_offset", defaults["tcp_offset"])
    return merged, True


class StaticRobotScene:
    def __init__(self, scene: rendering.Open3DScene, config: dict, piper_joint_overrides: dict, elite_joint_overrides: dict):
        self.scene = scene
        self.config = config
        self.piper_joint_overrides = piper_joint_overrides
        self.elite_joint_overrides = elite_joint_overrides

    def add(self) -> None:
        assets_root = (REPO_ROOT / self.config.get("assets_root", "robot_assets/standardized")).resolve()
        piper_urdf = assets_root / self.config["piper_urdf"]
        elite_urdf = assets_root / self.config["elite_urdf"]
        piper_pose = pose_matrix(self.config["piper_pose"])
        elite_pose = pose_matrix(self.config["elite_pose"])
        piper_joints = {**self.config.get("piper_joints", {}), **self.piper_joint_overrides}
        elite_joints = {**self.config.get("elite_joints", {}), **self.elite_joint_overrides}

        robot_mat = rendering.MaterialRecord()
        robot_mat.shader = "defaultLit"

        piper_mat = rendering.MaterialRecord()
        piper_mat.shader = "defaultLit"
        piper_mat.base_color = [0.23, 0.43, 0.88, 1.0]
        for index, mesh in enumerate(load_urdf_meshes(piper_urdf, piper_pose, (80, 135, 230, 255), piper_joints)):
            self.scene.add_geometry(f"piper_{index:02d}", trimesh_to_open3d(mesh), piper_mat)

        elite_mat = rendering.MaterialRecord()
        elite_mat.shader = "defaultLit"
        elite_mat.base_color = [0.92, 0.42, 0.22, 1.0]
        for index, mesh in enumerate(load_urdf_meshes(elite_urdf, elite_pose, (235, 130, 70, 255), elite_joints)):
            self.scene.add_geometry(f"elite_{index:02d}", trimesh_to_open3d(mesh), elite_mat)

        piper_tool_transform = get_urdf_link_transform_with_joints(
            piper_urdf,
            self.config.get("piper_tool_link", "gripper_base"),
            piper_joints,
        )
        piper_tool_center = (
            piper_pose @ piper_tool_transform @ translation_matrix(self.config.get("piper_tool_offset", [0.0, 0.0, 0.0]))
        )[:3, 3].tolist()
        elite_tool_transform = get_urdf_link_transform_with_joints(
            elite_urdf,
            self.config.get("elite_tool_link", "flan"),
            elite_joints,
        )
        elite_magnet_transform = (
            elite_pose @ elite_tool_transform @ translation_matrix(self.config.get("elite_magnet_offset", [0.0, 0.0, 0.0]))
        )

        piper_tool_mat = rendering.MaterialRecord()
        piper_tool_mat.shader = "defaultLit"
        piper_tool_mat.base_color = [0.05, 0.85, 0.25, 1.0]
        self.scene.add_geometry(
            "piper_tool_point",
            trimesh_to_open3d(make_marker(piper_tool_center, 0.025, (30, 210, 80, 255))),
            piper_tool_mat,
        )

        magnet_mat = rendering.MaterialRecord()
        magnet_mat.shader = "defaultLit"
        magnet_mat.base_color = [0.95, 0.12, 0.08, 1.0]
        self.scene.add_geometry("elite_magnet_tool", trimesh_to_open3d(make_magnet_tool(elite_magnet_transform)), magnet_mat)

        axis_mat = rendering.MaterialRecord()
        axis_mat.shader = "defaultLit"
        self.scene.add_geometry("world_axis", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25), axis_mat)


class RealTcpAlignmentApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = (REPO_ROOT / args.config).resolve()
        self.alignment_path = (REPO_ROOT / args.alignment).resolve()
        self.config = load_json(self.config_path)
        self.defaults = {
            "vessel_scale": float(self.config.get("vessel_scale", args.vessel_scale)),
            "tcp_yaw": float(args.tcp_yaw),
            "tcp_offset": [0.0, 0.0, 0.0],
            "tcp_stride": int(args.tcp_stride),
            "tcp_branch": args.tcp_branch,
        }
        self.alignment, self.loaded_existing_alignment = load_alignment(self.alignment_path, self.defaults)
        self.vessel_scale = float(self.alignment["vessel_scale"])
        self.tcp_yaw = float(self.alignment["tcp_yaw"])
        self.tcp_offset = [float(x) for x in self.alignment["tcp_offset"]]
        self.labels: dict[str, gui.Label] = {}

        self.raw_vessel = load_vessel((REPO_ROOT / self.config["vessel"]).resolve(), 1.0, (120, 150, 190, 115))
        self.vessel_center = self.raw_vessel.bounds.mean(axis=0)
        self.paths = load_real_tcp_paths((REPO_ROOT / args.root).resolve(), args.tcp_branch, args.tcp_stride)
        if not self.paths:
            raise RuntimeError(f"No real TCP paths found under {args.root}")
        self.raw_tcp_points = np.vstack([item["points_m"] for item in self.paths])
        self.raw_tcp_center = self.raw_tcp_points.mean(axis=0)
        self.initial_center_offset = self.vessel_center * self.vessel_scale - self.raw_tcp_center
        if not self.loaded_existing_alignment:
            for axis in range(3):
                self.tcp_offset[axis] += float(self.initial_center_offset[axis])

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("Real TCP / Vessel Alignment", 1480, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.94, 0.95, 0.96, 1.0])
        self.window.add_child(self.scene_widget)
        self.panel = gui.ScrollableVert(8, gui.Margins(12, 12, 12, 12))
        self.window.add_child(self.panel)
        self.status = gui.Label("")

        self._build_scene()
        self._build_panel()
        self.window.set_on_layout(self._on_layout)
        self._apply_transforms()
        self._reset_view()

    def _build_scene(self) -> None:
        vessel_mat = rendering.MaterialRecord()
        vessel_mat.shader = "defaultLitTransparency"
        vessel_mat.base_color = [0.40, 0.56, 0.82, 0.42]
        self.scene_widget.scene.add_geometry("vessel", trimesh_to_open3d(self.raw_vessel), vessel_mat)

        StaticRobotScene(
            self.scene_widget.scene,
            self.config,
            parse_joint_config(self.args.piper_joint),
            parse_joint_config(self.args.elite_joint),
        ).add()

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = float(self.args.line_width)
        colors = {"branch1": (1.0, 0.05, 0.05), "branch2": (0.05, 0.22, 1.0)}
        for index, item in enumerate(self.paths):
            name = f"tcp_{index:03d}_{item['branch']}_{item['path']}"
            self.scene_widget.scene.add_geometry(name, make_polyline(item["points_m"], colors.get(item["branch"], (0.1, 0.8, 0.3))), line_mat)

    def _build_panel(self) -> None:
        self.panel.add_child(gui.Label("Vessel"))
        self._add_slider("vessel_scale", "scale", self.vessel_scale, 0.02, 0.20, self._set_vessel_scale)
        self.panel.add_child(gui.Label(""))

        self.panel.add_child(gui.Label("Real TCP"))
        self._add_slider("tcp_yaw", "yaw", self.tcp_yaw, -math.pi, math.pi, self._set_tcp_yaw)
        self._add_slider("tcp_x", "x", self.tcp_offset[0], -0.8, 0.8, lambda value: self._set_tcp_offset(0, value))
        self._add_slider("tcp_y", "y", self.tcp_offset[1], -0.8, 0.8, lambda value: self._set_tcp_offset(1, value))
        self._add_slider("tcp_z", "z", self.tcp_offset[2], -0.8, 0.8, lambda value: self._set_tcp_offset(2, value))
        self.panel.add_child(gui.Label(""))

        center_btn = gui.Button("Center TCP To Vessel")
        center_btn.set_on_clicked(self._center_tcp_to_vessel)
        self.panel.add_child(center_btn)

        reset_yaw_btn = gui.Button("Yaw = pi")
        reset_yaw_btn.set_on_clicked(self._set_yaw_pi)
        self.panel.add_child(reset_yaw_btn)

        save_btn = gui.Button("Save Alignment")
        save_btn.set_on_clicked(self._save_alignment)
        self.panel.add_child(save_btn)

        save_scene_btn = gui.Button("Save Vessel Scale To Scene Config")
        save_scene_btn.set_on_clicked(self._save_scene_config)
        self.panel.add_child(save_scene_btn)

        reset_view_btn = gui.Button("Reset View")
        reset_view_btn.set_on_clicked(self._reset_view)
        self.panel.add_child(reset_view_btn)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(self.status)

    def _add_slider(self, key: str, label: str, value: float, lo: float, hi: float, callback) -> None:
        row = gui.Horiz(6)
        text = gui.Label(f"{label}: {value: .4f}")
        self.labels[key] = text
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(lo, hi)
        slider.double_value = float(value)

        def on_change(new_value):
            text.text = f"{label}: {new_value: .4f}"
            callback(float(new_value))

        slider.set_on_value_changed(on_change)
        setattr(self, f"{key}_slider", slider)
        row.add_child(text)
        row.add_child(slider)
        self.panel.add_child(row)

    def _on_layout(self, _layout_context) -> None:
        rect = self.window.content_rect
        panel_width = 390
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

    def _tcp_transform(self) -> np.ndarray:
        return translation(self.tcp_offset) @ yaw_matrix(self.tcp_yaw)

    def _apply_transforms(self) -> None:
        self.scene_widget.scene.set_geometry_transform("vessel", scale_matrix(self.vessel_scale))
        transform = self._tcp_transform()
        for index, item in enumerate(self.paths):
            name = f"tcp_{index:03d}_{item['branch']}_{item['path']}"
            self.scene_widget.scene.set_geometry_transform(name, transform)
        self._update_status()

    def _set_vessel_scale(self, value: float) -> None:
        self.vessel_scale = value
        self._apply_transforms()

    def _set_tcp_yaw(self, value: float) -> None:
        self.tcp_yaw = value
        self._apply_transforms()

    def _set_tcp_offset(self, axis: int, value: float) -> None:
        self.tcp_offset[axis] = value
        self._apply_transforms()

    def _center_tcp_to_vessel(self) -> None:
        rot = yaw_matrix(self.tcp_yaw)[:3, :3]
        rotated_center = rot @ self.raw_tcp_center
        target = self.vessel_center * self.vessel_scale
        self.tcp_offset = (target - rotated_center).astype(float).tolist()
        self._sync_sliders()
        self._apply_transforms()

    def _set_yaw_pi(self) -> None:
        self.tcp_yaw = math.pi
        self._sync_sliders()
        self._apply_transforms()

    def _sync_sliders(self) -> None:
        self.vessel_scale_slider.double_value = float(self.vessel_scale)
        self.tcp_yaw_slider.double_value = float(self.tcp_yaw)
        self.tcp_x_slider.double_value = float(self.tcp_offset[0])
        self.tcp_y_slider.double_value = float(self.tcp_offset[1])
        self.tcp_z_slider.double_value = float(self.tcp_offset[2])
        self.labels["vessel_scale"].text = f"scale: {self.vessel_scale: .4f}"
        self.labels["tcp_yaw"].text = f"yaw: {self.tcp_yaw: .4f}"
        self.labels["tcp_x"].text = f"x: {self.tcp_offset[0]: .4f}"
        self.labels["tcp_y"].text = f"y: {self.tcp_offset[1]: .4f}"
        self.labels["tcp_z"].text = f"z: {self.tcp_offset[2]: .4f}"

    def _save_alignment(self) -> None:
        data = {
            "vessel_scale": round(float(self.vessel_scale), 6),
            "tcp_yaw": round(float(self.tcp_yaw), 6),
            "tcp_offset": [round(float(x), 6) for x in self.tcp_offset],
            "tcp_stride": int(self.args.tcp_stride),
            "tcp_branch": self.args.tcp_branch,
            "root": self.args.root,
            "note": "Transform applied as T(tcp_offset) @ Rz(tcp_yaw) to real TCP points in meters.",
        }
        self.alignment_path.parent.mkdir(parents=True, exist_ok=True)
        self.alignment_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.text = f"Saved alignment:\n{self.alignment_path}"

    def _save_scene_config(self) -> None:
        self.config["vessel_scale"] = round(float(self.vessel_scale), 6)
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.text = f"Saved vessel scale to scene config:\n{self.config_path}"

    def _reset_view(self) -> None:
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(55.0, bounds, bounds.get_center())

    def _update_status(self) -> None:
        vessel_bounds = np.asarray(self.raw_vessel.bounds, dtype=float) * self.vessel_scale
        tcp_rot = yaw_matrix(self.tcp_yaw)[:3, :3]
        transformed_tcp = (self.raw_tcp_points @ tcp_rot.T) + np.asarray(self.tcp_offset, dtype=float)
        tcp_bounds = np.array([transformed_tcp.min(axis=0), transformed_tcp.max(axis=0)])
        self.status.text = (
            f"paths: {len(self.paths)}\n"
            f"branch: {self.args.tcp_branch or 'all'}\n"
            f"vessel scale: {self.vessel_scale:.5f}\n"
            f"tcp yaw: {self.tcp_yaw:.5f}\n"
            f"tcp offset: {[round(x, 4) for x in self.tcp_offset]}\n"
            f"vessel span m: {[round(x, 4) for x in (vessel_bounds[1] - vessel_bounds[0])]}\n"
            f"tcp span m: {[round(x, 4) for x in (tcp_bounds[1] - tcp_bounds[0])]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively align real TCP trajectories with vessel and dual robot scene.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--alignment", default=DEFAULT_ALIGNMENT)
    parser.add_argument("--root", default="branchs")
    parser.add_argument("--tcp-branch", default=None, choices=[None, "branch1", "branch2"])
    parser.add_argument("--tcp-stride", type=int, default=3)
    parser.add_argument("--tcp-yaw", type=float, default=math.pi)
    parser.add_argument("--vessel-scale", type=float, default=0.12)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--piper-joint", action="append", default=[])
    parser.add_argument("--elite-joint", action="append", default=[])
    args = parser.parse_args()

    RealTcpAlignmentApp(args)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
