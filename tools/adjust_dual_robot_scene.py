from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
from yourdfpy import URDF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_dual_robot_scene import (
    load_real_tcp_paths,
    load_vessel,
    make_magnet_tool,
    make_marker,
    pose_matrix,
    translation_matrix,
    trimesh_to_open3d,
)


DEFAULT_PIPER_POSE = [-0.55, -0.42, -0.08, 0.0, 0.0, 0.0]
DEFAULT_ELITE_POSE = [0.55, -0.42, -0.08, 0.0, 0.0, 3.14159265]
DEFAULT_PIPER_TOOL_POINT = [-0.20, -0.08, 0.02]
DEFAULT_PIPER_TOOL_LINK = "gripper_base"
DEFAULT_ELITE_TOOL_LINK = "flan"
DEFAULT_ELITE_MAGNET_OFFSET = [0.0, 0.0, 0.0]
DEFAULT_ALIGNMENT = "simulation_output/robot_scene_mvp/real_tcp_alignment.json"


def make_polyline(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points, dtype=np.float64, copy=True)),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32, copy=True)),
    )
    geom.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return geom


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.ones((len(points), 4), dtype=float)
    hom[:, :3] = points
    return (hom @ transform.T)[:, :3]


def load_alignment(path: str) -> dict | None:
    alignment_path = Path(path)
    if not alignment_path.exists():
        return None
    return json.loads(alignment_path.read_text(encoding="utf-8"))


def load_joint_specs(urdf_path: Path) -> list[dict]:
    robot = URDF.load(str(urdf_path))
    specs = []
    for joint in robot.robot.joints:
        if joint.type == "fixed":
            continue
        if joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
            lo = float(joint.limit.lower)
            hi = float(joint.limit.upper)
        elif joint.type == "prismatic":
            lo, hi = -0.05, 0.05
        else:
            lo, hi = -3.1416, 3.1416
        specs.append({"name": joint.name, "type": joint.type, "lo": lo, "hi": hi})
    return specs


def normalize_joint_values(specs: list[dict], values: dict | None) -> dict[str, float]:
    values = values or {}
    result = {}
    for spec in specs:
        value = float(values.get(spec["name"], 0.0))
        result[spec["name"]] = float(np.clip(value, spec["lo"], spec["hi"]))
    return result


class RobotVisualGroup:
    def __init__(self, scene, urdf_path: Path, prefix: str, material):
        self.scene = scene
        self.robot = URDF.load(str(urdf_path))
        self.items = []
        for index, node_name in enumerate(self.robot.scene.graph.nodes_geometry):
            _transform, geometry_name = self.robot.scene.graph.get(node_name)
            if geometry_name is None:
                continue
            geometry = self.robot.scene.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            name = f"{prefix}_{index:02d}"
            self.scene.add_geometry(name, trimesh_to_open3d(geometry), material)
            self.items.append((name, node_name))

    def update(self, joint_values: dict[str, float], base_transform: np.ndarray) -> None:
        self.robot.update_cfg({str(name): float(value) for name, value in joint_values.items()})
        for name, node_name in self.items:
            transform, _geometry_name = self.robot.scene.graph.get(node_name)
            self.scene.set_geometry_transform(name, base_transform @ transform)

    def link_transform(self, link_name: str) -> np.ndarray:
        return np.asarray(self.robot.get_transform(link_name), dtype=float)


class DualRobotSceneAdjuster:
    def __init__(self, args):
        self.args = args
        self.config_path = Path(args.config).resolve()
        self.assets_root = Path(args.assets_root).resolve()
        self.piper_urdf_path = self.assets_root / args.piper_urdf
        self.elite_urdf_path = self.assets_root / args.elite_urdf
        self.piper_pose = list(args.piper_pose)
        self.elite_pose = list(args.elite_pose)
        self.piper_tool_point = list(args.piper_tool_point)
        self.piper_tool_link = args.piper_tool_link
        self.piper_tool_offset = list(args.piper_tool_offset) if args.piper_tool_offset is not None else None
        self.piper_tool_transform = np.eye(4)
        self.elite_tool_link = args.elite_tool_link
        self.elite_magnet_offset = list(args.elite_magnet_offset)
        self.elite_tool_transform = np.eye(4)
        self.vessel_scale = float(args.vessel_scale)
        self.piper_names = []
        self.elite_names = []
        self.piper_visuals = None
        self.elite_visuals = None
        self.piper_joint_specs = load_joint_specs(self.piper_urdf_path)
        self.elite_joint_specs = load_joint_specs(self.elite_urdf_path)
        self.piper_joint_values = normalize_joint_values(self.piper_joint_specs, args.piper_joints)
        self.elite_joint_values = normalize_joint_values(self.elite_joint_specs, args.elite_joints)
        self.alignment = load_alignment(args.alignment)
        self.real_tcp_paths = []
        self.real_tcp_transform = np.eye(4)

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("Dual Robot Scene Pose Adjuster", 1460, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.94, 0.95, 0.96, 1.0])
        self.window.add_child(self.scene_widget)

        self.panel = gui.ScrollableVert(8, gui.Margins(12, 12, 12, 12))
        self.window.add_child(self.panel)
        self.status = gui.Label("")
        self.sliders = []

        self._build_scene()
        self._build_panel()
        self.window.set_on_layout(self._on_layout)
        self._apply_transforms()

    def _material(self, color):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = color
        return mat

    def _build_scene(self):
        self.piper_visuals = RobotVisualGroup(
            self.scene_widget.scene,
            self.piper_urdf_path,
            "piper",
            self._material([0.23, 0.43, 0.88, 1.0]),
        )
        self.elite_visuals = RobotVisualGroup(
            self.scene_widget.scene,
            self.elite_urdf_path,
            "elite",
            self._material([0.92, 0.42, 0.22, 1.0]),
        )
        self.piper_visuals.update(self.piper_joint_values, pose_matrix(self.piper_pose))
        self.elite_visuals.update(self.elite_joint_values, pose_matrix(self.elite_pose))

        self.piper_tool_transform = self.piper_visuals.link_transform(self.piper_tool_link)
        if self.piper_tool_offset is None:
            piper_link_world = pose_matrix(self.piper_pose) @ self.piper_tool_transform
            point = np.ones(4, dtype=float)
            point[:3] = np.asarray(self.piper_tool_point, dtype=float)
            self.piper_tool_offset = (np.linalg.inv(piper_link_world) @ point)[:3].astype(float).tolist()
        self.elite_tool_transform = self.elite_visuals.link_transform(self.elite_tool_link)

        vessel_mesh = load_vessel(Path(self.args.vessel).resolve(), self.vessel_scale, (120, 150, 190, 255))
        vessel_mat = self._material([0.45, 0.58, 0.78, 1.0])
        self.scene_widget.scene.add_geometry("vessel", trimesh_to_open3d(vessel_mesh), vessel_mat)

        self.scene_widget.scene.add_geometry(
            "world_axis",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.28),
            self._material([1, 1, 1, 1]),
        )
        self.scene_widget.scene.add_geometry(
            "piper_axis",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.22),
            self._material([1, 1, 1, 1]),
        )
        self.scene_widget.scene.add_geometry(
            "elite_axis",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.22),
            self._material([1, 1, 1, 1]),
        )
        self.scene_widget.scene.add_geometry(
            "piper_tool_point",
            trimesh_to_open3d(make_marker([0.0, 0.0, 0.0], 0.026, (30, 210, 80, 255))),
            self._material([0.05, 0.85, 0.25, 1.0]),
        )
        self.scene_widget.scene.add_geometry(
            "elite_magnet_tool",
            trimesh_to_open3d(make_magnet_tool(np.eye(4))),
            self._material([0.95, 0.12, 0.08, 1.0]),
        )
        self.scene_widget.scene.add_geometry(
            "elite_magnet_axis",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12),
            self._material([1, 1, 1, 1]),
        )

        if self.alignment:
            tcp_root = Path(self.alignment.get("root", "branchs")).resolve()
            self.real_tcp_paths = load_real_tcp_paths(tcp_root, self.alignment.get("tcp_branch"), int(self.alignment.get("tcp_stride", 3)))
            if self.real_tcp_paths:
                yaw = float(self.alignment.get("tcp_yaw", 0.0))
                offset = np.asarray(self.alignment.get("tcp_offset", [0.0, 0.0, 0.0]), dtype=float)
                c, s = np.cos(yaw), np.sin(yaw)
                self.real_tcp_transform = np.eye(4, dtype=float)
                self.real_tcp_transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                self.real_tcp_transform[:3, 3] = offset
                colors = {"branch1": (1.0, 0.08, 0.08), "branch2": (0.1, 0.28, 1.0)}
                line_mat = rendering.MaterialRecord()
                line_mat.shader = "unlitLine"
                line_mat.line_width = 4.0
                for index, item in enumerate(self.real_tcp_paths):
                    name = f"real_tcp_{index:03d}_{item['branch']}_{item['path']}"
                    points = transform_points(item["points_m"], self.real_tcp_transform)
                    self.scene_widget.scene.add_geometry(
                        name,
                        make_polyline(points, colors.get(item["branch"], (0.1, 0.8, 0.3))),
                        line_mat,
                    )

        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(55.0, bounds, bounds.get_center())

    def _build_panel(self):
        self.panel.add_child(gui.Label("Piper Pose"))
        self._add_pose_sliders("piper", self.piper_pose)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(gui.Label("Piper Joints"))
        self._add_joint_sliders("piper", self.piper_joint_specs, self.piper_joint_values)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(gui.Label("Elite Pose"))
        self._add_pose_sliders("elite", self.elite_pose)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(gui.Label("Elite Joints"))
        self._add_joint_sliders("elite", self.elite_joint_specs, self.elite_joint_values)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(gui.Label("Key Points"))
        self.panel.add_child(gui.Label(f"Piper outlet offset from {self.piper_tool_link}"))
        self._add_point_sliders("piper_tool_offset", self.piper_tool_offset, (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))
        self.panel.add_child(gui.Label(f"Elite magnet offset from {self.elite_tool_link}"))
        self._add_point_sliders("elite_magnet_offset", self.elite_magnet_offset, (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))
        self.panel.add_child(gui.Label(""))

        save_btn = gui.Button("Save Config")
        save_btn.set_on_clicked(self._save_config)
        self.panel.add_child(save_btn)

        reset_btn = gui.Button("Reset View")
        reset_btn.set_on_clicked(self._reset_view)
        self.panel.add_child(reset_btn)
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(self.status)
        self._update_status()

    def _add_pose_sliders(self, robot_name, values):
        specs = [
            ("x", -1.5, 1.5, 0.01),
            ("y", -1.5, 1.5, 0.01),
            ("z", -0.5, 1.0, 0.01),
            ("roll", -3.1416, 3.1416, 0.01),
            ("pitch", -3.1416, 3.1416, 0.01),
            ("yaw", -3.1416, 3.1416, 0.01),
        ]
        for idx, (label, lo, hi, _step) in enumerate(specs):
            row = gui.Horiz(6)
            text = gui.Label(f"{label}: {values[idx]: .3f}")
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(lo, hi)
            slider.double_value = float(values[idx])

            def on_change(value, robot=robot_name, axis=idx, text_label=text, axis_name=label):
                target = self.piper_pose if robot == "piper" else self.elite_pose
                target[axis] = float(value)
                text_label.text = f"{axis_name}: {value: .3f}"
                self._apply_transforms()

            slider.set_on_value_changed(on_change)
            row.add_child(text)
            row.add_child(slider)
            self.panel.add_child(row)
            self.sliders.append(slider)

    def _add_joint_sliders(self, robot_name, specs, values):
        for spec in specs:
            name = spec["name"]
            lo = float(spec["lo"])
            hi = float(spec["hi"])
            row = gui.Horiz(6)
            text = gui.Label(f"{name}: {values[name]: .3f}")
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(lo, hi)
            slider.double_value = float(values[name])

            def on_change(value, robot=robot_name, joint_name=name, text_label=text):
                target = self.piper_joint_values if robot == "piper" else self.elite_joint_values
                target[joint_name] = float(value)
                text_label.text = f"{joint_name}: {value: .3f}"
                self._apply_transforms()

            slider.set_on_value_changed(on_change)
            row.add_child(text)
            row.add_child(slider)
            self.panel.add_child(row)
            self.sliders.append(slider)

    def _add_point_sliders(self, point_name, values, x_limits, y_limits, z_limits):
        specs = [
            ("x", x_limits[0], x_limits[1]),
            ("y", y_limits[0], y_limits[1]),
            ("z", z_limits[0], z_limits[1]),
        ]
        for idx, (label, lo, hi) in enumerate(specs):
            row = gui.Horiz(6)
            text = gui.Label(f"{label}: {values[idx]: .3f}")
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(lo, hi)
            slider.double_value = float(values[idx])

            def on_change(value, point=point_name, axis=idx, text_label=text, axis_name=label):
                target = self.piper_tool_offset if point == "piper_tool_offset" else self.elite_magnet_offset
                target[axis] = float(value)
                text_label.text = f"{axis_name}: {value: .3f}"
                self._apply_transforms()

            slider.set_on_value_changed(on_change)
            row.add_child(text)
            row.add_child(slider)
            self.panel.add_child(row)
            self.sliders.append(slider)

    def _on_layout(self, _layout_context):
        rect = self.window.content_rect
        panel_width = 380
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

    def _apply_transforms(self):
        piper_tf = pose_matrix(self.piper_pose)
        elite_tf = pose_matrix(self.elite_pose)
        self.piper_visuals.update(self.piper_joint_values, piper_tf)
        self.elite_visuals.update(self.elite_joint_values, elite_tf)
        self.piper_tool_transform = self.piper_visuals.link_transform(self.piper_tool_link)
        piper_tool_tf = piper_tf @ self.piper_tool_transform @ translation_matrix(self.piper_tool_offset)
        self.piper_tool_point = piper_tool_tf[:3, 3].astype(float).tolist()
        self.elite_tool_transform = self.elite_visuals.link_transform(self.elite_tool_link)
        elite_magnet_tf = elite_tf @ self.elite_tool_transform @ translation_matrix(self.elite_magnet_offset)
        self.scene_widget.scene.set_geometry_transform("piper_axis", piper_tf)
        self.scene_widget.scene.set_geometry_transform("elite_axis", elite_tf)
        self.scene_widget.scene.set_geometry_transform("piper_tool_point", piper_tool_tf)
        self.scene_widget.scene.set_geometry_transform("elite_magnet_tool", elite_magnet_tf)
        self.scene_widget.scene.set_geometry_transform("elite_magnet_axis", elite_magnet_tf)
        self._update_status()

    def _reset_view(self):
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(55.0, bounds, bounds.get_center())

    def _save_config(self):
        data = {
            "assets_root": self.args.assets_root,
            "piper_urdf": self.args.piper_urdf,
            "elite_urdf": self.args.elite_urdf,
            "vessel": self.args.vessel,
            "vessel_scale": self.vessel_scale,
            "piper_pose": [round(float(x), 6) for x in self.piper_pose],
            "elite_pose": [round(float(x), 6) for x in self.elite_pose],
            "piper_tool_point": [round(float(x), 6) for x in self.piper_tool_point],
            "piper_tool_link": self.piper_tool_link,
            "piper_tool_offset": [round(float(x), 6) for x in self.piper_tool_offset],
            "piper_joints": {name: round(float(value), 6) for name, value in self.piper_joint_values.items()},
            "elite_tool_link": self.elite_tool_link,
            "elite_magnet_offset": [round(float(x), 6) for x in self.elite_magnet_offset],
            "elite_joints": {name: round(float(value), 6) for name, value in self.elite_joint_values.items()},
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.text = f"Saved: {self.config_path}"

    def _update_status(self):
        self.piper_tool_transform = self.piper_visuals.link_transform(self.piper_tool_link)
        piper_tool_tf = pose_matrix(self.piper_pose) @ self.piper_tool_transform @ translation_matrix(self.piper_tool_offset)
        piper_tool_world = piper_tool_tf[:3, 3].tolist()
        self.elite_tool_transform = self.elite_visuals.link_transform(self.elite_tool_link)
        elite_magnet_tf = pose_matrix(self.elite_pose) @ self.elite_tool_transform @ translation_matrix(self.elite_magnet_offset)
        elite_magnet_world = elite_magnet_tf[:3, 3].tolist()
        self.status.text = (
            f"Piper: {[round(x, 3) for x in self.piper_pose]}\n"
            f"Elite: {[round(x, 3) for x in self.elite_pose]}\n"
            f"Piper outlet link: {self.piper_tool_link}\n"
            f"Piper outlet offset: {[round(x, 3) for x in self.piper_tool_offset]}\n"
            f"Piper outlet world: {[round(x, 3) for x in piper_tool_world]}\n"
            f"Piper joints: {[round(self.piper_joint_values[s['name']], 3) for s in self.piper_joint_specs]}\n"
            f"Elite magnet link: {self.elite_tool_link}\n"
            f"Elite magnet offset: {[round(x, 3) for x in self.elite_magnet_offset]}\n"
            f"Elite magnet world: {[round(x, 3) for x in elite_magnet_world]}\n"
            f"Elite joints: {[round(self.elite_joint_values[s['name']], 3) for s in self.elite_joint_specs]}"
        )


def load_config_if_present(args):
    if not args.config or not Path(args.config).exists():
        return
    data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.assets_root = data.get("assets_root", args.assets_root)
    args.piper_urdf = data.get("piper_urdf", args.piper_urdf)
    args.elite_urdf = data.get("elite_urdf", args.elite_urdf)
    args.vessel = data.get("vessel", args.vessel)
    args.vessel_scale = float(data.get("vessel_scale", args.vessel_scale))
    args.piper_pose = data.get("piper_pose", args.piper_pose)
    args.elite_pose = data.get("elite_pose", args.elite_pose)
    args.piper_tool_point = data.get("piper_tool_point", args.piper_tool_point)
    args.piper_tool_link = data.get("piper_tool_link", args.piper_tool_link)
    args.piper_tool_offset = data.get("piper_tool_offset", args.piper_tool_offset)
    args.piper_joints = data.get("piper_joints", args.piper_joints)
    args.elite_tool_link = data.get("elite_tool_link", args.elite_tool_link)
    args.elite_magnet_offset = data.get("elite_magnet_offset", args.elite_magnet_offset)
    args.elite_joints = data.get("elite_joints", args.elite_joints)


def main():
    parser = argparse.ArgumentParser(description="Interactively adjust Piper and Elite base poses in one Open3D scene.")
    parser.add_argument("--assets-root", default="robot_assets/standardized")
    parser.add_argument("--piper-urdf", default="piper_description/urdf/piper_description.urdf")
    parser.add_argument("--elite-urdf", default="elite_description/urdf/ec66_description.urdf")
    parser.add_argument("--vessel", default="utils/interface/model/0422.stl")
    parser.add_argument("--vessel-scale", type=float, default=0.12)
    parser.add_argument("--piper-pose", nargs=6, type=float, default=DEFAULT_PIPER_POSE)
    parser.add_argument("--elite-pose", nargs=6, type=float, default=DEFAULT_ELITE_POSE)
    parser.add_argument("--piper-tool-point", nargs=3, type=float, default=DEFAULT_PIPER_TOOL_POINT)
    parser.add_argument("--piper-tool-link", default=DEFAULT_PIPER_TOOL_LINK)
    parser.add_argument("--piper-tool-offset", nargs=3, type=float, default=None)
    parser.add_argument("--piper-joints", default={})
    parser.add_argument("--elite-tool-link", default=DEFAULT_ELITE_TOOL_LINK)
    parser.add_argument("--elite-magnet-offset", nargs=3, type=float, default=DEFAULT_ELITE_MAGNET_OFFSET)
    parser.add_argument("--elite-joints", default={})
    parser.add_argument("--alignment", default=DEFAULT_ALIGNMENT)
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    args = parser.parse_args()
    load_config_if_present(args)

    DualRobotSceneAdjuster(args)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
