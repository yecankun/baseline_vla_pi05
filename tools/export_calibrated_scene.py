from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_dual_robot_scene import (
    add_axis,
    get_urdf_link_transform_with_joints,
    load_real_tcp_paths,
    load_urdf_meshes,
    load_vessel,
    make_magnet_tool,
    make_marker,
    parse_joint_config,
    polyline_to_tubes,
    pose_matrix,
    translation_matrix,
    transform_points,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def yaw_transform(yaw: float, offset: list[float]) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    transform[:3, 3] = np.asarray(offset, dtype=float)
    return transform


def bounds_or_none(points: np.ndarray | None) -> list[list[float]] | None:
    if points is None or len(points) == 0:
        return None
    return [
        [float(x) for x in np.min(points, axis=0)],
        [float(x) for x in np.max(points, axis=0)],
    ]


def export_scene(args: argparse.Namespace) -> dict:
    config_path = (REPO_ROOT / args.config).resolve()
    alignment_path = (REPO_ROOT / args.alignment).resolve()
    config = load_json(config_path)
    alignment = load_json(alignment_path) if alignment_path.exists() else {}

    assets_root = (REPO_ROOT / config.get("assets_root", "robot_assets/standardized")).resolve()
    piper_urdf = assets_root / config.get("piper_urdf", "piper_description/urdf/piper_description.urdf")
    elite_urdf = assets_root / config.get("elite_urdf", "elite_description/urdf/ec66_description.urdf")

    piper_joints = {**config.get("piper_joints", {}), **parse_joint_config(args.piper_joint)}
    elite_joints = {**config.get("elite_joints", {}), **parse_joint_config(args.elite_joint)}
    piper_pose = pose_matrix(config["piper_pose"])
    elite_pose = pose_matrix(config["elite_pose"])

    vessel = load_vessel((REPO_ROOT / config["vessel"]).resolve(), float(config["vessel_scale"]), (120, 150, 190, 112))
    piper_meshes = load_urdf_meshes(piper_urdf, piper_pose, (80, 135, 230, 255), piper_joints)
    elite_meshes = load_urdf_meshes(elite_urdf, elite_pose, (235, 130, 70, 255), elite_joints)

    piper_tool_transform = get_urdf_link_transform_with_joints(
        piper_urdf,
        config.get("piper_tool_link", "gripper_base"),
        piper_joints,
    )
    piper_tool_world = (
        piper_pose @ piper_tool_transform @ translation_matrix(config.get("piper_tool_offset", [0.0, 0.0, 0.0]))
    )[:3, 3]

    elite_tool_transform = get_urdf_link_transform_with_joints(
        elite_urdf,
        config.get("elite_tool_link", "flan"),
        elite_joints,
    )
    elite_magnet_transform = (
        elite_pose @ elite_tool_transform @ translation_matrix(config.get("elite_magnet_offset", [0.0, 0.0, 0.0]))
    )
    elite_magnet_world = elite_magnet_transform[:3, 3]

    scene = trimesh.Scene()
    scene.add_geometry(vessel, node_name="vessel")
    for index, mesh in enumerate(piper_meshes):
        scene.add_geometry(mesh, node_name=f"piper_{index:02d}")
    for index, mesh in enumerate(elite_meshes):
        scene.add_geometry(mesh, node_name=f"elite_{index:02d}")
    scene.add_geometry(make_marker(piper_tool_world.tolist(), 0.025, (30, 210, 80, 255)), node_name="piper_tool_point")
    scene.add_geometry(make_magnet_tool(elite_magnet_transform), node_name="elite_magnet_tool")
    add_axis(scene, np.eye(4), "world_axis")
    add_axis(scene, piper_pose, "piper_base_axis")
    add_axis(scene, elite_pose, "elite_base_axis")
    add_axis(scene, elite_magnet_transform, "elite_magnet_axis")

    tcp_points = None
    tcp_paths_count = 0
    if alignment:
        tcp_root = (REPO_ROOT / alignment.get("root", "branchs")).resolve()
        tcp_paths = load_real_tcp_paths(tcp_root, alignment.get("tcp_branch"), int(alignment.get("tcp_stride", 3)))
        tcp_transform = yaw_transform(float(alignment.get("tcp_yaw", 0.0)), alignment.get("tcp_offset", [0.0, 0.0, 0.0]))
        branch_colors = {"branch1": (255, 40, 40, 255), "branch2": (60, 110, 255, 255)}
        transformed = []
        for item in tcp_paths:
            points = transform_points(item["points_m"], tcp_transform)
            transformed.append(points)
            tube = polyline_to_tubes(
                points,
                radius=float(args.tcp_radius),
                color=branch_colors.get(item["branch"], (30, 220, 80, 255)),
            )
            if tube is not None:
                scene.add_geometry(tube, node_name=f"real_tcp_{item['branch']}_{item['path']}")
        tcp_paths_count = len(tcp_paths)
        if transformed:
            tcp_points = np.vstack(transformed)

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)

    report = {
        "config": str(config_path),
        "alignment": str(alignment_path) if alignment else None,
        "out": str(out_path),
        "vessel_scale": float(config["vessel_scale"]),
        "vessel_bounds_m": vessel.bounds.tolist(),
        "piper_tool_world_m": [float(x) for x in piper_tool_world],
        "elite_magnet_world_m": [float(x) for x in elite_magnet_world],
        "tool_distance_m": float(np.linalg.norm(piper_tool_world - elite_magnet_world)),
        "real_tcp_paths": tcp_paths_count,
        "real_tcp_bounds_m": bounds_or_none(tcp_points),
    }
    if tcp_points is not None:
        report["piper_to_real_tcp_start_m"] = float(np.linalg.norm(piper_tool_world - tcp_points[0]))
        report["elite_to_real_tcp_start_m"] = float(np.linalg.norm(elite_magnet_world - tcp_points[0]))

    report_path = (REPO_ROOT / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the calibrated vessel, dual-arm scene, and real TCP overlay.")
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--alignment", default="simulation_output/robot_scene_mvp/real_tcp_alignment.json")
    parser.add_argument("--out", default="simulation_output/robot_scene_mvp/calibrated_dual_robot_scene.glb")
    parser.add_argument("--report", default="simulation_output/robot_scene_mvp/calibrated_scene_report.json")
    parser.add_argument("--tcp-radius", type=float, default=0.0025)
    parser.add_argument("--piper-joint", action="append", default=[])
    parser.add_argument("--elite-joint", action="append", default=[])
    args = parser.parse_args()

    report = export_scene(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
