from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_branch_dataset import numeric_suffix, read_matrix
from tools.visualize_dual_robot_scene import (
    get_urdf_link_transform_with_joints,
    load_urdf_meshes,
    load_vessel,
    make_magnet_tool,
    make_marker,
    parse_joint_config,
    pose_matrix,
    translation_matrix,
    trimesh_to_open3d,
)


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_tcp_paths(root: Path, branch: str | None, stride: int) -> list[dict]:
    branch_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("branch")], key=numeric_suffix)
    if branch:
        branch_dirs = [p for p in branch_dirs if p.name == branch]
    paths = []
    for branch_dir in branch_dirs:
        for pose_path in sorted((branch_dir / "path").glob("pose*.txt"), key=numeric_suffix):
            pose = read_matrix(pose_path, expected_min_cols=6)
            if len(pose) == 0:
                continue
            xyz_m = pose[:: max(stride, 1), :3] * 0.001
            paths.append({"branch": branch_dir.name, "path": pose_path.stem, "points_m": xyz_m})
    return paths


def fit_real_to_vessel(paths: list[dict], vessel_mesh: trimesh.Trimesh, mode: str) -> np.ndarray:
    all_points = np.vstack([item["points_m"] for item in paths])
    real_min = np.min(all_points, axis=0)
    real_max = np.max(all_points, axis=0)
    real_center = (real_min + real_max) * 0.5
    vessel_center = vessel_mesh.bounds.mean(axis=0)
    tf = np.eye(4, dtype=float)
    if mode == "raw":
        return tf
    if mode == "center":
        tf[:3, 3] = vessel_center - real_center
        return tf
    if mode == "center_xy":
        tf[:2, 3] = vessel_center[:2] - real_center[:2]
        return tf
    raise ValueError(f"Unknown fit mode: {mode}")


def rotate_real_tcp_paths(paths: list[dict], yaw: float) -> list[dict]:
    if abs(yaw) < 1e-12:
        return paths
    center = np.vstack([item["points_m"] for item in paths]).mean(axis=0)
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotated = []
    for item in paths:
        points = (item["points_m"] - center) @ rot.T + center
        rotated.append({**item, "points_m": points})
    return rotated


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.ones((len(points), 4), dtype=float)
    hom[:, :3] = points
    return (hom @ transform.T)[:, :3]


def make_polyline(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    geom.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return geom


def cylinder_between_points(p0: np.ndarray, p1: np.ndarray, radius: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh | None:
    vec = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=10)
    direction = vec / length
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    transform[:3, 3] = (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float)) * 0.5
    cyl.apply_transform(transform)
    cyl.visual.face_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(cyl.faces), 1))
    return cyl


def polyline_to_tubes(points: np.ndarray, radius: float, color: tuple[int, int, int, int], max_segments: int = 120) -> trimesh.Trimesh | None:
    if len(points) < 2:
        return None
    step = max(int(np.ceil((len(points) - 1) / max_segments)), 1)
    segments = []
    for index in range(0, len(points) - 1, step):
        end_index = min(index + step, len(points) - 1)
        segment = cylinder_between_points(points[index], points[end_index], radius, color)
        if segment is not None:
            segments.append(segment)
    if not segments:
        return None
    return trimesh.util.concatenate(segments)


def make_scene_geometries(config: dict, piper_joint_overrides: dict, elite_joint_overrides: dict) -> tuple[list, trimesh.Trimesh]:
    assets_root = (REPO_ROOT / config.get("assets_root", "robot_assets/standardized")).resolve()
    piper_urdf = assets_root / config["piper_urdf"]
    elite_urdf = assets_root / config["elite_urdf"]
    piper_pose = pose_matrix(config["piper_pose"])
    elite_pose = pose_matrix(config["elite_pose"])
    piper_joints = {**config.get("piper_joints", {}), **piper_joint_overrides}
    elite_joints = {**config.get("elite_joints", {}), **elite_joint_overrides}

    vessel = load_vessel(REPO_ROOT / config["vessel"], float(config["vessel_scale"]), (120, 150, 190, 92))
    piper_meshes = load_urdf_meshes(piper_urdf, piper_pose, (80, 135, 230, 255), piper_joints)
    elite_meshes = load_urdf_meshes(elite_urdf, elite_pose, (235, 130, 70, 255), elite_joints)

    piper_tool_transform = get_urdf_link_transform_with_joints(
        piper_urdf,
        config.get("piper_tool_link", "gripper_base"),
        piper_joints,
    )
    piper_tool_center = (
        piper_pose @ piper_tool_transform @ translation_matrix(config.get("piper_tool_offset", [0.0, 0.0, 0.0]))
    )[:3, 3].tolist()
    elite_tool_transform = get_urdf_link_transform_with_joints(
        elite_urdf,
        config.get("elite_tool_link", "flan"),
        elite_joints,
    )
    elite_magnet_transform = (
        elite_pose @ elite_tool_transform @ translation_matrix(config.get("elite_magnet_offset", [0.0, 0.0, 0.0]))
    )

    geometries = [trimesh_to_open3d(vessel)]
    geometries.extend(trimesh_to_open3d(mesh) for mesh in piper_meshes)
    geometries.extend(trimesh_to_open3d(mesh) for mesh in elite_meshes)
    geometries.append(trimesh_to_open3d(make_marker(piper_tool_center, 0.025, (30, 210, 80, 255))))
    geometries.append(trimesh_to_open3d(make_magnet_tool(elite_magnet_transform)))
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.28))
    return geometries, vessel


def export_overlay_glb(config: dict, paths: list[dict], transform: np.ndarray, out_path: Path) -> None:
    scene = trimesh.Scene()
    vessel = load_vessel(REPO_ROOT / config["vessel"], float(config["vessel_scale"]), (120, 150, 190, 92))
    scene.add_geometry(vessel, node_name="vessel")
    colors = {"branch1": (255, 50, 50, 255), "branch2": (60, 100, 255, 255)}
    for item in paths:
        points = transform_points(item["points_m"], transform)
        color = colors.get(item["branch"], (30, 220, 80, 255))
        tube = polyline_to_tubes(points, radius=0.0025, color=color)
        if tube is not None:
            scene.add_geometry(tube, node_name=f"{item['branch']}_{item['path']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay real Elite TCP trajectories from branchs onto the current scene.")
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--root", default="branchs")
    parser.add_argument("--branch", default=None, choices=[None, "branch1", "branch2"])
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fit", choices=["raw", "center", "center_xy"], default="center")
    parser.add_argument("--yaw", type=float, default=3.14159265)
    parser.add_argument("--out", default="simulation_output/robot_scene_mvp/real_tcp_overlay.glb")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--piper-joint", action="append", default=[])
    parser.add_argument("--elite-joint", action="append", default=[])
    args = parser.parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    paths = load_tcp_paths((REPO_ROOT / args.root).resolve(), args.branch, args.stride)
    if not paths:
        raise RuntimeError(f"No TCP paths found under {args.root}")

    piper_joints = parse_joint_config(args.piper_joint)
    elite_joints = parse_joint_config(args.elite_joint)
    geometries, vessel = make_scene_geometries(config, piper_joints, elite_joints)
    paths = rotate_real_tcp_paths(paths, args.yaw)
    transform = fit_real_to_vessel(paths, vessel, args.fit)

    branch_colors = {"branch1": (1.0, 0.08, 0.08), "branch2": (0.1, 0.28, 1.0)}
    all_transformed = []
    for item in paths:
        points = transform_points(item["points_m"], transform)
        all_transformed.append(points)
        if len(points) > 1:
            geometries.append(make_polyline(points, branch_colors.get(item["branch"], (0.1, 0.8, 0.3))))
    all_transformed = np.vstack(all_transformed)

    export_overlay_glb(config, paths, transform, (REPO_ROOT / args.out).resolve())

    real_raw = np.vstack([item["points_m"] for item in paths])
    report = {
        "paths": len(paths),
        "fit": args.fit,
        "transform": transform.tolist(),
        "raw_bounds_m": [real_raw.min(axis=0).tolist(), real_raw.max(axis=0).tolist()],
        "overlay_bounds_m": [all_transformed.min(axis=0).tolist(), all_transformed.max(axis=0).tolist()],
        "vessel_bounds_m": vessel.bounds.tolist(),
        "out": str((REPO_ROOT / args.out).resolve()),
    }
    report_path = (REPO_ROOT / "simulation_output/robot_scene_mvp/real_tcp_overlay_report.json").resolve()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.show:
        o3d.visualization.draw_geometries(geometries, window_name="Real TCP Overlay")


if __name__ == "__main__":
    main()
