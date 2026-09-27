from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from yourdfpy import URDF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_branch_dataset import numeric_suffix, read_matrix


def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    rot = rz @ ry @ rx
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    return mat


def pose_matrix(values: list[float]) -> np.ndarray:
    if len(values) != 6:
        raise ValueError("Pose must contain x y z roll pitch yaw")
    mat = euler_matrix(values[3], values[4], values[5])
    mat[:3, 3] = values[:3]
    return mat


def translation_matrix(values: list[float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError("Translation must contain x y z")
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = values
    return mat


def colorize(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.visual.face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def load_urdf_meshes(
    urdf_path: Path,
    base_pose: np.ndarray,
    color: tuple[int, int, int, int],
    joint_config: dict[str, float] | None = None,
) -> list[trimesh.Trimesh]:
    robot = URDF.load(str(urdf_path))
    if joint_config:
        robot.update_cfg({str(name): float(value) for name, value in joint_config.items()})
    meshes = []
    for node_name in robot.scene.graph.nodes_geometry:
        transform, geometry_name = robot.scene.graph.get(node_name)
        if geometry_name is None:
            continue
        geometry = robot.scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = colorize(geometry, color)
        mesh.apply_transform(base_pose @ transform)
        meshes.append(mesh)
    return meshes


def load_vessel(vessel_path: Path, vessel_scale: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vessel = trimesh.load(vessel_path, force="mesh")
    if not isinstance(vessel, trimesh.Trimesh):
        raise TypeError(f"Expected vessel mesh, got {type(vessel)!r}")
    vessel = colorize(vessel, color)
    center = vessel.bounds.mean(axis=0)
    vessel.apply_translation(-center)
    vessel.apply_scale(vessel_scale)
    return vessel


def make_marker(point: list[float], radius: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    marker = trimesh.creation.uv_sphere(radius=radius, count=[24, 24])
    marker.apply_translation(np.asarray(point, dtype=float))
    return colorize(marker, color)


def make_magnet_tool(transform: np.ndarray, radius: float = 0.025, length: float = 0.09) -> trimesh.Trimesh:
    coil = trimesh.creation.cylinder(radius=radius, height=length, sections=36)
    tip = trimesh.creation.uv_sphere(radius=radius * 1.05, count=[24, 24])
    tip.apply_translation([0.0, 0.0, length * 0.5])
    tool = trimesh.util.concatenate([coil, tip])
    tool.apply_transform(transform)
    return colorize(tool, (240, 60, 40, 255))


def get_urdf_link_transform(urdf_path: Path, link_name: str) -> np.ndarray:
    robot = URDF.load(str(urdf_path))
    try:
        return np.array(robot.get_transform(link_name), dtype=float, copy=True)
    except KeyError as exc:
        raise KeyError(f"Link {link_name!r} was not found in {urdf_path}") from exc


def get_urdf_link_transform_with_joints(
    urdf_path: Path,
    link_name: str,
    joint_config: dict[str, float] | None = None,
) -> np.ndarray:
    robot = URDF.load(str(urdf_path))
    if joint_config:
        robot.update_cfg({str(name): float(value) for name, value in joint_config.items()})
    try:
        return np.array(robot.get_transform(link_name), dtype=float, copy=True)
    except KeyError as exc:
        raise KeyError(f"Link {link_name!r} was not found in {urdf_path}") from exc


def parse_joint_config(values: list[str] | None) -> dict[str, float]:
    joints: dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Joint config item must be name=value, got {item!r}")
        name, value = item.split("=", 1)
        joints[name.strip()] = float(value)
    return joints


def load_real_tcp_paths(root: Path, branch: str | None, stride: int) -> list[dict]:
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
    rot = euler_matrix(0.0, 0.0, yaw)[:3, :3]
    rotated = []
    for item in paths:
        points = (item["points_m"] - center) @ rot.T + center
        rotated.append({**item, "points_m": points})
    return rotated


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.ones((len(points), 4), dtype=float)
    hom[:, :3] = points
    return (hom @ transform.T)[:, :3]


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


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    geom = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64, copy=True)),
        triangles=o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32, copy=True)),
    )
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) == len(mesh.faces):
        face_color = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float64) / 255.0
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        counts = np.zeros((len(mesh.vertices), 1), dtype=np.float64)
        faces = np.asarray(mesh.faces)
        for tri, color in zip(faces, face_color):
            vertex_colors[tri] += color
            counts[tri] += 1.0
        vertex_colors /= np.maximum(counts, 1.0)
        geom.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors, dtype=np.float64, copy=True))
    geom.compute_vertex_normals()
    return geom


def add_axis(scene: trimesh.Scene, transform: np.ndarray, name: str) -> None:
    try:
        axis = trimesh.creation.axis(origin_size=0.025, axis_length=0.28, transform=transform)
        scene.add_geometry(axis, node_name=name)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal dual-arm robot scene for asset and frame checking.")
    parser.add_argument("--assets-root", default="robot_assets/standardized")
    parser.add_argument("--piper-urdf", default="piper_description/urdf/piper_description.urdf")
    parser.add_argument("--elite-urdf", default="elite_description/urdf/ec66_description.urdf")
    parser.add_argument("--vessel", default="utils/interface/model/0422.stl")
    parser.add_argument("--out", default="simulation_output/robot_scene_mvp/dual_arm_scene.glb")
    parser.add_argument("--config", default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--vessel-scale", type=float, default=0.12)
    parser.add_argument("--piper-pose", nargs=6, type=float, default=[-0.55, -0.42, -0.08, 0.0, 0.0, 0.0])
    parser.add_argument("--elite-pose", nargs=6, type=float, default=[0.55, -0.42, -0.08, 0.0, 0.0, 3.14159265])
    parser.add_argument("--piper-tool-point", nargs=3, type=float, default=[-0.20, -0.08, 0.02])
    parser.add_argument("--piper-tool-link", default="gripper_base")
    parser.add_argument("--piper-tool-offset", nargs=3, type=float, default=None)
    parser.add_argument("--elite-tool-link", default="flan")
    parser.add_argument("--elite-magnet-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--elite-magnet-point", nargs=3, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--piper-joint", action="append", default=[])
    parser.add_argument("--elite-joint", action="append", default=[])
    parser.add_argument("--real-tcp-root", default="")
    parser.add_argument("--real-tcp-branch", default=None, choices=[None, "branch1", "branch2"])
    parser.add_argument("--real-tcp-fit", choices=["raw", "center", "center_xy"], default="center")
    parser.add_argument("--real-tcp-yaw", type=float, default=3.14159265)
    parser.add_argument("--real-tcp-stride", type=int, default=3)
    parser.add_argument("--real-tcp-radius", type=float, default=0.0025)
    args = parser.parse_args()

    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        args.piper_pose = config.get("piper_pose", args.piper_pose)
        args.elite_pose = config.get("elite_pose", args.elite_pose)
        args.piper_tool_point = config.get("piper_tool_point", args.piper_tool_point)
        args.piper_tool_link = config.get("piper_tool_link", args.piper_tool_link)
        args.piper_tool_offset = config.get("piper_tool_offset", args.piper_tool_offset)
        args.elite_tool_link = config.get("elite_tool_link", args.elite_tool_link)
        args.elite_magnet_offset = config.get("elite_magnet_offset", args.elite_magnet_offset)
        config_piper_joints = config.get("piper_joints", {})
        config_elite_joints = config.get("elite_joints", {})
        args.vessel_scale = float(config.get("vessel_scale", args.vessel_scale))
        args.piper_urdf = config.get("piper_urdf", args.piper_urdf)
        args.elite_urdf = config.get("elite_urdf", args.elite_urdf)
    else:
        config_piper_joints = {}
        config_elite_joints = {}

    assets_root = Path(args.assets_root).resolve()
    piper_urdf = assets_root / args.piper_urdf
    elite_urdf = assets_root / args.elite_urdf
    vessel_path = Path(args.vessel).resolve()

    piper_pose = pose_matrix(args.piper_pose)
    elite_pose = pose_matrix(args.elite_pose)
    piper_joint_config = {**config_piper_joints, **parse_joint_config(args.piper_joint)}
    elite_joint_config = {**config_elite_joints, **parse_joint_config(args.elite_joint)}
    piper_tool_transform = get_urdf_link_transform_with_joints(piper_urdf, args.piper_tool_link, piper_joint_config)
    if args.piper_tool_offset is None:
        piper_tool_center = args.piper_tool_point
    else:
        piper_tool_center = (piper_pose @ piper_tool_transform @ translation_matrix(args.piper_tool_offset))[:3, 3].tolist()
    elite_tool_transform = get_urdf_link_transform_with_joints(elite_urdf, args.elite_tool_link, elite_joint_config)
    elite_magnet_transform = elite_pose @ elite_tool_transform @ translation_matrix(args.elite_magnet_offset)
    elite_magnet_center = elite_magnet_transform[:3, 3].tolist()

    scene = trimesh.Scene()
    vessel = load_vessel(vessel_path, args.vessel_scale, (120, 150, 190, 92))
    scene.add_geometry(vessel, node_name="vessel")
    add_axis(scene, np.eye(4), "world_axis")

    piper_meshes = load_urdf_meshes(piper_urdf, piper_pose, (80, 135, 230, 255), piper_joint_config)
    elite_meshes = load_urdf_meshes(elite_urdf, elite_pose, (235, 130, 70, 255), elite_joint_config)
    for index, mesh in enumerate(piper_meshes):
        scene.add_geometry(mesh, node_name=f"piper_{index:02d}")
    for index, mesh in enumerate(elite_meshes):
        scene.add_geometry(mesh, node_name=f"elite_{index:02d}")
    piper_marker = make_marker(piper_tool_center, 0.025, (30, 210, 80, 255))
    elite_marker = make_magnet_tool(elite_magnet_transform)
    scene.add_geometry(piper_marker, node_name="piper_tool_point")
    scene.add_geometry(elite_marker, node_name="elite_magnet_tool")
    add_axis(scene, piper_pose, "piper_base_axis")
    add_axis(scene, elite_pose, "elite_base_axis")
    add_axis(scene, elite_magnet_transform, "elite_magnet_axis")

    if args.real_tcp_root:
        tcp_root = Path(args.real_tcp_root).resolve()
        real_paths = load_real_tcp_paths(tcp_root, args.real_tcp_branch, args.real_tcp_stride)
        if real_paths:
            real_paths = rotate_real_tcp_paths(real_paths, args.real_tcp_yaw)
            tcp_transform = fit_real_to_vessel(real_paths, vessel, args.real_tcp_fit)
            branch_colors = {"branch1": (255, 40, 40, 255), "branch2": (60, 110, 255, 255)}
            for item in real_paths:
                points = transform_points(item["points_m"], tcp_transform)
                tube = polyline_to_tubes(points, radius=args.real_tcp_radius, color=branch_colors.get(item["branch"], (30, 220, 80, 255)))
                if tube is not None:
                    scene.add_geometry(tube, node_name=f"real_{item['branch']}_{item['path']}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)

    print(f"saved {out_path}")
    print(f"piper meshes: {len(piper_meshes)} from {piper_urdf}")
    print(f"elite meshes: {len(elite_meshes)} from {elite_urdf}")
    print(f"vessel scale: {args.vessel_scale}")
    print(f"piper tool: link={args.piper_tool_link} offset={args.piper_tool_offset} world={piper_tool_center}")
    print(f"elite magnet: link={args.elite_tool_link} offset={args.elite_magnet_offset} world={elite_magnet_center}")
    if args.real_tcp_root:
        print(f"real tcp overlay: root={args.real_tcp_root} branch={args.real_tcp_branch} fit={args.real_tcp_fit} yaw={args.real_tcp_yaw} stride={args.real_tcp_stride}")

    if args.show:
        geometries = [trimesh_to_open3d(vessel)]
        geometries.extend(trimesh_to_open3d(mesh) for mesh in piper_meshes)
        geometries.extend(trimesh_to_open3d(mesh) for mesh in elite_meshes)
        geometries.append(trimesh_to_open3d(piper_marker))
        geometries.append(trimesh_to_open3d(elite_marker))
        if args.real_tcp_root:
            tcp_root = Path(args.real_tcp_root).resolve()
            real_paths = load_real_tcp_paths(tcp_root, args.real_tcp_branch, args.real_tcp_stride)
            if real_paths:
                real_paths = rotate_real_tcp_paths(real_paths, args.real_tcp_yaw)
                tcp_transform = fit_real_to_vessel(real_paths, vessel, args.real_tcp_fit)
                branch_colors = {"branch1": (255, 40, 40, 255), "branch2": (60, 110, 255, 255)}
                for item in real_paths:
                    points = transform_points(item["points_m"], tcp_transform)
                    tube = polyline_to_tubes(points, radius=args.real_tcp_radius, color=branch_colors.get(item["branch"], (30, 220, 80, 255)))
                    if tube is not None:
                        geometries.append(trimesh_to_open3d(tube))
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.28))
        piper_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.22)
        piper_axis.transform(piper_pose)
        geometries.append(piper_axis)
        elite_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.22)
        elite_axis.transform(elite_pose)
        geometries.append(elite_axis)
        magnet_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        magnet_axis.transform(elite_magnet_transform)
        geometries.append(magnet_axis)
        o3d.visualization.draw_geometries(geometries, window_name="Dual Robot Scene MVP")


if __name__ == "__main__":
    main()
