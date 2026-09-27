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

from tools.visualize_dual_robot_scene import load_urdf_meshes, parse_joint_config, pose_matrix


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_robot_meshes(args: argparse.Namespace) -> dict:
    config_path = (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)
    assets_root = (REPO_ROOT / config.get("assets_root", "robot_assets/standardized")).resolve()
    out_root = (REPO_ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    piper_joints = {**config.get("piper_joints", {}), **parse_joint_config(args.piper_joint)}
    elite_joints = {**config.get("elite_joints", {}), **parse_joint_config(args.elite_joint)}
    robots = [
        {
            "name": "piper",
            "urdf": assets_root / config.get("piper_urdf", "piper_description/urdf/piper_description.urdf"),
            "pose": pose_matrix(config["piper_pose"]),
            "color": (80, 135, 230, 255),
            "joints": piper_joints,
        },
        {
            "name": "elite",
            "urdf": assets_root / config.get("elite_urdf", "elite_description/urdf/ec66_description.urdf"),
            "pose": pose_matrix(config["elite_pose"]),
            "color": (235, 130, 70, 255),
            "joints": elite_joints,
        },
    ]

    manifest = {
        "config": str(config_path),
        "out_root": str(out_root),
        "robots": [],
    }
    for robot in robots:
        robot_dir = out_root / robot["name"]
        robot_dir.mkdir(parents=True, exist_ok=True)
        meshes = load_urdf_meshes(robot["urdf"], robot["pose"], robot["color"], robot["joints"])
        robot_items = []
        for index, mesh in enumerate(meshes):
            mesh = mesh.copy()
            mesh.remove_unreferenced_vertices()
            if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
                continue
            out_path = robot_dir / f"{robot['name']}_{index:02d}.stl"
            mesh.export(out_path)
            bounds = np.asarray(mesh.bounds, dtype=float)
            robot_items.append(
                {
                    "name": f"{robot['name']}_{index:02d}",
                    "path": str(out_path.relative_to(out_root).as_posix()),
                    "bounds": bounds.tolist(),
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                }
            )
        manifest["robots"].append(
            {
                "name": robot["name"],
                "urdf": str(robot["urdf"]),
                "mesh_count": len(robot_items),
                "meshes": robot_items,
            }
        )

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake calibrated robot URDF meshes into MuJoCo-friendly STL assets.")
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_robot_assets")
    parser.add_argument("--piper-joint", action="append", default=[])
    parser.add_argument("--elite-joint", action="append", default=[])
    args = parser.parse_args()
    manifest = export_robot_meshes(args)
    print(json.dumps({k: manifest[k] for k in ["config", "out_root"]}, indent=2, ensure_ascii=False))
    for robot in manifest["robots"]:
        print(f"{robot['name']}: exported {robot['mesh_count']} meshes")


if __name__ == "__main__":
    main()
