from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from yourdfpy import URDF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def export_link_meshes(urdf_path: Path, robot_name: str, out_root: Path) -> dict:
    robot = URDF.load(str(urdf_path))
    robot_dir = out_root / robot_name
    robot_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, node_name in enumerate(robot.scene.graph.nodes_geometry):
        transform, geometry_name = robot.scene.graph.get(node_name)
        if geometry_name is None:
            continue
        geometry = robot.scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            continue
        out_path = robot_dir / f"{robot_name}_{index:02d}.stl"
        mesh.export(out_path)
        items.append(
            {
                "name": f"{robot_name}_{index:02d}",
                "node": str(node_name),
                "path": str(out_path.relative_to(out_root).as_posix()),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "initial_transform": np.asarray(transform, dtype=float).tolist(),
            }
        )
    return {"name": robot_name, "urdf": str(urdf_path), "mesh_count": len(items), "meshes": items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export robot visual meshes in their local node frames for MuJoCo FK syncing.")
    parser.add_argument("--out", default="simulation_output/mujoco_robot_link_assets")
    parser.add_argument("--piper-urdf", default="robot_assets/standardized/piper_description/urdf/piper_description.urdf")
    parser.add_argument("--elite-urdf", default="robot_assets/standardized/elite_description/urdf/ec66_description.urdf")
    args = parser.parse_args()

    out_root = (REPO_ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "out_root": str(out_root),
        "robots": [
            export_link_meshes((REPO_ROOT / args.piper_urdf).resolve(), "piper", out_root),
            export_link_meshes((REPO_ROOT / args.elite_urdf).resolve(), "elite", out_root),
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for robot in manifest["robots"]:
        print(f"{robot['name']}: exported {robot['mesh_count']} local meshes")


if __name__ == "__main__":
    main()
