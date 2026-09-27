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

from tools.visualize_dual_robot_scene import load_urdf_meshes, pose_matrix


def mesh_size(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)


def robot_size(root: Path, urdf_rel: str, pose: list[float], joints: dict) -> np.ndarray:
    meshes = load_urdf_meshes(root / urdf_rel, pose_matrix(pose), (1, 1, 1, 255), joints)
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
    return bounds[:, 1, :].max(axis=0) - bounds[:, 0, :].min(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and optionally calibrate scene scale using vessel physical length.")
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--target-vessel-length", type=float, default=None, help="Target longest vessel dimension in meters.")
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = (REPO_ROOT / config.get("assets_root", "robot_assets/standardized")).resolve()
    vessel = trimesh.load(REPO_ROOT / config["vessel"], force="mesh")
    if not isinstance(vessel, trimesh.Trimesh):
        raise TypeError(f"Expected vessel mesh, got {type(vessel)!r}")

    raw_vessel_size = mesh_size(vessel)
    raw_longest = float(raw_vessel_size.max())
    current_scale = float(config.get("vessel_scale", 1.0))
    current_vessel_size = raw_vessel_size * current_scale

    piper_size = robot_size(root, config["piper_urdf"], config["piper_pose"], config.get("piper_joints", {}))
    elite_size = robot_size(root, config["elite_urdf"], config["elite_pose"], config.get("elite_joints", {}))

    report = {
        "config": config_path.as_posix(),
        "vessel_raw_size_maybe_model_units": [float(x) for x in raw_vessel_size],
        "vessel_scale": current_scale,
        "vessel_scaled_size_m": [float(x) for x in current_vessel_size],
        "vessel_scaled_longest_m": float(current_vessel_size.max()),
        "piper_scene_size_m": [float(x) for x in piper_size],
        "elite_scene_size_m": [float(x) for x in elite_size],
    }

    if args.target_vessel_length is not None:
        new_scale = float(args.target_vessel_length) / raw_longest
        report["target_vessel_length_m"] = float(args.target_vessel_length)
        report["recommended_vessel_scale"] = new_scale
        report["recommended_vessel_size_m"] = [float(x) for x in raw_vessel_size * new_scale]
        if args.write_config:
            config["vessel_scale"] = round(new_scale, 6)
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            report["updated_config"] = config_path.as_posix()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
