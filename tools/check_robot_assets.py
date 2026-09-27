from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from yourdfpy import URDF


DEFAULT_URDFS = {
    "piper_no_gripper": "piper_description/urdf/piper_no_gripper_description.urdf",
    "piper_with_gripper": "piper_description/urdf/piper_description.urdf",
    "elite_ec66": "elite_description/urdf/ec66_description.urdf",
}


def parse_mesh_references(urdf_path: Path) -> list[dict]:
    tree = ET.parse(urdf_path)
    refs = []
    for mesh in tree.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue
        refs.append(
            {
                "filename": filename,
                "exists": (urdf_path.parent / filename).resolve().exists() if not filename.startswith("package://") else False,
            }
        )
    return refs


def summarize_urdf(name: str, urdf_path: Path) -> dict:
    robot = URDF.load(str(urdf_path))
    joints = list(robot.robot.joints)
    links = list(robot.robot.links)
    movable_joints = [joint for joint in joints if joint.type != "fixed"]
    mesh_refs = parse_mesh_references(urdf_path)
    missing_meshes = [ref["filename"] for ref in mesh_refs if not ref["exists"]]
    return {
        "name": name,
        "urdf": urdf_path.as_posix(),
        "loaded": True,
        "links": len(links),
        "joints": len(joints),
        "movable_joints": len(movable_joints),
        "movable_joint_names": [joint.name for joint in movable_joints],
        "mesh_refs": len(mesh_refs),
        "missing_meshes": missing_meshes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal sanity check for standardized robot URDF assets.")
    parser.add_argument("--root", default="robot_assets/standardized")
    parser.add_argument("--out", default="robot_assets/standardized/asset_check_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = {"root": root.as_posix(), "robots": []}
    for name, rel_path in DEFAULT_URDFS.items():
        urdf_path = root / rel_path
        if not urdf_path.exists():
            report["robots"].append({"name": name, "urdf": urdf_path.as_posix(), "loaded": False, "error": "missing_urdf"})
            continue
        try:
            report["robots"].append(summarize_urdf(name, urdf_path))
        except Exception as exc:
            report["robots"].append({"name": name, "urdf": urdf_path.as_posix(), "loaded": False, "error": repr(exc)})

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
