from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.robot_kinematics import DualRobotKinematics


def parse_joint_config(items: list[str]) -> dict[str, float]:
    joints: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Joint config item must be name=value, got {item!r}")
        name, value = item.split("=", 1)
        joints[name.strip()] = float(value)
    return joints


def update_config_tools(config_path: Path, kin: DualRobotKinematics, report: dict) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["piper_tool_link"] = kin.piper.tool_link
    config["piper_tool_offset"] = [round(float(x), 6) for x in kin.piper.tool_offset]
    config["piper_tool_point"] = [round(float(x), 6) for x in report["piper"]["tool_world"]]
    config["elite_tool_link"] = kin.elite.tool_link
    config["elite_magnet_offset"] = [round(float(x), 6) for x in kin.elite.tool_offset]
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FK for the dual robot scene tools.")
    parser.add_argument("--config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--out", default="simulation_output/robot_scene_mvp/kinematics_report.json")
    parser.add_argument("--piper-joint", action="append", default=[], help="Piper joint override, e.g. joint1=0.2")
    parser.add_argument("--elite-joint", action="append", default=[], help="Elite joint override, e.g. joint2=-0.4")
    parser.add_argument("--write-config", action="store_true", help="Write derived Piper tool link/offset back to scene_config.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    piper_joints = parse_joint_config(args.piper_joint)
    elite_joints = parse_joint_config(args.elite_joint)
    kin = DualRobotKinematics.from_scene_config(config_path)
    report = kin.summary(piper_joints=piper_joints, elite_joints=elite_joints)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_config:
        update_config_tools(config_path, kin, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved {out_path}")
    if args.write_config:
        print(f"updated {config_path}")


if __name__ == "__main__":
    main()
