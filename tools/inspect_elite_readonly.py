"""Read EC identity, readiness, tool frames and motor feedback; never send motion."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def run(args):
    from elite import EC

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(), "requested_ip": args.elite_ip,
        "motion_commands": 0, "feeder_packets": 0, "settings_changed": False,
        "query_errors": {}, "samples": [],
        "scope": "controller feedback only; does not establish visible physical motion",
    }

    def save():
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def read(name, getter):
        try:
            value = getter()
            return {"name": str(value), "value": value.value} if hasattr(value, "value") else value
        except Exception as exc:
            report["query_errors"][name] = f"{type(exc).__name__}: {exc}"
            return None

    ec = None
    save()
    try:
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["socket_peer"] = list(ec.sock_cmd.getpeername())
        getters = {
            "software_version": lambda: ec.soft_version,
            "robot_type": lambda: ec.robot_type,
            "state": lambda: ec.state, "mode": lambda: ec.mode,
            "servo_status": lambda: ec.servo_status, "synchronized": lambda: ec.sync_status,
            "estop": lambda: ec.estop_status, "global_run_speed": lambda: ec.run_speed,
            "precise_position": lambda: ec.get_servo_precise_position_status(is_block=False),
            "m472": lambda: ec.get_digital_io("M472"), "m473": lambda: ec.get_digital_io("M473"),
            "brakes_released": lambda: ec.send_CMD("get_servo_brake_off_status"),
            "tool_number_run": lambda: ec.tool_frame_num_in_run_mode,
            "active_tool_offset_mm_rad": lambda: ec._get_tool_coord(ec.tool_frame_num_in_run_mode.value, unit_type=1),
        }
        for name, getter in getters.items():
            report[name] = read(name, getter)
            save()
        for index in range(3):
            sample = {"timestamp": time.time()}
            for name, getter in {
                "tcp_pose_mm_rad": lambda: ec.current_pose,
                "computed_joints_deg": lambda: ec.current_joint,
                "motor_positions": lambda: ec.get_motor_pos(),
                "encoder_values": lambda: ec.encoder_values,
                "motor_speed": lambda: ec.motor_speed,
            }.items():
                sample[name] = read(f"sample_{index}.{name}", getter)
            report["samples"].append(sample)
            save()
            if index < 2:
                time.sleep(.3)
    finally:
        if ec is not None:
            ec.disconnect_ETController()
        save()
    print(json.dumps({key: report.get(key) for key in (
        "socket_peer", "robot_type", "state", "mode", "precise_position", "m472",
        "global_run_speed", "tool_number_run", "active_tool_offset_mm_rad", "query_errors",
    )}, ensure_ascii=False, indent=2))
    print(args.out / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite-ip", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
