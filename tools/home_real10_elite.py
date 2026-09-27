"""Return Elite to the recorded Real10 capture start, without model/cameras/feeder.

Requires --execute and an onsite HOME confirmation before connecting Elite.
The target is the capture start, not mechanical zero. There is no path planner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from run_real10_pi05_once import DEFAULT_HOME_POSE, dump_json, home_elite


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "real10_elite_home_only_v1",
        "status": "pending",
        "elite_ip": args.elite_ip,
        "execute": args.execute,
        "started_timestamp": time.time(),
        "homing": {
            "status": "pending",
            "target_pose": args.home_pose,
            "pose_source": args.home_pose_source,
            "speed_percent": args.home_speed,
            "max_distance_mm": args.home_max_distance_mm,
            "max_joint_step_deg": 60.0,
            "timeout_s": args.home_timeout_s,
            "position_tolerance_mm": .5,
            "rpy_tolerance_rad": .01,
            "elite_move_attempted": False,
            "elite_target_reached": False,
            "is_policy_action": False,
            "feeder_packets_allowed": 0,
        },
        "model_loaded": False,
        "cameras_opened": False,
        "feeder_packets_sent": 0,
    }

    def save():
        dump_json(args.out / "report.json", report)

    save()
    print(f"采集起点（非机械零位）：{args.home_pose}\n"
          "这是绝对位姿归位，无自动避障；确认设备交接、TCP/tool、导丝状态、回程净空和急停位置。",
          flush=True)
    if input("输入 HOME 确认归位；其他输入取消 > ").strip() != "HOME":
        report["status"] = "cancelled_before_connection"
        report["homing"]["status"] = "cancelled"
        report["finished_timestamp"] = time.time()
        save()
        print(f"已取消；未连接设备。报告：{args.out / 'report.json'}", flush=True)
        return

    ec = None
    try:
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3.0)
        home_elite(ec, args, report["homing"], save)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if ec is not None:
            try:
                ec.disconnect_ETController()
            except BaseException as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "failed_cleanup"
        report["finished_timestamp"] = time.time()
        save()
        print(f"结束：{report['status']}；报告：{args.out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite-ip", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="enable HOME prompt and possible physical return")
    parser.add_argument("--home-pose", nargs=6, type=float,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="capture start TCP in xyz mm / rpy rad")
    parser.add_argument("--home-speed", type=float, default=5.0,
                        help="joint speed percent, at most 5")
    parser.add_argument("--home-max-distance-mm", type=float, default=500.0)
    parser.add_argument("--home-timeout-s", type=float, default=60.0,
                        help="Elite arrival timeout; onsite may choose up to 300 s")
    args = parser.parse_args()
    if not args.execute:
        parser.error("physical homing requires explicit --execute")
    args.home_pose_source = ("explicit --home-pose" if args.home_pose is not None else
                             "recorded common capture start + fixed RPY")
    args.home_pose = list(DEFAULT_HOME_POSE if args.home_pose is None else args.home_pose)
    if not np.isfinite(args.home_pose).all():
        parser.error("--home-pose requires six finite mm/rad values")
    if not np.isfinite(args.home_speed) or not 0 < args.home_speed <= 5:
        parser.error("--home-speed must be in (0, 5] percent")
    if not np.isfinite(args.home_max_distance_mm) or not 0 < args.home_max_distance_mm <= 500:
        parser.error("--home-max-distance-mm must be in (0, 500] mm")
    if not np.isfinite(args.home_timeout_s) or not 10 <= args.home_timeout_s <= 300:
        parser.error("--home-timeout-s must be in [10, 300] seconds")
    run(args)


if __name__ == "__main__":
    main()
