"""Prepare the nominal Elite start, then run ONE live Real10 policy action.

With --execute: onsite HOME confirmation -> home/verify -> onsite EXECUTE
confirmation -> fresh inference/action. No loop, auto-retry or servo enable.
Without --execute there is no homing or action dispatch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from real10_pi05_bridge import (
    LatestColorCamera, ModelProcess, MODEL_PYTHON, ROOT, SIDE_SERIAL, TOP_SERIAL,
    TRAINING_ROOT, controller_preview, dump_json,
)


# Recorded common start: path/path1_pose.txt and path/path2_pose.txt, first XYZ;
# RPY from tools/play_elite_path.py DEFAULT_FIXED_RPY. Not mechanical zero.
DEFAULT_HOME_POSE = [-336.181546, 251.652310, 321.987936,
                     3.0661665148309702, .03647610536354759, .06842115274422467]
MAX_DEMO_STEP_MM = 10.0  # Software ceiling; actual onsite cap is operator-selected.


def pose_errors(current, target):
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    if current.shape != (6,) or target.shape != (6,) or not np.isfinite([current, target]).all():
        raise ValueError("TCP poses must each contain six finite mm/rad values.")
    rpy_error = (current[3:] - target[3:] + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(current[:3] - target[:3])), float(np.max(np.abs(rpy_error)))


def action_plan(prediction, pose, max_step_mm, *, translation_gain=1.0):
    mapped = controller_preview(prediction, pose)
    if mapped["piper_step_command"] == -1:
        raise ValueError("This first onsite adapter supports hold/feed only; retract is not dispatched.")
    if not np.isfinite(max_step_mm) or not 0 < max_step_mm <= MAX_DEMO_STEP_MM:
        raise ValueError(f"Demo translation norm limit must be in (0, {MAX_DEMO_STEP_MM:g}] mm.")
    if not np.isfinite(translation_gain) or not 0 < translation_gain <= 100:
        raise ValueError("Demo translation gain must be in (0, 100].")
    delta = np.asarray(prediction["elite_tcp_delta_6d"], dtype=float)
    scaled = delta.copy()
    scaled[:3] *= translation_gain
    norm = float(np.linalg.norm(scaled[:3]))
    guarded = scaled.copy()
    if norm > max_step_mm:
        guarded[:3] *= max_step_mm / norm
    return {
        "raw_elite_tcp_delta_6d": delta.tolist(),
        "demo_scaled_elite_tcp_delta_6d": scaled.tolist(),
        "translation_gain": translation_gain,
        "guarded_elite_tcp_delta_6d": guarded.tolist(),
        "translation_clipped": norm > max_step_mm,
        "translation_limit_mm": max_step_mm,
        "elite_target_tcp_pose_6d": (np.asarray(pose) + guarded).tolist(),
        "piper_intent_id": prediction["piper_intent_id"],
        "piper_step_command": mapped["piper_step_command"],
        "units": "xyz_mm, rpy_rad; fixed current orientation; displacement, not velocity",
    }


def robot_ready(ec):
    values = {"state": ec.state, "mode": ec.mode, "servo": ec.servo_status,
              "synchronized": ec.sync_status, "estop": ec.estop_status,
              "precise_position": ec.get_servo_precise_position_status(is_block=False)}
    if (values["state"] != ec.RobotState.STOP or values["mode"] != ec.RobotMode.REMOTE
            or values["servo"] not in (True, 1) or values["synchronized"] not in (True, 1)
            or values["estop"] != 0 or values["precise_position"] != 1):
        raise RuntimeError(f"Elite is not ready; no automatic mode/servo/alarm changes: {values}")
    return {key: str(value) for key, value in values.items()}


def checked_joints(ec, target_pose, max_joint_step_deg=5.0):
    current = np.asarray(ec.current_joint, dtype=float)
    target = np.asarray(ec.get_inverse_kinematic(pose=target_pose, ref_joint=current.tolist()), dtype=float)
    # The onsite controller returns six joints; preserve the SDK/controller's
    # exact output length instead of padding/truncating it based on old docs.
    if (current.ndim != 1 or target.shape != current.shape or current.size not in (6, 8)
            or not np.isfinite(current).all() or not np.isfinite(target).all()):
        raise RuntimeError("Invalid IK/current joint result; no motion.")
    if np.max(np.abs(target - current)) > max_joint_step_deg:
        raise RuntimeError(f"IK joint jump exceeds {max_joint_step_deg:g} degrees; no motion.")
    return target.tolist()


def load_feeder(args):
    # Use the verified onsite collection adapter with source-bind support, not
    # the older algorithm-root hardware copy. No source file is modified.
    spec = importlib.util.spec_from_file_location("real10_onsite_udp", args.feeder_adapter)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.UdpFeederConfig(host=args.feeder_host, port=8888,
                                   local_host=args.feeder_local_host, local_port=37011,
                                   timeout_s=2.0)
    feeder = module.UdpFeederDevice(config)
    feeder.connect()  # Bind only; this does not send a packet.
    return feeder


def execute_elite_move(ec, target, joints, speed, report, save, *, timeout_s=10.0,
                       position_tolerance_mm=.1):
    """Shared move/arrival/stop path; the caller owns home versus policy scope."""
    moving = False
    try:
        report["elite_move_attempted"] = True
        save()
        moving = True  # Also stop on an ambiguous SDK exception after send.
        result = ec.move_joint(target_joint=joints, speed=speed, block=False)
        report["elite_move_reply"] = result
        report["elite_move_accepted"] = result is True
        save()
        # SDK errors can be truthy tuples: do not interpret them as success.
        if result is not True:
            raise RuntimeError(f"Elite did not acknowledge move; no feed: {result!r}")
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            state = ec.state
            if state not in (ec.RobotState.STOP, ec.RobotState.PLAY):
                raise RuntimeError(f"Elite entered {state}; no feed.")
            actual = list(ec.current_pose)
            error_mm, error_rad = pose_errors(actual, target)
            if (state == ec.RobotState.STOP and time.monotonic() - started >= .3
                    and error_mm <= position_tolerance_mm and error_rad <= .01):
                moving = False
                report.update(elite_target_reached=True, elite_after_pose=actual,
                              elite_target_error_mm=error_mm, elite_target_rpy_error_rad=error_rad)
                save()
                return
            time.sleep(.05)
        report.update(elite_after_pose=actual, elite_target_error_mm=error_mm,
                      elite_target_rpy_error_rad=error_rad)
        save()
        raise TimeoutError(f"Elite did not reach the target within {timeout_s:g} s; no feed.")
    finally:
        if moving:
            report["elite_stop_attempted"] = True
            # A failed log write must not prevent a stop attempt.
            try:
                report["elite_stop_reply"] = ec.stop()
            except BaseException as exc:
                report["elite_stop_error"] = f"{type(exc).__name__}: {exc}; use onsite emergency stop"
            save()


def home_elite(ec, args, home, save):
    """Absolute preparation move, never a model label or a feeder operation."""
    try:
        home["robot_before"] = robot_ready(ec)
        current = list(ec.current_pose)
        distance, angle_error = pose_errors(current, args.home_pose)
        home.update(before_pose=current, distance_mm=distance, rpy_error_rad=angle_error)
        save()
        print(f"归位当前TCP：{current}\n目标TCP：{args.home_pose}\n"
              f"位置距离 {distance:.3f} mm；最大RPY差 {angle_error:.4f} rad。", flush=True)
        if distance <= .5 and angle_error <= .01:
            home.update(status="already_at_start", elite_target_reached=True,
                        elite_move_attempted=False, elite_after_pose=current,
                        elite_target_error_mm=distance, elite_target_rpy_error_rad=angle_error)
        else:
            if distance > args.home_max_distance_mm:
                raise RuntimeError("Home distance exceeds the configured limit; reposition manually, no inference/feed.")
            joints = checked_joints(ec, args.home_pose, max_joint_step_deg=60.0)
            home.update(status="moving", ik_target_joints=joints)
            save()
            execute_elite_move(ec, args.home_pose, joints, args.home_speed, home, save,
                               timeout_s=args.home_timeout_s, position_tolerance_mm=.5)
            home["status"] = "completed"
        home["completed_timestamp"] = time.time()
        save()
    except BaseException as exc:
        home.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def move_and_feed_once(ec, feeder, plan, joints, speed, report, save, *, wait_feeder_ack=True):
    if np.linalg.norm(plan["guarded_elite_tcp_delta_6d"][:3]) > 1e-8:
        execute_elite_move(ec, plan["elite_target_tcp_pose_6d"], joints, speed, report, save)
    else:
        report.update(elite_move_skipped="zero_translation", elite_target_reached=True,
                      elite_after_pose=list(ec.current_pose))
        save()
    if plan["piper_step_command"] == 1:
        robot_ready(ec)
        report.update(piper_send_attempted=True, piper_packets_sent=None)
        save()
        reply = feeder.feed_once(wait_response=wait_feeder_ack)  # Exactly once. Never retry.
        report.update(piper_packets_sent=1, piper_ack_requested=wait_feeder_ack,
                      piper_ack_received=(reply is not None if wait_feeder_ack else None),
                      piper_reply=None if reply is None else reply.decode("utf-8", errors="replace"),
                      piper_physical_execution_confirmed=None,
                      piper_dispatch_status=("ack_received" if reply is not None else
                                             "sent_without_ack_wait" if not wait_feeder_ack else
                                             "ack_missing"))
    else:
        report["piper_dispatch_status"] = "hold_no_packet"
    save()


def save_views(path, frames, title, font_path):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (1280, 420), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 23)
    draw.text((16, 10), title, font=font, fill="white")
    for index, view in enumerate(("side", "top")):
        picture = Image.fromarray(cv2.cvtColor(frames[view][0], cv2.COLOR_BGR2RGB))
        picture.thumbnail((620, 350))
        x = 16 + index * 632
        draw.text((x, 46), "侧视 side" if view == "side" else "俯视 top", font=font, fill="white")
        canvas.paste(picture, (x, 82))
    canvas.save(path)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "real10_once_v2_homing", "status": "starting", "execute": args.execute,
              "task": args.task, "elite_ip": args.elite_ip,
              "side_serial": args.side_serial, "top_serial": args.top_serial,
              "elite_move_attempted": False, "elite_target_reached": False,
              "piper_send_attempted": False, "piper_packets_sent": 0,
              "piper_physical_execution_confirmed": None,
              "piper_history": "missing; unchanged training encoder validity=0",
              "feeder_destination": f"{args.feeder_host}:8888",
              "feeder_source": f"{args.feeder_local_host}:37011",
              "feeder_adapter": str(args.feeder_adapter),
              "joint_speed_percent": args.speed, "visual_status": "not_viewed",
              "homing": {"status": "pending" if args.execute else "disabled_no_execution",
                         "target_pose": args.home_pose, "pose_source": args.home_pose_source,
                         "speed_percent": args.home_speed, "max_distance_mm": args.home_max_distance_mm,
                         "max_joint_step_deg": 60.0, "timeout_s": args.home_timeout_s,
                         "position_tolerance_mm": .5, "rpy_tolerance_rad": .01,
                         "elite_move_attempted": False, "elite_target_reached": False,
                         "is_policy_action": False, "feeder_packets_allowed": 0}}
    save = lambda: dump_json(args.out / "report.json", report)
    save()
    worker, ec, feeder, cameras = None, None, None, {}
    started = time.time()
    try:
        if args.execute:
            print(f"先归位到采集起点（非机械零点）：{args.home_pose}\n"
                  "归位是绝对位姿运动，不受模型1毫米限幅约束，且不含自动避障。\n"
                  "确认其他使用者已交接、其他控制程序停止、导丝状态适合归位、回程无障碍，急停可及。",
                  flush=True)
            if input("输入 HOME 确认交接并归位；其他输入取消 > ").strip() != "HOME":
                report["status"] = "cancelled_before_hardware"
                report["homing"]["status"] = "cancelled"
                return
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3.0)
        report["robot_before"] = robot_ready(ec)
        if args.execute:
            home_elite(ec, args, report["homing"], save)
        print("加载模型约30秒；归位阶段不会递丝，随后另行确认模型动作。", flush=True)
        worker = ModelProcess(args, args.out / "worker.log")
        report["model"] = worker.ready
        for view, serial in (("side", args.side_serial), ("top", args.top_serial)):
            cameras[view] = LatestColorCamera(serial)
            cameras[view].read()
        if args.execute:
            feeder = load_feeder(args)
            print(f"归位已到位。即将执行一次模型动作：Elite单步平移最多{args.max_step_mm:g}毫米、速度5%以内；feed时递丝一次。\n"
                  "停止其他控制程序，确认递丝已停止、人员离开运动范围，急停可及。", flush=True)
            if input("输入 EXECUTE 开始；其他输入取消 > ").strip() != "EXECUTE":
                report["status"] = "cancelled"
                return
        frames = {view: camera.read() for view, camera in cameras.items()}
        pose = [float(value) for value in ec.current_pose]
        if args.execute:
            report["robot_before_inference"] = robot_ready(ec)
            distance, angle_error = pose_errors(pose, args.home_pose)
            if distance > .5 or angle_error > .01:
                raise RuntimeError("Elite left the home pose while waiting; cancel without inference/feed.")
            if any(frame[1] < report["homing"]["completed_timestamp"] for frame in frames.values()):
                raise RuntimeError("Pre-homing camera frame rejected; no inference/feed.")
        report["observation"] = {"elite_tcp_pose_6d": pose, "task": args.task,
                                 "previous_controller_state": None,
                                 "camera_host_timestamps": {v: f[1] for v, f in frames.items()}}
        begin = time.perf_counter()
        reply = worker.predict(0, {v: f[0] for v, f in frames.items()}, pose, args.task, None)
        report.update(policy_output=reply["prediction"], state_32=reply["state_32"],
                      roundtrip_seconds=time.perf_counter() - begin)
        save_views(args.out / "before.png", frames, "真实模型单次执行｜动作前观测", args.font)
        report["robot_before_dispatch"] = robot_ready(ec)
        current = np.asarray(ec.current_pose, dtype=float)
        distance, angle_error = pose_errors(current, pose)
        if distance > .25 or angle_error > .01:
            raise RuntimeError("TCP changed since inference; do not execute a stale decision.")
        plan = action_plan(reply["prediction"], current, args.max_step_mm)
        joints = checked_joints(ec, plan["elite_target_tcp_pose_6d"])
        report.update(plan=plan, dispatch_base_pose=current.tolist(), ik_target_joints=joints)
        save()
        print(json.dumps(plan, ensure_ascii=False), flush=True)
        if any(not 0 <= time.time() - frame[1] <= 2.0 for frame in frames.values()):
            raise RuntimeError("Camera observation exceeds 2 s before dispatch; no action.")
        if args.execute:
            move_and_feed_once(ec, feeder, plan, joints, args.speed, report, save)
            report["status"] = "completed_one_action_attempt"
            time.sleep(.5)
            after = {view: camera.read() for view, camera in cameras.items()}
            save_views(args.out / "after.png", after, "真实模型单次执行｜动作后观测（不代表任务成功）", args.font)
        else:
            report["status"] = "inference_and_ik_only_no_action"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup = [(f"camera:{v}", c.close) for v, c in cameras.items()]
        if feeder is not None:
            cleanup.append(("feeder", feeder.close))
        if ec is not None:
            cleanup.append(("Elite", ec.disconnect_ETController))
        if worker is not None:
            cleanup.append(("model", worker.close))
        report["cleanup_errors"] = []
        for name, close in cleanup:
            try:
                close()
            except BaseException as exc:
                report["cleanup_errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        report["elapsed_seconds"] = time.time() - started
        save()
        print(f"结束：{report['status']}；日志：{args.out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("left", "right"))
    parser.add_argument("--elite-ip", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--home-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="absolute start TCP in mm/rad; default is the recorded common collection start")
    parser.add_argument("--home-speed", type=float, default=5.0, help="homing joint speed percent, at most 5")
    parser.add_argument("--home-max-distance-mm", type=float, default=500.0,
                        help="maximum allowed initial distance to start, at most 500 mm")
    parser.add_argument("--home-timeout-s", type=float, default=60.0,
                        help="Elite homing arrival timeout; onsite may choose up to 300 s")
    parser.add_argument("--max-step-mm", type=float, default=1.0,
                        help=f"operator-selected translation norm cap in mm; software ceiling {MAX_DEMO_STEP_MM:g}")
    parser.add_argument("--speed", type=float, default=5.0, help="joint speed percent, not mm/s")
    parser.add_argument("--feeder-host", default="192.168.5.13")
    parser.add_argument("--feeder-local-host", default="192.168.5.11")
    parser.add_argument("--feeder-adapter", type=Path, default=Path(
        "/home/zsw/PycharmProjects/real_collection/hardware/feeder_device/udp_controller.py"))
    parser.add_argument("--side-serial", default=SIDE_SERIAL)
    parser.add_argument("--top-serial", default=TOP_SERIAL)
    parser.add_argument("--model-python", type=Path, default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint", type=Path, default=TRAINING_ROOT / "elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint", type=Path, default=TRAINING_ROOT / "piper/mixed_head_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    args.home_pose_source = ("explicit --home-pose" if args.home_pose is not None else
                            "path/path1_pose.txt and path2_pose.txt first XYZ + play_elite_path.DEFAULT_FIXED_RPY")
    args.home_pose = list(DEFAULT_HOME_POSE if args.home_pose is None else args.home_pose)
    if not np.isfinite(args.home_pose).all():
        parser.error("--home-pose requires six finite mm/rad values")
    if not np.isfinite(args.home_speed) or not 0 < args.home_speed <= 5:
        parser.error("--home-speed must be in (0, 5] percent")
    if not np.isfinite(args.home_max_distance_mm) or not 0 < args.home_max_distance_mm <= 500:
        parser.error("--home-max-distance-mm must be in (0, 500]")
    if not np.isfinite(args.home_timeout_s) or not 10 <= args.home_timeout_s <= 300:
        parser.error("--home-timeout-s must be in [10, 300] seconds")
    if not np.isfinite(args.max_step_mm) or not 0 < args.max_step_mm <= MAX_DEMO_STEP_MM:
        parser.error(f"--max-step-mm must be in (0, {MAX_DEMO_STEP_MM:g}] mm")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 5:
        parser.error("--speed must be in (0, 5] percent")
    if args.side_serial == args.top_serial or not args.font.is_file():
        parser.error("two different camera serials and a Chinese font are required")
    run(args)


if __name__ == "__main__":
    main()
