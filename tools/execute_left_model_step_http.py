"""One authorized left-task model step, <=1 mm at 1% speed; no feeder or homing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from real10_pi05_bridge import MODEL_PYTHON, TRAINING_ROOT, ModelProcess, dump_json
from run_real10_pi05_once import (
    action_plan, checked_joints, execute_elite_move, pose_errors, robot_ready,
)


def capture(opener, base, out, stage):
    frames, timestamps = {}, {}
    for role in ("side", "top"):
        with opener.open(base.rstrip("/") + "/" + role + ".jpg", timeout=4) as response:
            encoded = response.read()
            timestamps[role] = float(response.headers["X-Capture-Timestamp"])
        frames[role] = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if frames[role] is None or frames[role].shape != (1080, 1920, 3):
            raise RuntimeError(f"invalid {role} image")
        (out / f"{role}_{stage}.jpg").write_bytes(encoded)
    return frames, timestamps


def validate_dispatch(prediction, observed_pose, current_pose, timestamps, now):
    plan = action_plan(prediction, current_pose, 1.0)
    if plan["piper_step_command"] != 0:
        raise RuntimeError("New feeder prediction is not hold; no movement permitted in this scope")
    distance, angle = pose_errors(current_pose, observed_pose)
    if distance > .05 or angle > .001:
        raise RuntimeError("TCP changed since observation; no motion")
    if any(not 0 <= now - timestamp <= 2 for timestamp in timestamps.values()):
        raise RuntimeError("Camera observation is stale; no motion")
    return plan


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "left_model_step_http_v1", "status": "loading_model",
              "started_at": datetime.now().astimezone().isoformat(), "task": "left",
              "execute_requested": args.execute, "translation_limit_mm": 1.0,
              "joint_speed_percent": 1.0, "max_joint_change_deg": 1.0,
              "motion_command_attempts": 0, "motion_commands_accepted": 0,
              "feeder_packets": 0, "homing": False, "automatic_retries": False,
              "hardware_executed": False, "previous_controller_state": None,
              "observation_transport": "live camera HTTP JPEG", "cleanup_errors": []}
    save = lambda: dump_json(args.out / "report.json", report)
    save()
    worker, ec = None, None
    start = time.monotonic()
    try:
        worker = ModelProcess(args, args.out / "worker.log")
        report["model"] = worker.ready
        print(json.dumps({"event": "model_ready", "elapsed_seconds": time.monotonic()-start}), flush=True)
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["robot_before"] = robot_ready(ec)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        images, stamps = capture(opener, args.preview_url, args.out, "before")
        pose = list(ec.current_pose)
        pose_time = time.time()
        ages = {role: pose_time-stamp for role, stamp in stamps.items()}
        if any(not 0 <= age <= .1 for age in ages.values()) or abs(stamps["side"]-stamps["top"]) > .1:
            raise RuntimeError("Image/pose timestamp difference exceeds 100 ms")
        report["observation"] = {"tcp_pose_mm_rad": pose, "pose_read_finished_timestamp": pose_time,
                                 "camera_host_timestamps": stamps, "image_age_s": ages}
        reply = worker.predict(0, images, pose, "left", None)
        report.update(policy_output=reply["prediction"], state_32=reply["state_32"])
        report["robot_before_dispatch"] = robot_ready(ec)
        current = list(ec.current_pose)
        plan = validate_dispatch(reply["prediction"], pose, current, stamps, time.time())
        joints_before = list(ec.current_joint)
        joints = checked_joints(ec, plan["elite_target_tcp_pose_6d"], max_joint_step_deg=1.0)
        report.update(plan=plan, dispatch_base_pose=current, joints_before=joints_before,
                      ik_target_joints=joints, status="ready_one_bounded_action")
        save()
        print(json.dumps({"event": "planned", "plan": plan}, ensure_ascii=False), flush=True)
        if not args.execute:
            report["status"] = "preview_only"
            return
        # Check again after IK and report I/O, immediately before the one dispatch.
        report["robot_final_pre_dispatch"] = robot_ready(ec)
        final_pose = list(ec.current_pose)
        validate_dispatch(reply["prediction"], current, final_pose, stamps, time.time())
        if pose_errors(final_pose, current)[0] > .01:
            raise RuntimeError("TCP drift after IK; no motion")
        norm = float(np.linalg.norm(plan["guarded_elite_tcp_delta_6d"][:3]))
        if norm <= 1e-8:
            report["status"] = "zero_translation_no_motion"
            return
        report.update(status="dispatching", motion_command_attempts=1,
                      motion_commands_accepted=None, dispatch_timestamp=time.time())
        save()
        execute_elite_move(ec, plan["elite_target_tcp_pose_6d"], joints, 1.0, report, save,
                           timeout_s=10.0, position_tolerance_mm=.02)
        report.update(hardware_executed=True, motion_commands_accepted=1,
                      status="completed_one_left_step", motion_completed_timestamp=time.time())
        report["robot_after"] = robot_ready(ec)
        report["tcp_after_mm_rad"] = list(ec.current_pose)
        report["joints_after"] = list(ec.current_joint)
        actual_delta = np.asarray(report["tcp_after_mm_rad"][:3])-np.asarray(current[:3])
        report["measured_tcp_delta_xyz_mm"] = actual_delta.tolist()
        report["measured_translation_norm_mm"] = float(np.linalg.norm(actual_delta))
        save()
        time.sleep(.2)
        _, after_stamps = capture(opener, args.preview_url, args.out, "after")
        report["after_camera_host_timestamps"] = after_stamps
        if any(stamp < report["motion_completed_timestamp"] for stamp in after_stamps.values()):
            raise RuntimeError("Post-motion camera frame predates motion completion")
    except BaseException as exc:
        if report.get("elite_move_accepted") is True:
            report.update(hardware_executed=True, motion_commands_accepted=1)
        report.update(status="failed_after_motion" if report["hardware_executed"] else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for name, close in (("elite", ec.disconnect_ETController if ec else None),
                            ("model", worker.close if worker else None)):
            if close:
                try:
                    close()
                except Exception as exc:
                    report["cleanup_errors"].append(f"{name}: {exc}")
        report["elapsed_seconds"] = time.monotonic()-start
        save()
        print(json.dumps({"status": report["status"], "report": str(args.out/"report.json"),
                          "measured_translation_norm_mm": report.get("measured_translation_norm_mm"),
                          "motion_commands_accepted": report["motion_commands_accepted"],
                          "feeder_packets": 0}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--elite-ip", default="192.168.5.66")
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--model-python", type=Path, default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint", type=Path, default=TRAINING_ROOT/"elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint", type=Path, default=TRAINING_ROOT/"piper/mixed_head_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-load-timeout-s", type=float, default=180)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
