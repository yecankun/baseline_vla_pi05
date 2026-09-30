"""One diagnostic left-task arm move, optionally fixed-distance <=10 mm, then one feed."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from execute_left_model_step_http import capture
from real10_pi05_bridge import MODEL_PYTHON, TRAINING_ROOT, ModelProcess, controller_preview, dump_json
from run_real10_pi05_once import (
    MAX_DEMO_STEP_MM, action_plan, checked_joints, execute_elite_move, load_feeder, pose_errors, robot_ready,
)
from restore_elite_precision_http import CameraRecording, snapshot


def check_observation(pose, current, stamps):
    distance, angle = pose_errors(current, pose)
    if distance > .01 or angle > .001:
        raise RuntimeError("TCP drift since observation; no motion")
    if any(not 0 <= time.time()-stamp <= 2 for stamp in stamps.values()):
        raise RuntimeError("Stale camera observation; no motion")


def coordination_plan(prediction, pose, fixed_step_mm=None):
    if fixed_step_mm is None:
        return action_plan(prediction, pose, 3.0, translation_gain=40.0)
    if not np.isfinite(fixed_step_mm) or not 0 < fixed_step_mm <= MAX_DEMO_STEP_MM:
        raise ValueError(f"Fixed diagnostic distance must be in (0, {MAX_DEMO_STEP_MM:g}] mm")
    mapped = controller_preview(prediction, pose)
    if mapped["piper_step_command"] == -1:
        raise ValueError("Policy requests retract; do not run a forward coordination trial")
    raw = np.asarray(prediction["elite_tcp_delta_6d"], dtype=float)
    norm = float(np.linalg.norm(raw[:3]))
    if norm <= 1e-6:
        raise ValueError("Policy direction is zero or too small to normalize")
    direction = raw[:3] / norm
    diagnostic = np.concatenate((direction * fixed_step_mm, np.zeros(3)))
    return {
        "translation_mode": "operator_fixed_distance_model_direction",
        "raw_elite_tcp_delta_6d": raw.tolist(), "raw_translation_norm_mm": norm,
        "direction_xyz": direction.tolist(), "fixed_step_mm": fixed_step_mm,
        "effective_direction_scale": fixed_step_mm / norm,
        "translation_gain": None,
        "guarded_elite_tcp_delta_6d": diagnostic.tolist(),
        "translation_limit_mm": fixed_step_mm, "translation_clipped": False,
        "elite_target_tcp_pose_6d": (np.asarray(pose) + diagnostic).tolist(),
        "piper_intent_id": prediction["piper_intent_id"],
        "piper_step_command": mapped["piper_step_command"],
        "units": "xyz_mm, rpy_rad; operator diagnostic distance, fixed orientation",
        "is_unmodified_policy_action": False,
    }


def coordinate_once(ec, feeder, plan, joints, report, save, before_feed, *, arm_only=False):
    """No feed until acknowledged motion, measured arrival and fresh images."""
    target = plan["elite_target_tcp_pose_6d"]
    limit = plan["translation_limit_mm"]
    if not np.isfinite(limit) or not 0 < limit <= MAX_DEMO_STEP_MM:
        raise RuntimeError("Requested move bound exceeds the existing executor ceiling")
    distance, _ = pose_errors(target, ec.current_pose)
    if not 0 < distance <= limit + 1e-9:
        raise RuntimeError(f"Actual target is outside the {limit:g} mm move bound")
    report.update(status="moving_arm", motion_command_attempts=1, motion_commands_accepted=None,
                  dispatch_timestamp=time.time())
    save()
    execute_elite_move(ec, target, joints, 1.0, report, save, position_tolerance_mm=.02)
    report.update(motion_commands_accepted=1, arm_completed_timestamp=time.time())
    robot_ready(ec)
    report["tcp_after_arm_mm_rad"] = list(ec.current_pose)
    actual = np.asarray(report["tcp_after_arm_mm_rad"][:3])-report["dispatch_base_pose"][:3]
    report.update(measured_tcp_delta_xyz_mm=actual.tolist(), measured_translation_norm_mm=float(np.linalg.norm(actual)))
    save()
    before_feed()
    if arm_only:
        report["status"] = "completed_arm_only_awaiting_physical_review"
        save()
        return
    report["robot_before_feed"] = robot_ready(ec)
    distance, angle = pose_errors(ec.current_pose, target)
    if distance > .02 or angle > .001:
        raise RuntimeError("Arm moved away from its target; no feed")
    report.update(status="sending_one_manual_feed", feeder_send_attempted=True,
                  feeder_packets=None, feeder_send_timestamp=time.time())
    save()
    reply = feeder.feed_once(wait_response=True)  # One datagram, never retry.
    report.update(feeder_packets=1, feeder_ack_received=reply is not None,
                  feeder_reply=reply.decode("utf-8",errors="replace") if reply is not None else None,
                  feeder_reply_received_timestamp=time.time(), feeder_physical_execution_confirmed=None,
                  status="completed_one_coordination_attempt")
    save()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema":"left_coordination_http_v1", "status":"loading_model", "task":"left",
              "started_at":datetime.now().astimezone().isoformat(), "execute_requested":args.execute,
              "arm_only":args.arm_only,
              "elite_ip":args.elite_ip, "tcp_measurement_source":"controller_tcp_readback",
              "arm_physical_execution_confirmed":None,
              "translation_mode":"operator_fixed_distance_model_direction" if args.fixed_step_mm else "scaled_policy",
              "translation_gain":None if args.fixed_step_mm else 40.0,
              "translation_limit_mm":args.fixed_step_mm or 3.0,
              "fixed_step_mm":args.fixed_step_mm, "joint_speed_percent":1.0,
              "max_joint_change_deg":5.0 if args.fixed_step_mm else 1.0,
              "feeder_action_source":"user_requested_diagnostic_coordination_not_policy_label",
              "feeder_action":{"command":"move","parameters":{"action":"forward","value":1}},
              "feeder_destination":f"{args.feeder_host}:8888", "feeder_source":f"{args.feeder_local_host}:37011",
              "motion_command_attempts":0, "motion_commands_accepted":0,
              "feeder_send_attempted":False, "feeder_packets":0,
              "feeder_physical_execution_confirmed":None, "homing":False, "automatic_retries":False,
              "previous_controller_state":None, "cleanup_errors":[], "visual_status":"not_viewed"}
    save=lambda:dump_json(args.out/"report.json",report)
    save()
    worker, ec, feeder, recording = None, None, None, None
    started=time.monotonic()
    try:
        worker=ModelProcess(args,args.out/"worker.log")
        report["model"]=worker.ready
        print(json.dumps({"event":"model_ready"}),flush=True)
        from elite import EC
        ec=EC(ip=args.elite_ip,auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["elite_socket_peer"]=list(ec.sock_cmd.getpeername())
        report["robot_before"]=robot_ready(ec)
        if args.execute and not args.arm_only:
            feeder=load_feeder(args)  # Bind before any motion; no packet sent here.
        if args.execute:
            recording=CameraRecording(args.preview_url,args.out)
            recording.start()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        images, stamps=capture(opener,args.preview_url,args.out,"before")
        pose=list(ec.current_pose)
        captured=time.time()
        if any(not 0 <= captured-stamp <= .1 for stamp in stamps.values()) or abs(stamps['side']-stamps['top'])>.1:
            raise RuntimeError("Image/pose timestamp difference exceeds 100 ms")
        report["observation"]={"tcp_pose_mm_rad":pose,"camera_host_timestamps":stamps,"pose_timestamp":captured}
        reply=worker.predict(0,images,pose,"left",None)
        report.update(policy_output=reply["prediction"],state_32=reply["state_32"])
        current=list(ec.current_pose)
        check_observation(pose,current,stamps)
        plan=coordination_plan(reply["prediction"],current,args.fixed_step_mm)
        joints=checked_joints(ec,plan["elite_target_tcp_pose_6d"],max_joint_step_deg=report["max_joint_change_deg"])
        report["ik_joint_change_deg"]=(np.asarray(joints)-np.asarray(ec.current_joint)).tolist()
        report.update(plan=plan,dispatch_base_pose=current,ik_target_joints=joints,
                      manual_feed_overrides_policy_hold=plan["piper_step_command"]==0)
        save()
        print(json.dumps({"event":"planned", "plan":plan, "manual_feed_once":not args.arm_only},ensure_ascii=False),flush=True)
        if not args.execute:
            report["status"]="preview_only_no_motion_or_feed"
            return
        report["robot_before_dispatch"]=robot_ready(ec)
        report["physical_feedback_before"]=snapshot(ec)
        check_observation(current,list(ec.current_pose),stamps)
        def before_feed():
            if recording: recording.check()
            report["physical_feedback_after"]=snapshot(ec)
            time.sleep(.2)
            _, fresh=capture(opener,args.preview_url,args.out,"before_feed")
            if any(not 0<=time.time()-stamp<=2 or stamp<report["arm_completed_timestamp"] for stamp in fresh.values()):
                raise RuntimeError("No fresh post-arm images; no feed")
            report["before_feed_camera_timestamps"]=fresh
            save()
        coordinate_once(ec,feeder,plan,joints,report,save,before_feed,arm_only=args.arm_only)
        time.sleep(.5)
        _, after=capture(opener,args.preview_url,args.out,"after")
        report["after_camera_timestamps"]=after
        report["robot_after"]=robot_ready(ec)
        report["tcp_after_mm_rad"]=list(ec.current_pose)
    except BaseException as exc:
        if report.get("elite_move_accepted") is True:
            report["motion_commands_accepted"]=1
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if recording:
            recording.close()
            report.update(camera_recording_frames=recording.frames,camera_recording_error=recording.error)
        for name,close in (("feeder",feeder.close if feeder else None),
                           ("elite",ec.disconnect_ETController if ec else None),
                           ("model",worker.close if worker else None)):
            if close:
                try: close()
                except Exception as exc: report["cleanup_errors"].append(f"{name}: {exc}")
        report["elapsed_seconds"]=time.monotonic()-started
        save()
        print(json.dumps({k:report.get(k) for k in ("status","measured_translation_norm_mm","motion_commands_accepted","feeder_packets","feeder_ack_received","feeder_reply","error")},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--arm-only",action="store_true",help="execute the arm once, recording physical feedback; no feeder packet")
    parser.add_argument("--fixed-step-mm",type=float,
                        help=f"operator-selected diagnostic distance in (0, {MAX_DEMO_STEP_MM:g}] mm; otherwise gain 40 and cap 3 mm")
    parser.add_argument("--elite-ip",default="192.168.5.66")
    parser.add_argument("--preview-url",default="http://192.168.5.11:8765")
    parser.add_argument("--feeder-host",default="192.168.5.13")
    parser.add_argument("--feeder-local-host",default="192.168.5.11")
    parser.add_argument("--feeder-adapter",type=Path,default=Path("/home/zsw/PycharmProjects/real_collection/hardware/feeder_device/udp_controller.py"))
    parser.add_argument("--model-python",type=Path,default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint",type=Path,default=TRAINING_ROOT/"elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint",type=Path,default=TRAINING_ROOT/"piper/mixed_head_policy.pt")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--model-load-timeout-s",type=float,default=180)
    args=parser.parse_args()
    if args.fixed_step_mm is not None and (not np.isfinite(args.fixed_step_mm) or not 0<args.fixed_step_mm<=MAX_DEMO_STEP_MM):
        parser.error(f"--fixed-step-mm must be in (0, {MAX_DEMO_STEP_MM:g}] mm")
    run(args)


if __name__=="__main__":
    main()
