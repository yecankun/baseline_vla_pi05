"""Three local 10 mm axis round trips; default IK preview, no feeder or homing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from execute_left_model_step_http import capture
from restore_elite_precision_http import CameraRecording, snapshot
from run_real10_pi05_once import checked_joints, pose_errors, robot_ready


def check_sample(base, current):
    if (current["state"] not in (0, 3) or current["mode"] != 2 or current["estop"] != 0
            or current["precise"] != 1 or current["servo"] not in (True, 1)
            or current["sync"] not in (True, 1)):
        raise RuntimeError("Robot lost operational readiness")
    p, q = np.asarray(current["pose"], float), np.asarray(current["joints"], float)
    if p.shape != (6,) or q.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise RuntimeError("Invalid robot feedback")
    distance = float(np.linalg.norm(p[:3]-base["pose"][:3]))
    angle = float((Rotation.from_euler("xyz", p[3:]) *
                   Rotation.from_euler("xyz", base["pose"][3:]).inv()).magnitude())
    if distance > 13 or angle > np.deg2rad(.5) or np.max(np.abs(q-base["joints"])) > 3:
        raise RuntimeError("Axis-probe motion exceeded local pose/joint envelope")
    return {"tcp_distance_mm": distance, "rotation_deg": float(np.rad2deg(angle)),
            "tool_point_bound_mm_at_500mm": distance+1000*np.sin(angle/2)}


def move_once(ec, base, target, joints, event, save, camera_check, *, timeout_s=12):
    robot_ready(ec)
    check_sample(base, snapshot(ec))
    distance, angle = pose_errors(ec.current_pose, target)
    if not 9.95 <= distance <= 10.05 or angle > .001:
        raise RuntimeError("This probe permits only the planned 10 mm translations")
    camera_check()
    event.update(attempted=True, dispatch_timestamp=time.time(), samples=[])
    save()
    complete = False
    try:
        event["reply"] = ec.move_joint(target_joint=joints, speed=1., block=False)
        save()
        if event["reply"] is not True:
            raise RuntimeError("Motion was not acknowledged; no retry")
        started = time.monotonic()
        while time.monotonic()-started < timeout_s:
            camera_check()
            current = snapshot(ec)
            event["samples"].append(current)
            save()
            check_sample(base, current)
            error_mm, error_rad = pose_errors(current["pose"], target)
            if (current["state"] == 0 and time.monotonic()-started >= .3
                    and error_mm <= .02 and error_rad <= .001):
                event.update(arrived=True, after=current, arrival_timestamp=time.time(),
                             target_error_mm=error_mm)
                complete = True
                save()
                return
            time.sleep(.05)
        raise TimeoutError("Axis-probe target not reached")
    finally:
        if not complete:
            try:
                event["stop_reply"] = ec.stop()
            except BaseException as exc:
                event["stop_error"] = str(exc)
            save()


def image_shift(before, after, roi):
    x, y, w, h = roi
    margin = 40
    if x < margin or y < margin or x+w+margin >= 1920 or y+h+margin >= 1080:
        raise ValueError("ROI must have a 40 px search margin inside the original image")
    a, b = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (before, after)]
    response = cv2.matchTemplate(b[y-margin:y+h+margin, x-margin:x+w+margin],
                                 a[y:y+h, x:x+w], cv2.TM_CCOEFF_NORMED)
    _, ncc, _, location = cv2.minMaxLoc(response)
    return {"delta_px": [location[0]-margin, location[1]-margin], "ncc": ncc,
            "method": "integer-pixel local normalized template correspondence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--elite-ip", default="192.168.5.66")
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rois = json.loads(args.rois.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "starting",
              "execute_requested": args.execute, "feeder_packets": 0, "events": [],
              "step_mm": 10, "joint_speed_percent": 1, "rois": rois,
              "bounds": {"tcp_from_start_mm": 13, "rotation_from_start_deg": .5,
                         "joint_from_start_deg": 3, "tool_reach_assumption_mm": 500},
              "not_established": ["Full scene registration", "Magnet TCP offset",
                                  "Tip tracking during feeding", "Full route clearance"],
              "valid_for_full_route_control": False, "visual_status": "not_viewed"}
    def save():
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    ec, cameras = None, None
    save()
    try:
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        base = snapshot(ec)
        if (base["state"] != 0 or base["mode"] != 2 or base["sync"] not in (True, 1)
                or base["precise"] != 1 or base["estop"] != 0):
            raise RuntimeError("IK preview requires synchronized and precise stopped robot")
        report["base"] = base
        plans = []
        for i, axis in enumerate("XYZ"):
            pose = list(base["pose"])
            pose[i] += 10
            joints = checked_joints(ec, pose, max_joint_step_deg=3)
            plans.append({"axis": axis, "target": pose, "joints": joints,
                          "max_joint_change_deg": float(np.max(np.abs(np.asarray(joints)-base["joints"])))})
        report.update(plans=plans, status="IK_preview_only")
        save()
        if not args.execute:
            return
        robot_ready(ec)
        report["status"] = "executing_three_axis_round_trips"
        save()
        cameras = CameraRecording(args.preview_url, args.out)
        cameras.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def photos(stage):
            cameras.check()
            frames, stamps = capture(opener, args.preview_url, args.out, stage)
            if any(not 0 <= time.time()-stamp <= 2 for stamp in stamps.values()):
                raise RuntimeError("Stale camera capture")
            return frames
        photos("initial")
        displacements = []
        for plan in plans:
            if pose_errors(ec.current_pose, base["pose"])[0] > .02:
                raise RuntimeError("Drift from axis-probe starting pose")
            before = photos(plan["axis"]+"_before")
            for outbound in (True, False):
                event = {"axis": plan["axis"], "direction": "outbound" if outbound else "return"}
                report["events"].append(event)
                target = plan["target"] if outbound else base["pose"]
                joints = plan["joints"] if outbound else base["joints"]
                move_once(ec, base, target, joints, event, save, cameras.check)
                time.sleep(.2)
                frames = photos(plan["axis"]+("_outbound" if outbound else "_return"))
                shifts = {v: image_shift(before[v], frames[v], rois[v]) for v in ("side", "top")}
                event["image_shifts"] = shifts
                save()
                if outbound:
                    displacements.append([value for v in ("side", "top") for value in shifts[v]["delta_px"]])
        matrix = np.asarray(displacements, float).T / 10
        report.update(status="completed_three_axis_round_trips", after=snapshot(ec),
                      local_image_jacobian_px_per_mm=matrix.tolist(),
                      singular_values=np.linalg.svd(matrix, compute_uv=False).tolist(),
                      condition_number=float(np.linalg.cond(matrix)))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if cameras:
            cameras.close()
            report.update(camera_frames=cameras.frames, camera_error=cameras.error)
        if ec:
            ec.disconnect_ETController()
        save()
        print(json.dumps({k: report.get(k) for k in ("status", "plans", "local_image_jacobian_px_per_mm", "condition_number", "error")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
