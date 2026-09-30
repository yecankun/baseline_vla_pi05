"""One measured -Y 10 mm left approach; no feed, default read-only preview.

Uses the immediately preceding local axis probe and manually selected target.
This is an operator-directed incremental experiment, not a PI05 prediction or
registered full-vessel trajectory. Each probe permits one execution directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from execute_left_model_step_http import capture
from probe_elite_camera_axes import image_shift, move_once
from restore_elite_precision_http import CameraRecording, snapshot
from run_real10_pi05_once import checked_joints, pose_errors, robot_ready


def local_plan(probe, annotation, current):
    if probe.get("status") != "completed_three_axis_round_trips":
        raise ValueError("A completed local axis probe is required")
    base = probe["after"]
    distance, angle = pose_errors(current["pose"], base["pose"])
    if distance > .02 or angle > .001:
        raise ValueError("Robot moved since the local axis measurement")
    outbound = [e for e in probe["events"] if e["axis"] == "Y" and e["direction"] == "outbound"]
    if len(outbound) != 1 or outbound[0].get("arrived") is not True:
        raise ValueError("Missing measured Y response")
    shifts = outbound[0]["image_shifts"]
    expected, horizontal = {}, []
    for role in ("side", "top"):
        points = annotation["views"][role]
        if any(points[n]["status"] != "visible" for n in ("tip", "target")):
            raise ValueError("Both manual tip and target points must be visible")
        tip, target = [np.asarray(points[n]["xy"], float) for n in ("tip", "target")]
        response = -np.asarray(shifts[role]["delta_px"], float)
        if (not np.isfinite(response).all() or shifts[role]["ncc"] < .9
                or abs(target[0]-tip[0]) < 30 or abs(response[0]) < 5
                or response[0]*(target[0]-tip[0]) <= 0):
            raise ValueError("Measured -Y direction does not reliably approach both selected targets")
        expected[role] = response.tolist()
        horizontal.append(float(target[0]-tip[0]))
    observed = np.array([expected[v][0] for v in ("side", "top")])
    similarity = float(np.dot(observed, horizontal)/(np.linalg.norm(observed)*np.linalg.norm(horizontal)))
    if similarity < .98:
        raise ValueError("Two views disagree with the proposed single-axis approach")
    target = list(current["pose"])
    target[1] -= 10
    return {"target_pose_mm_rad": target, "delta_xyz_mm": [0, -10, 0],
            "step_mm": 10, "expected_tool_shift_px": expected,
            "two_view_horizontal_similarity": similarity,
            "action_source": "operator_marked_target_and_measured_local_axis_not_policy",
            "height_and_orientation": "unchanged_from_operator_30mm_start",
            "full_route_registration": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axes-dir", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--elite-ip", default="192.168.5.66")
    p.add_argument("--preview-url", default="http://192.168.5.11:8765")
    args = p.parse_args()
    probe = json.loads((args.axes_dir/"report.json").read_text())
    annotation = json.loads(args.annotations.read_text().splitlines()[-1])
    if not 0 <= time.time()-probe["after"]["timestamp"] <= 1800:
        raise ValueError("Local axis probe is not current")
    # Never automatically repeat an ambiguous command or a completed trial.
    out = args.axes_dir / ("left_step" if args.execute else "left_step_preview")
    out.mkdir(exist_ok=False)
    report = {"status": "starting", "started_at": datetime.now().astimezone().isoformat(),
              "elite_ip": args.elite_ip, "local_axis_evidence": str(args.axes_dir.resolve()),
              "annotation_source": str(args.annotations.resolve()),
              "annotation_revision": annotation["revision"], "motion_commands_accepted": 0,
              "feeder_packets": 0, "automatic_retries": False, "homing": False,
              "arm_physical_execution_confirmed": None, "visual_status": "not_viewed"}
    def save():
        (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    ec = cameras = None
    save()
    try:
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        robot_ready(ec)
        base = snapshot(ec)
        plan = local_plan(probe, annotation, base)
        joints = checked_joints(ec, plan["target_pose_mm_rad"], max_joint_step_deg=3)
        report.update(plan=plan, base=base, ik_target_joints=joints)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def photos(stage):
            if cameras:
                cameras.check()
            frames, stamps = capture(opener, args.preview_url, out, stage)
            if any(not 0 <= time.time()-ts <= 2 for ts in stamps.values()):
                raise RuntimeError("Stale cameras")
            return frames
        before = photos("before")
        for role in ("side", "top"):
            original = cv2.imread(str(args.axes_dir/f"{role}_Y_before.jpg"))
            match = image_shift(original, before[role], probe["rois"][role])
            if match["ncc"] < .9 or np.linalg.norm(match["delta_px"]) > 1:
                raise RuntimeError("Camera/tool view changed since local axis measurement")
        report["status"] = "preview_only"
        save()
        if not args.execute:
            return
        cameras = CameraRecording(args.preview_url, out)
        cameras.start()
        event = {}
        report.update(status="moving_one_10mm_left_step", event=event)
        save()
        move_once(ec, base, plan["target_pose_mm_rad"], joints, event, save, cameras.check)
        report["motion_commands_accepted"] = 1
        time.sleep(.3)
        after = photos("after")
        report["image_shifts"] = {role: image_shift(before[role], after[role], probe["rois"][role])
                                  for role in ("side", "top")}
        report["after"] = snapshot(ec)
        report["tcp_after_mm_rad"] = report["after"]["pose"]
        report["measured_translation_norm_mm"] = pose_errors(base["pose"], report["after"]["pose"])[0]
        report["status"] = "completed_arm_only_awaiting_physical_review"
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
        print(json.dumps({k:report.get(k) for k in ("status", "plan", "image_shifts", "measured_translation_norm_mm", "error")},ensure_ascii=False))


if __name__ == "__main__":
    main()
