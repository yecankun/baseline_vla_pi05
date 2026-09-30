"""One local 10 mm magnet-following step from manual tip observations; no feeder."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from execute_left_model_step_http import capture
from probe_elite_camera_axes import image_shift, move_once
from restore_elite_precision_http import CameraRecording, snapshot
from run_real10_pi05_once import checked_joints, pose_errors, robot_ready


def reviewed_corridor(path, anchor):
    """Derive a finite corridor from the bounded targets already checked by IK.

    The preview records onsite route evidence, not execution permission. The
    caller still chooses --execute under the user's current instruction.
    """
    if path is None:
        return None
    d = json.loads(path.read_text())
    relief = d.get("status") == "one_10mm_upward_wall_relief_ik_available_not_executed"
    guided = d.get("status") == "two_10mm_guided_left_segments_ik_available_not_executed"
    lateral = guided or d.get("status") == "bounded_lateral_alignment_ik_available_not_executed"
    count = 1 if relief else 2 if lateral else 4
    if (d.get("status") not in ("four_10mm_segments_ik_available_not_executed",
                               "bounded_lateral_alignment_ik_available_not_executed",
                               "two_10mm_guided_left_segments_ik_available_not_executed",
                               "one_10mm_upward_wall_relief_ik_available_not_executed")
            or d.get("motion_commands") != 0 or d.get("feeder_packets") != 0
            or not d.get("onsite_confirmation", {}).get("answer")
            or len(d.get("candidate_segments", [])) != count):
        raise ValueError("Requires a reviewed bounded corridor preview")
    for field in ("pose", "joints"):
        if not np.allclose(d["original_anchor"][field], anchor[field], atol=1e-8, rtol=0):
            raise ValueError("Corridor must retain the original work-pose anchor")
    poses = np.asarray([d["current"]["pose"]] + [s["pose_mm_rad"] for s in d["candidate_segments"]], float)
    joints = np.asarray([d["current"]["joints"]] + [s["joints_deg"] for s in d["candidate_segments"]], float)
    if (poses.shape != (count+1, 6) or joints.shape != (count+1, 6)
            or not np.isfinite(poses).all() or not np.isfinite(joints).all()
            or not np.allclose(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1), 10, atol=.01)
            or np.max(np.abs(np.diff(joints, axis=0))) > 3
            or not np.allclose(poses[:, 3:], poses[0, 3:], atol=1e-8, rtol=0)):
        raise ValueError("Invalid corridor geometry or discontinuous IK")
    vector = poses[-1, :3] - poses[0, :3]
    if not np.isclose(np.linalg.norm(vector), count*10, atol=.01):
        raise ValueError("Expected the reviewed straight approach")
    if lateral and not np.allclose(np.diff(poses[:, :3], axis=0), [0, -10, 0], atol=.01):
        raise ValueError("Lateral alignment must retain height and X")
    if relief and (not np.allclose(np.diff(poses[:, :3], axis=0), [0, 0, 10], atol=.01)
                   or d.get("contact_observation", {}).get("upper_wall_contact") is not True
                   or d.get("upward_clearance_review", {}).get("reviewed") is not True):
        raise ValueError("Wall relief requires observed contact, reviewed clearance and exactly +Z 10 mm")
    if guided:
        route_start = np.asarray(d["onsite_route_start"]["pose"][:3], float)
        if (d["onsite_route_length_mm"] != 80 or not np.isfinite(route_start).all()
                or np.max(np.linalg.norm(poses[:, :3]-route_start, axis=1)) > 80):
            raise ValueError("Guided extension exceeds the already confirmed whole route")
    tcp_limit = math.ceil(float(np.max(np.linalg.norm(poses[:, :3]-anchor["pose"][:3], axis=1)))+2)
    joint_limit = math.ceil(float(np.max(np.abs(joints-anchor["joints"])))+.5)
    if tcp_limit > (112 if guided or relief else 92 if lateral else 75) or joint_limit > (15 if guided or relief else 12 if lateral else 10):
        raise ValueError("Preview exceeds this bounded approach implementation")
    return {"source": str(path.resolve()), "start": poses[0, :3].tolist(),
            "end": poses[-1, :3].tolist(), "radius_mm": 3,
            "tcp_from_original_mm": tcp_limit, "joint_deg": joint_limit}


def plan_lateral_alignment(jacobian, rois, annotation):
    """One horizontal centering trial; ROI center is only an approximate magnet center."""
    j = np.asarray(jacobian, float)
    if j.shape != (4, 3) or not np.isfinite(j).all():
        raise ValueError("Invalid local camera response")
    gap = np.asarray([rois[v][0]+rois[v][2]/2-annotation["views"][v]["tip"]["xy"][0]
                      for v in ("side", "top")], float)
    delta = np.array([0., -10., 0.])
    predicted = j@delta
    remaining = gap+predicted[[0, 2]]
    if (not np.isfinite(gap).all() or gap[0] <= 0 or gap[1] >= 0
            or not np.all(np.abs(remaining) < np.abs(gap)-2)):
        raise ValueError("Another 10 mm lateral step would not improve both view centers")
    return {"delta_xyz_mm": delta.tolist(), "expected_tool_shift_px": predicted.tolist(),
            "horizontal_center_minus_tip_px": gap.tolist(),
            "predicted_horizontal_center_minus_tip_px": remaining.tolist(),
            "approximate_centroid_alignment_only": True, "full_3d_registration": False,
            "surface_gap_independently_measured": False,
            "action_source": "manual_tip_horizontal_alignment_trial_not_policy"}


def plan_guided_left(jacobian, rois, annotation, review):
    if (review.get("wire_magnetic_guidance_confirmed") is not True
            or review.get("arm_physical_motion_confirmed") is not True):
        raise ValueError("Actual wire guidance must be confirmed before target following")
    j = np.asarray(jacobian, float)
    if j.shape != (4, 3) or not np.isfinite(j).all() or j[0, 1] <= .1 or j[2, 1] >= -.1:
        raise ValueError("Invalid left-axis camera response")
    if any(annotation["views"][v][p]["status"] != "visible"
           for v in ("side", "top") for p in ("tip", "target")):
        raise ValueError("Both current tips and targets must be visible")
    centers = np.array([rois[v][0]+rois[v][2]/2 for v in ("side", "top")])
    tips, targets = [np.array([annotation["views"][v][point]["xy"][0]
                              for v in ("side", "top")]) for point in ("tip", "target")]
    predicted = j@np.array([0., -10., 0.])
    remaining = targets-centers-predicted[[0, 2]]
    lead = (centers+predicted[[0, 2]]-tips)/j[[0, 2], 1]
    if (not np.isfinite(np.concatenate([tips, targets, centers])).all()
            or not (targets[0] < tips[0]-5 and targets[1] > tips[1]+5)
            or not np.all(np.abs(remaining) < np.abs(targets-centers)-2)
            or np.max(np.abs(lead)) > 12.5):
        raise ValueError("Target step would overshoot or separate magnet too far from observed tip")
    return {"delta_xyz_mm": [0., -10., 0.], "expected_tool_shift_px": predicted.tolist(),
            "horizontal_target_minus_center_px": (targets-centers).tolist(),
            "predicted_lateral_magnet_minus_tip_mm_by_view": lead.tolist(),
            "full_3d_registration": False, "surface_gap_independently_measured": False,
            "action_source": "confirmed_magnetic_response_and_manual_target_not_policy"}


def global_bound(anchor, current, corridor=None):
    p, q = np.asarray(current["pose"], float), np.asarray(current["joints"], float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Invalid feedback")
    angle = (Rotation.from_euler("xyz", p[3:]) * Rotation.from_euler("xyz", anchor["pose"][3:]).inv()).magnitude()
    tcp_limit = 31 if corridor is None else corridor["tcp_from_original_mm"]
    joint_limit = 5 if corridor is None else corridor["joint_deg"]
    if (np.linalg.norm(p[:3]-np.asarray(anchor["pose"][:3])) > tcp_limit
            or p[2] < anchor["pose"][2]-.1 or angle > np.deg2rad(.5)
            or np.max(np.abs(q-anchor["joints"])) > joint_limit):
        raise RuntimeError("Exceeded original work-pose envelope; do not reset the anchor")
    if corridor is not None:
        start, end = np.asarray(corridor["start"]), np.asarray(corridor["end"])
        direction = end-start
        fraction = np.clip(np.dot(p[:3]-start, direction)/np.dot(direction, direction), 0, 1)
        if np.linalg.norm(p[:3]-(start+fraction*direction)) > corridor["radius_mm"]:
            raise RuntimeError("Outside the reviewed finite approach corridor")


def plan_step(jacobian, tip_delta, tool_delta):
    j = np.asarray(jacobian, float)
    gap = np.asarray(tip_delta, float)-np.asarray(tool_delta, float)
    if j.shape != (4, 3) or gap.shape != (4,) or not np.isfinite(j).all() or not np.isfinite(gap).all():
        raise ValueError("Invalid image observations")
    # X is poorly observable with these two views. Restrict the incremental
    # experiment to the well-observed lateral/vertical response, retain the
    # residual, and never claim a full 3D registration or exact surface gap.
    yz, _, _, _ = np.linalg.lstsq(j[:, 1:], gap, rcond=None)
    if np.linalg.norm(yz) < 5 or yz[0] >= 0 or yz[1] < 0:
        raise ValueError("No eligible left/up 10 mm following step")
    delta = np.array([0., *yz])*10/np.linalg.norm(yz)
    predicted = j@delta
    if delta[2] > 4 or predicted[0] >= -5 or predicted[2] <= 5:
        raise ValueError("Step does not match the bounded left/up direction in both views")
    return {"delta_xyz_mm": delta.tolist(), "expected_tool_shift_px": predicted.tolist(),
            "remaining_image_gap_px": gap.tolist(), "restricted_yz_fit_mm": yz.tolist(),
            "fit_residual_px": (j[:,1:]@yz-gap).tolist(),
            "full_3d_registration": False, "surface_gap_independently_measured": False,
            "action_source": "manual_tip_update_and_local_axis_response_not_policy"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axes-dir",type=Path,required=True)
    p.add_argument("--prior-arm-dir",type=Path,required=True)
    p.add_argument("--annotation-before",type=Path,required=True)
    p.add_argument("--annotation-now",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--corridor-preview",type=Path,
                   help="reviewed bounded IK preview; finite extension with the original anchor preserved")
    p.add_argument("--align-over-tip",action="store_true",
                   help="one constant-height -Y 10 mm centering trial in a reviewed lateral corridor")
    p.add_argument("--guidance-review",type=Path,
                   help="confirmed arm-only wire response; selects one bounded -Y 10 mm target-following step")
    p.add_argument("--relieve-wall-contact",action="store_true",
                   help="one +Z 10 mm withdrawal away from vessel, only with reviewed relief preview")
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    probe=json.loads((args.axes_dir/"report.json").read_text())
    corridor=reviewed_corridor(args.corridor_preview,probe["base"])
    prior=json.loads((args.prior_arm_dir/"report.json").read_text())
    annotations=[json.loads(path.read_text().splitlines()[-1]) for path in (args.annotation_before,args.annotation_now)]
    if probe["status"]!="completed_three_axis_round_trips" or prior["status"]!="completed_arm_only_awaiting_physical_review":
        raise ValueError("Requires completed axis and prior arm observations")
    if any(a["views"][v]["tip"]["status"]!="visible" for a in annotations for v in ("side","top")):
        raise ValueError("Requires both current manual tips")
    tip_delta=np.concatenate([np.asarray(annotations[1]["views"][v]["tip"]["xy"])-annotations[0]["views"][v]["tip"]["xy"] for v in ("side","top")])
    tool_delta=np.asarray(prior.get("cumulative_tool_shift_px",[x for v in ("side","top") for x in prior["image_shifts"][v]["delta_px"]]),float)
    rois=prior.get("tool_rois_after",{v:[probe["rois"][v][0]+int(tool_delta[2*i]),probe["rois"][v][1]+int(tool_delta[2*i+1]),*probe["rois"][v][2:]] for i,v in enumerate(("side","top"))})
    if args.relieve_wall_contact:
        if args.align_over_tip or args.guidance_review or corridor is None or json.loads(args.corridor_preview.read_text())["status"] != "one_10mm_upward_wall_relief_ik_available_not_executed":
            raise ValueError("Wall relief requires its own reviewed single-step corridor")
        plan={"delta_xyz_mm":[0.,0.,10.],
              "expected_tool_shift_px":(np.asarray(probe["local_image_jacobian_px_per_mm"])@np.array([0.,0.,10.])).tolist(),
              "action_source":"operator_observed_wall_contact_magnet_withdrawal_not_policy",
              "full_3d_registration":False,"surface_gap_independently_measured":False,
              "contact_release_must_be_observed":True}
    elif args.guidance_review:
        if args.align_over_tip or corridor is None or json.loads(args.corridor_preview.read_text())["status"] != "two_10mm_guided_left_segments_ik_available_not_executed":
            raise ValueError("Guided target following requires its reviewed whole-route extension")
        plan=plan_guided_left(probe["local_image_jacobian_px_per_mm"],rois,annotations[1],
                              json.loads(args.guidance_review.read_text()))
        plan["guidance_review_source"]=str(args.guidance_review.resolve())
    elif args.align_over_tip:
        if corridor is None or json.loads(args.corridor_preview.read_text())["status"] != "bounded_lateral_alignment_ik_available_not_executed":
            raise ValueError("Horizontal alignment requires its reviewed lateral corridor")
        plan=plan_lateral_alignment(probe["local_image_jacobian_px_per_mm"],rois,annotations[1])
    else:
        plan=plan_step(probe["local_image_jacobian_px_per_mm"],tip_delta,tool_delta)
    args.out.mkdir(parents=True,exist_ok=False)
    report={"status":"starting","started_at":datetime.now().astimezone().isoformat(),"plan":plan,
            "elite_ip":"192.168.5.66","anchor_axes":str(args.axes_dir.resolve()),
            "prior_arm":str(args.prior_arm_dir.resolve()),"annotation_sources":[str(x.resolve()) for x in (args.annotation_before,args.annotation_now)],
            "annotation_revision":annotations[1]["revision"],"motion_commands_accepted":0,"feeder_packets":0,
            "global_bounds":{"tcp_from_original_mm":31 if corridor is None else corridor["tcp_from_original_mm"],
                             "rotation_deg":.5,"joint_deg":5 if corridor is None else corridor["joint_deg"],
                             "no_lowering":True,"corridor":corridor},
            "visual_status":"not_viewed","tool_rois_before":rois,"automatic_retries":False}
    def save():
        (args.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    ec=cameras=None
    response_seconds=5 if args.relieve_wall_contact else 2
    save()
    try:
        from elite import EC
        ec=EC(ip="192.168.5.66",auto_connect=True);ec.sock_cmd.settimeout(3);robot_ready(ec)
        base=snapshot(ec);global_bound(probe["base"],base,corridor)
        if any(x>limit for x,limit in zip(pose_errors(base["pose"],prior["tcp_after_mm_rad"]),(.02,.001))):
            raise RuntimeError("Pose changed since the prior completed arm stage")
        target=list(base["pose"]);target[:3]=(np.asarray(target[:3])+plan["delta_xyz_mm"]).tolist()
        joints=checked_joints(ec,target,max_joint_step_deg=3)
        global_bound(probe["base"],{"pose":target,"joints":joints},corridor)
        report.update(base=base,target_pose_mm_rad=target,ik_target_joints=joints)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def photos(stage):
            ims,ts=capture(opener,"http://192.168.5.11:8765",args.out,stage)
            if any(not 0<=time.time()-s<=2 for s in ts.values()):raise RuntimeError("Stale camera")
            return ims
        before=photos("before")
        for v in ("side","top"):
            reference=cv2.imread(str(args.prior_arm_dir/f"{v}_after.jpg"))
            match=image_shift(reference,before[v],rois[v])
            if match["ncc"]<.9 or np.linalg.norm(match["delta_px"])>2:raise RuntimeError("Tool/camera view changed")
        report["status"]="preview_only";save()
        if not args.execute:return
        cameras=CameraRecording("http://192.168.5.11:8765",args.out,save_original_frames=True);cameras.start()
        def monitor():
            cameras.check();global_bound(probe["base"],snapshot(ec),corridor)
        event={};report.update(status="moving_one_marked_tip_follow_step",event=event);save()
        move_once(ec,base,target,joints,event,save,monitor)
        report["motion_commands_accepted"]=1
        deadline=time.monotonic()+response_seconds
        while time.monotonic()<deadline:
            monitor();time.sleep(.1)
        after=photos("after");report["after"]=snapshot(ec);global_bound(probe["base"],report["after"],corridor)
        shifts={v:image_shift(before[v],after[v],rois[v]) for v in ("side","top")}
        report.update(image_shifts=shifts,tcp_after_mm_rad=report["after"]["pose"],
                      measured_translation_norm_mm=pose_errors(base["pose"],report["after"]["pose"])[0],
                      cumulative_tool_shift_px=(tool_delta+np.concatenate([shifts[v]["delta_px"] for v in ("side","top")])).tolist(),
                      tool_rois_after={v:[rois[v][0]+shifts[v]["delta_px"][0],rois[v][1]+shifts[v]["delta_px"][1],*rois[v][2:]] for v in ("side","top")},
                      status="completed_arm_only_awaiting_physical_review")
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        if cameras:cameras.close();report.update(camera_frames=cameras.frames,camera_error=cameras.error,
                                                original_camera_frames_saved=True,response_observation_seconds=response_seconds)
        if ec:ec.disconnect_ETController()
        save();print(json.dumps({k:report.get(k) for k in ("status","plan","image_shifts","measured_translation_norm_mm","error")},ensure_ascii=False))


if __name__=="__main__":main()
