"""Live cameras + PI05 + external reviewer handshake, with no actuation path.

Only reads Elite TCP. No IK, servo, robot motion, UDP or feeder implementation.
Each review references an immutable request; every next cycle captures new frames.
The external reviewer is the actual Codex session, not a simulated policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from real10_pi05_bridge import (
    ROOT, MODEL_PYTHON, TRAINING_ROOT, LatestColorCamera, ModelProcess,
    controller_preview, render_preview,
)

DECISIONS = ("continue_candidate", "pause_review", "observe_again")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def validate_review(value, request_id):
    if value.get("request_id") != request_id:
        raise ValueError("review request_id does not match the current observation")
    if value.get("decision") not in DECISIONS or value.get("next") not in ("observe", "finish"):
        raise ValueError("only known review decisions and observe/finish are allowed")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("review must include its evidence/reason")
    if set(value) - {"request_id", "decision", "next", "reason", "visual_observation", "limitations"}:
        raise ValueError("unexpected review fields; this interface accepts no commands or action overrides")
    return value


def wait_review(path, request_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return validate_review(json.loads(path.read_text(encoding="utf-8")), request_id)
        time.sleep(.1)
    raise TimeoutError("external review did not arrive; readonly session stops")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    worker, ec, cameras = None, None, {}
    started = time.monotonic()
    report = {"schema":"real10_reasoning_readonly_v1", "run_id":uuid.uuid4().hex,
              "source":"live", "status":"starting", "task":args.task,
              "elite_ip":args.elite_ip, "side_serial":args.side_serial, "top_serial":args.top_serial,
              "camera_profile":"1920x1080 BGR8 15fps; host receive times, not exposure-synchronized",
              "external_reviewer":"actual Codex session via request-bound JSON files",
              "execution_enabled":False, "hardware_executed":False, "motion_commands":0,
              "feeder_packets":0, "rounds":[], "cleanup_errors":[], "visual_status":"not_viewed"}
    save = lambda: atomic_json(args.out/"report.json", report)
    save()
    try:
        from elite import EC
        try:
            ec = EC(ip=args.elite_ip, auto_connect=True)
        except SystemExit as exc:
            raise RuntimeError("Elite read connection failed") from exc
        report["initial_tcp_pose"] = [float(x) for x in ec.current_pose]
        report["status"] = "loading_model"
        save()
        print(json.dumps({"event":"loading_model","initial_tcp_pose":report["initial_tcp_pose"],"hardware_executed":False}),flush=True)
        worker = ModelProcess(args, args.out/"worker.log")
        report["model"] = worker.ready
        report["model_ready_elapsed_s"] = time.monotonic()-started
        for name, serial in (("side",args.side_serial),("top",args.top_serial)):
            cameras[name] = LatestColorCamera(serial)
        previous_capture_end = 0.0
        for sequence in range(1,args.rounds+1):
            # Read side once first to allow its exposure warmup, then both again
            # after top warmup so their host receive times are recent together.
            for camera in cameras.values():
                camera.read()
            frames = {name:camera.read() for name,camera in cameras.items()}
            pose_start = time.time()
            pose = [float(x) for x in ec.current_pose]
            capture_end = time.time()
            if len(pose)!=6 or not np.isfinite(pose).all():
                raise RuntimeError("invalid read-only TCP pose")
            ages = {name:capture_end-frame[1] for name,frame in frames.items()}
            if any(not 0 <= age <= 2 for age in ages.values()) or capture_end-pose_start>2:
                raise RuntimeError("live observation is stale before policy inference")
            if any(frame[1] <= previous_capture_end for frame in frames.values()):
                raise RuntimeError("next round did not acquire fresh camera frames")
            previous_capture_end = capture_end
            images = {name:frame[0] for name,frame in frames.items()}
            inference_start = time.perf_counter()
            reply = worker.predict(sequence,images,pose,args.task,None)
            roundtrip = time.perf_counter()-inference_start
            request_id = f'{report["run_id"]}:{sequence}'
            cycle = args.out/f"round_{sequence:02d}"
            cycle.mkdir()
            for name, image in images.items():
                if not cv2.imwrite(str(cycle/f"{name}.png"),image):
                    raise RuntimeError("failed to save full-resolution live image")
            row = {"sequence":sequence,"source":"live", "request_id":request_id,
                   "observation":{"task":args.task,"elite_tcp_pose_6d":pose,
                                  "previous_controller_state":None,"controller_history_valid":False,
                                  "camera_host_timestamps":{name:f[1] for name,f in frames.items()},
                                  "capture_finished_timestamp":capture_end},
                   "policy_output":reply["prediction"],"state_32":reply["state_32"],
                   "legacy_controller_preview":controller_preview(reply["prediction"],pose),
                   "roundtrip_seconds":roundtrip,"hardware_executed":False}
            preview = render_preview(images,row,args.font)
            # Replace only the inherited GUI hint with this entrypoint's actual mode.
            from PIL import Image, ImageDraw, ImageFont
            pil = Image.fromarray(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.rectangle((0,639,1280,720),fill="#111827")
            draw.text((22,641),f"实时只读联调 第{sequence}轮｜等待当前 Codex 会话审核；可请求新观测或结束",font=ImageFont.truetype(str(args.font),18),fill="#e5e7eb")
            draw.text((22,677),"机械臂动作 / 递丝发送路径不存在；审核通过也不会运动。",font=ImageFont.truetype(str(args.font),18),fill="#e5e7eb")
            pil.save(cycle/"preview.png")
            published = time.time()
            row.update(published_timestamp=published,
                       observation_age_at_publication_s={name:published-f[1] for name,f in frames.items()},
                       image_files={"side":"side.png","top":"top.png","preview":"preview.png"},
                       review_response_file="review.json",allowed_next=["observe","finish"],
                       candidate_is_not_executable=True)
            atomic_json(cycle/"request.json",row)
            summary = {"sequence":sequence,"request_id":request_id,"cycle_dir":str(cycle),
                       "policy_roundtrip_s":roundtrip,"status":"waiting_for_review"}
            report["rounds"].append(summary)
            report["status"]="waiting_for_review"
            save()
            print(json.dumps({"event":"review_requested",**summary},ensure_ascii=False),flush=True)
            wait_started=time.monotonic()
            review=wait_review(cycle/"review.json",request_id,args.review_timeout_s)
            received=time.time()
            received_ages={name:received-f[1] for name,f in frames.items()}
            receipt={"request_id":request_id,"received_timestamp":received,"review":review,
                     "review_wait_s":time.monotonic()-wait_started,
                     "observation_age_when_review_received_s":received_ages,
                     "within_existing_2s_observation_limit":all(0<=age<=2 for age in received_ages.values()),
                     "motion_authorized":False,"hardware_executed":False,
                     "next_program_action":"capture_fresh_observation" if review["next"]=="observe" and sequence<args.rounds else "finish_readonly_session"}
            atomic_json(cycle/"receipt.json",receipt)
            summary.update(status="review_received",review_decision=review["decision"],next=review["next"],
                           review_wait_s=receipt["review_wait_s"],within_existing_2s_observation_limit=receipt["within_existing_2s_observation_limit"])
            report["status"]="review_received"
            save()
            print(json.dumps({"event":"review_received",**summary},ensure_ascii=False),flush=True)
            if review["next"]=="finish":
                break
        report["final_tcp_pose"]=[float(x) for x in ec.current_pose]
        report["tcp_translation_readback_difference_mm"]=float(np.linalg.norm(np.asarray(report["final_tcp_pose"][:3])-np.asarray(report["initial_tcp_pose"][:3])))
        report["status"]="completed_readonly"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup=[(f"camera:{name}",camera.close) for name,camera in cameras.items()]
        if ec is not None:
            cleanup.append(("Elite read connection",ec.disconnect_ETController))
        if worker is not None:
            cleanup.append(("model worker",worker.close))
        for name,close in cleanup:
            try:
                close()
            except Exception as exc:
                report["cleanup_errors"].append(f"{name}: {exc}")
        if report["cleanup_errors"]:
            report["status"]="failed_cleanup"
        report["elapsed_seconds"]=time.monotonic()-started
        save()
        print(json.dumps({"event":"session_finished","status":report["status"],"rounds":len(report["rounds"]),"elapsed_seconds":report["elapsed_seconds"],"hardware_executed":False,"cleanup_errors":report["cleanup_errors"]},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--elite-ip",required=True)
    parser.add_argument("--side-serial",required=True)
    parser.add_argument("--top-serial",required=True)
    parser.add_argument("--task",choices=("left","right"),required=True)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--review-timeout-s",type=float,default=600)
    parser.add_argument("--model-load-timeout-s",type=float,default=240)
    parser.add_argument("--startup-trace",action="store_true")
    parser.add_argument("--model-python",type=Path,default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint",type=Path,default=TRAINING_ROOT/"elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint",type=Path,default=TRAINING_ROOT/"piper/mixed_head_policy.pt")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args=parser.parse_args()
    if not 1<=args.rounds<=10 or not 0<args.review_timeout_s<=900 or not 0<args.model_load_timeout_s<=300 or args.side_serial==args.top_serial:
        parser.error("need 1..10 rounds, 0..900 s read-only review timeout, 0..300 s model load timeout and distinct cameras")
    run(args)


if __name__=="__main__":
    main()
