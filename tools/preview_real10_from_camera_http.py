"""Preview Real10 policy actions using the running camera page; never actuate."""
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
from run_real10_pi05_once import action_plan, checked_joints, robot_ready


def render(report, images, out):
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 18)
    canvas = Image.new("RGB", (1440, 920), "#142232")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "实机模型预测预览｜尚未执行机械臂或递丝动作", font=font, fill="#9bd5ff")
    for index, role in enumerate(("side", "top")):
        picture = Image.fromarray(cv2.cvtColor(images[role], cv2.COLOR_BGR2RGB))
        picture.thumbnail((680, 383))
        x = 24 + 716 * index
        draw.text((x, 58), f"{role} · {report['cameras'][role]['serial']}", font=small, fill="white")
        canvas.paste(picture, (x, 92))
    y = 493
    for row in report["predictions"]:
        direction = "左分支" if row["task"] == "left" else "右分支"
        raw = row["prediction"]["elite_tcp_delta_6d"][:3]
        draw.text((24, y), f"{direction}：原始 ΔXYZ = [{raw[0]:+.4f}, {raw[1]:+.4f}, {raw[2]:+.4f}] mm", font=font, fill="white")
        plan = row.get("plan")
        if plan:
            delta = plan["guarded_elite_tcp_delta_6d"][:3]
            feed = {0: "保持（0 包）", 1: "前进一步（1 包）"}[plan["piper_step_command"]]
            text = f"限幅后 [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] mm；递丝：{feed}；IK：{row['ik_status']}"
        else:
            text = "候选动作不可执行：" + row["plan_error"]
        draw.text((24, y+42), text, font=small, fill="#c5e5fd")
        y += 115
    draw.text((24, 752), "控制条件：平移最多 1 mm；保持当前姿态；建议关节速度 1%；未归位、未运动、未递丝。", font=small, fill="white")
    draw.text((24, 793), "递丝历史缺失，按现有编码置无效；相机图像来自 JPEG 预览，存在压缩与视角差异。", font=small, fill="#ffc781")
    draw.text((24, 834), "此图为一次预测记录。确认执行后仍需新观测与状态检查，不能直接执行本图中的过期目标。", font=small, fill="#ffc781")
    canvas.save(out / "prediction_preview.png")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "real10_http_prediction_preview_v1", "status": "loading_model",
              "started_at": datetime.now().astimezone().isoformat(),
              "source": "live_camera_http_jpeg", "preview_url": args.preview_url,
              "motion_commands": 0, "feeder_packets": 0, "hardware_executed": False,
              "previous_controller_state": None, "controller_history_valid": False,
              "translation_limit_mm": 1.0, "proposed_joint_speed_percent": 1.0,
              "homing": False, "visual_status": "not_viewed", "predictions": [],
              "cleanup_errors": []}
    save = lambda: dump_json(args.out / "report.json", report)
    save()
    worker, ec = None, None
    begin = time.monotonic()
    try:
        worker = ModelProcess(args, args.out / "worker.log")
        report["model"] = worker.ready
        report["model_load_seconds"] = time.monotonic() - begin
        print(json.dumps({"event": "model_ready", "seconds": report["model_load_seconds"]}), flush=True)
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["robot_before"] = robot_ready(ec)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = args.preview_url.rstrip("/")
        with opener.open(base + "/status.json", timeout=5) as response:
            report["cameras"] = json.load(response)
        images, stamps = {}, {}
        for role in ("side", "top"):
            with opener.open(base + "/" + role + ".jpg", timeout=5) as response:
                encoded = response.read()
                stamps[role] = float(response.headers["X-Capture-Timestamp"])
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (1080, 1920, 3):
                raise RuntimeError(f"invalid {role} image")
            images[role] = image
            (args.out / f"{role}.jpg").write_bytes(encoded)
        pose = [float(value) for value in ec.current_pose]
        captured = time.time()
        ages = {role: captured - stamp for role, stamp in stamps.items()}
        if any(not 0 <= age <= 2 for age in ages.values()):
            raise RuntimeError(f"stale observation: {ages}")
        if abs(stamps["side"]-stamps["top"]) > .1 or max(ages.values()) > .1:
            raise RuntimeError("camera/pose host timestamp difference exceeds 100 ms")
        report["observation"] = {"tcp_pose_mm_rad": pose, "pose_read_finished_timestamp": captured,
                                 "camera_host_timestamps": stamps, "image_age_s": ages,
                                 "same_observation_for_both_tasks": True}
        save()
        for sequence, task in enumerate(("left", "right")):
            started = time.monotonic()
            reply = worker.predict(sequence, images, pose, task, None)
            row = {"task": task, "prediction": reply["prediction"], "state_32": reply["state_32"],
                   "inference_seconds": time.monotonic()-started, "ik_status": "未检查"}
            try:
                row["plan"] = action_plan(reply["prediction"], pose, 1.0)
                row["ik_joints_deg"] = checked_joints(ec, row["plan"]["elite_target_tcp_pose_6d"], max_joint_step_deg=1.0)
                row["ik_status"] = "通过（未执行）"
            except (ValueError, RuntimeError) as exc:
                row["plan_error"] = str(exc)
                row["ik_status"] = "未通过"
            report["predictions"].append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["robot_after"] = robot_ready(ec)
        report["tcp_pose_after_mm_rad"] = list(ec.current_pose)
        report["tcp_translation_change_mm"] = float(np.linalg.norm(np.asarray(report["tcp_pose_after_mm_rad"][:3])-pose[:3]))
        report["status"] = "completed_predictions_no_execution"
        render(report, images, args.out)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for name, close in (("elite", ec.disconnect_ETController if ec else None),
                            ("model", worker.close if worker else None)):
            if close:
                try:
                    close()
                except Exception as exc:
                    report["cleanup_errors"].append(f"{name}: {exc}")
        report["elapsed_seconds"] = time.monotonic() - begin
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elite-ip", default="192.168.5.66")
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--model-python", type=Path, default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint", type=Path, default=TRAINING_ROOT / "elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint", type=Path, default=TRAINING_ROOT / "piper/mixed_head_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-load-timeout-s", type=float, default=180)
    parser.add_argument("--startup-trace", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
