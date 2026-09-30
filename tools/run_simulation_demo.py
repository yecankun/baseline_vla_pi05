"""Render the existing tip-guided environment and scripted expert, without hardware."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# EGL also works with Mesa software rendering when no NVIDIA device is exposed.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulation.mujoco_guided_wire_env import MuJoCoRoutePlanGuideExpert
from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_frame(env, task, font):
    overview = env.render_camera("overview")
    detail = env.render_camera("top", hide_robot_visuals=True)
    canvas = Image.new("RGB", (1280, 552), "#142232")
    canvas.paste(Image.fromarray(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB)), (0, 44))
    canvas.paste(Image.fromarray(cv2.cvtColor(detail, cv2.COLOR_BGR2RGB)), (640, 44))
    draw = ImageDraw.Draw(canvas)
    branch = "左" if task == "left" else "右"
    draw.text((16, 8), "机器人全景", font=font, fill="white")
    draw.text((656, 8), f"导丝特写 · 目标：{branch}分支 · 第 {env.step_count} 步", font=font, fill="white")
    draw.text((16, 521), "规则专家驱动的诊断仿真｜红色为导丝头｜非实机标定场景", font=font, fill="white")
    return np.asarray(canvas)


def run(args):
    os.chdir(ROOT)
    out = args.out.resolve()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save the demo video")
    font = ImageFont.truetype(str(args.font), 20)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "starting", "hardware_executed": False, "formal_data": False,
              "policy": "MuJoCoRoutePlanGuideExpert (scripted; no PI05 weights)",
              "scene_calibration": "diagnostic_only", "visual_status": "not_viewed",
              "tasks": [], "render_every": args.render_every, "video_fps": args.fps,
              "python": sys.executable, "render_backend": os.environ.get("MUJOCO_GL")}
    save_json(out / "report.json", report)
    cameras = {
        "overview": {"distance_scale": 3.2, "azimuth": 145, "elevation": -32,
                     "lookat_offset": [0, 0, 0.12]},
        "top": {"distance_scale": 1.4, "azimuth": 180, "elevation": -65},
    }
    save_json(out / "camera_config.json", cameras)
    config = TipGuidedWireConfig(
        scene_config_path=str(args.scene.resolve()), camera_config_path=str(out / "camera_config.json"),
        vessel_mesh_asset=str(out / "vessel.obj"), table_texture_asset=str(out / "table.png"),
        max_steps=args.max_steps, render_width=640, render_height=480, render_observation=False,
        guidance_mode="physical", robot_visual_mode="kinematic", wire_segments=160,
        wire_visual_mode="line", wire_visual_radius=0.0008, wire_visual_rgb="0.02 0.02 0.018",
        wire_tip_visual_rgb="0.78 0.04 0.02", wire_tip_visual_segments=8,
        wire_tip_visual_radius_scale=2.2, wire_tip_marker_radius=0.002,
        wire_tip_marker_alpha=0.95, show_path_tubes=False, show_tool_markers=False,
        piper_advection_scale=0.11,
    )
    save_json(out / "env_config.json", asdict(config))
    shutil.copyfile(args.scene, out / "scene_config.json")
    env = None
    encoder = None
    frames = 0
    snapshots = []
    try:
        print("Loading vessel and robot geometry...", flush=True)
        env = TipGuidedWireEnv(config, seed=args.seed)
        print(f"Scene loaded in {time.perf_counter() - started:.1f}s", flush=True)
        with (out / "ffmpeg.log").open("w") as encoder_log, (out / "states.jsonl").open("w") as log:
            encoder = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "1280x552", "-r", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out / "demo.mp4"),
            ], stdin=subprocess.PIPE, stderr=encoder_log)
            for task in args.tasks:
                _, info = env.reset(seed=args.seed, options={"task": task, "start_fraction": args.start_fraction})
                expert = MuJoCoRoutePlanGuideExpert(plan_step=0.24, elite_ahead=0.010, piper_cmd=0.70,
                                                  piper_command_period=40, piper_command_width=20,
                                                  route_plan_command_phase_lock=True)
                initial = info["obs_dict"]
                terminal = False
                task_frames = 0
                while True:
                    obs = info["obs_dict"]
                    row = {k: obs[k] for k in ["step", "task", "tip_pos", "path_progress",
                                               "distance_to_target", "success", "failure_reason"]}
                    row["robot_state"] = obs["robot_state"]
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if env.step_count % args.render_every == 0 or terminal:
                        frame = render_frame(env, task, font)
                        encoder.stdin.write(frame.tobytes())
                        frames += 1
                        task_frames += 1
                        if env.step_count == 0 or terminal:
                            image_path = out / f"{task}_{env.step_count:04d}.png"
                            Image.fromarray(frame).save(image_path)
                            snapshots.append(frame.copy())
                    if terminal:
                        break
                    _, _, terminated, truncated, info = env.step(expert.act(env))
                    terminal = terminated or truncated
                    if env.step_count % 40 == 0:
                        print(f"{task}: step={env.step_count}, progress={env.path_progress_float:.2f}", flush=True)
                report["tasks"].append({
                    "task": task, "steps": env.step_count, "frames": task_frames,
                    "success": bool(obs["success"]), "failure_reason": obs["failure_reason"],
                    "truncated": bool(truncated), "distance_to_target_m": obs["distance_to_target"],
                    "tip_displacement_m": float(np.linalg.norm(np.asarray(obs["tip_pos"]) - initial["tip_pos"])),
                    "elite_joint_max_change_rad": max(abs(obs["robot_state"]["elite_joints"][k] - v)
                                                       for k, v in initial["robot_state"]["elite_joints"].items()),
                })
                save_json(out / "report.json", report)
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise RuntimeError("video encoding failed; see ffmpeg.log")
        preview = Image.fromarray(np.concatenate(snapshots, axis=0))
        preview.save(out / "preview.png")
        (out / "index.html").write_text(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>导丝仿真演示</title>'
            '<style>body{background:#142232;color:white;font:18px sans-serif;max-width:1280px;margin:32px auto}'
            'video,img{width:100%}a{color:#9bd5ff}</style><h1>机器人与导丝仿真</h1>'
            '<p>规则专家驱动；诊断场景，尚未经实机标定。左侧机器人全景，右侧导丝特写。</p>'
            '<video controls autoplay muted loop src="demo.mp4"></video>'
            '<p><a href="report.json">运行记录</a> · <a href="demo.mp4">下载视频</a></p>'
            '<img src="preview.png" alt="各分支起点与终点"></html>', encoding="utf-8")
        report.update(status="completed", frames=frames, video_seconds=frames / args.fps)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait(timeout=60)
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.perf_counter() - started
        save_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output" / ("simulation_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--scene", type=Path, default=ROOT / "simulation/scene_configs/mujoco_scene_demo_v1.json")
    parser.add_argument("--tasks", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--start-fraction", type=float, default=0.48)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    if min(args.max_steps, args.render_every, args.fps) <= 0 or not 0 <= args.start_fraction <= 1:
        parser.error("steps, render-every and fps must be positive; start-fraction must be in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
