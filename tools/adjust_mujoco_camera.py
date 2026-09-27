from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import (  # noqa: E402
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoGuidedWireExpert,
)


WINDOW = "MuJoCo camera adjust"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_env(args: argparse.Namespace) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        wire_visual_mode=args.wire_visual_mode,
        wire_visual_radius=args.wire_visual_radius,
        wire_visual_rgb=args.wire_visual_rgb,
        wire_tip_visual_rgb=args.wire_tip_visual_rgb,
        wire_tip_visual_segments=args.wire_tip_visual_segments,
        wire_tip_visual_radius_scale=args.wire_tip_visual_radius_scale,
        wire_tip_marker_radius=args.wire_tip_marker_radius,
        wire_tip_marker_alpha=args.wire_tip_marker_alpha,
        wire_visual_offset=tuple(float(x) for x in args.wire_visual_offset),
        wire_line_width=4,
        wire_line_core_width=1,
        show_vessel_mesh=not args.hide_vessel_mesh,
        show_path_tubes=not args.hide_path_tubes,
        show_tool_markers=not args.hide_tool_markers,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    return MuJoCoGuidedWireEnv(config, seed=args.seed)


def advance_preview(env: MuJoCoGuidedWireEnv, args: argparse.Namespace) -> None:
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=0.0)
    _obs, info = env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
    for _ in range(max(args.preview_steps, 0)):
        action = expert.act(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated or info["obs_dict"]["success"] or info["obs_dict"]["failure_reason"] is not None:
            break


def default_preset(camera: str) -> dict:
    if camera == "top":
        return {"distance_scale": 0.98, "azimuth": 145.0, "elevation": -62.0, "lookat_offset": [0.0, 0.0, 0.0]}
    if camera == "side":
        return {"distance_scale": 0.72, "azimuth": 180.0, "elevation": -8.0, "lookat_offset": [0.0, 0.0, 0.0]}
    return {"distance_scale": 0.66, "azimuth": 145.0, "elevation": -24.0, "lookat_offset": [0.0, 0.0, 0.0]}


def clamp_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def setup_trackbars(preset: dict) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 860, 720)
    cv2.createTrackbar("azimuth", WINDOW, clamp_int(preset["azimuth"], 0, 360), 360, lambda _value: None)
    cv2.createTrackbar("elevation", WINDOW, clamp_int(preset["elevation"] + 89, 0, 89), 89, lambda _value: None)
    cv2.createTrackbar("distance_%", WINDOW, clamp_int(preset["distance_scale"] * 100.0, 35, 220), 220, lambda _value: None)
    cv2.createTrackbar("lookat_x_mm", WINDOW, clamp_int(preset["lookat_offset"][0] * 1000.0 + 250, 0, 500), 500, lambda _value: None)
    cv2.createTrackbar("lookat_y_mm", WINDOW, clamp_int(preset["lookat_offset"][1] * 1000.0 + 250, 0, 500), 500, lambda _value: None)
    cv2.createTrackbar("lookat_z_mm", WINDOW, clamp_int(preset["lookat_offset"][2] * 1000.0 + 250, 0, 500), 500, lambda _value: None)


def read_trackbars() -> dict:
    distance_percent = max(cv2.getTrackbarPos("distance_%", WINDOW), 35)
    return {
        "distance_scale": float(distance_percent) / 100.0,
        "azimuth": float(cv2.getTrackbarPos("azimuth", WINDOW)),
        "elevation": float(cv2.getTrackbarPos("elevation", WINDOW) - 89),
        "lookat_offset": [
            float(cv2.getTrackbarPos("lookat_x_mm", WINDOW) - 250) / 1000.0,
            float(cv2.getTrackbarPos("lookat_y_mm", WINDOW) - 250) / 1000.0,
            float(cv2.getTrackbarPos("lookat_z_mm", WINDOW) - 250) / 1000.0,
        ],
    }


def draw_overlay(image: np.ndarray, args: argparse.Namespace, preset: dict, saved: bool) -> np.ndarray:
    out = image.copy()
    lines = [
        f"camera={args.camera} task={args.task} | s save | r reset | q quit",
        f"az={preset['azimuth']:.1f} elev={preset['elevation']:.1f} dist={preset['distance_scale']:.2f}",
        "lookat_offset[m]=" + ", ".join(f"{v:.3f}" for v in preset["lookat_offset"]),
    ]
    if saved:
        lines.append(f"saved: {Path(args.camera_config).as_posix()}")
    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 25
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively adjust MuJoCo render camera parameters.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--camera", choices=["top", "side", "perspective"], default="top")
    parser.add_argument("--seed", type=int, default=970)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--preview-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--wire-segments", type=int, default=160)
    parser.add_argument("--wire-visual-mode", choices=["segments", "line", "both"], default="line")
    parser.add_argument("--wire-visual-radius", type=float, default=0.0008)
    parser.add_argument("--wire-visual-rgb", default="0.02 0.02 0.018")
    parser.add_argument("--wire-tip-visual-rgb", default="0.78 0.04 0.02")
    parser.add_argument("--wire-tip-visual-segments", type=int, default=5)
    parser.add_argument("--wire-tip-visual-radius-scale", type=float, default=2.2)
    parser.add_argument("--wire-tip-marker-radius", type=float, default=0.0020)
    parser.add_argument("--wire-tip-marker-alpha", type=float, default=0.95)
    parser.add_argument("--wire-visual-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--image-size", type=int, default=720)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--hide-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--hide-path-tubes", action="store_true")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.camera_config).resolve()
    camera_config = load_json(config_path)
    preset = default_preset(args.camera)
    preset.update(camera_config.get(args.camera, {}))

    env = build_env(args)
    advance_preview(env, args)
    setup_trackbars(preset)
    print("Use sliders to adjust. Press 's' to save, 'r' to reset this camera, 'q' or Esc to quit.")

    saved = False
    last_preset = None
    last_image = None
    try:
        while True:
            preset = read_trackbars()
            if preset != last_preset:
                env.camera_config[args.camera] = preset
                image = env.render_camera(args.camera)
                image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                last_image = draw_overlay(image, args, preset, saved=False)
                last_preset = json.loads(json.dumps(preset))
                saved = False
            if last_image is not None:
                cv2.imshow(WINDOW, last_image)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                preset = default_preset(args.camera)
                cv2.setTrackbarPos("azimuth", WINDOW, clamp_int(preset["azimuth"], 0, 360))
                cv2.setTrackbarPos("elevation", WINDOW, clamp_int(preset["elevation"] + 89, 0, 89))
                cv2.setTrackbarPos("distance_%", WINDOW, clamp_int(preset["distance_scale"] * 100.0, 35, 220))
                cv2.setTrackbarPos("lookat_x_mm", WINDOW, 250)
                cv2.setTrackbarPos("lookat_y_mm", WINDOW, 250)
                cv2.setTrackbarPos("lookat_z_mm", WINDOW, 250)
            if key == ord("s"):
                camera_config[args.camera] = preset
                save_json(config_path, camera_config)
                env.camera_config = camera_config
                if last_image is not None:
                    last_image = draw_overlay(env.render_camera(args.camera), args, preset, saved=True)
                    last_image = cv2.resize(last_image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                print(f"saved {args.camera}: {preset}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
