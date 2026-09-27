from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import (  # noqa: E402
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoGuidedWireExpert,
)


def build_env(args: argparse.Namespace, seed: int) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        show_vessel_mesh=not args.hide_vessel_mesh,
        show_path_tubes=args.show_path_tubes,
        show_tool_markers=args.show_tool_markers,
        render_overlay=args.show_overlay,
        render_observation=False,
        render_width=args.width,
        render_height=args.height,
    )
    return MuJoCoGuidedWireEnv(config, seed=seed)


def segment_min_distance_to_wall(env: MuJoCoGuidedWireEnv) -> float:
    return env.segment_min_distance_to_wall()


def render_task(args: argparse.Namespace, task: str) -> dict:
    env = build_env(args, args.seed)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})

    out_dir = Path(args.out) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for camera in args.cameras:
        path = out_dir / f"{camera}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        writers[camera] = writer

    min_tip_wall = float("inf")
    min_segment_wall = float("inf")
    max_contact = 0.0
    total_reward = 0.0

    try:
        while True:
            obs_dict = info["obs_dict"]
            min_tip_wall = min(min_tip_wall, float(obs_dict["distance_to_wall"]))
            min_segment_wall = min(min_segment_wall, segment_min_distance_to_wall(env))
            max_contact = max(max_contact, float(obs_dict["contact_strength"]))

            if obs_dict["step"] % max(args.render_every, 1) == 0:
                for camera, writer in writers.items():
                    writer.write(env.render_camera(camera))

            if obs_dict["success"] or obs_dict["failure_reason"] is not None or obs_dict["step"] >= args.max_steps:
                break

            action = expert.act(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                obs_dict = info["obs_dict"]
                min_tip_wall = min(min_tip_wall, float(obs_dict["distance_to_wall"]))
                min_segment_wall = min(min_segment_wall, segment_min_distance_to_wall(env))
                max_contact = max(max_contact, float(obs_dict["contact_strength"]))
                for _ in range(max(args.fps // 2, 1)):
                    for camera, writer in writers.items():
                        writer.write(env.render_camera(camera))
                break
    finally:
        for writer in writers.values():
            writer.release()
        env.close()

    final_obs = info["obs_dict"]
    meta = {
        "task": task,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "min_tip_distance_to_wall": min_tip_wall,
        "min_segment_distance_to_wall": min_segment_wall,
        "max_contact_strength": max_contact,
        "total_reward": total_reward,
        "videos": {camera: str((out_dir / f"{camera}.mp4").resolve()) for camera in args.cameras},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{task}: success={meta['success']} steps={meta['steps']} "
        f"min_seg={meta['min_segment_distance_to_wall']:.6f} max_contact={meta['max_contact_strength']:.3f}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MuJoCo guided-wire expert rollout from multiple cameras.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_clearance_view_check")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--cameras", nargs="+", default=["overview", "top", "side"], choices=["overview", "top", "side", "perspective"])
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--show-path-tubes", action="store_true")
    parser.add_argument("--show-tool-markers", action="store_true")
    parser.add_argument("--show-overlay", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    args = parser.parse_args()

    summaries = [render_task(args, task) for task in args.tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved videos to {out.resolve()}")


if __name__ == "__main__":
    main()
