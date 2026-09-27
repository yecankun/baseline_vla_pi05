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
    MuJoCoMagneticGuideExpert,
)


def build_env(args: argparse.Namespace) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        show_path_tubes=args.show_path_tubes,
        show_tool_markers=args.show_tool_markers,
        render_width=args.render_width,
        render_height=args.render_height,
        guidance_mode="physical",
        magnet_force=args.magnet_force,
        magnet_range=args.magnet_range,
        piper_force=args.piper_force,
        piper_advection_scale=args.piper_advection_scale,
        spring_k=args.spring_k,
        bend_k=args.bend_k,
        damping=args.damping,
        sim_substeps=args.sim_substeps,
        wall_k=args.wall_k,
        gravity_z=args.gravity_z,
        wire_max_turn_degrees=args.wire_max_turn_degrees,
        physical_curvature_smoothing=args.physical_curvature_smoothing,
        physical_curvature_iterations=args.physical_curvature_iterations,
        physical_tip_stiff_segments=args.physical_tip_stiff_segments,
        physical_tip_curvature_smoothing=args.physical_tip_curvature_smoothing,
        physical_tip_curvature_iterations=args.physical_tip_curvature_iterations,
        wire_max_speed=args.wire_max_speed,
    )
    return MuJoCoGuidedWireEnv(config, seed=args.seed)


def run_task(args: argparse.Namespace, task: str) -> dict:
    env = build_env(args)
    expert = MuJoCoMagneticGuideExpert(
        rng_seed=args.seed,
        steer_noise=args.steer_noise,
        lookahead_points=args.lookahead_points,
        elite_ahead=args.elite_ahead,
        lateral_gain=args.lateral_gain,
        piper_cmd=args.piper_cmd,
    )
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})
    start_progress = float(info["obs_dict"]["path_progress"])
    target_progress = min(start_progress + args.progress_delta, len(env.paths[task]) - 1)
    out_dir = Path(args.out) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        writer = cv2.VideoWriter(
            str(out_dir / "physical_guidance.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.render_width, args.render_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_dir / 'physical_guidance.mp4'}")

    min_wall = float("inf")
    max_contact = 0.0
    total_reward = 0.0
    states = []
    try:
        while True:
            obs = info["obs_dict"]
            if writer is not None and int(obs["step"]) % max(args.render_every, 1) == 0:
                writer.write(env.render_camera(args.video_camera))
            min_wall = min(min_wall, env.segment_min_distance_to_wall())
            max_contact = max(max_contact, float(obs["contact_strength"]))
            states.append(obs)
            short_success = float(obs["path_progress"]) >= target_progress or bool(obs["success"])
            if short_success or obs["failure_reason"] is not None or int(obs["step"]) >= args.max_steps:
                break
            action = expert.act(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    final_obs = info["obs_dict"]
    reached_progress_target = float(final_obs["path_progress"]) >= target_progress
    env_success = bool(final_obs["success"])
    short_success = reached_progress_target or env_success
    success_reason = "progress_target" if reached_progress_target else ("env_success" if env_success else None)
    meta = {
        "task": task,
        "short_success": bool(short_success),
        "success_reason": success_reason,
        "reached_progress_target": bool(reached_progress_target),
        "env_success": env_success,
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "start_progress": start_progress,
        "target_progress": target_progress,
        "final_progress": float(final_obs["path_progress"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "min_segment_distance_to_wall": float(min_wall),
        "max_contact_strength": float(max_contact),
        "total_reward": float(total_reward),
        "video": str((out_dir / "physical_guidance.mp4").resolve()) if args.video else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as f:
        for row in states:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{task}: short_success={meta['short_success']} env_success={meta['env_success']} "
        f"steps={meta['steps']} progress={meta['final_progress']:.2f}/{meta['target_progress']:.2f} "
        f"min_wall={meta['min_segment_distance_to_wall']:.6f} max_contact={meta['max_contact_strength']:.3f}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Test short-horizon physical magnetic guidance without centerline autopilot.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_physical_guidance_short_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--start-fraction", type=float, default=0.66)
    parser.add_argument("--progress-delta", type=float, default=14.0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--magnet-force", type=float, default=0.42)
    parser.add_argument("--magnet-range", type=float, default=0.16)
    parser.add_argument("--piper-force", type=float, default=0.30)
    parser.add_argument("--piper-advection-scale", type=float, default=0.22)
    parser.add_argument("--spring-k", type=float, default=220.0)
    parser.add_argument("--bend-k", type=float, default=18.0)
    parser.add_argument("--damping", type=float, default=2.2)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--wall-k", type=float, default=340.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--wire-max-turn-degrees", type=float, default=70.0)
    parser.add_argument("--physical-curvature-smoothing", type=float, default=0.38)
    parser.add_argument("--physical-curvature-iterations", type=int, default=4)
    parser.add_argument("--physical-tip-stiff-segments", type=int, default=8)
    parser.add_argument("--physical-tip-curvature-smoothing", type=float, default=0.86)
    parser.add_argument("--physical-tip-curvature-iterations", type=int, default=4)
    parser.add_argument("--wire-max-speed", type=float, default=1.2)
    parser.add_argument("--lookahead-points", type=int, default=7)
    parser.add_argument("--elite-ahead", type=float, default=0.018)
    parser.add_argument("--lateral-gain", type=float, default=0.75)
    parser.add_argument("--piper-cmd", type=float, default=0.30)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--show-path-tubes", action="store_true")
    parser.add_argument("--show-tool-markers", action="store_true")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-camera", choices=["overview", "top", "side", "perspective"], default="overview")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    results = [run_task(args, task) for task in args.tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved summary to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
