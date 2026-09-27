from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import (
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoGuidedWireExpert,
)


def run_task(args: argparse.Namespace, task: str) -> dict:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        max_steps=args.max_steps,
        render_width=args.width,
        render_height=args.height,
        wire_segments=args.wire_segments,
        render_observation=args.render_observation,
        robot_visual_mode=args.robot_visual_mode,
    )
    env = MuJoCoGuidedWireEnv(config, seed=args.seed)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})

    out_dir = Path(args.out) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        video_path = out_dir / "rollout.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

    states = []
    actions = []
    total_reward = 0.0
    try:
        while True:
            obs_dict = info["obs_dict"]
            if writer is not None and obs_dict["step"] % args.render_every == 0:
                writer.write(env.render())
            action = expert.act(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            states.append(obs_dict)
            actions.append(
                {
                    "step": obs_dict["step"],
                    "action": {
                        "piper_joints": {name: float(value) for name, value in action["piper_joints"].items()},
                        "elite_joints": {name: float(value) for name, value in action["elite_joints"].items()},
                    },
                    "reward": float(reward),
                }
            )
            total_reward += float(reward)
            if terminated or truncated:
                states.append(info["obs_dict"])
                if writer is not None:
                    for _ in range(args.fps):
                        writer.write(env.render())
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    final_obs = info["obs_dict"]
    meta = {
        "task": task,
        "success": final_obs["success"],
        "failure_reason": final_obs["failure_reason"],
        "steps": final_obs["step"],
        "distance_to_target": final_obs["distance_to_target"],
        "path_progress": final_obs["path_progress"],
        "boundary_projection_count": final_obs["boundary_projection_count"],
        "total_reward": total_reward,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")
    with (out_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for action in actions:
            f.write(json.dumps(action, ensure_ascii=False) + "\n")
    print(
        f"{task}: success={meta['success']} steps={meta['steps']} failure={meta['failure_reason']} "
        f"dist={meta['distance_to_target']:.3f} progress={meta['path_progress']:.1f} "
        f"projections={meta['boundary_projection_count']}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the MuJoCo guided wire MVP.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--out", default="simulation_output/mujoco_guided_wire_check")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--steer-noise", type=float, default=0.01)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-observation", action="store_true")
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    args = parser.parse_args()

    summaries = [run_task(args, task) for task in args.tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
