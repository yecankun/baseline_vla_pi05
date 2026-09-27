from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def segment_min_distance_to_wall(env: MuJoCoGuidedWireEnv) -> float:
    return env.segment_min_distance_to_wall()


def sample_state(env: MuJoCoGuidedWireEnv, obs: dict) -> dict:
    state = dict(obs)
    state["segment_min_distance_to_wall"] = segment_min_distance_to_wall(env)
    return state


def build_env(args: argparse.Namespace, seed: int) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        show_vessel_mesh=not args.hide_vessel_mesh,
        show_path_tubes=not args.hide_path_tubes,
        show_tool_markers=not args.hide_tool_markers,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    return MuJoCoGuidedWireEnv(config, seed=seed)


def action_dict(action: dict) -> dict:
    if "piper_joints" in action and "elite_joints" in action:
        return {
            "piper_joints": {name: float(value) for name, value in action["piper_joints"].items()},
            "elite_joints": {name: float(value) for name, value in action["elite_joints"].items()},
        }
    return {
        "piper": float(np.asarray(action["piper"], dtype=np.float32).reshape(-1)[0]),
        "elirobot_delta": np.asarray(action["elirobot_delta"], dtype=np.float32).reshape(3).astype(float).tolist(),
    }


def collect_episode(
    args: argparse.Namespace,
    episode_dir: Path,
    task: str,
    seed: int,
    start_fraction: float,
) -> dict:
    env = build_env(args, seed)
    expert = MuJoCoGuidedWireExpert(rng_seed=seed, steer_noise=args.steer_noise)
    side_dir = episode_dir / "frames" / "side"
    top_dir = episode_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    states: list[dict] = []
    actions: list[dict] = []
    total_reward = 0.0
    min_tip_wall = float("inf")
    min_segment_wall = float("inf")
    max_contact_strength = 0.0
    _gym_obs, info = env.reset(options={"task": task, "start_fraction": start_fraction})
    try:
        while True:
            obs = sample_state(env, info["obs_dict"])
            step = int(obs["step"])
            min_tip_wall = min(min_tip_wall, float(obs["distance_to_wall"]))
            min_segment_wall = min(min_segment_wall, float(obs["segment_min_distance_to_wall"]))
            max_contact_strength = max(max_contact_strength, float(obs["contact_strength"]))

            action = expert.act(env)
            if step % max(args.sample_every, 1) == 0:
                images = env.render_camera_pair(size=args.image_size)
                cv2.imwrite(str(side_dir / f"{step:06d}.png"), images["side"])
                cv2.imwrite(str(top_dir / f"{step:06d}.png"), images["top"])
                states.append(obs)
                actions.append({"step": step, "action": action_dict(action)})

            _gym_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                final_obs = sample_state(env, info["obs_dict"])
                min_tip_wall = min(min_tip_wall, float(final_obs["distance_to_wall"]))
                min_segment_wall = min(min_segment_wall, float(final_obs["segment_min_distance_to_wall"]))
                max_contact_strength = max(max_contact_strength, float(final_obs["contact_strength"]))
                break
    finally:
        env.close()

    action_by_step = {item["step"]: item["action"] for item in actions}
    meta = {
        "episode": episode_dir.name,
        "task": task,
        "scenario": "mujoco_expert",
        "instruction": final_obs["instruction"],
        "seed": seed,
        "start_fraction": start_fraction,
        "steps": int(final_obs["step"]),
        "samples": len(states),
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "boundary_projection_count": int(final_obs["boundary_projection_count"]),
        "min_tip_distance_to_wall": min_tip_wall,
        "min_segment_distance_to_wall": min_segment_wall,
        "max_contact_strength": max_contact_strength,
        "total_reward": total_reward,
        "action_schema": {
            "type": "joint_target",
            "piper_joint_names": env.piper_joint_names,
            "elite_joint_names": env.elite_joint_names,
        },
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(episode_dir / "states.jsonl", states)
    write_jsonl(episode_dir / "actions.jsonl", actions)

    samples = []
    for state in states:
        step = int(state["step"])
        samples.append(
            {
                "episode": episode_dir.name,
                "task": task,
                "scenario": "mujoco_expert",
                "step": step,
                "mode": "mujoco_guided_wire",
                "images": {
                    "side": str((episode_dir / "frames" / "side" / f"{step:06d}.png").as_posix()),
                    "top": str((episode_dir / "frames" / "top" / f"{step:06d}.png").as_posix()),
                },
                "instruction": state["instruction"],
                "state": {
                    "tip_pos": state["tip_pos"],
                    "heading": state["heading"],
                    "target_pos": state["target_pos"],
                    "contact_flag": state["contact_flag"],
                    "contact_direction": state["contact_direction"],
                    "contact_normal": state["contact_normal"],
                    "contact_strength": state["contact_strength"],
                    "distance_to_wall": state["distance_to_wall"],
                    "segment_min_distance_to_wall": state["segment_min_distance_to_wall"],
                    "path_progress": state["path_progress"],
                    "piper_step": state["piper_step"],
                    "piper_insertion_length": state["piper_insertion_length"],
                    "boundary_projection_count": state["boundary_projection_count"],
                    "boundary_projection_window": state["boundary_projection_window"],
                    "last_boundary_projection": state["last_boundary_projection"],
                    "elirobot_pose": state["elirobot_pose"],
                    "magnetic_pose": state["magnetic_pose"],
                    "lateral_offset": state["lateral_offset"],
                    "path_tangent": state["path_tangent"],
                    "local_radius": state["local_radius"],
                    "robot_state": state["robot_state"],
                },
                "action": action_by_step[step],
            }
        )
    return {"meta": meta, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful MuJoCo dual-arm guided-wire imitation episodes.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_guided_wire_dataset_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--max-attempts-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--start-fraction-min", type=float, default=0.52)
    parser.add_argument("--start-fraction-max", type=float, default=0.66)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--hide-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--hide-path-tubes", action="store_true")
    parser.add_argument("--keep-failures", action="store_true", help="Include failed attempts in the training manifest.")
    args = parser.parse_args()

    if args.start_fraction_min > args.start_fraction_max:
        raise ValueError("--start-fraction-min must be <= --start-fraction-max")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mesh": args.mesh,
        "route_config": args.route_config,
        "scene_config": args.scene_config,
        "mode": "mujoco_guided_wire",
        "action_schema": {"type": "joint_target"},
        "episodes": [],
        "rejected_episodes": [],
        "samples": [],
    }

    accepted = 0
    attempts = 0
    for task in args.tasks:
        for slot in range(1, args.episodes_per_task + 1):
            for retry in range(1, args.max_attempts_per_episode + 1):
                attempts += 1
                seed = args.seed + attempts
                start_fraction = float(rng.uniform(args.start_fraction_min, args.start_fraction_max))
                episode_dir = out_root / f"episode_{task}_{slot:04d}_try_{retry:02d}"
                result = collect_episode(args, episode_dir, task, seed, start_fraction)
                meta = result["meta"]
                print(
                    f"{meta['episode']}: task={task}, success={meta['success']}, steps={meta['steps']}, "
                    f"samples={meta['samples']}, min_segment_wall={meta['min_segment_distance_to_wall']:.6f}, "
                    f"failure={meta['failure_reason']}",
                    flush=True,
                )
                if meta["success"] or args.keep_failures:
                    manifest["action_schema"] = meta["action_schema"]
                    manifest["episodes"].append(meta)
                    manifest["samples"].extend(result["samples"])
                else:
                    manifest["rejected_episodes"].append(meta)
                if meta["success"]:
                    accepted += 1
                    break
            else:
                print(f"Warning: no successful episode collected for task={task}, slot={slot}", flush=True)

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}", flush=True)
    print(
        f"Accepted successful episodes: {accepted}, manifest episodes: {len(manifest['episodes'])}, "
        f"rejected attempts: {len(manifest['rejected_episodes'])}, samples: {len(manifest['samples'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
