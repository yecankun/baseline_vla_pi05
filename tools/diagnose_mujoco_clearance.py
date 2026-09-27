from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.eval_mujoco_guided_wire_rollout import (  # noqa: E402
    _guard_piper_feed,
    _joint_dict_from_vector,
    _scripted_piper_feed,
    image_to_tensor,
    load_model,
    make_sample_from_obs,
)
from simulation.mujoco_guided_wire_env import (  # noqa: E402
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoGuidedWireExpert,
)
from simulation.train_dual_arm_baseline import build_state_vector  # noqa: E402


CSV_FIELDS = [
    "step",
    "success",
    "distance_to_target",
    "path_progress",
    "tip_wall",
    "segment_min_wall",
    "segment_p01_wall",
    "segment_p05_wall",
    "segment_mean_wall",
    "segment_min_index",
    "segment_min_region",
    "segments_near_projection",
    "segments_near_wall",
    "contact_strength",
    "piper_insertion_length",
]


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
        render_overlay=False,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    return MuJoCoGuidedWireEnv(config, seed=seed)


def segment_wall_distances(env: MuJoCoGuidedWireEnv) -> tuple[np.ndarray, np.ndarray]:
    distances = []
    radii = []
    for index, point in enumerate(env.positions):
        center, radius = env._nearest_path_state(
            point,
            update_progress=False,
            progress_hint=env._segment_progress_hint(index),
        )
        distances.append(float(radius - np.linalg.norm(point - center)))
        radii.append(float(radius))
    return np.asarray(distances, dtype=np.float32), np.asarray(radii, dtype=np.float32)


def segment_region(index: int, count: int) -> str:
    if index >= count - 5:
        return "tip"
    if index < 5:
        return "tail"
    return "body"


def clearance_row(env: MuJoCoGuidedWireEnv, obs: dict, projection_margin: float, wall_threshold: float) -> dict:
    distances, _radii = segment_wall_distances(env)
    min_index = int(np.argmin(distances))
    return {
        "step": int(obs["step"]),
        "success": int(bool(obs["success"])),
        "distance_to_target": float(obs["distance_to_target"]),
        "path_progress": float(obs["path_progress"]),
        "tip_wall": float(obs["distance_to_wall"]),
        "segment_min_wall": float(np.min(distances)),
        "segment_p01_wall": float(np.percentile(distances, 1)),
        "segment_p05_wall": float(np.percentile(distances, 5)),
        "segment_mean_wall": float(np.mean(distances)),
        "segment_min_index": min_index,
        "segment_min_region": segment_region(min_index, len(distances)),
        "segments_near_projection": int(np.count_nonzero(distances <= projection_margin)),
        "segments_near_wall": int(np.count_nonzero(distances <= wall_threshold)),
        "contact_strength": float(obs["contact_strength"]),
        "piper_insertion_length": float(obs.get("piper_insertion_length", 0.0)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def summarize(task: str, rows: list[dict], final_obs: dict, projection_margin: float, wall_threshold: float) -> dict:
    def stats(key: str) -> dict:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        return {
            "min": float(np.min(values)),
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "final": float(values[-1]),
        }

    min_regions = Counter(str(row["segment_min_region"]) for row in rows)
    min_indices = Counter(int(row["segment_min_index"]) for row in rows)
    near_projection_steps = sum(int(row["segments_near_projection"] > 0) for row in rows)
    near_wall_steps = sum(int(row["segments_near_wall"] > 0) for row in rows)
    total = max(len(rows), 1)
    return {
        "task": task,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "projection_margin": float(projection_margin),
        "wall_threshold": float(wall_threshold),
        "near_projection_step_ratio": near_projection_steps / total,
        "near_wall_step_ratio": near_wall_steps / total,
        "min_segment_region_counts": dict(min_regions),
        "top_min_segment_indices": [{"index": idx, "count": count} for idx, count in min_indices.most_common(8)],
        "metrics": {
            "tip_wall": stats("tip_wall"),
            "segment_min_wall": stats("segment_min_wall"),
            "segment_p05_wall": stats("segment_p05_wall"),
            "segment_mean_wall": stats("segment_mean_wall"),
            "segments_near_projection": stats("segments_near_projection"),
            "segments_near_wall": stats("segments_near_wall"),
            "contact_strength": stats("contact_strength"),
        },
    }


def model_action(args: argparse.Namespace, env: MuJoCoGuidedWireEnv, obs_dict: dict, bundle: dict, device: torch.device, piper_direction: float):
    images = env.render_camera_pair(size=args.camera_size)
    side = image_to_tensor(images["side"], bundle["image_size"], device)
    top = image_to_tensor(images["top"], bundle["image_size"], device)
    piper_joint_names = bundle["piper_joint_names"]
    elite_joint_names = bundle["elite_joint_names"]
    state_vec = torch.tensor(
        build_state_vector(
            make_sample_from_obs(obs_dict, bundle["max_steps"]),
            bundle["max_steps"],
            piper_joint_names,
            elite_joint_names,
            bundle.get("observation_schema", "full_sim_state"),
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        pred = bundle["model"](side, top, state_vec).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if bundle["action_mode"] != "joint":
        piper = float(np.clip(pred[0], -1.0, 1.0))
        delta = np.tanh(pred[1:4]).astype(np.float32)
        return {"piper": np.array([piper], dtype=np.float32), "elirobot_delta": delta}, piper_direction

    piper_vec = pred[: len(piper_joint_names)]
    elite_vec = pred[len(piper_joint_names) : len(piper_joint_names) + len(elite_joint_names)]
    if args.scripted_piper_feed:
        piper_vec, piper_direction = _scripted_piper_feed(env, obs_dict, obs_dict["task"], piper_direction)
    elif args.piper_feed_guard:
        piper_vec = _guard_piper_feed(env, piper_vec, args)
    return {
        "piper_joints": _joint_dict_from_vector(piper_joint_names, piper_vec),
        "elite_joints": _joint_dict_from_vector(elite_joint_names, elite_vec),
    }, piper_direction


def run_task(args: argparse.Namespace, task: str, device: torch.device, bundle: dict | None) -> dict:
    env = build_env(args, seed=args.seed)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})
    projection_margin = float(env.config.wall_clearance * env.config.boundary_projection_relief + args.projection_eps)
    wall_threshold = float(args.wall_threshold if args.wall_threshold is not None else env.config.wall_margin)
    rows: list[dict] = []
    total_reward = 0.0
    piper_direction = 1.0

    try:
        while True:
            obs_dict = info["obs_dict"]
            rows.append(clearance_row(env, obs_dict, projection_margin, wall_threshold))
            if obs_dict["success"] or obs_dict["failure_reason"] is not None or obs_dict["step"] >= args.max_steps:
                break

            if args.policy == "expert":
                action = expert.act(env)
            else:
                if bundle is None:
                    raise ValueError("--checkpoint is required when --policy model")
                action, piper_direction = model_action(args, env, obs_dict, bundle, device, piper_direction)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                rows.append(clearance_row(env, info["obs_dict"], projection_margin, wall_threshold))
                break
    finally:
        env.close()

    final_obs = info["obs_dict"]
    summary = summarize(task, rows, final_obs, projection_margin, wall_threshold)
    summary["policy"] = args.policy
    summary["total_reward"] = total_reward
    out_dir = Path(args.out) / task
    write_csv(out_dir / "clearance.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{task}: success={summary['success']} steps={summary['steps']} "
        f"min_seg={summary['metrics']['segment_min_wall']['min']:.6f} "
        f"p05_seg={summary['metrics']['segment_p05_wall']['mean']:.6f} "
        f"near_projection={summary['near_projection_step_ratio']:.2%} "
        f"min_regions={summary['min_segment_region_counts']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose per-segment vessel-wall clearance in MuJoCo guided-wire rollouts.")
    parser.add_argument("--policy", choices=["expert", "model"], default="expert")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_clearance_diagnostics")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--max-steps", type=int, default=560)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--show-path-tubes", action="store_true")
    parser.add_argument("--show-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--piper-feed-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piper-feed-min", type=float, default=0.0032)
    parser.add_argument("--piper-feed-max", type=float, default=0.0062)
    parser.add_argument("--scripted-piper-feed", action="store_true")
    parser.add_argument("--projection-eps", type=float, default=5e-5)
    parser.add_argument("--wall-threshold", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    bundle = load_model(args.checkpoint, device) if args.policy == "model" else None
    summaries = [run_task(args, task, device, bundle) for task in args.tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved clearance diagnostics to {out.resolve()}")


if __name__ == "__main__":
    main()
