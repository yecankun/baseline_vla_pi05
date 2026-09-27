from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.eval_mujoco_guided_wire_rollout import (  # noqa: E402
    _joint_dict_from_vector,
    _scripted_piper_feed,
    image_to_tensor,
    load_model,
    make_sample_from_obs,
)
from simulation.mujoco_guided_wire_env import MuJoCoGuidedWireConfig, MuJoCoGuidedWireEnv  # noqa: E402
from simulation.train_dual_arm_baseline import build_state_vector  # noqa: E402


def build_config(args: argparse.Namespace, mode: str) -> MuJoCoGuidedWireConfig:
    kwargs = {
        "mesh_path": args.mesh,
        "route_config_path": args.route_config,
        "scene_config_path": args.scene_config,
        "camera_config_path": args.camera_config,
        "max_steps": args.max_steps,
        "wire_segments": args.wire_segments,
        "robot_visual_mode": args.robot_visual_mode,
        "show_path_tubes": args.show_path_tubes,
        "show_tool_markers": args.show_tool_markers,
        "render_overlay": False,
        "render_observation": False,
        "render_width": args.render_width,
        "render_height": args.render_height,
    }
    if "no_magnet" in mode:
        kwargs["magnet_force"] = 0.0
    if "weak_autopilot" in mode:
        kwargs.update(
            {
                "tip_path_drive_force": args.weak_tip_path_drive_force,
                "tip_centering_force": args.weak_tip_centering_force,
                "wire_centering_force": args.weak_wire_centering_force,
                "wire_centerline_relaxation": args.weak_wire_centerline_relaxation,
                "branch_centering_force_scale": 1.0,
                "branch_tip_centering_force_scale": 1.0,
                "branch_relaxation_scale": 1.0,
            }
        )
    return MuJoCoGuidedWireConfig(**kwargs)


def current_action(env: MuJoCoGuidedWireEnv, piper_vec: np.ndarray, elite_vec: np.ndarray) -> dict:
    return {
        "piper_joints": _joint_dict_from_vector(env.piper_joint_names, piper_vec),
        "elite_joints": _joint_dict_from_vector(env.elite_joint_names, elite_vec),
    }


def model_elite_action(
    env: MuJoCoGuidedWireEnv,
    obs_dict: dict,
    model_bundle: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    images = env.render_camera_pair(size=args.camera_size)
    side = image_to_tensor(images["side"], model_bundle["image_size"], device)
    top = image_to_tensor(images["top"], model_bundle["image_size"], device)
    state_vec = torch.tensor(
        build_state_vector(
            make_sample_from_obs(obs_dict, model_bundle["max_steps"]),
            model_bundle["max_steps"],
            model_bundle["piper_joint_names"],
            model_bundle["elite_joint_names"],
            model_bundle.get("observation_schema", "full_sim_state"),
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        pred = model_bundle["model"](side, top, state_vec).squeeze(0).detach().cpu().numpy().astype(np.float32)
    piper_dim = len(model_bundle["piper_joint_names"])
    return pred[piper_dim : piper_dim + len(model_bundle["elite_joint_names"])]


def run_one(args: argparse.Namespace, mode: str, task: str, model_bundle: dict | None, device: torch.device) -> dict:
    env = MuJoCoGuidedWireEnv(build_config(args, mode), seed=args.seed)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})
    rng = np.random.default_rng(args.seed + (0 if task == "left" else 1000))
    scripted_piper_direction = 1.0
    fixed_elite = env._robot_joint_vector("elite").copy()
    min_tip_wall = float("inf")
    min_segment_wall = float("inf")
    max_contact = 0.0
    total_reward = 0.0
    elite_ref_error_sum = 0.0
    elite_ref_error_count = 0

    while True:
        obs_dict = info["obs_dict"]
        piper_vec, scripted_piper_direction = _scripted_piper_feed(env, obs_dict, task, scripted_piper_direction)

        if mode.startswith("model"):
            assert model_bundle is not None
            elite_vec = model_elite_action(env, obs_dict, model_bundle, device, args)
        elif "random_elite" in mode:
            current = env._robot_joint_vector("elite")
            noise = rng.normal(0.0, args.random_elite_joint_std, size=current.shape).astype(np.float32)
            elite_vec = env._clip_robot_joint_vector("elite", current + noise)
        else:
            elite_vec = fixed_elite

        action = current_action(env, piper_vec, elite_vec)
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_obs = info["obs_dict"]
        min_tip_wall = min(min_tip_wall, float(step_obs["distance_to_wall"]))
        min_segment_wall = min(min_segment_wall, env.segment_min_distance_to_wall())
        max_contact = max(max_contact, float(step_obs["contact_strength"]))
        elite_pose = np.asarray(step_obs["elirobot_pose"], dtype=np.float32)
        elite_ref = np.asarray(step_obs["elite_reference_pose"], dtype=np.float32)
        elite_ref_error_sum += float(np.linalg.norm(elite_pose - elite_ref))
        elite_ref_error_count += 1

        if terminated or truncated:
            break

    final_obs = info["obs_dict"]
    env.close()
    return {
        "mode": mode,
        "task": task,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "min_tip_distance_to_wall": float(min_tip_wall),
        "min_segment_distance_to_wall": float(min_segment_wall),
        "max_contact_strength": float(max_contact),
        "mean_elite_reference_error": float(elite_ref_error_sum / max(elite_ref_error_count, 1)),
        "total_reward": float(total_reward),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate MuJoCo guidewire guidance sources.")
    parser.add_argument("--checkpoint", default="simulation_output/baseline_mujoco_wire_centerline_v1/best_model.pt")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_guidance_ablation_v1")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["model", "fixed_elite", "no_magnet_fixed_elite", "weak_autopilot_fixed_elite"],
        choices=["model", "fixed_elite", "random_elite", "no_magnet_fixed_elite", "weak_autopilot_fixed_elite"],
    )
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="none")
    parser.add_argument("--show-path-tubes", action="store_true")
    parser.add_argument("--show-tool-markers", action="store_true")
    parser.add_argument("--random-elite-joint-std", type=float, default=0.015)
    parser.add_argument("--weak-tip-path-drive-force", type=float, default=0.006)
    parser.add_argument("--weak-tip-centering-force", type=float, default=0.28)
    parser.add_argument("--weak-wire-centering-force", type=float, default=0.10)
    parser.add_argument("--weak-wire-centerline-relaxation", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    needs_model = any(mode.startswith("model") for mode in args.modes)
    model_bundle = load_model(args.checkpoint, device) if needs_model else None

    results = []
    for mode in args.modes:
        for task in args.tasks:
            result = run_one(args, mode, task, model_bundle, device)
            results.append(result)
            print(
                f"{mode}/{task}: success={result['success']} steps={result['steps']} "
                f"dist={result['distance_to_target']:.4f} min_seg={result['min_segment_distance_to_wall']:.6f} "
                f"max_contact={result['max_contact_strength']:.3f} elite_ref_err={result['mean_elite_reference_error']:.4f}",
                flush=True,
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
