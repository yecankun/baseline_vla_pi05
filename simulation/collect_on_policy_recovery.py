import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from simulation.dual_arm_guidewire_3d_env import DualArmGuidewire3DEnv, DualArmMesh3DEnvConfig, DualArmSuccessRecoveryExpert
from simulation.eval_dual_arm_rollout import image_to_tensor, load_model, make_sample_from_obs
from simulation.train_dual_arm_baseline import build_state_vector


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_action(model, obs_dict, images, image_size, max_steps, device, delta_scale):
    side = image_to_tensor(images["side"], image_size, device)
    top = image_to_tensor(images["top"], image_size, device)
    state_vec = torch.tensor(build_state_vector(make_sample_from_obs(obs_dict), max_steps), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        piper, delta = model(side, top, state_vec)
    piper = float(torch.clamp(torch.tanh(piper), -1.0, 1.0).item())
    delta = np.tanh(delta.squeeze(0).cpu().numpy()).astype(np.float32) * float(delta_scale)
    return {"piper": np.array([piper], dtype=np.float32), "elirobot_delta": delta}


def teacher_decision(
    obs_dict,
    hold_contact,
    hold_wall,
    teacher_stall_steps,
    teacher_progress_eps,
    stalled_steps,
    projection_hold_steps,
):
    contact_strength = float(obs_dict["contact_strength"])
    distance_to_wall = float(obs_dict["distance_to_wall"])
    boundary_projection_count = int(obs_dict.get("boundary_projection_count", 0))
    last_boundary_projection = bool(obs_dict.get("last_boundary_projection", False))

    if contact_strength >= hold_contact:
        return True, "contact"
    if distance_to_wall <= hold_wall:
        return True, "wall"
    if last_boundary_projection:
        return True, "projection"
    if projection_hold_steps > 0:
        return True, "projection_hold"
    if stalled_steps >= teacher_stall_steps:
        return True, "stall"
    if boundary_projection_count > 0 and distance_to_wall <= hold_wall * 1.5:
        return True, "projection_near_wall"
    return False, None


def collect_episode(args, episode_index: int, task: str, seed: int, model, image_size: int, ckpt_max_steps: int, device):
    env = DualArmGuidewire3DEnv(
        DualArmMesh3DEnvConfig(
            mesh_path=args.mesh,
            route_config_path=args.route_config,
            max_steps=args.max_steps,
        ),
        seed=seed,
    )
    expert = DualArmSuccessRecoveryExpert(rng_seed=seed, steer_noise=0.0, push_strength=0.35)
    _obs, info = env.reset(options={"task": task})

    episode_dir = Path(args.out) / f"episode_{episode_index:06d}_on_policy_recovery_{task}"
    side_dir = episode_dir / "frames" / "side"
    top_dir = episode_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    teacher_steps = 0
    model_steps = 0
    danger_steps = 0
    stalled_steps = 0
    projection_hold_steps = 0
    prev_progress = None
    prev_projection_count = 0
    total_reward = 0.0

    for _ in range(args.max_steps):
        obs_dict = info["obs_dict"]
        step = int(obs_dict["step"])
        progress = float(obs_dict["path_progress"])
        if prev_progress is not None and progress <= prev_progress + args.teacher_progress_eps:
            stalled_steps += 1
        else:
            stalled_steps = 0
        prev_progress = progress

        boundary_projection_count = int(obs_dict.get("boundary_projection_count", 0))
        if boundary_projection_count > prev_projection_count:
            projection_hold_steps = args.teacher_projection_hold
        else:
            projection_hold_steps = max(0, projection_hold_steps - 1)
        prev_projection_count = boundary_projection_count

        images = env.render_camera_pair(size=args.image_size)
        cv2.imwrite(str(side_dir / f"{step:06d}.png"), images["side"])
        cv2.imwrite(str(top_dir / f"{step:06d}.png"), images["top"])

        expert_action = expert.act(env)
        policy_action = model_action(model, obs_dict, images, image_size, ckpt_max_steps, device, args.delta_scale)
        danger, teacher_reason = teacher_decision(
            obs_dict,
            args.teacher_contact,
            args.teacher_wall,
            args.teacher_stall_steps,
            args.teacher_progress_eps,
            stalled_steps,
            projection_hold_steps,
        )
        use_teacher = danger or args.teacher_ratio >= 1.0
        if 0.0 < args.teacher_ratio < 1.0:
            rng_value = np.random.default_rng(seed + step).random()
            use_teacher = use_teacher or rng_value < args.teacher_ratio

        exec_action = expert_action if use_teacher else policy_action
        if use_teacher:
            teacher_steps += 1
        else:
            model_steps += 1
        if danger:
            danger_steps += 1

        states.append(obs_dict)
        actions.append(
            {
                "step": step,
                "action": {
                    "piper": float(expert_action["piper"][0]),
                    "elirobot_delta": expert_action["elirobot_delta"].astype(float).tolist(),
                },
                "executed_action": {
                    "piper": float(exec_action["piper"][0]),
                    "elirobot_delta": exec_action["elirobot_delta"].astype(float).tolist(),
                },
                "used_teacher": bool(use_teacher),
                "danger": bool(danger),
                "teacher_reason": teacher_reason,
            }
        )

        _next_obs, reward, terminated, truncated, info = env.step(exec_action)
        total_reward += float(reward)
        if terminated or truncated:
            final_obs = info["obs_dict"]
            final_step = int(final_obs["step"])
            images = env.render_camera_pair(size=args.image_size)
            cv2.imwrite(str(side_dir / f"{final_step:06d}.png"), images["side"])
            cv2.imwrite(str(top_dir / f"{final_step:06d}.png"), images["top"])
            states.append(final_obs)
            break

    final_obs = info["obs_dict"]
    meta = {
        "task": task,
        "scenario": "on_policy_recovery",
        "instruction": env.instruction,
        "steps": int(final_obs["step"]),
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "total_reward": float(total_reward),
        "episode": episode_dir.name,
        "teacher_steps": int(teacher_steps),
        "model_steps": int(model_steps),
        "danger_steps": int(danger_steps),
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(episode_dir / "states.jsonl", states)
    write_jsonl(episode_dir / "actions.jsonl", actions)
    return meta


def append_samples(manifest, episode_dir: Path, meta):
    states = [json.loads(x) for x in (episode_dir / "states.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    actions = [json.loads(x) for x in (episode_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    action_by_step = {item["step"]: item for item in actions}
    for state in states:
        step = state["step"]
        if step not in action_by_step:
            continue
        manifest["samples"].append(
            {
                "episode": episode_dir.name,
                "task": meta["task"],
                "scenario": meta["scenario"],
                "step": step,
                "mode": "dual_arm",
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
                    "boundary_projection_count": state.get("boundary_projection_count", 0),
                    "last_boundary_projection": state.get("last_boundary_projection", False),
                    "path_progress": state["path_progress"],
                    "piper_step": state["piper_step"],
                    "piper_insertion_length": state["piper_insertion_length"],
                    "elirobot_pose": state["elirobot_pose"],
                    "lateral_offset": state["lateral_offset"],
                    "path_tangent": state["path_tangent"],
                    "local_radius": state["local_radius"],
                },
                "action": action_by_step[step]["action"],
                "executed_action": action_by_step[step]["executed_action"],
                "used_teacher": action_by_step[step]["used_teacher"],
                "danger": action_by_step[step]["danger"],
                "teacher_reason": action_by_step[step].get("teacher_reason"),
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Collect DAgger-style on-policy recovery samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/dual_arm_dataset_on_policy_recovery_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--teacher-contact", type=float, default=0.20)
    parser.add_argument("--teacher-wall", type=float, default=0.014)
    parser.add_argument("--teacher-ratio", type=float, default=0.0)
    parser.add_argument("--teacher-stall-steps", type=int, default=4)
    parser.add_argument("--teacher-projection-hold", type=int, default=6)
    parser.add_argument("--teacher-progress-eps", type=float, default=0.015)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, model_image_size, ckpt_max_steps = load_model(args.checkpoint, device)
    manifest = {
        "mesh": args.mesh,
        "route_config": args.route_config,
        "mode": "dual_arm",
        "source_checkpoint": args.checkpoint,
        "episodes": [],
        "samples": [],
    }

    episode_index = 1
    for task in args.tasks:
        for _ in range(args.episodes_per_task):
            seed = args.seed + episode_index
            meta = collect_episode(args, episode_index, task, seed, model, model_image_size, ckpt_max_steps, device)
            manifest["episodes"].append(meta)
            append_samples(manifest, out_root / meta["episode"], meta)
            print(
                f"{meta['episode']}: task={task}, success={meta['success']}, steps={meta['steps']}, "
                f"failure={meta['failure_reason']}, teacher={meta['teacher_steps']}, model={meta['model_steps']}, danger={meta['danger_steps']}"
            )
            episode_index += 1

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}")
    print(f"Episodes: {len(manifest['episodes'])}, samples: {len(manifest['samples'])}")


if __name__ == "__main__":
    main()
