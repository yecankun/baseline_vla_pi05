import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from simulation.dual_arm_guidewire_3d_env import DualArmGuidewire3DEnv, DualArmMesh3DEnvConfig
from simulation.train_dual_arm_baseline import DualArmCameraStatePolicy, build_state_vector


def make_sample_from_obs(obs_dict):
    return {
        "task": obs_dict["task"],
        "step": obs_dict["step"],
        "state": {
            "tip_pos": obs_dict["tip_pos"],
            "heading": obs_dict["heading"],
            "target_pos": obs_dict["target_pos"],
            "contact_normal": obs_dict["contact_normal"],
            "distance_to_wall": obs_dict["distance_to_wall"],
            "contact_strength": obs_dict["contact_strength"],
            "contact_flag": obs_dict["contact_flag"],
            "path_progress": obs_dict["path_progress"],
            "piper_step": obs_dict["piper_step"],
            "piper_insertion_length": obs_dict["piper_insertion_length"],
            "elirobot_pose": obs_dict["elirobot_pose"],
            "lateral_offset": obs_dict["lateral_offset"],
            "path_tangent": obs_dict["path_tangent"],
            "local_radius": obs_dict["local_radius"],
        },
    }


def image_to_tensor(img_bgr_or_rgb, image_size, device):
    img = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualArmCameraStatePolicy(state_dim=ckpt["state_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt["image_size"]), int(ckpt.get("max_steps", 280))


def run_rollout(args, task):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, image_size, max_steps = load_model(args.checkpoint, device)
    env = DualArmGuidewire3DEnv(
        DualArmMesh3DEnvConfig(mesh_path=args.mesh, route_config_path=args.route_config, max_steps=max_steps),
        seed=args.seed,
    )
    _obs, info = env.reset(options={"task": task})

    out_dir = Path(args.out) / f"{task}_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "rollout.mp4"), fourcc, args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_dir / 'rollout.mp4'}")

    max_contact = 0.0
    min_distance_to_wall = float("inf")
    boundary_projection_count = 0
    total_reward = 0.0
    states = []
    actions = []

    try:
        for _step in range(args.max_steps):
            obs_dict = info["obs_dict"]
            if writer is not None and obs_dict["step"] % args.render_every == 0:
                frame = env.render()
                if frame.shape[0] != args.height or frame.shape[1] != args.width:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                writer.write(frame)

            images = env.render_camera_pair(size=args.camera_size)
            side = image_to_tensor(images["side"], image_size, device)
            top = image_to_tensor(images["top"], image_size, device)
            state_vec = torch.tensor(build_state_vector(make_sample_from_obs(obs_dict), max_steps), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                piper, delta = model(side, top, state_vec)
            piper = float(torch.clamp(torch.tanh(piper), -1.0, 1.0).item())
            if args.tactile_safety:
                contact_strength = float(obs_dict.get("contact_strength", 0.0))
                distance_to_wall = float(obs_dict.get("distance_to_wall", 1.0))
                if contact_strength > args.retreat_contact:
                    piper = min(piper, args.retreat_piper)
                elif contact_strength > args.hold_contact or distance_to_wall < args.hold_wall:
                    piper = min(piper, args.hold_piper)
            delta = delta.squeeze(0).cpu().numpy()
            delta = np.tanh(delta).astype(np.float32) * args.delta_scale
            action = {"piper": np.array([piper], dtype=np.float32), "elirobot_delta": np.asarray(delta, dtype=np.float32)}
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_obs = info["obs_dict"]
            max_contact = max(max_contact, float(step_obs["contact_strength"]))
            min_distance_to_wall = min(min_distance_to_wall, float(step_obs["distance_to_wall"]))
            boundary_projection_count = int(step_obs.get("boundary_projection_count", boundary_projection_count))
            states.append(obs_dict)
            actions.append({"step": obs_dict["step"], "action": {"piper": piper, "elirobot_delta": delta.astype(float).tolist()}, "reward": float(reward)})
            if terminated or truncated:
                final_obs = info["obs_dict"]
                if writer is not None:
                    for _ in range(args.fps):
                        frame = env.render()
                        if frame.shape[0] != args.height or frame.shape[1] != args.width:
                            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                        writer.write(frame)
                states.append(final_obs)
                break
    finally:
        if writer is not None:
            writer.release()

    final_obs = info["obs_dict"]
    meta = {
        "task": task,
        "scenario": args.scenario,
        "checkpoint": args.checkpoint,
        "success": final_obs["success"],
        "failure_reason": final_obs["failure_reason"],
        "steps": final_obs["step"],
        "max_contact_strength": max_contact,
        "min_distance_to_wall": min_distance_to_wall,
        "boundary_projection_count": boundary_projection_count,
        "total_reward": total_reward,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")
    with (out_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for action in actions:
            f.write(json.dumps(action, ensure_ascii=False) + "\n")
    print(
        f"{task}: success={meta['success']} steps={meta['steps']} failure={meta['failure_reason']} "
        f"max_contact={meta['max_contact_strength']:.3f} min_wall={meta['min_distance_to_wall']:.3f} "
        f"projections={meta['boundary_projection_count']}"
    )
    return meta


def main():
    parser = argparse.ArgumentParser(description="Evaluate the dual-arm baseline in closed-loop simulation.")
    parser.add_argument("--checkpoint", default="simulation_output/baseline_dual_arm_camera_state/best_model.pt")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/baseline_dual_arm_rollout")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--tactile-safety", action="store_true")
    parser.add_argument("--hold-contact", type=float, default=0.28)
    parser.add_argument("--retreat-contact", type=float, default=0.62)
    parser.add_argument("--hold-wall", type=float, default=0.010)
    parser.add_argument("--hold-piper", type=float, default=0.0)
    parser.add_argument("--retreat-piper", type=float, default=-0.08)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=1)
    args = parser.parse_args()

    summaries = [run_rollout(args, task) for task in args.tasks]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
