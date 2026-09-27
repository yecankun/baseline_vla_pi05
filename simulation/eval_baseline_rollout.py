import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, MeshGuidewire3DEnv
from simulation.render_mesh_3d_pyrender import PyrenderRolloutRenderer
from simulation.train_baseline import DualCameraStatePolicy, build_state_vector


def make_sample_from_obs(obs_dict):
    return {
        "task": obs_dict["task"],
        "state": {
            "tip_pos": obs_dict["tip_pos"],
            "heading": obs_dict["heading"],
            "target_pos": obs_dict["target_pos"],
            "contact_flag": obs_dict["contact_flag"],
            "contact_direction": obs_dict["contact_direction"],
            "contact_normal": obs_dict["contact_normal"],
            "contact_strength": obs_dict["contact_strength"],
            "distance_to_wall": obs_dict["distance_to_wall"],
        },
    }


def image_to_tensor(img_bgr_or_rgb, image_size, device):
    # render_camera_pair returns BGR-like OpenCV images; normalization matches training after RGB conversion.
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
    model = DualCameraStatePolicy(state_dim=ckpt["state_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt["image_size"])


def run_rollout(args, task):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, image_size = load_model(args.checkpoint, device)
    env = MeshGuidewire3DEnv(
        Mesh3DEnvConfig(mesh_path=args.mesh, route_config_path=args.route_config),
        seed=args.seed,
    )
    _obs, info = env.reset(options={"task": task})

    out_dir = Path(args.out) / f"{task}_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = None
    writer = None
    if args.video:
        renderer = PyrenderRolloutRenderer(
            env,
            width=args.width,
            height=args.height,
            yaw=args.yaw,
            pitch=args.pitch,
            distance_scale=args.distance_scale,
            vessel_alpha=args.vessel_alpha,
        )
        writer = cv2.VideoWriter(
            str(out_dir / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            renderer.close()
            raise RuntimeError(f"Could not open video writer: {out_dir / 'rollout.mp4'}")

    states = []
    actions = []
    max_contact = 0.0
    min_distance_to_wall = float("inf")
    boundary_projection_count = 0
    total_reward = 0.0
    try:
        for _step in range(args.max_steps):
            obs_dict = info["obs_dict"]
            if writer is not None and obs_dict["step"] % args.render_every == 0:
                writer.write(renderer.render())

            camera_images = env.render_camera_pair(size=args.camera_size)
            side = image_to_tensor(camera_images["side"], image_size, device)
            top = image_to_tensor(camera_images["top"], image_size, device)
            state_vec = torch.tensor(build_state_vector(make_sample_from_obs(obs_dict)), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                steer = model(side, top, state_vec).squeeze(0).cpu().numpy()
            steer = np.clip(steer, -args.max_steer_cmd, args.max_steer_cmd)
            if args.safety_recovery and obs_dict["contact_strength"] > args.safety_threshold:
                normal = np.asarray(obs_dict["contact_normal"], dtype=np.float32)
                steer = steer - normal * obs_dict["contact_strength"] * args.safety_gain
            action = np.array([steer[0], steer[1], steer[2], 1.0], dtype=np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_obs = info["obs_dict"]
            max_contact = max(max_contact, float(step_obs["contact_strength"]))
            min_distance_to_wall = min(min_distance_to_wall, float(step_obs["distance_to_wall"]))
            boundary_projection_count = int(step_obs.get("boundary_projection_count", boundary_projection_count))
            states.append(obs_dict)
            actions.append({"step": obs_dict["step"], "action": action.astype(float).tolist(), "reward": float(reward)})
            if terminated or truncated:
                if writer is not None:
                    for _ in range(args.fps):
                        writer.write(renderer.render())
                break
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()

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
        f"{task}: success={meta['success']} steps={meta['steps']} "
        f"failure={meta['failure_reason']} max_contact={meta['max_contact_strength']:.3f} "
        f"min_wall={meta['min_distance_to_wall']:.3f} projections={meta['boundary_projection_count']}"
    )
    return meta


def main():
    parser = argparse.ArgumentParser(description="Evaluate the dual-camera baseline in closed-loop simulation.")
    parser.add_argument("--checkpoint", default="simulation_output/baseline_dual_camera_state/best_model.pt")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/baseline_rollout")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--max-steer-cmd", type=float, default=0.55)
    parser.add_argument("--safety-recovery", action="store_true")
    parser.add_argument("--safety-threshold", type=float, default=0.25)
    parser.add_argument("--safety-gain", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=3)
    parser.add_argument("--yaw", type=float, default=-70.0)
    parser.add_argument("--pitch", type=float, default=-10.0)
    parser.add_argument("--distance-scale", type=float, default=0.95)
    parser.add_argument("--vessel-alpha", type=float, default=0.58)
    args = parser.parse_args()

    summaries = [run_rollout(args, task) for task in args.tasks]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
