import argparse
from pathlib import Path

import cv2

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, Mesh3DPathExpert, MeshGuidewire3DEnv


def main():
    parser = argparse.ArgumentParser(description="Render an automatic 3D guidewire rollout.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--out", default="simulation_output/mesh_3d_visual/rollout.mp4")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--yaw", type=float, default=-35.0)
    parser.add_argument("--pitch", type=float, default=18.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=320)
    args = parser.parse_args()

    env = MeshGuidewire3DEnv(Mesh3DEnvConfig(mesh_path=args.mesh), seed=args.seed)
    expert = Mesh3DPathExpert()
    _obs, info = env.reset(options={"task": args.task})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (env.config.render_size, env.config.render_size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    final_frame = None
    frame_count = 0
    while frame_count < args.max_frames:
        frame = env.render_perspective(yaw_deg=args.yaw, pitch_deg=args.pitch, zoom=args.zoom)
        writer.write(frame)
        final_frame = frame
        action = expert.act(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        frame_count += 1
        if terminated or truncated:
            frame = env.render_perspective(yaw_deg=args.yaw, pitch_deg=args.pitch, zoom=args.zoom)
            for _ in range(args.fps):
                writer.write(frame)
            final_frame = frame
            break

    writer.release()
    if args.snapshot:
        snap_path = Path(args.snapshot)
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap_path), final_frame)

    obs_dict = info["obs_dict"]
    print(
        f"saved {out_path} frames={frame_count} task={args.task} "
        f"success={obs_dict['success']} failure={obs_dict['failure_reason']}"
    )


if __name__ == "__main__":
    main()
