from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import (
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoGuidedWireExpert,
)


def configure_viewer_camera(viewer, env: MuJoCoGuidedWireEnv, camera: str) -> None:
    path = env.paths[env.task]
    lo = np.minimum(np.min(path, axis=0), env.scene_min)
    hi = np.maximum(np.max(path, axis=0), env.scene_max)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1e-6)
    distance = max(float(np.max(span)) * 0.88, 1.0)

    if camera == "top":
        viewer.cam.lookat[:] = center
        viewer.cam.distance = distance * 0.98
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -62.0
    elif camera == "side":
        viewer.cam.lookat[:] = center
        viewer.cam.distance = distance
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -8.0
    else:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = distance * 0.82
        viewer.cam.azimuth = 145.0 if env.task == "left" else 35.0
        viewer.cam.elevation = -24.0


def build_env(args: argparse.Namespace) -> MuJoCoGuidedWireEnv:
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
        render_width=args.width,
        render_height=args.height,
    )
    return MuJoCoGuidedWireEnv(config, seed=args.seed)


def run_headless_smoke(args: argparse.Namespace) -> None:
    env = build_env(args)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
    try:
        for _ in range(args.headless_steps):
            action = expert.act(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            final = info["obs_dict"]
            if truncated or final["success"] or (terminated and args.strict_failure):
                break
        final = info["obs_dict"]
        print(
            f"task={final['task']} step={final['step']} success={final['success']} "
            f"failure={final['failure_reason']} dist={final['distance_to_target']:.4f}"
        )
        print(f"piper_tool={np.round(final['robot_state']['piper_tool_world'], 4).tolist()}")
        print(f"elite_tool={np.round(final['robot_state']['elite_tool_world'], 4).tolist()}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open an interactive MuJoCo viewer for the guided-wire dual-robot MVP.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--step-hz", type=float, default=24.0)
    parser.add_argument("--camera", choices=["perspective", "overview", "side", "top"], default="perspective")
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--hide-tool-markers", action="store_true", help="Hide the Piper/Elite debug spheres.")
    parser.add_argument("--hide-vessel-mesh", action="store_true", help="Hide the real STL vessel mesh.")
    parser.add_argument("--hide-path-tubes", action="store_true", help="Hide the left/right path tube overlays.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--loop", action="store_true", help="Restart the same task when it terminates or truncates.")
    parser.add_argument("--strict-failure", action="store_true", help="Stop immediately on simulator failure flags.")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run this many steps without opening the GUI.")
    args = parser.parse_args()

    if args.headless_steps > 0:
        run_headless_smoke(args)
        return

    import mujoco.viewer

    env = build_env(args)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
    step_interval = 1.0 / max(float(args.step_hz), 1e-6)
    next_step = time.perf_counter()

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer_camera(viewer, env, args.camera)
            print("MuJoCo viewer is running. Close the viewer window to stop.")
            while viewer.is_running():
                now = time.perf_counter()
                if now >= next_step:
                    action = expert.act(env)
                    _obs, _reward, terminated, truncated, info = env.step(action)
                    final = info["obs_dict"]
                    if truncated or final["success"] or (terminated and args.strict_failure):
                        final = info["obs_dict"]
                        print(
                            f"{final['task']}: success={final['success']} "
                            f"failure={final['failure_reason']} step={final['step']} "
                            f"dist={final['distance_to_target']:.4f}"
                        )
                        if args.loop:
                            _obs, info = env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
                        else:
                            break
                    next_step = now + step_interval
                viewer.sync()
                time.sleep(0.002)
    finally:
        env.close()


if __name__ == "__main__":
    main()
