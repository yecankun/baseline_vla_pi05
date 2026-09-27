from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv, TipMagneticGuideExpert


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def run_task(args, task: str) -> dict:
    cfg = TipGuidedWireConfig(
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        render_width=args.render_width,
        render_height=args.render_height,
        render_observation=False,
        guidance_mode="physical",
        robot_visual_mode=args.robot_visual_mode,
        show_tool_markers=not args.hide_tool_markers,
        show_path_tubes=not args.hide_path_tubes,
        tip_progress_gain=args.tip_progress_gain,
        tip_lateral_magnetic_gain=args.tip_lateral_magnetic_gain,
        tip_lateral_centering_gain=args.tip_lateral_centering_gain,
        tip_contact_relief_gain=args.tip_contact_relief_gain,
    )
    env = TipGuidedWireEnv(cfg, seed=args.seed)
    expert = TipMagneticGuideExpert(
        rng_seed=args.seed,
        steer_noise=args.steer_noise,
        lookahead_points=args.lookahead_points,
        elite_ahead=args.elite_ahead,
        lateral_gain=args.lateral_gain,
        piper_cmd=args.piper_cmd,
    )
    states = []
    try:
        _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})
        final_obs = info["obs_dict"]
        for _ in range(args.max_steps):
            obs = info["obs_dict"]
            states.append(obs)
            if args.progress_log_every > 0 and int(obs["step"]) % args.progress_log_every == 0:
                print(
                    f"  {task}: step={obs['step']}, progress={obs['path_progress']:.2f}, "
                    f"dist={obs['distance_to_target']:.4f}, contact={obs['contact_strength']:.3f}",
                    flush=True,
                )
            action = expert.act(env)
            _obs, _reward, terminated, truncated, info = env.step(action)
            final_obs = info["obs_dict"]
            if terminated or truncated:
                states.append(final_obs)
                break
    finally:
        env.close()

    progress = [float(s["path_progress"]) for s in states]
    dprogress = [progress[i] - progress[i - 1] for i in range(1, len(progress))]
    contacts = [float(s["contact_strength"]) for s in states]
    walls = [float(s["distance_to_wall"]) for s in states]
    tip_to_mag = [dist(s["tip_pos"], s["magnetic_pose"]) for s in states]
    summary = {
        "task": task,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "progress_gain": float(progress[-1] - progress[0]) if progress else 0.0,
        "stall_fraction": float(sum(abs(x) < 1e-8 for x in dprogress) / max(len(dprogress), 1)),
        "slow_fraction": float(sum(0 <= x < args.slow_progress_threshold for x in dprogress) / max(len(dprogress), 1)),
        "max_contact_strength": max(contacts) if contacts else 0.0,
        "contact_p95": percentile(contacts, 95),
        "min_tip_distance_to_wall": min(walls) if walls else 0.0,
        "tip_to_magnetic_median": percentile(tip_to_mag, 50),
        "tip_to_magnetic_p95": percentile(tip_to_mag, 95),
        "tip_to_magnetic_max": max(tip_to_mag) if tip_to_mag else 0.0,
    }
    return summary


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the tip-centric guidewire prototype.")
    parser.add_argument("--out", default="")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=9711)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--hide-tool-markers", action="store_true")
    parser.add_argument("--hide-path-tubes", action="store_true")
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--lookahead-points", type=int, default=8)
    parser.add_argument("--elite-ahead", type=float, default=0.018)
    parser.add_argument("--lateral-gain", type=float, default=0.75)
    parser.add_argument("--piper-cmd", type=float, default=0.34)
    parser.add_argument("--tip-progress-gain", type=float, default=0.45)
    parser.add_argument("--tip-lateral-magnetic-gain", type=float, default=0.16)
    parser.add_argument("--tip-lateral-centering-gain", type=float, default=0.06)
    parser.add_argument("--tip-contact-relief-gain", type=float, default=0.42)
    parser.add_argument("--slow-progress-threshold", type=float, default=0.05)
    parser.add_argument("--progress-log-every", type=int, default=100)
    args = parser.parse_args()

    summaries = [run_task(args, task) for task in args.tasks]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tip_guided_wire_smoke_summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
