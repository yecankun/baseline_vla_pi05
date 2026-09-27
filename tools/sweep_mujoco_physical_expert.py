from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import (  # noqa: E402
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoMagneticGuideExpert,
)


CSV_FIELDS = [
    "task",
    "start_fraction",
    "progress_delta",
    "short_success",
    "success_reason",
    "reached_progress_target",
    "env_success",
    "failure_reason",
    "steps",
    "start_progress",
    "target_progress",
    "final_progress",
    "progress_gain",
    "distance_to_target",
    "min_segment_distance_to_wall",
    "max_contact_strength",
    "total_reward",
]


def build_env(args: argparse.Namespace, seed: int) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode="none",
        show_path_tubes=False,
        show_tool_markers=False,
        render_observation=False,
        guidance_mode="physical",
        magnet_force=args.magnet_force,
        magnet_range=args.magnet_range,
        piper_force=args.piper_force,
        piper_advection_scale=args.piper_advection_scale,
        spring_k=args.spring_k,
        bend_k=args.bend_k,
        damping=args.damping,
        sim_substeps=args.sim_substeps,
        wall_k=args.wall_k,
        gravity_z=args.gravity_z,
        wire_max_turn_degrees=args.wire_max_turn_degrees,
        physical_curvature_smoothing=args.physical_curvature_smoothing,
        physical_curvature_iterations=args.physical_curvature_iterations,
        physical_tip_stiff_segments=args.physical_tip_stiff_segments,
        physical_tip_curvature_smoothing=args.physical_tip_curvature_smoothing,
        physical_tip_curvature_iterations=args.physical_tip_curvature_iterations,
        wire_max_speed=args.wire_max_speed,
    )
    return MuJoCoGuidedWireEnv(config, seed=seed)


def run_case(args: argparse.Namespace, task: str, start_fraction: float, progress_delta: float, seed: int) -> dict:
    env = build_env(args, seed)
    expert = MuJoCoMagneticGuideExpert(
        rng_seed=seed,
        steer_noise=args.steer_noise,
        lookahead_points=args.lookahead_points,
        elite_ahead=args.elite_ahead,
        lateral_gain=args.lateral_gain,
        piper_cmd=args.piper_cmd,
    )
    _obs, info = env.reset(options={"task": task, "start_fraction": start_fraction})
    start_progress = float(info["obs_dict"]["path_progress"])
    target_progress = min(start_progress + progress_delta, len(env.paths[task]) - 1)
    min_wall = float("inf")
    max_contact = 0.0
    total_reward = 0.0

    while True:
        obs = info["obs_dict"]
        min_wall = min(min_wall, env.segment_min_distance_to_wall())
        max_contact = max(max_contact, float(obs["contact_strength"]))
        short_success = float(obs["path_progress"]) >= target_progress or bool(obs["success"])
        if short_success or obs["failure_reason"] is not None or int(obs["step"]) >= args.max_steps:
            break
        action = expert.act(env)
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    final_obs = info["obs_dict"]
    final_progress = float(final_obs["path_progress"])
    reached_progress_target = final_progress >= target_progress
    env_success = bool(final_obs["success"])
    short_success = reached_progress_target or env_success
    success_reason = "progress_target" if reached_progress_target else ("env_success" if env_success else None)
    env.close()
    return {
        "task": task,
        "start_fraction": float(start_fraction),
        "progress_delta": float(progress_delta),
        "short_success": bool(short_success),
        "success_reason": success_reason,
        "reached_progress_target": bool(reached_progress_target),
        "env_success": env_success,
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "start_progress": float(start_progress),
        "target_progress": float(target_progress),
        "final_progress": final_progress,
        "progress_gain": float(final_progress - start_progress),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "min_segment_distance_to_wall": float(min_wall),
        "max_contact_strength": float(max_contact),
        "total_reward": float(total_reward),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep physical MuJoCo magnetic expert curriculum settings.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_physical_expert_sweep_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--start-fractions", nargs="+", type=float, default=[0.58, 0.62, 0.66, 0.70])
    parser.add_argument("--progress-deltas", nargs="+", type=float, default=[14, 20, 28, 40])
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--magnet-force", type=float, default=0.36)
    parser.add_argument("--magnet-range", type=float, default=0.17)
    parser.add_argument("--piper-force", type=float, default=0.17)
    parser.add_argument("--piper-advection-scale", type=float, default=0.11)
    parser.add_argument("--spring-k", type=float, default=760.0)
    parser.add_argument("--bend-k", type=float, default=165.0)
    parser.add_argument("--damping", type=float, default=6.2)
    parser.add_argument("--sim-substeps", type=int, default=24)
    parser.add_argument("--wall-k", type=float, default=650.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--wire-max-turn-degrees", type=float, default=24.0)
    parser.add_argument("--physical-curvature-smoothing", type=float, default=0.78)
    parser.add_argument("--physical-curvature-iterations", type=int, default=12)
    parser.add_argument("--physical-tip-stiff-segments", type=int, default=7)
    parser.add_argument("--physical-tip-curvature-smoothing", type=float, default=0.82)
    parser.add_argument("--physical-tip-curvature-iterations", type=int, default=4)
    parser.add_argument("--wire-max-speed", type=float, default=0.26)
    parser.add_argument("--lookahead-points", type=int, default=7)
    parser.add_argument("--elite-ahead", type=float, default=0.018)
    parser.add_argument("--lateral-gain", type=float, default=0.75)
    parser.add_argument("--piper-cmd", type=float, default=0.17)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    case_index = 0
    for task in args.tasks:
        for start_fraction in args.start_fractions:
            for progress_delta in args.progress_deltas:
                case_seed = int(args.seed + case_index)
                row = run_case(args, task, start_fraction, progress_delta, case_seed)
                rows.append(row)
                print(
                    f"{task} start={start_fraction:.2f} delta={progress_delta:.1f}: "
                    f"success={row['short_success']} steps={row['steps']} "
                    f"gain={row['progress_gain']:.2f}/{progress_delta:.1f} "
                    f"min_wall={row['min_segment_distance_to_wall']:.6f} contact={row['max_contact_strength']:.3f}",
                    flush=True,
                )
                case_index += 1

    summary = {
        "total_cases": len(rows),
        "success_cases": sum(1 for row in rows if row["short_success"]),
        "success_rate": sum(1 for row in rows if row["short_success"]) / max(len(rows), 1),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "summary.csv", rows)
    print(f"saved sweep to {out}")


if __name__ == "__main__":
    main()
