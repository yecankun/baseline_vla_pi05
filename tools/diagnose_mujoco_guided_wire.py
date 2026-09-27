from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

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


METRIC_FIELDS = [
    "step",
    "reward",
    "piper_cmd",
    "elite_delta_norm",
    "path_progress",
    "distance_to_target",
    "tip_to_elite_distance",
    "tip_to_magnetic_distance",
    "elite_to_reference_distance",
    "elite_height_above_tip",
    "elite_reference_height_above_tip",
    "elite_tip_forward_offset",
    "elite_reference_forward_offset",
    "tip_speed",
    "elite_speed",
    "magnetic_speed",
    "distance_to_wall",
    "wall_clearance_fraction",
    "contact_strength",
    "boundary_projection_count",
    "wire_length",
    "wire_tip_to_tail_distance",
    "wire_curvature_mean",
    "wire_curvature_max",
    "success",
]


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
        render_overlay=args.show_overlay,
        render_observation=False,
        render_width=args.width,
        render_height=args.height,
    )
    return MuJoCoGuidedWireEnv(config, seed=args.seed)


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def mapping_to_array(mapping: dict, names: list[str]) -> np.ndarray:
    return np.asarray([float(mapping[name]) for name in names], dtype=np.float32)


def wire_curvature(points: np.ndarray) -> tuple[float, float]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0, 0.0
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-8
    if np.count_nonzero(valid) < 2:
        return 0.0, 0.0
    dirs = segments[valid] / lengths[valid, None]
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    local_lengths = 0.5 * (lengths[valid][:-1] + lengths[valid][1:])
    curvature = angles / np.maximum(local_lengths, 1e-6)
    return float(np.mean(curvature)), float(np.max(curvature))


def sample_metrics(
    env: MuJoCoGuidedWireEnv,
    obs: dict,
    *,
    action: dict | None,
    reward: float,
    previous: dict | None,
) -> dict:
    tip = as_array(obs["tip_pos"])
    elite = as_array(obs["robot_state"]["elite_tool_world"])
    magnetic = as_array(obs["magnetic_pose"])
    elite_ref = as_array(obs["elite_reference_pose"])
    _center, tangent, _n1, _n2, radius = env._local_path_frame(env.path_progress_float)

    dt = max(float(env.config.sim_dt) * float(env.config.sim_substeps), 1e-6)
    if previous is None:
        tip_speed = 0.0
        elite_speed = 0.0
        magnetic_speed = 0.0
    else:
        tip_speed = float(np.linalg.norm(tip - previous["tip"]) / dt)
        elite_speed = float(np.linalg.norm(elite - previous["elite"]) / dt)
        magnetic_speed = float(np.linalg.norm(magnetic - previous["magnetic"]) / dt)

    piper_cmd = 0.0
    elite_delta_norm = 0.0
    if action is not None:
        robot_state = obs.get("robot_state", {})
        if "piper_joints" in action and "elite_joints" in action:
            piper_names = list(robot_state.get("piper_joints", {}).keys())
            elite_names = list(robot_state.get("elite_joints", {}).keys())
            current_piper = mapping_to_array(robot_state.get("piper_joints", {}), piper_names)
            target_piper = mapping_to_array(action["piper_joints"], piper_names)
            current_elite = mapping_to_array(robot_state.get("elite_joints", {}), elite_names)
            target_elite = mapping_to_array(action["elite_joints"], elite_names)
            piper_cmd = float(np.linalg.norm(target_piper - current_piper))
            elite_delta_norm = float(np.linalg.norm(target_elite - current_elite))
        else:
            piper_cmd = float(np.asarray(action["piper"], dtype=np.float32).reshape(-1)[0])
            elite_delta_norm = float(np.linalg.norm(np.asarray(action["elirobot_delta"], dtype=np.float32).reshape(3)))

    wire = np.asarray(env.positions, dtype=np.float32)
    wire_segments = np.diff(wire, axis=0)
    wire_lengths = np.linalg.norm(wire_segments, axis=1) if len(wire) > 1 else np.asarray([], dtype=np.float32)
    curvature_mean, curvature_max = wire_curvature(wire)
    distance_to_wall = float(obs["distance_to_wall"])
    local_radius = max(float(obs["local_radius"]), 1e-6)

    return {
        "step": int(obs["step"]),
        "reward": float(reward),
        "piper_cmd": piper_cmd,
        "elite_delta_norm": elite_delta_norm,
        "path_progress": float(obs["path_progress"]),
        "distance_to_target": float(obs["distance_to_target"]),
        "tip_to_elite_distance": float(np.linalg.norm(elite - tip)),
        "tip_to_magnetic_distance": float(np.linalg.norm(magnetic - tip)),
        "elite_to_reference_distance": float(np.linalg.norm(elite - elite_ref)),
        "elite_height_above_tip": float(elite[2] - tip[2]),
        "elite_reference_height_above_tip": float(elite_ref[2] - tip[2]),
        "elite_tip_forward_offset": float(np.dot(elite - tip, tangent)),
        "elite_reference_forward_offset": float(np.dot(elite_ref - tip, tangent)),
        "tip_speed": tip_speed,
        "elite_speed": elite_speed,
        "magnetic_speed": magnetic_speed,
        "distance_to_wall": distance_to_wall,
        "wall_clearance_fraction": float(distance_to_wall / local_radius),
        "contact_strength": float(obs["contact_strength"]),
        "boundary_projection_count": int(obs["boundary_projection_count"]),
        "wire_length": float(np.sum(wire_lengths)) if len(wire_lengths) else 0.0,
        "wire_tip_to_tail_distance": float(np.linalg.norm(wire[-1] - wire[0])) if len(wire) > 1 else 0.0,
        "wire_curvature_mean": curvature_mean,
        "wire_curvature_max": curvature_max,
        "success": int(bool(obs["success"])),
        "_state": {"tip": tip, "elite": elite, "magnetic": magnetic},
    }


def summarize_metrics(task: str, final_obs: dict, metrics: list[dict], total_reward: float) -> dict:
    summary = {
        "task": task,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "boundary_projection_count": int(final_obs["boundary_projection_count"]),
        "total_reward": float(total_reward),
        "metrics": {},
    }
    for key in METRIC_FIELDS:
        if key in {"step", "success"}:
            continue
        values = np.asarray([float(item[key]) for item in metrics if key in item and math.isfinite(float(item[key]))], dtype=np.float32)
        if len(values) == 0:
            continue
        summary["metrics"][key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "final": float(values[-1]),
        }
    return summary


def write_metrics_csv(path: Path, metrics: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for item in metrics:
            writer.writerow({key: item.get(key, "") for key in METRIC_FIELDS})


def write_plot(path: Path, metrics: list[dict]) -> bool:
    groups = [
        ("Elite / tip relation", ["tip_to_elite_distance", "tip_to_magnetic_distance", "elite_to_reference_distance"]),
        ("Height and lead", ["elite_height_above_tip", "elite_tip_forward_offset", "elite_reference_forward_offset"]),
        ("Wall contact", ["distance_to_wall", "contact_strength", "wall_clearance_fraction"]),
        ("Motion smoothness", ["tip_speed", "elite_speed", "wire_curvature_mean"]),
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable, writing OpenCV fallback plot ({exc})")
        write_cv2_plot(path, metrics, groups)
        return True

    steps = np.asarray([m["step"] for m in metrics], dtype=np.float32)
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 10), sharex=True)
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            values = np.asarray([m[key] for m in metrics], dtype=np.float32)
            ax.plot(steps, values, label=key)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_cv2_plot(path: Path, metrics: list[dict], groups: list[tuple[str, list[str]]]) -> None:
    width = 1200
    panel_h = 260
    margin_l = 78
    margin_r = 28
    margin_t = 42
    margin_b = 42
    colors = [(35, 115, 230), (30, 170, 80), (200, 100, 30), (170, 70, 170)]
    img = np.full((panel_h * len(groups), width, 3), 250, dtype=np.uint8)
    steps = np.asarray([m["step"] for m in metrics], dtype=np.float32)
    x_min = float(np.min(steps)) if len(steps) else 0.0
    x_max = float(np.max(steps)) if len(steps) else 1.0
    if abs(x_max - x_min) < 1e-6:
        x_max = x_min + 1.0

    for group_idx, (title, keys) in enumerate(groups):
        y0 = group_idx * panel_h
        panel = img[y0 : y0 + panel_h]
        plot_x0 = margin_l
        plot_x1 = width - margin_r
        plot_y0 = margin_t
        plot_y1 = panel_h - margin_b
        cv2.rectangle(panel, (plot_x0, plot_y0), (plot_x1, plot_y1), (215, 215, 215), 1)
        cv2.putText(panel, title, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 45), 2, cv2.LINE_AA)

        series = [np.asarray([m[key] for m in metrics], dtype=np.float32) for key in keys]
        finite_values = np.concatenate([values[np.isfinite(values)] for values in series if np.any(np.isfinite(values))])
        if len(finite_values) == 0:
            continue
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 0.5
            y_max += 0.5
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

        for tick in range(5):
            alpha = tick / 4.0
            y = int(round(plot_y1 - alpha * (plot_y1 - plot_y0)))
            cv2.line(panel, (plot_x0, y), (plot_x1, y), (232, 232, 232), 1)
            label = f"{y_min + alpha * (y_max - y_min):.3g}"
            cv2.putText(panel, label, (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

        for idx, (key, values) in enumerate(zip(keys, series)):
            xs = plot_x0 + (steps - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)
            ys = plot_y1 - (values - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)
            pts = np.column_stack([xs, ys])
            pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(panel, [pts], False, colors[idx % len(colors)], 2, cv2.LINE_AA)
            legend_x = plot_x0 + 12 + idx * 280
            cv2.line(panel, (legend_x, panel_h - 16), (legend_x + 34, panel_h - 16), colors[idx % len(colors)], 2)
            cv2.putText(panel, key, (legend_x + 42, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def run_task(args: argparse.Namespace, task: str) -> dict:
    env = build_env(args)
    expert = MuJoCoGuidedWireExpert(rng_seed=args.seed, steer_noise=args.steer_noise)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})
    out_dir = Path(args.out) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    video_path = out_dir / "rollout.mp4"
    if not args.no_video:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

    metrics: list[dict] = []
    total_reward = 0.0
    previous_state = None
    reward = 0.0
    action = None

    try:
        while True:
            obs_dict = info["obs_dict"]
            metric = sample_metrics(env, obs_dict, action=action, reward=reward, previous=previous_state)
            previous_state = metric.pop("_state")
            metrics.append(metric)

            if writer is not None and obs_dict["step"] % max(args.render_every, 1) == 0:
                writer.write(env.render_camera(args.camera))

            if obs_dict["success"] or obs_dict["failure_reason"] is not None or obs_dict["step"] >= args.max_steps:
                break

            action = expert.act(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                obs_dict = info["obs_dict"]
                metric = sample_metrics(env, obs_dict, action=action, reward=reward, previous=previous_state)
                previous_state = metric.pop("_state")
                metrics.append(metric)
                if writer is not None:
                    for _ in range(max(args.fps // 2, 1)):
                        writer.write(env.render_camera(args.camera))
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    final_obs = info["obs_dict"]
    summary = summarize_metrics(task, final_obs, metrics, total_reward)
    summary["artifacts"] = {
        "metrics_csv": str((out_dir / "metrics.csv").resolve()),
        "summary_json": str((out_dir / "summary.json").resolve()),
        "plot_png": str((out_dir / "diagnostics.png").resolve()),
        "video_mp4": str(video_path.resolve()) if not args.no_video else None,
    }
    write_metrics_csv(out_dir / "metrics.csv", metrics)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_written = write_plot(out_dir / "diagnostics.png", metrics)
    if not plot_written:
        summary["artifacts"]["plot_png"] = None

    print(
        f"{task}: success={summary['success']} steps={summary['steps']} failure={summary['failure_reason']} "
        f"tip_elite_mean={summary['metrics']['tip_to_elite_distance']['mean']:.4f} "
        f"wall_min={summary['metrics']['distance_to_wall']['min']:.4f} "
        f"contact_max={summary['metrics']['contact_strength']['max']:.3f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MuJoCo guided-wire geometry, contact, and Elite/tip motion.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_guided_wire_diagnostics")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--camera", choices=["perspective", "side", "top"], default="perspective")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--hide-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--hide-path-tubes", action="store_true")
    parser.add_argument("--show-overlay", action="store_true", help="Draw debug 2D wire/trail markers over the MuJoCo render.")
    parser.add_argument("--no-video", action="store_true", help="Skip rollout.mp4 generation.")
    args = parser.parse_args()

    summaries = [run_task(args, task) for task in args.tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved diagnostics to {out.resolve()}")


if __name__ == "__main__":
    main()
