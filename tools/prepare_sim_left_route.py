"""Export and plot the recorded simulation's left route; no hardware interfaces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def run(args):
    if not np.isfinite(args.spacing_mm) or args.spacing_mm <= 0:
        raise ValueError("spacing-mm must be finite and positive")
    source = args.demo / "states.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["task"] == "left"]
    tcp = np.asarray([row["robot_state"]["elite_tcp_pose_6d"][:3] for row in rows], dtype=float)
    tip = np.asarray([row["tip_pos"] for row in rows], dtype=float) * 1000
    if len(tcp) < 2 or tcp.shape != tip.shape or tcp.shape[1:] != (3,) or not np.isfinite([tcp, tip]).all():
        raise ValueError("Source must contain finite left-task TCP and tip trajectories")
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(tcp, axis=0), axis=1))]
    if arc[-1] <= 0:
        raise ValueError("Source has no TCP travel")
    keep = np.r_[True, np.diff(arc) > 1e-9]
    stations = np.r_[np.arange(0., arc[-1], args.spacing_mm), arc[-1]]
    points = np.column_stack([np.interp(stations, arc[keep], tcp[keep, axis]) for axis in range(3)])
    chord = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1)]
    # Endpoint and distance checks concern this generated reference, never a robot command.
    if not np.allclose(points[[0, -1]], tcp[[0, -1]]) or np.any(chord > args.spacing_mm + 1e-7):
        raise RuntimeError("Reference resampling failed endpoint or segment-length checks")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "left_sim_waypoints.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["reference_index", "source_arc_mm", "sim_world_x_mm", "sim_world_y_mm", "sim_world_z_mm",
                         "relative_x_mm", "relative_y_mm", "relative_z_mm", "chord_from_previous_mm"])
        for i, (distance, point, length) in enumerate(zip(stations, points, chord)):
            writer.writerow([i, distance, *point, *(point - points[0]), length])
    report = {
        "schema": "simulation_left_route_reference_v1", "source": str(source.resolve()),
        "task": "left", "coordinate_frame": "simulation_world", "position_units": "mm",
        "input_samples": len(rows), "input_sim_steps": int(rows[-1]["step"]),
        "source_path_length_mm": float(arc[-1]), "net_displacement_mm": float(np.linalg.norm(tcp[-1] - tcp[0])),
        "reference_segments": len(points) - 1, "arc_spacing_mm": args.spacing_mm,
        "last_arc_interval_mm": float(stations[-1] - stations[-2]),
        "chord_length_range_mm": [float(chord[1:].min()), float(chord[1:].max())],
        "resampled_chord_total_mm": float(chord.sum()),
        "simulation_to_robot_transform": None, "live_magnet_tcp_verified": False,
        "executable_on_robot": False, "hardware_commands": 0, "feeder_commands": 0,
        "feeder_schedule": None,
        "physical_arm_motion_confirmed": None, "visual_status": "not_viewed",
        "limitations": [
            "Arc spacing is a reference sampling distance, not a minimum real motion distance.",
            "Reference chords approximate the recorded TCP curve and do not preserve its exact timing or orientation.",
            "Simulated tip trajectory is constrained by the model; it is not a measured real guidewire path.",
            "No current-scene registration, tool calibration, collision validation or live feeder distance calibration.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = FontProperties(fname=font_path)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = font.get_name()
    # Register explicitly for installations whose font cache predates this font.
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=140)
    delta, sampled_delta = tcp - tcp[0], points - tcp[0]
    axes[0].plot(delta[:, 0], delta[:, 1], color="#0b6fa4", label="仿真 TCP 原路径")
    axes[0].scatter(sampled_delta[:, 0], sampled_delta[:, 1], s=23, color="#e89419", label="分段参考点", zorder=3)
    axes[0].plot(tip[:, 0] - tcp[0, 0], tip[:, 1] - tcp[0, 1], "--", color="#9b4560", label="仿真导丝头（简化模型）")
    for index in (0, len(points)//2, len(points)-1):
        axes[0].annotate(str(index), sampled_delta[index, :2], xytext=(6, 5), textcoords="offset points")
    axes[0].set(xlabel="相对起点 X / mm", ylabel="相对起点 Y / mm", title="仿真世界系 XY 投影")
    axes[0].axis("equal")
    axes[0].legend(prop=font, loc="best")
    axes[1].plot(arc, delta[:, 2], color="#0b6fa4", label="仿真 TCP 高度变化")
    axes[1].scatter(stations, sampled_delta[:, 2], s=23, color="#e89419")
    axes[1].set(xlabel="原路径累计长度 / mm", ylabel="相对起点 Z / mm", title="连续路径的高度变化")
    axes[1].legend(prop=font)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle(f"左分支轨迹准备｜{len(points)-1} 段参考路径，尚不可用于实机执行", fontsize=19)
    fig.text(.05, .055, f"原仿真 {rows[-1]['step']} 步，TCP 累计 {arc[-1]:.1f} mm；按路径长度每 {args.spacing_mm:g} mm 取样，最后一段 {stations[-1]-stations[-2]:.2f} mm。", fontsize=12)
    fig.text(.05, .015, "缺少当前场景坐标对应、磁铁 TCP 和物理动作验证；本工具没有机器人或递丝接口。", fontsize=12, color="#a43b22")
    fig.tight_layout(rect=(.02, .10, .98, .94))
    fig.savefig(args.out / "left_route_preview.png")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("simulation_output/simulation_demo_20260929"))
    parser.add_argument("--spacing-mm", type=float, default=10.)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
