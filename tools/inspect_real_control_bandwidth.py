from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def numeric_suffix(path: Path, prefix: str) -> int:
    stem = path.stem
    if stem.startswith(prefix):
        return int(stem[len(prefix) :])
    return int(stem)


def read_pose_file(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = ast.literal_eval(line)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"Expected 6 pose values in {path}, got {arr.size}")
        rows.append(arr)
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.vstack(rows)


def read_label_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90, method="nearest")),
        "p95": float(np.quantile(arr, 0.95, method="nearest")),
        "p99": float(np.quantile(arr, 0.99, method="nearest")),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def unit_vectors(vectors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors), where=norms > eps)


def direction_reversal_stats(delta_pos: np.ndarray) -> dict:
    if len(delta_pos) < 2:
        return {"count": 0, "fraction": 0.0, "dot": quantiles([])}
    nonzero = np.linalg.norm(delta_pos, axis=1) > 1e-9
    pairs = nonzero[:-1] & nonzero[1:]
    if not np.any(pairs):
        return {"count": 0, "fraction": 0.0, "dot": quantiles([])}
    u = unit_vectors(delta_pos)
    dots = np.sum(u[:-1] * u[1:], axis=1)[pairs]
    reversals = dots < -0.25
    return {
        "count": int(np.sum(reversals)),
        "fraction": float(np.mean(reversals)),
        "dot": quantiles(dots),
    }


def run_lengths(labels: list[str]) -> list[dict]:
    if not labels:
        return []
    runs = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            runs.append({"label": current, "length": length})
            current = label
            length = 1
    runs.append({"label": current, "length": length})
    return runs


def label_transition_stats(labels: list[str]) -> dict:
    counts = Counter(labels)
    transitions = Counter()
    for prev, nxt in zip(labels, labels[1:]):
        if prev != nxt:
            transitions[f"{prev}->{nxt}"] += 1
    runs = run_lengths(labels)
    run_by_label: dict[str, list[int]] = {}
    for run in runs:
        run_by_label.setdefault(str(run["label"]), []).append(int(run["length"]))
    return {
        "count": len(labels),
        "counts": dict(sorted(counts.items())),
        "transition_count": int(sum(transitions.values())),
        "transitions": dict(sorted(transitions.items())),
        "transition_fraction": float(sum(transitions.values()) / max(len(labels) - 1, 1)) if len(labels) > 1 else 0.0,
        "run_length_frames": {label: quantiles(lengths) for label, lengths in sorted(run_by_label.items())},
    }


def finite_difference_stats(pose: np.ndarray, fps: float | None) -> dict:
    pos = pose[:, :3]
    rot = pose[:, 3:]
    delta = np.diff(pose, axis=0)
    delta_pos = np.diff(pos, axis=0)
    delta_rot = np.diff(rot, axis=0)
    accel_pos = np.diff(pos, n=2, axis=0)
    jerk_pos = np.diff(pos, n=3, axis=0)
    accel_rot = np.diff(rot, n=2, axis=0)
    jerk_rot = np.diff(rot, n=3, axis=0)

    result = {
        "position_step_mm_per_frame": quantiles(np.linalg.norm(delta_pos, axis=1)),
        "position_step_axis_linf_mm_per_frame": quantiles(np.max(np.abs(delta_pos), axis=1) if len(delta_pos) else []),
        "rotation_step_linf_rad_per_frame": quantiles(np.max(np.abs(delta_rot), axis=1) if len(delta_rot) else []),
        "pose_step_linf_per_frame": quantiles(np.max(np.abs(delta), axis=1) if len(delta) else []),
        "position_accel_l2_mm_per_frame2": quantiles(np.linalg.norm(accel_pos, axis=1) if len(accel_pos) else []),
        "position_jerk_l2_mm_per_frame3": quantiles(np.linalg.norm(jerk_pos, axis=1) if len(jerk_pos) else []),
        "rotation_accel_linf_rad_per_frame2": quantiles(np.max(np.abs(accel_rot), axis=1) if len(accel_rot) else []),
        "rotation_jerk_linf_rad_per_frame3": quantiles(np.max(np.abs(jerk_rot), axis=1) if len(jerk_rot) else []),
        "direction_reversal": direction_reversal_stats(delta_pos),
        "zero_position_step_fraction": float(np.mean(np.linalg.norm(delta_pos, axis=1) <= 1e-9)) if len(delta_pos) else 0.0,
    }
    if fps is not None and fps > 0:
        result["position_speed_mm_per_s"] = quantiles(np.linalg.norm(delta_pos, axis=1) * fps)
        result["position_accel_mm_per_s2"] = quantiles(np.linalg.norm(accel_pos, axis=1) * fps * fps if len(accel_pos) else [])
        result["position_jerk_mm_per_s3"] = quantiles(np.linalg.norm(jerk_pos, axis=1) * fps * fps * fps if len(jerk_pos) else [])
        result["rotation_speed_linf_rad_per_s"] = quantiles(np.max(np.abs(delta_rot), axis=1) * fps if len(delta_rot) else [])
        result["rotation_accel_linf_rad_per_s2"] = quantiles(np.max(np.abs(accel_rot), axis=1) * fps * fps if len(accel_rot) else [])
        result["rotation_jerk_linf_rad_per_s3"] = quantiles(np.max(np.abs(jerk_rot), axis=1) * fps * fps * fps if len(jerk_rot) else [])
    return result


def inspect_path(branch_dir: Path, pose_file: Path, fps: float | None) -> dict:
    path_id = numeric_suffix(pose_file, "pose")
    pose = read_pose_file(pose_file)
    labels = read_label_file(branch_dir / "piper" / f"label{path_id}.txt")
    stats = finite_difference_stats(pose, fps)
    stats.update(
        {
            "branch": branch_dir.name,
            "path_id": path_id,
            "pose_file": str(pose_file.as_posix()),
            "frames_pose": int(len(pose)),
            "frames_piper_labels": int(len(labels)),
            "piper_labels": label_transition_stats(labels),
        }
    )
    return stats


def aggregate_quantile(rows: list[dict], key: str, subkey: str = "p95") -> dict:
    values = []
    for row in rows:
        item = row
        for part in key.split("."):
            item = item.get(part, {})
        if isinstance(item, dict) and subkey in item:
            values.append(float(item[subkey]))
    return quantiles(values)


def summarize(rows: list[dict], fps: float | None) -> dict:
    label_counts = Counter()
    transitions = Counter()
    for row in rows:
        label_counts.update(row["piper_labels"]["counts"])
        transitions.update(row["piper_labels"]["transitions"])

    by_branch = {}
    for branch in sorted({row["branch"] for row in rows}):
        xs = [row for row in rows if row["branch"] == branch]
        by_branch[branch] = summarize_group(xs)

    return {
        "fps": fps,
        "units": {
            "position": "mm",
            "rotation": "rad",
            "per_frame_metrics": "computed from adjacent pose rows",
            "per_second_metrics": "only present when --fps is provided",
        },
        "total_paths": len(rows),
        "total_pose_frames": int(sum(row["frames_pose"] for row in rows)),
        "branches": by_branch,
        "all": summarize_group(rows),
        "piper_label_counts": dict(sorted(label_counts.items())),
        "piper_transitions": dict(sorted(transitions.items())),
    }


def summarize_group(rows: list[dict]) -> dict:
    return {
        "paths": len(rows),
        "pose_frames": int(sum(row["frames_pose"] for row in rows)),
        "position_step_p95_across_paths": aggregate_quantile(rows, "position_step_mm_per_frame", "p95"),
        "position_step_max_across_paths": aggregate_quantile(rows, "position_step_mm_per_frame", "max"),
        "position_accel_p95_across_paths": aggregate_quantile(rows, "position_accel_l2_mm_per_frame2", "p95"),
        "position_jerk_p95_across_paths": aggregate_quantile(rows, "position_jerk_l2_mm_per_frame3", "p95"),
        "rotation_step_linf_p95_across_paths": aggregate_quantile(rows, "rotation_step_linf_rad_per_frame", "p95"),
        "direction_reversal_fraction_across_paths": quantiles(row["direction_reversal"]["fraction"] for row in rows),
        "zero_step_fraction_across_paths": quantiles(row["zero_position_step_fraction"] for row in rows),
        "piper_transition_fraction_across_paths": quantiles(row["piper_labels"]["transition_fraction"] for row in rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "branch",
        "path_id",
        "frames_pose",
        "frames_piper_labels",
        "pos_step_median",
        "pos_step_p95",
        "pos_step_p99",
        "pos_step_max",
        "pos_accel_p95",
        "pos_jerk_p95",
        "rot_step_linf_p95",
        "direction_reversal_fraction",
        "zero_step_fraction",
        "piper_label_counts",
        "piper_transition_count",
        "piper_transition_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "branch": row["branch"],
                    "path_id": row["path_id"],
                    "frames_pose": row["frames_pose"],
                    "frames_piper_labels": row["frames_piper_labels"],
                    "pos_step_median": row["position_step_mm_per_frame"]["median"],
                    "pos_step_p95": row["position_step_mm_per_frame"]["p95"],
                    "pos_step_p99": row["position_step_mm_per_frame"]["p99"],
                    "pos_step_max": row["position_step_mm_per_frame"]["max"],
                    "pos_accel_p95": row["position_accel_l2_mm_per_frame2"]["p95"],
                    "pos_jerk_p95": row["position_jerk_l2_mm_per_frame3"]["p95"],
                    "rot_step_linf_p95": row["rotation_step_linf_rad_per_frame"]["p95"],
                    "direction_reversal_fraction": row["direction_reversal"]["fraction"],
                    "zero_step_fraction": row["zero_position_step_fraction"],
                    "piper_label_counts": json.dumps(row["piper_labels"]["counts"], ensure_ascii=False),
                    "piper_transition_count": row["piper_labels"]["transition_count"],
                    "piper_transition_fraction": row["piper_labels"]["transition_fraction"],
                }
            )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_markdown(path: Path, summary: dict) -> None:
    all_stats = summary["all"]
    lines = [
        "# Real Control Bandwidth Inspection",
        "",
        "This report treats `branchs` pose rows as real Elite/magnetic-arm pose samples, not guidewire-tip labels.",
        "Position units are millimeters because the real pose files appear to be millimeter-scale.",
        "",
        "## Overview",
        "",
        f"- Paths: {summary['total_paths']}",
        f"- Pose frames: {summary['total_pose_frames']}",
        f"- FPS assumption: {summary['fps'] if summary['fps'] else 'not provided; per-frame metrics only'}",
        f"- Piper labels: `{json.dumps(summary['piper_label_counts'], ensure_ascii=False)}`",
        f"- Piper transitions: `{json.dumps(summary['piper_transitions'], ensure_ascii=False)}`",
        "",
        "## Per-Frame Motion Anchor",
        "",
        "| Scope | Paths | Position Step P95 Median (mm/frame) | Position Step P95 Max | Position Accel P95 Median (mm/frame^2) | Position Jerk P95 Median (mm/frame^3) | Direction Reversal Fraction Median | Zero Step Fraction Median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def add_row(name: str, item: dict) -> None:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(item["paths"]),
                    fmt(item["position_step_p95_across_paths"]["median"]),
                    fmt(item["position_step_p95_across_paths"]["max"]),
                    fmt(item["position_accel_p95_across_paths"]["median"]),
                    fmt(item["position_jerk_p95_across_paths"]["median"]),
                    fmt(item["direction_reversal_fraction_across_paths"]["median"], 4),
                    fmt(item["zero_step_fraction_across_paths"]["median"], 4),
                ]
            )
            + " |"
        )

    add_row("all", all_stats)
    for branch, item in summary["branches"].items():
        add_row(branch, item)

    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- Use position-step p95 as a visible Elite/magnetic-arm motion scale anchor.",
            "- Use acceleration/jerk and direction reversals to detect whether simulator or policy motion is visibly jittery, not only whether step size is small.",
            "- Piper labels are binary and should not be silently equated with continuous simulated `piper_feed`.",
            "- If an FPS is known, re-run with `--fps` to convert frame-based step/acceleration/jerk into physical rates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real branch pose control bandwidth and Piper label switching.")
    parser.add_argument("--root", default="branchs")
    parser.add_argument("--out", default="simulation_output/real_control_bandwidth")
    parser.add_argument("--fps", type=float, default=None, help="Optional real sampling FPS for per-second speed/accel/jerk conversion.")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for branch_dir in sorted(root.glob("branch*")):
        pose_dir = branch_dir / "path"
        if not pose_dir.exists():
            continue
        pose_files = sorted(pose_dir.glob("pose*.txt"), key=lambda p: numeric_suffix(p, "pose"))
        for pose_file in pose_files:
            rows.append(inspect_path(branch_dir, pose_file, args.fps))

    summary = summarize(rows, args.fps)
    (out / "real_control_bandwidth_paths.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "real_control_bandwidth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "real_control_bandwidth_paths.csv", rows)
    write_markdown(out / "real_control_bandwidth.md", summary)

    all_stats = summary["all"]
    print(f"Inspected {summary['total_paths']} real paths from {root}")
    print(f"Total pose frames: {summary['total_pose_frames']}")
    print(
        "all: "
        f"pos_step_p95 median/max={all_stats['position_step_p95_across_paths']['median']:.3f}/"
        f"{all_stats['position_step_p95_across_paths']['max']:.3f} mm/frame, "
        f"accel_p95 median={all_stats['position_accel_p95_across_paths']['median']:.3f} mm/frame^2, "
        f"jerk_p95 median={all_stats['position_jerk_p95_across_paths']['median']:.3f} mm/frame^3"
    )
    print(f"Saved report to {out / 'real_control_bandwidth.md'}")


if __name__ == "__main__":
    main()
