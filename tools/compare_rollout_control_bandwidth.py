from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


POSE_KEYS = {
    "elirobot_pose": {
        "semantic": "Visible Elite tool pose in simulation.",
        "state_key": "elirobot_pose",
        "robot_state_key": "elite_tool_world",
    },
    "magnetic_pose": {
        "semantic": "Effective magnetic point in simulation.",
        "state_key": "magnetic_pose",
        "robot_state_key": "magnetic_effective_world",
    },
    "tip_pos": {
        "semantic": "Simulated guidewire tip; no direct real counterpart.",
        "state_key": "tip_pos",
        "robot_state_key": None,
    },
}


CSV_FIELDS = [
    "rollout",
    "pose_key",
    "cases",
    "frames",
    "step_p95_median_mm",
    "step_p95_ratio_to_real",
    "step_max_median_mm",
    "accel_p95_median_mm",
    "accel_p95_ratio_to_real",
    "jerk_p95_median_mm",
    "jerk_p95_ratio_to_real",
    "direction_reversal_fraction_median",
    "zero_step_fraction_median",
    "near_zero_step_fraction_median",
    "tip_to_elite_tool_median_mm",
    "tip_to_magnetic_median_mm",
    "findings",
]


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)


def quantiles(values: Iterable[float]) -> dict:
    arr = finite(values)
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


def unit_vectors(vectors: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors), where=norms > eps)


def direction_reversal_stats(delta_pos_mm: np.ndarray, eps_mm: float) -> dict:
    if len(delta_pos_mm) < 2:
        return {"count": 0, "fraction": 0.0, "dot": quantiles([])}
    nonzero = np.linalg.norm(delta_pos_mm, axis=1) > eps_mm
    pairs = nonzero[:-1] & nonzero[1:]
    if not np.any(pairs):
        return {"count": 0, "fraction": 0.0, "dot": quantiles([])}
    u = unit_vectors(delta_pos_mm, eps_mm)
    dots = np.sum(u[:-1] * u[1:], axis=1)[pairs]
    reversals = dots < -0.25
    return {
        "count": int(np.sum(reversals)),
        "fraction": float(np.mean(reversals)),
        "dot": quantiles(dots),
    }


def vec3(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def state_pose(state: dict, pose_key: str) -> np.ndarray | None:
    spec = POSE_KEYS[pose_key]
    value = vec3(state.get(spec["state_key"]))
    if value is not None:
        return value
    robot_key = spec["robot_state_key"]
    if robot_key:
        return vec3(state.get("robot_state", {}).get(robot_key))
    return None


def joint_vec(mapping: dict | None) -> np.ndarray | None:
    if not isinstance(mapping, dict) or not mapping:
        return None
    return np.asarray([float(mapping[key]) for key in mapping.keys()], dtype=np.float64)


def step_linf(values: list[np.ndarray]) -> np.ndarray:
    if len(values) < 2:
        return np.asarray([], dtype=np.float64)
    arr = np.vstack(values)
    return np.max(np.abs(np.diff(arr, axis=0)), axis=1)


def trajectory_motion_stats(points_m: list[np.ndarray], args: argparse.Namespace) -> dict:
    if len(points_m) < 2:
        empty = quantiles([])
        return {
            "frames": len(points_m),
            "step_mm_per_frame": empty,
            "accel_mm_per_frame2": empty,
            "jerk_mm_per_frame3": empty,
            "direction_reversal": {"count": 0, "fraction": 0.0, "dot": empty},
            "zero_step_fraction": 0.0,
            "near_zero_step_fraction": 0.0,
        }

    points_mm = np.vstack(points_m) * 1000.0
    delta = np.diff(points_mm, axis=0)
    step = np.linalg.norm(delta, axis=1)
    accel = np.linalg.norm(np.diff(points_mm, n=2, axis=0), axis=1) if len(points_mm) >= 3 else np.asarray([])
    jerk = np.linalg.norm(np.diff(points_mm, n=3, axis=0), axis=1) if len(points_mm) >= 4 else np.asarray([])
    return {
        "frames": int(len(points_m)),
        "step_mm_per_frame": quantiles(step),
        "accel_mm_per_frame2": quantiles(accel),
        "jerk_mm_per_frame3": quantiles(jerk),
        "direction_reversal": direction_reversal_stats(delta, args.direction_eps_mm),
        "zero_step_fraction": float(np.mean(step <= args.zero_step_threshold_mm)) if step.size else 0.0,
        "near_zero_step_fraction": float(np.mean(step <= args.near_zero_step_threshold_mm)) if step.size else 0.0,
    }


def distance_stats(states: list[dict]) -> dict:
    tip_to_elite = []
    tip_to_magnetic = []
    elite_to_magnetic = []
    for state in states:
        tip = state_pose(state, "tip_pos")
        elite = state_pose(state, "elirobot_pose")
        magnetic = state_pose(state, "magnetic_pose")
        if tip is not None and elite is not None:
            tip_to_elite.append(float(np.linalg.norm(elite - tip) * 1000.0))
        if tip is not None and magnetic is not None:
            tip_to_magnetic.append(float(np.linalg.norm(magnetic - tip) * 1000.0))
        if elite is not None and magnetic is not None:
            elite_to_magnetic.append(float(np.linalg.norm(elite - magnetic) * 1000.0))
    return {
        "tip_to_elite_tool_mm": quantiles(tip_to_elite),
        "tip_to_magnetic_mm": quantiles(tip_to_magnetic),
        "elite_tool_to_magnetic_mm": quantiles(elite_to_magnetic),
    }


def action_by_step(actions: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for row in actions:
        if "step" in row:
            result[int(row["step"])] = row.get("action", {})
    return result


def joint_motion_stats(states: list[dict], actions: list[dict]) -> dict:
    executed = []
    for row in states:
        joints = joint_vec(row.get("robot_state", {}).get("elite_joints"))
        if joints is not None:
            executed.append(joints)

    targets = []
    for row in actions:
        joints = joint_vec(row.get("action", {}).get("elite_joints"))
        if joints is not None:
            targets.append(joints)

    return {
        "elite_executed_joint_step_linf_rad": quantiles(step_linf(executed)),
        "elite_target_joint_step_linf_rad": quantiles(step_linf(targets)),
    }


def candidate_case_dirs(path: Path) -> list[Path]:
    if (path / "states.jsonl").exists():
        return [path]
    cases = []
    for child in sorted(path.iterdir()) if path.exists() else []:
        if child.is_dir() and (child / "states.jsonl").exists():
            cases.append(child)
    return cases


def load_case(case_dir: Path, args: argparse.Namespace) -> dict:
    states = read_jsonl(case_dir / "states.jsonl")
    actions = read_jsonl(case_dir / "actions.jsonl") if (case_dir / "actions.jsonl").exists() else []
    meta = read_json(case_dir / "meta.json") if (case_dir / "meta.json").exists() else {}
    pose_stats = {}
    for pose_key in POSE_KEYS:
        points = [state_pose(state, pose_key) for state in states]
        pose_stats[pose_key] = trajectory_motion_stats([point for point in points if point is not None], args)
    return {
        "case": str(case_dir),
        "task": str(meta.get("task", case_dir.name)),
        "success": bool(meta.get("success", False)) if "success" in meta else None,
        "frames": len(states),
        "pose_stats": pose_stats,
        "distance_stats": distance_stats(states),
        "joint_stats": joint_motion_stats(states, actions),
    }


def aggregate_case_quantile(cases: list[dict], pose_key: str, metric: str, subkey: str) -> dict:
    values = []
    for case in cases:
        item = case["pose_stats"][pose_key][metric]
        if isinstance(item, dict) and subkey in item:
            values.append(float(item[subkey]))
    return quantiles(values)


def aggregate_scalar_metric(cases: list[dict], pose_key: str, metric: str) -> dict:
    return quantiles(case["pose_stats"][pose_key][metric] for case in cases)


def aggregate_distance(cases: list[dict], metric: str, subkey: str) -> dict:
    return quantiles(case["distance_stats"][metric][subkey] for case in cases)


def aggregate_joint(cases: list[dict], metric: str, subkey: str) -> dict:
    return quantiles(case["joint_stats"][metric][subkey] for case in cases)


def real_anchor(real_summary: dict) -> dict:
    all_stats = real_summary["all"]
    return {
        "source": "real_control_bandwidth",
        "semantic": "Real Elite/magnetic-arm pose; not guidewire path.",
        "position_step_p95_median_mm": float(all_stats["position_step_p95_across_paths"]["median"]),
        "position_step_p95_max_mm": float(all_stats["position_step_p95_across_paths"]["max"]),
        "position_step_max_median_mm": float(all_stats["position_step_max_across_paths"]["median"]),
        "position_accel_p95_median_mm": float(all_stats["position_accel_p95_across_paths"]["median"]),
        "position_jerk_p95_median_mm": float(all_stats["position_jerk_p95_across_paths"]["median"]),
        "direction_reversal_fraction_median": float(all_stats["direction_reversal_fraction_across_paths"]["median"]),
        "zero_step_fraction_median": float(all_stats["zero_step_fraction_across_paths"]["median"]),
        "paths": int(all_stats["paths"]),
        "pose_frames": int(all_stats["pose_frames"]),
    }


def ratio(value: float, anchor: float) -> float:
    if anchor <= 0:
        return 0.0
    return float(value / anchor)


def pose_findings(summary: dict, pose_key: str, real: dict, args: argparse.Namespace) -> list[str]:
    pose = summary["pose_aggregate"][pose_key]
    findings = []
    step_ratio = pose["ratios_to_real"]["step_p95_median"]
    accel_ratio = pose["ratios_to_real"]["accel_p95_median"]
    jerk_ratio = pose["ratios_to_real"]["jerk_p95_median"]
    reversal = pose["direction_reversal_fraction_across_cases"]["median"]
    near_zero = pose["near_zero_step_fraction_across_cases"]["median"]

    if step_ratio > args.high_ratio:
        findings.append("step_too_high")
    elif step_ratio < args.low_ratio:
        findings.append("step_below_real_anchor")
    else:
        findings.append("step_close_to_real_anchor")

    if accel_ratio > args.high_ratio:
        findings.append("accel_too_high")
    if jerk_ratio > args.high_ratio:
        findings.append("jerk_too_high")
    if reversal > args.direction_reversal_threshold:
        findings.append("direction_reversal_high")
    if pose_key == "elirobot_pose" and near_zero < args.min_near_zero_fraction:
        findings.append("few_hold_like_frames")

    if pose_key == "elirobot_pose":
        tip_elite = summary["distance_aggregate"]["tip_to_elite_tool_mm"]["median_across_cases"]["median"]
        tip_magnetic = summary["distance_aggregate"]["tip_to_magnetic_mm"]["median_across_cases"]["median"]
        if tip_elite > args.tip_elite_drift_threshold_mm and tip_magnetic < args.tip_magnetic_good_threshold_mm:
            findings.append("visible_elite_tool_drift")

    return findings


def analyze_rollout(rollout_path: Path, real: dict, args: argparse.Namespace) -> dict:
    cases = [load_case(case_dir, args) for case_dir in candidate_case_dirs(rollout_path)]
    if not cases:
        raise FileNotFoundError(f"No rollout cases found under {rollout_path}")

    pose_aggregate = {}
    for pose_key in POSE_KEYS:
        step_p95 = aggregate_case_quantile(cases, pose_key, "step_mm_per_frame", "p95")
        step_max = aggregate_case_quantile(cases, pose_key, "step_mm_per_frame", "max")
        accel_p95 = aggregate_case_quantile(cases, pose_key, "accel_mm_per_frame2", "p95")
        jerk_p95 = aggregate_case_quantile(cases, pose_key, "jerk_mm_per_frame3", "p95")
        pose_aggregate[pose_key] = {
            "semantic": POSE_KEYS[pose_key]["semantic"],
            "step_p95_across_cases_mm": step_p95,
            "step_max_across_cases_mm": step_max,
            "accel_p95_across_cases_mm_per_frame2": accel_p95,
            "jerk_p95_across_cases_mm_per_frame3": jerk_p95,
            "direction_reversal_fraction_across_cases": quantiles(
                case["pose_stats"][pose_key]["direction_reversal"]["fraction"] for case in cases
            ),
            "zero_step_fraction_across_cases": aggregate_scalar_metric(cases, pose_key, "zero_step_fraction"),
            "near_zero_step_fraction_across_cases": aggregate_scalar_metric(cases, pose_key, "near_zero_step_fraction"),
            "ratios_to_real": {
                "step_p95_median": ratio(step_p95["median"], real["position_step_p95_median_mm"]),
                "step_max_median": ratio(step_max["median"], real["position_step_max_median_mm"]),
                "accel_p95_median": ratio(accel_p95["median"], real["position_accel_p95_median_mm"]),
                "jerk_p95_median": ratio(jerk_p95["median"], real["position_jerk_p95_median_mm"]),
            },
        }

    distance_aggregate = {
        "tip_to_elite_tool_mm": {
            "median_across_cases": aggregate_distance(cases, "tip_to_elite_tool_mm", "median"),
            "p95_across_cases": aggregate_distance(cases, "tip_to_elite_tool_mm", "p95"),
        },
        "tip_to_magnetic_mm": {
            "median_across_cases": aggregate_distance(cases, "tip_to_magnetic_mm", "median"),
            "p95_across_cases": aggregate_distance(cases, "tip_to_magnetic_mm", "p95"),
        },
        "elite_tool_to_magnetic_mm": {
            "median_across_cases": aggregate_distance(cases, "elite_tool_to_magnetic_mm", "median"),
            "p95_across_cases": aggregate_distance(cases, "elite_tool_to_magnetic_mm", "p95"),
        },
    }
    joint_aggregate = {
        "elite_executed_joint_step_linf_rad": {
            "p95_across_cases": aggregate_joint(cases, "elite_executed_joint_step_linf_rad", "p95"),
            "max_across_cases": aggregate_joint(cases, "elite_executed_joint_step_linf_rad", "max"),
        },
        "elite_target_joint_step_linf_rad": {
            "p95_across_cases": aggregate_joint(cases, "elite_target_joint_step_linf_rad", "p95"),
            "max_across_cases": aggregate_joint(cases, "elite_target_joint_step_linf_rad", "max"),
        },
    }
    summary = {
        "rollout": str(rollout_path),
        "cases": cases,
        "case_count": len(cases),
        "frames": int(sum(case["frames"] for case in cases)),
        "tasks": [case["task"] for case in cases],
        "successes": [case["success"] for case in cases],
        "pose_aggregate": pose_aggregate,
        "distance_aggregate": distance_aggregate,
        "joint_aggregate": joint_aggregate,
    }
    summary["findings"] = {pose_key: pose_findings(summary, pose_key, real, args) for pose_key in POSE_KEYS}
    return summary


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def csv_rows(result: dict) -> list[dict]:
    rows = []
    for rollout in result["rollouts"]:
        for pose_key in POSE_KEYS:
            pose = rollout["pose_aggregate"][pose_key]
            rows.append(
                {
                    "rollout": rollout["rollout"],
                    "pose_key": pose_key,
                    "cases": rollout["case_count"],
                    "frames": rollout["frames"],
                    "step_p95_median_mm": pose["step_p95_across_cases_mm"]["median"],
                    "step_p95_ratio_to_real": pose["ratios_to_real"]["step_p95_median"],
                    "step_max_median_mm": pose["step_max_across_cases_mm"]["median"],
                    "accel_p95_median_mm": pose["accel_p95_across_cases_mm_per_frame2"]["median"],
                    "accel_p95_ratio_to_real": pose["ratios_to_real"]["accel_p95_median"],
                    "jerk_p95_median_mm": pose["jerk_p95_across_cases_mm_per_frame3"]["median"],
                    "jerk_p95_ratio_to_real": pose["ratios_to_real"]["jerk_p95_median"],
                    "direction_reversal_fraction_median": pose["direction_reversal_fraction_across_cases"]["median"],
                    "zero_step_fraction_median": pose["zero_step_fraction_across_cases"]["median"],
                    "near_zero_step_fraction_median": pose["near_zero_step_fraction_across_cases"]["median"],
                    "tip_to_elite_tool_median_mm": rollout["distance_aggregate"]["tip_to_elite_tool_mm"]["median_across_cases"]["median"],
                    "tip_to_magnetic_median_mm": rollout["distance_aggregate"]["tip_to_magnetic_mm"]["median_across_cases"]["median"],
                    "findings": ";".join(rollout["findings"][pose_key]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(result: dict) -> str:
    real = result["real_anchor"]
    lines = [
        "# Rollout Control Bandwidth Comparison",
        "",
        "Generated by `tools/compare_rollout_control_bandwidth.py`.",
        "",
        "Important semantic constraint: real `branchs` pose is treated as Elite/magnetic-arm motion, not guidewire-tip truth.",
        "Simulated `tip_pos` remains a sim-only guidewire diagnostic.",
        "",
        "## Real Anchor",
        "",
        markdown_table(
            [
                "Source",
                "Pose Frames",
                "Step P95 Median (mm/frame)",
                "Step P95 Max",
                "Accel P95 Median (mm/frame^2)",
                "Jerk P95 Median (mm/frame^3)",
                "Direction Reversal Median",
                "Zero Step Median",
            ],
            [
                [
                    "real_control_bandwidth",
                    str(real["pose_frames"]),
                    fmt(real["position_step_p95_median_mm"]),
                    fmt(real["position_step_p95_max_mm"]),
                    fmt(real["position_accel_p95_median_mm"]),
                    fmt(real["position_jerk_p95_median_mm"]),
                    fmt(real["direction_reversal_fraction_median"], 4),
                    fmt(real["zero_step_fraction_median"], 4),
                ]
            ],
        ),
        "",
        "## Rollout Motion Comparison",
        "",
    ]
    motion_rows = []
    for rollout in result["rollouts"]:
        name = Path(rollout["rollout"]).name
        for pose_key in ("elirobot_pose", "magnetic_pose", "tip_pos"):
            pose = rollout["pose_aggregate"][pose_key]
            motion_rows.append(
                [
                    name,
                    pose_key,
                    str(rollout["case_count"]),
                    fmt(pose["step_p95_across_cases_mm"]["median"]),
                    fmt(pose["ratios_to_real"]["step_p95_median"], 2),
                    fmt(pose["accel_p95_across_cases_mm_per_frame2"]["median"]),
                    fmt(pose["ratios_to_real"]["accel_p95_median"], 2),
                    fmt(pose["jerk_p95_across_cases_mm_per_frame3"]["median"]),
                    fmt(pose["ratios_to_real"]["jerk_p95_median"], 2),
                    fmt(pose["direction_reversal_fraction_across_cases"]["median"], 4),
                    fmt(pose["near_zero_step_fraction_across_cases"]["median"], 4),
                    ", ".join(rollout["findings"][pose_key]),
                ]
            )
    lines.append(
        markdown_table(
            [
                "Rollout",
                "Pose",
                "Cases",
                "Step P95 Med",
                "Step Ratio",
                "Accel P95 Med",
                "Accel Ratio",
                "Jerk P95 Med",
                "Jerk Ratio",
                "DirRev Med",
                "NearZero Med",
                "Findings",
            ],
            motion_rows,
        )
    )
    lines.extend(["", "## Drift And Joint Summary", ""])
    drift_rows = []
    for rollout in result["rollouts"]:
        name = Path(rollout["rollout"]).name
        dist = rollout["distance_aggregate"]
        joints = rollout["joint_aggregate"]
        drift_rows.append(
            [
                name,
                ", ".join(str(item) for item in rollout["tasks"]),
                ", ".join(str(item) for item in rollout["successes"]),
                fmt(dist["tip_to_elite_tool_mm"]["median_across_cases"]["median"]),
                fmt(dist["tip_to_magnetic_mm"]["median_across_cases"]["median"]),
                fmt(dist["elite_tool_to_magnetic_mm"]["median_across_cases"]["median"]),
                fmt(joints["elite_target_joint_step_linf_rad"]["p95_across_cases"]["median"], 5),
                fmt(joints["elite_executed_joint_step_linf_rad"]["p95_across_cases"]["median"], 5),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Rollout",
                "Tasks",
                "Success",
                "Tip-Elite Med (mm)",
                "Tip-Magnetic Med (mm)",
                "Elite-Magnetic Med (mm)",
                "Target Joint P95",
                "Executed Joint P95",
            ],
            drift_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `Step Ratio`, `Accel Ratio`, and `Jerk Ratio` compare the rollout median-across-cases value to the real bandwidth anchor.",
            "- A rollout can match step p95 but still be suspicious if acceleration, jerk, or direction reversals are high.",
            "- `tip_pos` is included only as a sim-internal guidewire diagnostic and should not be interpreted as real-aligned guidewire ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MuJoCo rollout motion bandwidth against real branch pose bandwidth.")
    parser.add_argument("rollouts", nargs="+", help="Rollout directories, or case directories containing states.jsonl.")
    parser.add_argument("--real-summary", default="simulation_output/real_control_bandwidth/real_control_bandwidth_summary.json")
    parser.add_argument("--out", default="simulation_output/rollout_control_bandwidth_comparison")
    parser.add_argument("--zero-step-threshold-mm", type=float, default=1e-6)
    parser.add_argument("--near-zero-step-threshold-mm", type=float, default=0.05)
    parser.add_argument("--direction-eps-mm", type=float, default=1e-6)
    parser.add_argument("--low-ratio", type=float, default=0.75)
    parser.add_argument("--high-ratio", type=float, default=1.5)
    parser.add_argument("--direction-reversal-threshold", type=float, default=0.05)
    parser.add_argument("--min-near-zero-fraction", type=float, default=0.05)
    parser.add_argument("--tip-elite-drift-threshold-mm", type=float, default=80.0)
    parser.add_argument("--tip-magnetic-good-threshold-mm", type=float, default=10.0)
    args = parser.parse_args()

    real = real_anchor(read_json(Path(args.real_summary)))
    rollouts = [analyze_rollout(Path(path), real, args) for path in args.rollouts]
    result = {
        "real_summary": str(Path(args.real_summary)),
        "real_anchor": real,
        "settings": {
            "zero_step_threshold_mm": args.zero_step_threshold_mm,
            "near_zero_step_threshold_mm": args.near_zero_step_threshold_mm,
            "direction_eps_mm": args.direction_eps_mm,
            "low_ratio": args.low_ratio,
            "high_ratio": args.high_ratio,
            "direction_reversal_threshold": args.direction_reversal_threshold,
            "tip_elite_drift_threshold_mm": args.tip_elite_drift_threshold_mm,
            "tip_magnetic_good_threshold_mm": args.tip_magnetic_good_threshold_mm,
        },
        "rollouts": rollouts,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout_control_bandwidth_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "rollout_control_bandwidth_comparison.csv", csv_rows(result))
    (out / "rollout_control_bandwidth_comparison.md").write_text(build_markdown(result), encoding="utf-8")

    print(f"Saved JSON to {out / 'rollout_control_bandwidth_comparison.json'}")
    print(f"Saved CSV to {out / 'rollout_control_bandwidth_comparison.csv'}")
    print(f"Saved Markdown to {out / 'rollout_control_bandwidth_comparison.md'}")
    for rollout in rollouts:
        elite = rollout["pose_aggregate"]["elirobot_pose"]
        print(
            f"{Path(rollout['rollout']).name}: "
            f"elirobot step_p95={elite['step_p95_across_cases_mm']['median']:.3f} mm "
            f"({elite['ratios_to_real']['step_p95_median']:.2f}x real), "
            f"accel_p95={elite['accel_p95_across_cases_mm_per_frame2']['median']:.3f} mm/frame^2 "
            f"({elite['ratios_to_real']['accel_p95_median']:.2f}x), "
            f"jerk_p95={elite['jerk_p95_across_cases_mm_per_frame3']['median']:.3f} mm/frame^3 "
            f"({elite['ratios_to_real']['jerk_p95_median']:.2f}x), "
            f"findings={', '.join(rollout['findings']['elirobot_pose'])}"
        )


if __name__ == "__main__":
    main()
