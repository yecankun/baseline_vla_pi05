from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


POSE_KEYS = {
    "elirobot_pose": ("state", "elirobot_pose"),
    "elite_tool_world": ("robot_state", "elite_tool_world"),
    "elite_pose_robot_state": ("robot_state", "elite_pose"),
    "magnetic_pose": ("state", "magnetic_pose"),
    "magnetic_effective_world": ("robot_state", "magnetic_effective_world"),
    "tip_pos": ("state", "tip_pos"),
}


CSV_FIELDS = [
    "rollout",
    "case",
    "success",
    "video",
    "elite_target_step_p95",
    "elite_target_accel_p95",
    "elite_target_jerk_p95",
    "elite_executed_step_p95",
    "elite_executed_accel_p95",
    "elite_executed_jerk_p95",
    "elite_executed_direction_reversal_fraction",
    "elirobot_pose_step_p95_mm",
    "elirobot_pose_accel_p95_mm",
    "elirobot_pose_jerk_p95_mm",
    "elirobot_pose_direction_reversal_fraction",
    "tool_pose_mismatch_max_mm",
    "magnetic_pose_mismatch_max_mm",
    "tip_to_elite_median_mm",
    "tip_to_magnetic_median_mm",
    "video_frame_count",
    "video_mean_diff_p95",
    "video_mean_diff_spike_fraction",
    "diagnosis",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def vec3(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def pose_from_state(state: dict, key: str) -> np.ndarray | None:
    scope, name = POSE_KEYS[key]
    if scope == "state":
        return vec3(state.get(name))
    return vec3(state.get("robot_state", {}).get(name))


def joint_names(states: list[dict], actions: list[dict]) -> list[str]:
    for row in actions:
        elite = row.get("action", {}).get("elite_joints")
        if isinstance(elite, dict) and elite:
            return list(elite.keys())
    for row in states:
        elite = row.get("robot_state", {}).get("elite_joints")
        if isinstance(elite, dict) and elite:
            return list(elite.keys())
    return []


def joint_vec(mapping: dict | None, names: list[str]) -> np.ndarray | None:
    if not names or not isinstance(mapping, dict):
        return None
    return np.asarray([float(mapping.get(name, 0.0)) for name in names], dtype=np.float64)


def action_by_step(actions: list[dict]) -> dict[int, dict]:
    result = {}
    for row in actions:
        if "step" in row:
            result[int(row["step"])] = row.get("action", {})
    return result


def direction_reversal_fraction(delta: np.ndarray, eps: float) -> float:
    if len(delta) < 2:
        return 0.0
    norms = np.linalg.norm(delta, axis=1)
    mask = (norms[:-1] > eps) & (norms[1:] > eps)
    if not np.any(mask):
        return 0.0
    unit = delta / np.maximum(norms[:, None], eps)
    dots = np.sum(unit[:-1] * unit[1:], axis=1)[mask]
    return float(np.mean(dots < -0.25)) if dots.size else 0.0


def sequence_stats(values: list[np.ndarray], scale: float = 1.0, eps: float = 1e-9) -> dict:
    if len(values) < 2:
        empty = quantiles([])
        return {
            "frames": len(values),
            "step": empty,
            "accel": empty,
            "jerk": empty,
            "direction_reversal_fraction": 0.0,
        }
    arr = np.vstack(values) * scale
    delta = np.diff(arr, axis=0)
    step = np.linalg.norm(delta, axis=1)
    accel = np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1) if len(arr) >= 3 else []
    jerk = np.linalg.norm(np.diff(arr, n=3, axis=0), axis=1) if len(arr) >= 4 else []
    return {
        "frames": int(len(values)),
        "step": quantiles(step),
        "accel": quantiles(accel),
        "jerk": quantiles(jerk),
        "direction_reversal_fraction": direction_reversal_fraction(delta, eps * scale),
    }


def joint_linf_stats(values: list[np.ndarray], eps: float = 1e-9) -> dict:
    if len(values) < 2:
        empty = quantiles([])
        return {
            "frames": len(values),
            "step_linf": empty,
            "accel_linf": empty,
            "jerk_linf": empty,
            "direction_reversal_fraction": 0.0,
            "per_joint_step_linf_p95": {},
            "per_joint_accel_linf_p95": {},
            "per_joint_jerk_linf_p95": {},
        }
    arr = np.vstack(values)
    delta = np.diff(arr, axis=0)
    accel = np.diff(arr, n=2, axis=0) if len(arr) >= 3 else np.zeros((0, arr.shape[1]))
    jerk = np.diff(arr, n=3, axis=0) if len(arr) >= 4 else np.zeros((0, arr.shape[1]))
    return {
        "frames": int(len(values)),
        "step_linf": quantiles(np.max(np.abs(delta), axis=1)),
        "accel_linf": quantiles(np.max(np.abs(accel), axis=1) if len(accel) else []),
        "jerk_linf": quantiles(np.max(np.abs(jerk), axis=1) if len(jerk) else []),
        "direction_reversal_fraction": direction_reversal_fraction(delta, eps),
        "per_joint_step_linf_p95": {str(i): float(np.quantile(np.abs(delta[:, i]), 0.95, method="nearest")) for i in range(delta.shape[1])},
        "per_joint_accel_linf_p95": {
            str(i): float(np.quantile(np.abs(accel[:, i]), 0.95, method="nearest")) for i in range(accel.shape[1])
        }
        if len(accel)
        else {},
        "per_joint_jerk_linf_p95": {str(i): float(np.quantile(np.abs(jerk[:, i]), 0.95, method="nearest")) for i in range(jerk.shape[1])}
        if len(jerk)
        else {},
    }


def paired_distance_stats(values_a: list[np.ndarray | None], values_b: list[np.ndarray | None], scale: float = 1.0) -> dict:
    distances = []
    for a, b in zip(values_a, values_b):
        if a is not None and b is not None:
            distances.append(float(np.linalg.norm(a - b) * scale))
    return quantiles(distances)


def video_diff_stats(path: Path, sample_every: int) -> dict:
    if not path.exists():
        return {"exists": False, "frame_count": 0, "fps": 0.0, "mean_abs_diff": quantiles([]), "spike_fraction": 0.0}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": True, "opened": False, "frame_count": 0, "fps": 0.0, "mean_abs_diff": quantiles([]), "spike_fraction": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    diffs = []
    prev = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % max(sample_every, 1) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))))
            prev = gray
        index += 1
    cap.release()
    diff_stats = quantiles(diffs)
    threshold = diff_stats["median"] + 3.0 * max(diff_stats["p95"] - diff_stats["median"], 1e-6)
    spike_fraction = float(np.mean(np.asarray(diffs) > threshold)) if diffs else 0.0
    return {
        "exists": True,
        "opened": True,
        "frame_count": frame_count,
        "fps": fps,
        "sample_every": sample_every,
        "mean_abs_diff": diff_stats,
        "spike_threshold": threshold,
        "spike_fraction": spike_fraction,
    }


def candidate_case_dirs(paths: Iterable[Path]) -> list[Path]:
    result = []
    for path in paths:
        if (path / "states.jsonl").exists():
            result.append(path)
            continue
        for child in ("left", "right"):
            case = path / child
            if (case / "states.jsonl").exists():
                result.append(case)
    return result


def top_dict_values(mapping: dict[str, float], limit: int = 3) -> list[dict]:
    return [
        {"index": key, "value": value}
        for key, value in sorted(mapping.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
    ]


def diagnose(case_summary: dict, args: argparse.Namespace) -> list[str]:
    diagnosis = []
    target = case_summary["joint"]["target"]
    executed = case_summary["joint"]["executed"]
    pose = case_summary["poses"]["elirobot_pose"]
    tool_mismatch = case_summary["consistency"]["elirobot_vs_elite_tool_world_mm"]
    magnetic_mismatch = case_summary["consistency"]["magnetic_pose_vs_magnetic_effective_world_mm"]
    video = case_summary["video"]

    if target["step_linf"]["p95"] > args.target_step_p95_threshold:
        diagnosis.append("policy_target_jitter_remains")
    if executed["accel_linf"]["p95"] > args.executed_accel_p95_threshold:
        diagnosis.append("executed_joint_accel_high")
    if executed["jerk_linf"]["p95"] > args.executed_jerk_p95_threshold:
        diagnosis.append("executed_joint_jerk_high")
    if executed["direction_reversal_fraction"] > args.direction_reversal_threshold:
        diagnosis.append("executed_joint_direction_reversal_high")
    if pose["step"]["p95"] <= args.pose_step_p95_good_mm and pose["jerk"]["p95"] <= args.pose_jerk_p95_good_mm:
        diagnosis.append("logged_tool_pose_smooth")
    if tool_mismatch["max"] > args.pose_mismatch_threshold_mm:
        diagnosis.append("elirobot_pose_mismatch_with_robot_state_tool")
    if magnetic_mismatch["max"] > args.pose_mismatch_threshold_mm:
        diagnosis.append("magnetic_pose_mismatch_with_robot_state")
    if video.get("exists") and video.get("mean_abs_diff", {}).get("p95", 0.0) > args.video_diff_p95_threshold:
        diagnosis.append("video_frame_difference_spikes")
    if "logged_tool_pose_smooth" in diagnosis and not any(item.startswith("executed_joint") for item in diagnosis):
        diagnosis.append("suspect_rendering_or_link_visual_mapping")
    if not diagnosis:
        diagnosis.append("no_clear_jitter_source_from_logs")
    return diagnosis


def analyze_case(case_dir: Path, args: argparse.Namespace) -> dict:
    states = read_jsonl(case_dir / "states.jsonl")
    actions = read_jsonl(case_dir / "actions.jsonl") if (case_dir / "actions.jsonl").exists() else []
    meta = read_json(case_dir / "meta.json")
    names = joint_names(states, actions)
    actions_for_step = action_by_step(actions)

    target_joints = []
    executed_joints = []
    for state in states:
        step = int(state.get("step", len(executed_joints)))
        action = actions_for_step.get(step, {})
        target = joint_vec(action.get("elite_joints"), names)
        executed = joint_vec(state.get("robot_state", {}).get("elite_joints"), names)
        if target is not None:
            target_joints.append(target)
        if executed is not None:
            executed_joints.append(executed)

    pose_values = {}
    for key in POSE_KEYS:
        values = [pose_from_state(state, key) for state in states]
        pose_values[key] = values

    tip = pose_values["tip_pos"]
    elite = pose_values["elirobot_pose"]
    magnetic = pose_values["magnetic_pose"]
    video_path = Path(str(meta.get("video") or (case_dir / "rollout.mp4")))

    summary = {
        "case": str(case_dir),
        "task": meta.get("task", case_dir.name),
        "success": meta.get("success"),
        "meta": meta,
        "frames": len(states),
        "joint_names": names,
        "joint": {
            "target": joint_linf_stats(target_joints),
            "executed": joint_linf_stats(executed_joints),
            "top_executed_step_joints": [],
            "top_executed_accel_joints": [],
            "top_executed_jerk_joints": [],
        },
        "poses": {},
        "distances_mm": {
            "tip_to_elite": paired_distance_stats(tip, elite, scale=1000.0),
            "tip_to_magnetic": paired_distance_stats(tip, magnetic, scale=1000.0),
        },
        "consistency": {
            "elirobot_vs_elite_tool_world_mm": paired_distance_stats(
                pose_values["elirobot_pose"], pose_values["elite_tool_world"], scale=1000.0
            ),
            "elirobot_vs_elite_pose_robot_state_mm": paired_distance_stats(
                pose_values["elirobot_pose"], pose_values["elite_pose_robot_state"], scale=1000.0
            ),
            "magnetic_pose_vs_magnetic_effective_world_mm": paired_distance_stats(
                pose_values["magnetic_pose"], pose_values["magnetic_effective_world"], scale=1000.0
            ),
        },
        "video": video_diff_stats(video_path, args.video_sample_every),
    }

    for key, values in pose_values.items():
        summary["poses"][key] = sequence_stats([value for value in values if value is not None], scale=1000.0, eps=args.pose_eps_m)

    executed = summary["joint"]["executed"]
    summary["joint"]["top_executed_step_joints"] = top_dict_values(executed["per_joint_step_linf_p95"])
    summary["joint"]["top_executed_accel_joints"] = top_dict_values(executed["per_joint_accel_linf_p95"])
    summary["joint"]["top_executed_jerk_joints"] = top_dict_values(executed["per_joint_jerk_linf_p95"])
    summary["diagnosis"] = diagnose(summary, args)
    return summary


def aggregate_case_summaries(cases: list[dict]) -> dict:
    return {
        "case_count": len(cases),
        "successes": [case["success"] for case in cases],
        "elite_target_step_p95": quantiles(case["joint"]["target"]["step_linf"]["p95"] for case in cases),
        "elite_executed_step_p95": quantiles(case["joint"]["executed"]["step_linf"]["p95"] for case in cases),
        "elite_executed_accel_p95": quantiles(case["joint"]["executed"]["accel_linf"]["p95"] for case in cases),
        "elite_executed_jerk_p95": quantiles(case["joint"]["executed"]["jerk_linf"]["p95"] for case in cases),
        "elirobot_pose_step_p95_mm": quantiles(case["poses"]["elirobot_pose"]["step"]["p95"] for case in cases),
        "elirobot_pose_accel_p95_mm": quantiles(case["poses"]["elirobot_pose"]["accel"]["p95"] for case in cases),
        "elirobot_pose_jerk_p95_mm": quantiles(case["poses"]["elirobot_pose"]["jerk"]["p95"] for case in cases),
        "elirobot_direction_reversal_fraction": quantiles(case["poses"]["elirobot_pose"]["direction_reversal_fraction"] for case in cases),
        "tip_to_elite_median_mm": quantiles(case["distances_mm"]["tip_to_elite"]["median"] for case in cases),
        "tip_to_magnetic_median_mm": quantiles(case["distances_mm"]["tip_to_magnetic"]["median"] for case in cases),
    }


def csv_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        rows.append(
            {
                "rollout": str(Path(case["case"]).parent),
                "case": case["case"],
                "success": case["success"],
                "video": case["meta"].get("video"),
                "elite_target_step_p95": case["joint"]["target"]["step_linf"]["p95"],
                "elite_target_accel_p95": case["joint"]["target"]["accel_linf"]["p95"],
                "elite_target_jerk_p95": case["joint"]["target"]["jerk_linf"]["p95"],
                "elite_executed_step_p95": case["joint"]["executed"]["step_linf"]["p95"],
                "elite_executed_accel_p95": case["joint"]["executed"]["accel_linf"]["p95"],
                "elite_executed_jerk_p95": case["joint"]["executed"]["jerk_linf"]["p95"],
                "elite_executed_direction_reversal_fraction": case["joint"]["executed"]["direction_reversal_fraction"],
                "elirobot_pose_step_p95_mm": case["poses"]["elirobot_pose"]["step"]["p95"],
                "elirobot_pose_accel_p95_mm": case["poses"]["elirobot_pose"]["accel"]["p95"],
                "elirobot_pose_jerk_p95_mm": case["poses"]["elirobot_pose"]["jerk"]["p95"],
                "elirobot_pose_direction_reversal_fraction": case["poses"]["elirobot_pose"]["direction_reversal_fraction"],
                "tool_pose_mismatch_max_mm": case["consistency"]["elirobot_vs_elite_tool_world_mm"]["max"],
                "magnetic_pose_mismatch_max_mm": case["consistency"]["magnetic_pose_vs_magnetic_effective_world_mm"]["max"],
                "tip_to_elite_median_mm": case["distances_mm"]["tip_to_elite"]["median"],
                "tip_to_magnetic_median_mm": case["distances_mm"]["tip_to_magnetic"]["median"],
                "video_frame_count": case["video"]["frame_count"],
                "video_mean_diff_p95": case["video"]["mean_abs_diff"]["p95"],
                "video_mean_diff_spike_fraction": case["video"]["spike_fraction"],
                "diagnosis": ";".join(case["diagnosis"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(result: dict) -> str:
    lines = [
        "# Elite Visual Jitter Source Analysis",
        "",
        "Generated by `tools/analyze_elite_visual_jitter_source.py`.",
        "",
        "This report separates policy target jitter, executed joint jitter, logged tool-pose jitter, pose consistency, and coarse video frame-difference spikes.",
        "",
        "## Case Summary",
        "",
    ]
    rows = []
    for case in result["cases"]:
        rows.append(
            [
                Path(case["case"]).name,
                str(case["success"]),
                fmt(case["joint"]["target"]["step_linf"]["p95"], 5),
                fmt(case["joint"]["executed"]["step_linf"]["p95"], 5),
                fmt(case["joint"]["executed"]["accel_linf"]["p95"], 5),
                fmt(case["joint"]["executed"]["jerk_linf"]["p95"], 5),
                fmt(case["poses"]["elirobot_pose"]["step"]["p95"], 3),
                fmt(case["poses"]["elirobot_pose"]["jerk"]["p95"], 3),
                fmt(case["poses"]["elirobot_pose"]["direction_reversal_fraction"], 4),
                fmt(case["distances_mm"]["tip_to_elite"]["median"], 3),
                ", ".join(case["diagnosis"]),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Case",
                "Success",
                "Target Step P95",
                "Exec Step P95",
                "Exec Accel P95",
                "Exec Jerk P95",
                "Tool Step P95 mm",
                "Tool Jerk P95 mm",
                "Tool DirRev",
                "Tip-Elite Med mm",
                "Diagnosis",
            ],
            rows,
        )
    )
    lines.extend(["", "## Pose Consistency", ""])
    consistency_rows = []
    for case in result["cases"]:
        consistency_rows.append(
            [
                Path(case["case"]).name,
                fmt(case["consistency"]["elirobot_vs_elite_tool_world_mm"]["max"], 6),
                fmt(case["consistency"]["elirobot_vs_elite_pose_robot_state_mm"]["max"], 6),
                fmt(case["consistency"]["magnetic_pose_vs_magnetic_effective_world_mm"]["max"], 6),
                fmt(case["video"]["mean_abs_diff"]["p95"], 3),
                fmt(case["video"]["spike_fraction"], 4),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Case",
                "elirobot-tool max mm",
                "elirobot-robotstate max mm",
                "magnetic mismatch max mm",
                "Video diff p95",
                "Video spike frac",
            ],
            consistency_rows,
        )
    )
    lines.extend(["", "## Top Executed Joint Contributors", ""])
    contributor_rows = []
    for case in result["cases"]:
        contributor_rows.append(
            [
                Path(case["case"]).name,
                json.dumps(case["joint"]["top_executed_step_joints"], ensure_ascii=False),
                json.dumps(case["joint"]["top_executed_accel_joints"], ensure_ascii=False),
                json.dumps(case["joint"]["top_executed_jerk_joints"], ensure_ascii=False),
            ]
        )
    lines.append(markdown_table(["Case", "Step P95", "Accel P95", "Jerk P95"], contributor_rows))
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- If logged tool pose is smooth but video still jitters, suspect rendering cadence, robot link visual mapping, or intermediate-link motion not captured by tool pose.",
            "- If executed joint accel/jerk or direction reversals remain high, improve the execution controller before blaming rendering.",
            "- Pose consistency mismatches near zero mean `elirobot_pose` and `robot_state.elite_tool_world` are reporting the same tool position.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether Elite visual jitter comes from targets, executed joints, logged poses, or video/rendering.")
    parser.add_argument("rollouts", nargs="+", help="Rollout directories, or left/right case directories.")
    parser.add_argument("--out", default="docs/_elite_visual_jitter_source_check")
    parser.add_argument("--video-sample-every", type=int, default=1)
    parser.add_argument("--pose-eps-m", type=float, default=1e-9)
    parser.add_argument("--target-step-p95-threshold", type=float, default=0.045)
    parser.add_argument("--executed-accel-p95-threshold", type=float, default=0.0025)
    parser.add_argument("--executed-jerk-p95-threshold", type=float, default=0.0025)
    parser.add_argument("--direction-reversal-threshold", type=float, default=0.05)
    parser.add_argument("--pose-step-p95-good-mm", type=float, default=5.5)
    parser.add_argument("--pose-jerk-p95-good-mm", type=float, default=3.0)
    parser.add_argument("--pose-mismatch-threshold-mm", type=float, default=0.1)
    parser.add_argument("--video-diff-p95-threshold", type=float, default=20.0)
    args = parser.parse_args()

    cases = [analyze_case(case_dir, args) for case_dir in candidate_case_dirs(Path(path) for path in args.rollouts)]
    if not cases:
        raise FileNotFoundError("No rollout cases found. Expected states.jsonl under each case or left/right subdirectories.")
    result = {
        "settings": vars(args),
        "aggregate": aggregate_case_summaries(cases),
        "cases": cases,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "elite_visual_jitter_source.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "elite_visual_jitter_source.csv", csv_rows(cases))
    (out / "elite_visual_jitter_source.md").write_text(build_markdown(result), encoding="utf-8")

    print(f"Saved JSON to {out / 'elite_visual_jitter_source.json'}")
    print(f"Saved CSV to {out / 'elite_visual_jitter_source.csv'}")
    print(f"Saved Markdown to {out / 'elite_visual_jitter_source.md'}")
    for case in cases:
        print(
            f"{case['case']}: target_step_p95={case['joint']['target']['step_linf']['p95']:.5f}, "
            f"exec_accel_p95={case['joint']['executed']['accel_linf']['p95']:.5f}, "
            f"tool_jerk_p95={case['poses']['elirobot_pose']['jerk']['p95']:.3f} mm, "
            f"diagnosis={', '.join(case['diagnosis'])}"
        )


if __name__ == "__main__":
    main()
