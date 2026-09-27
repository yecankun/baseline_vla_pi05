from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


CSV_FIELDS = [
    "step",
    "path_progress",
    "progress_rate",
    "distance_to_target",
    "distance_to_wall",
    "contact_strength",
    "piper_feed",
    "piper_insertion_length",
    "elite_target_joint_step_linf",
    "elite_target_joint_step_l2",
    "elite_executed_joint_step_linf",
    "elite_executed_joint_step_l2",
    "elite_target_to_executed_joint_l2",
    "elite_tool_step",
    "magnetic_step",
    "tip_step",
    "tip_to_elite_tool",
    "tip_to_magnetic",
    "elite_tool_to_magnetic",
    "elite_height_above_tip",
    "magnetic_height_above_tip",
    "elite_forward_offset",
    "magnetic_forward_offset",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def vec3(value, default=None) -> np.ndarray:
    if value is None:
        value = default if default is not None else [0.0, 0.0, 0.0]
    return np.asarray(value, dtype=np.float64).reshape(3)


def joint_vec(mapping: dict | None, names: list[str]) -> np.ndarray:
    mapping = mapping or {}
    return np.asarray([float(mapping.get(name, 0.0)) for name in names], dtype=np.float64)


def finite_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)


def stats(values: Iterable[float]) -> dict:
    arr = finite_array(values)
    if len(arr) == 0:
        return {"min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def candidate_case_dirs(paths: Iterable[Path]) -> list[Path]:
    cases: list[Path] = []
    for path in paths:
        if (path / "states.jsonl").exists() and (path / "actions.jsonl").exists():
            cases.append(path)
            continue
        for child in ("left", "right"):
            case = path / child
            if (case / "states.jsonl").exists() and (case / "actions.jsonl").exists():
                cases.append(case)
    seen = set()
    unique = []
    for case in cases:
        resolved = case.resolve()
        if resolved not in seen:
            unique.append(case)
            seen.add(resolved)
    return unique


def infer_elite_names(actions: list[dict], states: list[dict]) -> list[str]:
    for row in actions:
        elite = row.get("action", {}).get("elite_joints")
        if isinstance(elite, dict) and elite:
            return list(elite.keys())
    for row in states:
        elite = row.get("robot_state", {}).get("elite_joints")
        if isinstance(elite, dict) and elite:
            return list(elite.keys())
    return []


def action_by_step(actions: list[dict]) -> dict[int, dict]:
    result = {}
    for row in actions:
        if "step" in row:
            result[int(row["step"])] = row.get("action", {})
    return result


def build_rows(states: list[dict], actions: list[dict]) -> list[dict]:
    names = infer_elite_names(actions, states)
    actions_for_step = action_by_step(actions)
    rows: list[dict] = []

    prev_progress = None
    prev_step = None
    prev_action_elite = None
    prev_state_elite = None
    prev_elite_tool = None
    prev_magnetic = None
    prev_tip = None

    for state in states:
        step = int(state.get("step", len(rows)))
        progress = float(state.get("path_progress", 0.0))
        dt = max(float(step - prev_step), 1.0) if prev_step is not None else 1.0
        progress_rate = 0.0 if prev_progress is None else (progress - prev_progress) / dt

        action = actions_for_step.get(step, {})
        has_elite_action = isinstance(action.get("elite_joints"), dict)
        if names and has_elite_action:
            target_elite = joint_vec(action.get("elite_joints"), names)
        elif prev_action_elite is not None:
            target_elite = prev_action_elite.copy()
        else:
            target_elite = np.asarray([], dtype=np.float64)
        state_elite = joint_vec(state.get("robot_state", {}).get("elite_joints"), names) if names else np.asarray([], dtype=np.float64)

        if prev_action_elite is None or len(target_elite) == 0 or not has_elite_action:
            target_step = np.zeros_like(target_elite)
        else:
            target_step = target_elite - prev_action_elite

        if prev_state_elite is None or len(state_elite) == 0:
            executed_step = np.zeros_like(state_elite)
        else:
            executed_step = state_elite - prev_state_elite

        tip = vec3(state.get("tip_pos"))
        elite_tool = vec3(state.get("robot_state", {}).get("elite_tool_world"), default=state.get("elirobot_pose"))
        magnetic = vec3(state.get("magnetic_pose"), default=elite_tool)
        tangent = vec3(state.get("path_tangent"), default=[0.0, 1.0, 0.0])
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-8:
            tangent = tangent / tangent_norm

        elite_delta = elite_tool - tip
        magnetic_delta = magnetic - tip

        row = {
            "step": step,
            "path_progress": progress,
            "progress_rate": float(progress_rate),
            "distance_to_target": float(state.get("distance_to_target", float("nan"))),
            "distance_to_wall": float(state.get("distance_to_wall", float("nan"))),
            "contact_strength": float(state.get("contact_strength", float("nan"))),
            "piper_feed": float(action.get("piper_feed", float("nan"))),
            "piper_insertion_length": float(state.get("piper_insertion_length", 0.0)),
            "elite_target_joint_step_linf": float(np.max(np.abs(target_step))) if len(target_step) else 0.0,
            "elite_target_joint_step_l2": float(np.linalg.norm(target_step)) if len(target_step) else 0.0,
            "elite_executed_joint_step_linf": float(np.max(np.abs(executed_step))) if len(executed_step) else 0.0,
            "elite_executed_joint_step_l2": float(np.linalg.norm(executed_step)) if len(executed_step) else 0.0,
            "elite_target_to_executed_joint_l2": float(np.linalg.norm(target_elite - state_elite)) if len(target_elite) and len(state_elite) else 0.0,
            "elite_tool_step": 0.0 if prev_elite_tool is None else float(np.linalg.norm(elite_tool - prev_elite_tool)),
            "magnetic_step": 0.0 if prev_magnetic is None else float(np.linalg.norm(magnetic - prev_magnetic)),
            "tip_step": 0.0 if prev_tip is None else float(np.linalg.norm(tip - prev_tip)),
            "tip_to_elite_tool": float(np.linalg.norm(elite_delta)),
            "tip_to_magnetic": float(np.linalg.norm(magnetic_delta)),
            "elite_tool_to_magnetic": float(np.linalg.norm(elite_tool - magnetic)),
            "elite_height_above_tip": float(elite_delta[2]),
            "magnetic_height_above_tip": float(magnetic_delta[2]),
            "elite_forward_offset": float(np.dot(elite_delta, tangent)),
            "magnetic_forward_offset": float(np.dot(magnetic_delta, tangent)),
        }
        rows.append(row)

        prev_progress = progress
        prev_step = step
        if has_elite_action:
            prev_action_elite = target_elite.copy()
        prev_state_elite = state_elite.copy()
        prev_elite_tool = elite_tool.copy()
        prev_magnetic = magnetic.copy()
        prev_tip = tip.copy()

    return rows


def worst_rows(rows: list[dict], field: str, n: int = 8) -> list[dict]:
    ordered = sorted(rows, key=lambda row: float(row.get(field, 0.0)), reverse=True)
    return [
        {
            "step": int(row["step"]),
            field: float(row.get(field, 0.0)),
            "path_progress": float(row.get("path_progress", 0.0)),
            "distance_to_target": float(row.get("distance_to_target", float("nan"))),
            "contact_strength": float(row.get("contact_strength", float("nan"))),
            "piper_insertion_length": float(row.get("piper_insertion_length", 0.0)),
        }
        for row in ordered[:n]
    ]


def likely_causes(summary: dict, args: argparse.Namespace) -> list[str]:
    causes = []
    target_max = summary["metrics"]["elite_target_joint_step_linf"]["max"] or 0.0
    executed_pose_max = summary["metrics"]["elite_tool_step"]["max"] or 0.0
    magnetic_step_max = summary["metrics"]["magnetic_step"]["max"] or 0.0
    tip_to_elite_med = summary["metrics"]["tip_to_elite_tool"]["median"] or 0.0
    tip_to_magnetic_med = summary["metrics"]["tip_to_magnetic"]["median"] or 0.0

    if target_max > args.target_jump_threshold:
        causes.append("policy_elite_joint_target_spikes")
    if executed_pose_max > args.elite_pose_jump_threshold:
        causes.append("visible_elite_pose_jumps")
    if executed_pose_max > args.elite_pose_jump_threshold and magnetic_step_max < args.magnetic_step_threshold:
        causes.append("robot_visual_jump_larger_than_magnetic_field_motion")
    if tip_to_elite_med > args.tip_elite_distance_threshold and tip_to_magnetic_med < args.tip_magnetic_distance_threshold:
        causes.append("magnetic_effective_pose_tracks_tip_but_robot_tool_is_far")
    if not causes:
        causes.append("no_large_elite_jump_detected")
    return causes


def summarize(case_dir: Path, rows: list[dict], meta: dict, args: argparse.Namespace) -> dict:
    metrics = {field: stats(row[field] for row in rows) for field in CSV_FIELDS if field != "step"}
    summary = {
        "case": str(case_dir.resolve()),
        "meta": meta,
        "n_rows": len(rows),
        "start_step": int(rows[0]["step"]) if rows else None,
        "end_step": int(rows[-1]["step"]) if rows else None,
        "progress_gain": float(rows[-1]["path_progress"] - rows[0]["path_progress"]) if len(rows) >= 2 else 0.0,
        "metrics": metrics,
        "worst_elite_target_joint_jumps": worst_rows(rows, "elite_target_joint_step_linf"),
        "worst_elite_executed_joint_jumps": worst_rows(rows, "elite_executed_joint_step_linf"),
        "worst_elite_tool_steps": worst_rows(rows, "elite_tool_step"),
        "likely_causes": [],
        "thresholds": {
            "target_jump_threshold": args.target_jump_threshold,
            "elite_pose_jump_threshold": args.elite_pose_jump_threshold,
            "magnetic_step_threshold": args.magnetic_step_threshold,
            "tip_elite_distance_threshold": args.tip_elite_distance_threshold,
            "tip_magnetic_distance_threshold": args.tip_magnetic_distance_threshold,
        },
    }
    summary["likely_causes"] = likely_causes(summary, args)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def maybe_plot(case_dir: Path, rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable, skip plots: {exc}")
        return []

    steps = [row["step"] for row in rows]
    plots = []

    def save(name: str, title: str, series: list[tuple[str, str]], ylabel: str) -> None:
        plt.figure(figsize=(10, 5.2))
        for label, field in series:
            plt.plot(steps, [row[field] for row in rows], label=label)
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = case_dir / name
        plt.savefig(out, dpi=150)
        plt.close()
        plots.append(str(out.resolve()))

    save(
        "elite_joint_jumps.png",
        "Elite Joint Target vs Executed Joint Step",
        [
            ("target joint step L-inf", "elite_target_joint_step_linf"),
            ("executed joint step L-inf", "elite_executed_joint_step_linf"),
            ("target-executed joint L2", "elite_target_to_executed_joint_l2"),
        ],
        "rad",
    )
    save(
        "elite_pose_following.png",
        "Elite / Magnetic / Tip Motion",
        [
            ("elite tool step", "elite_tool_step"),
            ("magnetic step", "magnetic_step"),
            ("tip step", "tip_step"),
        ],
        "m / step",
    )
    save(
        "elite_tip_distances.png",
        "Distances To Tip",
        [
            ("tip to elite tool", "tip_to_elite_tool"),
            ("tip to magnetic", "tip_to_magnetic"),
            ("elite tool to magnetic", "elite_tool_to_magnetic"),
        ],
        "m",
    )
    save(
        "task_progress_contact.png",
        "Task Progress And Contact",
        [
            ("path progress", "path_progress"),
            ("contact strength", "contact_strength"),
            ("distance to wall", "distance_to_wall"),
            ("piper insertion", "piper_insertion_length"),
        ],
        "mixed units",
    )
    return plots


def analyze_case(case_dir: Path, args: argparse.Namespace) -> dict:
    states = read_jsonl(case_dir / "states.jsonl")
    actions = read_jsonl(case_dir / "actions.jsonl")
    meta = read_json(case_dir / "meta.json")
    rows = build_rows(states, actions)
    write_csv(case_dir / "elite_diagnostics.csv", rows)
    summary = summarize(case_dir, rows, meta, args)
    if not args.no_plots:
        summary["plots"] = maybe_plot(case_dir, rows)
    (case_dir / "elite_diagnostics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def print_summary(case_dir: Path, summary: dict) -> None:
    metrics = summary["metrics"]
    target = metrics["elite_target_joint_step_linf"]
    executed = metrics["elite_executed_joint_step_linf"]
    pose = metrics["elite_tool_step"]
    magnetic = metrics["magnetic_step"]
    tip_elite = metrics["tip_to_elite_tool"]
    tip_magnetic = metrics["tip_to_magnetic"]
    print(
        f"{case_dir}: gain={summary['progress_gain']:.2f}, "
        f"target_jump_max={target['max']:.4f}, target_jump_p95={target['p95']:.4f}, "
        f"executed_joint_jump_max={executed['max']:.4f}, elite_pose_step_max={pose['max']:.4f}, "
        f"mag_step_max={magnetic['max']:.4f}, tip_elite_med={tip_elite['median']:.4f}, "
        f"tip_mag_med={tip_magnetic['median']:.4f}, causes={', '.join(summary['likely_causes'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Elite joint/pose smoothness in MuJoCo rollout outputs.")
    parser.add_argument("cases", nargs="+", help="Case directories, or root directories containing left/right subdirectories.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--target-jump-threshold", type=float, default=0.05)
    parser.add_argument("--elite-pose-jump-threshold", type=float, default=0.025)
    parser.add_argument("--magnetic-step-threshold", type=float, default=0.006)
    parser.add_argument("--tip-elite-distance-threshold", type=float, default=0.08)
    parser.add_argument("--tip-magnetic-distance-threshold", type=float, default=0.01)
    args = parser.parse_args()

    case_dirs = candidate_case_dirs(Path(path) for path in args.cases)
    if not case_dirs:
        raise FileNotFoundError("No rollout case found. Expected states.jsonl/actions.jsonl or left/right subdirectories.")

    combined = {}
    for case_dir in case_dirs:
        summary = analyze_case(case_dir, args)
        combined[str(case_dir.resolve())] = summary
        print_summary(case_dir, summary)

    if len(case_dirs) > 1:
        root = Path(args.cases[0])
        out = root / "elite_diagnostics_summary_all.json"
        out.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved combined summary to {out}")


if __name__ == "__main__":
    main()
