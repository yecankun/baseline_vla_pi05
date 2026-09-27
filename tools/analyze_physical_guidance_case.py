from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    "piper_insertion_length",
    "elite_reference_error",
    "heading_path_angle_deg",
    "magnetic_path_angle_deg",
    "tip_x",
    "tip_y",
    "tip_z",
]


def vec(values, default=None) -> np.ndarray:
    if values is None:
        values = default if default is not None else [0.0, 0.0, 0.0]
    return np.asarray(values, dtype=np.float32).reshape(3)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    seen = set()
    unique = []
    for path in result:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def build_rows(states: list[dict]) -> list[dict]:
    rows = []
    prev_progress = None
    prev_step = None
    for state in states:
        step = int(state.get("step", len(rows)))
        progress = float(state.get("path_progress", 0.0))
        if prev_progress is None:
            progress_rate = 0.0
        else:
            dt = max(float(step - prev_step), 1.0)
            progress_rate = (progress - prev_progress) / dt
        prev_progress = progress
        prev_step = step

        tip = vec(state.get("tip_pos"))
        heading = vec(state.get("heading"))
        tangent = vec(state.get("path_tangent"), default=[0.0, 1.0, 0.0])
        magnetic = vec(state.get("magnetic_pose"), default=tip)
        elite = vec(state.get("elirobot_pose"), default=magnetic)
        elite_ref = vec(state.get("elite_reference_pose"), default=elite)

        rows.append(
            {
                "step": step,
                "path_progress": progress,
                "progress_rate": float(progress_rate),
                "distance_to_target": float(state.get("distance_to_target", float("nan"))),
                "distance_to_wall": float(state.get("distance_to_wall", float("nan"))),
                "contact_strength": float(state.get("contact_strength", float("nan"))),
                "piper_insertion_length": float(state.get("piper_insertion_length", 0.0)),
                "elite_reference_error": float(np.linalg.norm(elite - elite_ref)),
                "heading_path_angle_deg": angle_deg(heading, tangent),
                "magnetic_path_angle_deg": angle_deg(magnetic - tip, tangent),
                "tip_x": float(tip[0]),
                "tip_y": float(tip[1]),
                "tip_z": float(tip[2]),
            }
        )
    return rows


def finite_stats(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if len(arr) == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(np.min(arr)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}


def summarize(rows: list[dict], meta: dict | None) -> dict:
    if not rows:
        return {"error": "no rows"}
    n = len(rows)
    tail_start = max(0, int(n * 0.75))
    head_end = max(1, int(n * 0.25))
    progress_gain = rows[-1]["path_progress"] - rows[0]["path_progress"]
    rates = [row["progress_rate"] for row in rows]
    tail_rates = [row["progress_rate"] for row in rows[tail_start:]]
    head_rates = [row["progress_rate"] for row in rows[:head_end]]
    contact = [row["contact_strength"] for row in rows]
    wall = [row["distance_to_wall"] for row in rows]
    heading_angle = [row["heading_path_angle_deg"] for row in rows]
    magnetic_angle = [row["magnetic_path_angle_deg"] for row in rows]
    elite_error = [row["elite_reference_error"] for row in rows]

    tail_rate_mean = float(np.mean(tail_rates)) if tail_rates else 0.0
    head_rate_mean = float(np.mean(head_rates)) if head_rates else 0.0
    stalled = tail_rate_mean < 0.35 * max(head_rate_mean, 1e-6)
    high_contact = max(contact) > 0.65 if contact else False
    poor_alignment = np.nanmean(heading_angle[tail_start:]) > 35.0 if heading_angle else False

    likely_causes = []
    if stalled and not high_contact and not poor_alignment:
        likely_causes.append("low_effective_push_or_overdamped")
    if stalled and high_contact:
        likely_causes.append("wall_contact_or_boundary_constraint")
    if poor_alignment:
        likely_causes.append("tip_direction_misaligned_with_path")
    if np.nanmean(magnetic_angle[tail_start:]) > 45.0:
        likely_causes.append("magnetic_pose_not_aligned_with_path")
    if not likely_causes:
        likely_causes.append("no_obvious_single_cause")

    return {
        "meta": meta or {},
        "steps": int(rows[-1]["step"]),
        "progress_gain": float(progress_gain),
        "first_quarter_progress_rate_mean": head_rate_mean,
        "last_quarter_progress_rate_mean": tail_rate_mean,
        "stalled_in_last_quarter": bool(stalled),
        "distance_to_wall": finite_stats(wall),
        "contact_strength": finite_stats(contact),
        "heading_path_angle_deg": finite_stats(heading_angle),
        "magnetic_path_angle_deg": finite_stats(magnetic_angle),
        "elite_reference_error": finite_stats(elite_error),
        "likely_causes": likely_causes,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def maybe_plot(case_dir: Path, rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional diagnostic dependency
        print(f"matplotlib unavailable, skip plots: {exc}")
        return []

    steps = [row["step"] for row in rows]
    progress = [row["path_progress"] for row in rows]
    rate = [row["progress_rate"] for row in rows]
    contact = [row["contact_strength"] for row in rows]
    wall = [row["distance_to_wall"] for row in rows]
    piper = [row["piper_insertion_length"] for row in rows]
    elite_error = [row["elite_reference_error"] for row in rows]
    heading_angle = [row["heading_path_angle_deg"] for row in rows]
    magnetic_angle = [row["magnetic_path_angle_deg"] for row in rows]

    plots = []

    def save_plot(name: str, title: str, series: list[tuple[str, list[float]]], ylabel: str) -> None:
        plt.figure(figsize=(9, 4.8))
        for label, values in series:
            plt.plot(steps, values, label=label)
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

    save_plot("progress.png", "Path Progress", [("progress", progress), ("rate", rate)], "progress / step")
    save_plot("contact_wall.png", "Contact And Wall Clearance", [("contact", contact), ("wall", wall)], "value")
    save_plot("angles.png", "Direction Alignment", [("heading-path", heading_angle), ("magnetic-path", magnetic_angle)], "degrees")
    save_plot("piper_elite.png", "Piper And Elite Error", [("piper insertion", piper), ("elite ref error", elite_error)], "meters")
    return plots


def analyze_case(case_dir: Path, make_plots: bool) -> dict:
    states_path = case_dir / "states.jsonl"
    meta_path = case_dir / "meta.json"
    states = read_jsonl(states_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rows = build_rows(states)
    write_csv(case_dir / "diagnostics.csv", rows)
    summary = summarize(rows, meta)
    if make_plots:
        summary["plots"] = maybe_plot(case_dir, rows)
    (case_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a physical MuJoCo guidance rollout case.")
    parser.add_argument("cases", nargs="+", help="Case directories, or a root directory containing left/right subdirectories.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    case_dirs = candidate_case_dirs(Path(p) for p in args.cases)
    if not case_dirs:
        raise FileNotFoundError("No states.jsonl found in provided cases or left/right subdirectories.")

    summaries = {}
    for case_dir in case_dirs:
        summary = analyze_case(case_dir, make_plots=not args.no_plots)
        summaries[str(case_dir)] = summary
        causes = ", ".join(summary.get("likely_causes", []))
        print(
            f"{case_dir}: gain={summary['progress_gain']:.2f}, "
            f"tail_rate={summary['last_quarter_progress_rate_mean']:.4f}, "
            f"contact_max={summary['contact_strength']['max']}, "
            f"wall_min={summary['distance_to_wall']['min']}, causes={causes}"
        )

    if len(case_dirs) > 1:
        root = case_dirs[0].parent
        (root / "diagnostics_all_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
