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
    "contact_strength",
    "distance_to_wall",
    "local_radius",
    "wall_clearance_fraction",
    "piper_feed",
    "piper_command_label",
    "policy_piper_command_label",
    "tip_to_magnetic",
    "magnet_wall_pull",
    "magnet_tangent_pull",
    "tip_step",
    "tip_wall_velocity",
    "estimated_image_distance_px",
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "tip_estimator_confidence",
    "estimated_wall_margin",
    "estimated_wall_margin_fraction",
    "route_estimator_confidence",
    "estimated_magnet_wall_pull",
    "estimated_wall_side_risk",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
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


def finite(values: Iterable[float]) -> np.ndarray:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=np.float64)


def stats(values: Iterable[float]) -> dict:
    arr = finite(values)
    if len(arr) == 0:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def corr(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    pairs = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if float(np.std(arr[:, 0])) < 1e-12 or float(np.std(arr[:, 1])) < 1e-12:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def action_by_step(actions: list[dict]) -> dict[int, dict]:
    return {int(row.get("step", -1)): row.get("action", {}) for row in actions}


def build_rows(states: list[dict], actions: list[dict]) -> list[dict]:
    action_map = action_by_step(actions)
    rows: list[dict] = []
    prev_tip = None
    prev_wall_distance = None
    for state in states:
        step = int(state.get("step", len(rows)))
        action = action_map.get(step, {})
        tip = vec3(state.get("tip_pos"))
        magnetic = vec3(state.get("magnetic_pose"), default=state.get("robot_state", {}).get("magnetic_effective_world"))
        normal = vec3(state.get("contact_normal"))
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-8:
            normal = normal / normal_norm
        else:
            normal = np.zeros(3, dtype=np.float64)
        tangent = vec3(state.get("path_tangent"), default=[0.0, 1.0, 0.0])
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-8:
            tangent = tangent / tangent_norm
        else:
            tangent = np.zeros(3, dtype=np.float64)

        magnet_vec = magnetic - tip
        wall_distance = float(state.get("distance_to_wall", float("nan")))
        local_radius = float(state.get("local_radius", float("nan")))
        tip_delta = np.zeros(3, dtype=np.float64) if prev_tip is None else tip - prev_tip
        wall_delta = 0.0 if prev_wall_distance is None else wall_distance - prev_wall_distance

        row = {
            "step": step,
            "path_progress": float(state.get("path_progress", float("nan"))),
            "contact_strength": float(state.get("contact_strength", float("nan"))),
            "distance_to_wall": wall_distance,
            "local_radius": local_radius,
            "wall_clearance_fraction": wall_distance / local_radius if math.isfinite(local_radius) and abs(local_radius) > 1e-12 else float("nan"),
            "piper_feed": float(action.get("piper_feed", float("nan"))),
            "piper_command_label": str(action.get("piper_command_label", "")),
            "policy_piper_command_label": str(action.get("policy_piper_command_label", "")),
            "tip_to_magnetic": float(np.linalg.norm(magnet_vec)),
            "magnet_wall_pull": float(np.dot(magnet_vec, normal)),
            "magnet_tangent_pull": float(np.dot(magnet_vec, tangent)),
            "tip_step": float(np.linalg.norm(tip_delta)),
            "tip_wall_velocity": float(wall_delta),
            "estimated_image_distance_px": state.get("estimated_image_distance_px"),
            "estimated_contact_flag": state.get("estimated_contact_flag"),
            "contact_estimator_confidence": state.get("contact_estimator_confidence"),
            "tip_estimator_confidence": state.get("tip_estimator_confidence"),
            "estimated_wall_margin": state.get("estimated_wall_margin"),
            "estimated_wall_margin_fraction": state.get("estimated_wall_margin_fraction"),
            "route_estimator_confidence": state.get("route_estimator_confidence"),
            "estimated_magnet_wall_pull": state.get("estimated_magnet_wall_pull"),
            "estimated_wall_side_risk": state.get("estimated_wall_side_risk"),
        }
        rows.append(row)
        prev_tip = tip
        prev_wall_distance = wall_distance
    return rows


def summarize_rows(rows: list[dict], high_contact_threshold: float) -> dict:
    high = [row for row in rows if float(row["contact_strength"]) >= high_contact_threshold]
    low = [row for row in rows if float(row["contact_strength"]) < high_contact_threshold]
    first_high = high[0] if high else None
    return {
        "n_rows": len(rows),
        "high_contact_threshold": float(high_contact_threshold),
        "high_contact_count": len(high),
        "high_contact_fraction": float(len(high) / max(len(rows), 1)),
        "first_high_contact": None
        if first_high is None
        else {
            "step": int(first_high["step"]),
            "path_progress": float(first_high["path_progress"]),
            "contact_strength": float(first_high["contact_strength"]),
            "estimated_image_distance_px": first_high.get("estimated_image_distance_px"),
            "magnet_wall_pull": float(first_high["magnet_wall_pull"]),
            "estimated_magnet_wall_pull": first_high.get("estimated_magnet_wall_pull"),
            "estimated_wall_side_risk": first_high.get("estimated_wall_side_risk"),
            "tip_to_magnetic": float(first_high["tip_to_magnetic"]),
        },
        "contact_strength": stats(row["contact_strength"] for row in rows),
        "distance_to_wall": stats(row["distance_to_wall"] for row in rows),
        "wall_clearance_fraction": stats(row["wall_clearance_fraction"] for row in rows),
        "tip_to_magnetic": stats(row["tip_to_magnetic"] for row in rows),
        "magnet_wall_pull": stats(row["magnet_wall_pull"] for row in rows),
        "magnet_wall_pull_high_contact": stats(row["magnet_wall_pull"] for row in high),
        "magnet_wall_pull_low_contact": stats(row["magnet_wall_pull"] for row in low),
        "magnet_tangent_pull": stats(row["magnet_tangent_pull"] for row in rows),
        "estimated_image_distance_px": stats(row["estimated_image_distance_px"] for row in rows),
        "estimated_image_distance_px_high_contact": stats(row["estimated_image_distance_px"] for row in high),
        "estimated_image_distance_px_low_contact": stats(row["estimated_image_distance_px"] for row in low),
        "contact_estimator_confidence": stats(row["contact_estimator_confidence"] for row in rows),
        "route_estimator_confidence": stats(row["route_estimator_confidence"] for row in rows),
        "estimated_wall_margin": stats(row["estimated_wall_margin"] for row in rows),
        "estimated_wall_margin_high_contact": stats(row["estimated_wall_margin"] for row in high),
        "estimated_wall_margin_low_contact": stats(row["estimated_wall_margin"] for row in low),
        "estimated_magnet_wall_pull": stats(row["estimated_magnet_wall_pull"] for row in rows),
        "estimated_magnet_wall_pull_high_contact": stats(row["estimated_magnet_wall_pull"] for row in high),
        "estimated_magnet_wall_pull_low_contact": stats(row["estimated_magnet_wall_pull"] for row in low),
        "estimated_wall_side_risk": stats(row["estimated_wall_side_risk"] for row in rows),
        "estimated_wall_side_risk_high_contact": stats(row["estimated_wall_side_risk"] for row in high),
        "estimated_wall_side_risk_low_contact": stats(row["estimated_wall_side_risk"] for row in low),
        "contact_flag_positive_count": int(sum(1 for row in rows if int(row.get("estimated_contact_flag") or 0) > 0)),
        "corr_contact_vs_estimated_image_distance_px": corr(
            [row["contact_strength"] for row in rows],
            [row["estimated_image_distance_px"] for row in rows],
        ),
        "corr_contact_vs_magnet_wall_pull": corr(
            [row["contact_strength"] for row in rows],
            [row["magnet_wall_pull"] for row in rows],
        ),
        "corr_contact_vs_tip_to_magnetic": corr(
            [row["contact_strength"] for row in rows],
            [row["tip_to_magnetic"] for row in rows],
        ),
        "corr_contact_vs_estimated_wall_margin": corr(
            [row["contact_strength"] for row in rows],
            [row["estimated_wall_margin"] for row in rows],
        ),
        "corr_contact_vs_estimated_magnet_wall_pull": corr(
            [row["contact_strength"] for row in rows],
            [row["estimated_magnet_wall_pull"] for row in rows],
        ),
        "corr_contact_vs_estimated_wall_side_risk": corr(
            [row["contact_strength"] for row in rows],
            [row["estimated_wall_side_risk"] for row in rows],
        ),
    }


def candidate_case_dirs(paths: Iterable[Path]) -> list[Path]:
    cases: list[Path] = []
    for path in paths:
        if (path / "states.jsonl").exists() and (path / "actions.jsonl").exists():
            cases.append(path)
            continue
        for task in ("left", "right"):
            case = path / task
            if (case / "states.jsonl").exists() and (case / "actions.jsonl").exists():
                cases.append(case)
    seen = set()
    unique = []
    for case in cases:
        resolved = case.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(case)
    return unique


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose wall-contact causes in a MuJoCo rollout.")
    parser.add_argument("paths", nargs="+", type=Path, help="Rollout root(s) or task case directories.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to each rollout root.")
    parser.add_argument("--high-contact-threshold", type=float, default=0.35)
    args = parser.parse_args()

    cases = candidate_case_dirs(args.paths)
    if not cases:
        raise SystemExit("No rollout case directories found.")

    combined = {}
    for case in cases:
        states = read_jsonl(case / "states.jsonl")
        actions = read_jsonl(case / "actions.jsonl")
        rows = build_rows(states, actions)
        summary = summarize_rows(rows, args.high_contact_threshold)
        out_dir = args.out or case.parent
        if len(cases) == 1 and args.out is not None:
            case_out = out_dir
        else:
            case_out = out_dir / case.name
        case_out.mkdir(parents=True, exist_ok=True)
        write_csv(case_out / "wall_contact_diagnostics.csv", rows)
        (case_out / "wall_contact_diagnostics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        combined[str(case.resolve())] = summary
        first = summary["first_high_contact"]
        first_text = "none" if first is None else f"step={first['step']} progress={first['path_progress']:.2f}"
        print(
            f"{case}: high_contact={summary['high_contact_count']}/{summary['n_rows']} "
            f"contact_p95={summary['contact_strength']['p95']:.3f} "
            f"magnet_wall_pull_high_med={summary['magnet_wall_pull_high_contact']['median']} "
            f"estdist_corr={summary['corr_contact_vs_estimated_image_distance_px']} "
            f"first_high={first_text}"
        )

    if len(combined) > 1:
        out_dir = args.out or cases[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wall_contact_diagnostics_summary_all.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
