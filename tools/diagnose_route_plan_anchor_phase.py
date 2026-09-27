from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quantile(values: Iterable[float], p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _sub(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _corr(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return float("nan")
    xs = [float(x) for x in xs[:n]]
    ys = [float(y) for y in ys[:n]]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values) if values else float("nan"),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else float("nan"),
    }


def _action_piper_feed(action: dict) -> float | None:
    payload = action.get("action", action)
    if "piper_feed" not in payload:
        return None
    return float(payload["piper_feed"])


def analyze_rollout(rollout_dir: Path) -> dict:
    results = {}
    for task_dir in sorted(p for p in rollout_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        states_path = task_dir / "states.jsonl"
        actions_path = task_dir / "actions.jsonl"
        meta_path = task_dir / "meta.json"
        if not states_path.exists() or not actions_path.exists():
            continue
        states = _read_jsonl(states_path)
        actions = _read_jsonl(actions_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if not states or not actions:
            continue
        plan_step = float(meta.get("elite_route_plan_anchor_step", 0.20))
        start_progress = float(states[0]["path_progress"])
        n = min(len(actions), len(states) - 1)
        rows = []
        for i in range(n):
            state = states[i]
            next_state = states[i + 1]
            plan_progress = start_progress + plan_step * (i + 1)
            actual_progress = float(state["path_progress"])
            tip_to_magnetic = _distance(state["tip_pos"], state["magnetic_pose"])
            magnetic_forward = _dot(_sub(state["magnetic_pose"], state["tip_pos"]), state["path_tangent"])
            rows.append(
                {
                    "step": int(state["step"]),
                    "plan_progress": plan_progress,
                    "actual_progress": actual_progress,
                    "plan_minus_actual": plan_progress - actual_progress,
                    "progress_delta": float(next_state["path_progress"]) - actual_progress,
                    "piper_feed": _action_piper_feed(actions[i]),
                    "contact_strength": float(state.get("contact_strength", 0.0)),
                    "tip_to_magnetic": tip_to_magnetic,
                    "magnetic_forward": magnetic_forward,
                    "piper_insertion_length": float(state.get("piper_insertion_length", 0.0)),
                }
            )
        results[task] = summarize_rows(rows, meta=meta)
        results[task]["rows"] = rows
    return results


def analyze_dataset(dataset_dir: Path, plan_step: float) -> dict:
    results: dict[str, dict] = {}
    for ep_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        meta_path = ep_dir / "meta.json"
        states_path = ep_dir / "states.jsonl"
        actions_path = ep_dir / "actions.jsonl"
        if not meta_path.exists() or not states_path.exists() or not actions_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task = str(meta.get("task", "") or ep_dir.name)
        states = _read_jsonl(states_path)
        actions = _read_jsonl(actions_path)
        start_progress = float(meta.get("start_progress", states[0]["path_progress"] if states else 0.0))
        n = min(len(actions), len(states) - 1)
        task_result = results.setdefault(task, {"rows": [], "episodes": 0, "episode_steps": []})
        task_result["episodes"] += 1
        if "steps" in meta:
            task_result["episode_steps"].append(float(meta["steps"]))
        for i in range(n):
            state = states[i]
            next_state = states[i + 1]
            env_step = int(state["step"])
            # The formal route-plan expert advances the plan once per env.step,
            # even though only every sample-every step is recorded.
            plan_progress = start_progress + plan_step * (env_step + 1)
            tip_to_magnetic = _distance(state["tip_pos"], state["magnetic_pose"])
            magnetic_forward = _dot(_sub(state["magnetic_pose"], state["tip_pos"]), state["path_tangent"])
            task_result["rows"].append(
                {
                    "episode": ep_dir.name,
                    "step": env_step,
                    "plan_progress": plan_progress,
                    "actual_progress": float(state["path_progress"]),
                    "plan_minus_actual": plan_progress - float(state["path_progress"]),
                    "progress_delta": float(next_state["path_progress"]) - float(state["path_progress"]),
                    "piper_feed": _action_piper_feed(actions[i]),
                    "contact_strength": float(state.get("contact_strength", 0.0)),
                    "tip_to_magnetic": tip_to_magnetic,
                    "magnetic_forward": magnetic_forward,
                    "piper_insertion_length": float(state.get("piper_insertion_length", 0.0)),
                }
            )
    for task, task_result in results.items():
        summary = summarize_rows(task_result["rows"], meta={})
        summary["episodes"] = task_result["episodes"]
        summary["episode_steps"] = _summary(task_result["episode_steps"])
        summary["rows"] = task_result["rows"]
        results[task] = summary
    return results


def summarize_rows(rows: list[dict], meta: dict) -> dict:
    plan_lead = [r["plan_minus_actual"] for r in rows]
    progress_delta = [r["progress_delta"] for r in rows]
    piper = [r["piper_feed"] for r in rows if r["piper_feed"] is not None]
    contact = [r["contact_strength"] for r in rows]
    tip_to_magnetic = [r["tip_to_magnetic"] for r in rows]
    magnetic_forward = [r["magnetic_forward"] for r in rows]
    insertion = [r["piper_insertion_length"] for r in rows]
    high_contact_idx = [i for i, r in enumerate(rows) if r["contact_strength"] > 0.1]
    lead_gt5_idx = [i for i, r in enumerate(rows) if r["plan_minus_actual"] > 5.0]

    def at(indices: list[int], key: str) -> list[float]:
        return [float(rows[i][key]) for i in indices]

    return {
        "meta": meta,
        "num_rows": len(rows),
        "plan_minus_actual": _summary(plan_lead),
        "progress_delta": _summary(progress_delta),
        "piper_feed": _summary(piper),
        "piper_negative_fraction": (sum(1 for x in piper if x < 0.0) / len(piper)) if piper else float("nan"),
        "piper_low_lt_0p2_fraction": (sum(1 for x in piper if x < 0.2) / len(piper)) if piper else float("nan"),
        "contact_strength": _summary(contact),
        "contact_gt_0p1_fraction": (len(high_contact_idx) / len(rows)) if rows else float("nan"),
        "tip_to_magnetic_m": _summary(tip_to_magnetic),
        "tip_to_magnetic_mm": _summary([x * 1000.0 for x in tip_to_magnetic]),
        "magnetic_forward_mm": _summary([x * 1000.0 for x in magnetic_forward]),
        "piper_insertion_mm": _summary([x * 1000.0 for x in insertion]),
        "corr_plan_lead_contact": _corr(plan_lead, contact),
        "corr_tip_to_magnetic_contact": _corr(tip_to_magnetic, contact),
        "corr_piper_progress_delta": _corr(piper, progress_delta[: len(piper)]),
        "high_contact_plan_minus_actual": _summary(at(high_contact_idx, "plan_minus_actual")),
        "high_contact_tip_to_magnetic_mm": _summary([x * 1000.0 for x in at(high_contact_idx, "tip_to_magnetic")]),
        "lead_gt5_contact_strength": _summary(at(lead_gt5_idx, "contact_strength")),
        "lead_gt5_tip_to_magnetic_mm": _summary([x * 1000.0 for x in at(lead_gt5_idx, "tip_to_magnetic")]),
    }


def strip_rows(results: dict) -> dict:
    stripped = {}
    for task, data in results.items():
        stripped[task] = {k: v for k, v in data.items() if k != "rows"}
    return stripped


def write_csv(path: Path, source: str, results: dict) -> None:
    rows = []
    for task, data in results.items():
        for row in data.get("rows", []):
            out = {"source": source, "task": task}
            out.update(row)
            rows.append(out)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, dataset_summary: dict, rollout_summary: dict) -> None:
    lines = [
        "# Route-Plan Anchor Phase Diagnostic",
        "",
        "This diagnostic reconstructs scheduled route-plan progress and compares it with actual tip path progress.",
        "",
        "## Summary",
        "",
    ]
    for source, summary in (("expert_dataset", dataset_summary), ("rollout", rollout_summary)):
        lines.append(f"### {source}")
        lines.append("")
        for task, data in summary.items():
            plan = data["plan_minus_actual"]
            contact = data["contact_strength"]
            tipmag = data["tip_to_magnetic_mm"]
            lines.append(
                f"- {task}: plan_minus_actual median/p95={plan['median']:.3f}/{plan['p95']:.3f}; "
                f"contact p95/max={contact['p95']:.3f}/{contact['max']:.3f}; "
                f"tip_to_magnetic median/p95={tipmag['median']:.1f}/{tipmag['p95']:.1f} mm; "
                f"corr(plan_lead, contact)={data['corr_plan_lead_contact']:.3f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- In the formal expert dataset, scheduled plan progress is slightly behind actual tip progress at sampled states.",
            "- In the anchored BC rollout, scheduled plan progress runs ahead of actual tip progress by several path indices.",
            "- High contact aligns with large positive plan lead and large tip-to-magnetic distance.",
            "- This indicates a phase mismatch between scheduled Elite route anchor and actual Piper/tip progress, not merely an Elite joint smoothness problem.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose route-plan anchor phase mismatch against actual tip progress.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-plan-step", type=float, default=0.20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dataset_results = analyze_dataset(args.dataset, plan_step=args.dataset_plan_step)
    rollout_results = analyze_rollout(args.rollout)
    dataset_summary = strip_rows(dataset_results)
    rollout_summary = strip_rows(rollout_results)

    (args.out / "route_plan_anchor_phase_summary.json").write_text(
        json.dumps({"dataset": dataset_summary, "rollout": rollout_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.out / "dataset_phase_rows.csv", "dataset", dataset_results)
    write_csv(args.out / "rollout_phase_rows.csv", "rollout", rollout_results)
    write_markdown(args.out / "route_plan_anchor_phase_summary.md", dataset_summary, rollout_summary)
    print(f"saved route-plan anchor phase diagnostic to {args.out}")


if __name__ == "__main__":
    main()
