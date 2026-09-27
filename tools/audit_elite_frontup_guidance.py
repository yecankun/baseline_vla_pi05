from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return vec / norm


def audit_episode(episode_dir: Path) -> dict:
    states = _load_jsonl(episode_dir / "states.jsonl")
    actions = {int(row["step"]): row.get("action", {}) for row in _load_jsonl(episode_dir / "actions.jsonl")}

    actual_forward: list[float] = []
    actual_up: list[float] = []
    desired_forward: list[float] = []
    desired_up: list[float] = []
    plan_gap: list[float] = []

    for state in states:
        tip = np.asarray(state["tip_pos"], dtype=float)
        tangent = _unit(np.asarray(state["path_tangent"], dtype=float))
        magnetic = np.asarray(state["magnetic_pose"], dtype=float)
        actual_delta = magnetic - tip
        actual_forward.append(float(np.dot(actual_delta, tangent)))
        actual_up.append(float(actual_delta[2]))

        action = actions.get(int(state["step"]), {})
        if "elite_desired_pose_3d" in action:
            desired = np.asarray(action["elite_desired_pose_3d"], dtype=float)
            desired_delta = desired - tip
            desired_forward.append(float(np.dot(desired_delta, tangent)))
            desired_up.append(float(desired_delta[2]))
        if "elite_plan_progress" in action and "path_progress" in state:
            plan_gap.append(float(action["elite_plan_progress"]) - float(state["path_progress"]))

    actual_negative = int(sum(value < 0.0 for value in actual_forward))
    desired_negative = int(sum(value < 0.0 for value in desired_forward))
    return {
        "episode": episode_dir.name,
        "samples": len(states),
        "actual_forward_m": _stats(actual_forward),
        "actual_up_m": _stats(actual_up),
        "desired_forward_m": _stats(desired_forward),
        "desired_up_m": _stats(desired_up),
        "elite_plan_minus_tip_progress": _stats(plan_gap),
        "actual_forward_negative_fraction": float(actual_negative / max(len(actual_forward), 1)),
        "desired_forward_negative_fraction": float(desired_negative / max(len(desired_forward), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether Elite magnetic guidance is in front/up of the guidewire tip.")
    parser.add_argument("dataset", type=Path, help="Dataset root or a single episode directory.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--print-details", action="store_true", help="Print per-episode details to stdout.")
    args = parser.parse_args()

    if (args.dataset / "states.jsonl").exists():
        episode_dirs = [args.dataset]
    else:
        episode_dirs = sorted(path for path in args.dataset.glob("episode_*") if (path / "states.jsonl").exists())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode states.jsonl found under {args.dataset}")

    episodes = [audit_episode(path) for path in episode_dirs]
    all_actual_forward = []
    all_desired_forward = []
    for episode in episodes:
        all_actual_forward.extend([episode["actual_forward_m"]["p50"]])
        if episode["desired_forward_m"]["count"]:
            all_desired_forward.extend([episode["desired_forward_m"]["p50"]])
    summary = {
        "dataset": str(args.dataset),
        "episodes": len(episodes),
        "episode_actual_forward_p50_m": _stats([float(x) for x in all_actual_forward if x is not None]),
        "episode_desired_forward_p50_m": _stats([float(x) for x in all_desired_forward if x is not None]),
        "episode_actual_negative_fraction_p50": _stats([float(ep["actual_forward_negative_fraction"]) for ep in episodes]),
        "episodes_detail": episodes,
    }

    output = summary if args.print_details else {key: value for key, value in summary.items() if key != "episodes_detail"}
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
