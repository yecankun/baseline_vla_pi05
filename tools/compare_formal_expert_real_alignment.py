from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
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


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def unit_vectors(vectors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors), where=norms > eps)


def direction_reversal_fraction(delta_pos: np.ndarray) -> float:
    if len(delta_pos) < 2:
        return 0.0
    nonzero = np.linalg.norm(delta_pos, axis=1) > 1e-9
    pairs = nonzero[:-1] & nonzero[1:]
    if not np.any(pairs):
        return 0.0
    u = unit_vectors(delta_pos)
    dots = np.sum(u[:-1] * u[1:], axis=1)[pairs]
    return float(np.mean(dots < -0.25)) if dots.size else 0.0


def pose_motion_stats(rows: list[dict], key: str) -> dict[str, Any]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        state = row.get("state", {})
        if key in state:
            by_episode[str(row.get("episode", ""))].append(row)

    per_episode = []
    all_saved_steps = []
    all_per_env_steps = []
    all_accel = []
    all_jerk = []
    all_zero_step = []
    all_direction_reversal = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda item: int(item.get("step", 0)))
        if len(episode_rows) < 2:
            continue
        steps = np.asarray([int(item.get("step", 0)) for item in episode_rows], dtype=np.float64)
        pos = np.asarray([item["state"][key][:3] for item in episode_rows], dtype=np.float64) * 1000.0
        dpos = np.diff(pos, axis=0)
        dstep = np.diff(steps)
        valid = dstep > 0
        if not np.any(valid):
            continue
        saved_step = np.linalg.norm(dpos[valid], axis=1)
        per_env_step = saved_step / dstep[valid]
        velocities = dpos[valid] / dstep[valid, None]
        accel = np.linalg.norm(np.diff(velocities, axis=0), axis=1) if len(velocities) >= 2 else np.asarray([])
        jerk = np.linalg.norm(np.diff(velocities, n=2, axis=0), axis=1) if len(velocities) >= 3 else np.asarray([])
        zero_fraction = float(np.mean(saved_step <= 1e-9)) if saved_step.size else 0.0
        reverse_fraction = direction_reversal_fraction(dpos[valid])
        all_saved_steps.extend(saved_step.tolist())
        all_per_env_steps.extend(per_env_step.tolist())
        all_accel.extend(accel.tolist())
        all_jerk.extend(jerk.tolist())
        all_zero_step.append(zero_fraction)
        all_direction_reversal.append(reverse_fraction)
        per_episode.append(
            {
                "saved_step_p95": float(np.quantile(saved_step, 0.95, method="nearest")),
                "per_env_step_p95": float(np.quantile(per_env_step, 0.95, method="nearest")),
                "accel_p95": float(np.quantile(accel, 0.95, method="nearest")) if accel.size else 0.0,
                "jerk_p95": float(np.quantile(jerk, 0.95, method="nearest")) if jerk.size else 0.0,
                "direction_reversal_fraction": reverse_fraction,
                "zero_step_fraction": zero_fraction,
            }
        )

    return {
        "episodes": len(per_episode),
        "saved_sample_step_l2_mm": quantiles(all_saved_steps),
        "per_env_step_l2_mm": quantiles(all_per_env_steps),
        "velocity_change_l2_mm_per_env_step": quantiles(all_accel),
        "velocity_jerk_l2_mm_per_env_step": quantiles(all_jerk),
        "direction_reversal_fraction_across_episodes": quantiles(all_direction_reversal),
        "zero_saved_step_fraction_across_episodes": quantiles(all_zero_step),
        "episode_p95": {
            "saved_sample_step_l2_mm": quantiles([item["saved_step_p95"] for item in per_episode]),
            "per_env_step_l2_mm": quantiles([item["per_env_step_p95"] for item in per_episode]),
            "velocity_change_l2_mm_per_env_step": quantiles([item["accel_p95"] for item in per_episode]),
            "velocity_jerk_l2_mm_per_env_step": quantiles([item["jerk_p95"] for item in per_episode]),
        },
    }


def piper_sim_stats(samples: list[dict]) -> dict[str, Any]:
    values = []
    step_commands = []
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        action = row.get("action", {})
        if "piper_feed" in action:
            value = float(action["piper_feed"])
            values.append(value)
            by_episode[str(row.get("episode", ""))].append(row)
        if "piper_step_command" in action:
            step_commands.append(int(np.clip(round(float(action["piper_step_command"])), -1, 1)))

    transitions = []
    run_lengths: list[int] = []
    sign_counts = Counter()
    for value in values:
        if value < -0.05:
            sign_counts["negative"] += 1
        elif value > 0.05:
            sign_counts["positive"] += 1
        else:
            sign_counts["hold"] += 1

    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda item: int(item.get("step", 0)))
        signs = []
        for row in episode_rows:
            value = float(row.get("action", {}).get("piper_feed", 0.0))
            signs.append("-" if value < -0.05 else "+" if value > 0.05 else "0")
        if len(signs) < 2:
            continue
        changes = [prev != nxt for prev, nxt in zip(signs, signs[1:])]
        transitions.append(float(np.mean(changes)))
        current = signs[0]
        length = 1
        for sign in signs[1:]:
            if sign == current:
                length += 1
            else:
                run_lengths.append(length)
                current = sign
                length = 1
        run_lengths.append(length)

    total = max(len(values), 1)
    return {
        "count": len(values),
        "value": quantiles(values),
        "sign_counts": dict(sorted(sign_counts.items())),
        "sign_fractions": {key: float(count / total) for key, count in sorted(sign_counts.items())},
        "step_command_counts": dict(sorted(Counter(step_commands).items())),
        "step_command_fractions": {
            str(key): float(count / max(len(step_commands), 1))
            for key, count in sorted(Counter(step_commands).items())
        },
        "transition_fraction_across_episodes": quantiles(transitions),
        "run_length_samples": quantiles(run_lengths),
    }


def elite_action_stats(samples: list[dict]) -> dict[str, Any]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        action = row.get("action", {})
        if isinstance(action.get("elite_joints"), dict):
            by_episode[str(row.get("episode", ""))].append(row)

    all_steps = []
    all_per_env_steps = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda item: int(item.get("step", 0)))
        if len(episode_rows) < 2:
            continue
        steps = np.asarray([int(item.get("step", 0)) for item in episode_rows], dtype=np.float64)
        joints = []
        for row in episode_rows:
            elite = row["action"]["elite_joints"]
            joints.append([float(elite[key]) for key in sorted(elite.keys())])
        arr = np.asarray(joints, dtype=np.float64)
        djoint = np.max(np.abs(np.diff(arr, axis=0)), axis=1)
        dstep = np.diff(steps)
        valid = dstep > 0
        all_steps.extend(djoint[valid].tolist())
        all_per_env_steps.extend((djoint[valid] / dstep[valid]).tolist())

    return {
        "saved_sample_step_linf_rad": quantiles(all_steps),
        "per_env_step_linf_rad": quantiles(all_per_env_steps),
    }


def summarize_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    samples = manifest.get("samples", [])
    episodes = manifest.get("episodes", [])
    image_missing = 0
    image_total = 0
    for sample in samples:
        for image_path in sample.get("images", {}).values():
            image_total += 1
            if not Path(image_path).exists():
                image_missing += 1

    return {
        "path": str(path),
        "root": str(path.parent),
        "mode": manifest.get("mode"),
        "guidance_mode": manifest.get("guidance_mode"),
        "validity_audit": manifest.get("validity_audit"),
        "episodes": len(episodes),
        "samples": len(samples),
        "tasks": dict(sorted(Counter(str(ep.get("task")) for ep in episodes).items())),
        "env_success": int(sum(bool(ep.get("env_success")) for ep in episodes)),
        "rejected": len(manifest.get("rejected_episodes", [])),
        "image_refs": {"total": image_total, "missing": image_missing},
        "start_fraction": quantiles([float(ep.get("start_fraction", 0.0)) for ep in episodes]),
        "contact_max": max([float(ep.get("max_contact_strength", 0.0)) for ep in episodes], default=0.0),
        "min_segment_distance_to_wall_m": min(
            [float(ep.get("min_segment_distance_to_wall", 0.0)) for ep in episodes], default=0.0
        ),
        "tip_to_magnetic_m": {
            "episode_median": quantiles([float(ep.get("tip_to_magnetic_median", 0.0)) for ep in episodes]),
            "episode_p95": quantiles([float(ep.get("tip_to_magnetic_p95", 0.0)) for ep in episodes]),
            "episode_max": quantiles([float(ep.get("tip_to_magnetic_max", 0.0)) for ep in episodes]),
        },
        "motion": {
            "elirobot_pose": pose_motion_stats(samples, "elirobot_pose"),
            "magnetic_pose": pose_motion_stats(samples, "magnetic_pose"),
            "tip_pos": pose_motion_stats(samples, "tip_pos"),
        },
        "piper_feed": piper_sim_stats(samples),
        "elite_action": elite_action_stats(samples),
    }


def build_markdown(real_summary: dict[str, Any], datasets: list[dict[str, Any]]) -> str:
    real_all = real_summary["all"]
    real_labels = real_summary.get("piper_label_counts", {})
    lines = [
        "# Formal Expert vs Real Alignment",
        "",
        "This report compares formal route-plan expert datasets with real `branchs` motion anchors.",
        "",
        "Semantic guardrail: real `branchs` pose is treated as Elite/magnetic-arm motion, not guidewire-tip path truth. Simulated `tip_pos` is reported as an internal simulator signal only.",
        "",
        "## Real Anchor",
        "",
        markdown_table(
            [
                "Source",
                "Paths",
                "Pose Frames",
                "Pose Step P95 Median (mm/frame)",
                "Accel P95 Median (mm/frame^2)",
                "Jerk P95 Median (mm/frame^3)",
                "Direction Reversal Median",
                "Zero Step Median",
                "Piper Labels",
            ],
            [
                [
                    "branchs",
                    str(real_summary.get("total_paths", "")),
                    str(real_summary.get("total_pose_frames", "")),
                    fmt(real_all["position_step_p95_across_paths"]["median"]),
                    fmt(real_all["position_accel_p95_across_paths"]["median"]),
                    fmt(real_all["position_jerk_p95_across_paths"]["median"]),
                    fmt(real_all["direction_reversal_fraction_across_paths"]["median"], 4),
                    fmt(real_all["zero_step_fraction_across_paths"]["median"], 4),
                    f"0=stop, 1=advance; counts {json.dumps(real_labels, ensure_ascii=False)}",
                ]
            ],
        ),
        "",
        "## Dataset Quality",
        "",
    ]
    quality_rows = []
    for data in datasets:
        quality_rows.append(
            [
                Path(data["root"]).name,
                str(data["episodes"]),
                str(data["env_success"]),
                str(data["rejected"]),
                str(data["samples"]),
                f"{data['image_refs']['total']}/{data['image_refs']['missing']} missing",
                fmt(data["start_fraction"]["min"], 4)
                + " / "
                + fmt(data["start_fraction"]["median"], 4)
                + " / "
                + fmt(data["start_fraction"]["max"], 4),
                fmt(data["contact_max"]),
                fmt(data["min_segment_distance_to_wall_m"] * 1000.0),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Dataset",
                "Episodes",
                "Env Success",
                "Rejected",
                "Samples",
                "Image Refs",
                "Start Fraction Min/Med/Max",
                "Contact Max",
                "Min Wall Clearance (mm)",
            ],
            quality_rows,
        )
    )

    lines.extend(["", "## Motion Scale", ""])
    motion_rows = []
    motion_rows.append(
        [
            "real branchs",
            "Elite/magnetic-arm pose",
            fmt(real_all["position_step_p95_across_paths"]["median"]),
            "N/A",
            fmt(real_all["position_accel_p95_across_paths"]["median"]),
            fmt(real_all["position_jerk_p95_across_paths"]["median"]),
            fmt(real_all["direction_reversal_fraction_across_paths"]["median"], 4),
            "real frame",
        ]
    )
    for data in datasets:
        for key, semantic in [
            ("elirobot_pose", "sim Elite/magnetic pose"),
            ("magnetic_pose", "sim magnetic point"),
            ("tip_pos", "sim-only guidewire tip"),
        ]:
            stats = data["motion"][key]
            ep = stats["episode_p95"]
            motion_rows.append(
                [
                    Path(data["root"]).name,
                    semantic,
                    fmt(ep["saved_sample_step_l2_mm"]["median"]),
                    fmt(ep["per_env_step_l2_mm"]["median"]),
                    fmt(ep["velocity_change_l2_mm_per_env_step"]["median"]),
                    fmt(ep["velocity_jerk_l2_mm_per_env_step"]["median"]),
                    fmt(stats["direction_reversal_fraction_across_episodes"]["median"], 4),
                    "saved sample and normalized env step",
                ]
            )
    lines.append(
        markdown_table(
            [
                "Source",
                "Signal",
                "Step P95 Median (mm/saved sample)",
                "Step P95 Median (mm/env step)",
                "Velocity Change P95 Median",
                "Velocity Jerk P95 Median",
                "Direction Reversal Median",
                "Notes",
            ],
            motion_rows,
        )
    )

    lines.extend(["", "## Action Semantics", ""])
    action_rows = [
        [
            "real branchs",
            "binary stop/advance labels",
            f"0=stop, 1=advance; counts {json.dumps(real_labels, ensure_ascii=False)}",
            fmt(real_all["piper_transition_fraction_across_paths"]["median"], 4),
            "N/A",
            "From intervention/bc_piper/inference.py: action 1 calls piper.step_forward; action 0 holds.",
        ]
    ]
    for data in datasets:
        piper = data["piper_feed"]
        elite = data["elite_action"]
        piper_distribution = piper["step_command_fractions"] if piper.get("step_command_fractions") else piper["sign_fractions"]
        action_rows.append(
            [
                Path(data["root"]).name,
                "continuous piper_feed plus optional signed-step label",
                json.dumps(piper_distribution, ensure_ascii=False),
                fmt(piper["transition_fraction_across_episodes"]["median"], 4),
                fmt(elite["saved_sample_step_linf_rad"]["p95"], 5),
                "Not covered by senior stop/advance labels; retract needs an explicit real command label.",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Source",
                "Piper Representation",
                "Piper Distribution",
                "Piper Transition Fraction Median",
                "Elite Joint Target Step P95 (rad/saved sample)",
                "Notes",
            ],
            action_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The formal route-plan expert datasets remain clean as synthetic expert data: high `env_success`, no rejected attempts, zero contact, positive wall clearance, and millimeter-scale `tip_to_magnetic`.",
            "- Motion-scale comparison is not a proof of real alignment because real data and sim samples may not share the same time base. The report therefore shows both per-saved-sample and normalized per-env-step values.",
            "- The largest confirmed semantic gap is Piper: senior real data uses binary stop/advance labels, while the simulator emits continuous feed/retract values with its own switching pattern.",
            "- Retract should not be rejected just because the senior BC interface omitted it; it needs an explicit real command path and labels before formal data is real-action aligned.",
            "- `tip_pos` should remain an internal simulator quality metric unless a real guidewire-tip sensing path is defined.",
            "- The next realism work should focus on time-base/calibration, Piper-command semantics, camera/domain matching, and real-plausible route/scene perturbations before returning to BC optimization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare formal route-plan expert datasets with real branchs anchors.")
    parser.add_argument(
        "--real-summary",
        default="simulation_output/real_control_bandwidth/real_control_bandwidth_summary.json",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Formal expert manifest path. Repeat for multiple datasets.",
    )
    parser.add_argument("--out", default="docs/_formal_expert_real_alignment")
    args = parser.parse_args()

    real_summary = read_json(Path(args.real_summary))
    datasets = [summarize_manifest(Path(path)) for path in args.manifest]
    report = {
        "real_summary_path": args.real_summary,
        "datasets": datasets,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "formal_expert_real_alignment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown = build_markdown(real_summary, datasets)
    (out / "formal_expert_real_alignment.md").write_text(markdown, encoding="utf-8")
    print(f"Saved JSON to {out / 'formal_expert_real_alignment.json'}")
    print(f"Saved Markdown to {out / 'formal_expert_real_alignment.md'}")


if __name__ == "__main__":
    main()
