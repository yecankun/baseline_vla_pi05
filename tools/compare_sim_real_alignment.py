from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def q(values, quantile: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, quantile, method="nearest"))


def summary(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": q(arr, 0.95),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def step_l2(values: list[list[float]], scale: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1) * scale


def step_linf(values: list[list[float]], scale: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return np.asarray([], dtype=np.float64)
    return np.max(np.abs(np.diff(arr, axis=0)), axis=1) * scale


def image_shape(path: Path) -> list[int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return [int(h), int(w), int(img.shape[2])]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def summarize_path_step_stats(path_stats: list[dict], key: str) -> dict:
    p95_values = [item[key]["p95"] for item in path_stats]
    max_values = [item[key]["max"] for item in path_stats]
    median_values = [item[key]["median"] for item in path_stats]
    return {
        "step_median_across_paths": summary(median_values),
        "step_p95_across_paths": summary(p95_values),
        "step_max_across_paths": summary(max_values),
    }


def real_alignment(real_paths_path: Path) -> dict:
    rows = read_json(real_paths_path)
    label_counts = Counter()
    for row in rows:
        label_counts.update(row.get("label_counts", {}))
    image_shapes = Counter(json.dumps(row.get("image_shape"), ensure_ascii=False) for row in rows)
    position = summarize_path_step_stats(rows, "position_step")
    rotation = summarize_path_step_stats(rows, "rotation_step_linf")
    return {
        "source": "real_branchs",
        "semantic": "Real Elite/magnetic-arm pose trajectory; not guidewire trajectory.",
        "paths": len(rows),
        "frames": int(sum(row.get("frames_pose", 0) for row in rows)),
        "image_frames": int(sum(row.get("frames_images", 0) for row in rows)),
        "image_shapes": dict(image_shapes),
        "position_step_l2_mm": position,
        "rotation_step_linf": rotation,
        "piper_labels": dict(sorted(label_counts.items())),
        "notes": [
            "Real pose values are treated as millimeter-scale Cartesian pose.",
            "Use this only as an Elite/magnetic-arm motion anchor, not as guidewire path ground truth.",
        ],
    }


def samples_by_episode(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("episode", ""))].append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("step", 0)))
    return grouped


def joint_vec(mapping: dict) -> list[float]:
    return [float(mapping[key]) for key in mapping.keys()]


def summarize_sim_samples(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    samples = manifest.get("samples", [])
    grouped = samples_by_episode(samples)
    pose_keys = {
        "elirobot_pose": "Sim visible Elite tool pose.",
        "magnetic_pose": "Sim magnetic effective point.",
        "tip_pos": "Sim guidewire tip; no direct real counterpart in branchs.",
    }
    pose_stats = {}
    for key, semantic in pose_keys.items():
        per_episode = []
        for rows in grouped.values():
            values = [row["state"][key] for row in rows if key in row.get("state", {})]
            steps_mm = step_l2(values, scale=1000.0)
            if steps_mm.size:
                per_episode.append({"median": float(np.median(steps_mm)), "p95": q(steps_mm, 0.95), "max": float(np.max(steps_mm))})
        pose_stats[key] = {
            "semantic": semantic,
            "step_l2_mm": summarize_path_step_stats([{"step": item} for item in per_episode], "step"),
        }

    elite_joint_steps = []
    piper_feed = []
    for rows in grouped.values():
        elite_values = []
        for row in rows:
            action = row.get("action", {})
            if "piper_feed" in action:
                piper_feed.append(float(action["piper_feed"]))
            elite = action.get("elite_joints")
            if isinstance(elite, dict):
                elite_values.append(joint_vec(elite))
        elite_joint_steps.extend(step_linf(elite_values).tolist())

    first_shape = {}
    if samples:
        for view in ("side", "top"):
            image = samples[0].get("images", {}).get(view)
            if image:
                first_shape[view] = image_shape(resolve_path(image))

    return {
        "source": str(manifest_path.parent),
        "semantic": "Sim expert dataset generated by MuJoCo physical guidance.",
        "episodes": len(grouped),
        "frames": len(samples),
        "image_shapes": first_shape,
        "pose_stats": pose_stats,
        "elite_joint_target_step_linf_rad": summary(elite_joint_steps),
        "piper_feed": summarize_piper_feed(piper_feed),
    }


def summarize_piper_feed(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "negative": int(np.sum(arr < -0.05)),
        "hold": int(np.sum(np.abs(arr) <= 0.05)),
        "positive": int(np.sum(arr > 0.05)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": q(arr, 0.95),
        "max": float(np.max(arr)),
        "abs_p95": q(np.abs(arr), 0.95),
    }


def summarize_rollout(rollout_dir: Path) -> dict:
    tasks = []
    pose_keys = {
        "elirobot_pose": "Closed-loop sim visible Elite tool pose.",
        "magnetic_pose": "Closed-loop sim magnetic effective point.",
        "tip_pos": "Closed-loop sim guidewire tip; no direct real counterpart in branchs.",
    }
    pose_stats = {key: [] for key in pose_keys}
    executed_joint_steps = []
    target_joint_steps = []
    piper_feed = []
    frames = 0
    for task_dir in sorted([p for p in rollout_dir.iterdir() if p.is_dir()]):
        states_path = task_dir / "states.jsonl"
        actions_path = task_dir / "actions.jsonl"
        if not states_path.exists():
            continue
        tasks.append(task_dir.name)
        states = read_jsonl(states_path)
        frames += len(states)
        for key in pose_keys:
            values = [row[key] for row in states if key in row]
            steps_mm = step_l2(values, scale=1000.0)
            if steps_mm.size:
                pose_stats[key].append({"median": float(np.median(steps_mm)), "p95": q(steps_mm, 0.95), "max": float(np.max(steps_mm))})

        executed = []
        for row in states:
            elite = row.get("robot_state", {}).get("elite_joints")
            if isinstance(elite, dict):
                executed.append(joint_vec(elite))
        executed_joint_steps.extend(step_linf(executed).tolist())

        if actions_path.exists():
            actions = read_jsonl(actions_path)
            targets = []
            for row in actions:
                action = row.get("action", {})
                if "piper_feed" in action:
                    piper_feed.append(float(action["piper_feed"]))
                elite = action.get("elite_joints")
                if isinstance(elite, dict):
                    targets.append(joint_vec(elite))
            target_joint_steps.extend(step_linf(targets).tolist())

    return {
        "source": str(rollout_dir),
        "semantic": "Closed-loop policy rollout in MuJoCo.",
        "tasks": tasks,
        "frames": frames,
        "pose_stats": {
            key: {
                "semantic": semantic,
                "step_l2_mm": summarize_path_step_stats([{"step": item} for item in values], "step"),
            }
            for key, semantic, values in [(k, pose_keys[k], pose_stats[k]) for k in pose_keys]
        },
        "elite_executed_joint_step_linf_rad": summary(executed_joint_steps),
        "elite_target_joint_step_linf_rad": summary(target_joint_steps),
        "piper_feed": summarize_piper_feed(piper_feed),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def motion_row(source: str, semantic: str, stats: dict, notes: str = "") -> list[str]:
    p95 = stats["step_l2_mm"]["step_p95_across_paths"]
    mx = stats["step_l2_mm"]["step_max_across_paths"]
    med = stats["step_l2_mm"]["step_median_across_paths"]
    return [
        source,
        semantic,
        fmt(med["median"]),
        fmt(p95["median"]),
        fmt(p95["max"]),
        fmt(mx["median"]),
        notes,
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def build_markdown(real: dict, sim_dataset: dict, rollout: dict) -> str:
    lines = [
        "# Sim-vs-Real Alignment",
        "",
        "Generated by `tools/compare_sim_real_alignment.py`.",
        "",
        "Important semantic constraint: the real `branchs` pose files are treated as Elite/magnetic-arm motion trajectories, not guidewire trajectories. They should be compared with simulated `elirobot_pose` / `magnetic_pose`, not with simulated `tip_pos` as ground truth.",
        "",
        "## Source Overview",
        "",
    ]
    lines.append(
        markdown_table(
            ["Source", "Frames", "Trajectories", "Image Shape", "Action/Piper Semantics"],
            [
                [
                    "Real `branchs`",
                    str(real["frames"]),
                    str(real["paths"]),
                    ", ".join(real["image_shapes"].keys()),
                    f"Binary labels {real['piper_labels']}",
                ],
                [
                    "Sim expert dataset",
                    str(sim_dataset["frames"]),
                    str(sim_dataset["episodes"]),
                    json.dumps(sim_dataset["image_shapes"], ensure_ascii=False),
                    "Continuous normalized `piper_feed`",
                ],
                [
                    "Sim policy rollout",
                    str(rollout["frames"]),
                    ", ".join(rollout["tasks"]),
                    "rendered video frames, not dataset frames",
                    "Continuous normalized `piper_feed`",
                ],
            ],
        )
    )
    lines.extend(["", "## Mechanical/Magnetic Motion Scale", ""])
    motion_rows = []
    motion_rows.append(
        motion_row(
            "Real `branchs`",
            "Elite/magnetic-arm pose, not guidewire",
            {"step_l2_mm": real["position_step_l2_mm"]},
            "Real pose appears millimeter-scale.",
        )
    )
    for key in ("elirobot_pose", "magnetic_pose", "tip_pos"):
        note = "Sim-only guidewire tip; no direct real counterpart." if key == "tip_pos" else ""
        motion_rows.append(motion_row("Sim expert", key, sim_dataset["pose_stats"][key], note))
    for key in ("elirobot_pose", "magnetic_pose", "tip_pos"):
        note = "Sim-only guidewire tip; no direct real counterpart." if key == "tip_pos" else ""
        motion_rows.append(motion_row("Sim rollout", key, rollout["pose_stats"][key], note))
    lines.append(
        markdown_table(
            [
                "Source",
                "Trajectory Semantic",
                "Median Step Median (mm)",
                "Median Step P95 (mm)",
                "Worst Step P95 (mm)",
                "Median Step Max (mm)",
                "Notes",
            ],
            motion_rows,
        )
    )
    lines.extend(["", "## Action Semantics", ""])
    lines.append(
        markdown_table(
            ["Source", "Piper", "Elite Joint Target Step P95 (rad)", "Elite Joint Target Step Max (rad)", "Notes"],
            [
                [
                    "Real `branchs`",
                    f"Binary labels {real['piper_labels']}",
                    "N/A",
                    "N/A",
                    "Real files provide pose trajectory, not six-joint target labels in this inspection.",
                ],
                [
                    "Sim expert dataset",
                    json.dumps(sim_dataset["piper_feed"], ensure_ascii=False),
                    fmt(sim_dataset["elite_joint_target_step_linf_rad"]["p95"], 5),
                    fmt(sim_dataset["elite_joint_target_step_linf_rad"]["max"], 5),
                    "Expert labels are smooth.",
                ],
                [
                    "Sim policy rollout",
                    json.dumps(rollout["piper_feed"], ensure_ascii=False),
                    fmt(rollout["elite_target_joint_step_linf_rad"]["p95"], 5),
                    fmt(rollout["elite_target_joint_step_linf_rad"]["max"], 5),
                    "Policy target jitter should be compared with expert labels, not real pose directly.",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- Real data is usable as a mechanical/magnetic-arm motion anchor, not as guidewire path ground truth.",
            "- The simulator has a Piper semantic gap: real data uses binary `0/1` labels, while MuJoCo currently uses continuous `piper_feed`.",
            "- Simulated mechanical/magnetic motion scale should be checked against the real pose step scale before further closed-loop tuning.",
            "- Simulated `tip_pos` is still useful for internal guidewire evaluation, but it should not be called real-aligned until a real guidewire-tip signal is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sim-vs-real alignment table for Project 2026.")
    parser.add_argument("--real-paths", default="simulation_output/real_branch_data_inspection/real_branch_paths.json")
    parser.add_argument("--sim-manifest", default="simulation_output/mujoco_physical_feed_action_dataset_v2/manifest.json")
    parser.add_argument("--rollout", default="simulation_output/baseline_mujoco_physical_feed_action_v2_rollout")
    parser.add_argument("--out", default="simulation_output/sim_vs_real_alignment")
    args = parser.parse_args()

    real = real_alignment(Path(args.real_paths))
    sim_dataset = summarize_sim_samples(Path(args.sim_manifest))
    rollout = summarize_rollout(Path(args.rollout))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"real": real, "sim_dataset": sim_dataset, "sim_rollout": rollout}
    (out / "sim_vs_real_alignment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = build_markdown(real, sim_dataset, rollout)
    (out / "sim_vs_real_alignment.md").write_text(markdown, encoding="utf-8")

    docs_path = Path("docs/sim-vs-real-alignment.md")
    docs_path.write_text(markdown, encoding="utf-8")

    print(f"Saved JSON to {out / 'sim_vs_real_alignment.json'}")
    print(f"Saved Markdown to {out / 'sim_vs_real_alignment.md'}")
    print(f"Updated docs copy at {docs_path}")


if __name__ == "__main__":
    main()
