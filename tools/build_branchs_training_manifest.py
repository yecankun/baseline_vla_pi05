from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PIPER_LABELS = {0: "hold", 1: "feed"}


def numeric_suffix(path: Path, prefix: str) -> int:
    text = path.stem if path.is_file() else path.name
    if not text.startswith(prefix):
        raise ValueError(f"Expected {prefix}* name, got {path.name}")
    return int(text[len(prefix) :])


def read_pose_file(path: Path) -> list[list[float]]:
    poses: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            pose = ast.literal_eval(line)
            if len(pose) != 6:
                raise ValueError(f"{path}:{line_no} expected 6D pose, got {len(pose)}")
            poses.append([float(v) for v in pose])
    return poses


def read_label_file(path: Path) -> list[int]:
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = int(float(line))
            if value not in PIPER_LABELS:
                raise ValueError(f"{path}:{line_no} expected Piper label 0/1, got {value}")
            labels.append(value)
    return labels


def branch_task(branch_name: str, args: argparse.Namespace) -> str:
    if branch_name == "branch1":
        return args.branch1_task
    if branch_name == "branch2":
        return args.branch2_task
    return args.default_task


def build_sample(
    *,
    branch_name: str,
    path_name: str,
    frame_number: int,
    image_path: Path,
    pose: list[float],
    next_pose: list[float],
    piper_step_before_action: int,
    piper_label: int,
    task: str,
    zero_orientation_delta: bool,
    piper_feed_value: float,
) -> dict:
    current = np.asarray(pose, dtype=np.float32).reshape(6)
    target = np.asarray(next_pose, dtype=np.float32).reshape(6)
    delta = (target - current).astype(np.float32)
    if zero_orientation_delta:
        delta[3:] = 0.0
        target = current.copy()
        target[:3] = current[:3] + delta[:3]
    piper_feed = float(piper_feed_value) if int(piper_label) > 0 else 0.0
    image_text = str(image_path.as_posix())
    return {
        "task": task,
        "step": int(frame_number - 1),
        "branch": branch_name,
        "path": path_name,
        "frame_number": int(frame_number),
        "images": {
            "side": image_text,
            "top": image_text,
        },
        "state": {
            "elite_tcp_pose_6d": current.astype(float).tolist(),
            "piper_step": float(piper_step_before_action),
        },
        "action": {
            "piper_feed": piper_feed,
            "piper_step_command": int(piper_label),
            "piper_command_label": PIPER_LABELS[int(piper_label)],
            "elite_tcp_delta_6d": delta.astype(float).tolist(),
            "elite_tcp_pose_6d": target.astype(float).tolist(),
            # Compatibility placeholder: train_dual_arm_baseline gates this
            # action mode on the presence of piper_feed + elite_joints, but
            # tcp_delta training uses elite_tcp_delta_6d as the target.
            "elite_joints": {},
        },
        "source": {
            "dataset": "branchs",
            "single_camera_duplicated_as_top": True,
            "elite_delta_label": "next_pose_minus_current_pose",
            "zero_orientation_delta": bool(zero_orientation_delta),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert senior branchs data to a train_dual_arm_baseline manifest.")
    parser.add_argument("--branch-root", default="branchs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--branch1-task", choices=["left", "right"], default="left")
    parser.add_argument("--branch2-task", choices=["left", "right"], default="right")
    parser.add_argument("--default-task", choices=["left", "right"], default="left")
    parser.add_argument("--max-frames-per-path", type=int, default=0)
    parser.add_argument("--piper-feed-value", type=float, default=0.7)
    parser.add_argument("--zero-orientation-delta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    branch_root = Path(args.branch_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    episodes: list[dict] = []
    feed_count = 0
    hold_count = 0
    for branch_dir in sorted(branch_root.glob("branch*"), key=lambda p: p.name):
        if not branch_dir.is_dir():
            continue
        task = branch_task(branch_dir.name, args)
        pose_files = {numeric_suffix(p, "pose"): p for p in (branch_dir / "path").glob("pose*.txt")}
        label_files = {numeric_suffix(p, "label"): p for p in (branch_dir / "piper").glob("label*.txt")}
        image_dirs = {numeric_suffix(p, "path"): p for p in (branch_dir / "image1").glob("path*") if p.is_dir()}
        for path_idx in sorted(set(pose_files) & set(label_files) & set(image_dirs)):
            poses = read_pose_file(pose_files[path_idx])
            labels = read_label_file(label_files[path_idx])
            images = sorted(image_dirs[path_idx].glob("*.png"), key=lambda p: numeric_suffix(p, ""))
            # Need t+1 pose for Elite TCP-delta label.
            n = min(len(poses) - 1, len(labels), len(images) - 1)
            if args.max_frames_per_path and args.max_frames_per_path > 0:
                n = min(n, int(args.max_frames_per_path))
            piper_step = 0
            episode_name = f"{branch_dir.name}_path{path_idx}"
            episode_samples = 0
            for i in range(max(n, 0)):
                frame_number = numeric_suffix(images[i], "")
                sample = build_sample(
                    branch_name=branch_dir.name,
                    path_name=f"path{path_idx}",
                    frame_number=frame_number,
                    image_path=images[i],
                    pose=poses[i],
                    next_pose=poses[i + 1],
                    piper_step_before_action=piper_step,
                    piper_label=labels[i],
                    task=task,
                    zero_orientation_delta=bool(args.zero_orientation_delta),
                    piper_feed_value=float(args.piper_feed_value),
                )
                samples.append(sample)
                episode_samples += 1
                if labels[i] > 0:
                    piper_step += 1
                    feed_count += 1
                else:
                    hold_count += 1
            episodes.append({"episode": episode_name, "task": task, "success": True, "samples": episode_samples})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "branchs_real_shadow_training",
        "source": str(branch_root),
        "samples": samples,
        "episodes": episodes,
        "action_mode": "piper_feed_elite_joint",
        "observation_schema_recommended": "senior_piper_real_like",
        "elite_action_representation_recommended": "tcp_delta",
        "piper_head_recommended": "step_classification",
        "image_semantics": "single real camera duplicated as side/top for compatibility",
        "label_semantics": {
            "elite_tcp_delta_6d": "next Elite TCP pose minus current pose; orientation delta zeroed by default",
            "piper_step_command": "senior Piper label, 0=hold, 1=feed",
        },
        "counts": {
            "samples": len(samples),
            "feed": feed_count,
            "hold": hold_count,
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"wrote branchs training manifest to {out_path}")


if __name__ == "__main__":
    main()
