from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def numeric_suffix(path: Path, prefix: str) -> int:
    stem = path.stem
    if stem.startswith(prefix):
        return int(stem[len(prefix) :])
    return int(stem)


def read_pose_file(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = ast.literal_eval(line)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"Expected 6 pose values in {path}, got {arr.size}")
        rows.append(arr)
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.vstack(rows)


def read_label_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_camera_file(path: Path) -> list[list[float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rows.append([float(part) for part in parts])
    return rows


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="nearest")),
        "p99": float(np.quantile(values, 0.99, method="nearest")),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def image_shape(image_dir: Path) -> list[int] | None:
    images = sorted(image_dir.glob("*.png"), key=lambda p: numeric_suffix(p, ""))
    if not images:
        return None
    img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return [int(h), int(w), int(img.shape[2])]


def inspect_path(branch_dir: Path, pose_file: Path) -> dict:
    path_id = numeric_suffix(pose_file, "pose")
    pose = read_pose_file(pose_file)
    labels = read_label_file(branch_dir / "piper" / f"label{path_id}.txt")
    camera_rows = read_camera_file(branch_dir / "camera1" / f"pos{path_id}.txt")
    image_dir = branch_dir / "image1" / f"path{path_id}"
    images = sorted(image_dir.glob("*.png"), key=lambda p: numeric_suffix(p, ""))

    if len(pose) >= 2:
        delta = np.diff(pose, axis=0)
        pos_step = np.linalg.norm(delta[:, :3], axis=1)
        rot_step_linf = np.max(np.abs(delta[:, 3:]), axis=1)
        pose_step_linf = np.max(np.abs(delta), axis=1)
    else:
        pos_step = np.asarray([], dtype=np.float64)
        rot_step_linf = np.asarray([], dtype=np.float64)
        pose_step_linf = np.asarray([], dtype=np.float64)

    label_counts = Counter(labels)
    camera_arr = np.asarray(camera_rows, dtype=np.float64) if camera_rows else np.zeros((0, 3), dtype=np.float64)
    visible_camera_rows = int(np.sum(camera_arr[:, -1] > 0)) if camera_arr.size else 0

    return {
        "branch": branch_dir.name,
        "path_id": path_id,
        "pose_file": str(pose_file.as_posix()),
        "frames_pose": int(len(pose)),
        "frames_images": int(len(images)),
        "frames_piper_labels": int(len(labels)),
        "frames_camera_rows": int(len(camera_rows)),
        "image_shape": image_shape(image_dir),
        "label_counts": dict(sorted(label_counts.items())),
        "camera_visible_rows": visible_camera_rows,
        "pose_start": pose[0].tolist() if len(pose) else None,
        "pose_end": pose[-1].tolist() if len(pose) else None,
        "position_step": quantiles(pos_step),
        "rotation_step_linf": quantiles(rot_step_linf),
        "pose_step_linf": quantiles(pose_step_linf),
    }


def summarize(rows: list[dict]) -> dict:
    by_branch = {}
    for branch in sorted({row["branch"] for row in rows}):
        xs = [row for row in rows if row["branch"] == branch]
        pos_p95 = np.asarray([row["position_step"]["p95"] for row in xs], dtype=np.float64)
        pos_max = np.asarray([row["position_step"]["max"] for row in xs], dtype=np.float64)
        rot_p95 = np.asarray([row["rotation_step_linf"]["p95"] for row in xs], dtype=np.float64)
        frame_counts = np.asarray([row["frames_pose"] for row in xs], dtype=np.float64)
        label_counts = Counter()
        for row in xs:
            label_counts.update(row["label_counts"])
        by_branch[branch] = {
            "paths": len(xs),
            "pose_frames": int(np.sum(frame_counts)),
            "pose_frames_min": int(np.min(frame_counts)) if frame_counts.size else 0,
            "pose_frames_max": int(np.max(frame_counts)) if frame_counts.size else 0,
            "position_step_p95_across_paths": quantiles(pos_p95),
            "position_step_max_across_paths": quantiles(pos_max),
            "rotation_step_linf_p95_across_paths": quantiles(rot_p95),
            "piper_label_counts": dict(sorted(label_counts.items())),
        }
    return {
        "branches": by_branch,
        "total_paths": len(rows),
        "total_pose_frames": int(sum(row["frames_pose"] for row in rows)),
        "total_image_frames": int(sum(row["frames_images"] for row in rows)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "branch",
        "path_id",
        "frames_pose",
        "frames_images",
        "frames_piper_labels",
        "frames_camera_rows",
        "position_step_median",
        "position_step_p95",
        "position_step_max",
        "rotation_step_linf_p95",
        "rotation_step_linf_max",
        "pose_step_linf_p95",
        "pose_step_linf_max",
        "label_counts",
        "image_shape",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "branch": row["branch"],
                    "path_id": row["path_id"],
                    "frames_pose": row["frames_pose"],
                    "frames_images": row["frames_images"],
                    "frames_piper_labels": row["frames_piper_labels"],
                    "frames_camera_rows": row["frames_camera_rows"],
                    "position_step_median": row["position_step"]["median"],
                    "position_step_p95": row["position_step"]["p95"],
                    "position_step_max": row["position_step"]["max"],
                    "rotation_step_linf_p95": row["rotation_step_linf"]["p95"],
                    "rotation_step_linf_max": row["rotation_step_linf"]["max"],
                    "pose_step_linf_p95": row["pose_step_linf"]["p95"],
                    "pose_step_linf_max": row["pose_step_linf"]["max"],
                    "label_counts": json.dumps(row["label_counts"], ensure_ascii=False),
                    "image_shape": json.dumps(row["image_shape"], ensure_ascii=False),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real branch data under branchs/.")
    parser.add_argument("--root", default="branchs")
    parser.add_argument("--out", default="simulation_output/real_branch_data_inspection")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for branch_dir in sorted(root.glob("branch*")):
        pose_dir = branch_dir / "path"
        if not pose_dir.exists():
            continue
        pose_files = sorted(pose_dir.glob("pose*.txt"), key=lambda p: numeric_suffix(p, "pose"))
        for pose_file in pose_files:
            rows.append(inspect_path(branch_dir, pose_file))

    summary = summarize(rows)
    (out / "real_branch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "real_branch_paths.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "real_branch_paths.csv", rows)

    print(f"Inspected {summary['total_paths']} real paths from {root}")
    print(f"Total pose frames: {summary['total_pose_frames']}")
    print(f"Total image frames: {summary['total_image_frames']}")
    for branch, item in summary["branches"].items():
        p95 = item["position_step_p95_across_paths"]
        rot = item["rotation_step_linf_p95_across_paths"]
        print(
            f"{branch}: paths={item['paths']} pose_frames={item['pose_frames']} "
            f"pos_step_p95 median/max={p95['median']:.4f}/{p95['max']:.4f} "
            f"rot_step_linf_p95 median/max={rot['median']:.6f}/{rot['max']:.6f} "
            f"labels={item['piper_label_counts']}"
        )
    print(f"Saved summary to {out / 'real_branch_summary.json'}")


if __name__ == "__main__":
    main()
