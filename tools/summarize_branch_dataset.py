from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def parse_numeric_line(line: str) -> list[float]:
    text = line.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    return [float(p) for p in parts]


def read_matrix(path: Path, expected_min_cols: int = 1) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = parse_numeric_line(line)
        if len(values) >= expected_min_cols:
            rows.append(values)
    if not rows:
        return np.empty((0, expected_min_cols), dtype=float)
    width = max(len(row) for row in rows)
    padded = [row + [np.nan] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=float)


def count_images(path_dir: Path) -> int:
    if not path_dir.exists():
        return 0
    return len([p for p in path_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def summarize_path(branch_dir: Path, path_index: int) -> dict:
    pose_path = branch_dir / "path" / f"pose{path_index}.txt"
    pos_path = branch_dir / "camera1" / f"pos{path_index}.txt"
    label_path = branch_dir / "piper" / f"label{path_index}.txt"
    image_dir = branch_dir / "image1" / f"path{path_index}"

    pose = read_matrix(pose_path, expected_min_cols=6) if pose_path.exists() else np.empty((0, 6), dtype=float)
    pos = read_matrix(pos_path, expected_min_cols=3) if pos_path.exists() else np.empty((0, 3), dtype=float)
    labels = read_matrix(label_path, expected_min_cols=1) if label_path.exists() else np.empty((0, 1), dtype=float)
    pose_xyz = pose[:, :3] if pose.shape[1] >= 3 else np.empty((0, 3), dtype=float)
    pose_rpy = pose[:, 3:6] if pose.shape[1] >= 6 else np.empty((0, 3), dtype=float)

    label_values = labels[:, 0].astype(int).tolist() if len(labels) else []
    label_counts = {str(value): int(label_values.count(value)) for value in sorted(set(label_values))}

    result = {
        "path_index": path_index,
        "frames": int(len(pose)),
        "images": int(count_images(image_dir)),
        "camera_pos_rows": int(len(pos)),
        "piper_label_rows": int(len(labels)),
        "piper_label_counts": label_counts,
    }
    if len(pose):
        deltas = np.diff(pose[:, :6], axis=0)
        result.update(
            {
                "initial_pose": [float(x) for x in pose[0, :6]],
                "final_pose": [float(x) for x in pose[-1, :6]],
                "xyz_min": [float(x) for x in np.nanmin(pose_xyz, axis=0)],
                "xyz_max": [float(x) for x in np.nanmax(pose_xyz, axis=0)],
                "xyz_span": [float(x) for x in (np.nanmax(pose_xyz, axis=0) - np.nanmin(pose_xyz, axis=0))],
                "rpy_min": [float(x) for x in np.nanmin(pose_rpy, axis=0)],
                "rpy_max": [float(x) for x in np.nanmax(pose_rpy, axis=0)],
                "mean_abs_delta_xyz": [float(x) for x in np.nanmean(np.abs(deltas[:, :3]), axis=0)] if len(deltas) else [0.0, 0.0, 0.0],
                "max_abs_delta_xyz": [float(x) for x in np.nanmax(np.abs(deltas[:, :3]), axis=0)] if len(deltas) else [0.0, 0.0, 0.0],
            }
        )
    if len(pos):
        result["camera_pos_first"] = [float(x) for x in pos[0, : min(4, pos.shape[1])]]
        result["camera_pos_last"] = [float(x) for x in pos[-1, : min(4, pos.shape[1])]]
    return result


def summarize_dataset(root: Path) -> dict:
    branches = []
    for branch_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("branch")], key=numeric_suffix):
        pose_files = sorted((branch_dir / "path").glob("pose*.txt"), key=numeric_suffix)
        paths = [summarize_path(branch_dir, numeric_suffix(path)) for path in pose_files]
        all_initial = np.asarray([item["initial_pose"] for item in paths if "initial_pose" in item], dtype=float)
        all_xyz_min = np.asarray([item["xyz_min"] for item in paths if "xyz_min" in item], dtype=float)
        all_xyz_max = np.asarray([item["xyz_max"] for item in paths if "xyz_max" in item], dtype=float)
        branch_summary = {
            "branch": branch_dir.name,
            "num_paths": len(paths),
            "total_frames": int(sum(item["frames"] for item in paths)),
            "total_images": int(sum(item["images"] for item in paths)),
            "paths": paths,
        }
        if len(all_initial):
            branch_summary["mean_initial_pose"] = [float(x) for x in np.nanmean(all_initial, axis=0)]
            branch_summary["xyz_min_all"] = [float(x) for x in np.nanmin(all_xyz_min, axis=0)]
            branch_summary["xyz_max_all"] = [float(x) for x in np.nanmax(all_xyz_max, axis=0)]
            branch_summary["xyz_span_all"] = [float(x) for x in (np.nanmax(all_xyz_max, axis=0) - np.nanmin(all_xyz_min, axis=0))]
        branches.append(branch_summary)
    return {
        "root": root.as_posix(),
        "branches": branches,
        "total_frames": int(sum(branch["total_frames"] for branch in branches)),
        "total_images": int(sum(branch["total_images"] for branch in branches)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real branch dataset poses, images, and piper labels.")
    parser.add_argument("--root", default="branchs")
    parser.add_argument("--out", default="simulation_output/real_branch_dataset_summary.json")
    args = parser.parse_args()

    report = summarize_dataset(Path(args.root).resolve())
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved {out_path}")
    print(f"total frames: {report['total_frames']}")
    print(f"total images: {report['total_images']}")
    for branch in report["branches"]:
        print(
            f"{branch['branch']}: paths={branch['num_paths']} frames={branch['total_frames']} "
            f"images={branch['total_images']} mean_initial={branch.get('mean_initial_pose')}"
        )


if __name__ == "__main__":
    main()
