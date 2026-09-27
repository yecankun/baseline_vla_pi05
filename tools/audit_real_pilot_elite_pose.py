from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def pose_array(values: list[Any], *, name: str) -> np.ndarray:
    if not values:
        return np.empty((0, 6), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx6 array, got {array.shape}")
    return array


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def scalar_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def vector_min_max_span(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "span": np.ptp(values, axis=0).tolist(),
    }


def motion_summary(poses: np.ndarray) -> dict[str, Any]:
    xyz = poses[:, :3]
    rpy = poses[:, 3:6]
    relative_rpy = wrap_radians(rpy - rpy[0])
    delta_xyz = np.diff(xyz, axis=0)
    delta_rpy = wrap_radians(np.diff(rpy, axis=0))
    translation_step = np.linalg.norm(delta_xyz, axis=1)
    rotation_step_deg = np.rad2deg(np.linalg.norm(delta_rpy, axis=1))
    max_translation_index = int(np.argmax(translation_step)) + 1 if translation_step.size else None
    max_rotation_index = int(np.argmax(rotation_step_deg)) + 1 if rotation_step_deg.size else None
    return {
        "records": int(poses.shape[0]),
        "initial_pose_xyz_mm_rpy_rad": poses[0].tolist(),
        "final_pose_xyz_mm_rpy_rad": poses[-1].tolist(),
        "position_mm": vector_min_max_span(xyz),
        "position_total_displacement_mm": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "position_path_length_mm": float(translation_step.sum()),
        "orientation_absolute_rad": vector_min_max_span(rpy),
        "orientation_relative_deg": vector_min_max_span(np.rad2deg(relative_rpy)),
        "orientation_max_abs_from_initial_deg": np.max(np.abs(np.rad2deg(relative_rpy)), axis=0).tolist(),
        "consecutive_translation_l2_mm": scalar_stats(translation_step),
        "consecutive_translation_l2_mm_when_moving": scalar_stats(translation_step[translation_step > 0.01]),
        "consecutive_rotation_l2_deg": scalar_stats(rotation_step_deg),
        "consecutive_rotation_l2_deg_when_position_moving": scalar_stats(rotation_step_deg[translation_step > 0.01]),
        "moving_record_pairs_translation_gt_0_01mm": int(np.sum(translation_step > 0.01)),
        "moving_record_pairs_rotation_gt_0_01deg": int(np.sum(rotation_step_deg > 0.01)),
        "max_translation_step_record_index": max_translation_index,
        "max_rotation_step_record_index": max_rotation_index,
    }


def read_xyz_path(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.replace(",", " ").split()]
        if values:
            rows.append(values)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    width = {len(row) for row in rows}
    if len(width) != 1:
        raise ValueError(f"path file has inconsistent column counts: {sorted(width)}")
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Elite TCP position/orientation in a real pilot JSONL.")
    parser.add_argument("dataset", type=Path, help="Pilot directory or records.jsonl path.")
    parser.add_argument("--path-file", type=Path, help="Optional Elite path file used by the collector.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset if args.dataset.is_dir() else args.dataset.parent
    records_path = dataset_dir / "records.jsonl" if args.dataset.is_dir() else args.dataset
    rows = load_jsonl(records_path)
    if not rows:
        raise ValueError(f"no records found: {records_path}")

    state_poses: list[Any] = []
    executed_poses: list[Any] = []
    reference_poses: list[Any] = []
    requested_poses: list[Any] = []
    reference_deltas: list[Any] = []
    requested_row_indices: list[int] = []
    timestamps: list[float] = []
    image_pose_ms: list[float] = []
    stale_flags: list[bool] = []
    path_indices: list[int] = []
    path_index_by_record: list[int | None] = []

    for record_index, row in enumerate(rows):
        state = row.get("state", {})
        controller = state.get("controller_state", {})
        state_poses.append(state.get("elite_tcp_pose_6d"))
        executed_poses.append(controller.get("elite_executed_tcp_pose_6d"))
        reference = row.get("reference_action", {})
        reference_poses.append(reference.get("elite_tcp_pose_6d"))
        reference_deltas.append(reference.get("elite_tcp_delta_6d"))
        requested = controller.get("elite_requested_tcp_pose_6d")
        if requested is not None:
            requested_poses.append(requested)
            requested_row_indices.append(record_index)
        path_index = controller.get("elite_path_index")
        if isinstance(path_index, int):
            path_indices.append(path_index)
            path_index_by_record.append(path_index)
        else:
            path_index_by_record.append(None)
        timestamps.append(float(row["timestamp"]))
        image_pose_ms.append(float(row.get("sync", {}).get("image_pose_time_delta_ms", np.nan)))
        stale_flags.append(bool(controller.get("elite_pose_stale", False)))

    state_values = pose_array(state_poses, name="state Elite poses")
    executed_values = pose_array(executed_poses, name="executed Elite poses")
    reference_values = pose_array(reference_poses, name="reference Elite poses")
    reference_delta_values = pose_array(reference_deltas, name="reference Elite deltas")
    requested_values = pose_array(requested_poses, name="requested Elite poses")

    state_executed_abs = np.abs(state_values - executed_values)
    state_reference_abs = np.abs(state_values - reference_values)
    reference_delta_rotation_deg = np.rad2deg(np.abs(reference_delta_values[:, 3:6]))
    reference_delta_translation = np.linalg.norm(reference_delta_values[:, :3], axis=1)
    finite_image_pose_ms = np.asarray(image_pose_ms, dtype=np.float64)
    finite_image_pose_ms = finite_image_pose_ms[np.isfinite(finite_image_pose_ms)]
    timestamp_values = np.asarray(timestamps, dtype=np.float64)
    executed_delta_xyz = np.diff(executed_values[:, :3], axis=0)
    executed_delta_rpy = wrap_radians(np.diff(executed_values[:, 3:6], axis=0))
    executed_translation = np.linalg.norm(executed_delta_xyz, axis=1)
    executed_rotation_deg = np.rad2deg(np.linalg.norm(executed_delta_rpy, axis=1))
    movement_events: list[dict[str, Any]] = []
    for pair_index in np.flatnonzero(executed_translation > 0.01).tolist():
        record_index = pair_index + 1
        movement_events.append(
            {
                "record_index": int(record_index),
                "line_no": int(rows[record_index]["_line_no"]),
                "timestamp": float(timestamp_values[record_index]),
                "path_index": path_index_by_record[record_index],
                "translation_delta_xyz_mm": executed_delta_xyz[pair_index].tolist(),
                "translation_l2_mm": float(executed_translation[pair_index]),
                "rotation_l2_deg": float(executed_rotation_deg[pair_index]),
                "image_pose_time_delta_ms": float(image_pose_ms[record_index]),
                "pose_stale": bool(stale_flags[record_index]),
            }
        )

    requested_report: dict[str, Any] = {"records": 0}
    if requested_values.size:
        requested_state = state_values[np.asarray(requested_row_indices, dtype=np.int64)]
        requested_error_xyz = np.linalg.norm(requested_values[:, :3] - requested_state[:, :3], axis=1)
        requested_error_rpy = np.rad2deg(
            np.linalg.norm(wrap_radians(requested_values[:, 3:6] - requested_state[:, 3:6]), axis=1)
        )
        requested_report = {
            **motion_summary(requested_values),
            "executed_to_requested_translation_l2_mm": scalar_stats(requested_error_xyz),
            "executed_to_requested_rotation_l2_deg": scalar_stats(requested_error_rpy),
        }

    manifest = load_json(dataset_dir / "manifest.json")
    summary = load_json(dataset_dir / "summary.json")
    report: dict[str, Any] = {
        "scope": "read-only Elite pose reference audit; failed pilot episode, not training-data acceptance",
        "dataset": str(dataset_dir),
        "records_path": str(records_path),
        "records": len(rows),
        "manifest_pose_semantics": manifest.get("pose_semantics"),
        "manifest_zero_orientation_delta": manifest.get("zero_orientation_delta"),
        "manifest_elite_path": manifest.get("elite_path"),
        "run_summary": {
            "collection_mode": summary.get("collection_mode"),
            "final_piper_step": summary.get("final_piper_step"),
            "piper_command_counts": summary.get("piper_command_counts"),
        },
        "timing": {
            "duration_s": float(timestamp_values[-1] - timestamp_values[0]),
            "record_interval_s": scalar_stats(np.diff(timestamp_values)),
            "image_pose_time_delta_ms": scalar_stats(finite_image_pose_ms),
            "elite_pose_stale_records": int(sum(stale_flags)),
            "elite_pose_stale_fraction": float(np.mean(stale_flags)),
        },
        "executed_pose": motion_summary(executed_values),
        "executed_movement_events": movement_events,
        "requested_pose": requested_report,
        "path_indices": {
            "observed_min": min(path_indices) if path_indices else None,
            "observed_max": max(path_indices) if path_indices else None,
            "observed_unique": sorted(set(path_indices)),
        },
        "field_consistency": {
            "state_vs_controller_executed_max_abs_by_dim": state_executed_abs.max(axis=0).tolist(),
            "state_vs_reference_pose_max_abs_by_dim": state_reference_abs.max(axis=0).tolist(),
        },
        "recorded_reference_delta": {
            "translation_nonzero_records_gt_0_01mm": int(np.sum(reference_delta_translation > 0.01)),
            "rotation_nonzero_records_gt_0_01deg": int(np.sum(np.max(reference_delta_rotation_deg, axis=1) > 0.01)),
            "translation_l2_mm": scalar_stats(reference_delta_translation),
            "rotation_max_abs_deg": scalar_stats(np.max(reference_delta_rotation_deg, axis=1)),
        },
        "training_eligibility": {
            "accepted": False,
            "reason": "episode failed; Elite trajectory may be used only as motion-reference evidence",
        },
    }

    if args.path_file is not None:
        path_values = read_xyz_path(args.path_file)
        path_step = np.linalg.norm(np.diff(path_values[:, :3], axis=0), axis=1) if len(path_values) > 1 else np.empty(0)
        report["path_file"] = {
            "path": str(args.path_file),
            "rows": int(path_values.shape[0]),
            "columns": int(path_values.shape[1]) if path_values.ndim == 2 else 0,
            "xyz_only": bool(path_values.ndim == 2 and path_values.shape[1] == 3),
            "position_mm": vector_min_max_span(path_values[:, :3]) if path_values.size else None,
            "consecutive_target_translation_l2_mm": scalar_stats(path_step),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
