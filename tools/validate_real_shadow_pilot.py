from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def numeric_list(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    return all(isinstance(x, (int, float)) for x in value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a real_shadow_pilot records.jsonl before shadow-mode evaluation."
    )
    parser.add_argument("dataset", help="Pilot directory or records.jsonl path.")
    parser.add_argument("--image-base-dir", help="Base directory for relative image paths. Defaults to dataset dir.")
    parser.add_argument("--max-side-top-ms", type=float, default=100.0)
    parser.add_argument("--max-image-pose-ms", type=float, default=100.0)
    parser.add_argument("--check-readable-images", action="store_true")
    parser.add_argument("--out", help="Optional JSON summary output path.")
    args = parser.parse_args()

    dataset_arg = Path(args.dataset)
    if dataset_arg.is_dir():
        dataset_dir = dataset_arg
        records_path = dataset_dir / "records.jsonl"
    else:
        records_path = dataset_arg
        dataset_dir = records_path.parent
    base_dir = Path(args.image_base_dir) if args.image_base_dir else dataset_dir

    errors: list[str] = []
    warnings: list[str] = []
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")

    manifest_path = dataset_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        warnings.append(f"missing manifest.json in {dataset_dir}")
    summary_path = dataset_dir / "summary.json"
    run_summary = {}
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    rows = load_jsonl(records_path)
    if not rows:
        errors.append("records.jsonl has no records")

    label_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    executed_counts: Counter[str] = Counter()
    piper_steps: list[float] = []
    side_top_ms: list[float] = []
    image_pose_ms: list[float] = []
    image_action_ms: list[float] = []
    elite_pose_age_ms: list[float] = []
    elite_pose_stale_count = 0
    elite_path_connection_modes: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    color_resolution_counts: Counter[str] = Counter()
    depth_resolution_counts: Counter[str] = Counter()

    camera_setup = manifest.get("camera_setup", {})
    record_depth = bool(camera_setup.get("record_depth"))
    depth_expected = {
        "side": record_depth and bool(camera_setup.get("side_runtime", {}).get("depth_enabled")),
        "top": record_depth and bool(camera_setup.get("top_runtime", {}).get("depth_enabled")),
    }
    for role in ("side", "top"):
        runtime = camera_setup.get(f"{role}_runtime", {})
        requested_color = camera_setup.get(f"{role}_color_requested", {})
        actual_color = runtime.get("color_profile", {})
        if runtime.get("source") in {"opencv", "realsense"}:
            for dimension in ("width", "height"):
                requested_value = requested_color.get(dimension)
                actual_value = actual_color.get(dimension)
                if isinstance(requested_value, (int, float)) and actual_value != requested_value:
                    errors.append(
                        f"manifest {role} actual color {dimension}={actual_value} "
                        f"does not match requested {requested_value}"
                    )
        if depth_expected[role]:
            if not isinstance(runtime.get("depth_scale_m_per_unit"), (int, float)):
                errors.append(f"manifest camera_setup.{role}_runtime is missing depth_scale_m_per_unit")
            if not isinstance(runtime.get("color_intrinsics"), dict):
                errors.append(f"manifest camera_setup.{role}_runtime is missing color_intrinsics")
            if not isinstance(runtime.get("aligned_depth_intrinsics"), dict):
                errors.append(f"manifest camera_setup.{role}_runtime is missing aligned_depth_intrinsics")

    cv2 = None
    if args.check_readable_images:
        import cv2 as _cv2

        cv2 = _cv2

    for idx, row in enumerate(rows):
        line = row.get("_line_no", idx + 1)
        task = row.get("task")
        if task not in {"left", "right"}:
            errors.append(f"line {line}: task should be left/right, got {task!r}")
        else:
            task_counts[str(task)] += 1
        if row.get("step") is None:
            errors.append(f"line {line}: missing step")

        color_images: dict[str, Any] = {}
        for image_key in ("side_image", "top_image"):
            image_ref = row.get(image_key)
            if not image_ref:
                errors.append(f"line {line}: missing {image_key}")
                continue
            image_path = resolve_path(str(image_ref), base_dir)
            if not image_path.exists():
                errors.append(f"line {line}: missing image file {image_path}")
            elif cv2 is not None:
                color_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if color_image is None:
                    errors.append(f"line {line}: unreadable image file {image_path}")
                else:
                    role = image_key.split("_", maxsplit=1)[0]
                    color_images[role] = color_image
                    color_resolution_counts[f"{role}:{color_image.shape[1]}x{color_image.shape[0]}"] += 1
                    actual_profile = camera_setup.get(f"{role}_runtime", {}).get("color_profile", {})
                    expected_shape = (actual_profile.get("height"), actual_profile.get("width"))
                    if all(isinstance(value, (int, float)) for value in expected_shape):
                        if color_image.shape[:2] != tuple(int(value) for value in expected_shape):
                            errors.append(
                                f"line {line}: {image_key} shape {color_image.shape[:2]} "
                                f"does not match runtime profile {expected_shape}"
                            )

        for role in ("side", "top"):
            depth_key = f"{role}_depth"
            depth_ref = row.get(depth_key)
            if not depth_ref:
                if depth_expected[role]:
                    errors.append(f"line {line}: missing {depth_key}")
                continue
            depth_counts[role] += 1
            depth_path = resolve_path(str(depth_ref), base_dir)
            if not depth_path.exists():
                errors.append(f"line {line}: missing depth file {depth_path}")
                continue
            if cv2 is not None:
                depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    errors.append(f"line {line}: unreadable depth file {depth_path}")
                elif str(depth_image.dtype) != "uint16":
                    errors.append(f"line {line}: {depth_key} must be uint16, got {depth_image.dtype}")
                else:
                    depth_resolution_counts[f"{role}:{depth_image.shape[1]}x{depth_image.shape[0]}"] += 1
                    if role in color_images and depth_image.shape[:2] != color_images[role].shape[:2]:
                        errors.append(
                            f"line {line}: aligned {depth_key} shape {depth_image.shape[:2]} "
                            f"does not match color shape {color_images[role].shape[:2]}"
                        )

        state = row.get("state", {})
        if not numeric_list(state.get("elite_tcp_pose_6d"), 6):
            errors.append(f"line {line}: state.elite_tcp_pose_6d must be six numeric values")
        piper_step = state.get("piper_step")
        if not isinstance(piper_step, (int, float)):
            errors.append(f"line {line}: state.piper_step must be numeric")
        else:
            piper_steps.append(float(piper_step))
        controller_state = state.get("controller_state", {})
        executed = controller_state.get("piper_executed_command")
        if executed is not None:
            executed_counts[str(executed)] += 1
        if controller_state.get("elite_pose_stale") is True:
            elite_pose_stale_count += 1
        pose_age = controller_state.get("elite_pose_age_ms")
        if isinstance(pose_age, (int, float)):
            elite_pose_age_ms.append(float(pose_age))
        connection_mode = controller_state.get("elite_path_connection_mode")
        if connection_mode is not None:
            elite_path_connection_modes[str(connection_mode)] += 1

        reference_action = row.get("reference_action", {})
        command = reference_action.get("piper_step_command")
        if command not in {-1, 0, 1}:
            errors.append(f"line {line}: reference_action.piper_step_command should be -1/0/1")
        else:
            label_counts[str(command)] += 1
        if not numeric_list(reference_action.get("elite_tcp_delta_6d"), 6):
            errors.append(f"line {line}: reference_action.elite_tcp_delta_6d must be six numeric values")

        sync = row.get("sync", {})
        side_top = sync.get("side_top_time_delta_ms")
        image_pose = sync.get("image_pose_time_delta_ms")
        image_action = sync.get("image_action_time_delta_ms")
        if isinstance(side_top, (int, float)):
            side_top_ms.append(float(side_top))
            if side_top > args.max_side_top_ms:
                warnings.append(f"line {line}: side/top delta {side_top:.1f} ms exceeds {args.max_side_top_ms:.1f}")
        else:
            warnings.append(f"line {line}: missing side_top_time_delta_ms")
        if isinstance(image_pose, (int, float)):
            image_pose_ms.append(float(image_pose))
            if image_pose > args.max_image_pose_ms:
                warnings.append(f"line {line}: image/pose delta {image_pose:.1f} ms exceeds {args.max_image_pose_ms:.1f}")
        else:
            warnings.append(f"line {line}: missing image_pose_time_delta_ms")
        if isinstance(image_action, (int, float)):
            image_action_ms.append(float(image_action))

    top_duplicated = manifest.get("camera_setup", {}).get("top_duplicated_from_side")
    if top_duplicated:
        warnings.append("top camera is duplicated from side according to manifest")
    if elite_pose_stale_count:
        warnings.append(
            f"{elite_pose_stale_count}/{len(rows)} records use a stale Elite pose; "
            "use a separate Elite command connection during path execution"
        )

    piper_step_decreases = 0
    for prev, cur in zip(piper_steps, piper_steps[1:]):
        if cur < prev:
            piper_step_decreases += 1

    summary = {
        "dataset": str(dataset_dir),
        "records_path": str(records_path),
        "records": len(rows),
        "errors": errors,
        "warnings": warnings[:50],
        "warning_count": len(warnings),
        "task_counts": dict(task_counts),
        "piper_step_command_counts": dict(label_counts),
        "piper_executed_command_counts": dict(executed_counts),
        "piper_step_min": min(piper_steps) if piper_steps else None,
        "piper_step_max": max(piper_steps) if piper_steps else None,
        "summary_final_piper_step": run_summary.get("final_piper_step"),
        "summary_auto_feed_count": run_summary.get("auto_feed_count"),
        "piper_step_decreases": piper_step_decreases,
        "depth_expected": depth_expected,
        "depth_file_counts": dict(depth_counts),
        "color_resolution_counts": dict(color_resolution_counts),
        "depth_resolution_counts": dict(depth_resolution_counts),
        "side_top_time_delta_ms_p95": percentile(side_top_ms, 0.95),
        "side_top_time_delta_ms_max": max(side_top_ms) if side_top_ms else None,
        "image_pose_time_delta_ms_p95": percentile(image_pose_ms, 0.95),
        "image_pose_time_delta_ms_max": max(image_pose_ms) if image_pose_ms else None,
        "elite_pose_stale_count": elite_pose_stale_count,
        "elite_pose_stale_fraction": elite_pose_stale_count / len(rows) if rows else None,
        "elite_pose_age_ms_p95": percentile(elite_pose_age_ms, 0.95),
        "elite_pose_age_ms_max": max(elite_pose_age_ms) if elite_pose_age_ms else None,
        "elite_path_connection_modes": dict(elite_path_connection_modes),
        "image_action_time_delta_ms_p95": percentile(image_action_ms, 0.95),
        "image_action_time_delta_ms_max": max(image_action_ms) if image_action_ms else None,
        "manifest_robot_command_mode": manifest.get("robot_command_mode"),
        "manifest_piper_label_source": manifest.get("piper_label_source"),
        "manifest_piper_control_source": manifest.get("piper_control_source"),
        "manifest_elite_control_source": manifest.get("elite_control_source"),
        "ok": not errors,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
