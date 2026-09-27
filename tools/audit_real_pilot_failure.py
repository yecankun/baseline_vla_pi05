from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def resolve_dataset(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path, path / "records.jsonl"
    return path.parent, path


def resolve_record_image(dataset_dir: Path, row: dict[str, Any], key: str) -> Path | None:
    value = row.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return dataset_dir / path


def pose_xyz(row: dict[str, Any]) -> list[float] | None:
    pose = row.get("state", {}).get("elite_tcp_pose_6d")
    if isinstance(pose, list) and len(pose) >= 3 and all(isinstance(v, (int, float)) for v in pose[:3]):
        return [float(v) for v in pose[:3]]
    return None


def xyz_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def read_gray_resized(cv2: Any, path: Path, width: int = 160) -> Any | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height = max(1, int(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_stats(cv2: Any, path: Path) -> dict[str, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return {
        "mean": float(image.mean()),
        "std": float(image.std()),
        "blur_laplacian_var": float(cv2.Laplacian(image, cv2.CV_64F).var()),
    }


def image_diffs(cv2: Any, dataset_dir: Path, rows: list[dict[str, Any]], image_key: str) -> list[float | None]:
    diffs: list[float | None] = [None]
    previous = None
    for row in rows:
        path = resolve_record_image(dataset_dir, row, image_key)
        current = read_gray_resized(cv2, path) if path is not None else None
        if previous is None or current is None or previous.shape != current.shape:
            diffs.append(None)
        else:
            diffs.append(float(cv2.absdiff(previous, current).mean()))
        previous = current
    return diffs[: len(rows)]


def image_stat_samples(cv2: Any, dataset_dir: Path, rows: list[dict[str, Any]], image_key: str) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    indices = sorted(set([0, len(rows) // 4, len(rows) // 2, (len(rows) * 3) // 4, len(rows) - 1]))
    stats = []
    for index in indices:
        path = resolve_record_image(dataset_dir, rows[index], image_key)
        if path is None:
            continue
        one = image_stats(cv2, path)
        if one is not None:
            stats.append(one)
    if not stats:
        return {"samples": 0}
    return {
        "samples": len(stats),
        "mean_luminance_median": percentile([s["mean"] for s in stats], 0.5),
        "mean_luminance_min": min(s["mean"] for s in stats),
        "mean_luminance_max": max(s["mean"] for s in stats),
        "blur_laplacian_var_median": percentile([s["blur_laplacian_var"] for s in stats], 0.5),
    }


def command_label(row: dict[str, Any]) -> int | None:
    value = row.get("reference_action", {}).get("piper_step_command")
    if isinstance(value, int):
        return value
    return None


def controller_state(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state", {})
    if not isinstance(state, dict):
        return {}
    controller = state.get("controller_state", {})
    return controller if isinstance(controller, dict) else {}


def feed_event_steps(rows: list[dict[str, Any]]) -> list[int]:
    events: list[int] = []
    for index, row in enumerate(rows):
        controller = controller_state(row)
        executed = str(controller.get("piper_executed_command", ""))
        command = command_label(row)
        if command == 1 or "feed" in executed:
            events.append(index)
    return events


def feed_response_summary(diffs: list[float | None], events: list[int], window: int) -> dict[str, Any]:
    responses: list[float] = []
    per_event: list[dict[str, Any]] = []
    for event in events:
        window_values = [v for v in diffs[event + 1 : event + 1 + window] if isinstance(v, (int, float))]
        response = percentile([float(v) for v in window_values], 0.5)
        per_event.append({"step": int(event), "median_next_frame_diff": response})
        if response is not None:
            responses.append(float(response))
    early = responses[: max(1, min(3, len(responses)))]
    late = responses[-max(1, min(3, len(responses))) :] if responses else []
    early_median = percentile(early, 0.5)
    late_median = percentile(late, 0.5)
    ratio = None
    if early_median is not None and early_median > 1e-6 and late_median is not None:
        ratio = float(late_median / early_median)
    return {
        "events": len(events),
        "response_median": percentile(responses, 0.5),
        "response_p95": percentile(responses, 0.95),
        "early_response_median": early_median,
        "late_response_median": late_median,
        "late_over_early_ratio": ratio,
        "per_event": per_event,
    }


def representative_indices(rows: list[dict[str, Any]], events: list[int]) -> list[int]:
    if not rows:
        return []
    indices: set[int] = {0, len(rows) - 1, len(rows) // 2}
    for event in events:
        for offset in (-1, 0, 1, 3):
            candidate = event + offset
            if 0 <= candidate < len(rows):
                indices.add(candidate)
    return sorted(indices)


def annotate_image(cv2: Any, image: Any, lines: list[str]) -> Any:
    result = image.copy()
    line_height = 22
    pad = 8
    overlay_height = pad * 2 + line_height * len(lines)
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (result.shape[1], overlay_height), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.55, result, 0.45, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (pad, pad + line_height * (index + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def make_contact_sheet(
    cv2: Any,
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    indices: list[int],
    out_path: Path,
    max_tiles: int,
) -> str | None:
    tiles = []
    selected = indices[:max_tiles]
    for index in selected:
        row = rows[index]
        side_path = resolve_record_image(dataset_dir, row, "side_image")
        top_path = resolve_record_image(dataset_dir, row, "top_image")
        side = cv2.imread(str(side_path)) if side_path is not None else None
        top = cv2.imread(str(top_path)) if top_path is not None else None
        if side is None and top is None:
            continue
        if side is None:
            side = top.copy()
        if top is None:
            top = side.copy()
        height = 180
        side = cv2.resize(side, (int(side.shape[1] * height / side.shape[0]), height), interpolation=cv2.INTER_AREA)
        top = cv2.resize(top, (int(top.shape[1] * height / top.shape[0]), height), interpolation=cv2.INTER_AREA)
        if side.shape[1] != top.shape[1]:
            target_width = min(side.shape[1], top.shape[1])
            side = cv2.resize(side, (target_width, height), interpolation=cv2.INTER_AREA)
            top = cv2.resize(top, (target_width, height), interpolation=cv2.INTER_AREA)
        controller = controller_state(row)
        lines = [
            f"step={row.get('step')} idx={index} task={row.get('task')}",
            f"piper={row.get('state', {}).get('piper_step')} cmd={command_label(row)} exec={controller.get('piper_executed_command')}",
            f"elite_idx={controller.get('elite_path_index')} status={controller.get('elite_path_status')}",
        ]
        tile = annotate_image(cv2, cv2.vconcat([side, top]), lines)
        tiles.append(tile)
    if not tiles:
        return None
    max_width = max(tile.shape[1] for tile in tiles)
    normalized = []
    for tile in tiles:
        if tile.shape[1] < max_width:
            pad = max_width - tile.shape[1]
            tile = cv2.copyMakeBorder(tile, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        normalized.append(tile)
    columns = min(4, len(normalized))
    rows_tiles = []
    for start in range(0, len(normalized), columns):
        row_tiles = normalized[start : start + columns]
        if len(row_tiles) < columns:
            blank = np.zeros_like(row_tiles[0])
            row_tiles.extend([blank] * (columns - len(row_tiles)))
        rows_tiles.append(cv2.hconcat(row_tiles))
    sheet = cv2.vconcat(rows_tiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return str(out_path)


def audit_dataset(dataset_arg: Path, output_dir: Path, max_tiles: int, feed_window: int) -> dict[str, Any]:
    dataset_dir, records_path = resolve_dataset(dataset_arg)
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")

    import cv2

    rows = load_jsonl(records_path)
    manifest = load_json(dataset_dir / "manifest.json")
    summary = load_json(dataset_dir / "summary.json")

    commands: Counter[int] = Counter()
    executed: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    elite_indices: list[int] = []
    stale_count = 0
    stale_ages: list[float] = []
    side_top_ms: list[float] = []
    image_pose_ms: list[float] = []
    image_action_ms: list[float] = []
    piper_steps: list[float] = []
    pose_steps: list[float] = []
    missing_images = 0

    previous_xyz = None
    for row in rows:
        tasks[str(row.get("task"))] += 1
        command = command_label(row)
        if command is not None:
            commands[command] += 1
        controller = controller_state(row)
        if controller.get("piper_executed_command") is not None:
            executed[str(controller.get("piper_executed_command"))] += 1
        index = controller.get("elite_path_index")
        if isinstance(index, int):
            elite_indices.append(index)
        if controller.get("elite_pose_stale"):
            stale_count += 1
        age = controller.get("elite_pose_age_ms")
        if isinstance(age, (int, float)):
            stale_ages.append(float(age))
        sync = row.get("sync", {})
        for target, key in (
            (side_top_ms, "side_top_time_delta_ms"),
            (image_pose_ms, "image_pose_time_delta_ms"),
            (image_action_ms, "image_action_time_delta_ms"),
        ):
            value = sync.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))
        piper_step = row.get("state", {}).get("piper_step")
        if isinstance(piper_step, (int, float)):
            piper_steps.append(float(piper_step))
        xyz = pose_xyz(row)
        if previous_xyz is not None and xyz is not None:
            pose_steps.append(xyz_distance(previous_xyz, xyz))
        previous_xyz = xyz
        for image_key in ("side_image", "top_image"):
            image_path = resolve_record_image(dataset_dir, row, image_key)
            if image_path is None or not image_path.exists():
                missing_images += 1

    side_diffs = image_diffs(cv2, dataset_dir, rows, "side_image")
    top_diffs = image_diffs(cv2, dataset_dir, rows, "top_image")
    events = feed_event_steps(rows)
    side_feed_response = feed_response_summary(side_diffs, events, feed_window)
    top_feed_response = feed_response_summary(top_diffs, events, feed_window)
    sheet_indices = representative_indices(rows, events)
    safe_name = dataset_dir.name
    sheet_path = make_contact_sheet(
        cv2,
        dataset_dir,
        rows,
        sheet_indices,
        output_dir / f"{safe_name}_contact_sheet.jpg",
        max_tiles=max_tiles,
    )

    piper_step_decreases = sum(1 for prev, cur in zip(piper_steps, piper_steps[1:]) if cur < prev)
    result = {
        "dataset": str(dataset_dir),
        "records_path": str(records_path),
        "records": len(rows),
        "tasks": counter_to_dict(tasks),
        "manifest_robot_command_mode": manifest.get("robot_command_mode"),
        "manifest_collection_mode": manifest.get("collection_mode"),
        "manifest_piper_label_source": manifest.get("piper_label_source"),
        "manifest_piper_control_source": manifest.get("piper_control_source"),
        "manifest_elite_control_source": manifest.get("elite_control_source"),
        "summary_final_piper_step": summary.get("final_piper_step"),
        "summary_stopped_by_user": summary.get("stopped_by_user"),
        "piper_step_min": min(piper_steps) if piper_steps else None,
        "piper_step_max": max(piper_steps) if piper_steps else None,
        "piper_step_decreases": piper_step_decreases,
        "piper_step_command_counts": counter_to_dict(commands),
        "piper_executed_command_counts": counter_to_dict(executed),
        "feed_event_steps": events,
        "elite_path_index_min": min(elite_indices) if elite_indices else None,
        "elite_path_index_max": max(elite_indices) if elite_indices else None,
        "elite_path_index_last": elite_indices[-1] if elite_indices else None,
        "elite_pose_stale_count": stale_count,
        "elite_pose_age_ms_p95": percentile(stale_ages, 0.95),
        "sync": {
            "side_top_time_delta_ms_p95": percentile(side_top_ms, 0.95),
            "side_top_time_delta_ms_max": max(side_top_ms) if side_top_ms else None,
            "image_pose_time_delta_ms_p95": percentile(image_pose_ms, 0.95),
            "image_pose_time_delta_ms_max": max(image_pose_ms) if image_pose_ms else None,
            "image_action_time_delta_ms_p95": percentile(image_action_ms, 0.95),
            "image_action_time_delta_ms_max": max(image_action_ms) if image_action_ms else None,
        },
        "elite_pose_step_mm": {
            "median": percentile(pose_steps, 0.5),
            "p95": percentile(pose_steps, 0.95),
            "max": max(pose_steps) if pose_steps else None,
        },
        "image_motion": {
            "side_adjacent_diff_median": percentile([v for v in side_diffs if v is not None], 0.5),
            "side_adjacent_diff_p95": percentile([v for v in side_diffs if v is not None], 0.95),
            "top_adjacent_diff_median": percentile([v for v in top_diffs if v is not None], 0.5),
            "top_adjacent_diff_p95": percentile([v for v in top_diffs if v is not None], 0.95),
            "side_feed_response": side_feed_response,
            "top_feed_response": top_feed_response,
        },
        "image_quality_samples": {
            "side": image_stat_samples(cv2, dataset_dir, rows, "side_image"),
            "top": image_stat_samples(cv2, dataset_dir, rows, "top_image"),
        },
        "missing_image_refs": missing_images,
        "contact_sheet": sheet_path,
        "interpretation": [
            "This audit is a data-integrity and visual-review aid; it cannot prove mechanical sticking by itself.",
            "Treat this run as failed/stuck pilot data if onsite observation confirms guidewire blockage.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{safe_name}_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_markdown(results: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Real Pilot Failure Audit",
        "",
        "This report summarizes local integrity and timing checks for real pilot datasets.",
        "It is a review aid, not an automatic physical-cause detector.",
        "",
    ]
    for result in results:
        image_pose_p95 = result["sync"].get("image_pose_time_delta_ms_p95")
        image_pose_max = result["sync"].get("image_pose_time_delta_ms_max")
        sync_flag = "ok"
        if isinstance(image_pose_p95, (int, float)) and image_pose_p95 > 100.0:
            sync_flag = "not train/eval suitable"
        lines.extend(
            [
                f"## {Path(result['dataset']).name}",
                "",
                f"- records: {result['records']}",
                f"- tasks: `{result['tasks']}`",
                f"- robot command mode: `{result.get('manifest_robot_command_mode')}`",
                f"- piper final step: `{result.get('summary_final_piper_step')}`",
                f"- piper commands: `{result.get('piper_step_command_counts')}`",
                f"- piper executed: `{result.get('piper_executed_command_counts')}`",
                f"- elite path index max/last: `{result.get('elite_path_index_max')}` / `{result.get('elite_path_index_last')}`",
                f"- sync image-pose p95/max ms: `{result['sync'].get('image_pose_time_delta_ms_p95')}` / `{result['sync'].get('image_pose_time_delta_ms_max')}`",
                f"- sync image-action p95/max ms: `{result['sync'].get('image_action_time_delta_ms_p95')}` / `{result['sync'].get('image_action_time_delta_ms_max')}`",
                f"- synchronization judgment: `{sync_flag}`",
                f"- Elite pose step p95/max mm: `{result['elite_pose_step_mm'].get('p95')}` / `{result['elite_pose_step_mm'].get('max')}`",
                f"- side image adjacent diff median/p95: `{result['image_motion'].get('side_adjacent_diff_median')}` / `{result['image_motion'].get('side_adjacent_diff_p95')}`",
                f"- top image adjacent diff median/p95: `{result['image_motion'].get('top_adjacent_diff_median')}` / `{result['image_motion'].get('top_adjacent_diff_p95')}`",
                f"- contact sheet: `{result.get('contact_sheet')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Current Conclusion",
            "",
            "The current onsite pilot runs should be treated as failed/stuck pilot data, not successful demonstrations.",
            "They are still useful for validating the real collection schema, camera setup, command logs,",
            "and for locating the physical guidewire blockage before the next lab visit. They should not be used",
            "as successful policy-training or strict shadow-evaluation data because the old collection run shows",
            "large image-pose/action timestamp deltas.",
            "",
            "Next local use:",
            "",
            "1. Review the generated contact sheets around feed events.",
            "2. Compare the stuck frames with the physical vessel groove/entry geometry.",
            "3. Keep these datasets out of successful policy-training manifests.",
            "4. Re-run this audit immediately after the next onsite collection.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real pilot datasets and generate visual review artifacts.")
    parser.add_argument("datasets", nargs="+", help="Dataset directories or records.jsonl files.")
    parser.add_argument("--out", default="docs/_real_pilot_failure_audit", help="Output directory for JSON and sheets.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report path.")
    parser.add_argument("--max-tiles", type=int, default=28)
    parser.add_argument("--feed-response-window", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.out)
    results = [
        audit_dataset(Path(dataset), output_dir=output_dir, max_tiles=args.max_tiles, feed_window=args.feed_response_window)
        for dataset in args.datasets
    ]
    if args.markdown_out:
        write_markdown(results, Path(args.markdown_out))
    print(json.dumps({"output_dir": str(output_dir), "datasets": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
