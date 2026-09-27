from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def resolve_record_path(dataset_dir: Path, path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = dataset_dir / path
    return path


def read_image(dataset_dir: Path, row: dict[str, Any], camera: str) -> np.ndarray | None:
    path = resolve_record_path(dataset_dir, row.get(f"{camera}_image"))
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def get_controller_state(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    if not isinstance(state, dict):
        return {}
    controller = state.get("controller_state")
    return controller if isinstance(controller, dict) else {}


def get_reference_action(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("reference_action")
    return ref if isinstance(ref, dict) else {}


def get_pose(row: dict[str, Any]) -> list[float] | None:
    state = row.get("state")
    if not isinstance(state, dict):
        return None
    pose = state.get("elite_tcp_pose_6d")
    if not isinstance(pose, list) or len(pose) < 6:
        return None
    if not all(isinstance(x, (int, float)) for x in pose[:6]):
        return None
    return [float(x) for x in pose[:6]]


def event_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    controller = get_controller_state(row)
    event = controller.get("user_event")
    if isinstance(event, dict):
        return event
    ref = get_reference_action(row)
    if ref.get("piper_step_command") == 1:
        return {
            "type": "piper_command",
            "key": None,
            "piper_command": 1,
            "piper_burst_count": ref.get("piper_burst_count", 1),
            "elite_command": None,
            "elite_submit_status": None,
        }
    return None


def image_diff_metrics(
    dataset_dir: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    camera: str,
) -> dict[str, float | None]:
    img_a = read_image(dataset_dir, before, camera)
    img_b = read_image(dataset_dir, after, camera)
    if img_a is None or img_b is None:
        return {
            f"{camera}_absdiff_mean": None,
            f"{camera}_absdiff_p95": None,
            f"{camera}_motion_frac_gt15": None,
        }
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return {
        f"{camera}_absdiff_mean": float(np.mean(diff)),
        f"{camera}_absdiff_p95": float(np.percentile(diff, 95)),
        f"{camera}_motion_frac_gt15": float(np.mean(diff > 15.0)),
    }


def pose_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    pose_a = get_pose(before)
    pose_b = get_pose(after)
    if pose_a is None or pose_b is None:
        return {
            "elite_tcp_delta_mm_l2": None,
            "elite_tcp_delta_mm_linf": None,
            "elite_rot_delta_l2": None,
        }
    pos_delta = np.asarray(pose_b[:3], dtype=np.float64) - np.asarray(pose_a[:3], dtype=np.float64)
    rot_delta = np.asarray(pose_b[3:6], dtype=np.float64) - np.asarray(pose_a[3:6], dtype=np.float64)
    return {
        "elite_tcp_delta_mm_l2": float(np.linalg.norm(pos_delta)),
        "elite_tcp_delta_mm_linf": float(np.max(np.abs(pos_delta))),
        "elite_rot_delta_l2": float(np.linalg.norm(rot_delta)),
    }


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    event_index: int,
    out_path: Path,
    pre_frames: int,
    post_frames: int,
    thumb_width: int,
) -> None:
    offsets = [-pre_frames, 0, max(1, post_frames // 2), post_frames]
    row_images: list[np.ndarray] = []
    for camera in ("side", "top"):
        cells: list[np.ndarray] = []
        for offset in offsets:
            idx = min(len(rows) - 1, max(0, event_index + offset))
            image = read_image(dataset_dir, rows[idx], camera)
            if image is None:
                image = np.zeros((120, 160, 3), dtype=np.uint8)
            scale = thumb_width / max(1, image.shape[1])
            thumb = cv2.resize(
                image,
                (thumb_width, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            cells.append(put_label(thumb, f"{camera} idx={idx} off={offset}"))
        row_images.append(np.concatenate(cells, axis=1))
    min_width = min(img.shape[1] for img in row_images)
    row_images = [img[:, :min_width] for img in row_images]
    sheet = np.concatenate(row_images, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def audit_dataset(
    dataset_dir: Path,
    out_dir: Path,
    pre_frames: int,
    post_frames: int,
    max_sheets_per_dataset: int,
    thumb_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_path = dataset_dir / "records.jsonl"
    rows = load_jsonl(records_path)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    events: list[dict[str, Any]] = []
    image_pose_ms: list[float] = []
    side_top_ms: list[float] = []

    for idx, row in enumerate(rows):
        sync = row.get("sync") if isinstance(row.get("sync"), dict) else {}
        if isinstance(sync.get("image_pose_time_delta_ms"), (int, float)):
            image_pose_ms.append(float(sync["image_pose_time_delta_ms"]))
        if isinstance(sync.get("side_top_time_delta_ms"), (int, float)):
            side_top_ms.append(float(sync["side_top_time_delta_ms"]))
        event = event_from_row(row)
        if event is None:
            continue
        before_idx = min(len(rows) - 1, max(0, idx - pre_frames))
        after_idx = min(len(rows) - 1, max(0, idx + post_frames))
        before = rows[before_idx]
        after = rows[after_idx]
        ref = get_reference_action(row)
        controller = get_controller_state(row)
        item: dict[str, Any] = {
            "dataset": dataset_dir.name,
            "event_id": len(events),
            "line_no": row.get("_line_no"),
            "record_index": idx,
            "before_index": before_idx,
            "after_index": after_idx,
            "timestamp": row.get("timestamp"),
            "task": row.get("task", manifest.get("task")),
            "event_type": event.get("type"),
            "key": event.get("key"),
            "piper_command": event.get("piper_command", ref.get("piper_step_command")),
            "piper_burst_count": event.get("piper_burst_count", ref.get("piper_burst_count")),
            "elite_command": event.get("elite_command"),
            "elite_submit_status": event.get("elite_submit_status"),
            "piper_executed_command": controller.get("piper_executed_command"),
            "piper_step_before": get_controller_state(before).get("piper_step_count"),
            "piper_step_event": controller.get("piper_step_count"),
            "piper_step_after": get_controller_state(after).get("piper_step_count"),
            "elite_path_index_before": get_controller_state(before).get("elite_path_index"),
            "elite_path_index_event": controller.get("elite_path_index"),
            "elite_path_index_after": get_controller_state(after).get("elite_path_index"),
        }
        item.update(pose_delta(before, after))
        item.update(image_diff_metrics(dataset_dir, before, after, "side"))
        item.update(image_diff_metrics(dataset_dir, before, after, "top"))
        if len(events) < max_sheets_per_dataset:
            sheet_path = out_dir / "contact_sheets" / dataset_dir.name / f"event_{len(events):03d}.png"
            make_contact_sheet(dataset_dir, rows, idx, sheet_path, pre_frames, post_frames, thumb_width)
            item["contact_sheet"] = str(sheet_path)
        events.append(item)

    summary = {
        "dataset": dataset_dir.name,
        "records": len(rows),
        "events": len(events),
        "manifest_task": manifest.get("task"),
        "dataset_intent": manifest.get("dataset_intent"),
        "collection_mode": manifest.get("collection_mode"),
        "piper_control_source": manifest.get("piper_control_source"),
        "elite_control_source": manifest.get("elite_control_source"),
        "image_pose_time_delta_ms_p95": percentile(image_pose_ms, 0.95),
        "image_pose_time_delta_ms_max": max(image_pose_ms) if image_pose_ms else None,
        "side_top_time_delta_ms_p95": percentile(side_top_ms, 0.95),
        "side_top_time_delta_ms_max": max(side_top_ms) if side_top_ms else None,
    }
    return events, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real 20260707 alignment calibration datasets by key-event windows."
    )
    parser.add_argument(
        "--root",
        default="collected_data",
        help="Directory containing real_align_*/real_pilot_* datasets.",
    )
    parser.add_argument(
        "--out",
        default="docs/_real_alignment_20260707_audit",
        help="Output directory for summaries and contact sheets.",
    )
    parser.add_argument("--pattern", action="append", default=["real_align_20260707_*"])
    parser.add_argument("--pre-frames", type=int, default=2)
    parser.add_argument("--post-frames", type=int, default=10)
    parser.add_argument("--max-sheets-per-dataset", type=int, default=40)
    parser.add_argument("--thumb-width", type=int, default=180)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    datasets: list[Path] = []
    for pattern in args.pattern:
        datasets.extend(path for path in root.glob(pattern) if (path / "records.jsonl").exists())
    datasets = sorted(set(datasets))
    if not datasets:
        raise FileNotFoundError(f"No datasets found under {root} for patterns: {args.pattern}")

    all_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset_dir in datasets:
        events, summary = audit_dataset(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
            max_sheets_per_dataset=args.max_sheets_per_dataset,
            thumb_width=args.thumb_width,
        )
        all_events.extend(events)
        summaries.append(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in all_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_csv(out_dir / "events.csv", all_events)
    print(json.dumps({"datasets": len(summaries), "events": len(all_events), "out": str(out_dir)}, indent=2))
    for summary in summaries:
        print(
            f"{summary['dataset']}: records={summary['records']} events={summary['events']} "
            f"image_pose_p95={summary['image_pose_time_delta_ms_p95']}"
        )


if __name__ == "__main__":
    main()
