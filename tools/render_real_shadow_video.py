from __future__ import annotations

import argparse
import json
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


def infer_fps(rows: list[dict[str, Any]], fallback: float) -> float:
    timestamps: list[float] = []
    for row in rows:
        value = row.get("timestamp")
        if isinstance(value, (int, float)):
            timestamps.append(float(value))
    if len(timestamps) < 2:
        return fallback
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if 0.001 <= b - a <= 10.0]
    if not deltas:
        return fallback
    deltas.sort()
    median_delta = deltas[len(deltas) // 2]
    if median_delta <= 0:
        return fallback
    return max(0.2, min(120.0, 1.0 / median_delta))


def resize_to_height(cv2: Any, image: Any, target_height: int) -> Any:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    target_width = max(1, round(width * target_height / height))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def compose_frame(cv2: Any, dataset_dir: Path, row: dict[str, Any], camera: str) -> Any:
    if camera in {"side", "top"}:
        image_ref = row.get(f"{camera}_image")
        if not image_ref:
            raise ValueError(f"line {row.get('_line_no')}: missing {camera}_image")
        image_path = resolve_path(str(image_ref), dataset_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(image_path))
        return image

    side_ref = row.get("side_image")
    top_ref = row.get("top_image")
    if not side_ref or not top_ref:
        raise ValueError(f"line {row.get('_line_no')}: missing side_image/top_image")
    side = cv2.imread(str(resolve_path(str(side_ref), dataset_dir)), cv2.IMREAD_COLOR)
    top = cv2.imread(str(resolve_path(str(top_ref), dataset_dir)), cv2.IMREAD_COLOR)
    if side is None:
        raise FileNotFoundError(str(resolve_path(str(side_ref), dataset_dir)))
    if top is None:
        raise FileNotFoundError(str(resolve_path(str(top_ref), dataset_dir)))

    if camera == "hstack":
        top = resize_to_height(cv2, top, side.shape[0])
        return cv2.hconcat([side, top])
    if camera == "vstack":
        top = cv2.resize(top, (side.shape[1], round(top.shape[0] * side.shape[1] / top.shape[1])))
        return cv2.vconcat([side, top])
    raise ValueError(f"unsupported camera layout: {camera}")


def overlay_text(cv2: Any, frame: Any, row: dict[str, Any], camera: str, frame_idx: int) -> Any:
    state = row.get("state", {}) if isinstance(row.get("state"), dict) else {}
    ref = row.get("reference_action", {}) if isinstance(row.get("reference_action"), dict) else {}
    controller = state.get("controller_state", {}) if isinstance(state.get("controller_state"), dict) else {}
    piper_step = state.get("piper_step")
    piper_label = ref.get("piper_command_label")
    executed = controller.get("piper_executed_command")
    task = row.get("task")
    step = row.get("step")
    text = f"{camera} frame={frame_idx} task={task} step={step} piper={piper_step} label={piper_label} exec={executed}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def default_output_path(dataset_dir: Path, camera: str) -> Path:
    return dataset_dir / "videos" / f"{dataset_dir.name}_{camera}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render real_shadow_pilot / real alignment PNG frames back into an MP4 video."
    )
    parser.add_argument("dataset", help="Dataset directory or records.jsonl path.")
    parser.add_argument(
        "--camera",
        choices=["side", "top", "hstack", "vstack"],
        default="hstack",
        help="Video view to render. hstack places side and top images next to each other.",
    )
    parser.add_argument("--out", help="Output video path. Defaults to <dataset>/videos/<dataset>_<camera>.mp4.")
    parser.add_argument("--fps", type=float, help="Frames per second. Defaults to inferred timestamp FPS.")
    parser.add_argument("--fallback-fps", type=float, default=5.0, help="FPS used when timestamps cannot be inferred.")
    parser.add_argument("--start-index", type=int, default=0, help="First record index to render.")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to render.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth record.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not draw step/action metadata on frames.")
    args = parser.parse_args()

    import cv2

    dataset_arg = Path(args.dataset)
    if dataset_arg.is_dir():
        dataset_dir = dataset_arg
        records_path = dataset_dir / "records.jsonl"
    else:
        records_path = dataset_arg
        dataset_dir = records_path.parent
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    rows = load_jsonl(records_path)
    selected = rows[args.start_index :: args.stride]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    if not selected:
        raise ValueError("no records selected")

    fps = args.fps if args.fps is not None else infer_fps(selected, args.fallback_fps)
    out_path = Path(args.out) if args.out else default_output_path(dataset_dir, args.camera)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = compose_frame(cv2, dataset_dir, selected[0], args.camera)
    if not args.no_overlay:
        first = overlay_text(cv2, first, selected[0], args.camera, 0)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")

    written = 0
    try:
        writer.write(first)
        written += 1
        for frame_idx, row in enumerate(selected[1:], start=1):
            frame = compose_frame(cv2, dataset_dir, row, args.camera)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if not args.no_overlay:
                frame = overlay_text(cv2, frame, row, args.camera, frame_idx)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(
        json.dumps(
            {
                "dataset": str(dataset_dir),
                "records": len(rows),
                "rendered_frames": written,
                "fps": fps,
                "camera": args.camera,
                "out": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
