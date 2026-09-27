from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW = "Annotate Piper forward progress"


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


def load_events(path: Path, dataset_patterns: list[str], include_physical_limit: bool) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"events.csv not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(
            f"events.csv is empty: {path}. Re-run tools/audit_real_alignment_calibration.py first."
        )
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "")
            if not any(Path(dataset).match(pattern) for pattern in dataset_patterns):
                continue
            is_piper_calib = "piper_burst_calib" in dataset
            is_physical = "physical_limit" in dataset
            if not is_piper_calib and not (include_physical_limit and is_physical):
                continue
            if str(row.get("piper_command", "")) != "1":
                continue
            events.append(row)
    return events


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def annotation_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["dataset"]), int(row["event_id"]), str(row["camera"])


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def write_annotations(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (str(r["dataset"]), int(r["event_id"]), str(r["camera"])))
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = path.with_suffix(".csv")
    if not ordered:
        csv_path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in ordered:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(ordered)


def autofill_forward_axis(rows: list[dict[str, Any]], quality_filter: set[str]) -> list[dict[str, Any]]:
    by_camera: dict[str, list[np.ndarray]] = {}
    for row in rows:
        if str(row.get("quality")) not in quality_filter:
            continue
        dx = row.get("dx_px")
        dy = row.get("dy_px")
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            continue
        vec = np.asarray([float(dx), float(dy)], dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        by_camera.setdefault(str(row.get("camera", "side")), []).append(vec)

    axis_by_camera: dict[str, np.ndarray] = {}
    for camera, vectors in by_camera.items():
        mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-6:
            axis_by_camera[camera] = mean_vec / norm

    updated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        camera = str(item.get("camera", "side"))
        axis = axis_by_camera.get(camera)
        dx = item.get("dx_px")
        dy = item.get("dy_px")
        if axis is not None and isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
            delta = np.asarray([float(dx), float(dy)], dtype=np.float64)
            item["forward_px"] = float(np.dot(delta, axis))
            item["axis_x"] = float(axis[0])
            item["axis_y"] = float(axis[1])
            item["axis_source"] = "autofill_mean_delta"
        updated.append(item)
    return updated


def resolve_image_path(dataset_dir: Path, row: dict[str, Any], camera: str) -> Path:
    value = row.get(f"{camera}_image")
    if not value:
        raise ValueError(f"Record has no {camera}_image field")
    path = Path(str(value))
    if not path.is_absolute():
        path = dataset_dir / path
    return path


def read_image(dataset_dir: Path, row: dict[str, Any], camera: str) -> np.ndarray:
    path = resolve_image_path(dataset_dir, row, camera)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def parse_int(row: dict[str, Any], key: str) -> int:
    return int(float(str(row[key])))


def build_display(
    before: np.ndarray,
    after: np.ndarray,
    state: dict[str, Any],
    event: dict[str, Any],
    axis: tuple[tuple[float, float], tuple[float, float]] | None,
    scale: float,
) -> tuple[np.ndarray, float, int]:
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = before.shape[:2]
    view_mode = str(state.get("view_mode", "split"))
    blink_phase = int(state.get("blink_phase", 0))
    blink_frame = "after" if blink_phase % 2 else "before"
    if view_mode == "overlay":
        canvas = after.copy() if blink_frame == "after" else before.copy()
    else:
        canvas = np.concatenate([before, after], axis=1)
    split = w

    def to_display(point: tuple[float, float], pane: str) -> tuple[int, int]:
        x, y = point
        if view_mode != "overlay" and pane == "after":
            x += split
        return int(round(x)), int(round(y))

    if view_mode != "overlay":
        cv2.line(canvas, (split, 0), (split, h - 1), (255, 255, 255), 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    title = (
        f"{event['dataset']} event={event['event_id']} burst={event.get('piper_burst_count')} "
        f"camera={state['camera']}"
    )
    if view_mode == "overlay":
        title += f" view=blink:{blink_frame}"
    else:
        title += " view=split"
    help_text = (
        "split: click before/after. o=blink, b=before/after, x=axis, h=hide axis, "
        "1=usable, 2=uncertain, 3=stuck, 4=not_visible, n=skip, q=quit"
    )
    cv2.putText(canvas, title[:150], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, help_text, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)

    before_pt = state.get("before_point")
    after_pt = state.get("after_point")
    if before_pt is not None:
        cv2.circle(canvas, to_display(before_pt, "before"), 5, (0, 255, 255), -1)
        cv2.putText(canvas, "before", to_display((before_pt[0] + 6, before_pt[1] - 6), "before"), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if after_pt is not None:
        cv2.circle(canvas, to_display(after_pt, "after"), 5, (0, 180, 255), -1)
        cv2.putText(canvas, "after", to_display((after_pt[0] + 6, after_pt[1] - 6), "after"), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
    if before_pt is not None and after_pt is not None:
        cv2.line(canvas, to_display(before_pt, "before"), to_display(after_pt, "after"), (0, 255, 0), 1)

    if axis is not None and state.get("show_axis", True):
        start, end = axis
        panes = (blink_frame,) if view_mode == "overlay" else ("before", "after")
        for pane in panes:
            cv2.arrowedLine(canvas, to_display(start, pane), to_display(end, pane), (255, 0, 255), 2, tipLength=0.15)
            cv2.putText(canvas, "forward axis", to_display((end[0] + 5, end[1] + 5), pane), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
    if state.get("axis_mode"):
        cv2.putText(canvas, "AXIS MODE: click axis start then forward direction", (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if scale != 1.0:
        canvas = cv2.resize(canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return canvas, scale, split


def fit_scale_for_canvas(canvas: np.ndarray, max_width: int, max_height: int) -> float:
    if max_width <= 0 or max_height <= 0:
        return 1.0
    h, w = canvas.shape[:2]
    if h <= 0 or w <= 0:
        return 1.0
    return min(1.0, float(max_width) / float(w), float(max_height) / float(h))


def compute_annotation(
    event: dict[str, Any],
    camera: str,
    before_point: tuple[float, float] | None,
    after_point: tuple[float, float] | None,
    axis: tuple[tuple[float, float], tuple[float, float]] | None,
    quality: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": event["dataset"],
        "event_id": int(event["event_id"]),
        "task": event.get("task"),
        "camera": camera,
        "burst_count": int(float(str(event.get("piper_burst_count", 1)))),
        "record_index": parse_int(event, "record_index"),
        "before_index": parse_int(event, "before_index"),
        "after_index": parse_int(event, "after_index"),
        "quality": quality,
    }
    if before_point is None or after_point is None:
        row.update(
            {
                "before_x": None,
                "before_y": None,
                "after_x": None,
                "after_y": None,
                "dx_px": None,
                "dy_px": None,
                "euclidean_px": None,
                "forward_px": None,
                "axis_x": None,
                "axis_y": None,
            }
        )
        return row

    before_xy = np.asarray(before_point, dtype=np.float64)
    after_xy = np.asarray(after_point, dtype=np.float64)
    delta = after_xy - before_xy
    row.update(
        {
            "before_x": float(before_xy[0]),
            "before_y": float(before_xy[1]),
            "after_x": float(after_xy[0]),
            "after_y": float(after_xy[1]),
            "dx_px": float(delta[0]),
            "dy_px": float(delta[1]),
            "euclidean_px": float(np.linalg.norm(delta)),
            "forward_px": None,
            "axis_x": None,
            "axis_y": None,
        }
    )
    if axis is not None:
        axis_start = np.asarray(axis[0], dtype=np.float64)
        axis_end = np.asarray(axis[1], dtype=np.float64)
        axis_vec = axis_end - axis_start
        norm = float(np.linalg.norm(axis_vec))
        if norm > 1e-6:
            unit = axis_vec / norm
            row["forward_px"] = float(np.dot(delta, unit))
            row["axis_x"] = float(unit[0])
            row["axis_y"] = float(unit[1])
    return row


def annotate_event(
    dataset_dir: Path,
    records: list[dict[str, Any]],
    event: dict[str, Any],
    camera: str,
    axis_by_camera: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    display_scale: float,
    max_window_width: int,
    max_window_height: int,
) -> tuple[dict[str, Any] | None, bool]:
    before = read_image(dataset_dir, records[parse_int(event, "before_index")], camera)
    after = read_image(dataset_dir, records[parse_int(event, "after_index")], camera)
    state: dict[str, Any] = {
        "camera": camera,
        "before_point": None,
        "after_point": None,
        "axis_mode": False,
        "axis_points": [],
        "view_mode": "split",
        "blink_phase": 0,
        "render_scale": display_scale,
        "max_window_width": max_window_width,
        "max_window_height": max_window_height,
        "show_axis": True,
    }

    def on_mouse(event_id: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event_id != cv2.EVENT_LBUTTONDOWN:
            return
        render_scale = float(state.get("render_scale", display_scale))
        x0 = float(x) / max(render_scale, 1e-6)
        y0 = float(y) / max(render_scale, 1e-6)
        split = before.shape[1]
        view_mode = str(state.get("view_mode", "split"))
        blink_frame = "after" if int(state.get("blink_phase", 0)) % 2 else "before"
        if state.get("axis_mode"):
            if view_mode != "overlay" and x0 >= split:
                x0 -= split
            points = state.setdefault("axis_points", [])
            points.append((float(x0), float(y0)))
            if len(points) >= 2:
                axis_by_camera[camera] = (points[-2], points[-1])
                state["axis_mode"] = False
                state["axis_points"] = []
            return
        if view_mode == "overlay":
            if blink_frame == "after":
                state["after_point"] = (float(x0), float(y0))
            else:
                state["before_point"] = (float(x0), float(y0))
        elif x0 < split:
            state["before_point"] = (float(x0), float(y0))
        else:
            state["after_point"] = (float(x0 - split), float(y0))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        preview, _, _ = build_display(before, after, state, event, axis_by_camera.get(camera), 1.0)
        auto_scale = fit_scale_for_canvas(
            preview,
            int(state.get("max_window_width", 1800)),
            int(state.get("max_window_height", 1000)),
        )
        render_scale = min(float(display_scale), auto_scale)
        state["render_scale"] = render_scale
        canvas, _, _ = build_display(before, after, state, event, axis_by_camera.get(camera), render_scale)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            return None, True
        if key == ord("o"):
            state["view_mode"] = "overlay" if state.get("view_mode") != "overlay" else "split"
        elif key == ord("b"):
            if state.get("view_mode") != "overlay":
                state["view_mode"] = "overlay"
            state["blink_phase"] = int(state.get("blink_phase", 0)) + 1
        elif key == ord("x"):
            state["axis_mode"] = True
            state["axis_points"] = []
        elif key == ord("c"):
            state["before_point"] = None
            state["after_point"] = None
        elif key == ord("h"):
            state["show_axis"] = not bool(state.get("show_axis", True))
        elif key == ord("n"):
            return None, False
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            quality = {
                ord("1"): "usable",
                ord("2"): "uncertain",
                ord("3"): "stuck",
                ord("4"): "not_visible",
            }[key]
            annotation = compute_annotation(
                event=event,
                camera=camera,
                before_point=state.get("before_point"),
                after_point=state.get("after_point"),
                axis=axis_by_camera.get(camera),
                quality=quality,
            )
            return annotation, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually annotate guidewire forward progress in Piper-only real calibration windows."
    )
    parser.add_argument("--root", default="collected_data")
    parser.add_argument("--events", default="docs/_real_alignment_20260707_audit/events.csv")
    parser.add_argument(
        "--out",
        default="docs/_real_alignment_20260707_piper_forward_annotations/annotations.jsonl",
    )
    parser.add_argument(
        "--dataset-pattern",
        action="append",
        default=["real_align_20260707_*_piper_burst_calib_001"],
        help="Dataset name glob. Can be repeated.",
    )
    parser.add_argument("--camera", action="append", choices=["side", "top"], default=None)
    parser.add_argument("--include-physical-limit", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already annotated dataset/event/camera rows.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Maximum display scale. Aspect ratio is preserved.")
    parser.add_argument("--max-window-width", type=int, default=1800)
    parser.add_argument("--max-window-height", type=int, default=1000)
    parser.add_argument(
        "--autofill-forward-axis",
        action="store_true",
        help="Post-process existing annotations: estimate per-camera forward axis from usable/uncertain deltas and fill forward_px.",
    )
    parser.add_argument("--list-events", action="store_true", help="Print selected events and exit.")
    args = parser.parse_args()

    root = Path(args.root)
    cameras = unique_preserve_order(args.camera or ["side"])
    events = load_events(Path(args.events), args.dataset_pattern, args.include_physical_limit)
    if not events:
        raise ValueError(
            "No Piper-only events matched the requested filters. Check --events and --dataset-pattern, "
            "or re-run tools/audit_real_alignment_calibration.py."
        )
    if args.list_events:
        print(json.dumps({"events": len(events), "cameras": cameras}, indent=2))
        for event in events:
            print(
                f"{event['dataset']} event={event['event_id']} burst={event.get('piper_burst_count')} "
                f"before={event.get('before_index')} after={event.get('after_index')}"
            )
        return

    out_path = Path(args.out)
    annotations = load_annotations(out_path)
    if args.autofill_forward_axis:
        if not annotations:
            raise ValueError(f"No annotations to post-process: {out_path}")
        annotations = autofill_forward_axis(annotations, {"usable", "uncertain"})
        write_annotations(out_path, annotations)
        print(f"autofilled forward axis for {len(annotations)} annotations at {out_path}")
        return
    by_key = {annotation_key(row): row for row in annotations}
    records_cache: dict[str, list[dict[str, Any]]] = {}
    axis_by_camera: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

    try:
        for event in events:
            dataset = str(event["dataset"])
            dataset_dir = root / dataset
            if dataset not in records_cache:
                records_cache[dataset] = load_jsonl(dataset_dir / "records.jsonl")
            for camera in cameras:
                key = (dataset, int(event["event_id"]), camera)
                if not args.no_resume and key in by_key:
                    continue
                print(f"Annotating {dataset} event={event['event_id']} camera={camera}")
                annotation, should_quit = annotate_event(
                    dataset_dir=dataset_dir,
                    records=records_cache[dataset],
                    event=event,
                    camera=camera,
                    axis_by_camera=axis_by_camera,
                    display_scale=args.display_scale,
                    max_window_width=args.max_window_width,
                    max_window_height=args.max_window_height,
                )
                if annotation is not None:
                    by_key[annotation_key(annotation)] = annotation
                    write_annotations(out_path, list(by_key.values()))
                    print(f"saved {annotation['quality']} forward_px={annotation.get('forward_px')}")
                if should_quit:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Stopped. Existing annotations were preserved.")
    finally:
        write_annotations(out_path, list(by_key.values()))
        cv2.destroyAllWindows()
        print(f"wrote {len(by_key)} annotations to {out_path}")


if __name__ == "__main__":
    main()
