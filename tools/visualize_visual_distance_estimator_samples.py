from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = {
    "tip": (40, 40, 255),
    "wall": (255, 220, 40),
    "line": (40, 190, 255),
    "flag": (0, 140, 255),
    "text": (245, 245, 245),
    "shadow": (0, 0, 0),
}


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def sample_key(sample: dict[str, Any]) -> tuple[str, int]:
    return str(sample.get("episode")), int(sample.get("step", -1))


def state_float(sample: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = sample.get("state", {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def state_optional_float(sample: dict[str, Any], key: str) -> float | None:
    value = sample.get("state", {}).get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def has_finite_state(sample: dict[str, Any], key: str) -> bool:
    value = sample.get("state", {}).get(key)
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    flagged = [s for s in samples if int(s.get("state", {}).get("estimated_contact_flag", 0)) == 1]
    unflagged = [s for s in samples if int(s.get("state", {}).get("estimated_contact_flag", 0)) == 0]
    distance_visible = [s for s in unflagged if has_finite_state(s, "estimated_image_distance_px")]
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, int]] = set()

    def add(group: str, rows: list[dict[str, Any]], limit: int) -> None:
        for sample in rows[: max(int(limit), 0)]:
            key = sample_key(sample)
            if key in seen:
                continue
            seen.add(key)
            selected.append((group, sample))

    add("flagged", sorted(flagged, key=lambda s: state_float(s, "estimated_image_distance_px")), args.max_flagged)
    add("near_miss", sorted(distance_visible, key=lambda s: state_float(s, "estimated_image_distance_px")), args.near_miss)
    add("high_contact_diag", sorted(unflagged, key=lambda s: state_float(s, "contact_strength"), reverse=True), args.high_contact_diag)
    add("far_clear", sorted(distance_visible, key=lambda s: state_float(s, "estimated_image_distance_px"), reverse=True), args.far_clear)

    rng = random.Random(args.seed)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for sample in unflagged:
        if has_finite_state(sample, "estimated_image_distance_px"):
            by_task.setdefault(str(sample.get("task")), []).append(sample)
    for task, rows in sorted(by_task.items()):
        rows = list(rows)
        rng.shuffle(rows)
        add(f"random_{task}", rows, args.random_per_task)
    return selected


def draw_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.42) -> None:
    x, y = org
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS["shadow"], 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS["text"], 1, cv2.LINE_AA)


def point_in_image(point: list[float] | None, image: np.ndarray) -> bool:
    if point is None or len(point) < 2:
        return False
    h, w = image.shape[:2]
    return 0 <= float(point[0]) < w and 0 <= float(point[1]) < h


def draw_marker(image: np.ndarray, point: list[float] | None, color: tuple[int, int, int], label: str) -> bool:
    if not point_in_image(point, image):
        return False
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.circle(image, (x, y), 5, color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 1, color, -1, cv2.LINE_AA)
    draw_text(image, label, (x + 7, y - 7), 0.36)
    return True


def annotate_view(image_path: Path, state: dict[str, Any], view: str) -> tuple[np.ndarray, dict[str, bool]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(image_path))
    annotated = image.copy()
    tip = state.get(f"estimated_tip_pixel_{view}")
    wall = state.get(f"estimated_wall_pixel_{view}")
    tip_visible = draw_marker(annotated, tip, COLORS["tip"], "tip")
    wall_visible = draw_marker(annotated, wall, COLORS["wall"], "wall")
    if tip_visible and wall_visible:
        p0 = (int(round(float(tip[0]))), int(round(float(tip[1]))))
        p1 = (int(round(float(wall[0]))), int(round(float(wall[1]))))
        cv2.line(annotated, p0, p1, COLORS["line"], 1, cv2.LINE_AA)
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    draw_text(annotated, view, (8, 18), 0.5)
    if not tip_visible:
        draw_text(annotated, "tip off-frame", (8, h - 12), 0.4)
    if not wall_visible:
        draw_text(annotated, "wall off-frame", (100, h - 12), 0.4)
    return annotated, {"tip_visible": tip_visible, "wall_visible": wall_visible}


def make_panel(group: str, sample: dict[str, Any], manifest_base: Path, out_dir: Path, index: int) -> dict[str, Any]:
    state = sample.get("state", {})
    images = sample.get("images", {})
    side_path = resolve_path(images["side"], manifest_base)
    top_path = resolve_path(images["top"], manifest_base)
    side, side_vis = annotate_view(side_path, state, "side")
    top, top_vis = annotate_view(top_path, state, "top")
    if side.shape[:2] != top.shape[:2]:
        top = cv2.resize(top, (side.shape[1], side.shape[0]), interpolation=cv2.INTER_AREA)

    flag = int(state.get("estimated_contact_flag", 0))
    distance_value = state_optional_float(sample, "estimated_image_distance_px")
    confidence_value = state_optional_float(sample, "contact_estimator_confidence")
    distance = 0.0 if distance_value is None else distance_value
    confidence = 0.0 if confidence_value is None else confidence_value
    wall = state_float(sample, "segment_min_distance_to_wall")
    contact = state_float(sample, "contact_strength")
    title_h = 78
    h, w = side.shape[:2]
    panel = np.zeros((h + title_h, w * 2, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    panel[title_h:, :w] = side
    panel[title_h:, w:] = top
    border_color = COLORS["flag"] if flag else (80, 80, 80)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), border_color, 3 if flag else 1)
    title = (
        f"{index:03d} {group} | {sample.get('episode')} step={sample.get('step')} "
        f"task={sample.get('task')} flag={flag}"
    )
    dist_text = "NA" if distance_value is None else f"{distance_value:.3f}px"
    conf_text = "NA" if confidence_value is None else f"{confidence_value:.3f}"
    metrics = f"img_dist={dist_text} conf={conf_text} wall={wall:.6f} contact={contact:.6f}"
    piper = (
        f"piper={sample.get('action', {}).get('piper_step_command')} "
        f"feed={sample.get('action', {}).get('piper_feed')}"
    )
    draw_text(panel, title, (10, 22), 0.5)
    draw_text(panel, metrics, (10, 45), 0.45)
    draw_text(panel, piper, (10, 67), 0.42)

    name = f"{index:03d}_{group}_{sample.get('task')}_{sample.get('episode')}_{int(sample.get('step', -1)):06d}.png"
    name = name.replace(":", "_").replace("\\", "_").replace("/", "_")
    out_path = out_dir / "annotated" / name
    cv2.imwrite(str(out_path), panel)
    return {
        "group": group,
        "episode": sample.get("episode"),
        "task": sample.get("task"),
        "step": sample.get("step"),
        "flag": flag,
        "estimated_image_distance_px": distance_value,
        "contact_estimator_confidence": confidence_value,
        "segment_min_distance_to_wall": wall,
        "contact_strength": contact,
        "side_tip_visible": side_vis["tip_visible"],
        "side_wall_visible": side_vis["wall_visible"],
        "top_tip_visible": top_vis["tip_visible"],
        "top_wall_visible": top_vis["wall_visible"],
        "annotated_image": str(out_path.relative_to(out_dir).as_posix()),
        "source_side": str(side_path),
        "source_top": str(top_path),
    }


def stats(values: list[float]) -> dict[str, float] | None:
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_html(out_dir: Path, manifest_path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        cls = "flagged" if row["flag"] else "normal"
        dist_text = "NA" if row["estimated_image_distance_px"] is None else f"{row['estimated_image_distance_px']:.3f}px"
        conf_text = "NA" if row["contact_estimator_confidence"] is None else f"{row['contact_estimator_confidence']:.3f}"
        cards.append(
            f"""
<article class="card {cls}">
  <a href="{html.escape(row['annotated_image'])}"><img src="{html.escape(row['annotated_image'])}" alt="sample"></a>
  <div class="meta">
    <b>{html.escape(row['group'])}</b>
    <span>{html.escape(str(row['episode']))} step={html.escape(str(row['step']))} task={html.escape(str(row['task']))}</span>
    <span>flag={row['flag']} dist={dist_text} conf={conf_text}</span>
    <span>wall={row['segment_min_distance_to_wall']:.6f} contact={row['contact_strength']:.6f}</span>
    <span>visible side(tip/wall)={row['side_tip_visible']}/{row['side_wall_visible']} top(tip/wall)={row['top_tip_visible']}/{row['top_wall_visible']}</span>
  </div>
</article>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Visual Distance Estimator Check</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #151515; color: #e8e8e8; }}
    h1, h2 {{ margin: 0 0 12px; }}
    code {{ color: #b7e3ff; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 16px 0 24px; }}
    .box {{ border: 1px solid #3d3d3d; padding: 12px; background: #202020; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #3a3a3a; background: #202020; padding: 10px; }}
    .card.flagged {{ border-color: #ff9a2f; }}
    img {{ width: 100%; height: auto; display: block; background: #000; }}
    .meta {{ display: grid; gap: 4px; margin-top: 8px; font-size: 13px; color: #d4d4d4; }}
  </style>
</head>
<body>
  <h1>Visual Distance Estimator Check</h1>
  <p>Manifest: <code>{html.escape(str(manifest_path))}</code></p>
  <div class="summary">
    <div class="box">samples total: <b>{summary['samples_total']}</b></div>
    <div class="box">selected panels: <b>{len(rows)}</b></div>
    <div class="box">flagged total: <b>{summary['flagged_total']}</b> ({summary['flagged_rate']:.2%})</div>
    <div class="box">selected groups: <b>{html.escape(str(summary['selected_groups']))}</b></div>
  </div>
  <h2>Annotated Samples</h2>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual audit panels for visual-distance estimator samples.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/_visual_distance_estimator_visual_check"))
    parser.add_argument("--max-flagged", type=int, default=40)
    parser.add_argument("--near-miss", type=int, default=24)
    parser.add_argument("--high-contact-diag", type=int, default=24)
    parser.add_argument("--far-clear", type=int, default=16)
    parser.add_argument("--random-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest_base = Path.cwd()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if not samples:
        raise SystemExit(f"No samples found in {manifest_path}")

    out_dir = args.out.resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    selected = select_samples(samples, args)
    rows = [make_panel(group, sample, manifest_base, out_dir, idx) for idx, (group, sample) in enumerate(selected, start=1)]
    flagged_total = sum(int(s.get("state", {}).get("estimated_contact_flag", 0)) for s in samples)
    summary = {
        "manifest": str(manifest_path),
        "samples_total": len(samples),
        "selected_panels": len(rows),
        "flagged_total": int(flagged_total),
        "flagged_rate": float(flagged_total / len(samples)),
        "selected_groups": dict(Counter(row["group"] for row in rows)),
        "estimated_image_distance_px": stats([state_optional_float(s, "estimated_image_distance_px") for s in samples]),
        "contact_strength": stats([state_optional_float(s, "contact_strength") for s in samples]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "selected_samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_html(out_dir, manifest_path, rows, summary)
    print(f"wrote {len(rows)} panels to {out_dir}")
    print(f"open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
