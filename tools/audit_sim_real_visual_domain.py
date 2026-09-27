from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ImageStats:
    luminance_mean: float
    luminance_std: float
    saturation_mean: float
    saturation_std: float
    red_ratio: float
    dark_ratio: float
    bright_ratio: float
    edge_density: float
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples list")
    return samples


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_image_samples(path_text: str, kind: str) -> tuple[list[dict[str, Any]], Path]:
    path = Path(path_text)
    if kind == "manifest":
        return _load_manifest(path), Path.cwd()
    if kind == "records":
        records_path = path / "records.jsonl" if path.is_dir() else path
        samples = _load_jsonl_records(records_path)
        return samples, records_path.parent
    if kind == "auto":
        if path.is_dir() and (path / "records.jsonl").exists():
            samples = _load_jsonl_records(path / "records.jsonl")
            return samples, path
        if path.suffix.lower() == ".jsonl":
            samples = _load_jsonl_records(path)
            return samples, path.parent
        return _load_manifest(path), Path.cwd()
    raise ValueError(f"Unknown sample kind: {kind}")


def _load_multiple_image_samples(path_texts: list[str], kind: str) -> tuple[list[dict[str, Any]], list[Path]]:
    all_samples: list[dict[str, Any]] = []
    base_dirs: list[Path] = []
    for path_text in path_texts:
        samples, base_dir = _load_image_samples(path_text, kind)
        base_index = len(base_dirs)
        base_dirs.append(base_dir)
        for sample in samples:
            row = dict(sample)
            row["_image_base_index"] = base_index
            row["_image_base_dir"] = str(base_dir)
            all_samples.append(row)
    return all_samples, base_dirs


def _image_path(sample: dict[str, Any], camera: str) -> str | None:
    images = sample.get("images")
    if isinstance(images, dict):
        value = images.get(camera) or images.get("side") or images.get("top")
        return str(value) if value else None
    direct_key = f"{camera}_image"
    value = sample.get(direct_key)
    if value:
        return str(value)
    if camera == "side" and sample.get("side_image"):
        return str(sample["side_image"])
    if camera == "top" and sample.get("top_image"):
        return str(sample["top_image"])
    return None


def _resolve(path_text: str, base_dir: Path | None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return base_dir / path


def _group_by_task(samples: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, sample in enumerate(samples):
        task = str(sample.get("task") or sample.get("branch") or "unknown")
        groups.setdefault(task, []).append((index, sample))
    return groups


def _pick_evenly(items: list[tuple[int, dict[str, Any]]], count: int) -> list[tuple[int, dict[str, Any]]]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, count)
    return [items[int(round(pos))] for pos in positions]


def _stats(image: Image.Image, resize: int) -> ImageStats:
    rgb = image.convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb).astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    red_mask = (r > 90.0) & (r > g * 1.35) & (r > b * 1.25)
    dark_mask = luminance < 45.0
    bright_mask = luminance > 220.0
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    edge_density = float((dx > 25.0).mean() * 0.5 + (dy > 25.0).mean() * 0.5)
    return ImageStats(
        luminance_mean=float(luminance.mean()),
        luminance_std=float(luminance.std()),
        saturation_mean=float(saturation.mean()),
        saturation_std=float(saturation.std()),
        red_ratio=float(red_mask.mean()),
        dark_ratio=float(dark_mask.mean()),
        bright_ratio=float(bright_mask.mean()),
        edge_density=edge_density,
        rgb_mean=tuple(float(x) for x in arr.mean(axis=(0, 1))),
        rgb_std=tuple(float(x) for x in arr.std(axis=(0, 1))),
    )


def _mean_stats(stats: list[ImageStats]) -> dict[str, Any]:
    if not stats:
        return {}
    scalar_fields = [
        "luminance_mean",
        "luminance_std",
        "saturation_mean",
        "saturation_std",
        "red_ratio",
        "dark_ratio",
        "bright_ratio",
        "edge_density",
    ]
    result: dict[str, Any] = {}
    for field in scalar_fields:
        values = np.array([getattr(item, field) for item in stats], dtype=np.float64)
        result[field] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for field in ["rgb_mean", "rgb_std"]:
        values = np.array([getattr(item, field) for item in stats], dtype=np.float64)
        result[field] = {
            "mean": [float(x) for x in values.mean(axis=0)],
            "std": [float(x) for x in values.std(axis=0)],
        }
    return result


def _thumb(image: Image.Image, size: int) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _make_contact_sheet(
    pairs: list[dict[str, Any]],
    out_path: Path,
    thumb_size: int,
) -> None:
    font = _font()
    label_h = 42
    gap = 10
    cols = 2
    rows = max(1, len(pairs))
    width = cols * thumb_size + (cols + 1) * gap
    height = rows * (thumb_size + label_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        y = gap + row * (thumb_size + label_h + gap)
        for col, domain in enumerate(["sim", "real"]):
            x = gap + col * (thumb_size + gap)
            image = Image.open(_resolve(pair[f"{domain}_image"], Path(pair[f"{domain}_base_dir"])))
            sheet.paste(_thumb(image, thumb_size), (x, y + label_h))
            label = (
                f"{domain.upper()} {pair['task']} idx={pair[f'{domain}_index']}\n"
                f"{Path(pair[f'{domain}_image']).name}"
            )
            draw.multiline_text((x, y), label, fill=(0, 0, 0), font=font, spacing=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _metric_line(name: str, sim: dict[str, Any], real: dict[str, Any], digits: int = 3) -> str:
    s = sim.get(name, {}).get("mean", float("nan"))
    r = real.get(name, {}).get("mean", float("nan"))
    delta = s - r if math.isfinite(s) and math.isfinite(r) else float("nan")
    return f"| `{name}` | {s:.{digits}f} | {r:.{digits}f} | {delta:.{digits}f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit visual-domain gap between sim and branchs manifests.")
    parser.add_argument("--sim-manifest", required=True)
    parser.add_argument(
        "--real-manifest",
        nargs="+",
        required=True,
        help="Real manifest(s), records.jsonl file(s), or real pilot dataset directorie(s).",
    )
    parser.add_argument("--sim-kind", choices=["auto", "manifest", "records"], default="manifest")
    parser.add_argument("--real-kind", choices=["auto", "manifest", "records"], default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="side")
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_samples, sim_base_dirs = _load_multiple_image_samples([args.sim_manifest], args.sim_kind)
    real_samples, real_base_dirs = _load_multiple_image_samples(args.real_manifest, args.real_kind)
    sim_groups = _group_by_task(sim_samples)
    real_groups = _group_by_task(real_samples)
    tasks = sorted(set(sim_groups) & set(real_groups))
    if not tasks:
        raise ValueError("No shared task names between sim and real manifests")

    pairs: list[dict[str, Any]] = []
    sim_stats: list[ImageStats] = []
    real_stats: list[ImageStats] = []
    missing: list[str] = []
    for task in tasks:
        sim_picks = _pick_evenly(sim_groups[task], args.samples_per_task)
        real_picks = _pick_evenly(real_groups[task], args.samples_per_task)
        count = min(len(sim_picks), len(real_picks))
        for (sim_idx, sim_sample), (real_idx, real_sample) in zip(sim_picks[:count], real_picks[:count]):
            sim_image = _image_path(sim_sample, args.camera)
            real_image = _image_path(real_sample, args.camera)
            if not sim_image or not real_image:
                missing.append(f"{task}: missing image key at sim={sim_idx} real={real_idx}")
                continue
            sim_base_dir = sim_base_dirs[int(sim_sample.get("_image_base_index", 0))]
            real_base_dir = real_base_dirs[int(real_sample.get("_image_base_index", 0))]
            sim_path = _resolve(sim_image, sim_base_dir)
            real_path = _resolve(real_image, real_base_dir)
            if not sim_path.exists() or not real_path.exists():
                missing.append(f"{task}: missing file sim={sim_path} real={real_path}")
                continue
            sim_img = Image.open(sim_path)
            real_img = Image.open(real_path)
            sim_stats.append(_stats(sim_img, args.image_size))
            real_stats.append(_stats(real_img, args.image_size))
            pairs.append(
                {
                    "task": task,
                    "sim_index": sim_idx,
                    "real_index": real_idx,
                    "sim_image": sim_image,
                    "real_image": real_image,
                    "sim_base_dir": str(sim_base_dir),
                    "real_base_dir": str(real_base_dir),
                    "sim_size": list(sim_img.size),
                    "real_size": list(real_img.size),
                }
            )

    _make_contact_sheet(pairs, out_dir / "contact_sheet.png", args.thumb_size)

    summary = {
        "sim_manifest": args.sim_manifest,
        "real_manifest": args.real_manifest,
        "sim_kind": args.sim_kind,
        "real_kind": args.real_kind,
        "sim_base_dirs": [str(path) for path in sim_base_dirs],
        "real_base_dirs": [str(path) for path in real_base_dirs],
        "camera": args.camera,
        "tasks": tasks,
        "pairs": len(pairs),
        "missing": missing,
        "sim": _mean_stats(sim_stats),
        "real": _mean_stats(real_stats),
        "sample_pairs": pairs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sim_summary = summary["sim"]
    real_summary = summary["real"]
    lines = [
        "# Sim/Real Visual-Domain Audit",
        "",
        f"sim manifest: `{args.sim_manifest}`",
        f"real manifest: `{args.real_manifest}`",
        f"camera: `{args.camera}`",
        f"sample pairs: `{len(pairs)}`",
        "",
        "contact sheet: `contact_sheet.png`",
        "",
        "## Mean Metrics",
        "",
        "| metric | sim mean | real mean | sim-real |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "luminance_mean",
        "luminance_std",
        "saturation_mean",
        "saturation_std",
        "red_ratio",
        "dark_ratio",
        "bright_ratio",
        "edge_density",
    ]:
        lines.append(_metric_line(metric, sim_summary, real_summary))
    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- Use this audit as a visual/domain diagnostic, not as proof of policy quality.",
            "- Large luminance/saturation/edge differences mean the sim renderer and real camera image distribution differ before the policy sees state.",
            "- Compare the contact sheet manually for camera angle, crop, robot occlusion, guidewire visibility, background, vessel contrast, and red-tip appearance.",
            "- If the sheet shows obvious mismatch, fix rendering/camera/domain randomization or collect aligned real data before another BC rollout-tuning loop.",
        ]
    )
    if missing:
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(f"- {item}" for item in missing)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ["camera", "tasks", "pairs", "missing"]}, indent=2, ensure_ascii=False))
    print(f"wrote visual-domain audit to {out_dir}")


if __name__ == "__main__":
    main()
