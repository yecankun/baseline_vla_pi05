from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SAMPLE_SCHEMA = "project_2026_pi_style_v0"
EXPECTED_MANIFEST_SCHEMA = "project_2026_pi_style_manifest_v0"
DEFAULT_PIPER_INTENT_TO_ID = {
    "retract": 0,
    "hold": 1,
    "feed": 2,
}


def numeric_list(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(x, (int, float)) for x in value)


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    float_value = float(value)
    if not math.isfinite(float_value):
        return None
    return float_value


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def scalar_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_records is not None and len(rows) >= max_records:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def resolve_manifest(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent, manifest_path, manifest


def sample_image_path(export_dir: Path, image_entry: dict[str, Any]) -> Path | None:
    path_text = image_entry.get("path")
    if not isinstance(path_text, str) or not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return export_dir / path


def make_contact_sheet(export_dir: Path, samples: list[dict[str, Any]], out_path: Path, max_samples: int) -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --contact-sheet") from exc

    tiles: list[tuple[str, Image.Image]] = []
    for sample in samples[:max_samples]:
        sample_id = str(sample.get("sample_id", "unknown"))
        images = sample.get("images") if isinstance(sample.get("images"), dict) else {}
        for camera in ("side", "top"):
            entry = images.get(camera) if isinstance(images.get(camera), dict) else {}
            image_path = sample_image_path(export_dir, entry)
            if image_path is None or not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((224, 224))
            tiles.append((f"{sample_id} {camera}", image.copy()))

    if not tiles:
        raise RuntimeError("no existing images available for contact sheet")

    tile_w, tile_h = 224, 248
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(tiles):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        sheet.paste(image, (x, y + 20))
        draw.text((x + 4, y + 4), label[:36], fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def validate_export(
    export_path: Path,
    *,
    max_records: int | None,
    require_images_existing: bool,
    warn_elite_delta_linf_p95_above: float,
    contact_sheet: Path | None,
    contact_sheet_samples: int,
) -> dict[str, Any]:
    export_dir, manifest_path, manifest = resolve_manifest(export_path)
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        errors.append(f"manifest schema is {manifest.get('schema')!r}, expected {EXPECTED_MANIFEST_SCHEMA!r}")

    samples_ref = manifest.get("output", {}).get("samples_jsonl") if isinstance(manifest.get("output"), dict) else None
    if not isinstance(samples_ref, str) or not samples_ref:
        errors.append("manifest output.samples_jsonl is missing")
        samples_path = export_dir / "samples.jsonl"
    else:
        samples_path = export_dir / samples_ref
    if not samples_path.exists():
        errors.append(f"samples jsonl not found: {samples_path}")
        samples: list[dict[str, Any]] = []
    else:
        samples = read_jsonl(samples_path, max_records=max_records)

    manifest_count = None
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    if isinstance(counts.get("exported_samples"), int):
        manifest_count = counts["exported_samples"]
        if max_records is None and manifest_count != len(samples):
            errors.append(f"manifest exported_samples={manifest_count}, samples.jsonl rows={len(samples)}")

    piper_map = manifest.get("adapter", {}).get("piper_intent_to_id") if isinstance(manifest.get("adapter"), dict) else {}
    if not isinstance(piper_map, dict) or not piper_map:
        piper_map = DEFAULT_PIPER_INTENT_TO_ID
    piper_map = {str(key): int(value) for key, value in piper_map.items()}

    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    piper_intents: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    image_exists_counts: Counter[str] = Counter()
    tactile_signal_count = 0
    contact_sources: Counter[str] = Counter()
    elite_delta_linf: list[float] = []
    piper_steps: list[float] = []
    image_distance_px: list[float] = []
    confidence_values: list[float] = []

    for row_index, sample in enumerate(samples):
        line_label = f"line {sample.get('_line_no', row_index + 1)}"
        if sample.get("schema") != EXPECTED_SAMPLE_SCHEMA:
            errors.append(f"{line_label}: sample schema is {sample.get('schema')!r}")

        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{line_label}: missing sample_id")
        elif sample_id in sample_ids:
            duplicate_ids.append(sample_id)
        else:
            sample_ids.add(sample_id)

        task = sample.get("task")
        if isinstance(task, str):
            tasks[task] += 1
        else:
            warnings.append(f"{line_label}: missing task")

        images = sample.get("images") if isinstance(sample.get("images"), dict) else {}
        for camera in ("side", "top"):
            entry = images.get(camera) if isinstance(images.get(camera), dict) else {}
            image_path = sample_image_path(export_dir, entry)
            if image_path is None:
                errors.append(f"{line_label}: missing {camera} image path")
                continue
            if Path(str(entry.get("path"))).is_absolute():
                warnings.append(f"{line_label}: {camera} image path is absolute; copied exports should be relative")
            image_counts[camera] += 1
            if image_path.exists():
                image_exists_counts[camera] += 1
            elif require_images_existing:
                errors.append(f"{line_label}: {camera} image does not exist relative to export dir: {image_path}")

        state = sample.get("state") if isinstance(sample.get("state"), dict) else {}
        if not numeric_list(state.get("elite_tcp_pose_6d"), 6):
            errors.append(f"{line_label}: missing state.elite_tcp_pose_6d")

        piper = state.get("piper") if isinstance(state.get("piper"), dict) else {}
        piper_step = finite_float(piper.get("piper_step"))
        if piper_step is not None:
            piper_steps.append(piper_step)

        tactile = state.get("tactile_context") if isinstance(state.get("tactile_context"), dict) else {}
        contact_flag = tactile.get("estimated_contact_flag")
        image_distance = finite_float(tactile.get("estimated_image_distance_px"))
        confidence = finite_float(tactile.get("contact_estimator_confidence"))
        if contact_flag is not None or image_distance is not None:
            tactile_signal_count += 1
        if image_distance is not None:
            image_distance_px.append(image_distance)
        if confidence is not None:
            confidence_values.append(confidence)
        contact_sources[str(tactile.get("contact_source") or "missing")] += 1

        action = sample.get("action") if isinstance(sample.get("action"), dict) else {}
        delta = action.get("elite_tcp_delta_6d")
        if numeric_list(delta, 6):
            elite_delta_linf.append(max(abs(float(value)) for value in delta))
        else:
            errors.append(f"{line_label}: missing action.elite_tcp_delta_6d")

        piper_intent = action.get("piper_intent")
        piper_intent_id = action.get("piper_intent_id")
        if piper_intent not in piper_map:
            errors.append(f"{line_label}: unknown piper_intent={piper_intent!r}")
        else:
            piper_intents[str(piper_intent)] += 1
            if piper_intent_id != piper_map[str(piper_intent)]:
                errors.append(
                    f"{line_label}: piper_intent_id={piper_intent_id!r} does not match "
                    f"{piper_intent!r}->{piper_map[str(piper_intent)]}"
                )

    if duplicate_ids:
        errors.append(f"duplicate sample_id values found, first examples: {duplicate_ids[:5]}")

    for camera in ("side", "top"):
        if image_counts[camera] != len(samples):
            errors.append(f"{camera} image coverage {image_counts[camera]}/{len(samples)}")
        if image_exists_counts[camera] != image_counts[camera]:
            warnings.append(f"{camera} existing image coverage {image_exists_counts[camera]}/{image_counts[camera]}")

    if not elite_delta_linf or math.isclose(max(elite_delta_linf), 0.0, abs_tol=1e-12):
        warnings.append("Elite TCP delta appears all-zero or missing; weak for Elite action learning")
    elite_linf_stats = scalar_stats(elite_delta_linf)
    if elite_linf_stats["p95"] is not None and elite_linf_stats["p95"] > warn_elite_delta_linf_p95_above:
        warnings.append(
            "Elite TCP delta linf p95 "
            f"{elite_linf_stats['p95']:.4f} exceeds warning threshold {warn_elite_delta_linf_p95_above:.4f}"
        )
    if len(piper_intents) <= 1:
        warnings.append("Piper intent labels are single-class or missing")
    if tactile_signal_count == 0:
        warnings.append("No tactile/contact signal rows found")

    sheet_path = None
    if contact_sheet is not None:
        sheet_path = make_contact_sheet(export_dir, samples, contact_sheet, max_samples=contact_sheet_samples)

    return {
        "export_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "samples_path": str(samples_path),
        "records_checked": len(samples),
        "manifest_exported_samples": manifest_count,
        "counts": {
            "tasks": dict(tasks),
            "piper_intents": dict(piper_intents),
            "tactile_signal_samples": tactile_signal_count,
            "contact_sources": dict(contact_sources),
            "image_paths": dict(image_counts),
            "existing_images": dict(image_exists_counts),
        },
        "stats": {
            "action.elite_tcp_delta_6d.linf": elite_linf_stats,
            "state.piper.piper_step": scalar_stats(piper_steps),
            "state.tactile_context.estimated_image_distance_px": scalar_stats(image_distance_px),
            "state.tactile_context.contact_estimator_confidence": scalar_stats(confidence_values),
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
        "contact_sheet": sheet_path,
    }


def print_summary(result: dict[str, Any]) -> None:
    print(f"{result['export_dir']}")
    print(f"  records_checked={result['records_checked']} ok={result['ok']}")
    print(f"  tasks={result['counts']['tasks']} piper={result['counts']['piper_intents']}")
    print(
        "  tactile={tactile} contact_sources={sources}".format(
            tactile=result["counts"]["tactile_signal_samples"],
            sources=result["counts"]["contact_sources"],
        )
    )
    print(
        "  images={images} existing={existing}".format(
            images=result["counts"]["image_paths"],
            existing=result["counts"]["existing_images"],
        )
    )
    print(f"  elite_delta_linf={result['stats']['action.elite_tcp_delta_6d.linf']}")
    if result.get("contact_sheet"):
        print(f"  contact_sheet={result['contact_sheet']}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    for error in result["errors"]:
        print(f"  error: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a project_2026 pi-style intermediate export.")
    parser.add_argument("export", help="Pi-style export directory or manifest.json.")
    parser.add_argument("--max-records", type=int, help="Validate only the first N samples.")
    parser.add_argument(
        "--require-images-existing",
        action="store_true",
        help="Fail when image paths do not exist relative to the export directory.",
    )
    parser.add_argument(
        "--warn-elite-delta-linf-p95-above",
        type=float,
        default=100.0,
        help="Warn when action.elite_tcp_delta_6d linf p95 exceeds this value.",
    )
    parser.add_argument("--contact-sheet", help="Optional output PNG contact sheet for first samples.")
    parser.add_argument("--contact-sheet-samples", type=int, default=8, help="Number of samples for contact sheet.")
    parser.add_argument("--out", help="Optional JSON summary path.")
    parser.add_argument("--strict-warnings", action="store_true", help="Exit non-zero when warnings are present.")
    args = parser.parse_args()

    result = validate_export(
        Path(args.export),
        max_records=args.max_records,
        require_images_existing=args.require_images_existing,
        warn_elite_delta_linf_p95_above=args.warn_elite_delta_linf_p95_above,
        contact_sheet=Path(args.contact_sheet) if args.contact_sheet else None,
        contact_sheet_samples=args.contact_sheet_samples,
    )
    print_summary(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out_path}")

    if result["errors"] or (args.strict_warnings and result["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
