from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SCHEMA = "project_2026_openpi_compat_pack_v0"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def finite_distance(distance: np.ndarray) -> np.ndarray:
    value = np.asarray(distance, dtype=np.float64).reshape(-1)
    return value[np.isfinite(value)]


def pick_threshold(distance: np.ndarray, *, threshold_px: float | None, positive_rate: float | None) -> float:
    finite = finite_distance(distance)
    if finite.size == 0:
        raise ValueError("estimated_image_distance_px has no finite values")
    if threshold_px is not None:
        return float(threshold_px)
    if positive_rate is None:
        raise ValueError("either --threshold-px or --target-positive-rate is required")
    rate = float(np.clip(positive_rate, 0.0, 1.0))
    return float(np.quantile(finite, rate))


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    finite = finite_distance(values)
    if finite.size == 0:
        return {"count": int(np.asarray(values).size), "finite_count": 0}
    return {
        "count": int(np.asarray(values).size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p15": float(np.quantile(finite, 0.15)),
        "p50": float(np.quantile(finite, 0.50)),
        "p85": float(np.quantile(finite, 0.85)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def relabel_confidence(distance: np.ndarray, flag: np.ndarray, threshold: float, band_px: float) -> np.ndarray:
    dist = np.asarray(distance, dtype=np.float32)
    valid = np.isfinite(dist)
    margin = np.where(flag.astype(bool), np.maximum(threshold - dist, 0.0), np.maximum(dist - threshold, 0.0))
    confidence = 0.35 + 0.65 * np.minimum(margin / max(float(band_px), 1e-6), 1.0)
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    confidence[~valid] = 0.0
    return confidence


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a contact-rich OpenPI-compatible pack by relabeling "
            "estimated_contact_flag from estimated_image_distance_px. This is "
            "for tactile-signal ablation/prototyping, not a real contact-truth label."
        )
    )
    parser.add_argument("pack", type=Path, help="OpenPI-compatible pack directory or manifest.")
    parser.add_argument("--out", type=Path, required=True, help="Output relabeled pack directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--threshold-px", type=float, help="Set contact-like flag when image distance <= threshold.")
    group.add_argument(
        "--target-positive-rate",
        type=float,
        help="Pick threshold by distance quantile so approximately this fraction is positive.",
    )
    parser.add_argument("--confidence-band-px", type=float, default=2.0)
    parser.add_argument("--source-suffix", default="posthoc_distance_threshold_contact_rich")
    args = parser.parse_args()

    pack_dir, manifest = resolve_pack(args.pack)
    arrays_path = pack_dir / str(manifest.get("output", {}).get("arrays_npz", "openpi_arrays.npz"))
    index_path = pack_dir / str(manifest.get("output", {}).get("index_jsonl", "index.jsonl"))
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays not found: {arrays_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    arrays = {name: value for name, value in np.load(arrays_path).items()}
    tactile = arrays["tactile_context"].astype(np.float32, copy=True)
    state = arrays["state_32"].astype(np.float32, copy=True)
    distance = tactile[:, 2].astype(np.float32)
    threshold = pick_threshold(distance, threshold_px=args.threshold_px, positive_rate=args.target_positive_rate)
    flag = (np.isfinite(distance) & (distance <= threshold)).astype(np.float32)
    confidence = relabel_confidence(distance, flag, threshold, args.confidence_band_px)

    tactile[:, 0] = flag
    tactile[:, 1] = confidence
    state[:, 8:11] = tactile
    arrays["tactile_context"] = tactile
    arrays["state_32"] = state

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, out_dir / "index.jsonl")
    save_arrays(out_dir / "openpi_arrays.npz", arrays)

    positive = int(flag.sum())
    total = int(flag.size)
    out_manifest = dict(manifest)
    out_manifest["source_pack"] = str(args.pack)
    out_manifest["output"] = dict(manifest.get("output", {}))
    out_manifest["output"]["arrays_npz"] = "openpi_arrays.npz"
    out_manifest["output"]["index_jsonl"] = "index.jsonl"
    out_manifest["tactile_relabel"] = {
        "method": "estimated_image_distance_px_threshold",
        "warning": (
            "posthoc contact-rich label for ablation/prototyping only; "
            "not real contact truth and not a formal sim-to-real validation label"
        ),
        "threshold_px": threshold,
        "requested_threshold_px": args.threshold_px,
        "requested_target_positive_rate": args.target_positive_rate,
        "actual_positive_count": positive,
        "actual_total": total,
        "actual_positive_rate": float(positive / max(total, 1)),
        "confidence_band_px": float(args.confidence_band_px),
        "source_suffix": str(args.source_suffix),
        "distance_summary": scalar_summary(distance),
        "confidence_summary": scalar_summary(confidence),
    }
    write_json(out_dir / "manifest.json", out_manifest)
    print(f"wrote relabeled OpenPI-compatible pack: {out_dir}")
    print(
        "threshold_px={:.4f} positives={}/{} positive_rate={:.4f}".format(
            threshold,
            positive,
            total,
            positive / max(total, 1),
        )
    )


if __name__ == "__main__":
    main()
