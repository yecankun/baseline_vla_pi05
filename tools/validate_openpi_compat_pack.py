from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SCHEMA = "project_2026_openpi_compat_pack_v0"


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def resolve_ref(base: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    direct = base / path
    if direct.exists():
        return direct
    return path


def finite_summary(name: str, value: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(value)
    return {
        "name": name,
        "shape": list(value.shape),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
        "min": float(np.nanmin(value)) if value.size else None,
        "max": float(np.nanmax(value)) if value.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a project_2026 OpenPI-compatible pack.")
    parser.add_argument("pack", type=Path, help="OpenPI-compatible pack directory or manifest.")
    parser.add_argument("--sample-images", type=int, default=32, help="How many image rows to check.")
    parser.add_argument("--out", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    pack_dir, manifest = resolve_pack(args.pack)
    arrays_path = pack_dir / str(manifest.get("output", {}).get("arrays_npz", "openpi_arrays.npz"))
    index_path = pack_dir / str(manifest.get("output", {}).get("index_jsonl", "index.jsonl"))
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays not found: {arrays_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    arrays = dict(np.load(arrays_path))
    rows = read_jsonl(index_path)
    n = int(manifest.get("counts", {}).get("samples", -1))
    errors: list[str] = []
    warnings: list[str] = []

    required_shapes = {
        "state_32": (n, 32),
        "state_mask_32": (n, 32),
        "action_32": (n, 32),
        "action_mask_32": (n, 32),
        "elite_tcp_delta_6d": (n, 6),
        "piper_intent_id": (n,),
    }
    for key, expected in required_shapes.items():
        if key not in arrays:
            errors.append(f"missing array: {key}")
            continue
        if tuple(arrays[key].shape) != expected:
            errors.append(f"{key} shape {arrays[key].shape} != {expected}")

    if len(rows) != n:
        errors.append(f"index rows {len(rows)} != manifest count {n}")

    if "state_mask_32" in arrays:
        state_active = arrays["state_mask_32"].sum(axis=1)
        if not np.all(state_active == 16):
            warnings.append(f"state active dim range: {int(state_active.min())}-{int(state_active.max())}, expected 16")

    if "action_mask_32" in arrays:
        action_active = arrays["action_mask_32"].sum(axis=1)
        if not np.all(action_active == 9):
            warnings.append(f"action active dim range: {int(action_active.min())}-{int(action_active.max())}, expected 9")

    if "piper_intent_id" in arrays:
        ids = set(int(x) for x in np.unique(arrays["piper_intent_id"]))
        if not ids.issubset({0, 1, 2}):
            errors.append(f"unexpected Piper intent ids: {sorted(ids)}")
        if ids == {1} or ids == {2}:
            warnings.append(f"Piper intent has one class only: {sorted(ids)}")

    finite = {
        key: finite_summary(key, arrays[key])
        for key in ["state_32", "action_32", "elite_tcp_delta_6d"]
        if key in arrays
    }
    for key, item in finite.items():
        if not item["finite"]:
            errors.append(f"{key} has {item['nonfinite_count']} non-finite values")

    image_root = resolve_ref(pack_dir, str(manifest.get("output", {}).get("image_root", "")))
    checked_images = 0
    missing_images = 0
    for row in rows[: max(0, args.sample_images)]:
        for key in ["observation.images.side", "observation.images.top"]:
            path_text = row.get(key)
            if not isinstance(path_text, str):
                errors.append(f"missing image field {key} in sample {row.get('sample_id')}")
                continue
            checked_images += 1
            if not (image_root / path_text).exists():
                missing_images += 1
    if missing_images:
        errors.append(f"missing sampled images: {missing_images}/{checked_images}")

    report = {
        "pack": str(args.pack),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "samples": n,
        "checked_images": checked_images,
        "missing_sampled_images": missing_images,
        "finite": finite,
        "semantics": manifest.get("openpi_compat", {}).get("important_semantics", []),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
