from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


EXPORT_SCHEMA = "project_2026_pi_style_manifest_v0"
PACK_SCHEMA = "project_2026_pi_style_training_pack_v0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def posix_rel_path(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    return path_text.replace("\\", "/")


def normalize_image_paths(row: dict[str, Any]) -> None:
    images = row.get("images")
    if not isinstance(images, dict):
        return
    for image_info in images.values():
        if not isinstance(image_info, dict):
            continue
        path_text = image_info.get("path")
        if isinstance(path_text, str):
            image_info["path"] = posix_rel_path(path_text)


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def validate_source_dirs(export_dir: Path, pack_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    export_manifest_path = export_dir / "manifest.json"
    pack_manifest_path = pack_dir / "manifest.json"
    if not export_manifest_path.exists():
        raise FileNotFoundError(f"export manifest not found: {export_manifest_path}")
    if not pack_manifest_path.exists():
        raise FileNotFoundError(f"pack manifest not found: {pack_manifest_path}")
    export_manifest = load_json(export_manifest_path)
    pack_manifest = load_json(pack_manifest_path)
    if export_manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"unexpected export schema: {export_manifest.get('schema')!r}")
    if pack_manifest.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected pack schema: {pack_manifest.get('schema')!r}")
    return export_manifest, pack_manifest


def rewrite_export(export_dst: Path, export_manifest: dict[str, Any]) -> int:
    samples_name = export_manifest.get("output", {}).get("samples_jsonl", "samples.jsonl")
    samples_path = export_dst / str(samples_name)
    rows = read_jsonl(samples_path)
    for row in rows:
        normalize_image_paths(row)
        source = row.get("source")
        if isinstance(source, dict):
            dataset = source.get("dataset")
            if isinstance(dataset, str):
                source["dataset"] = posix_rel_path(dataset)
        images = row.get("images")
        if isinstance(images, dict):
            for image_info in images.values():
                if isinstance(image_info, dict):
                    image_info.pop("source_path", None)
    write_jsonl(samples_path, rows)

    export_manifest["source_dataset"] = posix_rel_path(str(export_manifest.get("source_dataset", "")))
    export_manifest.setdefault("output", {})["samples_jsonl"] = posix_rel_path(str(samples_name))
    write_json(export_dst / "manifest.json", export_manifest)
    return len(rows)


def rewrite_pack(pack_dst: Path, pack_manifest: dict[str, Any]) -> int:
    index_name = pack_manifest.get("output", {}).get("index_jsonl", "index.jsonl")
    index_path = pack_dst / str(index_name)
    rows = read_jsonl(index_path)
    for row in rows:
        normalize_image_paths(row)
        source = row.get("source")
        if isinstance(source, dict):
            dataset = source.get("dataset")
            if isinstance(dataset, str):
                source["dataset"] = posix_rel_path(dataset)
    write_jsonl(index_path, rows)

    pack_manifest["source_export"] = "../pi_style_export"
    output = pack_manifest.setdefault("output", {})
    output["index_jsonl"] = posix_rel_path(str(index_name))
    output["arrays_npz"] = posix_rel_path(str(output.get("arrays_npz", "arrays.npz")))
    output["image_root"] = "../pi_style_export"
    output["image_paths_are_relative_to_image_root"] = True
    write_json(pack_dst / "manifest.json", pack_manifest)
    return len(rows)


def write_readme(out_dir: Path, export_rows: int, pack_rows: int) -> None:
    readme = f"""# Project 2026 Pi-Style 4090 Bundle

This bundle is prepared for Linux/Ubuntu training. Paths inside JSONL and
manifests have been normalized to POSIX-style separators.

## Contents

- `pi_style_export/`: image-backed pi-style intermediate export.
- `pi_style_pack/`: array-backed training pack plus relative image paths.

## Counts

- export samples: `{export_rows}`
- pack samples: `{pack_rows}`

## Suggested Ubuntu Smoke Check

From the repository root on the 4090 machine, place or extract this bundle so
the following paths exist:

```text
simulation_output/pi_style_4090_bundle/pi_style_export
simulation_output/pi_style_4090_bundle/pi_style_pack
```

Then run a lightweight verification/training smoke before OpenPI/LeRobot
integration:

```bash
python tools/train_pi_style_pack_smoke.py \\
  simulation_output/pi_style_4090_bundle/pi_style_pack \\
  --out simulation_output/pi_style_4090_bundle_smoke_model \\
  --epochs 1 \\
  --batch-size 64 \\
  --image-size 128
```

For OpenPI/LeRobot work, keep the mixed target semantics explicit:

```text
dual-view images + Elite/Piper state + tactile context + task
-> Elite TCP delta regression + Piper discrete intent classification
```
"""
    (out_dir / "README_4090.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package pi-style export/pack for Linux 4090 training.")
    parser.add_argument("export_dir", type=Path, help="Pi-style export directory with images and samples.jsonl.")
    parser.add_argument("pack_dir", type=Path, help="Pi-style training pack directory with arrays.npz and index.jsonl.")
    parser.add_argument("--out", type=Path, required=True, help="Output bundle directory.")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive next to the bundle directory.")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    pack_dir = args.pack_dir.resolve()
    out_dir = args.out.resolve()
    export_manifest, pack_manifest = validate_source_dirs(export_dir, pack_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    export_dst = out_dir / "pi_style_export"
    pack_dst = out_dir / "pi_style_pack"
    copytree_clean(export_dir, export_dst)
    copytree_clean(pack_dir, pack_dst)

    export_rows = rewrite_export(export_dst, export_manifest)
    pack_rows = rewrite_pack(pack_dst, pack_manifest)
    write_readme(out_dir, export_rows, pack_rows)

    bundle_manifest = {
        "schema": "project_2026_pi_style_4090_bundle_v0",
        "export_dir": "pi_style_export",
        "pack_dir": "pi_style_pack",
        "export_samples": export_rows,
        "pack_samples": pack_rows,
        "path_convention": "posix_relative_paths_for_linux_training",
    }
    write_json(out_dir / "bundle_manifest.json", bundle_manifest)

    if args.zip:
        zip_base = out_dir.with_suffix("")
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=out_dir)
        print(f"wrote bundle: {out_dir}")
        print(f"wrote zip: {zip_path}")
    else:
        print(f"wrote bundle: {out_dir}")
    print(f"export samples={export_rows} pack samples={pack_rows}")


if __name__ == "__main__":
    main()
