from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PACK_SCHEMA = "project_2026_pi_style_training_pack_v0"
OUT_SCHEMA = "project_2026_openpi_compat_pack_v0"
STATE_DIM = 32
ACTION_DIM = 32


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


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"pack manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected pack schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def resolve_ref(base: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    direct = base / path
    if direct.exists():
        return direct
    return path


def portable_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start.resolve()).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def compute_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": np.where(values.std(axis=0) < 1e-6, 1.0, values.std(axis=0)).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def make_state_32(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    n = arrays["elite_tcp_pose_6d"].shape[0]
    state = np.zeros((n, STATE_DIM), dtype=np.float32)
    mask = np.zeros((n, STATE_DIM), dtype=np.bool_)
    layout = {
        "elite_tcp_pose_6d": (0, 6),
        "piper_state": (6, 8),
        "tactile_context": (8, 11),
        "tactile_context_valid": (11, 14),
        "task_one_hot": (14, 16),
    }
    for key, (start, end) in layout.items():
        value = arrays[key].astype(np.float32)
        if value.shape[1] != end - start:
            raise ValueError(f"{key} has shape {value.shape}, expected second dim {end - start}")
        state[:, start:end] = value
        mask[:, start:end] = True
    return state, mask, layout


def make_action_32(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    n = arrays["elite_tcp_delta_6d"].shape[0]
    action = np.zeros((n, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    layout = {
        "elite_tcp_delta_6d": (0, 6),
        "piper_intent_one_hot_compat": (6, 9),
    }
    action[:, 0:6] = arrays["elite_tcp_delta_6d"].astype(np.float32)
    action[:, 6:9] = arrays["piper_intent_one_hot"].astype(np.float32)
    mask[:, 0:9] = True
    return action, mask, layout


def rewrite_index_rows(rows: list[dict[str, Any]], *, image_root: str) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        images = row.get("images", {})
        side = images.get("side", {}) if isinstance(images, dict) else {}
        top = images.get("top", {}) if isinstance(images, dict) else {}
        rewritten.append(
            {
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "task": row.get("task"),
                "task_id": row.get("task_id"),
                "language_instruction": row.get("language_instruction"),
                "image_root": image_root,
                "observation.images.side": side.get("path"),
                "observation.images.top": top.get("path"),
                "observation.state": "state_32",
                "observation.state_mask": "state_mask_32",
                "action": "action_32",
                "action_mask": "action_mask_32",
                "elite_tcp_delta_6d": "elite_tcp_delta_6d",
                "piper_intent_id": "piper_intent_id",
                "piper_intent": row.get("action_row", {}).get("piper_intent") if isinstance(row.get("action_row"), dict) else None,
                "source": row.get("source"),
                "source_index": row.get("source_index"),
                "pack_index": row.get("pack_index"),
            }
        )
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a project_2026 pi-style training pack into an OpenPI-compatible "
            "32-dim state/action pack while preserving the preferred mixed action labels."
        )
    )
    parser.add_argument("pack", type=Path, help="Pi-style training pack directory or manifest.")
    parser.add_argument("--out", type=Path, required=True, help="Output OpenPI-compatible pack directory.")
    args = parser.parse_args()

    pack_dir, manifest = resolve_pack(args.pack)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = resolve_ref(pack_dir, str(manifest.get("output", {}).get("arrays_npz", "arrays.npz")))
    index_path = resolve_ref(pack_dir, str(manifest.get("output", {}).get("index_jsonl", "index.jsonl")))
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays not found: {arrays_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    arrays = dict(np.load(arrays_path))
    index_rows = read_jsonl(index_path)
    n = arrays["elite_tcp_pose_6d"].shape[0]
    if len(index_rows) != n:
        raise ValueError(f"index rows ({len(index_rows)}) != array rows ({n})")

    state_32, state_mask_32, state_layout = make_state_32(arrays)
    action_32, action_mask_32, action_layout = make_action_32(arrays)

    np.savez_compressed(
        out_dir / "openpi_arrays.npz",
        state_32=state_32,
        state_mask_32=state_mask_32,
        action_32=action_32,
        action_mask_32=action_mask_32,
        elite_tcp_pose_6d=arrays["elite_tcp_pose_6d"],
        piper_state=arrays["piper_state"],
        tactile_context=arrays["tactile_context"],
        tactile_context_valid=arrays["tactile_context_valid"],
        task_one_hot=arrays["task_one_hot"],
        elite_tcp_delta_6d=arrays["elite_tcp_delta_6d"],
        piper_intent_id=arrays["piper_intent_id"],
        piper_intent_one_hot=arrays["piper_intent_one_hot"],
    )

    source_image_root_ref = str(manifest.get("output", {}).get("image_root", ""))
    source_image_root = resolve_ref(pack_dir, source_image_root_ref)
    image_root = portable_relpath(source_image_root, out_dir) if source_image_root.exists() else source_image_root_ref
    rewritten_rows = rewrite_index_rows(index_rows, image_root=image_root)
    write_jsonl(out_dir / "index.jsonl", rewritten_rows)

    piper_ids, piper_counts = np.unique(arrays["piper_intent_id"], return_counts=True)
    out_manifest = {
        "schema": OUT_SCHEMA,
        "source_pack": str(args.pack),
        "source_pack_schema": manifest.get("schema"),
        "counts": {
            "samples": int(n),
            "splits": manifest.get("counts", {}).get("splits", {}),
            "piper_intent_id": {str(int(k)): int(v) for k, v in zip(piper_ids, piper_counts)},
        },
        "output": {
            "arrays_npz": "openpi_arrays.npz",
            "index_jsonl": "index.jsonl",
            "image_root": image_root,
            "image_paths_are_relative_to_image_root": bool(
                manifest.get("output", {}).get("image_paths_are_relative_to_image_root", True)
            ),
        },
        "openpi_compat": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_layout": {k: [v[0], v[1]] for k, v in state_layout.items()},
            "action_layout": {k: [v[0], v[1]] for k, v in action_layout.items()},
            "important_semantics": [
                "action_32 is a compatibility tensor, not the preferred final control contract",
                "preferred target remains Elite TCP delta regression plus Piper intent classification",
                "piper_intent_one_hot_compat occupies action dims 6:9 only for framework smoke tests",
                "do not interpret padded zero dims as robot controls",
            ],
        },
        "stats": {
            "state_32": compute_stats(state_32),
            "action_32": compute_stats(action_32),
            "elite_tcp_delta_6d": compute_stats(arrays["elite_tcp_delta_6d"].astype(np.float32)),
        },
    }
    write_json(out_dir / "manifest.json", out_manifest)
    print(f"wrote OpenPI-compatible pack: {out_dir}")
    print(f"samples={n} state_dim={STATE_DIM} action_dim={ACTION_DIM}")


if __name__ == "__main__":
    main()
