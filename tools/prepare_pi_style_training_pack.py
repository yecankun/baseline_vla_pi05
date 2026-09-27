from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SAMPLE_SCHEMA = "project_2026_pi_style_v0"
EXPECTED_MANIFEST_SCHEMA = "project_2026_pi_style_manifest_v0"
PIPER_INTENT_TO_ID = {
    "retract": 0,
    "hold": 1,
    "feed": 2,
}


def numeric_list(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(x, (int, float)) for x in value)


def finite_float_or_nan(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return float("nan")
    float_value = float(value)
    if not math.isfinite(float_value):
        return float("nan")
    return float_value


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_export(path: Path) -> tuple[Path, dict[str, Any], Path]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected export manifest schema: {manifest.get('schema')!r}")
    samples_ref = manifest.get("output", {}).get("samples_jsonl")
    if not isinstance(samples_ref, str) or not samples_ref:
        raise ValueError("export manifest missing output.samples_jsonl")
    samples_path = manifest_path.parent / samples_ref
    if not samples_path.exists():
        raise FileNotFoundError(f"samples jsonl not found: {samples_path}")
    return manifest_path.parent, manifest, samples_path


def resolve_image(export_dir: Path, path_text: str | None) -> tuple[str | None, bool]:
    if not path_text:
        return None, False
    path = Path(path_text)
    resolved = path if path.is_absolute() else export_dir / path
    return str(path_text), resolved.exists()


def piper_one_hot(intent_id: int) -> list[float]:
    return [1.0 if intent_id == index else 0.0 for index in range(3)]


def task_id(task: Any) -> int:
    if task == "left":
        return 0
    if task == "right":
        return 1
    return -1


def task_one_hot(task: Any) -> list[float]:
    tid = task_id(task)
    return [1.0 if tid == index else 0.0 for index in range(2)]


def build_pack_sample(
    sample: dict[str, Any],
    *,
    export_dir: Path,
    require_tactile_context: bool,
    strict_images: bool,
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None, list[str]]:
    errors: list[str] = []
    line_label = f"line {sample.get('_line_no')}"
    if sample.get("schema") != EXPECTED_SAMPLE_SCHEMA:
        errors.append(f"{line_label}: bad sample schema {sample.get('schema')!r}")

    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        errors.append(f"{line_label}: missing sample_id")

    images = sample.get("images") if isinstance(sample.get("images"), dict) else {}
    image_paths: dict[str, str | None] = {}
    image_exists: dict[str, bool] = {}
    for camera in ("side", "top"):
        entry = images.get(camera) if isinstance(images.get(camera), dict) else {}
        image_ref, exists = resolve_image(export_dir, entry.get("path"))
        image_paths[camera] = image_ref
        image_exists[camera] = exists
        if image_ref is None:
            errors.append(f"{line_label}: missing {camera} image")
        elif strict_images and not exists:
            errors.append(f"{line_label}: {camera} image does not exist")

    state = sample.get("state") if isinstance(sample.get("state"), dict) else {}
    action = sample.get("action") if isinstance(sample.get("action"), dict) else {}
    elite_pose = state.get("elite_tcp_pose_6d")
    elite_delta = action.get("elite_tcp_delta_6d")
    if not numeric_list(elite_pose, 6):
        errors.append(f"{line_label}: missing elite_tcp_pose_6d")
    if not numeric_list(elite_delta, 6):
        errors.append(f"{line_label}: missing elite_tcp_delta_6d")

    piper_intent = action.get("piper_intent")
    piper_intent_id = action.get("piper_intent_id")
    if piper_intent not in PIPER_INTENT_TO_ID:
        errors.append(f"{line_label}: unknown piper_intent={piper_intent!r}")
    elif piper_intent_id != PIPER_INTENT_TO_ID[piper_intent]:
        errors.append(f"{line_label}: mismatched piper_intent_id={piper_intent_id!r}")

    piper = state.get("piper") if isinstance(state.get("piper"), dict) else {}
    tactile = state.get("tactile_context") if isinstance(state.get("tactile_context"), dict) else {}
    has_tactile = tactile.get("estimated_contact_flag") is not None or tactile.get("estimated_image_distance_px") is not None
    if require_tactile_context and not has_tactile:
        errors.append(f"{line_label}: missing tactile context")

    if errors:
        return None, None, errors

    assert isinstance(sample_id, str)
    assert numeric_list(elite_pose, 6)
    assert numeric_list(elite_delta, 6)
    assert isinstance(piper_intent_id, int)

    contact_flag = finite_float_or_nan(tactile.get("estimated_contact_flag"))
    contact_confidence = finite_float_or_nan(tactile.get("contact_estimator_confidence"))
    image_distance = finite_float_or_nan(tactile.get("estimated_image_distance_px"))

    elite_delta_array = np.asarray(elite_delta, dtype=np.float32)
    piper_id_array = np.asarray([piper_intent_id], dtype=np.int64)
    piper_onehot_array = np.asarray(piper_one_hot(piper_intent_id), dtype=np.float32)
    task_onehot_array = np.asarray(task_one_hot(sample.get("task")), dtype=np.float32)
    tactile_array = np.asarray([contact_flag, contact_confidence, image_distance], dtype=np.float32)
    tactile_valid_array = np.asarray([not math.isnan(value) for value in tactile_array], dtype=np.bool_)

    arrays = {
        "elite_tcp_pose_6d": np.asarray(elite_pose, dtype=np.float32),
        "piper_state": np.asarray(
            [
                finite_float_or_nan(piper.get("piper_step")),
                finite_float_or_nan(piper.get("piper_insertion_length")),
            ],
            dtype=np.float32,
        ),
        "tactile_context": tactile_array,
        "tactile_context_valid": tactile_valid_array,
        "task_one_hot": task_onehot_array,
        "elite_tcp_delta_6d": elite_delta_array,
        "piper_intent_id": piper_id_array,
        "piper_intent_one_hot": piper_onehot_array,
        "action_continuous_padded": np.concatenate([elite_delta_array, piper_onehot_array]).astype(np.float32),
    }
    index = {
        "sample_id": sample_id,
        "source": sample.get("source", {}),
        "task": sample.get("task"),
        "task_id": task_id(sample.get("task")),
        "language_instruction": sample.get("language_instruction"),
        "images": {
            "side": {"path": image_paths["side"], "exists": image_exists["side"]},
            "top": {"path": image_paths["top"], "exists": image_exists["top"]},
        },
        "state_row": {
            "has_tactile_signal": bool(has_tactile),
            "contact_source": tactile.get("contact_source") or "missing",
        },
        "action_row": {
            "piper_intent": piper_intent,
            "piper_intent_id": piper_intent_id,
        },
    }
    return index, arrays, []


def stack_or_empty(items: list[np.ndarray], shape_tail: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not items:
        return np.empty((0, *shape_tail), dtype=dtype)
    return np.stack(items).astype(dtype, copy=False)


def summarize_array(name: str, values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"name": name, "shape": list(values.shape), "finite_count": 0}
    finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values.reshape(-1)
    if finite.size == 0:
        return {"name": name, "shape": list(values.shape), "finite_count": 0}
    return {
        "name": name,
        "shape": list(values.shape),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def split_indices(count: int, val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(count * val_ratio))
    if count > 1:
        val_count = min(max(1, val_count), count - 1)
    else:
        val_count = 0
    val = set(indices[:val_count])
    train = set(indices[val_count:])
    return train, val


def prepare_pack(args: argparse.Namespace) -> dict[str, Any]:
    export_dir, export_manifest, samples_path = resolve_export(Path(args.export))
    samples = read_jsonl(samples_path, max_records=args.max_records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    arrays_by_name: dict[str, list[np.ndarray]] = {
        "elite_tcp_pose_6d": [],
        "piper_state": [],
        "tactile_context": [],
        "tactile_context_valid": [],
        "task_one_hot": [],
        "elite_tcp_delta_6d": [],
        "piper_intent_id": [],
        "piper_intent_one_hot": [],
        "action_continuous_padded": [],
    }

    piper_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    contact_sources: Counter[str] = Counter()
    tactile_count = 0
    _train_split, val_split = split_indices(len(samples), args.val_ratio, args.seed)

    for source_index, sample in enumerate(samples):
        index_row, arrays, errors = build_pack_sample(
            sample,
            export_dir=export_dir,
            require_tactile_context=args.require_tactile_context,
            strict_images=args.strict_images,
        )
        if index_row is None or arrays is None:
            skipped.append({"source_index": source_index, "errors": errors})
            continue

        packed_index = len(index_rows)
        split = "val" if source_index in val_split else "train"
        index_row["pack_index"] = packed_index
        index_row["source_index"] = source_index
        index_row["split"] = split
        index_rows.append(index_row)

        for name, value in arrays.items():
            arrays_by_name[name].append(value)
        piper_counts[str(index_row["action_row"]["piper_intent"])] += 1
        if isinstance(index_row.get("task"), str):
            task_counts[str(index_row["task"])] += 1
        if index_row["state_row"]["has_tactile_signal"]:
            tactile_count += 1
        contact_sources[str(index_row["state_row"]["contact_source"])] += 1

    arrays_npz = {
        "elite_tcp_pose_6d": stack_or_empty(arrays_by_name["elite_tcp_pose_6d"], (6,), np.float32),
        "piper_state": stack_or_empty(arrays_by_name["piper_state"], (2,), np.float32),
        "tactile_context": stack_or_empty(arrays_by_name["tactile_context"], (3,), np.float32),
        "tactile_context_valid": stack_or_empty(arrays_by_name["tactile_context_valid"], (3,), np.bool_),
        "task_one_hot": stack_or_empty(arrays_by_name["task_one_hot"], (2,), np.float32),
        "elite_tcp_delta_6d": stack_or_empty(arrays_by_name["elite_tcp_delta_6d"], (6,), np.float32),
        "piper_intent_id": stack_or_empty(arrays_by_name["piper_intent_id"], (1,), np.int64).reshape(-1),
        "piper_intent_one_hot": stack_or_empty(arrays_by_name["piper_intent_one_hot"], (3,), np.float32),
        "action_continuous_padded": stack_or_empty(arrays_by_name["action_continuous_padded"], (9,), np.float32),
    }

    index_path = out_dir / "index.jsonl"
    arrays_path = out_dir / "arrays.npz"
    write_jsonl(index_path, index_rows)
    np.savez_compressed(arrays_path, **arrays_npz)

    manifest = {
        "schema": "project_2026_pi_style_training_pack_v0",
        "source_export": str(Path(args.export)),
        "source_export_manifest_schema": export_manifest.get("schema"),
        "output": {
            "index_jsonl": str(index_path.relative_to(out_dir)),
            "arrays_npz": str(arrays_path.relative_to(out_dir)),
            "image_root": str(export_dir),
            "image_paths_are_relative_to_image_root": True,
        },
        "counts": {
            "source_samples": len(samples),
            "packed_samples": len(index_rows),
            "skipped_samples": len(skipped),
            "tasks": dict(task_counts),
            "piper_intents": dict(piper_counts),
            "tactile_signal_samples": tactile_count,
            "contact_sources": dict(contact_sources),
            "splits": dict(Counter(row["split"] for row in index_rows)),
        },
        "schema_notes": {
            "observation_arrays": [
                "elite_tcp_pose_6d",
                "piper_state",
                "tactile_context",
                "tactile_context_valid",
                "task_one_hot",
            ],
            "image_keys": ["side", "top"],
            "language_key": "language_instruction",
            "preferred_action_heads": {
                "elite_tcp_delta_6d": "continuous",
                "piper_intent_id": "classification",
            },
            "smoke_test_action": {
                "action_continuous_padded": "elite_tcp_delta_6d concatenated with Piper intent one-hot; use only as a smoke-test shortcut",
            },
        },
        "stats": {
            name: summarize_array(name, values)
            for name, values in arrays_npz.items()
            if name != "tactile_context_valid"
        },
        "skipped_preview": skipped[:20],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a framework-neutral training pack from a project_2026 pi-style export."
    )
    parser.add_argument("export", help="Pi-style export directory or manifest.json.")
    parser.add_argument("--out", required=True, help="Output directory for the training pack.")
    parser.add_argument("--max-records", type=int, help="Limit samples for smoke tests.")
    parser.add_argument("--require-tactile-context", action="store_true", help="Skip rows without tactile/contact signal.")
    parser.add_argument("--strict-images", action="store_true", help="Fail rows whose side/top image paths do not exist.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Deterministic validation split ratio.")
    parser.add_argument("--seed", type=int, default=123, help="Split seed.")
    args = parser.parse_args()

    manifest = prepare_pack(args)
    counts = manifest["counts"]
    print(f"wrote {Path(args.out) / 'manifest.json'}")
    print(
        "packed={packed_samples} skipped={skipped_samples} tactile={tactile_signal_samples} "
        "splits={splits} piper={piper_intents}".format(**counts)
    )


if __name__ == "__main__":
    main()
