from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from audit_pi_style_dataset import (
    action_elite_delta,
    action_piper_label,
    load_dataset,
    resolve_path,
    row_action,
    row_images,
    row_instruction,
    row_state,
    summarize_dataset,
)


PIPER_INTENT_TO_ID = {
    "retract": 0,
    "hold": 1,
    "feed": 2,
}


def numeric_list(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(x, (int, float)) for x in value)


def non_null(value: Any) -> bool:
    return value is not None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    float_value = float(value)
    if not math.isfinite(float_value):
        return None
    return float_value


def scalar_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def vector_stats(vectors: list[list[float]]) -> dict[str, Any]:
    if not vectors:
        return {
            "count": 0,
            "dim": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "linf": scalar_stats([]),
        }
    dim = len(vectors[0])
    columns = [[vector[index] for vector in vectors] for index in range(dim)]
    return {
        "count": len(vectors),
        "dim": dim,
        "mean": [scalar_stats(column)["mean"] for column in columns],
        "std": [scalar_stats(column)["std"] for column in columns],
        "min": [min(column) for column in columns],
        "max": [max(column) for column in columns],
        "linf": scalar_stats([max(abs(value) for value in vector) for vector in vectors]),
    }


def build_export_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    elite_pose_vectors: list[list[float]] = []
    elite_delta_vectors: list[list[float]] = []
    piper_steps: list[float] = []
    piper_insertion_lengths: list[float] = []
    contact_confidences: list[float] = []
    image_distances_px: list[float] = []
    contact_flags: Counter[str] = Counter()
    contact_sources: Counter[str] = Counter()

    for sample in samples:
        state = sample.get("state", {})
        action = sample.get("action", {})
        pose = state.get("elite_tcp_pose_6d")
        delta = action.get("elite_tcp_delta_6d")
        if numeric_list(pose, 6):
            elite_pose_vectors.append([float(value) for value in pose])
        if numeric_list(delta, 6):
            elite_delta_vectors.append([float(value) for value in delta])

        piper = state.get("piper") if isinstance(state.get("piper"), dict) else {}
        piper_step = finite_float(piper.get("piper_step"))
        insertion = finite_float(piper.get("piper_insertion_length"))
        if piper_step is not None:
            piper_steps.append(piper_step)
        if insertion is not None:
            piper_insertion_lengths.append(insertion)

        tactile = state.get("tactile_context") if isinstance(state.get("tactile_context"), dict) else {}
        confidence = finite_float(tactile.get("contact_estimator_confidence"))
        distance = finite_float(tactile.get("estimated_image_distance_px"))
        if confidence is not None:
            contact_confidences.append(confidence)
        if distance is not None:
            image_distances_px.append(distance)
        flag = tactile.get("estimated_contact_flag")
        if flag is None:
            contact_flags["missing"] += 1
        else:
            contact_flags[str(int(flag)) if isinstance(flag, (int, float, bool)) else str(flag)] += 1
        source = tactile.get("contact_source") or "missing"
        contact_sources[str(source)] += 1

    return {
        "normalization_hints": {
            "state.elite_tcp_pose_6d": vector_stats(elite_pose_vectors),
            "action.elite_tcp_delta_6d": vector_stats(elite_delta_vectors),
            "state.piper.piper_step": scalar_stats(piper_steps),
            "state.piper.piper_insertion_length": scalar_stats(piper_insertion_lengths),
            "state.tactile_context.contact_estimator_confidence": scalar_stats(contact_confidences),
            "state.tactile_context.estimated_image_distance_px": scalar_stats(image_distances_px),
        },
        "class_balance": {
            "piper_intent": dict(Counter(sample["action"]["piper_intent"] for sample in samples)),
            "estimated_contact_flag": dict(contact_flags),
            "contact_source": dict(contact_sources),
        },
    }


def make_sample_id(index: int, row: dict[str, Any]) -> str:
    episode = row.get("episode") or "record"
    task = row.get("task") or "unknown"
    step = row.get("step")
    if step is None:
        step = row.get("_line_no", index)
    safe_episode = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(episode))
    return f"{index:08d}_{task}_{safe_episode}_{int(step):06d}"


def tactile_context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    context = {
        "estimated_contact_flag": state.get("estimated_contact_flag"),
        "contact_estimator_confidence": state.get("contact_estimator_confidence"),
        "estimated_image_distance_px": state.get("estimated_image_distance_px"),
        "contact_source": state.get("contact_source"),
    }
    if context["contact_source"] is None:
        if non_null(context["estimated_contact_flag"]) or non_null(context["estimated_image_distance_px"]):
            context["contact_source"] = "dataset_estimated_visual_distance"
        else:
            context["contact_source"] = "missing"
    return context


def has_tactile_signal(context: dict[str, Any]) -> bool:
    return non_null(context.get("estimated_contact_flag")) or non_null(context.get("estimated_image_distance_px"))


def image_output_path(out_dir: Path, sample_id: str, camera: str, source_path: Path) -> Path:
    suffix = source_path.suffix if source_path.suffix else ".png"
    return out_dir / "images" / camera / f"{sample_id}{suffix}"


def maybe_copy_image(
    *,
    image_ref: str | None,
    camera: str,
    sample_id: str,
    source_base_dir: Path,
    out_dir: Path,
    copy_images: bool,
) -> tuple[str | None, str | None, bool]:
    if not image_ref:
        return None, None, False
    source_path = resolve_path(str(image_ref), source_base_dir)
    if not copy_images:
        return str(image_ref), str(source_path), source_path.exists()
    if not source_path.exists():
        return None, str(source_path), False
    target = image_output_path(out_dir, sample_id, camera, source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target.relative_to(out_dir)), str(source_path), True


def build_sample(
    *,
    row: dict[str, Any],
    index: int,
    kind: str,
    source_base_dir: Path,
    source_dataset: Path,
    out_dir: Path,
    copy_images: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    sample_id = make_sample_id(index, row)
    state = row_state(row)
    action = row_action(row, kind)
    images = row_images(row, kind)
    instruction = row_instruction(row)
    elite_tcp_pose_6d = state.get("elite_tcp_pose_6d")
    elite_tcp_delta_6d = action_elite_delta(action)
    piper_intent = action_piper_label(action)
    tactile_context = tactile_context_from_state(state)

    if not instruction:
        errors.append("missing instruction/task")
    if not numeric_list(elite_tcp_pose_6d, 6):
        errors.append("missing elite_tcp_pose_6d")
    if elite_tcp_delta_6d is None:
        errors.append("missing elite_tcp_delta_6d")
    if piper_intent not in PIPER_INTENT_TO_ID:
        errors.append("missing piper intent")

    image_entries: dict[str, Any] = {}
    for camera in ("side", "top"):
        output_ref, source_ref, exists = maybe_copy_image(
            image_ref=images.get(camera),
            camera=camera,
            sample_id=sample_id,
            source_base_dir=source_base_dir,
            out_dir=out_dir,
            copy_images=copy_images,
        )
        if output_ref is None:
            errors.append(f"missing {camera} image")
        image_entries[camera] = {
            "path": output_ref,
            "source_path": source_ref,
            "exists": exists,
        }

    if errors:
        return None, errors

    piper_state = {
        "piper_step": state.get("piper_step"),
        "piper_insertion_length": state.get("piper_insertion_length"),
    }
    controller_state = state.get("controller_state") if isinstance(state.get("controller_state"), dict) else {}
    if controller_state:
        piper_state["controller"] = {
            "piper_motion_state": controller_state.get("piper_motion_state"),
            "piper_executed_command": controller_state.get("piper_executed_command"),
            "piper_busy": controller_state.get("piper_busy"),
        }

    sample = {
        "schema": "project_2026_pi_style_v0",
        "sample_id": sample_id,
        "source": {
            "dataset": str(source_dataset),
            "kind": kind,
            "episode": row.get("episode"),
            "line_no": row.get("_line_no"),
            "step": row.get("step"),
        },
        "task": row.get("task"),
        "language_instruction": instruction,
        "images": image_entries,
        "state": {
            "elite_tcp_pose_6d": [float(x) for x in elite_tcp_pose_6d],
            "piper": piper_state,
            "tactile_context": tactile_context,
        },
        "action": {
            "elite_tcp_delta_6d": elite_tcp_delta_6d,
            "piper_intent": piper_intent,
            "piper_intent_id": PIPER_INTENT_TO_ID[piper_intent],
            "piper_step_command": action.get("piper_step_command"),
            "piper_command_label": action.get("piper_command_label") or piper_intent,
        },
        "pi_style": {
            "observation_keys": [
                "images.side",
                "images.top",
                "state.elite_tcp_pose_6d",
                "state.piper",
                "state.tactile_context",
                "language_instruction",
            ],
            "action_heads": {
                "elite_tcp_delta_6d": "continuous",
                "piper_intent": "classification",
            },
            "context_conditioning": {
                "style": "pi0.7_inspired",
                "fields": ["state.tactile_context"],
                "has_tactile_signal": has_tactile_signal(tactile_context),
            },
        },
    }
    return sample, []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kind, source_base_dir, rows, source_manifest = load_dataset(source_path, max_records=args.max_records)
    audit = summarize_dataset(source_path, max_records=args.max_records, check_images=False)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    piper_intents: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    tactile_signal_count = 0

    for index, row in enumerate(rows):
        sample, errors = build_sample(
            row=row,
            index=index,
            kind=kind,
            source_base_dir=source_base_dir,
            source_dataset=source_path,
            out_dir=out_dir,
            copy_images=args.copy_images,
        )
        if sample is None:
            skipped.append({"index": index, "errors": errors})
            continue
        if args.require_tactile_context and not sample["pi_style"]["context_conditioning"]["has_tactile_signal"]:
            skipped.append({"index": index, "sample_id": sample["sample_id"], "errors": ["missing tactile context"]})
            continue
        samples.append(sample)
        piper_intents[sample["action"]["piper_intent"]] += 1
        if isinstance(sample.get("task"), str):
            tasks[str(sample["task"])] += 1
        if sample["pi_style"]["context_conditioning"]["has_tactile_signal"]:
            tactile_signal_count += 1

    samples_path = out_dir / "samples.jsonl"
    write_jsonl(samples_path, samples)

    manifest = {
        "schema": "project_2026_pi_style_manifest_v0",
        "source_dataset": str(source_path),
        "source_kind": kind,
        "source_manifest_mode": source_manifest.get("mode"),
        "output": {
            "samples_jsonl": str(samples_path.relative_to(out_dir)),
            "images_copied": bool(args.copy_images),
        },
        "adapter": {
            "require_tactile_context": bool(args.require_tactile_context),
            "piper_intent_to_id": PIPER_INTENT_TO_ID,
            "action_heads": {
                "elite_tcp_delta_6d": "continuous",
                "piper_intent": "classification",
            },
            "context_conditioning": "pi0.7_inspired_tactile_context",
        },
        "counts": {
            "source_records": len(rows),
            "exported_samples": len(samples),
            "skipped_samples": len(skipped),
            "tactile_signal_samples": tactile_signal_count,
            "tasks": dict(tasks),
            "piper_intents": dict(piper_intents),
        },
        "stats": build_export_stats(samples),
        "source_audit": audit,
        "skipped_preview": skipped[:20],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sim/real datasets into the project_2026 pi-style intermediate JSONL format."
    )
    parser.add_argument("dataset", help="Dataset directory, manifest.json, or records.jsonl.")
    parser.add_argument("--out", required=True, help="Output directory for pi-style samples.")
    parser.add_argument("--max-records", type=int, help="Limit source records for smoke exports.")
    parser.add_argument("--copy-images", action="store_true", help="Copy referenced images into the output directory.")
    parser.add_argument(
        "--require-tactile-context",
        action="store_true",
        help="Skip samples without estimated_contact_flag or estimated_image_distance_px.",
    )
    args = parser.parse_args()

    manifest = export_dataset(args)
    counts = manifest["counts"]
    print(f"wrote {Path(args.out) / 'manifest.json'}")
    print(
        "exported={exported_samples} skipped={skipped_samples} "
        "tactile={tactile_signal_samples} tasks={tasks} piper={piper_intents}".format(**counts)
    )


if __name__ == "__main__":
    main()
