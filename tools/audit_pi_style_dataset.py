from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PI_STATE_FIELDS = [
    "elite_tcp_pose_6d",
    "piper_step",
    "piper_insertion_length",
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "estimated_image_distance_px",
]

TACTILE_FIELDS = [
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "estimated_image_distance_px",
]


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


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


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


def load_dataset(path: Path, max_records: int | None = None) -> tuple[str, Path, list[dict[str, Any]], dict[str, Any]]:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        records_path = path / "records.jsonl"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest.get("samples"), list):
                return "sim_manifest", path, list(manifest.get("samples", []))[:max_records], manifest
            if records_path.exists():
                return "real_records", path, read_jsonl(records_path, max_records), manifest
        if records_path.exists():
            manifest = {}
            optional_manifest = path / "manifest.json"
            if optional_manifest.exists():
                manifest = json.loads(optional_manifest.read_text(encoding="utf-8"))
            return "real_records", path, read_jsonl(records_path, max_records), manifest
        raise FileNotFoundError(f"dataset dir has no manifest.json or records.jsonl: {path}")

    if path.name == "manifest.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return "sim_manifest", path.parent, list(manifest.get("samples", []))[:max_records], manifest
    if path.name.endswith(".jsonl"):
        return "real_records", path.parent, read_jsonl(path, max_records), {}
    raise ValueError(f"unsupported dataset path: {path}")


def row_images(row: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "sim_manifest":
        images = row.get("images", {}) if isinstance(row.get("images"), dict) else {}
        return {"side": images.get("side"), "top": images.get("top")}
    return {"side": row.get("side_image"), "top": row.get("top_image")}


def row_instruction(row: dict[str, Any]) -> str | None:
    instruction = row.get("instruction")
    if isinstance(instruction, str) and instruction:
        return instruction
    task = row.get("task")
    if task in {"left", "right"}:
        return f"enter {task} branch"
    return None


def row_state(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    return state if isinstance(state, dict) else {}


def row_action(row: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "sim_manifest":
        action = row.get("action")
    else:
        action = row.get("reference_action")
    return action if isinstance(action, dict) else {}


def action_piper_label(action: dict[str, Any]) -> str | None:
    label = action.get("piper_command_label")
    if isinstance(label, str):
        return label
    command = action.get("piper_step_command")
    if command == -1:
        return "retract"
    if command == 0:
        return "hold"
    if command == 1:
        return "feed"
    feed = action.get("piper_feed")
    if isinstance(feed, (int, float)):
        if feed > 0.05:
            return "feed"
        if feed < -0.05:
            return "retract"
        return "hold"
    return None


def action_elite_delta(action: dict[str, Any]) -> list[float] | None:
    value = action.get("elite_tcp_delta_6d")
    if numeric_list(value, 6):
        return [float(x) for x in value]
    return None


def summarize_dataset(path: Path, max_records: int | None, check_images: bool) -> dict[str, Any]:
    kind, base_dir, rows, manifest = load_dataset(path, max_records=max_records)
    counts = {
        "records": len(rows),
        "side_image": 0,
        "top_image": 0,
        "side_image_exists": 0,
        "top_image_exists": 0,
        "instruction": 0,
        "elite_tcp_pose_6d": 0,
        "piper_state": 0,
        "elite_tcp_delta_6d": 0,
        "piper_intent": 0,
    }
    state_field_counts = Counter()
    tactile_non_null = Counter()
    tactile_any = 0
    tactile_signal_rows = 0
    piper_intents: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    warnings: list[str] = []
    elite_delta_linf: list[float] = []
    exact_sim_contact_rows = 0

    for row in rows:
        task = row.get("task")
        if isinstance(task, str):
            tasks[task] += 1

        images = row_images(row, kind)
        for image_key in ("side", "top"):
            image_ref = images.get(image_key)
            if image_ref:
                counts[f"{image_key}_image"] += 1
                if check_images and resolve_path(str(image_ref), base_dir).exists():
                    counts[f"{image_key}_image_exists"] += 1

        if row_instruction(row):
            counts["instruction"] += 1

        state = row_state(row)
        for field in PI_STATE_FIELDS:
            if field in state:
                state_field_counts[field] += 1
            if field in TACTILE_FIELDS and non_null(state.get(field)):
                tactile_non_null[field] += 1
        if any(non_null(state.get(field)) for field in TACTILE_FIELDS):
            tactile_any += 1
        if non_null(state.get("estimated_contact_flag")) or non_null(state.get("estimated_image_distance_px")):
            tactile_signal_rows += 1
        if numeric_list(state.get("elite_tcp_pose_6d"), 6):
            counts["elite_tcp_pose_6d"] += 1
        if non_null(state.get("piper_step")) or non_null(state.get("piper_insertion_length")):
            counts["piper_state"] += 1
        if "contact_flag" in state and not any(non_null(state.get(field)) for field in TACTILE_FIELDS):
            exact_sim_contact_rows += 1

        action = row_action(row, kind)
        elite_delta = action_elite_delta(action)
        if elite_delta is not None:
            counts["elite_tcp_delta_6d"] += 1
            elite_delta_linf.append(max(abs(x) for x in elite_delta))
        piper_label = action_piper_label(action)
        if piper_label:
            counts["piper_intent"] += 1
            piper_intents[piper_label] += 1

    required = [
        "side_image",
        "top_image",
        "instruction",
        "elite_tcp_pose_6d",
        "piper_state",
        "elite_tcp_delta_6d",
        "piper_intent",
    ]
    missing_required = [field for field in required if counts[field] < len(rows)]
    if check_images:
        for image_key in ("side", "top"):
            if counts[f"{image_key}_image_exists"] < counts[f"{image_key}_image"]:
                warnings.append(f"{image_key} image path existence is incomplete")
    if exact_sim_contact_rows:
        warnings.append(
            "exact simulator contact fields exist without estimator-style tactile fields; do not use them as policy context"
        )
    if tactile_signal_rows == 0:
        warnings.append("no non-null contact flag or image-distance tactile signal found")
    if len(piper_intents) <= 1:
        warnings.append("piper intent labels are single-class or missing; weak for supervised Piper intent learning")
    if not elite_delta_linf or math.isclose(max(elite_delta_linf), 0.0, abs_tol=1e-12):
        warnings.append("Elite TCP delta appears all-zero or missing; weak for Elite action learning")

    readiness = {
        "pi_style_adapter_minimum": bool(rows) and not missing_required,
        "tactile_context_ready": tactile_signal_rows > 0,
        "piper_intent_multiclass": len(piper_intents) > 1,
        "elite_action_nonzero": bool(elite_delta_linf and max(elite_delta_linf) > 0.0),
    }
    if readiness["pi_style_adapter_minimum"] and readiness["tactile_context_ready"]:
        recommendation = "candidate_for_pi_style_with_tactile_context"
    elif readiness["pi_style_adapter_minimum"]:
        recommendation = "candidate_for_pi_style_without_tactile_context_or_needs_contact_estimator_fill"
    else:
        recommendation = "not_ready_for_pi_style_adapter"

    return {
        "path": str(path),
        "kind": kind,
        "base_dir": str(base_dir),
        "manifest_mode": manifest.get("mode"),
        "manifest_action_schema": (manifest.get("action_schema") or {}).get("type"),
        "records": len(rows),
        "counts": counts,
        "tasks": dict(tasks),
        "piper_intents": dict(piper_intents),
        "state_field_presence": dict(state_field_counts),
        "tactile_non_null": dict(tactile_non_null),
        "tactile_any_non_null": tactile_any,
        "tactile_signal_rows": tactile_signal_rows,
        "elite_delta_linf": {
            "max": max(elite_delta_linf) if elite_delta_linf else None,
            "p50": percentile(elite_delta_linf, 0.50),
            "p95": percentile(elite_delta_linf, 0.95),
        },
        "missing_required_full_coverage": missing_required,
        "warnings": warnings,
        "readiness": readiness,
        "recommendation": recommendation,
    }


def print_text_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"\n{result['path']}")
        if "error" in result:
            print(f"  error={result['error']}")
            continue
        print(f"  kind={result['kind']} records={result['records']} recommendation={result['recommendation']}")
        print(f"  tasks={result['tasks']} piper_intents={result['piper_intents']}")
        print(f"  counts={result['counts']}")
        print(
            f"  tactile_non_null={result['tactile_non_null']} "
            f"any={result['tactile_any_non_null']} signal_rows={result['tactile_signal_rows']}"
        )
        print(f"  elite_delta_linf={result['elite_delta_linf']}")
        if result["missing_required_full_coverage"]:
            print(f"  missing_required_full_coverage={result['missing_required_full_coverage']}")
        for warning in result["warnings"]:
            print(f"  warning: {warning}")


def expand_default_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in (Path("simulation_output"), Path("collected_data")):
        if not root.exists():
            continue
        for name in ("manifest.json", "records.jsonl"):
            candidates.extend(path.parent for path in root.rglob(name))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit existing sim/real datasets for pi-style VLA sample compatibility."
    )
    parser.add_argument("datasets", nargs="*", help="Dataset dirs, manifest.json files, or records.jsonl files.")
    parser.add_argument("--scan-defaults", action="store_true", help="Scan all manifest/records datasets under defaults.")
    parser.add_argument("--max-records", type=int, help="Limit records per dataset for a quick audit.")
    parser.add_argument("--check-images", action="store_true", help="Check whether referenced image paths exist.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    dataset_paths = [Path(value) for value in args.datasets]
    if args.scan_defaults:
        dataset_paths.extend(expand_default_candidates())
    if not dataset_paths:
        raise SystemExit("provide at least one dataset path or --scan-defaults")

    results: list[dict[str, Any]] = []
    for path in dataset_paths:
        try:
            results.append(summarize_dataset(path, max_records=args.max_records, check_images=args.check_images))
        except Exception as exc:
            results.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    print_text_summary(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"datasets": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
