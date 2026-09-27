"""CPU-only LIBERO feature-pack/window preparation. No model or optimizer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pi05_libero_world_model_adapter import (
    LAYOUT, SCHEMA, SPLIT_SCHEMA, LiberoFeaturePack, LiberoWindowDataset,
    fit_train_normalization, sha256_file,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(pack_root: Path, split_path: Path, out: Path, *, allow_synthetic_fixture: bool = False,
            context_len: int = 4, horizon: int = 3) -> dict:
    if out.exists():
        raise FileExistsError(f"output exists; use a new attempt directory: {out}")
    with LiberoFeaturePack(pack_root, allow_synthetic_fixture=allow_synthetic_fixture) as pack:
        return _prepare_checked(pack, split_path, out, context_len=context_len, horizon=horizon)


def _prepare_checked(pack: LiberoFeaturePack, split_path: Path, out: Path, *, context_len: int, horizon: int) -> dict:
    split = pack.load_split(split_path)
    stats = fit_train_normalization(pack, split)
    datasets = {name: LiberoWindowDataset(pack, split, partition=name, normalization=stats,
                                        context_len=context_len, horizon=horizon)
                for name in ("train", "validation")}
    report = {
        "schema": "pi05_libero_native_preparation_report_v1", "status": "passed",
        "evidence_scope": "synthetic_interface_only" if pack.synthetic else "declared_public_feature_pack_schema_only",
        "feature_pack_manifest_sha256": pack.manifest_sha256, "split_sha256": split["split_sha256"],
        "source": pack.manifest["source"], "extractor": pack.manifest["extractor"], "layout": LAYOUT,
        "source_dataset_independently_verified": False, "training_ready": False,
        "optimizer_steps": 0, "training_started": False, "policy_loaded": False,
        "guidewire_model_changed": False, "b4b_outputs_requested_or_opened_by_entrypoint": False,
        "episode_completeness_and_source_identity": "manifest_declarations_pending_source_audit",
        "context_len": context_len, "horizon": horizon, "fps": pack.manifest["fps"],
        "history_span_seconds": (context_len - 1) / pack.manifest["fps"],
        "prediction_horizon_seconds": horizon / pack.manifest["fps"],
        "task_mapping_count": len(pack.tasks), "visual_dim": pack.visual_dim, "partitions": {},
        "normalization_scope": "world_model_only_train_episodes_not_pi05_processors",
        "missing_state": "normalized_zero_with_false_mask; no_train_support_disabled_even_in_validation",
        "state_target_semantics": LAYOUT["state_target"],
        "source_acceptance_and_real_pi05_probe_pending": True,
    }
    for name, dataset in datasets.items():
        valid_state_targets = valid_visual_targets = 0
        for item in dataset:
            valid_state_targets += int(item["targets"]["state_target_valid"].sum())
            valid_visual_targets += int(item["targets"]["future_visual_valid"].sum())
        if not valid_state_targets or not valid_visual_targets:
            raise ValueError(f"{name}: no valid state/visual targets")
        first = dataset[0]
        report["partitions"][name] = {
            "episodes": len(split[f"{name}_episode_indices"]), "windows": len(dataset),
            "valid_state_target_coordinates": valid_state_targets,
            "valid_visual_target_views": valid_visual_targets,
            "input_shapes": {key: list(value.shape) for key, value in first["inputs"].items()
                             if isinstance(value, np.ndarray)},
            "target_shapes": {key: list(value.shape) for key, value in first["targets"].items()},
            "first_window_metadata": first["metadata"],
            "task_instruction_preserved": first["inputs"]["task_instruction"],
        }
    # All checks precede output creation. Existing source and report artifacts stay untouched.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "normalization.json", stats)
    for name, dataset in datasets.items():
        with (out / f"{name}_windows.jsonl").open("x", encoding="utf-8") as stream:
            for window in dataset.windows:
                row = {key: ([int(v) for v in value] if isinstance(value, tuple) else int(value))
                       for key, value in asdict(window).items()}
                stream.write(json.dumps(row) + "\n")
    report["outputs_sha256"] = {path.name: sha256_file(path) for path in out.iterdir() if path.is_file()}
    report["adapter_sha256"] = sha256_file(Path(__file__).with_name("pi05_libero_world_model_adapter.py"))
    report["entrypoint_sha256"] = sha256_file(Path(__file__))
    write_json(out / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-contract", action="store_true")
    parser.add_argument("--feature-pack", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps({"feature_pack_schema": SCHEMA, "split_schema": SPLIT_SCHEMA, "layout": LAYOUT,
                          "task_mapping": "explicit source_task_index plus exact native instruction to suite ID",
                          "training_started": False, "optimizer_steps": 0}, indent=2))
        return
    if not all((args.feature_pack, args.split, args.out)):
        parser.error("--feature-pack, --split and --out are required without --describe-contract")
    report = prepare(args.feature_pack, args.split, args.out,
                     allow_synthetic_fixture=args.allow_synthetic_fixture,
                     context_len=args.context_len, horizon=args.horizon)
    print(json.dumps({"status": report["status"], "scope": report["evidence_scope"],
                      "windows": {k: v["windows"] for k, v in report["partitions"].items()},
                      "optimizer_steps": 0, "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
