"""Fixed public feature pair -> untrained native world-model forward checks.

No PI0.5 load, loss, backward, optimizer, policy selection or environment step.
The two-episode pair is a pipeline diagnostic, not a generalization benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from pi05_libero_world_model_adapter import (
    ARRAY_NAMES, DEMO_REPO, DEMO_REVISION, LiberoFeaturePack, LiberoWindowDataset,
    fit_train_normalization, read_json, sha256_file,
)

SCHEMA = "pi05_libero_world_model_forward_smoke_v1"
SOURCE_SHA = "ab691452b5901836a1638096955b15b294c09f32c2b2a0915a566cc36181187f"
PLAN_SHA = "bad36b6a93769d4210792c60a42892a325d30694d8a087316deeafab7de48a53"
CHECKPOINT_SHA = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
SPLIT = {"train_episode_indices": [1400], "validation_episode_indices": [1402]}
COUNTS = {"train": 134, "validation": 167}
SEED = 20260912
BATCH_SIZE = 16
PRED_KEYS = {"pred_future_visual_latent", "pred_state_delta"}
FROZEN_FEATURE_IMPLEMENTATIONS = {
    "build_pi05_libero_feature_pack.py", "pi05_libero_feature_extraction.py",
    "pi05_action_effect_world_model.py", "probe_pi05_libero_public_features.py",
    "pi05_pretrained_loader.py", "pi05_libero_world_model_adapter.py",
    "prepare_pi05_libero_world_model.py",
}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_value(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("external SHA256 must be 64 lowercase hex characters")
    return value


def contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute():
        raise ValueError("relative POSIX artifact name required")
    if any(part in {".", ".."} for part in name.split("/")):
        raise ValueError("artifact path traversal rejected")
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError(f"required artifact missing: {name}")
    return path


def check_report_contract(report: dict) -> None:
    expected = {
        "schema": "pi05_libero_feature_pair_build_v1", "status": "passed",
        "source": {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION},
        "source_report_sha256": SOURCE_SHA, "plan_sha256": PLAN_SHA, "split": SPLIT,
        "latent_shape": [313, 2, 2048], "state_shape": [313, 8], "action_shape": [313, 7],
        "complete_episode_counts": {"1400": 140, "1402": 173}, "windows": COUNTS,
        "window_context_len": 4, "window_horizon": 3,
        "normalization_state_rows": 140, "normalization_action_rows": 139,
        "feature_pack_complete_for_fixed_two_episodes": True,
        "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
        "training_ready": False, "training_started": False,
        "optimizer_steps": 0, "actions_generated": 0, "simulation_steps": 0,
        "source_files_unchanged": True,
    }
    for key, value in expected.items():
        if report.get(key) != value or type(report.get(key)) is not type(value):
            raise ValueError(f"fixed feature-pair report contract differs: {key}")
    extraction = report.get("extraction_evidence", {})
    for key, value in {
        "complete_rows_extracted": 313, "preprocessing_call_count": 313,
        "real_view_embedding_call_count": 626, "empty_camera_embedded": False,
        "all_modules_eval": True, "all_parameters_frozen": True,
        "all_parameter_gradients_absent": True, "parameter_versions_unchanged": True,
        "all_calls_inference_mode": True,
    }.items():
        if extraction.get(key) != value or type(extraction.get(key)) is not type(value):
            raise ValueError(f"frozen extraction evidence differs: {key}")
    checkpoint = report.get("checkpoint", {})
    if (checkpoint.get("model_sha256") != CHECKPOINT_SHA
            or checkpoint.get("load", {}).get("loaded_parameter_fraction") != 1.0):
        raise ValueError("pinned frozen extractor checkpoint provenance required")


def validate_artifacts(root: Path, report_sha256: str) -> tuple[dict, dict]:
    root = root.resolve()
    report_path = contained(root, "report.json")
    if sha256_file(report_path) != digest_value(report_sha256):
        raise ValueError("feature build report differs from external SHA256")
    report = read_json(report_path)
    check_report_contract(report)
    outputs = report.get("output_sha256")
    required = {f"{name}.npy" for name in ARRAY_NAMES} | {
        "manifest.json", "split.json", "extractor_provenance.json", "extraction_trace.jsonl",
        "preparation/normalization.json", "preparation/train_windows.jsonl",
        "preparation/validation_windows.jsonl",
    }
    if not isinstance(outputs, dict) or not required <= set(outputs):
        raise ValueError("complete feature output inventory required")
    if {"report.json", "status.json", "run.log"} & set(outputs):
        raise ValueError("self-referential/mutable output hash entry rejected")
    verified = {"report.json": report_sha256, **outputs,
                "preparation/report.json": digest_value(report.get("preparation_report_sha256"))}
    for name, digest in verified.items():
        if sha256_file(contained(root, name)) != digest_value(digest):
            raise ValueError(f"feature artifact hash mismatch: {name}")
    manifest = read_json(root / "manifest.json")
    if (verified["manifest.json"] != report.get("manifest_sha256")
            or verified["split.json"] != report.get("split_sha256")
            or manifest.get("source", {}).get("metadata_sha256") != SOURCE_SHA
            or manifest.get("extractor", {}).get("checkpoint_sha256") != CHECKPOINT_SHA):
        raise ValueError("manifest/split/source/checkpoint linkage differs")
    provenance = read_json(root / "extractor_provenance.json")
    if (provenance.get("plan_sha256") != PLAN_SHA
            or provenance.get("source_report_sha256") != SOURCE_SHA
            or provenance.get("checkpoint") != report["checkpoint"]
            or manifest.get("extractor", {}).get("code_sha256") != outputs["extractor_provenance.json"]
            or set(provenance.get("repo_sources", {})) != FROZEN_FEATURE_IMPLEMENTATIONS):
        raise ValueError("extractor provenance linkage differs")
    for name, digest in provenance.get("repo_sources", {}).items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest_value(digest):
            raise ValueError(f"frozen feature implementation differs: {name}")
    return report, verified


def tensor_sha(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + value.numpy().tobytes()).hexdigest()


def parameter_sha(model) -> str:
    content = {key: tensor_sha(value) for key, value in model.state_dict().items()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def input_fingerprints(inputs: dict) -> dict:
    return {key: list(value) if key == "task_instruction" else tensor_sha(value)
            for key, value in inputs.items()}


def prediction_check(outputs: dict, batch: int, candidates: int) -> None:
    import torch
    if set(outputs) != PRED_KEYS:
        raise ValueError("native world model may output only visual/state predictions")
    expected = {"pred_future_visual_latent": (batch, candidates, 3, 2, 2048),
                "pred_state_delta": (batch, candidates, 3, 8)}
    for key, value in outputs.items():
        if (tuple(value.shape) != expected[key] or value.dtype != torch.float32
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"invalid inference-only prediction: {key}")


def candidate_wiring_probe(model, inputs: dict) -> dict:
    """Synthetic normalized tensor perturbations; never dispatched as actions."""
    import torch
    base = {**inputs, "candidate_actions": inputs["candidate_actions"].repeat(1, 3, 1, 1)}
    reference = model(**base)
    prediction_check(reference, len(base["task_instruction"]), 3)
    repeated_difference = max(float((value[:, 0] - value[:, 1]).abs().max()) for value in reference.values())
    changed = {**base, "candidate_actions": base["candidate_actions"].clone()}
    changed["candidate_actions"][:, 1, -1, 0] += 0.25
    changed["candidate_actions"][:, 2, 0, 0] -= 0.25
    predicted = model(**changed)
    prediction_check(predicted, len(base["task_instruction"]), 3)
    permutation = [2, 0, 1]
    permuted = model(**{**changed, "candidate_actions": changed["candidate_actions"][:, permutation]})
    unchanged_candidate = max(float((predicted[key][:, 0] - reference[key][:, 0]).abs().max()) for key in PRED_KEYS)
    early_difference = max(float((predicted[key][:, 1, :-1] - reference[key][:, 1, :-1]).abs().max()) for key in PRED_KEYS)
    changed_final = max(float((predicted[key][:, 1, -1] - reference[key][:, 1, -1]).abs().max()) for key in PRED_KEYS)
    permutation_difference = max(float((permuted[key] - predicted[key][:, permutation]).abs().max()) for key in PRED_KEYS)
    for key in PRED_KEYS:
        torch.testing.assert_close(permuted[key], predicted[key][:, permutation], atol=1e-6, rtol=1e-5)
    if max(repeated_difference, unchanged_candidate, early_difference) != 0 or changed_final <= 1e-8:
        raise ValueError("candidate isolation/causality/action-dependency wiring check failed")
    return {"scope": "untrained wiring_only_not_prediction_quality_or_action_selection",
            "batch_size": len(base["task_instruction"]), "candidate_count": 3,
            "duplicate_candidate_max_abs_difference": repeated_difference,
            "unmodified_candidate_max_abs_difference": unchanged_candidate,
            "future_action_to_earlier_prediction_max_abs_difference": early_difference,
            "changed_last_action_to_final_prediction_max_abs_difference": changed_final,
            "candidate_permutation_max_abs_difference": permutation_difference,
            "perturbation_units": "0.25 in train-normalized action coordinate0",
            "policy_actions_generated": 0, "commands_dispatched": 0}


def run(root: Path, report_sha256: str, out: Path, *, device: str = "cpu") -> dict:
    source_report, verified = validate_artifacts(root, report_sha256)
    implementations = {name: sha256_file(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "pi05_libero_world_model.py", "pi05_libero_world_model_adapter.py",
        "pi05_action_effect_world_model.py", "train_pi05_action_effect_world_model.py",
    )}
    print("FROZEN_FEATURE_ARTIFACTS_VERIFIED rows=313", flush=True)
    import torch
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
    torch.set_num_threads(1)
    config = LiberoWorldModelConfig()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    with LiberoFeaturePack(root) as pack:
        split = pack.load_split(root / "split.json")
        if any(split[key] != value for key, value in SPLIT.items()):
            raise ValueError("only the predeclared two-episode split is in scope")
        if {e: len(rows) for e, rows in pack.indices.items()} != {1400: 140, 1402: 173} or pack.visual_dim != 2048:
            raise ValueError("fixed native complete episode counts/visual dimension differ")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(root / "preparation" / "normalization.json"):
            raise ValueError("saved normalization differs from training-only recomputation")
        datasets = {part: LiberoWindowDataset(pack, split, partition=part, normalization=stats)
                    for part in COUNTS}
        if {part: len(dataset) for part, dataset in datasets.items()} != COUNTS:
            raise ValueError("full native window counts differ")
        # Initialization only; no optimizer/loss/checkpoint saving or loading.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = LiberoWorldModel(config, task_registry=pack.manifest["task_registry"])
        model = model.to(device).eval().requires_grad_(False)
        before = parameter_sha(model)
        parameter_versions = [p._version for p in model.parameters()]
        metadata = model.metadata()
        write_json(out / "model_contract.json", metadata)
        parts = {}
        with (out / "forward_trace.jsonl").open("x", encoding="utf-8") as trace:
            with torch.inference_mode():
                for part, dataset in datasets.items():
                    calls = 0
                    for start in range(0, len(dataset), BATCH_SIZE):
                        items = [dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))]
                        inputs = collate_window_inputs([item["inputs"] for item in items], device=device)
                        inputs_before = input_fingerprints(inputs)
                        outputs = model(**inputs)
                        prediction_check(outputs, len(items), 1)
                        if input_fingerprints(inputs) != inputs_before:
                            raise ValueError("model modified caller input tensors")
                        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                            raise ValueError("inference-only context lost")
                        calls += 1
                        for offset, item in enumerate(items):
                            record = {"partition": part, "window_index": start + offset,
                                      "metadata": item["metadata"],
                                      "task_instruction": inputs["task_instruction"][offset],
                                      "inputs": {key: tensor_sha(value[offset:offset+1])
                                                 for key, value in inputs.items() if key != "task_instruction"},
                                      "predictions": {key: tensor_sha(value[offset:offset+1]) for key, value in outputs.items()},
                                      "inference_mode": True, "grad_enabled": False}
                            trace.write(json.dumps(record, allow_nan=False) + "\n")
                        if start == 0:
                            with (out / f"first_{part}_predictions.npz").open("xb") as stream:
                                np.savez(stream, **{key: value[:1].detach().cpu().numpy() for key, value in outputs.items()})
                        if (start + len(items)) % 64 == 0 or start + len(items) == len(dataset):
                            print(f"ZERO_STEP_FORWARD partition={part} windows={start+len(items)}/{len(dataset)}", flush=True)
                    parts[part] = {"windows": len(dataset), "forward_batches": calls,
                                   "all_outputs_finite": True, "caller_inputs_unchanged": True}
                probe_inputs = collate_window_inputs([datasets[part][0]["inputs"] for part in COUNTS], device=device)
                probe_before = input_fingerprints(probe_inputs)
                wiring = candidate_wiring_probe(model, probe_inputs)
                if input_fingerprints(probe_inputs) != probe_before:
                    raise ValueError("candidate probe mutated recorded inputs")
        after = parameter_sha(model)
        if (before != after or parameter_versions != [p._version for p in model.parameters()]
                or any(module.training for module in model.modules())
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise ValueError("untrained frozen smoke model changed or gradient appeared")
        task_count = len(pack.tasks)
        parameter_count = sum(p.numel() for p in model.parameters())
    for name, digest in verified.items():
        if sha256_file(contained(root, name)) != digest:
            raise ValueError(f"source artifact changed during forward: {name}")
    for name, digest in implementations.items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"implementation changed during forward: {name}")
    return {"schema": SCHEMA, "status": "passed", "scope": "untrained_forward_interface_only",
            "feature_pack": str(root.resolve()), "feature_report_sha256": report_sha256,
            "source_report_sha256": SOURCE_SHA, "plan_sha256": PLAN_SHA,
            "manifest_sha256": source_report["manifest_sha256"], "split_sha256": source_report["split_sha256"],
            "split": SPLIT, "seed": SEED, "batch_size": BATCH_SIZE, "device": device,
            "torch_version": torch.__version__, "model_contract": metadata, "parameter_count": parameter_count,
            "parameter_sha256_before": before, "parameter_sha256_after": after,
            "all_parameters_frozen": True, "all_parameter_gradients_absent": True, "all_modules_eval": True,
            "parameter_versions_unchanged": True, "partitions": parts, "candidate_wiring": wiring,
            "normalization_state_rows": stats["state_record_count"], "normalization_action_rows": stats["action_record_count"],
            "real_task_count": task_count, "real_multitask_conditioning_tested": False,
            "task_conditioning_scope": "exact registered instruction -> closed-set embedding, not language grounding",
            "inputs_targets_metadata_separated": True, "source_artifacts_unchanged": True,
            "implementation_sha256": implementations, "verified_feature_artifacts": verified,
            "output_sha256": {path.name: sha256_file(path) for path in out.iterdir()
                              if path.is_file() and path.name not in {"report.json", "status.json", "run.log"}},
            "world_model_randomly_initialized": True, "world_model_weights_saved": False,
            "pi05_loaded": False, "training_started": False, "training_ready": False,
            "optimizer_steps": 0, "backward_calls": 0, "loss_computed": False,
            "policy_actions_generated": 0, "actions_dispatched": 0, "simulation_steps": 0,
            "candidate_ranking_implemented": False, "world_model_quality_evaluated": False,
            "policy_generalization_evaluated": False, "family_independence_verified": False,
            "checkpoint_training_overlap_unknown": True, "real_system_validated": False,
            "visual_review_required": False, "visual_scope": "no pixels or generated rollout behavior changed"}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-pack", type=Path, required=True)
    parser.add_argument("--feature-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "pid": os.getpid(),
                                          "training_started": False, "optimizer_steps": 0})
    with (args.out / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                result = run(args.feature_pack, args.feature_report_sha256, args.out, device=args.device)
                result["runtime_seconds"] = time.monotonic() - started
                write_json(args.out / "report.json", result)
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "completed",
                           "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 0})
                print("NATIVE_WORLD_MODEL_ZERO_STEP_PASSED", json.dumps({"windows": COUNTS,
                      "runtime_seconds": result["runtime_seconds"], "optimizer_steps": 0}), flush=True)
            except BaseException as error:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "failed",
                           "error_type": type(error).__name__, "error": str(error), "optimizer_steps": 0,
                           "training_started": False})
                raise


if __name__ == "__main__":
    main()
