"""Read-only numerical-path diagnosis after a pre-optimizer step0 failure.

Never imports a training utility, creates an optimizer, performs backward,
changes model/loss source, or replaces a historical result. All eight forward
configurations are reported, not selected by quality. Only inference is run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import product
from pathlib import Path
import json
import platform
import time

import numpy as np

from pi05_libero_world_model_adapter import (
    LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization, read_json, sha256_file,
)
from smoke_pi05_libero_world_model import (
    BATCH_SIZE, COUNTS, SEED, SPLIT, contained, input_fingerprints, parameter_sha,
    prediction_check, tensor_sha, validate_artifacts, write_json,
)
from eval_pi05_libero_world_model_persistence import (
    PROTOCOL_SHA, assert_matching_batch, load_protocol, validate_reference,
)


def compare_array(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("reference array shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite prediction/reference")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"element_count": actual.size, "different_elements": int(np.count_nonzero(actual != expected)),
            "max_absolute_difference": float(delta.max()), "mean_absolute_difference": float(delta.mean())}


def run(args) -> dict:
    protocol = load_protocol(args.protocol, PROTOCOL_SHA)
    _, verified = validate_artifacts(args.feature_pack, protocol["data"]["feature_report_sha256"])
    _, traces, references = validate_reference(args.random_reference, protocol, verified)
    failure_root = args.failed_attempt
    failed_status = read_json(failure_root / "status.json")
    expected_status = {"status": "failed", "training_started": False, "completed_updates": 0,
                       "last_attempted_step": 0, "automatic_retry_allowed": False}
    if any(type(failed_status.get(k)) is not type(v) or failed_status.get(k) != v for k, v in expected_status.items()):
        raise ValueError("only the preserved zero-update failure is in diagnostic scope")
    failed_hashes = {p.name: sha256_file(p) for p in failure_root.iterdir() if p.is_file()}
    if set(failed_hashes) != {"status.json", "run.log", "sampling_plan.json", "evaluation_step000_trace.jsonl"}:
        raise ValueError("unexpected failed-attempt files; no training artifact should exist")

    import torch
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
    from pi05_libero_world_model_objectives import collate_window_targets

    if str(torch.__version__) != protocol["future_learning_diagnostic"]["torch_version"]:
        raise ValueError("requires the pinned remote Torch runtime")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"torch": str(torch.__version__), "threads": torch.get_num_threads(),
               "interop_threads": torch.get_num_interop_threads(), "platform": platform.platform(),
               "processor": platform.processor(), "torch_config": torch.__config__.show(),
               "mkldnn_enabled": torch.backends.mkldnn.enabled,
               "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    implementation_names = (Path(__file__).name, "pi05_libero_world_model.py",
                           "pi05_libero_world_model_adapter.py", "pi05_libero_world_model_objectives.py",
                           "train_pi05_libero_world_model.py", "run_pi05_libero_world_model_learning_diagnostic.py")
    implementations = {name: sha256_file(Path(__file__).with_name(name)) for name in implementation_names}
    results, full_checks = [], []
    started = time.monotonic()
    with LiberoFeaturePack(args.feature_pack) as pack:
        split = pack.load_split(args.feature_pack / "split.json")
        if any(split[k] != v for k, v in SPLIT.items()):
            raise ValueError("split mismatch")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(args.feature_pack / "preparation" / "normalization.json"):
            raise ValueError("normalization mismatch")
        datasets = {p: LiberoWindowDataset(pack, split, partition=p, normalization=stats) for p in COUNTS}
        if {p: len(ds) for p, ds in datasets.items()} != COUNTS:
            raise ValueError("window counts mismatch")
        rows = {p: [ds[i] for i in range(len(ds))] for p, ds in datasets.items()}
        saved = {}
        for part in COUNTS:
            with np.load(args.random_reference / f"first_{part}_audit.npz", allow_pickle=False) as archive:
                saved[part] = {k: archive[k].copy() for k in archive.files}

        def model_for(require_grad):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                model = LiberoWorldModel(LiberoWorldModelConfig(**protocol["model"]["config"]), pack.manifest["task_registry"])
            model.to("cpu").eval().requires_grad_(require_grad)
            if parameter_sha(model) != protocol["baselines"]["random_reference_initial_parameter_sha256"]:
                raise ValueError("initial parameter SHA mismatch")
            return model

        def predict(model, part, start, inference_inputs):
            batch = rows[part][start:start + BATCH_SIZE]
            with torch.inference_mode() if inference_inputs else nullcontext():
                inputs = collate_window_inputs([r["inputs"] for r in batch], device="cpu")
                targets = collate_window_targets([r["targets"] for r in batch], device="cpu")
            assert_matching_batch(inputs, targets, targets, batch, traces[(part, start)])
            before = input_fingerprints(inputs)
            with torch.inference_mode():
                predictions = model(**inputs)
                prediction_check(predictions, len(batch), 1)
                hashes = {k: tensor_sha(v) for k, v in predictions.items()}
                arrays = {k: v.detach().cpu().numpy().copy() for k, v in predictions.items()}
            if before != input_fingerprints(inputs) or any(p.grad is not None for p in model.parameters()):
                raise ValueError("input mutation or unexpected gradients")
            info = {k: {"is_inference": torch.is_inference(v), "requires_grad": v.requires_grad,
                        "stride": list(v.stride()), "contiguous": v.is_contiguous()}
                    for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            return hashes, arrays, info

        # Two independent repeats of every case, both saved first batches.
        for deterministic, requires_grad, inference_inputs in product((False, True), repeat=3):
            torch.use_deterministic_algorithms(deterministic)
            case = {"deterministic_algorithms": deterministic, "parameter_requires_grad": requires_grad,
                    "collate_in_inference_context": inference_inputs, "first_batch_repeats": []}
            for repeat in range(2):
                model = model_for(requires_grad)
                before, versions = parameter_sha(model), [p._version for p in model.parameters()]
                result = {}
                for part in COUNTS:
                    hashes, arrays, input_info = predict(model, part, 0, inference_inputs)
                    result[part] = {"predictions": hashes,
                        "exact_reference_hashes": hashes == traces[(part, 0)]["predictions"],
                        "array_differences": {k: compare_array(v, saved[part][k]) for k, v in arrays.items()},
                        "input_tensor_properties": input_info}
                if parameter_sha(model) != before or [p._version for p in model.parameters()] != versions:
                    raise ValueError("inference changed parameters")
                case["first_batch_repeats"].append(result)
            results.append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "first_batch_repeats"}),
                  {p: case["first_batch_repeats"][0][p]["exact_reference_hashes"] for p in COUNTS}, flush=True)

        # Predeclared full-trace checks: reproduce failed path and test ONLY
        # requires_grad=False while leaving ordinary inputs + deterministic=True.
        # This is no automatic retry of the learning runner and performs no loss.
        torch.use_deterministic_algorithms(True)
        for requires_grad in (True, False):
            model = model_for(requires_grad)
            before = parameter_sha(model)
            batches = []
            for part in COUNTS:
                for start in range(0, COUNTS[part], BATCH_SIZE):
                    hashes, _, _ = predict(model, part, start, False)
                    batches.append({"partition": part, "window_start": start,
                                    "batch_size": min(BATCH_SIZE, COUNTS[part] - start),
                                    "predictions": hashes,
                                    "exact_reference_hashes": hashes == traces[(part, start)]["predictions"]})
            if before != parameter_sha(model):
                raise ValueError("full trace changed parameters")
            full_checks.append({"deterministic_algorithms": True, "parameter_requires_grad": requires_grad,
                                "collate_in_inference_context": False, "batches": batches,
                                "exact_matching_batches": sum(r["exact_reference_hashes"] for r in batches),
                                "batch_count": len(batches), "window_count": sum(r["batch_size"] for r in batches)})
    for root, hashes in ((args.feature_pack, verified), (args.random_reference, references), (failure_root, failed_hashes)):
        if any(sha256_file(contained(root, k)) != v for k, v in hashes.items()):
            raise ValueError("bound evidence changed during diagnosis")
    if any(sha256_file(Path(__file__).with_name(k)) != v for k, v in implementations.items()):
        raise ValueError("implementation changed during diagnosis")
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA initialization")
    return {"schema": "pi05_libero_step0_reproducibility_probe_v1", "status": "completed",
            "scope": "read_only_forward_path_diagnosis_not_learning_retry",
            "runtime": runtime, "runtime_seconds": time.monotonic() - started,
            "protocol_sha256": PROTOCOL_SHA, "implementation_sha256": implementations,
            "verified_feature_artifacts": verified, "verified_reference_artifacts": references,
            "preserved_failure_status": failed_status, "preserved_failure_sha256": failed_hashes,
            "parameter_sha256": protocol["baselines"]["random_reference_initial_parameter_sha256"],
            "first_batch_factorial": results, "full_trace_checks": full_checks,
            "optimizer_steps": 0, "backward_calls": 0, "policy_actions": 0,
            "pi05_loaded": False, "model_loss_or_protocol_changed": False,
            "automatic_training_retry_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-pack", "random-reference", "failed-attempt", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("diagnostic output must not already exist")
    result = run(args)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "report.json", result)
    print("STEP0_READ_ONLY_DIAGNOSIS_COMPLETED", sha256_file(args.out / "report.json"), flush=True)


if __name__ == "__main__":
    main()
