"""Fixed native persistence baseline; no learned model load, forward or training."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from pi05_libero_world_model_adapter import (
    LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization, read_json, sha256_file,
)
from smoke_pi05_libero_world_model import (
    BATCH_SIZE, COUNTS, SEED, SPLIT, Tee, contained, input_fingerprints,
    tensor_sha, validate_artifacts, write_json,
)
from train_pi05_libero_world_model import ObjectiveTotals

SCHEMA = "pi05_libero_persistence_evaluation_v1"
PROTOCOL_SHA = "5d1b67c75c03dcaeba6c53bdf7c0f8dc14d2fc5bb4d4e0d93f5c63c2ce58c2ff"
OWN_FILES = ("eval_pi05_libero_world_model_persistence.py", "pi05_libero_world_model_persistence.py")


def load_protocol(path: Path, expected_sha: str) -> dict:
    if expected_sha != PROTOCOL_SHA or sha256_file(path) != PROTOCOL_SHA:
        raise ValueError("only the frozen v1 diagnostic protocol bytes are accepted")
    return read_json(path)


def change(candidate, reference) -> dict:
    for value in (candidate, reference):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("finite nonnegative errors or unsupported null required")
    delta = candidate - reference if candidate is not None and reference is not None else None
    relative = 100.0 * delta / reference if delta is not None and reference != 0 else None
    if any(v is not None and not math.isfinite(v) for v in (delta, relative)):
        raise ValueError("nonfinite comparison")
    return {"candidate": candidate, "reference": reference, "absolute_change": delta,
            "relative_change_percent": relative, "lower_is_better": True}


def compare_stats(candidate, reference):
    if isinstance(candidate, dict):
        if type(reference) is not dict or set(candidate) != set(reference):
            raise ValueError("metric structures differ")
        if "count" in candidate:
            if candidate["count"] != reference["count"]:
                raise ValueError("metric valid counts differ")
            return {"count": candidate["count"], **{key: change(candidate[key], reference[key]) for key in ("mae", "rmse")}}
        return {key: compare_stats(value, reference[key]) for key, value in candidate.items()}
    if isinstance(candidate, list):
        if type(reference) is not list or len(candidate) != len(reference):
            raise ValueError("metric axes differ")
        return [compare_stats(a, b) for a, b in zip(candidate, reference)]
    raise ValueError("unexpected metric node")


def validate_reference(root: Path, protocol: dict, verified: dict) -> tuple[dict, dict, dict]:
    path = contained(root, "report.json")
    expected_sha = protocol["baselines"]["random_reference_report_sha256"]
    if sha256_file(path) != expected_sha:
        raise ValueError("random reference report differs from pinned SHA")
    report = read_json(path)
    expected = {"schema": "pi05_libero_native_objective_dry_run_v1", "status": "passed",
                "feature_report_sha256": protocol["data"]["feature_report_sha256"],
                "split": SPLIT, "seed": SEED, "batch_size": BATCH_SIZE, "device": "cpu",
                "optimizer_steps": 0, "backward_calls": 0, "training_started": False,
                "training_ready": False, "world_model_randomly_initialized": True,
                "parameter_sha256_before": protocol["baselines"]["random_reference_initial_parameter_sha256"],
                "parameter_sha256_after": protocol["baselines"]["random_reference_initial_parameter_sha256"],
                "verified_feature_artifacts": verified}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report.get(key) != value:
            raise ValueError(f"random reference contract differs: {key}")
    if report["model_contract"]["config"] != protocol["model"]["config"]:
        raise ValueError("model configuration differs")
    outputs = {"report.json": expected_sha, **report["output_sha256"]}
    for name, digest in outputs.items():
        if sha256_file(contained(root, name)) != digest:
            raise ValueError(f"reference artifact hash mismatch: {name}")
    for name, digest in report["implementation_sha256"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"reference implementation hash mismatch: {name}")
    traces = {}
    with (root / "dry_run_trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["partition"], row["window_start"])
            if key in traces:
                raise ValueError("duplicate reference batch")
            traces[key] = row
    expected_batches = {(part, start) for part, count in COUNTS.items() for start in range(0, count, BATCH_SIZE)}
    if set(traces) != expected_batches:
        raise ValueError("reference trace does not cover exactly all fixed windows")
    return report, traces, outputs


def assert_matching_batch(inputs: dict, targets: dict, common: dict, items: list, reference: dict) -> None:
    import torch
    for key in ("future_visual_valid", "state_target_valid"):
        if not torch.equal(targets[key], common[key]):
            raise ValueError("common mask differs: cannot reuse old random aggregate; both models need a new matched-support evaluation")
    if (reference["batch_size"] != len(items)
            or reference["metadata"] != [item["metadata"] for item in items]
            or reference["inputs"] != input_fingerprints(inputs)
            or reference["targets"] != {k: tensor_sha(v) for k, v in targets.items()}):
        raise ValueError("exact input/target/metadata reference batch mismatch")


def sampler_plan_hash(protocol: dict) -> str:
    import torch
    plan = protocol["future_learning_diagnostic"]
    generator = torch.Generator(device="cpu").manual_seed(plan["sampling_seed"])
    draws = torch.randint(protocol["data"]["train_windows"],
                          (plan["optimizer_steps"], plan["batch_size"]), generator=generator).tolist()
    actual = hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode("ascii")).hexdigest()
    if actual != plan["sampling_plan_sha256"]:
        raise ValueError("future diagnostic sampler sequence differs; no training attempted")
    return actual


def run(feature_root: Path, reference_root: Path, protocol_path: Path, protocol_sha: str, out: Path) -> dict:
    protocol = load_protocol(protocol_path, protocol_sha)
    feature_report, verified = validate_artifacts(feature_root, protocol["data"]["feature_report_sha256"])
    if (verified["manifest.json"] != protocol["data"]["manifest_sha256"]
            or verified["split.json"] != protocol["data"]["split_sha256"]
            or verified["preparation/normalization.json"] != protocol["data"]["normalization_sha256"]):
        raise ValueError("frozen protocol data linkage differs")
    reference, traces, reference_hashes = validate_reference(reference_root, protocol, verified)
    implementations = {**reference["implementation_sha256"],
                       **{name: sha256_file(Path(__file__).with_name(name)) for name in OWN_FILES}}
    print("PROTOCOL_FEATURE_REFERENCE_HASHES_VERIFIED", flush=True)
    import torch
    from pi05_libero_world_model import collate_window_inputs
    from pi05_libero_world_model_objectives import NativeWorldModelMetrics, collate_window_targets, native_world_model_loss
    from pi05_libero_world_model_persistence import persistence_predictions, persistence_scoring_targets, persistence_contract

    torch.set_num_threads(1)
    sample_hash = sampler_plan_hash(protocol)  # Offline plan only, not training samples consumed by an optimizer.
    write_json(out / "persistence_contract.json", persistence_contract())
    parts = {}
    with LiberoFeaturePack(feature_root) as pack:
        split = pack.load_split(feature_root / "split.json")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(feature_root / "preparation" / "normalization.json"):
            raise ValueError("train-only normalization mismatch")
        with (out / "persistence_trace.jsonl").open("x", encoding="utf-8") as stream:
            with torch.inference_mode():
                for part, expected_count in COUNTS.items():
                    dataset = LiberoWindowDataset(pack, split, partition=part, normalization=stats)
                    if len(dataset) != expected_count:
                        raise ValueError("full fixed window count differs")
                    totals = ObjectiveTotals()
                    metrics = NativeWorldModelMetrics(stats["state_std"])
                    for start in range(0, len(dataset), BATCH_SIZE):
                        items = [dataset[i] for i in range(start, min(start+BATCH_SIZE, len(dataset)))]
                        inputs = collate_window_inputs([item["inputs"] for item in items], device="cpu")
                        targets = collate_window_targets([item["targets"] for item in items], device="cpu")
                        before_i, before_t = input_fingerprints(inputs), {k: tensor_sha(v) for k, v in targets.items()}
                        common = persistence_scoring_targets(inputs, targets)
                        assert_matching_batch(inputs, targets, common, items, traces[(part, start)])
                        predictions = persistence_predictions(inputs)
                        loss, details = native_world_model_loss(predictions, common)
                        totals.update(details)
                        metrics.update(predictions, common)
                        if (loss.requires_grad or torch.is_grad_enabled() or not torch.is_inference_mode_enabled()
                                or input_fingerprints(inputs) != before_i
                                or {k: tensor_sha(v) for k, v in targets.items()} != before_t):
                            raise ValueError("inference-only or caller nonmutation guard failed")
                        stream.write(json.dumps({"partition": part, "window_start": start, "batch_size": len(items),
                            "metadata": [i["metadata"] for i in items], "inputs": before_i, "targets": before_t,
                            "common_masks_equal_original": True, "predictions": {k: tensor_sha(v) for k, v in predictions.items()},
                            "loss": float(loss), "objective_sufficient_statistics": {
                                k: float(v) if k.endswith("_sum") else v for k, v in details.items()
                                if k.endswith("_sum") or k.endswith("_count")}}, allow_nan=False) + "\n")
                    summary = metrics.summary()
                    old = reference["partitions"][part]["diagnostic_untrained_metrics"]
                    objective = totals.summary()
                    if (summary["window_count"] != old["window_count"]
                            or summary["state_std"] != old["state_std"]
                            or summary["visual"]["aggregate"]["count"] != totals.counts["visual"]
                            or summary["state_normalized"]["aggregate"]["count"] != totals.counts["state"]):
                        raise ValueError("metric normalization/coverage differs")
                    parts[part] = {"windows": len(dataset), "batches": summary["update_count"],
                        "original_and_common_support_identical": True,
                        "objective": objective, "persistence_metrics": summary,
                        "comparison_to_random_untrained": {
                            "objective": change(objective["objective"], reference["partitions"][part]["objective"]["objective"]),
                            **{key: compare_stats(summary[key], old[key]) for key in ("visual", "state_normalized", "state_native")}}}
                    print(f"PERSISTENCE_COMPLETE partition={part} windows={len(dataset)}", flush=True)
    for root, mapping in ((feature_root, verified), (reference_root, reference_hashes)):
        for name, digest in mapping.items():
            if sha256_file(contained(root, name)) != digest:
                raise ValueError(f"bound input changed during evaluation: {name}")
    for name, digest in implementations.items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"implementation changed during evaluation: {name}")
    load_protocol(protocol_path, protocol_sha)
    return {"schema": SCHEMA, "status": "passed", "scope": "fixed_two_episode_persistence_diagnostic",
        "protocol_sha256": protocol_sha, "protocol": protocol,
        "feature_report_sha256": protocol["data"]["feature_report_sha256"],
        "random_reference_report_sha256": protocol["baselines"]["random_reference_report_sha256"],
        "split": SPLIT, "seed": SEED, "batch_size": BATCH_SIZE, "device": "cpu", "torch_version": torch.__version__,
        "partitions": parts, "implementation_sha256": implementations, "verified_feature_artifacts": verified,
        "verified_reference_artifacts": reference_hashes, "bound_artifacts_unchanged": True,
        "future_sampling_plan_sha256_verified": sample_hash,
        "training_started": False, "training_ready": False, "optimizer_steps": 0, "backward_calls": 0,
        "model_instantiated": False, "model_forward_calls": 0, "pi05_loaded": False, "policy_actions_generated": 0,
        "future_training_implemented": False, "future_training_authorized": False,
        "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
        "visual_review_required": False, "visual_scope": "numeric only; no pixels or rollout behavior changed",
        "decision": "persistence baseline and diagnostic protocol ready; no trained-model, action-utility or generalization conclusion",
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir()
                          if p.is_file() and p.name not in {"report.json", "status.json", "run.log"}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-pack", type=Path, required=True)
    parser.add_argument("--random-reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    load_protocol(args.protocol, args.protocol_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "pid": os.getpid(), "optimizer_steps": 0})
    with (args.out / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                report = run(args.feature_pack, args.random_reference, args.protocol, args.protocol_sha256, args.out)
                report["runtime_seconds"] = time.monotonic() - started
                write_json(args.out / "report.json", report)
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "completed",
                    "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 0})
                print("NATIVE_PERSISTENCE_EVALUATION_PASSED", report["runtime_seconds"], flush=True)
            except BaseException as error:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "failed",
                    "error_type": type(error).__name__, "error": str(error), "optimizer_steps": 0})
                raise


if __name__ == "__main__":
    main()
