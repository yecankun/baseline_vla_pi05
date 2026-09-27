"""Native LIBERO objective/metric dry-run only; optimization is not implemented.

The fixed two-episode public feature pair is an interface diagnostic. Random
model errors below are not a quality benchmark or permission to train.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from pi05_libero_world_model_adapter import (
    LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization, read_json, sha256_file,
)
from smoke_pi05_libero_world_model import (
    BATCH_SIZE, COUNTS, PLAN_SHA, SEED, SOURCE_SHA, SPLIT, Tee, contained,
    input_fingerprints, parameter_sha, prediction_check, tensor_sha, validate_artifacts, write_json,
)

SCHEMA = "pi05_libero_native_objective_dry_run_v1"
IMPLEMENTATIONS = (
    "train_pi05_libero_world_model.py", "pi05_libero_world_model_objectives.py",
    "pi05_libero_world_model.py", "smoke_pi05_libero_world_model.py",
    "pi05_libero_world_model_adapter.py", "pi05_action_effect_world_model.py",
    "train_pi05_action_effect_world_model.py",
)


def validate_execution(*, dry_run: bool, max_steps: int, device: str) -> None:
    if dry_run is not True or type(max_steps) is not int or max_steps != 0 or device != "cpu":
        raise ValueError("only explicit --dry-run --max-steps 0 --device cpu is implemented")


class ObjectiveTotals:
    """Separate branch sufficient statistics, not a mean of batch means."""

    def __init__(self, visual_weight=1.0, state_weight=0.25):
        for value in (visual_weight, state_weight):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative objective weights required")
        if visual_weight + state_weight <= 0:
            raise ValueError("one positive objective weight required")
        self.weights = {"visual": visual_weight, "state": state_weight}
        self.sums = {"visual": 0.0, "state": 0.0}
        self.counts = {"visual": 0, "state": 0}

    def update(self, details: dict) -> None:
        sums, counts = dict(self.sums), dict(self.counts)
        for key in self.weights:
            value, count = float(details[f"{key}_sum"]), details[f"{key}_count"]
            if (type(count) is not int or count < 0 or not math.isfinite(value)
                    or value < 0 or (count == 0 and value != 0)):
                raise ValueError("invalid branch sufficient statistics")
            sums[key] += value
            counts[key] += count
            if not math.isfinite(sums[key]):
                raise ValueError("objective sum overflow")
        self.sums, self.counts = sums, counts

    def summary(self) -> dict:
        if not any(self.weights[k] > 0 and self.counts[k] > 0 for k in self.weights):
            raise ValueError("objective has no positively weighted valid support")
        branches = {k: {"smooth_l1_sum": self.sums[k], "valid_scalar_count": self.counts[k],
                        "mean": self.sums[k] / self.counts[k] if self.counts[k] else 0.0,
                        "weight": self.weights[k]} for k in self.weights}
        total = sum(v["weight"] * v["mean"] for v in branches.values())
        if not math.isfinite(total):
            raise ValueError("nonfinite aggregated objective")
        return {"aggregation": "per_branch_global_sum_over_valid_scalar_count",
                "objective": total, "branches": branches}


def run(root: Path, report_sha256: str, out: Path, *, dry_run: bool = False,
        max_steps: int = 0, device: str = "cpu") -> dict:
    validate_execution(dry_run=dry_run, max_steps=max_steps, device=device)
    source_report, verified = validate_artifacts(root, report_sha256)
    implementations = {name: sha256_file(Path(__file__).with_name(name)) for name in IMPLEMENTATIONS}
    print("FROZEN_FEATURE_ARTIFACTS_VERIFIED rows=313", flush=True)
    # Heavy imports occur only after explicit execution and source-hash gates.
    import torch
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
    from pi05_libero_world_model_objectives import (
        NativeWorldModelLossConfig, NativeWorldModelMetrics, collate_window_targets,
        native_world_model_loss, objective_contract,
    )

    torch.set_num_threads(1)
    loss_config = NativeWorldModelLossConfig()
    with LiberoFeaturePack(root) as pack:
        split = pack.load_split(root / "split.json")
        if any(split[key] != value for key, value in SPLIT.items()):
            raise ValueError("only the predeclared episode split is in scope")
        if {e: len(rows) for e, rows in pack.indices.items()} != {1400: 140, 1402: 173} or pack.visual_dim != 2048:
            raise ValueError("fixed complete episode counts/visual width differ")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(root / "preparation" / "normalization.json"):
            raise ValueError("normalization differs from training-only recomputation")
        datasets = {part: LiberoWindowDataset(pack, split, partition=part, normalization=stats) for part in COUNTS}
        if {part: len(dataset) for part, dataset in datasets.items()} != COUNTS:
            raise ValueError("full fixed window counts differ")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = LiberoWorldModel(LiberoWorldModelConfig(), task_registry=pack.manifest["task_registry"])
        model = model.to(device).eval().requires_grad_(False)
        before = parameter_sha(model)
        versions = [p._version for p in model.parameters()]
        metadata = model.metadata()
        write_json(out / "model_contract.json", metadata)
        write_json(out / "objective_contract.json", objective_contract())
        parts = {}
        with (out / "dry_run_trace.jsonl").open("x", encoding="utf-8") as trace:
            with torch.inference_mode():
                for part, dataset in datasets.items():
                    totals = ObjectiveTotals(loss_config.visual_weight, loss_config.state_weight)
                    metrics = NativeWorldModelMetrics(stats["state_std"])
                    calls = 0
                    for start in range(0, len(dataset), BATCH_SIZE):
                        items = [dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))]
                        inputs = collate_window_inputs([item["inputs"] for item in items], device=device)
                        targets = collate_window_targets([item["targets"] for item in items], device=device)
                        input_before = input_fingerprints(inputs)
                        target_before = {k: tensor_sha(v) for k, v in targets.items()}
                        predictions = model(**inputs)  # No targets or metadata enter the model.
                        prediction_check(predictions, len(items), 1)
                        loss, details = native_world_model_loss(predictions, targets, loss_config)
                        if loss.requires_grad or not bool(torch.isfinite(loss)):
                            raise ValueError("dry-run loss must be finite and detached")
                        totals.update(details)
                        metrics.update(predictions, targets)
                        if (input_fingerprints(inputs) != input_before
                                or {k: tensor_sha(v) for k, v in targets.items()} != target_before):
                            raise ValueError("model/objective/metrics modified caller inputs or targets")
                        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                            raise ValueError("inference-only context lost")
                        calls += 1
                        record = {
                            "partition": part, "window_start": start, "batch_size": len(items),
                            "metadata": [item["metadata"] for item in items],
                            "inputs": input_before, "targets": target_before,
                            "predictions": {k: tensor_sha(v) for k, v in predictions.items()},
                            "loss": float(loss), "objective_sufficient_statistics": {
                                k: float(v) if k.endswith("_sum") else v for k, v in details.items()
                                if k.endswith("_sum") or k.endswith("_count")},
                            "inference_mode": True, "grad_enabled": False,
                        }
                        trace.write(json.dumps(record, allow_nan=False) + "\n")
                        if start == 0:
                            # One complete batch per partition permits independent numeric readback.
                            with (out / f"first_{part}_audit.npz").open("xb") as stream:
                                np.savez_compressed(stream, **{k: v.detach().cpu().numpy()
                                                              for k, v in {**predictions, **targets}.items()})
                        print(f"ZERO_STEP_OBJECTIVE partition={part} windows={start+len(items)}/{len(dataset)}", flush=True)
                    summary = metrics.summary()
                    objective = totals.summary()
                    if (summary["window_count"] != len(dataset)
                            or summary["visual"]["aggregate"]["count"] != totals.counts["visual"]
                            or summary["state_normalized"]["aggregate"]["count"] != totals.counts["state"]):
                        raise ValueError("objective and metric support counts differ")
                    parts[part] = {"windows": len(dataset), "forward_batches": calls,
                                   "objective": objective, "diagnostic_untrained_metrics": summary,
                                   "caller_inputs_unchanged": True, "caller_targets_unchanged": True}
        after = parameter_sha(model)
        if (before != after or versions != [p._version for p in model.parameters()]
                or any(module.training for module in model.modules())
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise ValueError("frozen random model changed or gradients appeared")
        task_count = len(pack.tasks)
        parameter_count = sum(p.numel() for p in model.parameters())
    for name, digest in verified.items():
        if sha256_file(contained(root, name)) != digest:
            raise ValueError(f"feature input changed during dry-run: {name}")
    for name, digest in implementations.items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"implementation changed during dry-run: {name}")
    return {
        "schema": SCHEMA, "status": "passed", "scope": "untrained_numerical_dry_run_only",
        "feature_pack": str(root.resolve()), "feature_report_sha256": report_sha256,
        "source_report_sha256": SOURCE_SHA, "plan_sha256": PLAN_SHA,
        "manifest_sha256": source_report["manifest_sha256"], "split_sha256": source_report["split_sha256"],
        "split": SPLIT, "seed": SEED, "batch_size": BATCH_SIZE, "device": device,
        "torch_version": torch.__version__, "model_contract": metadata,
        "parameter_count": parameter_count, "parameter_sha256_before": before, "parameter_sha256_after": after,
        "parameter_versions_unchanged": True, "all_parameters_frozen": True,
        "all_parameter_gradients_absent": True, "all_modules_eval": True,
        "partitions": parts, "normalization_state_rows": stats["state_record_count"],
        "normalization_action_rows": stats["action_record_count"],
        "normalization_policy": "train-only; constant valid scale1 is numerical fallback; never infer validation scales",
        "statistical_unit_warning": "overlapping windows from one train and one validation episode, not independent samples",
        "real_task_count": task_count, "real_multitask_conditioning_tested": False,
        "inputs_targets_metadata_separated": True, "source_artifacts_unchanged": True,
        "implementation_sha256": implementations, "verified_feature_artifacts": verified,
        "output_sha256": {path.name: sha256_file(path) for path in out.iterdir()
                          if path.is_file() and path.name not in {"report.json", "status.json", "run.log"}},
        "world_model_randomly_initialized": True, "world_model_weights_saved": False,
        "pi05_loaded": False, "training_started": False, "training_ready": False,
        "optimization_implemented": False, "optimizer_constructed": False,
        "optimizer_steps": 0, "backward_calls": 0, "loss_computed": True,
        "policy_actions_generated": 0, "actions_dispatched": 0, "simulation_steps": 0,
        "candidate_ranking_implemented": False, "world_model_quality_evaluated": False,
        "policy_generalization_evaluated": False, "family_independence_verified": False,
        "checkpoint_training_overlap_unknown": True, "real_system_validated": False,
        "visual_review_required": False, "visual_scope": "numeric/schema-only, no pixel or rollout changes",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-pack", type=Path, required=True)
    parser.add_argument("--feature-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    validate_execution(dry_run=args.dry_run, max_steps=args.max_steps, device=args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "pid": os.getpid(),
                                          "training_started": False, "optimizer_steps": 0})
    with (args.out / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                result = run(args.feature_pack, args.feature_report_sha256, args.out,
                             dry_run=args.dry_run, max_steps=args.max_steps, device=args.device)
                result["runtime_seconds"] = time.monotonic() - started
                write_json(args.out / "report.json", result)
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "completed",
                           "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 0,
                           "training_started": False})
                print("NATIVE_OBJECTIVE_DRY_RUN_PASSED", json.dumps({"windows": COUNTS,
                      "runtime_seconds": result["runtime_seconds"], "optimizer_steps": 0}), flush=True)
            except BaseException as error:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "failed",
                           "error_type": type(error).__name__, "error": str(error),
                           "optimizer_steps": 0, "training_started": False})
                raise


if __name__ == "__main__":
    main()
