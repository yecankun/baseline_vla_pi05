"""Frozen DP scorer train/validation fit diagnosis; no optimizer or environment.

Reuse the original source-macro selection metrics. Labels enter reporting only,
after forward(current_grid, observable_agent_xy, candidate_actions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from . import train_diffusion_pusht_action_effect as training
else:
    import train_diffusion_pusht_action_effect as training

shared, pair = training.shared, training.pair
OUT = shared.ROOT / "simulation_output/diffusion_pusht_scorer_fit_diagnostic_v1"


def regression_fit(target, prediction, sources, training_mean, training_std):
    error = prediction - target
    centered_error = error - error.mean(1, keepdims=True)
    centered_target = target - target.mean(1, keepdims=True)
    values = {
        "mae": np.abs(error).mean(1),
        "mse": np.square(error).mean(1),
        "context_mean_error_mse": np.square(error.mean(1)),
        "within_context_error_mse": np.square(centered_error).mean(1),
        "within_context_target_variance": np.square(centered_target).mean(1),
        "training_mean_constant_mse": np.square(training_mean - target).mean(1),
        "prediction_candidate_std": prediction.std(1),
        "target_candidate_std": target.std(1),
    }
    result = {}
    for aggregation in ("context_mean", "source_macro"):
        row = {name: float(value.mean()) if aggregation == "context_mean" else
               float(np.mean([value[sources == s].mean() for s in np.unique(sources)]))
               for name, value in values.items()}
        row.update({
            "rmse": float(np.sqrt(row["mse"])),
            "standardized_mse": row["mse"] / training_std ** 2,
            "mse_over_training_mean_constant": row["mse"] / row["training_mean_constant_mse"],
            "within_context_mse_over_flat_prediction": (
                row["within_context_error_mse"] / row["within_context_target_variance"]),
        })
        result[aggregation] = row
    result["raw_prediction_range"] = [float(prediction.min()), float(prediction.max())]
    return result


def compact_arm(result, arm):
    coverage = result["coverage"]["methods"][arm]
    success = result["success"]["methods"][arm]["source_macro"]
    fit = result["regression_fit"][arm]["source_macro"]
    return {
        "pair_accuracy": result["coverage"]["pair_agreement"][arm]["source_seed_macro_accuracy"],
        "informative_top1": coverage["informative_top1"]["source_seed_macro"],
        "coverage": coverage["source_seed_macro"]["coverage"],
        "coverage_gain_vs_DP": coverage["source_seed_macro"]["gain_vs_DP"],
        "success": success["success"], "success_gain_vs_DP": success["gain_vs_DP"],
        "rescue": success["rescue"], "harm": success["harm"],
        **{key: fit[key] for key in ("mae", "rmse", "mse_over_training_mean_constant",
                                    "within_context_mse_over_flat_prediction")},
    }


@torch.inference_mode()
def run(args):
    started = time.perf_counter()
    original = shared.read(args.training_root / "report.json")
    if original["status"] != "completed_fixed_DP_paired_training_and_validation":
        raise ValueError("requires the completed six fixed finals; do not train from this entrypoint")
    data = training.DPData(args.data_root)
    if data.contract != original["contract"]:
        raise ValueError("data/protocol must match the frozen training run")
    with np.load(args.training_root / "data_snapshot.npz", allow_pickle=False) as saved:
        snapshot = data.snapshot()
        if set(saved.files) != set(snapshot) or any(not np.array_equal(saved[k], v) for k, v in snapshot.items()):
            raise ValueError("frozen training inputs, targets or ordering changed")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = data.contract["protocol"]
    seeds = protocol["paired_training_seeds"]
    norm = data.contract["normalization"]["target"]
    packs = {"train": data.train, "validation": data.validation}
    inputs = {split: pack.inputs(slice(None), device="cuda") for split, pack in packs.items()}
    results, parity = {}, {}
    for seed in seeds:
        predictions = {split: {} for split in packs}
        for arm in pair.ARMS:
            model, payload = training.load_final(args.training_root / str(seed) / arm / "final.pt")
            if payload["contract"] != data.contract or payload["training_seed"] != seed or payload["arm"] != arm:
                raise ValueError("fixed-final checkpoint identity changed")
            for split in packs:
                predictions[split][arm] = model.predict_effect(*inputs[split]).cpu().numpy().astype(np.float64)
                if not np.isfinite(predictions[split][arm]).all():
                    raise ValueError("nonfinite frozen predictions")
            del model, payload
        with np.load(args.training_root / str(seed) / "validation_predictions.npz", allow_pickle=False) as saved:
            for arm in pair.ARMS:
                if not np.array_equal(saved[arm + "_scores"], predictions["validation"][arm]):
                    raise ValueError("frozen validation predictions differ from the original run")
        parity[str(seed)] = "all_validation_scores_bitwise_equal"
        results[str(seed)], arrays = {}, {}
        for split, pack in packs.items():
            keys = np.array([(r["source_seed"], r["anchor_step"]) for r in pack.contexts], dtype=np.int64)
            sources = keys[:, 0]
            # Future targets are joined only here, after prediction; never forward inputs.
            scores = predictions[split]
            result = {
                "coverage": shared.selection_metrics(pack.targets, scores, sources, protocol, fixed_candidates=True),
                "success": training.success_metrics(data.success[split], scores, sources),
                "regression_fit": {arm: regression_fit(pack.targets, scores[arm], sources, norm["mean"], norm["std"])
                                   for arm in pair.ARMS},
            }
            results[str(seed)][split] = result
            arrays[split + "_keys"] = keys
            arrays.update({split + "_" + arm + "_scores": value for arm, value in scores.items()})
        shared.save_arrays(args.out / f"predictions_{seed}.npz", **arrays)
    summary = {}
    for split in packs:
        summary[split] = {}
        for arm in pair.ARMS:
            values = [compact_arm(results[str(seed)][split], arm) for seed in seeds]
            summary[split][arm] = {key: float(np.mean([value[key] for value in values])) for key in values[0]}
    report = {
        "schema": "diffusion_pusht_scorer_fit_diagnostic_v1", "status": "completed_frozen_fit_diagnosis",
        "training_root": str(args.training_root), "data_root": str(args.data_root),
        "training_seeds": seeds, "aggregation": "source_macro_then_mean_training_seeds",
        "cohort": {split: {"contexts": len(pack.contexts),
                           "sources": len({r["source_seed"] for r in pack.contexts}),
                           "target_mean": float(pack.targets.mean()), "target_std": float(pack.targets.std())}
                   for split, pack in packs.items()},
        "mean_across_training_seeds": summary, "per_training_seed": results,
        "validation_readback": parity, "constant_predictor": "fixed_new_training_target_mean",
        "flat_within_context_reference": "equal_candidate_scores_zero_predicted_differences_not_a_fitted_model",
        "interpretation": "posthoc_fit_and_generalization_diagnostic_not_new_test_or_causal_attribution",
        "seconds": time.perf_counter() - started, "scorer_forward_calls": 12,
        "optimizer_steps": 0, "environment_steps": 0, "DP_inference_calls": 0,
        "hardware_actions": 0, "test_sources_executed": 0,
        "weights_modified": False, "original_reports_modified": False,
        "decision": "preserve_all_six_finals_no_selection_or_promotion",
    }
    shared.write(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "summary": summary}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=training.OUT)
    parser.add_argument("--data-root", type=Path, default=training.data_source.OUT)
    parser.add_argument("--out", type=Path, default=OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
