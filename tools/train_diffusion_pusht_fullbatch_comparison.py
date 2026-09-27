"""Frozen full-cohort optimizer/loss comparison, then one development evaluation.

All twelve fixed-budget finals precede validation loading. Preserve the existing
source split, inputs, targets, goal and normalization. No test or environment run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal

import numpy as np
import torch

if __package__:
    from . import probe_diffusion_pusht_scorer_fullbatch as optimization
    from . import diagnose_diffusion_pusht_scorer_fit as diagnostic
else:
    import probe_diffusion_pusht_scorer_fullbatch as optimization
    import diagnose_diffusion_pusht_scorer_fit as diagnostic

tiny, shared, pair = optimization.tiny, optimization.shared, optimization.pair
SCHEMA = "diffusion_pusht_fullbatch_comparison_v1"
OUT = tiny.ORIGINAL_ROOT.parent / SCHEMA
SEEDS = (20260918, 20260919, 20260920)


def keys(pack):
    return np.asarray([(r["source_seed"], r["anchor_step"]) for r in pack.contexts], dtype=np.int64)


def verify_pack(pack, split, original_root, protocol):
    low, high = protocol["future_source_seed_ranges"][split]
    expected = {"train": (51, 30), "validation": (15, 8)}[split]
    if (pack.manifest["data_role"] != split
            or (len(pack.contexts), len(np.unique(keys(pack)[:, 0]))) != expected
            or any(r["split"] != split or not low <= r["source_seed"] <= high for r in pack.contexts)):
        raise ValueError("requires the unchanged full source-grouped cohort")
    # NPZ is lazy: before all finals exist only train_* arrays are read here.
    with np.load(original_root / "data_snapshot.npz", allow_pickle=False) as saved:
        arrays = {**pack.observations, "targets": pack.targets, "keys": keys(pack)}
        if any(not np.array_equal(saved[split + "_" + name], value) for name, value in arrays.items()):
            raise ValueError(f"{split} differs from the original frozen inputs, targets or order")


class FullTrainingData:
    def __init__(self, root, original_root):
        self.root, self.original_root = root, original_root
        self.protocol = shared.read(tiny.PROTOCOL)
        self.norm = shared.read(root / "normalization.json")
        self.train = pair.PairPack(root / "train/pack", for_training=True)
        verify_pack(self.train, "train", original_root, self.protocol)
        if (self.norm["validation_used"]
                or self.norm["target"] != pair.training_target_stats(self.train.targets)
                or self.protocol["paired_training_seeds"] != list(SEEDS)):
            raise ValueError("original training-only normalization or fixed seeds changed")
        self.initialize(SEEDS[0])

    def initialize(self, seed):
        original = torch.load(self.original_root / str(seed) / "initial_state.pt", map_location="cpu", weights_only=True)
        torch.manual_seed(seed)
        self.initial = pair.DirectActionScorer(self.norm["input"], self.norm["target"], original["goal"])
        if (set(original) != set(self.initial.state_dict())
                or any(not torch.equal(value, original[name]) for name, value in self.initial.state_dict().items())):
            raise ValueError("paired seed initialization or frozen buffers differ from the original experiment")


def load_success(pack, split, original_root):
    with np.load(pack.root / "evaluation_targets.npz", allow_pickle=False) as saved:
        success = saved["success"].copy()
    if success.dtype != bool or success.shape != pack.targets.shape:
        raise ValueError("unaligned success sidecar")
    with np.load(original_root / "data_snapshot.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved[split + "_success"], success):
            raise ValueError("success labels differ from the original frozen sidecar")
    return success


def evaluate(data, out, contract, trained):
    # Called only after every optimizer/loss/seed completed its fixed budget.
    validation = pair.PairPack(data.root / "validation/pack")
    verify_pack(validation, "validation", data.original_root, data.protocol)
    pair.require_disjoint_sources(data.train, validation)
    packs = {"train": data.train, "validation": validation}
    success = {split: load_success(pack, split, data.original_root) for split, pack in packs.items()}
    inputs = {split: pack.inputs(slice(None), device="cuda") for split, pack in packs.items()}
    norm = data.norm["target"]
    results = {}
    for seed in SEEDS:
        seed_key = str(seed)
        results[seed_key] = {}
        arrays = {}
        for method in optimization.OPTIMIZERS:
            predictions = {split: {} for split in packs}
            for arm in pair.ARMS:
                leaf = out / seed_key / (method + "__" + arm)
                saved = torch.load(leaf / "final.pt", map_location="cpu", weights_only=False)
                if (saved["schema"] != SCHEMA or saved["kind"] != "fixed_budget_final"
                        or saved["contract"] != {**contract, "training_seed": seed}
                        or saved["gradient_evaluations"] != trained[seed_key][method][arm]["gradient_evaluations"]):
                    raise ValueError("requires the matching fixed-budget final, not a selected checkpoint")
                model = deepcopy(data.initial).cuda().eval().requires_grad_(False)
                model.load_state_dict(saved["model"], strict=True)
                with torch.inference_mode():
                    for split in packs:
                        values = model.predict_effect(*inputs[split]).cpu().numpy().astype(np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError("nonfinite final prediction; do not substitute another model")
                        predictions[split][arm] = values
                        arrays[split + "__" + method + "__" + arm] = values
                with np.load(leaf / "predictions.npz", allow_pickle=False) as prior:
                    if not np.array_equal(prior["scores"], predictions["train"][arm]):
                        raise ValueError("saved final and in-memory final training predictions differ")
                del model, saved
            results[seed_key][method] = {}
            for split, pack in packs.items():
                scores, sources = predictions[split], keys(pack)[:, 0]
                item = {
                    "coverage": shared.selection_metrics(pack.targets, scores, sources, data.protocol, fixed_candidates=True),
                    "success": diagnostic.training.success_metrics(success[split], scores, sources),
                    "regression_fit": {arm: diagnostic.regression_fit(pack.targets, scores[arm], sources,
                                        norm["mean"], norm["std"]) for arm in pair.ARMS},
                }
                item["compact"] = {arm: {**diagnostic.compact_arm(item, arm),
                    "mse": item["regression_fit"][arm]["source_macro"]["mse"]} for arm in pair.ARMS}
                results[seed_key][method][split] = item
        shared.save_arrays(out / f"predictions_{seed}.npz", **arrays)
    shared.save_arrays(out / "evaluation_targets.npz", **{
        split + "__" + name: value for split, pack in packs.items()
        for name, value in {"keys": keys(pack), "coverage": pack.targets, "success": success[split]}.items()})
    summary, deltas, baselines = {}, {}, {}
    for split in packs:
        summary[split], deltas[split] = {}, {}
        for method in optimization.OPTIMIZERS:
            summary[split][method] = {}
            for arm in pair.ARMS:
                rows = [results[str(seed)][method][split]["compact"][arm] for seed in SEEDS]
                summary[split][method][arm] = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
            deltas[split][method] = {
                metric: {str(seed): results[str(seed)][method][split]["compact"][pair.ARMS[1]][metric]
                    - results[str(seed)][method][split]["compact"][pair.ARMS[0]][metric] for seed in SEEDS}
                for metric in ("success", "coverage", "pair_accuracy", "mse")}
        first = results[str(SEEDS[0])][next(iter(optimization.OPTIMIZERS))][split]
        baselines[split] = {
            name: {**first["success"]["methods"][name]["source_macro"],
                   "coverage": first["coverage"]["methods"][name]["source_seed_macro"]["coverage"]}
            for name in ("DP_reference", "uniform_expectation", "oracle_offline", *[f"fixed_candidate_{k}" for k in range(5)])}
        # Coverage oracle and success oracle have distinct offline label objectives.
        baselines[split]["oracle_offline"]["note"] = "success_and_coverage_oracles_are_separately_maximized_not_one_policy"
    return results, summary, deltas, baselines


def run(args, stop):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.__version__.startswith("2.10."):
        raise ValueError("line-search evaluation cap was verified against torch2.10")
    data = FullTrainingData(args.data_root, args.original_root)
    previous = shared.read(optimization.OUT / "run.json")["contract"]
    if (previous["optimizer_configs"] != optimization.OPTIMIZERS
            or previous["gradient_evaluation_cap_per_model"] != optimization.BUDGET):
        raise ValueError("do not tune the configurations after the tiny-fit experiment")
    contract = {
        "schema": SCHEMA, "optimizer_configs": optimization.OPTIMIZERS,
        "gradient_evaluation_cap_per_model": optimization.BUDGET,
        "paired_training_seeds": list(SEEDS), "normalization": data.norm,
        "original_protocol": data.protocol, "train_manifest": data.train.manifest,
        "fullbatch_contexts": 51, "training_sources": 30, "candidates_per_context": 5,
        "validation_contexts": 15, "validation_sources": 8,
        "initialization": "original_initial_state_for_each_paired_seed_no_warm_start",
        "torch_version": torch.__version__, "precision": "FP32_no_TF32_no_AMP",
        "training": "all51_contexts_each_gradient_call_same_data_losses_no_input_or_architecture_change",
        "checkpoint_selection": "all12_last_completed_updates_at_fixed_budget_no_best_seed_step_or_ensemble",
        "validation_schedule": "load_original_validation_arrays_only_after_all12_training_finals_exist",
        "primary": "validation_source_macro_success_effect_minus_pointwise_within_each_optimizer",
        "secondary": "all_train_validation_fit_and_selection_metrics_for_every_seed_optimizer_and_loss",
        "aggregation": "mean_contexts_within_source_then_mean_sources_then_mean_training_seeds",
        "evaluation_scope": "conditional_saved_candidate_continuations_not_repeated_closed_loop_or_fresh_test",
        "lbfgs_confounds": previous["lbfgs_confounds"],
        "budget_caveat": "equal_gradient_evaluation_cap_not_equal_FLOPs_wall_time_or_parameter_updates",
        "success_sidecars": "read_only_after_training_for_metrics_never_forward_or_loss",
        "original_root": str(args.original_root), "test_read": False,
        "environment_steps": 0, "DP_inference_calls": 0, "hardware_actions": 0,
        "deployment_allowed": False,
    }
    if args.resume:
        if shared.read(args.out / "run.json")["contract"] != contract:
            raise ValueError("resume contract changed")
        if (args.out / "report.json").exists():
            print("Full-cohort comparison already complete; no work repeated.", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        shared.write(args.out / "run.json", {"schema": SCHEMA, "contract": contract})
        shared.save_arrays(args.out / "training_snapshot.npz", **data.train.observations,
                           targets=data.train.targets, keys=keys(data.train), goal=data.initial.goal.numpy())
    trained = {}
    for seed in SEEDS:
        data.initialize(seed)
        seed_key = str(seed)
        (args.out / seed_key).mkdir(exist_ok=True)
        trained[seed_key] = {}
        for method in optimization.OPTIMIZERS:
            trained[seed_key][method] = {}
            for arm in pair.ARMS:
                shared.write(args.out / "status.json", {"status": "training", "seed": seed, "method": method, "arm": arm})
                result = optimization.train_one(data, args.out / seed_key / (method + "__" + arm),
                                               {**contract, "training_seed": seed}, method, arm, stop)
                if result is None or stop["requested"]:
                    shared.write(args.out / "status.json", {"status": "interrupted", "seed": seed, "method": method, "arm": arm})
                    return 130
                trained[seed_key][method][arm] = result
    flat = [r for methods in trained.values() for arms in methods.values() for r in arms.values()]
    if len(flat) != 12 or any(r["status"] != "completed_fixed_budget" for r in flat):
        raise ValueError("all twelve fixed-budget models must finish before validation")
    shared.write(args.out / "training_completed.json", {"models": 12, "training": trained,
        "gradient_evaluations_total": sum(r["gradient_evaluations"] for r in flat), "validation_arrays_loaded": False})
    shared.write(args.out / "status.json", {"status": "evaluating_all_fixed_finals", "completed_models": 12})
    results, summary, deltas, baselines = evaluate(data, args.out, contract, trained)
    report = {"schema": SCHEMA, "status": "completed_full_cohort_fullbatch_comparison",
        "contract": contract, "training": trained, "results": results, "summary": summary,
        "effect_minus_pointwise_by_seed": deltas, "baselines": baselines,
        "gradient_evaluations_total": sum(r["gradient_evaluations"] for r in flat),
        "optimization_seconds_total": sum(r["optimization_seconds"] for r in flat),
        "train_final_predictions_bitwise_verified": True, "validation_after_all12_finals": True,
        "fresh_test": False, "checkpoint_or_seed_selection": False, "promotion": False}
    shared.write(args.out / "report.json", report)
    shared.write(args.out / "status.json", {"status": report["status"], "completed_models": 12,
        "gradient_evaluations_total": report["gradient_evaluations_total"]})
    print(json.dumps({"status": report["status"], "summary": summary, "baselines": baselines,
                      "optimization_seconds_total": report["optimization_seconds_total"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--data-root", type=Path, default=tiny.DATA_ROOT)
    parser.add_argument("--original-root", type=Path, default=tiny.ORIGINAL_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    return run(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
