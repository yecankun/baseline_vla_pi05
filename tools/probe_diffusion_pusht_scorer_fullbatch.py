"""Fixed full-batch optimization probe on the existing four train contexts.

AdamW and L-BFGS each receive at most 2000 full-batch loss/gradient evaluations
per loss arm. No validation, test, environment, architecture or input changes.
L-BFGS changes the optimization bundle (no AdamW decay or gradient clipping);
this is a memorization diagnostic, not an isolated optimizer or policy claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

if __package__:
    from . import probe_diffusion_pusht_scorer_tiny_fit as tiny
else:
    import probe_diffusion_pusht_scorer_tiny_fit as tiny

shared, pair = tiny.shared, tiny.pair
SCHEMA = "diffusion_pusht_scorer_fullbatch_optimization_v1"
OUT = tiny.ORIGINAL_ROOT.parent / SCHEMA
BUDGET = 2000
OPTIMIZERS = {
    "adamw_fullbatch": {
        "name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001, "grad_clip": 1.0,
    },
    "lbfgs_fullbatch": {
        "name": "LBFGS", "lr": 1.0, "history_size": 20, "max_iter": 1,
        "max_eval_nominal": 24, "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-7, "tolerance_change": 1e-9,
        "weight_decay": 0.0, "grad_clip": None,
    },
}


def make_optimizer(model, method):
    if method == "adamw_fullbatch":
        return torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    return torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=20,
                             max_iter=1, max_eval=24, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-7, tolerance_change=1e-9)


def train_one(data, out, contract, method, arm, stop):
    """Shared fixed-budget optimizer; caller owns the cohort, seed and schema."""
    schema = contract["schema"]
    out.mkdir(exist_ok=True)
    if (out / "training_report.json").exists():
        return shared.read(out / "training_report.json")
    model = deepcopy(data.initial).cuda().train()
    optimizer = make_optimizer(model, method)
    inputs = data.train.inputs(slice(None), device="cuda")
    targets = torch.as_tensor(data.train.targets, device="cuda", dtype=torch.float32)
    targets = (targets - model.target_mean) / model.target_std
    weight = 0.0 if arm == pair.ARMS[0] else 1.0
    calls, evaluations, elapsed = 0, 0, 0.0
    last_path = out / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cuda", weights_only=False)
        if saved["contract"] != contract or saved["method"] != method or saved["arm"] != arm:
            raise ValueError("checkpoint belongs to a different optimization probe")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        calls, evaluations, elapsed = saved["step_calls"], saved["gradient_evaluations"], saved["optimization_seconds"]
        del saved
    resume_from = evaluations

    def save():
        shared.save_torch(last_path, {
            "schema": schema, "contract": contract, "method": method, "arm": arm,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step_calls": calls, "gradient_evaluations": evaluations, "optimization_seconds": elapsed,
        })

    def closure():
        nonlocal evaluations
        if evaluations >= BUDGET:
            raise RuntimeError("gradient budget exceeded; this is a failed probe, not an accepted final")
        optimizer.zero_grad(set_to_none=True)
        loss = pair.effect_difference_loss(model(*inputs), targets, pair_weight=weight)["loss"]
        loss.backward()
        evaluations += 1
        if method == "adamw_fullbatch":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        return loss

    next_log = (evaluations // 200 + 1) * 200
    while evaluations < BUDGET:
        if stop["requested"]:
            save()
            return None
        remaining = BUDGET - evaluations
        if method == "lbfgs_fullbatch":
            if remaining < 2:
                break  # One base evaluation plus at least one line-search trial.
            # Torch 2.10 supplies max_eval - current_evals as max_ls. Its initial
            # trial is additional to that loop, so reserve one evaluation here.
            optimizer.param_groups[0]["max_eval"] = min(24, remaining - 1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if method == "adamw_fullbatch":
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        calls += 1
        if evaluations >= next_log or stop["requested"]:
            with torch.no_grad():
                losses = pair.effect_difference_loss(model(*inputs), targets, pair_weight=weight)
            row = {"method": method, "arm": arm, "step_calls": calls,
                   "gradient_evaluations": evaluations, "resume_from_gradient_evaluations": resume_from,
                   "fullbatch_losses_after_step": {k: float(v) for k, v in losses.items()},
                   "optimization_seconds": elapsed}
            save()
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            shared.write(out.parent / "status.json", {"status": "training", **row})
            print(json.dumps(row), flush=True)
            next_log = (evaluations // 200 + 1) * 200
    if method == "adamw_fullbatch":
        if not shared.optimizer_steps_match(optimizer, evaluations) or calls != evaluations:
            raise ValueError("AdamW step and evaluation counts differ")
        internal = {"parameter_updates": evaluations}
    else:
        state = optimizer.state[next(iter(model.parameters()))]
        if int(state["func_evals"]) != evaluations:
            raise ValueError("L-BFGS closure and optimizer evaluation counts differ")
        internal = {"func_evals": int(state["func_evals"]), "n_iter": int(state["n_iter"])}
    with torch.inference_mode():
        scores = model.eval().predict_effect(*inputs).cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite final; retain failure and do not select an earlier checkpoint")
    buffers = {name: value for name, value in data.initial.named_buffers()}
    if any(not torch.equal(value.cpu(), buffers[name]) for name, value in model.named_buffers()):
        raise ValueError("normalization or fixed goal changed")
    shared.save_torch(out / "final.pt", {
        "schema": schema, "kind": "fixed_budget_final", "contract": contract,
        "method": method, "arm": arm, "step_calls": calls, "gradient_evaluations": evaluations,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "deployment_allowed": False,
    })
    shared.save_arrays(out / "predictions.npz", scores=scores)
    report = {"method": method, "arm": arm, "status": "completed_fixed_budget",
              "gradient_evaluations": evaluations, "step_calls": calls, "optimizer_internal": internal,
              "unused_gradient_budget": BUDGET - evaluations, "optimization_seconds": elapsed,
              "resume_from_gradient_evaluations": resume_from, "parameter_count": shared.PARAMETERS,
              "frozen_buffers_unchanged": True, "finite_final_scores": True}
    shared.write(out / "training_report.json", report)
    return report


def run(args, stop):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # The line-search budget convention was inspected in the installed 2.10 implementation.
    if not torch.__version__.startswith("2.10."):
        raise ValueError("recheck the installed L-BFGS evaluation cap before using another torch version")
    data = tiny.TinyData(args.data_root, args.original_root)
    previous = shared.read(args.previous_root / "report.json")
    if previous["contract"] != data.contract:
        raise ValueError("requires exactly the previous four training contexts and initialization")
    with np.load(args.previous_root / "selected_training_snapshot.npz", allow_pickle=False) as saved:
        if any(not np.array_equal(saved[k], v) for k, v in data.snapshot().items()):
            raise ValueError("previous tiny-fit training snapshot changed")
    contract = {
        "schema": SCHEMA, "previous_tiny_fit_contract": data.contract,
        "previous_root": str(args.previous_root), "optimizer_configs": OPTIMIZERS,
        "gradient_evaluation_cap_per_model": BUDGET, "fullbatch_contexts": 4, "candidates_per_context": 5,
        "training_seed": tiny.SEED, "parameter_count": shared.PARAMETERS,
        "initialization": "same_original_initial_state_not_previous_trained_weights",
        "precision": "FP32_no_AMP_no_TF32", "torch_version": torch.__version__,
        "checkpoint_selection": "last_completed_update_at_fixed_gradient_budget_no_best_selection",
        "lbfgs_budget": "max_iter1_max_eval_min24_remaining_minus1_stop_if_fewer_than2_evals_left",
        "lbfgs_confounds": ["no_decoupled_weight_decay", "no_gradient_clipping", "line_search_and_curvature_updates"],
        "evaluation": "same_four_training_contexts_only_no_success_sidecar",
        "historical_comparison": "previous_minibatch16_with_replacement_is_not_equal_example_exposure",
        "validation_read": False, "test_read": False, "environment_steps": 0,
        "DP_inference_calls": 0, "hardware_actions": 0, "deployment_allowed": False,
    }
    if args.resume:
        if shared.read(args.out / "run.json")["contract"] != contract:
            raise ValueError("resume contract changed")
        if (args.out / "report.json").exists():
            print("Full-batch probe already complete; no work repeated.", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        shared.write(args.out / "run.json", {"schema": SCHEMA, "contract": contract})
    trained, results, arrays = {}, {}, {"targets": data.train.targets, "keys": data.keys}
    for method in OPTIMIZERS:
        trained[method], scores = {}, {}
        for arm in pair.ARMS:
            leaf = args.out / (method + "__" + arm)
            result = train_one(data, leaf, contract, method, arm, stop)
            if result is None:
                shared.write(args.out / "status.json", {"status": "interrupted", "method": method, "arm": arm})
                return 130
            trained[method][arm] = result
            with np.load(leaf / "predictions.npz", allow_pickle=False) as saved:
                scores[arm] = saved["scores"].copy()
            arrays[method + "__" + arm] = scores[arm]
        results[method] = tiny.fit_metrics(data.train.targets, scores, data.keys[:, 0], data.contract["protocol"])
    report = {
        "schema": SCHEMA, "status": "completed_fixed_fullbatch_optimization_probe",
        "contract": contract, "training": trained, "results": results,
        "historical_minibatch": previous["final"]["summary"], "initial": previous["initial"]["summary"],
        "selected_contexts": data.train.contexts, "targets": data.train.targets.tolist(),
        "gradient_evaluations_total": sum(r["gradient_evaluations"] for arms in trained.values() for r in arms.values()),
        "optimization_seconds_total": sum(r["optimization_seconds"] for arms in trained.values() for r in arms.values()),
        "scope": "in_sample_memorization_not_generalization_policy_success_or_architecture_novelty",
    }
    shared.save_arrays(args.out / "predictions.npz", **arrays)
    shared.write(args.out / "report.json", report)
    shared.write(args.out / "status.json", {"status": report["status"], "gradient_evaluations_total": report["gradient_evaluations_total"]})
    print(json.dumps({"status": report["status"], "results": {k: v["summary"] for k, v in results.items()},
                      "optimization_seconds_total": report["optimization_seconds_total"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--data-root", type=Path, default=tiny.DATA_ROOT)
    parser.add_argument("--original-root", type=Path, default=tiny.ORIGINAL_ROOT)
    parser.add_argument("--previous-root", type=Path, default=tiny.OUT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    return run(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
