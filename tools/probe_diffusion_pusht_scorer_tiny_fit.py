"""Training-only memorization probe; never load validation or success sidecars.

Reuse the unchanged paired trainer, optimizer, loss, normalization and initial
weights. The four contexts are chosen by source/anchor order, not their labels.
Probe checkpoints use a separate schema and are not deployment candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import numpy as np
import torch

if __package__:
    from . import train_pusht_action_effect_contrast as shared
else:
    import train_pusht_action_effect_contrast as shared

pair = shared.pair
SCHEMA = "diffusion_pusht_scorer_tiny_fit_v1"
SEED = 20260918
CONTEXT_COUNT = 4
DATA_ROOT = shared.ROOT / "simulation_output/diffusion_pusht_action_effect_pairs_v1"
ORIGINAL_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1")
OUT = ORIGINAL_ROOT.parent / SCHEMA
PROTOCOL = shared.ROOT / "docs/diffusion-pusht-action-effect-protocol-v1.json"


class TrainingSubset:
    def __init__(self, pack, rows):
        self.pack, self.rows = pack, np.asarray(rows, dtype=np.int64)
        self.contexts = [pack.contexts[i] for i in rows]
        self.targets = pack.targets[self.rows].copy()

    def inputs(self, rows, *, device="cpu"):
        return self.pack.inputs(self.rows[rows], device=device)


class TinyData:
    def __init__(self, root, original_root):
        protocol = shared.read(PROTOCOL)
        norm = shared.read(root / "normalization.json")
        pack = pair.PairPack(root / "train/pack", for_training=True)
        low, high = protocol["future_source_seed_ranges"]["train"]
        if (norm["validation_used"]
                or norm["target"] != pair.training_target_stats(pack.targets)
                or any(r["split"] != "train" or not low <= r["source_seed"] <= high
                       for r in pack.contexts)):
            raise ValueError("requires the unchanged training-only pack and full-training normalization")
        sources = sorted({r["source_seed"] for r in pack.contexts})[:CONTEXT_COUNT]
        rows = [min((i for i, r in enumerate(pack.contexts) if r["source_seed"] == source),
                    key=lambda i: pack.contexts[i]["anchor_step"]) for source in sources]
        if len(rows) != CONTEXT_COUNT:
            raise ValueError("need four distinct available training sources; do not replace by label quality")
        self.train = TrainingSubset(pack, rows)
        self.keys = np.asarray([(r["source_seed"], r["anchor_step"])
                                for r in self.train.contexts], dtype=np.int64)
        initial_path = original_root / str(SEED) / "initial_state.pt"
        original = torch.load(initial_path, map_location="cpu", weights_only=True)
        self.goal = original["goal"].numpy().copy()
        torch.manual_seed(SEED)
        self.initial = pair.DirectActionScorer(norm["input"], norm["target"], self.goal)
        if (set(original) != set(self.initial.state_dict())
                or any(not torch.equal(v, original[k]) for k, v in self.initial.state_dict().items())):
            raise ValueError("initial weights, goal or normalization differ from the original first paired seed")
        self.contract = {
            "training_schema": SCHEMA,
            "protocol": {k: protocol[k] for k in (
                "reference_name", "optimizer", "pair_weight", "coverage_tie_epsilon",
                "prediction_tie_epsilon", "target", "current_inputs", "not_model_inputs")},
            "normalization": norm,
            "selection_rule": "lowest_four_available_train_source_ids_earliest_anchor_each_no_label_filter",
            "selected_train_rows": rows,
            "selected_keys": self.keys.tolist(),
            "full_training_contexts": len(pack.contexts),
            "training_seed": SEED,
            "original_initial_state": str(initial_path),
            "initial_state_bitwise_equal_to_original": True,
            "sampling": "unchanged_uniform_with_replacement_batch16_all5_candidates_shared_schedule",
            "checkpoint_selection": "fixed_final2000_each_arm_no_best_step_seed_or_subset_selection",
            "evaluation": "initial_and_final_on_the_same_four_training_contexts_only",
            "validation_read": False, "test_read": False, "success_sidecars_read": False,
            "normalization_refit_on_subset": False,
            "environment_steps": 0, "DP_inference_calls": 0, "hardware_actions": 0,
            "deployment_allowed": False,
        }

    def snapshot(self):
        return {**{name: self.train.pack.observations[name][self.train.rows]
                   for name in pair.INPUT_FIELDS},
                "targets": self.train.targets, "keys": self.keys, "goal": self.goal}


def fit_metrics(target, scores, sources, protocol):
    selection = shared.selection_metrics(target, scores, sources, protocol)
    centered_target = target - target.mean(1, keepdims=True)
    flat_mse = float(np.square(centered_target).mean())
    result = {}
    for arm in pair.ARMS:
        error = scores[arm] - target
        centered_error = error - error.mean(1, keepdims=True)
        result[arm] = {
            "coverage_mae": float(np.abs(error).mean()),
            "coverage_mse": float(np.square(error).mean()),
            "within_context_error_mse": float(np.square(centered_error).mean()),
            "within_context_mse_over_flat_prediction": (
                float(np.square(centered_error).mean()) / flat_mse if flat_mse > 0 else None),
            "pair_accuracy_source_macro": selection["pair_agreement"][arm]["source_seed_macro_accuracy"],
            "pair_counts": selection["pair_agreement"][arm]["pooled_pairs"],
            "best_candidate_top1": selection["methods"][arm]["informative_top1"]["source_seed_macro"],
            "mean_coverage_regret": selection["methods"][arm]["source_seed_macro"]["regret"],
            "selected_indices": scores[arm].argmax(1).tolist(),
            "per_context_mae": np.abs(error).mean(1).tolist(),
            "raw_prediction_range": [float(scores[arm].min()), float(scores[arm].max())],
        }
    return {"summary": result, "selection": selection}


def run(args, stop):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = TinyData(args.data_root, args.original_root)
    if args.out.resolve() in (args.original_root.resolve(), args.data_root.resolve()):
        raise ValueError("probe must use a separate output directory")
    if args.resume:
        if shared.read(args.out / "run.json")["contract"] != data.contract:
            raise ValueError("resume contract differs")
        with np.load(args.out / "selected_training_snapshot.npz", allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k], v) for k, v in data.snapshot().items()):
                raise ValueError("selected training data changed on resume")
        if (args.out / "report.json").exists():
            print("Probe already complete; no optimization or evaluation repeated.", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        shared.write(args.out / "run.json", {"schema": SCHEMA, "contract": data.contract})
        shared.save_arrays(args.out / "selected_training_snapshot.npz", **data.snapshot())
    trained = shared.train_seed(data, args.out, SEED, stop)
    if trained is None:
        return 130
    inputs = data.train.inputs(slice(None), device="cuda")
    with torch.inference_mode():
        initial_scores = data.initial.cuda().eval().predict_effect(*inputs).cpu().numpy().astype(np.float64)
        scores = {}
        for arm in pair.ARMS:
            model, payload = shared.load_final(args.out / str(SEED) / arm / "final.pt", expected_schema=SCHEMA)
            if payload["contract"] != data.contract:
                raise ValueError("final checkpoint contract differs")
            scores[arm] = model.predict_effect(*inputs).cpu().numpy().astype(np.float64)
            del model
    protocol = data.contract["protocol"]
    targets, sources = data.train.targets, data.keys[:, 0]
    initial = fit_metrics(targets, dict.fromkeys(pair.ARMS, initial_scores), sources, protocol)
    final = fit_metrics(targets, scores, sources, protocol)
    report = {
        "schema": SCHEMA, "status": "completed_training_only_tiny_fit_probe",
        "contract": data.contract, "optimizer_steps_total": 4000,
        "training": trained, "initial": initial, "final": final,
        "selected_contexts": data.train.contexts, "targets": targets.tolist(),
        "predictions": {arm: values.tolist() for arm, values in scores.items()},
        "target_diagnostics": {
            "informative_contexts": final["selection"]["informative_contexts"],
            "range_per_context": np.ptp(targets, axis=1).tolist(),
            "uniform_best_candidate_top1": final["selection"]["methods"]["uniform_expectation"]["informative_top1"]["source_seed_macro"],
        },
        "scope": "in_sample_memorization_only_not_generalization_policy_success_or_new_architecture_evidence",
        "visual_status": "not_applicable_numeric_training_diagnostic",
    }
    shared.save_arrays(args.out / "predictions.npz", initial=initial_scores, targets=targets,
                       keys=data.keys, **scores)
    shared.write(args.out / "report.json", report)
    shared.write(args.out / "status.json", {"status": report["status"], "optimizer_steps_total": 4000})
    print(json.dumps({"status": report["status"], "initial": initial["summary"],
                      "final": final["summary"], "out": str(args.out)}, ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    return run(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
