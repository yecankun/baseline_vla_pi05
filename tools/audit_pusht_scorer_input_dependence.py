"""Frozen scorer input interventions on existing TRAIN/VALIDATION only.

No fitting, native environment, ACT inference, hardware or test-set access.
Cross-source input combinations are sensitivity probes, not labeled futures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

if __package__:
    from . import train_pusht_action_effect_contrast as training
else:
    import train_pusht_action_effect_contrast as training

OUT = training.ROOT / "simulation_output/pusht_scorer_input_dependence_v1"
SEEDS = (20260918, 20260919, 20260920)
ARMS = training.pair.ARMS
MODES = {"grid_donor": (0,), "xy_donor": (1,), "state_donor": (0, 1), "action_donor": (2,)}
PERMUTATION = [4, 2, 0, 3, 1]
PLAN = {
    "schema": "pusht_scorer_input_dependence_v1",
    "role": "posthoc_mechanism_diagnostic_not_new_policy_test",
    "splits": ["train", "validation"], "expected_contexts": {"train": 92, "validation": 21},
    "checkpoint_rule": "all_six_frozen_final2000_no_selection_no_ensemble",
    "observations": ["current_RGB_derived_grid", "observable_agent_XY", "candidate_8step_absolute_XY"],
    "fixed_goal_and_normalization": "unchanged_checkpoint_buffers_no_new_fit",
    "donor_rule": "all_other_sources_at_the_same_anchor_within_the_same_split_no_target_selection",
    "interventions": list(MODES),
    "controls": ["candidate_permutation_equivariance", "five_identical_ACT_candidate_chunks"],
    "candidate_permutation": PERMUTATION,
    "aggregation": "mean_donors_per_recipient_then_mean_anchors_per_source_then_mean_sources",
    "training_repeats": "report_each_then_average_three_no_independent_sample_inflation",
    "sensitivity_metrics": ["raw_score_change_MAE", "common_offset_abs", "centered_score_change_RMS",
                            "top1_index_change_rate", "pair_order_change_rate", "shared_shift_energy_fraction"],
    "interaction": "S(i,i)-S(d,i)-S(i,d)+S(d,d)_raw_and_candidate_centered",
    "local_linearity": "symmetric_XY_candidate_first_order_reconstruction_no_fitted_parameters",
    "target_use": "original_unmodified_inputs_only_no_targets_or_quality_scores_for_mismatched_inputs",
    "limits": ["cross_source_combinations_can_be_jointly_OOD", "flip_does_not_imply_correctness",
               "common_offset_can_change_without_useful_ranking", "no_p_values_or_significance_search",
               "constant_goal_dependency_not_identifiable_from_single_goal_data"],
    "test_data_accessed": False, "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "policy_promotion": False,
}


def centered(values):
    return values - values.mean(1, keepdims=True)


def mean_or_none(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def aggregate(values, keys, recipient=None):
    """Equal donor weight inside recipient, equal recipient anchors inside source."""
    ids = np.arange(len(keys)) if recipient is None else np.asarray(recipient)
    unique = np.unique(ids)
    per_context = {name: np.array([np.mean(v[ids == i]) for i in unique]) for name, v in values.items()}
    sources = keys[unique, 0]
    per_source = {str(int(s)): {name: float(v[sources == s].mean()) for name, v in per_context.items()}
                  for s in np.unique(sources)}
    return {"recipient_contexts": len(unique), "source_count": len(per_source),
            "context_mean": {name: float(v.mean()) for name, v in per_context.items()},
            "source_macro": {name: float(np.mean([row[name] for row in per_source.values()])) for name in values},
            "per_source": per_source}


def sensitivity(base, changed, keys, recipient):
    original = base[recipient]
    difference = changed - original
    relative = centered(changed) - centered(original)
    i, j = np.triu_indices(5, 1)

    def ordering(x):
        gap = x[:, i] - x[:, j]
        return np.where(np.abs(gap) <= 1e-7, 0, np.sign(gap))

    values = {"raw_score_change_MAE": np.abs(difference).mean(1),
              "common_offset_abs": np.abs(difference.mean(1)),
              "centered_score_change_RMS": np.sqrt(np.square(relative).mean(1)),
              "top1_index_change_rate": (changed.argmax(1) != original.argmax(1)).astype(float),
              "pair_order_change_rate": (ordering(changed) != ordering(original)).mean(1),
              "raw_change_MSE": np.square(difference).mean(1),
              "centered_change_MSE": np.square(relative).mean(1)}
    result = aggregate(values, keys, recipient)
    macro = result["source_macro"]
    macro["shared_shift_energy_fraction"] = (1 - macro["centered_change_MSE"] / macro["raw_change_MSE"]
                                               if macro["raw_change_MSE"] > 0 else None)
    result["candidate_selection_quality_evaluated"] = False
    return result


def first_order_residual(values):
    # Candidate order is ACT, +X, -X, +Y, -Y; each ramp has identical magnitude.
    gx, gy = (values[:, 1] - values[:, 2]) / 2, (values[:, 3] - values[:, 4]) / 2
    approximate = values[:, :1] + np.stack((np.zeros_like(gx), gx, -gx, gy, -gy), axis=1)
    return np.sqrt(np.square(values - approximate).mean(1))


def local_linearity(scores, target, keys):
    top = np.sort(scores, axis=1)
    values = {"predicted_span": np.ptp(scores, axis=1), "actual_span": np.ptp(target, axis=1),
              "predicted_centered_RMS": np.sqrt(np.square(centered(scores)).mean(1)),
              "actual_centered_RMS": np.sqrt(np.square(centered(target)).mean(1)),
              "predicted_first_order_residual_RMS": first_order_residual(scores),
              "actual_first_order_residual_RMS": first_order_residual(target),
              "top1_top2_margin": top[:, -1] - top[:, -2]}
    informative = np.ptp(target, axis=1) > 1e-6
    return {"all": aggregate(values, keys),
            "informative_only_descriptive": aggregate({k: v[informative] for k, v in values.items()}, keys[informative]),
            "selected_histogram": np.bincount(scores.argmax(1), minlength=5).tolist()}


def load_packs(frozen):
    packs, keys, mappings = {}, {}, {}
    with np.load(training.TRAIN_ROOT / "data_snapshot.npz", allow_pickle=False) as snapshot:
        for split in PLAN["splits"]:
            pack = training.pair.PairPack(training.DATA_ROOT / split / "pack")
            key = np.array([(r["source_seed"], r["anchor_step"]) for r in pack.contexts], dtype=np.int64)
            low, high = frozen["contract"]["protocol"]["future_source_seed_ranges"][split]
            if (pack.manifest != frozen["contract"]["packs"][split] or pack.manifest["data_role"] != split
                    or len(key) != PLAN["expected_contexts"][split]
                    or not ((key[:, 0] >= low) & (key[:, 0] <= high)).all()):
                raise ValueError("only the unchanged completed training/validation cohort is allowed")
            for name, value in {**pack.observations, "targets": pack.targets, "keys": key}.items():
                if not np.array_equal(value, snapshot[f"{split}_{name}"]):
                    raise ValueError("pack changed from the original training snapshot")
            recipient, donor = np.where((key[:, None, 1] == key[None, :, 1]) & (key[:, None, 0] != key[None, :, 0]))
            if len(np.unique(recipient)) != len(key):
                raise ValueError("each recipient needs another-source same-anchor donor")
            # Necessary for the parameter-free local linearity diagnostic.
            actions = pack.observations["actions"].astype(np.float64)
            midpoint_error = max(float(np.abs(actions[:, a] + actions[:, b] - 2 * actions[:, 0]).max())
                                 for a, b in ((1, 2), (3, 4)))
            if midpoint_error > 1e-4:
                raise ValueError("candidate ramps are no longer symmetric")
            packs[split], keys[split] = pack, key
            mappings[split] = {"recipient": recipient, "donor": donor, "midpoint_error": midpoint_error}
    training.pair.require_disjoint_sources(*packs.values())
    return packs, keys, mappings


@torch.inference_mode()
def execute(out):
    if out.exists():
        raise FileExistsError("preserve the completed/partial audit; do not overwrite or sweep")
    frozen = training.read(training.TRAIN_ROOT / "report.json")
    if frozen["status"] != "completed_fixed_paired_training_and_validation" or frozen["total_optimizer_steps"] != 12000:
        raise ValueError("six completed frozen scorer checkpoints required")
    packs, keys, mappings = load_packs(frozen)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("run this frozen forward audit on the remote4090")
    identities = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = training.TRAIN_ROOT / str(seed) / arm / "final.pt"
            stat = path.stat()
            identities[f"{seed}__{arm}"] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    out.mkdir(parents=True, exist_ok=False)
    training.write(out / "protocol.json", {"plan": PLAN, "checkpoint_identity": identities,
        "training_contract": frozen["contract"], "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(), "batch_rows": 256})
    arrays, results = {}, {split: {} for split in packs}
    counts = {"forward_calls": 0, "context_batches_rows": 0, "candidate_scores": 0}
    started = time.monotonic()

    def predict(model, inputs):
        result = model.predict_effect(*inputs).cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite frozen forward")
        counts["forward_calls"] += 1
        counts["context_batches_rows"] += len(result)
        counts["candidate_scores"] += result.size
        return result

    try:
        for split, pack in packs.items():
            arrays[f"{split}__keys"] = keys[split]
            for name in ("recipient", "donor"):
                arrays[f"{split}__{name}"] = mappings[split][name]
            tensors = pack.inputs(slice(None), device="cuda")
            r, d = mappings[split]["recipient"], mappings[split]["donor"]
            for seed in SEEDS:
                results[split][str(seed)] = {}
                for arm in ARMS:
                    name = f"{seed}__{arm}"
                    model, payload = training.load_final(Path(identities[name]["path"]))
                    if payload["contract"] != frozen["contract"] or payload["training_seed"] != seed or payload["arm"] != arm:
                        raise ValueError("checkpoint identity/contract mismatch")
                    original = predict(model, tensors)
                    if split == "validation":
                        with np.load(training.TRAIN_ROOT / str(seed) / "validation_predictions.npz", allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved["context_keys"], keys[split])
                            np.testing.assert_allclose(original, saved[arm + "_scores"], rtol=0, atol=2e-6)
                    permuted = predict(model, (tensors[0], tensors[1], tensors[2][:, PERMUTATION]))
                    same = predict(model, (tensors[0], tensors[1], tensors[2][:, :1].expand(-1, 5, -1, -1)))
                    equivariance_error = float(np.abs(permuted - original[:, PERMUTATION]).max())
                    identical_spread = float(np.ptp(same, axis=1).max())
                    if max(equivariance_error, identical_spread) > 2e-6:
                        raise ValueError("candidate indexing/identical-action positive control failed")
                    item = {"controls": {"permutation_max_abs_error": equivariance_error,
                                         "identical_actions_max_score_span": identical_spread},
                            "local_linearity": local_linearity(original, pack.targets, keys[split]), "interventions": {}}
                    changed = {}
                    for mode, fields in MODES.items():
                        chunks = []
                        for start in range(0, len(r), 256):
                            own = torch.as_tensor(r[start:start + 256], device="cuda")
                            other = torch.as_tensor(d[start:start + 256], device="cuda")
                            modified = tuple(t[other if index in fields else own] for index, t in enumerate(tensors))
                            chunks.append(predict(model, modified))
                        changed[mode] = np.concatenate(chunks)
                        item["interventions"][mode] = sensitivity(original, changed[mode], keys[split], r)
                    interaction = original[r] - changed["state_donor"] - changed["action_donor"] + original[d]
                    item["state_action_cross_difference"] = aggregate({
                        "raw_RMS": np.sqrt(np.square(interaction).mean(1)),
                        "candidate_centered_RMS": np.sqrt(np.square(centered(interaction)).mean(1))}, keys[split], r)
                    for mode, value in {"original": original, "permuted": permuted, "identical_actions": same, **changed}.items():
                        arrays[f"{split}__{name}__{mode}"] = value
                    results[split][str(seed)][arm] = item
                    del model, payload
                predictions = {arm: arrays[f"{split}__{seed}__{arm}__original"] for arm in ARMS}
                results[split][str(seed)]["original_input_metrics"] = training.selection_metrics(
                    pack.targets, predictions, keys[split][:, 0], frozen["contract"]["protocol"])
                print(json.dumps({"split": split, "training_seed": seed, "status": "frozen_inputs_audited"}), flush=True)
        for identity in identities.values():
            stat = Path(identity["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
                raise ValueError("checkpoint changed during the diagnostic")
        mean_interventions = {split: {arm: {mode: {metric: mean_or_none([
            results[split][str(seed)][arm]["interventions"][mode]["source_macro"][metric] for seed in SEEDS])
            for metric in results[split][str(SEEDS[0])][arm]["interventions"][mode]["source_macro"]}
            for mode in MODES} for arm in ARMS} for split in packs}
        report = {"status": "completed_train_validation_only_input_dependence_audit", "plan": PLAN,
            "checkpoint_identity": identities, "cohorts": {split: {"contexts": len(keys[split]),
                "sources": len(np.unique(keys[split][:, 0])), "cross_source_same_anchor_pairs": len(mappings[split]["recipient"]),
                "candidate_midpoint_max_native_error": mappings[split]["midpoint_error"]} for split in packs},
            "per_split_seed_arm": results, "mean_source_macro_interventions_across_training_repeats": mean_interventions,
            "execution": {**counts, "runtime_seconds_after_loading_packs": time.monotonic() - started,
                "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0, "test_data_accessed": False},
            "validation_reference_scores_reproduced": True, "no_modified_input_quality_metrics": True,
            "interpretation": "numeric_sensitivity_and_original_fit_only_no_causal_validity_or_policy_benefit",
            "policy_promotion": False}
        training.save_arrays(out / "scores_and_donor_mappings.npz", **arrays)
        training.write(out / "report.json", report)
        training.write(out / "status.json", {"status": report["status"], **report["execution"]})
        print(json.dumps({"status": report["status"], "execution": report["execution"], "out": str(out)}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        training.write(out / "status.json", {"status": "failed_partial_audit_preserved", "error": str(exc), **counts})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    return execute(parser.parse_args().out)


if __name__ == "__main__":
    raise SystemExit(main())
