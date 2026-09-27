"""Frozen-checkpoint paired observation sensitivity; clean targets only.

No feature extraction, optimizer, training, or output mutation at import.
Existing native model/loss/dataset/checkpoint helpers remain unchanged.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

import run_pi05_libero_action_study_step0 as base
from pi05_libero_action_ablation import action_ablation_inputs
from pi05_libero_world_model import collate_window_inputs
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_world_model_persistence import persistence_predictions, persistence_scoring_targets

CONDITIONS = ("clean", "low_contrast")
VALIDATION_COUNTS = {1530: 126, 1476: 125, 1458: 130, 1566: 143}


def validate_feature_rows(pack, rows, latents):
    base.require(isinstance(rows, np.ndarray) and rows.dtype == np.int64 and rows.ndim == 1,
                 "row map must be int64 vector")
    expected = np.concatenate([pack.indices[eid] for eid in VALIDATION_COUNTS])
    base.require(np.array_equal(rows, expected) and len(rows) == 524,
                 "exact complete validation-only row order required")
    base.require(latents.dtype == np.float32 and latents.shape == (524, 2, 2048)
                 and np.isfinite(latents).all(), "finite native validation latents required")
    return {int(row): index for index, row in enumerate(rows)}


def condition_inputs(raw, windows, indices, row_lookup, latents, *, arm, condition):
    """Return owned six inputs; never access or rewrite future targets."""
    base.require(condition in CONDITIONS, "unknown image condition")
    base.require(arm in (*base.ARMS, "persistence"), "unknown model arm")
    before = base.fingerprints(raw)
    inputs = action_ablation_inputs(raw, base.ARMS[0] if arm == "persistence" else arm)
    paired_before = base.fingerprints(inputs)
    if condition == "low_contrast":
        positions = []
        for index in indices:
            history = windows[index].history
            base.require(all(row in row_lookup for row in history), "missing historical degraded row")
            positions.append([row_lookup[row] for row in history])
        selected = np.asarray(latents[np.asarray(positions)], dtype=np.float32)
        replacement = torch.from_numpy(selected.copy())
        base.require(replacement.shape == raw["history_visual_latent"].shape
                     and bool(torch.isfinite(replacement).all()), "degraded history shape/finiteness mismatch")
        inputs["history_visual_latent"] = torch.where(inputs["history_visual_valid"][..., None], replacement, 0.0)
    after = base.fingerprints(inputs)
    base.require(all(after[key] == digest for key, digest in paired_before.items()
                     if key != "history_visual_latent"), "nonvisual condition input changed")
    base.require(base.fingerprints(raw) == before, "source inputs mutated")
    return inputs


def evaluate_condition(model, dataset, stats, *, arm, seed, condition, row_lookup,
                       latents, references, trace=None):
    """Score identical clean targets; clean learned predictions must replay exactly.

    references are saved validation trace rows from final200 for learned models,
    or the original step0 rows for parameter-free persistence. No new sampling.
    """
    base.require((model is None) == (arm == "persistence"), "model/reference arm mismatch")
    before = None
    if model is not None:
        base.require(all(not m.training for m in model.modules())
                     and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                     "frozen eval-only checkpoint required")
        before = base.parameter_hash(model), [p._version for p in model.parameters()]
    grouped = {}
    for index, window in enumerate(dataset.windows):
        grouped.setdefault(window.episode_index, []).append(index)
    totals, episodes = base.Totals(stats), {}
    count, started = 0, time.monotonic()
    for episode, indices in grouped.items():
        subtotal = base.Totals(stats)
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[index] for index in selected]
            raw = collate_window_inputs([item["inputs"] for item in items])
            targets = collate_window_targets([item["targets"] for item in items])
            source_fp, target_fp = base.fingerprints(raw), base.fingerprints(targets)
            inputs = condition_inputs(raw, dataset.windows, selected, row_lookup, latents,
                                      arm=arm, condition=condition)
            input_fp = base.fingerprints(inputs)
            ref = references[tuple(selected)]
            if arm == "persistence":
                reference_input = ref["source_input_sha256"]
                reference_target = ref["common_target_sha256"]
                reference_prediction = ref["predictions_sha256"]["persistence"]
            else:
                reference_input = ref["input_sha256"]
                reference_target = ref["target_sha256"]
                reference_prediction = ref["prediction_sha256"]
            base.require(target_fp == reference_target, "clean target bytes changed")
            base.require(all(input_fp[k] == v for k, v in reference_input.items()
                             if condition == "clean" or k != "history_visual_latent"),
                         "clean or nonvisual input bytes drifted")
            with torch.inference_mode():
                common = persistence_scoring_targets(inputs, targets)
                base.require(base.fingerprints(common) == target_fp, "original clean support changed")
                predictions = persistence_predictions(inputs) if model is None else model(**inputs)
                prediction_fp = base.fingerprints(predictions)
                if condition == "clean":
                    base.require(prediction_fp == reference_prediction, "clean final prediction replay differs")
                totals.update(predictions, common)
                subtotal.update(predictions, common)
                base.require(base.fingerprints(common) == target_fp, "scoring mutated clean targets")
            base.require(base.fingerprints(raw) == source_fp and base.fingerprints(targets) == target_fp
                         and base.fingerprints(inputs) == input_fp, "evaluation changed caller bytes")
            count += len(selected)
            if trace is not None:
                trace.write(base.canonical({"seed": seed, "arm": arm, "condition": condition,
                    "episode_index": episode, "window_indices": selected,
                    "source_input_sha256": source_fp, "input_sha256": input_fp,
                    "target_sha256": target_fp, "prediction_sha256": prediction_fp}) + "\n")
                trace.flush()
        episodes[str(episode)] = subtotal.summary()
    base.require(count == len(dataset), "missing evaluation windows")
    if model is not None:
        base.require(before == (base.parameter_hash(model), [p._version for p in model.parameters()])
                     and all(p.grad is None and not p.requires_grad for p in model.parameters())
                     and all(not m.training for m in model.modules()), "evaluation changed frozen model")
    return {"per_episode": episodes, "episode_macro": base.macro_metrics(episodes),
            "micro": totals.summary(), "forward_and_scoring_seconds": time.monotonic()-started}


def summarize(runs, persistence):
    """Descriptive paired arithmetic, never tune or select a model/severity."""
    base.require(set(runs) == set(map(str, base.SEEDS)) and set(persistence) == set(CONDITIONS),
                 "full frozen seed/condition coverage required")
    comparisons, paired = {}, {condition: [] for condition in CONDITIONS}
    means = {condition: {} for condition in CONDITIONS}
    for seed in map(str, base.SEEDS):
        arms = runs[seed]
        base.require(set(arms) == set(base.ARMS)
                     and all(set(row) == set(CONDITIONS) for row in arms.values()), "six frozen checkpoints required")
        comparison = {"low_contrast_minus_clean": {}, "observed_minus_zero": {}, "arm_minus_persistence": {},
                      "per_episode": {}}
        for arm in base.ARMS:
            comparison["low_contrast_minus_clean"][arm] = base.changes(
                arms[arm]["low_contrast"]["episode_macro"], arms[arm]["clean"]["episode_macro"])
        for condition in CONDITIONS:
            values = {arm: arms[arm][condition]["episode_macro"] for arm in base.ARMS}
            comparison["observed_minus_zero"][condition] = base.changes(values[base.ARMS[0]], values[base.ARMS[1]])
            comparison["arm_minus_persistence"][condition] = {
                arm: base.changes(values[arm], persistence[condition]["episode_macro"]) for arm in base.ARMS}
            paired[condition].append(comparison["observed_minus_zero"][condition]["normalized_state_mae"]["absolute"])
        ids = arms[base.ARMS[0]]["clean"]["per_episode"]
        for eid in ids:
            values = {arm: {condition: base.scalar_metrics(arms[arm][condition]["per_episode"][eid])
                           for condition in CONDITIONS} for arm in base.ARMS}
            comparison["per_episode"][eid] = {
                "low_contrast_minus_clean": {arm: base.changes(values[arm]["low_contrast"], values[arm]["clean"])
                                            for arm in base.ARMS},
                "observed_minus_zero": {condition: base.changes(values[base.ARMS[0]][condition], values[base.ARMS[1]][condition])
                                        for condition in CONDITIONS}}
        comparisons[seed] = comparison
    for condition in CONDITIONS:
        for arm in base.ARMS:
            values = [runs[str(seed)][arm][condition]["episode_macro"] for seed in base.SEEDS]
            means[condition][arm] = {key: statistics.mean(row[key] for row in values) for key in values[0]}
        means[condition]["persistence"] = persistence[condition]["episode_macro"]
    retained = all(value < 0 for value in paired["low_contrast"]) and all(
        comparisons[str(seed)]["arm_minus_persistence"]["low_contrast"][base.ARMS[0]]["normalized_state_mae"]["absolute"] < 0
        for seed in base.SEEDS)
    return {"comparisons": comparisons, "seed_mean_episode_macro": means,
            "mean_low_contrast_minus_clean": {arm: base.changes(means["low_contrast"][arm], means["clean"][arm])
                                             for arm in (*base.ARMS, "persistence")},
            "paired_state_macro": {condition: {"by_seed": dict(zip(map(str, base.SEEDS), values)),
                "mean": statistics.mean(values), "descriptive_population_std": statistics.pstdev(values)}
                                   for condition, values in paired.items()},
            "decision": "degraded_observed_action_predictive_association_retained" if retained
                        else "degraded_action_association_mixed_or_not_retained",
            "robustness_certified": False}
