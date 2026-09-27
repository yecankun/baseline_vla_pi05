"""Read-only visual-input interventions on all six pinned final200 checkpoints.

No training, new feature extraction, model mutation, or rollout. Clean replay
is checked before interventions for each batch; nothing is published as complete
unless every batch and the original aggregate statistics replay exactly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

import run_pi05_libero_action_study_low_contrast as frozen
import run_pi05_libero_action_study_step0 as base
import run_pi05_libero_action_study_training as previous
import pi05_libero_action_study_training as checkpoint_io
from pi05_libero_visual_dependence import CONDITIONS, DriftTotals, capture_prediction, transform_inputs
from pi05_libero_world_model import collate_window_inputs
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, read_json, sha256_file
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_world_model_persistence import persistence_predictions, persistence_scoring_targets

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-visual-dependence-plan-v1.json"
OUT = "simulation_output/pi05_libero_visual_dependence_v1"
OLD_PLAN_SHA = "0558af2fc68e5cc8c0a6e869e34fda74782246a35120a871db9e86b3327ca307"
ALLOW = dict(frozen_checkpoint_evaluation=True, optimizer_updates=False,
             feature_extraction=False, normalization_refit=False, checkpoint_selection=False,
             policy_rollout=False, project_data_inspection=False)
LAYOUT = dict(history=4, horizon=3, views=2, visual_dim=2048, state_dim=8, action_dim=7,
              batch_size=16, validation_windows=500, checkpoint_step=200)
ANCHOR = dict(selection="minimum_train_episode_unique_frame_zero", episode_index=1312,
              frame_index=0, source="unchanged_clean_feature_cache", fitting=False)
NEW_CODE = {"tools/run_pi05_libero_visual_dependence.py", "tools/pi05_libero_visual_dependence.py"}
DRIFT_KEYS = ("state_output_delta", "visual_output_delta", "visual_residual_delta",
              "anchor_skip_delta", "history_input_delta")


def validate(repo, plan_sha):
    plan = read_json(base.checked(repo, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_visual_dependence_plan_v1"
                 and plan["allowed"] == ALLOW and plan["layout"] == LAYOUT
                 and plan["anchor"] == ANCHOR and plan["conditions"] == list(CONDITIONS)
                 and plan["seeds"] == list(base.SEEDS) and plan["arms"] == list(base.ARMS)
                 and plan["output"] == OUT and plan["previous_plan_sha256"] == OLD_PLAN_SHA
                 and plan["evaluation_order"] == "per_batch_clean_replay_then_two_interventions",
                 "fixed diagnostic scope/authorization differs")
    base.require(set(plan["input_sha256"]) == NEW_CODE, "explicit complete new code inventory required")
    old = frozen.validate(repo, OLD_PLAN_SHA)
    pins = {**old["input_sha256"], **plan["input_sha256"], PLAN: plan_sha}
    frozen.rehash(repo, pins)
    return dict(status="passed_visual_dependence_preflight", plan=plan, plan_sha256=plan_sha,
                input_sha256=pins, optimizer_steps=0)


def select_anchor(pack, split):
    """Actual training first-frame pair, never a fitted mean or future target."""
    train = split["train_episode_indices"]
    val = split["validation_episode_indices"]
    eid = min(train)
    base.require(eid == ANCHOR["episode_index"] and eid not in val, "fixed training anchor episode differs")
    indices = pack.indices[eid]
    zeros = indices[pack.arrays["frame_index"][indices] == 0]
    base.require(len(zeros) == 1, "unique training frame zero required")
    row = int(zeros[0])
    base.require(int(pack.arrays["episode_index"][row]) == eid
                 and bool(pack.arrays["visual_valid"][row].all()), "anchor provenance or masks differ")
    task = int(pack.arrays["task_id"][row])
    base.require(all(pack.episodes[e]["task_id"] == task for e in val), "anchor task differs from validation")
    value = np.array(pack.arrays["visual_latent"][row], copy=True)
    base.require(value.dtype == np.float32 and value.shape == (2, 2048)
                 and bool(np.isfinite(value).all()), "finite native anchor required")
    anchor = torch.from_numpy(value)
    return anchor, dict(**ANCHOR, global_cache_row=row, task_id=task,
                        pair_sha256=base.tensor_hash(anchor),
                        view_sha256=[base.tensor_hash(anchor[i]) for i in range(2)])


def drift_macro(episodes):
    base.require(bool(episodes), "empty drift episodes")
    return {key: {metric: statistics.mean(row[key][metric] for row in episodes.values())
                  for metric in ("mae", "rmse")} for key in DRIFT_KEYS}


def persistence_capture(inputs):
    prediction = persistence_predictions(inputs)
    total = prediction["pred_future_visual_latent"]
    return dict(predictions=prediction, residual=torch.zeros_like(total), anchor=total.clone())


def evaluate_paired(model, dataset, stats, *, arm, seed, anchor, references, trace=None):
    base.require((model is None) == (arm == "persistence"), "model/arm mismatch")
    grouped = {}
    for index, window in enumerate(dataset.windows):
        grouped.setdefault(window.episode_index, []).append(index)
    totals = {c: base.Totals(stats) for c in CONDITIONS}
    drifts = {c: DriftTotals() for c in CONDITIONS}
    episodes, episode_drifts = {c: {} for c in CONDITIONS}, {c: {} for c in CONDITIONS}
    counts = 0
    started = time.monotonic()
    for episode, indices in grouped.items():
        sub = {c: base.Totals(stats) for c in CONDITIONS}
        subdrift = {c: DriftTotals() for c in CONDITIONS}
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[i] for i in selected]
            raw = collate_window_inputs([item["inputs"] for item in items])
            targets = collate_window_targets([item["targets"] for item in items])
            source_fp, target_fp = base.fingerprints(raw), base.fingerprints(targets)
            base.require(tuple(selected) in references, "missing saved clean batch reference")
            ref = references[tuple(selected)]
            ref_input = ref["source_input_sha256"] if model is None else ref["input_sha256"]
            ref_target = ref["common_target_sha256"] if model is None else ref["target_sha256"]
            ref_pred = ref["predictions_sha256"]["persistence"] if model is None else ref["prediction_sha256"]
            base.require(target_fp == ref_target, "clean future target bytes changed")
            clean_inputs, clean_capture = None, None
            for condition in CONDITIONS:
                inputs = transform_inputs(raw, base.ARMS[0] if model is None else arm, condition, anchor)
                input_fp = base.fingerprints(inputs)
                base.require(all(input_fp[k] == v for k, v in ref_input.items()
                                 if condition == "clean" or k != "history_visual_latent"),
                             "clean or nonvisual input bytes changed")
                with torch.no_grad():
                    common = persistence_scoring_targets(inputs, targets)
                    base.require(base.fingerprints(common) == target_fp, "original target support changed")
                    captured = persistence_capture(inputs) if model is None else capture_prediction(model, inputs)
                    predictions = captured["predictions"]
                    pred_fp = base.fingerprints(predictions)
                    if condition == "clean":
                        base.require(pred_fp == ref_pred, "saved clean prediction replay differs")
                        clean_inputs, clean_capture = inputs, captured
                    batch_score, batch_drift = base.Totals(stats), DriftTotals()
                    for accumulator in (totals[condition], sub[condition], batch_score):
                        accumulator.update(predictions, common)
                    for accumulator in (drifts[condition], subdrift[condition], batch_drift):
                        accumulator.update(clean_capture, captured, clean_inputs, inputs)
                    drift = batch_drift.summary()
                    if condition == "clean":
                        base.require(all(drift[k]["exact_changed_count"] == 0 for k in DRIFT_KEYS), "clean drift nonzero")
                    if condition == "repeat_current":
                        base.require(drift["anchor_skip_delta"]["exact_changed_count"] == 0, "repeat-current skip changed")
                        if model is None:
                            base.require(pred_fp == ref_pred, "repeat-current persistence changed")
                    base.require(base.fingerprints(common) == target_fp, "scoring changed clean targets")
                base.require(base.fingerprints(raw) == source_fp and base.fingerprints(targets) == target_fp
                             and base.fingerprints(inputs) == input_fp, "caller input/target bytes mutated")
                if trace is not None:
                    trace.write(base.canonical(dict(seed=seed, arm=arm, condition=condition,
                        episode_index=episode, window_indices=selected, source_input_sha256=source_fp,
                        input_sha256=input_fp, target_sha256=target_fp, prediction_sha256=pred_fp,
                        actual_residual_sha256=base.tensor_hash(captured["residual"]),
                        broadcast_anchor_sha256=base.tensor_hash(captured["anchor"]),
                        score=batch_score.summary(), drift=drift)) + "\n")
            counts += len(selected)
        for c in CONDITIONS:
            episodes[c][str(episode)] = sub[c].summary()
            episode_drifts[c][str(episode)] = subdrift[c].summary()
    base.require(counts == len(dataset), "incomplete validation windows")
    return {c: dict(per_episode=episodes[c], episode_macro=base.macro_metrics(episodes[c]),
                    micro=totals[c].summary(), drift=dict(per_episode=episode_drifts[c],
                    episode_macro=drift_macro(episode_drifts[c]), micro=drifts[c].summary()),
                    paired_evaluation_seconds=time.monotonic()-started) for c in CONDITIONS}


def summarize(runs, persistence):
    base.require(set(runs) == set(map(str, base.SEEDS)) and set(persistence) == set(CONDITIONS), "complete seed/condition coverage required")
    for pair in runs.values():
        base.require(set(pair) == set(base.ARMS) and all(set(v) == set(CONDITIONS) for v in pair.values()), "six complete checkpoints required")
    means, drift_means, changes, paired = {}, {}, {}, {}
    for c in CONDITIONS:
        means[c], drift_means[c] = {}, {}
        for arm in base.ARMS:
            rows = [runs[str(s)][arm][c] for s in base.SEEDS]
            means[c][arm] = {k: statistics.mean(r["episode_macro"][k] for r in rows) for k in rows[0]["episode_macro"]}
            drift_means[c][arm] = {k: {m: dict(mean=statistics.mean(r["drift"]["episode_macro"][k][m] for r in rows),
                descriptive_population_std=statistics.pstdev(r["drift"]["episode_macro"][k][m] for r in rows))
                for m in ("mae", "rmse")} for k in DRIFT_KEYS}
        means[c]["persistence"] = persistence[c]["episode_macro"]
        drift_means[c]["persistence"] = persistence[c]["drift"]["episode_macro"]
    for c in CONDITIONS[1:]:
        changes[c] = {arm: base.changes(means[c][arm], means["clean"][arm]) for arm in (*base.ARMS, "persistence")}
        paired[c] = {}
        for arm in base.ARMS:
            metrics = {}
            for metric in means[c][arm]:
                values = {str(s): runs[str(s)][arm][c]["episode_macro"][metric] - runs[str(s)][arm]["clean"]["episode_macro"][metric] for s in base.SEEDS}
                metrics[metric] = dict(by_seed=values, mean=statistics.mean(values.values()),
                    descriptive_population_std=statistics.pstdev(values.values()))
            paired[c][arm] = metrics
    return dict(seed_mean_episode_macro=means, seed_summary_output_drift=drift_means,
                mean_intervention_minus_clean=changes, paired_error_changes=paired,
                decision="report_local_visual_sensitivity_without_module_selection", robustness_certified=False)


def run(repo, plan_sha, execute):
    base.require(execute is True and not (repo / OUT).exists(), "explicit fresh evaluation required; no retry")
    preflight = validate(repo, plan_sha)
    previous.runtime_settings()
    train = read_json(repo / frozen.TRAIN_REPORT)
    step0 = read_json(repo / base.OUT_PATH / "report.json")
    out = frozen.claim(repo, OUT, preflight)
    started = time.monotonic()
    try:
        final = [json.loads(s) for s in (repo / previous.TRAIN_ROOT / "evaluation_final.jsonl").read_text().splitlines()]
        initial = [json.loads(s) for s in (repo / base.OUT_PATH / "evaluation_trace.jsonl").read_text().splitlines()]
        runs, checks = {}, {}
        with LiberoFeaturePack(repo / base.PACK_PATH) as pack:
            split = pack.load_split(repo / base.PACK_PATH / "split.json")
            stats = read_json(repo / base.PACK_PATH / "preparation/normalization.json")
            dataset = LiberoWindowDataset(pack, split, partition="validation", normalization=stats)
            base.require(len(dataset) == 500 and {e: len(pack.indices[e]) for e in frozen.VALIDATION_COUNTS} == frozen.VALIDATION_COUNTS,
                         "fixed complete validation coverage differs")
            anchor, anchor_record = select_anchor(pack, split)
            with (out / "evaluation_trace.jsonl").open("x", encoding="utf-8") as trace:
                refs = {tuple(r["window_indices"]): r for r in initial if r["seed"] == base.SEEDS[0] and r["partition"] == "validation"}
                base.require(len(refs) == 33, "complete persistence reference required")
                persistence = evaluate_paired(None, dataset, stats, arm="persistence", seed=None,
                                              anchor=anchor, references=refs, trace=trace)
                previous.compare_initial(persistence["clean"], step0["runs"][str(base.SEEDS[0])]["validation"]["arms"]["persistence"])
                for seed in base.SEEDS:
                    pair = base.initialized_pair(seed, pack.manifest["task_registry"])
                    runs[str(seed)], checks[str(seed)] = {}, {}
                    for arm, model in pair.items():
                        record = train["runs"][str(seed)][arm]["checkpoint"]
                        envelope = record["envelope"]
                        binding = deepcopy(envelope["binding"])
                        base.require(binding["seed"] == seed and binding["arm"] == arm
                                     and binding["training_authorization_sha256"] == train["training_authorization_sha256"], "checkpoint binding differs")
                        metadata = deepcopy(envelope["legacy_metadata"])
                        metadata["authorization_sha256"] = train["training_authorization_sha256"]
                        checkpoint_io.load_final_checkpoint(repo / previous.TRAIN_ROOT / record["path"], model,
                            envelope, binding, expected_metadata=metadata)
                        before = base.parameter_hash(model)
                        refs = {tuple(r["window_indices"]): r for r in final if r["seed"] == seed and r["arm"] == arm and r["partition"] == "validation"}
                        base.require(len(refs) == 33, "complete final200 reference required")
                        print(f"VISUAL_DEPENDENCE seed={seed} arm={arm}; conditions=3 optimizer_steps=0", flush=True)
                        result = evaluate_paired(model, dataset, stats, arm=arm, seed=seed, anchor=anchor, references=refs, trace=trace)
                        previous.compare_initial(result["clean"], train["runs"][str(seed)][arm]["final"]["validation"])
                        after = base.parameter_hash(model)
                        base.require(before == after, "checkpoint parameters mutated")
                        runs[str(seed)][arm] = result
                        checks[str(seed)][arm] = dict(checkpoint_sha256=envelope["checkpoint_sha256"],
                            parameter_sha256_before=before, parameter_sha256_after=after, step=200)
            base.require(base.tensor_hash(anchor) == anchor_record["pair_sha256"], "fixed anchor mutated")
        frozen.rehash(repo, preflight["input_sha256"])
        base.require(not torch.cuda.is_initialized(), "CPU diagnostic initialized CUDA")
        model_batches = sum(v["micro"]["update_count"] for r in runs.values() for a in r.values() for v in a.values())
        persistence_batches = sum(v["micro"]["update_count"] for v in persistence.values())
        base.require(model_batches == 594 and persistence_batches == 99, "complete diagnostic batch counts required")
        report = dict(schema="libero_visual_dependence_result_v1", status="completed_frozen_visual_dependence",
            plan_sha256=plan_sha, training_report_sha256=frozen.TRAIN_SHA,
            input_sha256=preflight["input_sha256"], input_files_unchanged=True,
            anchor=anchor_record, runs=runs, persistence=persistence, checkpoint_checks=checks,
            **summarize(runs, persistence), wall_seconds=time.monotonic()-started, runtime=previous.runtime_evidence(),
            optimizer_steps=0, training_started=False, normalization_refitted=False, new_feature_encodings=0,
            clean_reference_prediction_and_statistics_replayed=True, future_targets_remain_clean=True,
            validation_episode_count=4, validation_windows=500, model_prediction_batches=model_batches,
            scored_persistence_prediction_batches=persistence_batches,
            persistence_helper_calls_including_support_checks=model_batches+2*persistence_batches,
            visual_residual_hook_readonly=True, exact_skip_plus_actual_residual_reconstruction=True,
            policy_performance_evaluated=False, family_independence_verified=False, checkpoint_training_overlap_unknown=True,
            training_ready=False, formal_data_allowed=False, real_system_validated=False,
            limitations=preflight["plan"]["limitations"],
            output_sha256={p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
        base.write_json(out / "report.json", report)
        base.write_json(out / "status.json", dict(status="completed", report_sha256=sha256_file(out / "report.json"), optimizer_steps=0))
        print(base.canonical(dict(status=report["status"], wall_seconds=report["wall_seconds"],
            report_sha256=sha256_file(out / "report.json"), optimizer_steps=0)), flush=True)
        return report
    except BaseException as error:
        base.write_json(out / "failure.json", dict(status="failed_preserve_no_retry", error=str(error),
            traceback=traceback.format_exc(), optimizer_steps=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "evaluate"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    base.require(args.execute == (args.stage == "evaluate"), "execution flag required only for evaluation")
    if args.stage == "preflight":
        result = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=result["status"], hashes_checked=len(result["input_sha256"]), optimizer_steps=0)))
    else:
        run(ROOT, args.plan_sha256, args.execute)


if __name__ == "__main__":
    main()
