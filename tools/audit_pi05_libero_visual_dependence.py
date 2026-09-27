"""Independent stdout-only audit of the completed frozen visual diagnostic.

Reads hashes, cached arrays and saved sufficient statistics. No model import,
forward, embedding, optimizer, checkpoint deserialization or input/output writes.
Learned prediction/residual values were not saved: their drift is checked via
recorded sufficient statistics, not regenerated or inferred from hash strings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import statistics
import time

import numpy as np

import audit_pi05_libero_action_study_step0 as a
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULT = "simulation_output/pi05_libero_visual_dependence_v1"
REPORT_SHA = "1849406c3a434b3dd43ed1d7859c0d401f69022acd4f9f1200a8af3951b25e41"
PLAN = "docs/libero-visual-dependence-plan-v1.json"
PLAN_SHA = "b81ed7f66ee10625d66a9e34e582283690930833c91e0f2c7bdb77905f6d7aea"
TRAIN = "simulation_output/pi05_libero_action_study_training_v1"
TRAIN_SHA = "837cb7b8df45a3b7230e979f9378d91b0177242e98449f6a60ed1608ad2fe5bf"
CONDITIONS = ("clean", "repeat_current", "fixed_train_anchor")
WIDTHS = {"state_output_delta": 24, "visual_output_delta": 12288,
          "visual_residual_delta": 12288, "anchor_skip_delta": 12288,
          "history_input_delta": 16384}


def drift_check(value, windows, updates):
    a.require(set(value) == set(WIDTHS) | {"windows", "updates"}, "drift keys differ")
    a.require(value["windows"] == windows and value["updates"] == updates, "drift coverage differs")
    for key, width in WIDTHS.items():
        row = value[key]
        a.require(set(row) == {"count", "sum_abs", "sum_sq", "max_abs", "exact_changed_count", "mae", "rmse"}, "drift statistics differ")
        a.require(type(row["count"]) is int and row["count"] == windows * width, "coordinate support differs")
        a.require(type(row["exact_changed_count"]) is int and 0 <= row["exact_changed_count"] <= row["count"], "invalid changed-coordinate count")
        a.require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                      for k in ("sum_abs", "sum_sq", "max_abs")), "invalid drift magnitudes")
        a.close_number(row["mae"], row["sum_abs"] / row["count"], "drift MAE")
        a.close_number(row["rmse"], math.sqrt(row["sum_sq"] / row["count"]), "drift RMSE")
        a.require(row["max_abs"] <= row["sum_abs"] + 1e-12, "drift maximum exceeds sum")
        if row["exact_changed_count"] == 0:
            a.require(row["sum_abs"] == row["sum_sq"] == row["max_abs"] == 0, "zero changed count has nonzero drift")


def pooled_drift(actual, parts):
    a.require(actual["windows"] == sum(p["windows"] for p in parts)
              and actual["updates"] == sum(p["updates"] for p in parts), "pooled drift coverage differs")
    for key in WIDTHS:
        for field in ("count", "exact_changed_count"):
            a.require(actual[key][field] == sum(p[key][field] for p in parts), "pooled drift count differs")
        for field in ("sum_abs", "sum_sq"):
            a.close_number(actual[key][field], math.fsum(p[key][field] for p in parts), "pooled drift " + field)
        a.close_number(actual[key]["max_abs"], max(p[key]["max_abs"] for p in parts), "pooled drift maximum")


def raw_drift(actual, clean, changed):
    """Independently recompute cache-derived drift, with promotion before subtraction."""
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    flattened = delta.reshape(delta.shape[0], -1)
    expected = {"count": delta.size, "sum_abs": math.fsum(float(np.abs(row).sum()) for row in flattened),
                "sum_sq": math.fsum(float(np.square(row).sum()) for row in flattened),
                "max_abs": float(np.abs(delta).max()), "exact_changed_count": int(np.count_nonzero(delta))}
    for key, value in expected.items():
        if key in ("count", "exact_changed_count"):
            a.require(actual[key] == value, "cache-derived drift count differs")
        else:
            a.close_number(actual[key], value, "cache-derived drift " + key)


def pooled_score(actual, parts):
    for key in ("visual", "state_normalized", "state_native"):
        a.merged_stats(actual[key], [p[key] for p in parts])
    for key in ("visual_sum", "state_sum", "visual_count", "state_count"):
        a.close_number(actual["masked_objective"][key], math.fsum(p["masked_objective"][key] for p in parts), "pooled score " + key)


def check_scoring_group(run, records, grouped, norm, reference):
    a.require(set(run["per_episode"]) == set(map(str, grouped)), "score episode membership differs")
    for eid, indices in grouped.items():
        batches = [row for row in records if row["episode_index"] == eid]
        score, drift = run["per_episode"][str(eid)], run["drift"]["per_episode"][str(eid)]
        a.check_summary(score, len(indices), math.ceil(len(indices) / 16), norm)
        drift_check(drift, len(indices), len(batches))
        pooled_score(score, [row["score"] for row in batches])
        pooled_drift(drift, [row["drift"] for row in batches])
    a.check_summary(run["micro"], 500, 33, norm)
    drift_check(run["drift"]["micro"], 500, 33)
    pooled_score(run["micro"], list(run["per_episode"].values()))
    pooled_drift(run["drift"]["micro"], list(run["drift"]["per_episode"].values()))
    values = [a.scalars(row) for row in run["per_episode"].values()]
    for key in values[0]:
        a.close_number(run["episode_macro"][key], statistics.mean(v[key] for v in values), "episode-equal score macro")
    for key in WIDTHS:
        for metric in ("mae", "rmse"):
            a.close_number(run["drift"]["episode_macro"][key][metric],
                           statistics.mean(v[key][metric] for v in run["drift"]["per_episode"].values()), "episode-equal drift macro")
    if reference is not None:
        for key in ("per_episode", "episode_macro", "micro"):
            a.require(run[key] == reference[key], "original clean statistics do not replay exactly")


def audit(root, report_sha):
    started = time.monotonic()
    root = Path(root).resolve()
    a.require(report_sha == REPORT_SHA, "only the fixed completed report is supported")
    report = a.read(a.checked(root, RESULT + "/report.json", report_sha))
    a.checked(root, PLAN, PLAN_SHA)
    out = root / RESULT
    a.require(report["plan_sha256"] == PLAN_SHA and report["status"] == "completed_frozen_visual_dependence", "wrong completed protocol")
    a.require(a.read(out / "status.json") == {"status": "completed", "report_sha256": report_sha, "optimizer_steps": 0}
              and not (out / "failure.json").exists(), "completion status differs")
    for name, digest in report["input_sha256"].items():
        a.checked(root, name, digest)
    for name, digest in report["output_sha256"].items():
        a.checked(root, RESULT + "/" + name, digest)
    a.require(len(report["input_sha256"]) == 176, "complete input inventory required")
    train = a.read(a.checked(root, TRAIN + "/report.json", TRAIN_SHA))
    step0 = a.read(a.checked(root, a.RESULT + "/report.json", a.REPORT_SHA))
    learned_refs = {(row["seed"], row["arm"], tuple(row["window_indices"])): row
                    for row in a.lines(root / TRAIN / "evaluation_final.jsonl") if row["partition"] == "validation"}
    persistence_refs = {tuple(row["window_indices"]): row for row in a.lines(root / a.RESULT / "evaluation_trace.jsonl")
                        if row["seed"] == a.SEEDS[0] and row["partition"] == "validation"}
    trace = a.lines(out / "evaluation_trace.jsonl")
    index = {(r["seed"], r["arm"], r["condition"], tuple(r["window_indices"])): r for r in trace}
    a.require(len(trace) == len(index) == 693, "complete unique trace required")
    expected_order, group_count, raw_drift_checks = [], 0, 0
    with LiberoFeaturePack(root / a.PACK) as pack:
        split = pack.load_split(root / a.PACK / "split.json")
        norm = a.read(root / a.PACK / "preparation/normalization.json")
        dataset = LiberoWindowDataset(pack, split, partition="validation", normalization=norm)
        eid = min(split["train_episode_indices"])
        candidate = pack.indices[eid][pack.arrays["frame_index"][pack.indices[eid]] == 0]
        a.require(eid == 1312 and len(candidate) == 1 and int(candidate[0]) == 447
                  and eid not in split["validation_episode_indices"], "anchor must be fixed train frame zero")
        row = int(candidate[0]); anchor = np.array(pack.arrays["visual_latent"][row], copy=True)
        a.require(int(pack.arrays["task_id"][row]) == 9 and bool(pack.arrays["visual_valid"][row].all()), "anchor task/validity differs")
        anchor_record = report["anchor"]
        a.require(anchor_record["episode_index"] == eid and anchor_record["global_cache_row"] == row
                  and anchor_record["frame_index"] == 0 and anchor_record["task_id"] == 9
                  and anchor_record["pair_sha256"] == a.array_hash(anchor)
                  and anchor_record["view_sha256"] == [a.array_hash(v) for v in anchor], "anchor bytes or metadata differ")
        grouped = {}
        for i, window in enumerate(dataset.windows):
            grouped.setdefault(window.episode_index, []).append(i)
        a.require(len(dataset) == 500 and set(grouped) == {1458, 1476, 1530, 1566}, "fixed validation windows differ")
        batches = []
        for ep, ids in grouped.items():
            for start in range(0, len(ids), 16):
                selected = ids[start:start+16]; items = [dataset[i] for i in selected]
                inputs, targets = a.batch_fingerprints(items, "inputs"), a.batch_fingerprints(items, "targets")
                history = np.stack([x["inputs"]["history_visual_latent"] for x in items])
                a.require(all(x["inputs"]["history_visual_valid"].all() and x["inputs"]["history_state_valid"].all() for x in items), "all history masks must remain true")
                zero = a.array_hash(np.zeros_like(np.stack([x["inputs"]["candidate_actions"] for x in items])))
                histories = {"clean": history, "repeat_current": np.broadcast_to(history[:, -1:], history.shape),
                             "fixed_train_anchor": np.broadcast_to(anchor[None, None], history.shape)}
                skips = {c: np.broadcast_to(v[:, -1, None, None], (len(items), 1, 3, 2, 2048)) for c, v in histories.items()}
                batches.append((ep, selected, items, inputs, targets, zero, histories, skips))
        a.require(len(batches) == 33, "fixed batch count differs")
        for seed, arm in [(None, "persistence"), *[(s, arm) for s in a.SEEDS for arm in a.ARMS]]:
            records = {c: [] for c in CONDITIONS}
            for ep, selected, items, inputs, targets, zero, histories, skips in batches:
                for condition in CONDITIONS:
                    key = (seed, arm, condition, tuple(selected)); expected_order.append(key)
                    entry = index[key]; records[condition].append(entry)
                    expected_input = {**inputs, "history_visual_latent": a.array_hash(histories[condition])}
                    if arm == a.ARMS[1]:
                        expected_input["candidate_actions"] = zero
                    a.require(entry["episode_index"] == ep and entry["source_input_sha256"] == inputs
                              and entry["input_sha256"] == expected_input and entry["target_sha256"] == targets, "intervention input/clean target differs")
                    a.require(entry["broadcast_anchor_sha256"] == a.array_hash(skips[condition]), "broadcast skip differs")
                    for value in [entry["actual_residual_sha256"], *entry["prediction_sha256"].values()]:
                        a.digest(value)
                    a.check_summary(entry["score"], len(selected), 1, norm)
                    drift_check(entry["drift"], len(selected), 1)
                    for field, arrays in (("history_input_delta", histories), ("anchor_skip_delta", skips)):
                        raw_drift(entry["drift"][field], arrays["clean"], arrays[condition]); raw_drift_checks += 1
                    if condition == "clean":
                        ref = persistence_refs[tuple(selected)] if seed is None else learned_refs[(seed, arm, tuple(selected))]
                        prediction = ref["predictions_sha256"]["persistence"] if seed is None else ref["prediction_sha256"]
                        a.require(entry["prediction_sha256"] == prediction, "clean prediction SHA differs")
                        a.require(all(entry["drift"][k]["exact_changed_count"] == 0 for k in WIDTHS), "clean drift is nonzero")
                    if condition == "repeat_current":
                        a.require(entry["drift"]["anchor_skip_delta"]["exact_changed_count"] == 0, "repeat-current skip changed")
                    if seed is None:
                        zero_visual = np.zeros_like(skips[condition]); zero_state = np.zeros((len(items), 1, 3, 8), np.float32)
                        a.require(entry["prediction_sha256"] == {"pred_future_visual_latent": a.array_hash(skips[condition]),
                                  "pred_state_delta": a.array_hash(zero_state)} and entry["actual_residual_sha256"] == a.array_hash(zero_visual), "persistence predictions/residual differ")
                        raw_drift(entry["drift"]["visual_output_delta"], skips["clean"], skips[condition]); raw_drift_checks += 1
                        a.require(all(entry["drift"][k]["exact_changed_count"] == 0 for k in ("state_output_delta", "visual_residual_delta")), "persistence state/residual drifted")
                        for name, prediction, target in (("visual", skips[condition][:, 0], np.stack([x["targets"]["future_visual_latent"] for x in items])),
                                                        ("state_normalized", zero_state[:, 0], np.stack([x["targets"]["state_delta"] for x in items]))):
                            delta = prediction.astype(np.float64) - target.astype(np.float64)
                            a.close_number(entry["score"][name]["aggregate"]["absolute_error_sum"], float(np.abs(delta).sum()), "raw persistence error sum")
                            a.close_number(entry["score"][name]["aggregate"]["squared_error_sum"], float(np.square(delta).sum()), "raw persistence squared sum")
            for condition in CONDITIONS:
                seen = [i for entry in records[condition] for i in entry["window_indices"]]
                a.require(Counter(seen) == Counter(range(500)), "each condition must cover every validation window once")
                run = report["persistence"][condition] if seed is None else report["runs"][str(seed)][arm][condition]
                reference = None
                if condition == "clean":
                    reference = step0["runs"][str(a.SEEDS[0])]["validation"]["arms"]["persistence"] if seed is None else train["runs"][str(seed)][arm]["final"]["validation"]
                check_scoring_group(run, records[condition], grouped, norm, reference); group_count += 1
            if seed is not None:
                check = report["checkpoint_checks"][str(seed)][arm]
                a.require(check["parameter_sha256_before"] == check["parameter_sha256_after"] and check["step"] == 200
                          and check["checkpoint_sha256"] == train["runs"][str(seed)][arm]["checkpoint"]["envelope"]["checkpoint_sha256"], "frozen checkpoint hash declaration differs")
    a.require([(r["seed"], r["arm"], r["condition"], tuple(r["window_indices"])) for r in trace] == expected_order, "trace file ordering differs")
    for condition in CONDITIONS:
        for arm in a.ARMS:
            values = [report["runs"][str(s)][arm][condition] for s in a.SEEDS]
            for key in values[0]["episode_macro"]:
                a.close_number(report["seed_mean_episode_macro"][condition][arm][key], statistics.mean(v["episode_macro"][key] for v in values), "seed score mean")
            for key in WIDTHS:
                for metric in ("mae", "rmse"):
                    numbers = [v["drift"]["episode_macro"][key][metric] for v in values]
                    actual = report["seed_summary_output_drift"][condition][arm][key][metric]
                    a.close_number(actual["mean"], statistics.mean(numbers), "seed drift mean")
                    a.close_number(actual["descriptive_population_std"], statistics.pstdev(numbers), "seed drift std")
        a.require(report["seed_mean_episode_macro"][condition]["persistence"] == report["persistence"][condition]["episode_macro"]
                  and report["seed_summary_output_drift"][condition]["persistence"] == report["persistence"][condition]["drift"]["episode_macro"], "persistence summaries differ")
        if condition != "clean":
            for arm in (*a.ARMS, "persistence"):
                a.check_changes(report["mean_intervention_minus_clean"][condition][arm], report["seed_mean_episode_macro"][condition][arm], report["seed_mean_episode_macro"]["clean"][arm])
            for arm in a.ARMS:
                for metric, row in report["paired_error_changes"][condition][arm].items():
                    values = {str(s): report["runs"][str(s)][arm][condition]["episode_macro"][metric] - report["runs"][str(s)][arm]["clean"]["episode_macro"][metric] for s in a.SEEDS}
                    for seed, value in values.items():
                        a.close_number(row["by_seed"][seed], value, "paired error difference")
                    a.close_number(row["mean"], statistics.mean(values.values()), "paired error mean")
                    a.close_number(row["descriptive_population_std"], statistics.pstdev(values.values()), "paired error std")
    a.require(report["optimizer_steps"] == report["new_feature_encodings"] == 0 and report["model_prediction_batches"] == 594
              and report["scored_persistence_prediction_batches"] == 99 and report["persistence_helper_calls_including_support_checks"] == 792, "execution budget differs")
    a.require(report["decision"] == "report_local_visual_sensitivity_without_module_selection"
              and report["robustness_certified"] is False and report["training_started"] is False
              and report["future_targets_remain_clean"] is True, "evidence scope differs")
    for name, digest in report["input_sha256"].items():
        a.checked(root, name, digest)
    for name, digest in report["output_sha256"].items():
        a.checked(root, RESULT + "/" + name, digest)
    a.checked(root, RESULT + "/report.json", report_sha)
    return dict(status="passed_independent_visual_dependence_readback", report_sha256=report_sha,
                audit_code_sha256=sha256_file(Path(__file__)), input_files_rehashed=len(report["input_sha256"]),
                output_files_rehashed=len(report["output_sha256"]), trace_rows_checked=len(trace), scoring_groups_checked=group_count,
                cache_derived_drift_field_checks=raw_drift_checks, anchor_row=447, anchor_episode=1312,
                all_condition_input_target_hashes_reconstructed=True, clean_prediction_and_metric_reference_match=True,
                model_forward_calls=0, embedding_calls=0, optimizer_updates=0, checkpoint_deserialization=False,
                limitations="History/skip/persistence values recomputed from immutable cache; learned output/residual drift checked through saved sufficient statistics and hashes, not regenerated predictions. No causal importance or robustness claim.",
                wall_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(a.canonical(audit(ROOT, args.report_sha256)))


if __name__ == "__main__":
    main()
