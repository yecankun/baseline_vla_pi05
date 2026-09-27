"""Fixed human-label retest: rescore old predictions, then refit cached features.

No images, neural models, source-label writes, hyperparameter search or hardware.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_pre_action_response import ARMS, BLOCKS, ALPHA, CLIP, group_metrics
from run_real10_response_baseline import LABELS, SEED, fit_ridge, predict_ridge, folds_for, metrics


SCHEMA = "real10_pre_action_label_revision_retest_v1"
STAGES = ("original", "revised_labels_frozen_predictions", "revised_labels_refit")
FLAGS = {"policy_input_allowed": False, "policy_training_ready": False,
         "formal_data_allowed": False, "deployable": False,
         "PI05_training_executed": False, "world_model_training_executed": False,
         "hardware_executed": False, "neural_model_loaded": False, "images_encoded": 0}


def majority_predictions(y, folds):
    result = np.full(len(y), -1, dtype=np.int64)
    for fold in folds:
        counts = np.bincount(y[fold["train_indices"]], minlength=2)
        result[fold["test_indices"]] = int(counts[1] >= counts[0])
    return result


def difference(a, b):
    return {"absolute_change": b - a, "percentage_points": 100 * (b - a),
            "relative_change_percent": 100 * (b - a) / a if a else None}


def summarize(y, scores, groups, majority):
    predictions = {arm: (scores[arm] >= .5).astype(int) for arm in ARMS}
    results = {}
    for arm in ARMS:
        result = {"pooled": metrics(y, predictions[arm])}
        result.update({key: group_metrics(y, predictions[arm], values) for key, values in groups.items()})
        result["episode_macro_accuracy"] = float(np.mean([v["accuracy"] for v in result["episode"].values()]))
        results[arm] = result
    paired = {key: difference(*[results[arm]["pooled"][key] for arm in ARMS])
              for key in ("accuracy", "balanced_accuracy")}
    episode_delta = {e: results[ARMS[1]]["episode"][e]["accuracy"] - results[ARMS[0]]["episode"][e]["accuracy"]
                     for e in sorted(set(groups["episode"]))}
    correct = [predictions[arm] == y for arm in ARMS]
    paired.update(improved_windows=int(np.sum(~correct[0] & correct[1])),
                  worsened_windows=int(np.sum(correct[0] & ~correct[1])),
                  changed_predictions=int(np.sum(predictions[ARMS[0]] != predictions[ARMS[1]])),
                  episode_accuracy_changes=episode_delta,
                  episode_win_tie_loss=[sum(v > 0 for v in episode_delta.values()),
                                        sum(v == 0 for v in episode_delta.values()),
                                        sum(v < 0 for v in episode_delta.values())],
                  mean_absolute_score_change=float(np.mean(np.abs(scores[ARMS[1]] - scores[ARMS[0]]))),
                  episode_macro_accuracy_change=float(np.mean(list(episode_delta.values()))))
    return {"methods": results, "paired_request_minus_observation": paired,
            "references": {"always_advance": metrics(y, np.ones_like(y)),
                           "train_majority": metrics(y, majority)}}


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve prior artifacts; choose a fresh --out")
    old_protocol = read_json(args.reference / "protocol.json")
    old_report = read_json(args.reference / "report.json")
    old_rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    old_oof = read_jsonl(args.reference / "oof_predictions.jsonl")
    folds = read_json(args.reference / "folds.json")
    frozen = {r["window_id"]: r for r in read_jsonl(args.revisions / "annotation_snapshot.jsonl")}
    current = {r["window_id"]: r for r in read_jsonl(args.pack / "annotations_joint_v2.jsonl")}
    expected = {"schema": "real10_pre_action_request_comparison_v1", "arms": list(ARMS),
                "block_sizes": BLOCKS, "ridge_alpha": ALPHA, "standardized_clip": CLIP,
                "threshold": .5, "seed": SEED, "samples": 45, "episodes": 10}
    if any(old_protocol[k] != v for k, v in expected.items()) or old_report["status"] != "completed":
        raise ValueError("reference is not the fixed completed comparison")
    if len(old_rows) != 45 or len(old_oof) != 45 or len(folds) != 10 or folds != folds_for(old_rows):
        raise ValueError("sample set or original folds changed")
    ids = [r["window_id"] for r in old_rows]
    if len(set(ids)) != 45 or set(ids) != set(frozen) or frozen != current:
        raise ValueError("revision snapshot/current source/sample set mismatch")
    new_rows, label_changes = [], []
    for i, (old, prediction) in enumerate(zip(old_rows, old_oof)):
        annotation = frozen[old["window_id"]]
        if (prediction["sample_index"] != i or prediction["window_id"] != old["window_id"]
                or prediction["ui_index"] != old["ui_index"]
                or prediction["label"] != old["human_annotation"]["joint_motion_response"]
                or annotation["source_episode"] != old["source_episode"]
                or annotation["split_group"] != old["source_episode"]
                or annotation["source_split"] != old["original_source_split"]
                or annotation["status"] != "reviewed"):
            raise ValueError("row identity, provenance or human label mismatch")
        new_rows.append({**old, "human_annotation": annotation})
        if annotation["joint_motion_response"] != prediction["label"]:
            label_changes.append({"ui_index": old["ui_index"], "window_id": old["window_id"],
                                  "old": prediction["label"], "new": annotation["joint_motion_response"]})
    if {c["ui_index"]: (c["old"], c["new"]) for c in label_changes} != {
            62: ("advance", "stationary"), 100: ("stationary", "advance")}:
        raise ValueError("only the two explicitly approved response corrections are in scope")
    if folds != folds_for(new_rows):
        raise ValueError("new labels must not change the folds")
    old_y = np.array([LABELS.index(r["label"]) for r in old_oof])
    y = np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in new_rows])
    with np.load(args.reference / "features.npz", allow_pickle=False) as cache:
        arrays = {arm: cache[arm].copy() for arm in ARMS}
        if not np.array_equal(cache["y"], old_y):
            raise ValueError("cached labels/order disagree with old snapshot")
    if any(x.shape != (45, sum(BLOCKS)) for x in arrays.values()):
        raise ValueError("unexpected cached feature dimensions")
    a, b = (arrays[arm] for arm in ARMS)
    request = np.array([[r["piper_request"] == "hold", r["piper_request"] == "feed"] for r in old_oof])
    if (not np.array_equal(a[:, :-2], b[:, :-2], equal_nan=True) or np.any(a[:, -2:] != 0)
            or not np.array_equal(b[:, -2:], request) or not np.isfinite(b[:, 12:]).all()):
        raise ValueError("cached arms are not the fixed observation/request pair")
    # Treat input cache as immutable; new y is saved separately in the new output.
    for x in arrays.values():
        x.setflags(write=False)
    old_scores = {arm: np.array([r["methods"][arm]["score"] for r in old_oof]) for arm in ARMS}
    for arm in ARMS:
        if any(r["methods"][arm]["prediction"] != LABELS[int(s >= .5)] for r, s in zip(old_oof, old_scores[arm])):
            raise ValueError("old labels/scores disagree at the fixed threshold")
    old_models, replay_errors, test_coverage = {}, [], np.zeros(45, dtype=int)
    for number, fold in enumerate(folds):
        test = np.array(fold["test_indices"])
        test_coverage[test] += 1
        for arm in ARMS:
            with np.load(args.reference / "fold_models" / arm / f"fold_{number:02d}.npz", allow_pickle=False) as data:
                model = {k: data[k] for k in data.files}
            if float(model["alpha"]) != ALPHA or float(model["standardized_clip"]) != CLIP:
                raise ValueError("old model hyperparameters differ")
            replay = predict_ridge(model, arrays[arm][test])
            error = float(np.max(np.abs(replay - old_scores[arm][test])))
            if error > 1e-12 or not np.array_equal(replay >= .5, old_scores[arm][test] >= .5):
                raise ValueError("old model/cache does not reproduce stored predictions")
            replay_errors.append(error)
            old_models[number, arm] = model
    if not np.all(test_coverage == 1):
        raise ValueError("not exactly one OOF prediction per window")
    # Audit group membership is reporting metadata; it never enters fit_ridge.
    groups = {k: [r[k] for r in old_oof] for k in ("episode", "task", "piper_request", "later_action_audit")}
    original = summarize(old_y, old_scores, groups, majority_predictions(old_y, folds))
    if any(original[k] != old_report[k] for k in original):
        raise ValueError("old metrics/paired results/references do not reproduce")
    frozen_result = summarize(y, old_scores, groups, majority_predictions(old_y, folds))
    protocol = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), **FLAGS,
                "reference": str(args.reference), "revision_snapshot": str(args.revisions / "annotation_snapshot.jsonl"),
                "pack": str(args.pack), "samples": 45, "episodes": 10,
                "stages": list(STAGES), "frozen_parameters": expected, "label_changes": label_changes,
                "inputs": "immutable original feature matrices; no image encoding or new inputs",
                "refit": "20 fixed balanced ridge fits; each training fold recalculates class weights from revised labels",
                "statistics": "descriptive paired development diagnostic; no independent-test or significance claim",
                "full_action_supervision": "Elite candidate still missing; no world API",
                "scope": "human-label sensitivity and original request comparison only; no tuning"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "folds.json", folds)
    write_jsonl(args.out / "annotation_snapshot.jsonl", new_rows)
    np.savez_compressed(args.out / "features.npz", **arrays, y=y, old_y=old_y)
    new_scores = {arm: np.full(45, np.nan) for arm in ARMS}
    fold_log, fit_started = [], time.perf_counter()
    for number, fold in enumerate(folds):
        train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        fitted = {}
        for arm in ARMS:
            model = fit_ridge(arrays[arm][train], y[train], alpha=ALPHA,
                              block_sizes=BLOCKS, standardized_clip=CLIP)
            old_model = old_models[number, arm]
            if any(not np.array_equal(model[k], old_model[k]) for k in ("mean", "std", "feature_scale")):
                raise ValueError("label revision unexpectedly changed feature normalization")
            if not np.array_equal(model["train_class_counts"], np.bincount(y[train], minlength=2)):
                raise ValueError("training class weights not derived from current training fold")
            new_scores[arm][test] = predict_ridge(model, arrays[arm][test])
            folder = args.out / "fold_models" / arm
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"fold_{number:02d}.npz", **model)
            fitted[arm] = model
        if (any(not np.array_equal(fitted[ARMS[0]][k][:-2], fitted[ARMS[1]][k][:-2]) for k in ("mean", "std"))
                or not np.array_equal(fitted[ARMS[0]]["feature_scale"], fitted[ARMS[1]]["feature_scale"])):
            raise ValueError("common training normalization differs between arms")
        fold_log.append({"fold": number, "heldout_episode": fold["heldout_episode"],
                         "train_n": len(train), "test_n": len(test),
                         "old_train_class_counts": np.bincount(old_y[train], minlength=2).tolist(),
                         "new_train_class_counts": np.bincount(y[train], minlength=2).tolist(),
                         "changed_training_labels": int(np.sum(old_y[train] != y[train])),
                         "normalization_equal_to_original": True, "common_normalization_equal": True})
    fit_seconds = time.perf_counter() - fit_started
    if any(not np.isfinite(v).all() for v in new_scores.values()):
        raise ValueError("incomplete/nonfinite new OOF scores")
    refit = summarize(y, new_scores, groups, majority_predictions(y, folds))
    stages = dict(zip(STAGES, (original, frozen_result, refit)))
    for stage, details in stages.items():
        details["diagnostic_ridge_fits_this_stage"] = 20 if stage == STAGES[2] else 0
        details["evaluation_labels"] = "original" if stage == STAGES[0] else "revised"
    changes = {}
    for arm in ARMS:
        previous, new = old_scores[arm] >= .5, new_scores[arm] >= .5
        correct_before, correct_after = previous == y, new == y
        changes[arm] = {"metrics": {k: difference(frozen_result["methods"][arm]["pooled"][k],
                                                    refit["methods"][arm]["pooled"][k])
                                    for k in ("accuracy", "balanced_accuracy")},
                        "changed_prediction_ui_indices": [r["ui_index"] for i, r in enumerate(old_oof) if previous[i] != new[i]],
                        "improved_windows": int(np.sum(~correct_before & correct_after)),
                        "worsened_windows": int(np.sum(correct_before & ~correct_after)),
                        "mean_absolute_score_change": float(np.mean(np.abs(new_scores[arm] - old_scores[arm])))}
    records = []
    for i, row in enumerate(old_oof):
        records.append({**{k: row[k] for k in ("sample_index", "window_id", "ui_index", *groups)},
                        "old_label": LABELS[old_y[i]], "revised_label": LABELS[y[i]],
                        "label_changed": bool(old_y[i] != y[i]), "frozen_methods": row["methods"],
                        "refit_methods": {arm: {"score": float(new_scores[arm][i]),
                                                "prediction": LABELS[int(new_scores[arm][i] >= .5)]} for arm in ARMS}})
    report = {"schema": SCHEMA, "status": "completed", **FLAGS, "diagnostic_ridge_training_executed": True,
              "samples": 45, "episodes": 10, "class_counts": dict(Counter(LABELS[v] for v in y)),
              "label_changes": label_changes, "stages": stages,
              "refit_minus_frozen_on_revised_labels": changes,
              "label_only_metric_changes": {arm: {k: difference(original["methods"][arm]["pooled"][k],
                                                                   frozen_result["methods"][arm]["pooled"][k])
                                                 for k in ("accuracy", "balanced_accuracy")} for arm in ARMS},
              "checks": {"frozen_sample_order_and_ten_folds": "passed", "only_approved_response_changes": "passed",
                         "current_human_source_matches_revision_snapshot": "passed", "cached_y_matches_original": "passed",
                         "only_request_block_differs": "passed", "old_score_replay_max_absolute_error": max(replay_errors),
                         "old_report_exactly_reproduced": True, "old_and_new_normalization_equal_model_count": 20,
                         "common_normalization_equal_folds": 10, "new_training_fold_class_counts_checked": True,
                         "complete_oof": True, "source_artifacts_modified": False},
              "fit_seconds": fit_seconds, "elapsed_s": time.perf_counter() - started,
              "execution": {"device": "cpu_numpy", "numpy": np.__version__, "new_fits": 20},
              "visual_status": "not_viewed",
              "claim_scope": "reused development set after targeted human review; no causal/world-model/policy/contact or independent-test claim"}
    write_json(args.out / "fold_log.json", fold_log)
    write_jsonl(args.out / "oof_predictions.jsonl", records)
    write_json(args.out / "report.json", report)
    print(json.dumps({"elapsed_s": report["elapsed_s"], "fit_seconds": fit_seconds,
                      "checks": report["checks"], "stages": {s: {"pooled": {a: d["methods"][a]["pooled"] for a in ARMS},
                          "paired": d["paired_request_minus_observation"]} for s, d in stages.items()}}, ensure_ascii=False, indent=2))


def render(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    report = read_json(out / "report.json")
    if report["schema"] != SCHEMA or report["status"] != "completed":
        raise ValueError("completed real retest required")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    data = {stage: {arm: report["stages"][stage]["methods"][arm]["pooled"] for arm in ARMS} for stage in STAGES}
    write_json(figures / "data-manifest.json", {"real_or_mock": "real", "source": "../report.json",
        "script": "tools/retest_real10_pre_action_response.py --stage render", "plotted_data": data,
        "outputs": ["label_retest_zh.png", "label_retest_zh.svg"], "visual_status": "not_viewed"})
    (figures / "data-manifest.md").write_text(
        "# Figure data manifest\n\n| Figure | Data | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Human-label retest | ../report.json | Real | Same 45 windows / 10 LOEO development folds | "
        "tools/retest_real10_pre_action_response.py --stage render | label_retest_zh.png / .svg |\n\n"
        "First row uses old labels. Second/third rows use revised labels. No independent test or policy claim.\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.65, 1, 1]})
    for offset, arm, color, label in zip([-.17, .17], ARMS, ["#4E79A7", "#F28E2B"], ["仅动作前观测", "观测＋Piper请求"]):
        values = [100 * data[s][arm]["balanced_accuracy"] for s in STAGES]
        bars = axes[0].barh(np.arange(3) + offset, values, height=.29, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.2f}%", va="center", fontsize=10)
    axes[0].set_yticks(np.arange(3), ["原标签＋旧预测", "新标签＋旧预测\n（不重新拟合）", "新标签＋重新拟合"])
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(2.55, -.65)
    axes[0].set_xlabel("平衡准确率（%）")
    axes[0].set_title("A  标签影响与重拟合影响分开", loc="left", weight="bold")
    axes[0].grid(axis="x", alpha=.18)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.07), ncol=2, frameon=False, fontsize=9)
    for ax, arm, title in zip(axes[1:], ARMS, ["B  重拟合：仅观测", "C  重拟合：加入请求"]):
        matrix = np.asarray(data[STAGES[2]][arm]["confusion_matrix_true_rows_predicted_columns"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(report["class_counts"].values()))
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, str(value), ha="center", va="center", fontsize=22,
                    color="white" if value > max(report["class_counts"].values())*.55 else "#243447")
        ax.set_xticks([0, 1], ["无明显推进", "推进"])
        ax.set_yticks([0, 1], ["无明显推进", "推进"])
        ax.set_xlabel("预测响应")
        ax.set_ylabel("修订后的人工标签")
        ax.set_title(title, loc="left", weight="bold")
    paired = report["stages"][STAGES[2]]["paired_request_minus_observation"]
    fig.suptitle("人工复核后的固定配置复测", y=.97, fontsize=18, weight="bold")
    fig.text(.5, .14, f"重新拟合后加入请求：平衡准确率 {paired['balanced_accuracy']['percentage_points']:+.2f} 个百分点；"
             f"{paired['improved_windows']} 窗改对 / {paired['worsened_windows']} 窗改错。", ha="center", fontsize=11)
    fig.text(.5, .065, "同45窗、同10折、同特征/参数；仅修订#100与#62的响应标签。\n"
             "反复使用的开发诊断集；不是独立测试、因果效应或策略成功率。", ha="center", fontsize=10, color="#4B5563", linespacing=1.6)
    fig.subplots_adjust(left=.135, right=.985, top=.72, bottom=.28, wspace=.48)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"label_retest_zh.{ext}", dpi=450)
    plt.close(fig)
    print(figures / "label_retest_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "render"), default="run")
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_pre_action_response_v1"))
    parser.add_argument("--revisions", type=Path, default=Path("simulation_output/real10_hardcase_human_review_20260921_v2/applied"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pre_action_label_retest_v1"))
    args = parser.parse_args()
    run(args) if args.stage == "run" else render(args.out)


if __name__ == "__main__":
    main()
