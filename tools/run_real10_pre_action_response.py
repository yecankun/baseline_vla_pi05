"""Fixed pre-action response comparison; small diagnostic ridge fits, no PI05.

Both arms share observable history. Only the Piper request block differs.
Future observations and post-submit logs are never prediction features.
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
from run_real10_response_baseline import fit_ridge, predict_ridge, metrics, folds_for, LABELS, SEED


SCHEMA = "real10_pre_action_request_comparison_v1"
ARMS = ("observation_only", "observation_plus_request")
BLOCKS = [12, 2048, 2]
ALPHA, CLIP = 1.0, 5.0
STATE_NAMES = ([f"current_elite_{a}_mm" for a in "xyz"]
               + [f"past_elite_change_{a}_mm" for a in "xyz"]
               + ["task_left", "task_right", "previous_busy", "past_busy_mean",
                  "past_event_count_change_not_mm", "past_observation_span_s"])


def encode_past_images(inputs):
    """Only first/last PRE-ACTION images; never accepts target rows or logs."""
    import PIL
    from PIL import Image
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights

    if not torch.cuda.is_available():
        raise RuntimeError("expected remote CUDA environment")
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cached weights required; no download: {checkpoint}")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval().to("cuda")
    keys = sorted({h["images"][view]["path"] for x in inputs
                   for h in (x["history"][0], x["history"][-1]) for view in ("side", "top")})
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    embeddings = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(keys), 32):
            batch = keys[offset:offset + 32]
            pixels = []
            for path in batch:
                with Image.open(path) as image:
                    if image.size != (224, 224):
                        raise ValueError("only the existing 224px pack; no implicit crop/resize")
                    pixels.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
            x = torch.stack(pixels).to(device="cuda", dtype=torch.float32) / 255.0
            z = encoder((x - mean) / std).cpu().numpy()
            embeddings.update(zip(batch, z))
            print(f"pre-action images {min(offset+32, len(keys))}/{len(keys)}", flush=True)
    return embeddings, {"encoder": "torchvision_resnet18_IMAGENET1K_V1_frozen_eval",
        "weights": str(checkpoint), "real10_finetuning": False, "batchnorm_updates": False,
        "images_encoded": len(keys), "future_images_encoded": 0,
        "preprocessing": "existing 224px RGB /255 -> ImageNet mean/std; no additional ROI/enhancement",
        "seconds": time.perf_counter() - started,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "Pillow": PIL.__version__}}


def feature_blocks(x, embeddings):
    """Raw features before fold-specific normalization; query model_input only."""
    history = x["history"]
    s = np.array([h["state_32"] for h in history], dtype=np.float64)
    valid = np.array([h["state_valid_32"] for h in history], dtype=bool)
    s[~valid] = np.nan
    for h in history:
        times = [h["elite_pose_relative_time_s"], *[h["images"][v]["relative_time_s"] for v in ("side", "top")]]
        if h["controller_snapshot_relative_time_s"] is not None:
            times.append(h["controller_snapshot_relative_time_s"])
        if not all(np.isfinite(t) and t <= 0 for t in times):
            raise ValueError("post-cutoff information in query history")
    if any(b["elite_pose_relative_time_s"] <= a["elite_pose_relative_time_s"] for a, b in zip(history, history[1:])):
        raise ValueError("history timestamps must be ordered")
    busy = s[:, 16][np.isfinite(s[:, 16])]
    state = np.concatenate((s[-1, :3], s[-1, :3] - s[0, :3], s[-1, 14:16],
        [s[-1, 16], float(busy.mean()) if len(busy) else np.nan,
         s[-1, 6] - s[0, 6], history[-1]["elite_pose_relative_time_s"] - history[0]["elite_pose_relative_time_s"]]))
    visual = []
    for view in ("side", "top"):
        first = embeddings[history[0]["images"][view]["path"]]
        current = embeddings[history[-1]["images"][view]["path"]]
        visual.extend((current, current - first))
    action = x["candidate_action"]
    if (action["elite_tcp_delta_6d"] is not None or any(action["valid"]["elite_tcp_delta_6d"])
            or action["valid"]["piper_intent_id"] is not True or action["piper_intent_id"] not in (1, 2)
            or x["request_context"]["piper_burst_count"] != 1 or x["nominal_horizon_s"] != 1.5):
        raise ValueError("this fixed probe expects unknown Elite, hold/feed requests, burst1, horizon1.5")
    request = np.array([action["piper_intent_id"] == i for i in (1, 2)], dtype=np.float64)
    return state, np.concatenate(visual), request


def group_metrics(y, prediction, values):
    return {v: metrics(y[np.asarray(values) == v], prediction[np.asarray(values) == v])
            for v in sorted(set(values))}


def later_action_group(audit):
    keys = ("later_piper_requests_before_last_image", "piper_request_brackets_overlap_last_image",
            "later_elite_submissions_before_last_image", "elite_submission_brackets_overlap_last_image",
            "elite_sent_timestamps_inside_response")
    return "later_action_logged" if any(audit[k] for k in keys) else "no_later_action_logged"


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve prior artifacts; choose a fresh --out")
    source_protocol = read_json(args.source / "protocol.json")
    if source_protocol["schema"] != "real10_pre_action_response_interface_v1":
        raise ValueError("unexpected pre-action interface")
    rows = read_jsonl(args.source / "queries.jsonl")
    targets = read_jsonl(args.source / "response_targets.jsonl")
    snapshot = read_jsonl(args.source / "annotation_snapshot.jsonl")
    folds = read_json(args.source / "folds.json")
    latest = {r["window_id"]: r for r in read_jsonl(Path(source_protocol["pack"]) / "annotations_joint_v2.jsonl")}
    if len(rows) != 45 or not (len(rows) == len(targets) == len(snapshot)) or folds != folds_for(snapshot):
        raise ValueError("frozen sample set/split changed")
    for i, (q, target, original) in enumerate(zip(rows, targets, snapshot)):
        if (q["sample_index"] != i or target["sample_index"] != i
                or q["window_id"] != target["window_id"] or q["window_id"] != original["window_id"]
                or latest[q["window_id"]] != original["human_annotation"]
                or target["joint_motion_response"] != original["human_annotation"]["joint_motion_response"]):
            raise ValueError("query/target/current human annotation mismatch")
    flags = {"policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
             "deployable": False, "PI05_training_executed": False, "world_model_training_executed": False,
             "hardware_executed": False}
    protocol = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), **flags,
        "source": str(args.source), "samples": len(rows), "episodes": len(folds),
        "arms": list(ARMS), "block_sizes": BLOCKS, "state_names": STATE_NAMES,
        "visual": "side/top: anchor512 + anchor-minus-first-past512; only query history pixels",
        "request": "observation_only: zeros(2); observation_plus_request: hold/feed onehot",
        "ridge_alpha": ALPHA, "standardized_clip": CLIP, "threshold": .5, "seed": SEED,
        "fit": "balanced ridge, train-fold mean/std/missing bits, fixed block scaling, unpenalized intercept",
        "score_semantics": "unbounded ridge decision score, NOT calibrated probability",
        "split": "unchanged ten whole-episode LOEO development folds; not an independent test",
        "primary_metric": "pooled OOF balanced accuracy", "tuning": "none",
        "statistics": "descriptive paired differences only; correlated windows/folds, no significance claim",
        "later_action_audit": "post-prediction strata only, not input/filter/model selection",
        "full_action_missing": "Elite remains null; no full-action world API invocation"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "folds.json", folds)
    write_jsonl(args.out / "annotation_snapshot.jsonl", snapshot)
    inputs = [r["model_input"] for r in rows]
    embeddings, encoder_info = encode_past_images(inputs)
    blocks = [feature_blocks(x, embeddings) for x in inputs]
    state, visual, request = [np.stack([b[j] for b in blocks]) for j in range(3)]
    if [state.shape[1], visual.shape[1], request.shape[1]] != BLOCKS or not np.isfinite(visual).all():
        raise ValueError("feature block mismatch")
    common = np.concatenate((state, visual), axis=1)
    arrays = {ARMS[0]: np.concatenate((common, np.zeros_like(request)), axis=1),
              ARMS[1]: np.concatenate((common, request), axis=1)}
    if not np.array_equal(arrays[ARMS[0]][:, :-2], arrays[ARMS[1]][:, :-2], equal_nan=True):
        raise ValueError("non-request features differ")
    y = np.array([LABELS.index(t["joint_motion_response"]) for t in targets])
    np.savez_compressed(args.out / "features.npz", **arrays, y=y)
    scores = {arm: np.full(len(y), np.nan) for arm in ARMS}
    majority = np.full(len(y), -1, dtype=np.int64)
    fit_started = time.perf_counter()
    fold_log = []
    for number, fold in enumerate(folds):
        train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        majority[test] = int(np.bincount(y[train], minlength=2)[1] >= np.bincount(y[train], minlength=2)[0])
        models = {}
        for arm in ARMS:
            model = fit_ridge(arrays[arm][train], y[train], alpha=ALPHA,
                              block_sizes=BLOCKS, standardized_clip=CLIP)
            scores[arm][test] = predict_ridge(model, arrays[arm][test])
            folder = args.out / "fold_models" / arm
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"fold_{number:02d}.npz", **model)
            models[arm] = model
        for key in ("mean", "std"):
            if not np.array_equal(models[ARMS[0]][key][:-2], models[ARMS[1]][key][:-2]):
                raise ValueError("shared training normalization differs between arms")
        if not np.array_equal(models[ARMS[0]]["feature_scale"], models[ARMS[1]]["feature_scale"]):
            raise ValueError("adding request changed block scales")
        fold_log.append({"fold": number, "heldout_episode": fold["heldout_episode"],
                         "train_n": len(train), "test_n": len(test), "common_normalization_equal": True})
        print(f"fold {number+1}/{len(folds)} complete, heldout n={len(test)}", flush=True)
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError("incomplete/nonfinite OOF predictions")
    fit_seconds = time.perf_counter() - fit_started
    predictions = {arm: (s >= .5).astype(int) for arm, s in scores.items()}
    # Read audit strata only AFTER feature extraction and both fits/predictions.
    audits = {r["window_id"]: r for r in read_jsonl(args.source / "timing_audit.jsonl")}
    groups = {
        "episode": [r["source_episode"] for r in snapshot],
        "task": [r["task"] for r in snapshot],
        "piper_request": ["feed" if x["candidate_action"]["piper_intent_id"] == 2 else "hold" for x in inputs],
        "later_action_audit": [later_action_group(audits[r["window_id"]]) for r in rows],
    }
    results = {}
    for arm in ARMS:
        result = {"pooled": metrics(y, predictions[arm])}
        result.update({name: group_metrics(y, predictions[arm], values) for name, values in groups.items()})
        result["episode_macro_accuracy"] = float(np.mean([v["accuracy"] for v in result["episode"].values()]))
        results[arm] = result
    paired = {}
    for key in ("accuracy", "balanced_accuracy"):
        a, b = [results[arm]["pooled"][key] for arm in ARMS]
        paired[key] = {"absolute_change": b-a, "percentage_points": 100*(b-a),
                       "relative_change_percent": 100*(b-a)/a if a else None}
    episode_delta = {e: results[ARMS[1]]["episode"][e]["accuracy"] - results[ARMS[0]]["episode"][e]["accuracy"]
                     for e in sorted(set(groups["episode"]))}
    correct = [predictions[arm] == y for arm in ARMS]
    paired.update({"improved_windows": int(np.sum(~correct[0] & correct[1])),
        "worsened_windows": int(np.sum(correct[0] & ~correct[1])),
        "changed_predictions": int(np.sum(predictions[ARMS[0]] != predictions[ARMS[1]])),
        "episode_accuracy_changes": episode_delta,
        "episode_win_tie_loss": [sum(v > 0 for v in episode_delta.values()),
                                  sum(v == 0 for v in episode_delta.values()), sum(v < 0 for v in episode_delta.values())],
        "mean_absolute_score_change": float(np.mean(np.abs(scores[ARMS[1]] - scores[ARMS[0]]))),
        "episode_macro_accuracy_change": float(np.mean(list(episode_delta.values())))})
    oof = [{"sample_index": i, "window_id": r["window_id"], "ui_index": snapshot[i]["ui_index"],
            "episode": groups["episode"][i], "task": groups["task"][i],
            "piper_request": groups["piper_request"][i], "later_action_audit": groups["later_action_audit"][i],
            "label": LABELS[y[i]],
            "methods": {arm: {"score": float(scores[arm][i]), "prediction": LABELS[predictions[arm][i]]} for arm in ARMS}}
           for i, r in enumerate(rows)]
    report = {"schema": SCHEMA, "status": "completed", **flags, "diagnostic_ridge_training_executed": True,
        "samples": len(y), "episodes": len(folds), "class_counts": dict(Counter(t["joint_motion_response"] for t in targets)),
        "encoder": encoder_info, "methods": results, "paired_request_minus_observation": paired,
        "references": {"always_advance": metrics(y, np.ones_like(y)), "train_majority": metrics(y, majority)},
        "fold_log": fold_log, "state_nonconstant": [n for j,n in enumerate(STATE_NAMES) if len(np.unique(state[:,j][np.isfinite(state[:,j])])) > 1],
        "checks": {"frozen_samples_and_folds": "passed", "pre_cutoff_feature_times": "passed",
                   "only_request_block_differs": "passed", "shared_normalization_identical_folds": len(folds),
                   "both_request_classes_present": sorted(np.unique(request, axis=0).tolist()), "oof_complete": True},
        "fit_seconds": fit_seconds, "elapsed_s": time.perf_counter() - started,
        "visual_status": "not_viewed",
        "claim_scope": "reused small diagnostic development batch; observational response only, not causal effect/world-model/policy/contact performance"}
    write_jsonl(args.out / "oof_predictions.jsonl", oof)
    write_json(args.out / "report.json", report)
    print(json.dumps({"elapsed_s": report["elapsed_s"], "pooled": {a: results[a]["pooled"] for a in ARMS}, "paired": paired}, ensure_ascii=False), flush=True)


def render(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    report = read_json(out / "report.json")
    if report["schema"] != SCHEMA or report["status"] != "completed":
        raise ValueError("completed experiment required")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    data = {arm: report["methods"][arm]["pooled"] for arm in ARMS}
    write_json(figures / "data-manifest.json", {"real_or_mock": "real", "source": "../report.json",
        "script": "tools/run_real10_pre_action_response.py --stage render", "plotted_data": data,
        "outputs": ["pre_action_response_zh.png", "pre_action_response_zh.svg"], "visual_status": "not_viewed"})
    (figures / "data-manifest.md").write_text(
        "# Figure data manifest\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Pre-action response | ../report.json | Real | 45 windows, 10 fixed LOEO development folds | "
        "tools/run_real10_pre_action_response.py --stage render | pre_action_response_zh.png / .svg |\n\n"
        "No policy, causal-effect or independent-test claim; all plotted values in data-manifest.json.\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    blue, orange = "#4E79A7", "#F28E2B"
    x = np.arange(2)
    for offset, arm, color, label in zip([-.19, .19], ARMS, [blue, orange], ["仅动作前信息", "加入 Piper 请求"]):
        values = [100*data[arm][key] for key in ("accuracy", "balanced_accuracy")]
        bars = axes[0].bar(x+offset, values, width=.34, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(x, ["准确率", "平衡准确率"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("汇总 OOF 指标（%）")
    axes[0].set_title("A  同划分、同分类器对照", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=.18)
    for ax, arm, title in zip(axes[1:], ARMS, ["B  仅动作前信息", "C  加入 Piper 请求"]):
        matrix = np.array(data[arm]["confusion_matrix_true_rows_predicted_columns"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(report["class_counts"].values()))
        for (r,c), value in np.ndenumerate(matrix):
            ax.text(c, r, str(value), ha="center", va="center", fontsize=22,
                    color="white" if value > max(report["class_counts"].values())*.55 else "#243447")
        ax.set_xticks([0,1], ["无明显推进", "推进"])
        ax.set_yticks([0,1], ["无明显推进", "推进"])
        ax.set_xlabel("预测响应")
        ax.set_ylabel("人工响应标签")
        ax.set_title(title, loc="left", weight="bold")
    change = report["paired_request_minus_observation"]
    fig.suptitle("动作前信息能否预测随后响应？", y=.96, fontsize=17, weight="bold")
    fig.text(.5, .09, f"加入请求：平衡准确率变化 {change['balanced_accuracy']['percentage_points']:+.2f} 个百分点；"
             f"{change['improved_windows']} 窗改对 / {change['worsened_windows']} 窗改错。\n"
             "45 个已反复使用的诊断窗口，10 条轨迹整条留出；未来图像不作输入；非因果效果、非策略成功率。",
             ha="center", fontsize=10, color="#4B5563", linespacing=1.7)
    fig.subplots_adjust(left=.065, right=.98, top=.80, bottom=.29, wspace=.45)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"pre_action_response_zh.{ext}", dpi=450)
    plt.close(fig)
    print(figures / "pre_action_response_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "render"), default="run")
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_action_effect_interface_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pre_action_response_v1"))
    args = parser.parse_args()
    run(args) if args.stage == "run" else render(args.out)


if __name__ == "__main__":
    main()
