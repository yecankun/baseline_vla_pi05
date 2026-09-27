"""Small, offline post-action response recognizers; NOT a contact/policy model.

Freeze ImageNet ResNet18, then fit fixed linear ridge classifiers in leave-one-
episode-out folds. Original PI05 training splits, human revisions, action targets,
and deployed models are read-only. --stage all is the short diagnostic fit/eval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import math
import os
from pathlib import Path
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from review_real10_event_windows import ANNOTATION_SCHEMA, annotation_status


SCHEMA = "real10_post_action_response_baseline_v1"
LABELS = ("stationary", "advance")
LEARNED = ("controller_only", "dual_static", "dual_temporal")
METHODS = ("always_advance", "train_majority", *LEARNED)
NAMES = {"always_advance": "总预测推进", "train_majority": "训练折多数类",
         "controller_only": "仅控制器与Elite状态", "dual_static": "双视角静态",
         "dual_temporal": "双视角时序"}
LABEL_ZH = {"stationary": "无明显推进", "advance": "推进"}
ALPHA = 1.0
SEED = 20260920


def select_windows(pack):
    """Latest explicit v2 human revisions only; metadata never becomes features."""
    manifest = read_json(pack / "manifest.json")
    windows = read_jsonl(pack / "windows.jsonl")
    targets = {r["window_id"]: r for r in read_jsonl(pack / "window_targets.jsonl")}
    revisions = read_jsonl(pack / "annotations_joint_v2.jsonl")
    latest = {}
    for row in revisions:
        if row["annotation_schema"] != ANNOTATION_SCHEMA:
            raise ValueError("only joint-response v2 human annotations are supported")
        latest[row["window_id"]] = row
    by_id = {w["window_id"]: w for w in windows}
    if set(latest) - set(by_id):
        raise ValueError("annotations contain windows outside this pack")
    selected, excluded = [], []
    for number, w in enumerate(windows, 1):
        key = w["window_id"]
        a = latest.get(key)
        if a is None or annotation_status(a) != "reviewed":
            excluded.append({"window_id": key, "reason": "not_complete_v2"})
            continue
        if a["annotation_source"] != "human_visual_review":
            raise ValueError("not an explicit human revision")
        for field in ("source_episode", "split_group", "source_split"):
            if a[field] != w[field]:
                raise ValueError(f"annotation provenance mismatch: {key}/{field}")
        if w["split_group"] != w["source_episode"] or w["source_split"] != "train":
            raise ValueError("expected original whole-episode train-only provenance")
        if a["joint_motion_response"] not in LABELS:
            excluded.append({"window_id": key, "reason": a["joint_motion_response"]})
            continue
        if a["motion_evidence"] not in ("side", "top", "both"):
            raise ValueError("binary response needs visual evidence")
        future = targets[key]["future_frame_ids"]
        if not future:
            raise ValueError(f"no response frames: {key}")
        selected.append({"window_id": key, "ui_index": number,
                         "source_episode": w["source_episode"], "task": w["task"],
                         "original_source_split": w["source_split"],
                         "anchor_frame_id": w["anchor_frame_id"], "future_frame_ids": future,
                         "human_annotation": a,
                         "selection_reason_audit_only": w["review_selection_reason"]})
    if len({r["source_episode"] for r in selected}) < 3:
        raise ValueError("need at least three source episodes for this diagnostic")
    if {r["human_annotation"]["joint_motion_response"] for r in selected} != set(LABELS):
        raise ValueError("this baseline needs both advance and stationary labels")
    if not {r["source_episode"] for r in selected} <= set(manifest["source_episodes"]):
        raise ValueError("window episode is outside the original real10 allowlist")
    return selected, excluded, len(revisions)


def numeric(value):
    return float(value) if isinstance(value, (bool, int, float)) and math.isfinite(value) else np.nan


def controller_features(sequence):
    """Observable states over the SAME response interval as vision, no log sidecar.

    Post-anchor observations are permitted for response recognition only. Use each
    frame's causal previous-record snapshot. No event IDs, absolute counters/time,
    requested pose deltas, completion sidecars or human labels are read.
    """
    values, names = [], []

    def add(name, value):
        names.append(name)
        values.append(numeric(value))

    def state(frame):
        h = frame["controller_history"]
        return h["values"] if h["valid"] else {}

    first, last = sequence[0], sequence[-1]
    states = [state(f) for f in sequence]
    add("task_right", first["task"] == "right")
    add("response_span_s", last["observation_available_at_s"] - first["observation_available_at_s"])
    poses = np.array([f["elite"]["tcp_pose_6d_xyz_mm_rpy_rad"] if f["elite"]["valid"]
                      else [np.nan] * 6 for f in sequence], dtype=np.float64)
    for j, axis in enumerate(("x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad")):
        add(f"anchor_elite_{axis}", poses[0, j])
        add(f"observed_elite_change_{axis}", poses[-1, j] - poses[0, j])
    add("observed_elite_xyz_path_mm", np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum())
    for prefix, f, s in (("anchor", first, states[0]), ("end", last, states[-1])):
        for name in ("piper_busy", "piper_async_command", "piper_cooldown",
                     "piper_request_accepted", "piper_executed_feed"):
            add(f"{prefix}_{name}", s.get(name))
        status = s.get("piper_async_status")
        for option in ("idle", "running", "other"):
            match = status not in ("idle", "running") if option == "other" else status == option
            add(f"{prefix}_status_{option}", match if status is not None else None)
        for event in ("start", "done"):
            timestamp = numeric(s.get(f"piper_async_{event}_timestamp"))
            age = f["observation_available_at_s"] - timestamp
            add(f"{prefix}_age_since_{event}_s", age if age >= 0 else None)
    add("controller_observed_fraction", np.mean([f["controller_history"]["valid"] for f in sequence]))
    for name, field, target in (("busy_fraction", "piper_busy", True),
                                ("feed_command_history_fraction", "piper_async_command", 1),
                                ("running_history_fraction", "piper_async_status", "running")):
        observed = [s[field] == target for s in states if s.get(field) is not None]
        add(name, np.mean(observed) if observed else None)
    counts = [numeric(s.get("piper_step_after_command")) for s in (states[0], states[-1])]
    add("logged_counter_change_not_physical_feed", counts[-1] - counts[0])
    return np.asarray(values, dtype=np.float64), names


def image_path(image_pack, frame_id, view):
    episode, step = frame_id.split("/")
    if Path(episode).name != episode or not step.isdigit() or view not in ("side", "top"):
        raise ValueError("invalid image reference")
    return image_pack / "images" / episode / view / f"{int(step):06d}.png"


def embed_images(rows, image_pack, device, batch_size):
    import PIL
    from PIL import Image
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu if intended")
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cached ImageNet weights required, no automatic download: {checkpoint}")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval().to(device)
    keys = sorted({(f, view) for r in rows for f in [r["anchor_frame_id"], *r["future_frame_ids"]]
                   for view in ("side", "top")})
    result = {}
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(keys), batch_size):
            batch = keys[offset:offset + batch_size]
            tensors = []
            for frame, view in batch:
                with Image.open(image_path(image_pack, frame, view)) as source:
                    if source.size != (224, 224):
                        raise ValueError("use the existing 224px real10 image pack; no implicit resize/crop")
                    rgb = np.array(source.convert("RGB"), copy=True)
                tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
            x = torch.stack(tensors).to(device=device, dtype=torch.float32) / 255.0
            z = encoder((x - mean) / std).cpu().numpy()
            for key, vector in zip(batch, z):
                result[key] = vector
            print(f"encoded {min(offset + batch_size, len(keys))}/{len(keys)} images", flush=True)
    return result, {"encoder": "torchvision_resnet18_IMAGENET1K_V1_frozen_eval",
                    "weights_path": str(checkpoint), "weights_url": weights.url,
                    "real10_finetuning": False, "batchnorm_running_stats_updated": False,
                    "image_size": [224, 224], "additional_crop_or_enhancement": False,
                    "preprocessing": "existing square INTER_AREA PNG -> RGB [0,1] -> ImageNet mean/std",
                    "images_encoded": len(keys), "device": device,
                    "feature_seconds": time.perf_counter() - started,
                    "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                 "numpy": np.__version__, "Pillow": PIL.__version__}}


def folds_for(rows):
    groups = sorted({r["source_episode"] for r in rows})
    return [{"heldout_episode": g,
             "train_indices": [i for i, r in enumerate(rows) if r["source_episode"] != g],
             "test_indices": [i for i, r in enumerate(rows) if r["source_episode"] == g]}
            for g in groups]


def prepare(args):
    marker = args.out / "prepared.json"
    if marker.exists():
        p = read_json(marker)
        if p["schema"] != SCHEMA or not (args.out / "features.npz").is_file():
            raise ValueError("incompatible/incomplete prepared artifact")
        print("reuse frozen preparation; newer annotations require a NEW --out", flush=True)
        return p
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("incomplete output exists: preserve it and choose a new --out")
    started = time.perf_counter()
    rows, excluded, revisions = select_windows(args.pack)
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    image_manifest = read_json(args.image_pack / "manifest.json")
    if image_manifest["adapter_version"] != "real10_pi05_observable_history_v1":
        raise ValueError("unexpected resized image pack")
    if not {r["source_episode"] for r in rows} <= set(image_manifest["source_episodes"]):
        raise ValueError("image pack does not cover selected episodes")
    controls, names = [], None
    for r in rows:
        seq = [observations[f] for f in [r["anchor_frame_id"], *r["future_frame_ids"]]]
        if any(f["source_episode"] != r["source_episode"] for f in seq):
            raise ValueError("response sequence crosses episode boundary")
        if any(b["observation_available_at_s"] <= a["observation_available_at_s"]
               for a, b in zip(seq, seq[1:])):
            raise ValueError("nonmonotonic response observations")
        control, names = controller_features(seq)
        controls.append(control)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": SCHEMA, "stage": "offline_post_action_response_recognition",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "pack": str(args.pack.resolve()), "image_pack": str(args.image_pack.resolve()),
                "annotation_authority": "latest explicit human v2 revision, snapshot below",
                "annotation_revisions": revisions, "samples": len(rows),
                "episodes": len({r["source_episode"] for r in rows}),
                "label_order": list(LABELS),
                "class_counts": dict(Counter(r["human_annotation"]["joint_motion_response"] for r in rows)),
                "excluded": excluded, "split": "leave_one_whole_source_episode_out",
                "original_PI05_train_only_split_changed": False,
                "methods": list(METHODS), "seed": SEED, "ridge_alpha": ALPHA,
                "classifier": "class-balanced ridge, unpenalized intercept, score threshold 0.5; not calibrated probability",
                "normalization": "per-fold train-only finite mean/std; zero after imputation; explicit missing bits; divide by sqrt(input_dimension)",
                "tuning": "none: fixed features/alpha/threshold, every fold reported, no selected checkpoint",
                "controller_feature_names": names,
                "controller_scope": "anchor and full response interval observed Elite + causal controller_history, same end horizon as temporal vision",
                "visual_static": "side anchor 512 + top anchor 512",
                "visual_temporal": "per view: anchor512, mean(future)-anchor512, last(future)-anchor512; concatenate side/top",
                "forbidden_inputs": ["human visibility/evidence/response", "window/episode IDs", "sample selection reason",
                                     "absolute timestamps/counters", "execution_log", "factual_transitions",
                                     "future completion diagnostic fields", "contact/wall/tip/route truth"],
                "future_images": "allowed ONLY as offline post-action recognizer inputs, NOT action-before prediction or policy observations",
                "metrics": ["pooled OOF accuracy", "pooled OOF balanced accuracy", "confusion matrix",
                            "each-episode accuracy", "episode-macro accuracy", "left/right metrics", "runtime"],
                "claim_scope": "selected weak-label feasibility on this recording batch, not contact, force, policy improvement or new-geometry generalization",
                "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False}
    # Written BEFORE feature extraction/fitting; no choice is tuned on OOF results.
    write_json(args.out / "protocol.json", protocol)
    write_jsonl(args.out / "annotation_snapshot.jsonl", rows)
    write_json(args.out / "folds.json", folds_for(rows))
    embeddings, info = embed_images(rows, args.image_pack, args.device, args.batch_size)
    static, temporal = [], []
    for r in rows:
        still, moving = [], []
        for view in ("side", "top"):
            anchor = embeddings[(r["anchor_frame_id"], view)]
            future = np.stack([embeddings[(f, view)] for f in r["future_frame_ids"]])
            still.append(anchor)
            moving.extend((anchor, future.mean(axis=0) - anchor, future[-1] - anchor))
        static.append(np.concatenate(still))
        temporal.append(np.concatenate(moving))
    arrays = {"controller_only": np.stack(controls), "dual_static": np.stack(static),
              "dual_temporal": np.stack(temporal),
              "y": np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])}
    if not all(np.isfinite(arrays[name]).all() for name in ("dual_static", "dual_temporal")):
        raise ValueError("nonfinite visual features")
    np.savez_compressed(args.out / "features.npz", **arrays)
    p = {"schema": SCHEMA, "samples": len(rows), "episodes": protocol["episodes"],
         "class_counts": protocol["class_counts"], "encoder": info,
         "feature_dimensions": {k: arrays[k].shape[1] for k in LEARNED},
         "controller_nonconstant_features": [names[j] for j in range(len(names))
             if len(np.unique(arrays["controller_only"][:, j][np.isfinite(arrays["controller_only"][:, j])])) > 1],
         "prepare_seconds": time.perf_counter() - started, "training_executed": False}
    write_json(marker, p)
    return p


def _ridge_standardized_features(x, mean, std, standardized_clip=None):
    """Impute to train mean; optional raw-feature clipping leaves missing bits intact."""
    finite = np.isfinite(x)
    values = np.where(finite, (x - mean) / std, 0)
    if standardized_clip is not None:
        values = np.clip(values, -standardized_clip, standardized_clip)
    return np.concatenate((values, (~finite).astype(float)), axis=1)


def fit_ridge(x, y, alpha=ALPHA, block_sizes=None, *, standardized_clip=None):
    """Closed form in sample space (<=44x44); fit statistics on train rows only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if standardized_clip is not None and (not np.isfinite(standardized_clip) or standardized_clip <= 0):
        raise ValueError("standardized_clip must be finite and positive")
    counts = np.bincount(y, minlength=2)
    if not counts.all():
        raise ValueError("a training fold is missing a class")
    finite = np.isfinite(x)
    n = finite.sum(axis=0)
    mean = np.where(finite, x, 0).sum(axis=0) / np.maximum(n, 1)
    var = np.where(finite, x - mean, 0) ** 2
    std = np.sqrt(var.sum(axis=0) / np.maximum(n, 1))
    std[std < 1e-6] = 1.0
    z = _ridge_standardized_features(x, mean, std, standardized_clip)
    feature_scale = None
    if block_sizes is None:
        z /= np.sqrt(z.shape[1])
    else:
        # Keep the controller's scale unchanged when adding a high-dimensional
        # visual block. Each block includes its own missing indicators.
        if any(size <= 0 for size in block_sizes) or sum(block_sizes) != x.shape[1]:
            raise ValueError("feature block sizes do not match the input")
        feature_scale = np.tile(np.sqrt(np.repeat(2 * np.asarray(block_sizes), block_sizes)), 2)
        z /= feature_scale
    weight = len(y) / (2.0 * counts[y])
    center = np.average(z, weights=weight, axis=0)
    target_center = np.average(y, weights=weight)
    root = np.sqrt(weight)
    design = (z - center) * root[:, None]
    dual = np.linalg.solve(design @ design.T + alpha * np.eye(len(y)), (y - target_center) * root)
    coef = design.T @ dual
    model = {"mean": mean, "std": std, "center": center, "coef": coef,
             "target_center": np.array(target_center), "alpha": np.array(alpha), "train_class_counts": counts}
    if feature_scale is not None:
        model["feature_scale"] = feature_scale
    if standardized_clip is not None:
        model["standardized_clip"] = np.array(float(standardized_clip))
    return model


def predict_ridge(model, x):
    x = np.asarray(x, dtype=np.float64)
    clip = model["standardized_clip"] if "standardized_clip" in model else None
    z = _ridge_standardized_features(x, model["mean"], model["std"], clip)
    z /= model["feature_scale"] if "feature_scale" in model else np.sqrt(z.shape[1])
    return (z - model["center"]) @ model["coef"] + model["target_center"]


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    support = matrix.sum(axis=1)
    recall = [float(matrix[j, j] / support[j]) if support[j] else None for j in range(2)]
    return {"n": len(y), "accuracy": float(np.mean(y == prediction)),
            "balanced_accuracy": float(np.mean(recall)) if all(support) else None,
            "recall_stationary": recall[0], "recall_advance": recall[1],
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist()}


def write_examples(out, rows, predictions, report, image_pack):
    """Representative cases reference the existing image pack; no copied images."""
    lookup = {(r["window_id"], r["method"]): r for r in predictions}
    chosen = []
    for truth in LABELS:
        for guess in LABELS:
            pool = [r for r in rows if r["human_annotation"]["joint_motion_response"] == truth
                    and lookup[(r["window_id"], "dual_temporal")]["prediction"] == guess]
            if pool:
                chosen.append(pool[0])
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实数据短时响应基线</title>',
             '<style>body{font:16px "Microsoft YaHei",sans-serif;background:#f5f6f8;color:#202939;max-width:1120px;margin:25px auto}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #d5dce6}section{background:white;padding:18px;margin:20px 0;border-radius:10px}.frames{display:flex;gap:12px}figure{margin:0}img{width:224px;height:224px}h2{font-size:20px}.bad{color:#b42318}.good{color:#067647}</style>',
             '<h1>真实数据：动作后短时响应识别</h1><p>整条轨迹留一评估；人工弱标签，不是碰壁真值。图像为已有224像素版本。以下按时序模型每个真实/预测组合选首个窗口，不代表总体成功率。</p>',
             '<table><tr><th>方法</th><th>准确率</th><th>平衡准确率</th><th>轨迹宏平均准确率</th></tr>']
    for method in METHODS:
        m = report["methods"][method]
        parts.append(f'<tr><td>{NAMES[method]}</td><td>{m["pooled"]["accuracy"]:.1%}</td>'
                     f'<td>{m["pooled"]["balanced_accuracy"]:.1%}</td><td>{m["episode_macro_accuracy"]:.1%}</td></tr>')
    parts.append('</table>')
    for row in chosen:
        a = row["human_annotation"]
        pred = lookup[(row["window_id"], "dual_temporal")]["prediction"]
        correct = pred == a["joint_motion_response"]
        parts.append(f'<section><h2>窗口{row["ui_index"]} · 人工：{LABEL_ZH[a["joint_motion_response"]]} · '
                     f'<span class="{"good" if correct else "bad"}">时序预测：{LABEL_ZH[pred]}</span></h2>'
                     f'<p>{html.escape(row["source_episode"])}</p><div class="frames">')
        for frame, title in ((row["anchor_frame_id"], "锚点"), (row["future_frame_ids"][-1], "窗口末帧")):
            for view, view_name in (("side", "侧视"), ("top", "俯视")):
                relative = Path(os.path.relpath(image_path(image_pack, frame, view), out)).as_posix()
                parts.append(f'<figure><img src="{html.escape(relative, quote=True)}"><figcaption>{title} · {view_name}</figcaption></figure>')
        descriptions = [f'{NAMES[m]}：{LABEL_ZH[lookup[(row["window_id"], m)]["prediction"]]}' for m in METHODS]
        parts.append('</div><p>' + '；'.join(descriptions) + '</p></section>')
    parts.append('</html>')
    (out / "examples.html").write_text('\n'.join(parts), encoding="utf-8")


def evaluate(args):
    if (args.out / "report.json").exists():
        raise FileExistsError("result already exists; inspect it instead of repeating/model selection")
    prepared = read_json(args.out / "prepared.json")
    protocol = read_json(args.out / "protocol.json")
    if prepared["schema"] != SCHEMA or protocol["ridge_alpha"] != ALPHA:
        raise ValueError("incompatible prepared feature/protocol version")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    folds = read_json(args.out / "folds.json")
    with np.load(args.out / "features.npz", allow_pickle=False) as z:
        arrays = {name: z[name] for name in (*LEARNED, "y")}
    y = arrays["y"]
    expected_y = np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    if not np.array_equal(y, expected_y) or folds != folds_for(rows):
        raise ValueError("frozen labels/folds differ from prepared rows")
    started = time.perf_counter()
    all_predictions, results, timings = [], {}, {}
    for method in METHODS:
        method_start = time.perf_counter()
        scores = np.full(len(rows), np.nan)
        fold_metrics = []
        for fold_index, fold in enumerate(folds):
            train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
            if method == "always_advance":
                score = np.ones(len(test))
            elif method == "train_majority":
                counts = np.bincount(y[train], minlength=2)
                score = np.full(len(test), float(counts[1] >= counts[0]))
            else:
                model = fit_ridge(arrays[method][train], y[train])
                score = predict_ridge(model, arrays[method][test])
                folder = args.out / "fold_models" / method
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"fold_{fold_index:02d}.npz", **model)
            scores[test] = score
            fold_metrics.append({"episode": fold["heldout_episode"],
                                 "train_n": len(train), "test_n": len(test),
                                 **metrics(y[test], (score >= .5).astype(int))})
        if not np.isfinite(scores).all():
            raise ValueError("not every selected window received an OOF prediction")
        predicted = (scores >= .5).astype(int)
        by_task = {}
        for task in ("left", "right"):
            subset = np.array([i for i, r in enumerate(rows) if r["task"] == task], dtype=int)
            by_task[task] = metrics(y[subset], predicted[subset]) if len(subset) else None
        results[method] = {"pooled": metrics(y, predicted), "episodes": fold_metrics, "by_task": by_task,
                           "episode_macro_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics]))}
        timings[method] = time.perf_counter() - method_start
        for i, row in enumerate(rows):
            all_predictions.append({"window_id": row["window_id"], "ui_index": row["ui_index"],
                                    "heldout_episode": row["source_episode"], "task": row["task"],
                                    "method": method, "target": LABELS[y[i]], "prediction": LABELS[predicted[i]],
                                    "score_not_probability": float(scores[i]), "correct": bool(y[i] == predicted[i])})
    write_jsonl(args.out / "oof_predictions.jsonl", all_predictions)
    report = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds), "label_order": list(LABELS),
              "class_counts": prepared["class_counts"], "methods": results,
              "temporal_minus_static_balanced_accuracy_pp": 100 * (results["dual_temporal"]["pooled"]["balanced_accuracy"]
                  - results["dual_static"]["pooled"]["balanced_accuracy"]),
              "timing": {"prepare_seconds": prepared["prepare_seconds"], "fit_evaluate_seconds": time.perf_counter() - started,
                         "method_seconds_including_save": timings},
              "evidence": "small selected-batch episode-grouped diagnostic; no independent final test, no tuning or significance claim",
              "recognizer_fit_executed": True, "PI05_training_executed": False,
              "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
              "hardware_executed": False, "visual_status": "not_viewed"}
    write_json(args.out / "report.json", report)
    finalize_examples(args)
    print({name: results[name]["pooled"] for name in METHODS}, flush=True)
    return report


def finalize_examples(args):
    """Recover presentation only after fitting, without another OOF evaluation."""
    report = read_json(args.out / "report.json")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    predictions = read_jsonl(args.out / "oof_predictions.jsonl")
    write_examples(args.out, rows, predictions, report, args.image_pack)
    write_json(args.out / "completed.json", {"schema": SCHEMA, "complete": True,
               "samples": len(rows), "methods": list(METHODS), "folds_per_method": report["episodes"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    parser.add_argument("--stage", choices=("prepare", "evaluate", "all", "examples"), default="prepare")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    if args.stage == "examples":
        finalize_examples(args)


if __name__ == "__main__":
    main()
