"""Fixed three-arm, post-action response diagnostic; no policy/hardware changes.

Local --stage crop reads original RGB and writes a small immutable ROI image pack.
After sequential SCP, remote --stage all reuses frozen baseline rows/full features,
encodes only ROI images, then fits the same ridge with fixed per-modality scaling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import os
from pathlib import Path
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_baseline import (
    ALPHA, SEED, LABELS, LABEL_ZH, embed_images, fit_ridge, folds_for,
    image_path, metrics, predict_ridge,
)


SCHEMA = "real10_response_matched_roi_v1"
METHODS = ("controller_only", "controller_full_temporal", "controller_roi_temporal")
NAMES = {"controller_only": "控制器＋任务", "controller_full_temporal": "控制器＋任务＋全图时序",
         "controller_roi_temporal": "控制器＋任务＋ROI时序"}
# [left, top, right, bottom], exclusive right/bottom, ORIGINAL 1920x1080 pixels.
# One unlabeled scene-setup pair, chosen before feature extraction/OOF evaluation.
# No window-specific tracking, tip masks, task-specific crops, or label-based edits.
ROI = {"side": [0, 540, 1440, 1040], "top": [480, 560, 1920, 1080]}
CALIBRATION_FRAME = "real_pilot_20260817_left_s_bend_replacement_tip_fixedudp_002/000000"
BLOCKS = {"controller_only": [40], "controller_full_temporal": [40, 3072],
          "controller_roi_temporal": [40, 3072]}


def baseline_snapshot(args):
    protocol = read_json(args.baseline / "protocol.json")
    rows = read_jsonl(args.baseline / "annotation_snapshot.jsonl")
    folds = read_json(args.baseline / "folds.json")
    if protocol["schema"] != "real10_post_action_response_baseline_v1" or folds != folds_for(rows):
        raise ValueError("expected the completed baseline's immutable episode folds")
    if protocol["ridge_alpha"] != ALPHA or protocol["seed"] != SEED:
        raise ValueError("baseline classifier/seed changed")
    return protocol, rows, folds


def crop_and_resize(bgr, view):
    import cv2
    if bgr is None or bgr.shape != (1080, 1920, 3):
        raise ValueError("ROI source must be original 1920x1080 RGB, not a resized pack")
    left, top, right, bottom = ROI[view]
    return cv2.resize(bgr[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_AREA)


def render_crop_preview(args, observations):
    """A data-preprocessing contact sheet, not a generated/retouched scene."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 20)
    title_font = ImageFont.truetype(str(font_path), 27)
    sheet = Image.new("RGB", (1280, 1070), "#f5f6f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), "固定ROI：先裁原图，再缩小到224像素", font=title_font, fill="#202939")
    draw.text((24, 62), "仅按场景首帧确定；所有轨迹共用，不追踪导丝，不参考人工响应或模型分数。", font=font, fill="#202939")
    obs = observations[CALIBRATION_FRAME]
    for row, (view, title) in enumerate((("side", "侧视 Side"), ("top", "俯视 Top"))):
        y = 110 + 465 * row
        bgr = cv2.imread(str(args.source_root / obs["images"][view]["path"]))
        roi = crop_and_resize(bgr, view)
        draw.text((24, y), f"{title}  原图裁剪框 {ROI[view]}", font=font, fill="#202939")
        original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((768, 432))
        box_draw = ImageDraw.Draw(original)
        box_draw.rectangle(tuple(round(v * .4) for v in ROI[view]), outline="#ff593b", width=3)
        sheet.paste(original, (24, y + 30))
        full = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        for x, img, label in ((804, full, "原有全图输入"), (1040, roi, "固定ROI输入")):
            sheet.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x, y + 70))
            draw.text((x, y + 310), label, font=font, fill="#202939")
    sheet.save(args.roi_pack / "crop_preview.jpg", quality=92)


def crop(args):
    import cv2
    started = time.perf_counter()
    old, rows, folds = baseline_snapshot(args)
    marker = args.roi_pack / "crop_completed.json"
    if marker.exists():
        if read_json(args.roi_pack / "protocol.json")["roi_xyxy_original"] != ROI:
            raise ValueError("existing crop pack uses another fixed ROI")
        print("reuse completed immutable crop pack", flush=True)
        return read_json(marker)
    args.roi_pack.mkdir(parents=True, exist_ok=True)
    if any(args.roi_pack.iterdir()):
        raise FileExistsError("preserve incomplete crop output; use a NEW --roi-pack")
    protocol = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "offline_post_action_response_recognition", "baseline": str(args.baseline),
        "samples": len(rows), "episodes": len(folds), "methods": list(METHODS),
        "annotation_authority": "exact frozen v1 baseline snapshot, NOT latest live revisions",
        "split": old["split"], "original_PI05_split_changed": False,
        "class_counts": old["class_counts"], "label_order": list(LABELS),
        "seed": SEED, "ridge_alpha": ALPHA, "score_threshold": .5,
        "encoder": "same frozen ImageNet ResNet18, no new weights or training",
        "common_input": "identical 40-d controller/Elite/task block over identical response interval",
        "visual_temporal": old["visual_temporal"], "feature_blocks": BLOCKS,
        "normalization": "train-fold mean/std/missing bits, each modality divided by sqrt(2*raw_block_dimension); fixed equal block scale, no searched weights",
        "normalization_reason": "adding 3072 visual dimensions must not dilute the same 40 controller dimensions; controller arm exactly reproduces v1",
        "roi_xyxy_original": ROI, "roi_coordinate_convention": "xyxy exclusive right/bottom",
        "source_image_hw": [1080, 1920], "model_image_hw": [224, 224],
        "resize": "original BGR crop -> cv2.INTER_AREA 224 square -> PNG -> RGB/ImageNet mean/std",
        "calibration_frame": CALIBRATION_FRAME,
        "roi_selection": "single unlabeled first-frame scene pair, before this experiment; one fixed rectangle per camera for ALL episodes/tasks; no score-driven retries",
        "calibration_limitation": "scene layout comes from this same capture batch (left002 first frame); no heldout response labels used for ROI, but not an unseen-camera calibration test",
        "full_image_features": "reuse existing frozen dual_temporal feature cache without re-encoding",
        "forbidden_inputs": old["forbidden_inputs"], "future_images": old["future_images"],
        "metrics": old["metrics"], "tuning": "none; one predeclared ROI and three methods",
        "interpretation": "exploratory comparison on already-inspected 45 weak labels; a gain needs new evidence, not OOF-based deployment/model selection",
        "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
    }
    # Lock the protocol BEFORE even creating the selected-window crops.
    write_json(args.roi_pack / "protocol.json", protocol)
    write_jsonl(args.roi_pack / "annotation_snapshot.jsonl", rows)
    write_json(args.roi_pack / "folds.json", folds)
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    keys = sorted({(f, view) for r in rows for f in [r["anchor_frame_id"], *r["future_frame_ids"]]
                   for view in ("side", "top")})
    index = []
    for i, (frame, view) in enumerate(keys):
        source = observations[frame]["images"][view]["path"]
        rgb = cv2.imread(str(args.source_root / source))
        resized = crop_and_resize(rgb, view)
        path = image_path(args.roi_pack, frame, view)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"could not write crop: {path}")
        index.append({"frame_id": frame, "view": view, "original_path": source,
                      "derived_path": path.relative_to(args.roi_pack).as_posix(), "bytes": path.stat().st_size})
        if (i + 1) % 100 == 0 or i + 1 == len(keys):
            print(f"cropped {i + 1}/{len(keys)} original images", flush=True)
    write_jsonl(args.roi_pack / "crop_index.jsonl", index)
    render_crop_preview(args, observations)
    result = {"schema": SCHEMA, "images": len(index), "image_bytes": sum(r["bytes"] for r in index),
              "crop_seconds": time.perf_counter() - started, "source_images_modified": False,
              "recognizer_fit_executed": False, "visual_status": "not_viewed"}
    write_json(marker, result)
    print(result, flush=True)
    return result


def temporal_features(rows, embeddings):
    vectors = []
    for row in rows:
        parts = []
        for view in ("side", "top"):
            anchor = embeddings[(row["anchor_frame_id"], view)]
            future = np.stack([embeddings[(frame, view)] for frame in row["future_frame_ids"]])
            parts.extend((anchor, future.mean(axis=0) - anchor, future[-1] - anchor))
        vectors.append(np.concatenate(parts))
    return np.stack(vectors)


def prepare(args):
    marker = args.out / "prepared.json"
    if marker.exists():
        print("reuse frozen preparation", flush=True)
        return read_json(marker)
    old, rows, folds = baseline_snapshot(args)
    protocol = read_json(args.roi_pack / "protocol.json")
    crop_info = read_json(args.roi_pack / "crop_completed.json")
    if protocol["schema"] != SCHEMA or protocol["roi_xyxy_original"] != ROI:
        raise ValueError("unexpected crop protocol")
    if rows != read_jsonl(args.roi_pack / "annotation_snapshot.jsonl") or folds != read_json(args.roi_pack / "folds.json"):
        raise ValueError("ROI and full-frame arms must have exactly the same frozen rows/folds")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("preserve incomplete output; use a NEW --out")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "protocol.json", protocol)
    write_jsonl(args.out / "annotation_snapshot.jsonl", rows)
    write_json(args.out / "folds.json", folds)
    started = time.perf_counter()
    with np.load(args.baseline / "features.npz", allow_pickle=False) as cached:
        control, full, y = cached["controller_only"], cached["dual_temporal"], cached["y"]
    expected = np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    if not np.array_equal(y, expected) or control.shape != (len(rows), 40) or full.shape != (len(rows), 3072):
        raise ValueError("baseline frozen features/labels mismatch")
    old_info = read_json(args.baseline / "prepared.json")["encoder"]
    embeddings, info = embed_images(rows, args.roi_pack, args.device, args.batch_size)
    if any(info[key] != old_info[key] for key in ("encoder", "weights_url", "versions", "device")):
        raise ValueError("encoder/runtime differs from cached full-frame experiment")
    info["additional_crop_or_enhancement"] = "fixed original-resolution ROI only, before embed_images"
    info["preprocessing"] = protocol["resize"]
    roi = temporal_features(rows, embeddings)
    if not np.isfinite(roi).all() or np.array_equal(roi, full):
        raise ValueError("ROI representation is nonfinite or identical to full-frame features")
    arrays = {"controller_only": control, "controller_full_temporal": np.concatenate((control, full), axis=1),
              "controller_roi_temporal": np.concatenate((control, roi), axis=1), "y": y}
    np.savez_compressed(args.out / "features.npz", **arrays)
    result = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds), "encoder": info,
              "feature_dimensions": {m: arrays[m].shape[1] for m in METHODS}, "class_counts": old["class_counts"],
              "shared_controller_task_features_exactly_equal": True, "roi_images": crop_info["images"],
              "prepare_seconds": time.perf_counter() - started, "recognizer_fit_executed": False}
    write_json(marker, result)
    return result


def evaluate(args):
    if (args.out / "report.json").exists():
        raise FileExistsError("report already exists; inspect, do not repeat or tune")
    prepared = read_json(args.out / "prepared.json")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    folds = read_json(args.out / "folds.json")
    protocol = read_json(args.out / "protocol.json")
    if prepared["schema"] != SCHEMA or protocol["feature_blocks"] != BLOCKS or folds != folds_for(rows):
        raise ValueError("frozen protocol/folds mismatch")
    with np.load(args.out / "features.npz", allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in (*METHODS, "y")}
    y = arrays["y"]
    if not np.array_equal(y, [LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows]):
        raise ValueError("frozen labels mismatch")
    for method in METHODS[1:]:
        if not np.array_equal(arrays[method][:, :40], arrays["controller_only"], equal_nan=True):
            raise ValueError("common controller/task inputs differ between methods")
    baseline_predictions = {r["window_id"]: r for r in read_jsonl(args.baseline / "oof_predictions.jsonl")
                            if r["method"] == "controller_only"}
    started = time.perf_counter()
    predictions, results, scores_by_method, timings = [], {}, {}, {}
    for method in METHODS:
        method_start = time.perf_counter()
        scores = np.full(len(rows), np.nan)
        fold_metrics = []
        for fold_index, fold in enumerate(folds):
            train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
            model = fit_ridge(arrays[method][train], y[train], block_sizes=BLOCKS[method])
            scores[test] = predict_ridge(model, arrays[method][test])
            folder = args.out / "fold_models" / method
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"fold_{fold_index:02d}.npz", **model)
            fold_metrics.append({"episode": fold["heldout_episode"], "train_n": len(train),
                                 **metrics(y[test], (scores[test] >= .5).astype(int))})
        if not np.isfinite(scores).all():
            raise ValueError("incomplete/nonfinite OOF predictions")
        if method == "controller_only":
            old_scores = [baseline_predictions[r["window_id"]]["score_not_probability"] for r in rows]
            np.testing.assert_allclose(scores, old_scores, rtol=0, atol=1e-10)
        predicted = (scores >= .5).astype(int)
        by_task = {}
        for task in ("left", "right"):
            selected = np.array([i for i, r in enumerate(rows) if r["task"] == task])
            by_task[task] = metrics(y[selected], predicted[selected])
        results[method] = {"pooled": metrics(y, predicted), "by_task": by_task, "episodes": fold_metrics,
                           "episode_macro_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics]))}
        scores_by_method[method] = predicted
        timings[method] = time.perf_counter() - method_start
        for i, row in enumerate(rows):
            predictions.append({"window_id": row["window_id"], "ui_index": row["ui_index"], "task": row["task"],
                                "heldout_episode": row["source_episode"], "method": method,
                                "target": LABELS[y[i]], "prediction": LABELS[predicted[i]],
                                "score_not_probability": float(scores[i]), "correct": bool(predicted[i] == y[i])})
    comparisons = {}
    for reference, candidate in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
        ref_ok, cand_ok = scores_by_method[reference] == y, scores_by_method[candidate] == y
        differences = [c["accuracy"] - r["accuracy"] for r, c in zip(results[reference]["episodes"], results[candidate]["episodes"])]
        comparisons[f"{candidate}_minus_{reference}"] = {
            "accuracy_pp": 100 * (results[candidate]["pooled"]["accuracy"] - results[reference]["pooled"]["accuracy"]),
            "balanced_accuracy_pp": 100 * (results[candidate]["pooled"]["balanced_accuracy"] - results[reference]["pooled"]["balanced_accuracy"]),
            "helped_windows": int(np.sum(cand_ok & ~ref_ok)), "hurt_windows": int(np.sum(ref_ok & ~cand_ok)),
            "episodes_improved_tied_worse": [sum(d > 0 for d in differences), sum(d == 0 for d in differences), sum(d < 0 for d in differences)]}
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    report = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds), "label_order": list(LABELS),
              "methods": results, "comparisons": comparisons, "baseline_scores_reproduced_atol": 1e-10,
              "timing": {"prepare_seconds": prepared["prepare_seconds"], "fit_evaluate_seconds": time.perf_counter() - started,
                         "method_seconds_including_save": timings},
              "evidence": "one fixed ROI, same 45 previously inspected windows, not an independent final test or stable-gain claim",
              "recognizer_fit_executed": True, "PI05_training_executed": False, "hardware_executed": False,
              "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
              "visual_status": "not_viewed"}
    write_json(args.out / "report.json", report)
    examples(args)
    print({m: results[m]["pooled"] for m in METHODS}, flush=True)
    print(comparisons, flush=True)
    return report


def examples(args):
    report = read_json(args.out / "report.json")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    predictions = read_jsonl(args.out / "oof_predictions.jsonl")
    lookup = {(r["window_id"], r["method"]): r for r in predictions}
    def relative(path):
        return html.escape(Path(os.path.relpath(path, args.out)).as_posix(), quote=True)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定ROI响应识别对照</title>',
             '<style>body{font:16px "Microsoft YaHei",sans-serif;max-width:1180px;margin:24px auto;background:#f5f6f8;color:#202939}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #ccd3dc}section{background:white;padding:18px;margin:20px 0}.frames{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}img.frame{width:224px;height:224px}h2{font-size:20px}</style>',
             '<h1>固定ROI：三组动作后响应识别对照</h1><p>相同控制器／Elite／任务输入，45个人工弱标注窗口，10折整轨迹留一。不是接触识别、动作前预测或独立最终测试。</p>',
             '<table><tr><th>方法</th><th>准确率</th><th>平衡准确率</th><th>左任务平衡准确率</th><th>右任务平衡准确率</th><th>轨迹宏平均准确率</th></tr>']
    for method in METHODS:
        m = report["methods"][method]
        values = [m["pooled"]["accuracy"], m["pooled"]["balanced_accuracy"], m["by_task"]["left"]["balanced_accuracy"],
                  m["by_task"]["right"]["balanced_accuracy"], m["episode_macro_accuracy"]]
        parts.append('<tr><td>' + NAMES[method] + '</td>' + ''.join(f'<td>{v:.2%}</td>' for v in values) + '</tr>')
    parts.append(f'</table><h2>评估前锁定的裁剪范围</h2><img style="max-width:100%" src="{relative(args.roi_pack / "crop_preview.jpg")}">')
    # Deterministic first case in each true/predicted ROI cell, not cherry-picked wins.
    for truth in LABELS:
        for predicted in LABELS:
            selected = [r for r in rows if r["human_annotation"]["joint_motion_response"] == truth
                        and lookup[(r["window_id"], METHODS[2])]["prediction"] == predicted]
            if not selected:
                continue
            row = selected[0]
            parts.append(f'<section><h2>窗口{row["ui_index"]}：人工 {LABEL_ZH[truth]} ／ ROI预测 {LABEL_ZH[predicted]}</h2>'
                         f'<p>{html.escape(row["source_episode"])}</p>')
            for image_pack, label in ((args.image_pack, "全图"), (args.roi_pack, "ROI")):
                parts.append(f'<p>{label}：锚点与同一窗口末帧</p><div class="frames">')
                for frame, when in ((row["anchor_frame_id"], "锚点"), (row["future_frame_ids"][-1], "末帧")):
                    for view in ("side", "top"):
                        src = relative(image_path(image_pack, frame, view))
                        parts.append(f'<figure><img class="frame" src="{src}"><figcaption>{when} · {view}</figcaption></figure>')
                parts.append('</div>')
            parts.append('<p>' + '；'.join(f'{NAMES[m]}：{LABEL_ZH[lookup[(row["window_id"], m)]["prediction"]]}' for m in METHODS) + '</p></section>')
    parts.append('</html>')
    (args.out / "examples.html").write_text('\n'.join(parts), encoding="utf-8")
    write_json(args.out / "completed.json", {"schema": SCHEMA, "complete": True, "methods": list(METHODS),
                                             "samples": len(rows), "folds_per_method": report["episodes"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("crop", "prepare", "evaluate", "all", "examples"), required=True)
    p.add_argument("--baseline", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--source-root", type=Path, default=Path("collected_data"))
    p.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    p.add_argument("--roi-pack", type=Path, default=Path("simulation_output/real10_response_roi_images_v1"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_response_matched_roi_v1"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    if args.stage == "crop":
        crop(args)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    if args.stage == "examples":
        examples(args)


if __name__ == "__main__":
    main()
