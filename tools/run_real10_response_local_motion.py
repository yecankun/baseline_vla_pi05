"""Frozen-window, post-action local-motion diagnostic; never a policy/contact input.

Prepare on Windows from original RGB; transfer the small numerical cache and fit
on project4090. No new encoder, annotation edits, threshold search or hardware.
The vessel/color and dark-line masks are observable heuristics, NOT wire truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import html
from pathlib import Path
import time

import cv2
import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_baseline import ALPHA, SEED, LABELS, fit_ridge, folds_for, metrics, predict_ridge
from run_real10_response_roi_comparison import CALIBRATION_FRAME, ROI


SCHEMA = "real10_vessel_relative_local_motion_v1"
VIEWS = ("side", "top")
GRID = (2, 6)  # rows, columns; fixed scene coordinates, not a tip/route tracker
WIDTH = 720
STATS = ("dx_px_s", "dy_px_s", "speed_mean_px_s", "speed_p90_px_s",
         "ridge_change_s", "ridge_abs_change_s")
FEATURE_NAMES = [f"{view}_{pool}_r{r}c{c}_{stat}" for view in VIEWS
                 for pool in ("mean", "std") for r in range(GRID[0])
                 for c in range(GRID[1]) for stat in STATS]
METHODS = ("controller_only", "local_motion_only", "controller_local_motion")
NAMES = {"controller_only": "控制器＋任务", "local_motion_only": "仅局部运动",
         "controller_local_motion": "控制器＋任务＋局部运动"}
BLOCKS = {"controller_only": [40], "local_motion_only": [len(FEATURE_NAMES)],
          "controller_local_motion": [40, len(FEATURE_NAMES)]}
PREVIEW_WINDOWS = (1, 59, 65, 81)  # a first left window, right control and prior audited errors
PARAMETERS = {
    "roi_xyxy_original": ROI, "width": WIDTH, "preserve_aspect_ratio": True,
    "vessel_hsv_low": [5, 45, 40], "vessel_hsv_high": [40, 255, 255],
    "vessel_close_kernel": 9, "vessel_open_kernel": 3, "min_component_area_fraction": .01,
    "interior_erosion_kernel": 5, "reference_rim_erosion_kernel": 13,
    "blackhat_kernel": 9, "dark_line_min_gray": 6, "min_cell_line_pixels": 8,
    "lk_max_corners": 300, "lk_quality": .01, "lk_min_distance": 7,
    "lk_window": 21, "lk_levels": 3, "lk_fb_max_px": 1.0, "min_reference_tracks": 8,
    "max_reference_translation_px": 12.0,
    "farneback": {"pyr_scale": .5, "levels": 3, "winsize": 15, "iterations": 3,
                  "poly_n": 5, "poly_sigma": 1.2, "flags": 0},
    "grid_rows_columns": list(GRID), "temporal_pool": ["duration_weighted_mean", "duration_weighted_std"],
}


def ellipse(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def crop_original(bgr, view):
    if bgr is None or bgr.shape != (1080, 1920, 3):
        raise ValueError("local motion needs the original 1920x1080 image")
    x0, y0, x1, y1 = ROI[view]
    height = round((y1 - y0) * WIDTH / (x1 - x0))
    return cv2.resize(bgr[y0:y1, x0:x1], (WIDTH, height), interpolation=cv2.INTER_AREA)


def vessel_masks(calibration_bgr):
    """One unlabeled scene calibration; no task, response label or wire annotation."""
    p = PARAMETERS
    mask = cv2.inRange(cv2.cvtColor(calibration_bgr, cv2.COLOR_BGR2HSV),
                       np.array(p["vessel_hsv_low"]), np.array(p["vessel_hsv_high"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(p["vessel_close_kernel"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(p["vessel_open_kernel"]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= mask.size * p["min_component_area_fraction"]
    mask = (keep[labels] * 255).astype(np.uint8)
    interior = cv2.erode(mask, ellipse(p["interior_erosion_kernel"])) > 0
    rim = (mask > 0) & (cv2.erode(mask, ellipse(p["reference_rim_erosion_kernel"])) == 0)
    if interior.sum() < 100 or rim.sum() < 100:
        raise ValueError("calibration color mask is insufficient; inspect without silently widening it")
    return interior, rim


def pair_motion(previous, current, interior, rim, dt):
    """Track vessel-rim texture, subtract its robust TRANSLATION from local flow.

    This is not full vessel deformation compensation, semantic wire segmentation,
    physical tip displacement, or calibrated optical-flow confidence. Failures
    remain missing, not falsely stationary. No labels/state enter this function.
    """
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("image timestamps must strictly increase within each view")
    p = PARAMETERS
    diagnostic = {"dt_s": float(dt), "valid": False, "reference_tracks": 0}
    missing = np.full((GRID[0] * GRID[1], len(STATS)), np.nan)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=p["lk_max_corners"],
        qualityLevel=p["lk_quality"], minDistance=p["lk_min_distance"],
        mask=rim.astype(np.uint8) * 255, blockSize=7)
    if points is None:
        return missing, diagnostic, None
    lk = {"winSize": (p["lk_window"], p["lk_window"]), "maxLevel": p["lk_levels"],
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk)
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk)
    fb = np.linalg.norm(backward[:, 0] - points[:, 0], axis=1)
    valid = (status[:, 0] != 0) & (reverse_status[:, 0] != 0) & (fb <= p["lk_fb_max_px"])
    diagnostic["reference_tracks"] = int(valid.sum())
    if valid.sum() < p["min_reference_tracks"]:
        return missing, diagnostic, None
    shift = np.median((forward - points)[valid, 0], axis=0)
    diagnostic["reference_translation_px"] = shift.tolist()
    diagnostic["median_reference_fb_error_px"] = float(np.median(fb[valid]))
    if np.linalg.norm(shift) > p["max_reference_translation_px"]:
        return missing, diagnostic, None
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **p["farneback"])
    residual = flow - shift
    h, w = previous.shape
    # forward flow goes prev->current; sampling current at (x+shift) aligns it to prev.
    y, x = np.mgrid[:h, :w].astype(np.float32)
    aligned = cv2.remap(current, x + shift[0], y + shift[1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    a = cv2.morphologyEx(previous, cv2.MORPH_BLACKHAT, ellipse(p["blackhat_kernel"]))
    b = cv2.morphologyEx(aligned, cv2.MORPH_BLACKHAT, ellipse(p["blackhat_kernel"]))
    support = interior & (np.maximum(a, b) >= p["dark_line_min_gray"])
    in_bounds = (x + shift[0] >= 0) & (x + shift[0] < w - 1) & (y + shift[1] >= 0) & (y + shift[1] < h - 1)
    support &= in_bounds
    ridge_delta = (b.astype(np.float32) - a.astype(np.float32)) / 255.0
    speed = np.linalg.norm(residual, axis=2)
    vectors, counts = [], []
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            region = np.s_[r * h // GRID[0]:(r + 1) * h // GRID[0],
                           c * w // GRID[1]:(c + 1) * w // GRID[1]]
            selected = support[region]
            counts.append(int(selected.sum()))
            if selected.sum() < p["min_cell_line_pixels"]:
                vectors.append([np.nan] * len(STATS))
                continue
            f = residual[region][selected]
            s = speed[region][selected]
            d = ridge_delta[region][selected]
            vectors.append([f[:, 0].mean() / dt, f[:, 1].mean() / dt, s.mean() / dt,
                            np.quantile(s, .9) / dt, d.mean() / dt, np.abs(d).mean() / dt])
    diagnostic.update(valid=True, line_pixels_by_cell=counts,
                      line_fraction_in_interior=float(support.sum() / interior.sum()))
    # Quality/proxy support is saved separately, never added to learned features.
    visual = {"residual_px": residual.astype(np.float32), "support": support,
              "ridge_change": ridge_delta}
    return np.array(vectors, dtype=np.float64), diagnostic, visual


def pool_pairs(values, dt):
    a = np.stack(values)
    weights = np.asarray(dt)[:, None, None] * np.isfinite(a)
    denominator = weights.sum(axis=0)
    mean = np.divide((np.nan_to_num(a) * weights).sum(axis=0), denominator,
                     out=np.full(a.shape[1:], np.nan), where=denominator > 0)
    variance = np.divide((np.nan_to_num(a - mean) ** 2 * weights).sum(axis=0), denominator,
                         out=np.full(a.shape[1:], np.nan), where=denominator > 0)
    return np.concatenate((mean.ravel(), np.sqrt(variance).ravel()))


def check():
    """One targeted numerical check for compensation sign and actual local changes."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(140, 200, (180, 360), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    interior = np.zeros(a.shape, bool)
    interior[55:125, 20:340] = True
    rim = np.zeros_like(interior)
    rim[35:50, 20:340] = True
    rim[130:145, 20:340] = True
    cv2.line(a, (100, 90), (180, 90), 30, 3)
    same, _, _ = pair_motion(a, a, interior, rim, .2)
    moved = cv2.warpAffine(a, np.float32([[1, 0, 3], [0, 1, 2]]), (360, 180), borderMode=cv2.BORDER_REFLECT_101)
    common, d, _ = pair_motion(a, moved, interior, rim, .2)
    local = a.copy()
    cv2.line(local, (181, 90), (196, 90), 30, 3)
    changed, _, _ = pair_motion(a, local, interior, rim, .2)
    np.testing.assert_allclose(d["reference_translation_px"], [3, 2], atol=.15)
    assert np.nanmean(same[:, 2]) < .01, "identical images must have negligible motion"
    assert np.nanmean(common[:, 5]) < np.nanmean(changed[:, 5]) * .25, "common translation compensation failed"
    assert np.nanmean(changed[:, 2]) > np.nanmean(same[:, 2]) + .1, "local movement was lost"
    result = {"identical_mean_speed": float(np.nanmean(same[:, 2])),
              "common_translation_estimate": d["reference_translation_px"],
              "common_mean_ridge_change": float(np.nanmean(common[:, 5])),
              "local_mean_ridge_change": float(np.nanmean(changed[:, 5])),
              "local_mean_speed": float(np.nanmean(changed[:, 2])), "opencv": cv2.__version__}
    print(result, flush=True)
    return result


def prepare(args):
    started = time.perf_counter()
    rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    folds = read_json(args.reference / "folds.json")
    old = read_json(args.reference / "protocol.json")
    if folds != folds_for(rows) or old["ridge_alpha"] != ALPHA or old["seed"] != SEED:
        raise ValueError("frozen rows/folds/classifier changed")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("preserve existing artifacts; use a new --out for incomplete work")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": PARAMETERS, "calibration_frame": CALIBRATION_FRAME,
        "calibration": "same unlabeled first frame as fixed ROI; one color-support mask per camera, all episodes/tasks",
        "samples": len(rows), "episodes": len(folds), "methods": list(METHODS), "feature_blocks": BLOCKS,
        "feature_names": FEATURE_NAMES, "ridge_alpha": ALPHA, "seed": SEED, "threshold": .5,
        "split": "same frozen leave-one-whole-source-episode-out folds", "labels_changed": False,
        "original_PI05_split_changed": False, "reference": str(args.reference),
        "representation": "vessel-rim LK translation reference + dark-line-supported residual dense flow and ridge change in a fixed 2x6 grid; no absolute appearance embedding",
        "units": "motion in resized-ROI pixels/second, not mm or reconstructed tip displacement; appearance change per second",
        "normalization": "existing train-fold-only mean/std/missing indicators, equal per-block scaling",
        "missing": "failed reference tracking or <8 dark-line pixels per cell -> NaN; train-fold imputation + missing bits; never force stationary",
        "model_inputs": "motion arms use only each view's ordered RGB and relative image timestamps; fusion appends unchanged 40-d controller/Elite/task block",
        "no_model_inputs": ["human visibility/response/evidence", "window/episode IDs", "mask quality summaries",
                            "absolute timestamps", "tip/contact/wall/route truth", "diagnostic_targets"],
        "future_frames": "post-action recognition only; cannot feed back into a decision made before the response window ends",
        "limitations": ["heuristic mask may include texture/reflections or miss wire; not validated semantic wire tracking",
                        "translation-only reference, not full vessel deformation compensation",
                        "45 windows already used for development; OOF is not independent final validation",
                        "fixed-mask calibration uses the same capture batch, not unseen camera geometry",
                        "residual motion does not imply axial advance or wall contact"],
        "tuning": "none; parameters locked before extraction/evaluation; no score-driven retries",
        "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
    }
    write_json(args.out / "protocol.json", protocol)
    write_jsonl(args.out / "annotation_snapshot.jsonl", rows)
    write_json(args.out / "folds.json", folds)
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    masks, visual_arrays = {}, {}
    for view in VIEWS:
        image = crop_original(cv2.imread(str(args.source_root / observations[CALIBRATION_FRAME]["images"][view]["path"])), view)
        masks[view] = vessel_masks(image)
        visual_arrays[f"calibration_{view}"] = image
        visual_arrays[f"interior_{view}"], visual_arrays[f"rim_{view}"] = masks[view]

    @lru_cache(maxsize=64)
    def gray(frame, view):
        source = args.source_root / observations[frame]["images"][view]["path"]
        return cv2.cvtColor(crop_original(cv2.imread(str(source)), view), cv2.COLOR_BGR2GRAY)

    matrix, quality = [], []
    pair_cache = {}
    for i, row in enumerate(rows):
        frame_ids = [row["anchor_frame_id"], *row["future_frame_ids"]]
        parts, details = [], {}
        for view in VIEWS:
            values, durations, diagnostics, displays = [], [], [], []
            for a, b in zip(frame_ids, frame_ids[1:]):
                dt = observations[b]["images"][view]["recorded_timestamp_s"] - observations[a]["images"][view]["recorded_timestamp_s"]
                key = (view, a, b)
                # Cache numeric stats only; previews do not cause a different representation.
                if key not in pair_cache or row["ui_index"] in PREVIEW_WINDOWS:
                    stats, diagnostic, visual = pair_motion(gray(a, view), gray(b, view), *masks[view], dt)
                    pair_cache[key] = (stats, diagnostic)
                    if visual is not None and row["ui_index"] in PREVIEW_WINDOWS:
                        displays.append(visual)
                stats, diagnostic = pair_cache[key]
                values.append(stats)
                durations.append(dt)
                diagnostics.append({"from_frame": a, "to_frame": b, **diagnostic})
            parts.append(pool_pairs(values, durations))
            details[view] = diagnostics
            if row["ui_index"] in PREVIEW_WINDOWS:
                prefix = f"w{row['ui_index']:03d}_{view}"
                visual_arrays[prefix + "_first"] = gray(frame_ids[0], view)
                visual_arrays[prefix + "_last"] = gray(frame_ids[-1], view)
                if displays:
                    # Mean pair displacement is a visualization, not integrated tip tracking.
                    visual_arrays[prefix + "_flow"] = np.mean([v["residual_px"] for v in displays], axis=0)
                    visual_arrays[prefix + "_support"] = np.any([v["support"] for v in displays], axis=0)
        matrix.append(np.concatenate(parts))
        quality.append({"window_id": row["window_id"], "ui_index": row["ui_index"], "by_view": details})
        print(f"local motion {i + 1}/{len(rows)}: window {row['ui_index']}", flush=True)
    motion = np.stack(matrix)
    with np.load(args.reference / "features.npz", allow_pickle=False) as z:
        control, y = z["controller_only"], z["y"]
    if control.shape != (len(rows), 40) or not np.array_equal(y, [LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows]):
        raise ValueError("cached controller/label order mismatch")
    if motion.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(motion).any():
        raise ValueError("local motion extraction produced no usable features")
    np.savez_compressed(args.out / "features.npz", controller_only=control, local_motion_only=motion,
                        controller_local_motion=np.concatenate((control, motion), axis=1), y=y)
    np.savez_compressed(args.out / "preview_arrays.npz", **visual_arrays)
    write_jsonl(args.out / "motion_quality.jsonl", quality)
    all_pairs = [d for q in quality for v in VIEWS for d in q["by_view"][v]]
    result = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds),
              "feature_dimensions": {k: sum(v) for k, v in BLOCKS.items()},
              "frame_pair_instances": len(all_pairs), "valid_reference_pair_instances": sum(d["valid"] for d in all_pairs),
              "finite_motion_fraction": float(np.isfinite(motion).mean()),
              "source_images_modified": False, "labels_modified": False,
              "recognizer_fit_executed": False, "policy_input_allowed": False,
              "prepare_seconds": time.perf_counter() - started,
              "feature_runtime": {"opencv": cv2.__version__, "numpy": np.__version__, "machine": "local Windows original RGB"},
              "visual_status": "not_viewed"}
    write_json(args.out / "prepared.json", result)
    print(result, flush=True)


def evaluate(args):
    if (args.out / "report.json").exists():
        raise FileExistsError("report exists; inspect instead of repeating/tuning")
    started = time.perf_counter()
    protocol = read_json(args.out / "protocol.json")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    folds = read_json(args.out / "folds.json")
    if protocol["parameters"] != PARAMETERS or protocol["feature_names"] != FEATURE_NAMES or folds != folds_for(rows):
        raise ValueError("locked feature protocol or folds mismatch")
    if rows != read_jsonl(args.reference / "annotation_snapshot.jsonl") or folds != read_json(args.reference / "folds.json"):
        raise ValueError("comparison rows/folds changed")
    with np.load(args.out / "features.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in (*METHODS, "y")}
    y = arrays["y"]
    np.testing.assert_array_equal(y, [LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    np.testing.assert_allclose(arrays["controller_local_motion"][:, :40], arrays["controller_only"], equal_nan=True)
    np.testing.assert_allclose(arrays["controller_local_motion"][:, 40:], arrays["local_motion_only"], equal_nan=True)
    old_predictions = read_jsonl(args.reference / "oof_predictions.jsonl")
    old_control = {p["window_id"]: p["score_not_probability"] for p in old_predictions if p["method"] == "controller_only"}
    predictions, results, correctness = [], {}, {}
    for method in METHODS:
        scores = np.full(len(rows), np.nan)
        fold_results = []
        folder = args.out / "fold_models" / method
        folder.mkdir(parents=True, exist_ok=True)
        for k, fold in enumerate(folds):
            train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
            model = fit_ridge(arrays[method][train], y[train], block_sizes=BLOCKS[method])
            scores[test] = predict_ridge(model, arrays[method][test])
            np.savez_compressed(folder / f"fold_{k:02d}.npz", **model)
            fold_results.append({"episode": fold["heldout_episode"], **metrics(y[test], (scores[test] >= .5).astype(int))})
        if not np.isfinite(scores).all():
            raise ValueError("incomplete OOF predictions")
        if method == "controller_only":
            np.testing.assert_allclose(scores, [old_control[r["window_id"]] for r in rows], rtol=0, atol=1e-10)
        predicted = (scores >= .5).astype(int)
        by_task = {}
        for task in ("left", "right"):
            idx = np.array([i for i, r in enumerate(rows) if r["task"] == task])
            by_task[task] = metrics(y[idx], predicted[idx])
        results[method] = {"pooled": metrics(y, predicted), "by_task": by_task, "episodes": fold_results,
                           "episode_macro_accuracy": float(np.mean([f["accuracy"] for f in fold_results]))}
        correctness[method] = predicted == y
        for i, row in enumerate(rows):
            predictions.append({"window_id": row["window_id"], "ui_index": row["ui_index"], "task": row["task"],
                "heldout_episode": row["source_episode"], "method": method, "target": LABELS[y[i]],
                "prediction": LABELS[predicted[i]], "score_not_probability": float(scores[i]), "correct": bool(predicted[i] == y[i])})
    comparisons = {}
    for method in METHODS[1:]:
        differences = [b["accuracy"] - a["accuracy"] for a, b in zip(results[METHODS[0]]["episodes"], results[method]["episodes"])]
        comparisons[method + "_minus_controller_only"] = {
            "accuracy_pp": 100 * (results[method]["pooled"]["accuracy"] - results[METHODS[0]]["pooled"]["accuracy"]),
            "balanced_accuracy_pp": 100 * (results[method]["pooled"]["balanced_accuracy"] - results[METHODS[0]]["pooled"]["balanced_accuracy"]),
            "helped_windows": int(np.sum(correctness[method] & ~correctness[METHODS[0]])),
            "hurt_windows": int(np.sum(correctness[METHODS[0]] & ~correctness[method])),
            "episodes_improved_tied_worse": [sum(d > 0 for d in differences), sum(d == 0 for d in differences), sum(d < 0 for d in differences)]}
    reference_report = read_json(args.reference / "report.json")
    report = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds), "methods": results,
              "comparisons": comparisons, "prior_same_window_reference_methods": reference_report["methods"],
              "baseline_scores_reproduced_atol": 1e-10, "label_order": list(LABELS),
              "fit_evaluate_seconds": time.perf_counter() - started,
              "prepared": read_json(args.out / "prepared.json"), "recognizer_fit_executed": True,
              "PI05_training_executed": False, "hardware_executed": False, "policy_input_allowed": False,
              "formal_data_allowed": False, "visual_status": "not_viewed",
              "interpretation": "exploratory development OOF only; no parameter tuning, contact label, stable gain or deployment claim"}
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_json(args.out / "report.json", report)
    print({k: v["pooled"] for k, v in results.items()}, flush=True)
    print(comparisons, flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    with np.load(args.out / "preview_arrays.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    sheet = Image.new("RGB", (1500, 740), "#f4f6fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((22, 16), "局部运动参照：绿色为候选血管内部，橙色为参考纹理带", font=title, fill="#172134")
    draw.text((22, 63), "同一无标签首帧校准；颜色启发式不是导丝分割真值，不能保证排除全部反光和器械。", font=font, fill="#172134")
    for i, view in enumerate(VIEWS):
        bgr = arrays[f"calibration_{view}"]
        marked = bgr.copy()
        for mask, color in ((arrays[f"interior_{view}"], [60, 220, 60]), (arrays[f"rim_{view}"], [20, 145, 255])):
            marked[mask] = (.6 * marked[mask] + .4 * np.asarray(color)).astype(np.uint8)
        y = 110 + 300 * i
        draw.text((22, y), f"{view} 原图固定ROI（保留宽高比）", font=font, fill="#172134")
        draw.text((762, y), "固定候选区域 / 参考带", font=font, fill="#172134")
        for x, image in ((22, bgr), (762, marked)):
            sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, y + 32))
    sheet.save(args.out / "mask_preview.jpg", quality=92)
    links = ["<h2>固定校准区域</h2><img src='mask_preview.jpg'>"]
    for window in PREVIEW_WINDOWS:
        sheet = Image.new("RGB", (1500, 765), "#f4f6fa")
        draw = ImageDraw.Draw(sheet)
        draw.text((22, 15), f"第{window}窗：局部运动可视化（不是导丝轨迹或接触图）", font=title, fill="#172134")
        draw.text((22, 62), "青色标出候选暗线；红箭头为相邻帧平均残余位移×8，仅用于显示，不代表累计位移。", font=font, fill="#172134")
        for i, view in enumerate(VIEWS):
            prefix = f"w{window:03d}_{view}"
            first, last = arrays[prefix + "_first"], arrays[prefix + "_last"]
            marked = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
            if prefix + "_flow" in arrays:
                flow, support = arrays[prefix + "_flow"], arrays[prefix + "_support"]
                marked[support] = (.6 * marked[support] + .4 * np.array([255, 210, 30])).astype(np.uint8)
                h, w = first.shape
                for y in range(10, h, 20):
                    for x in range(10, w, 20):
                        area = np.s_[max(0, y-5):y+6, max(0, x-5):x+6]
                        good = support[area]
                        if good.sum() < 3:
                            continue
                        f = np.median(flow[area][good], axis=0)
                        delta = np.clip(f * 8, -35, 35)
                        if np.linalg.norm(delta) >= 1:
                            cv2.arrowedLine(marked, (x, y), (round(x + delta[0]), round(y + delta[1])), (30, 30, 255), 1, tipLength=.3)
            y = 112 + i * 310
            draw.text((22, y), f"{view} 首帧及候选运动", font=font, fill="#172134")
            draw.text((762, y), f"{view} 末帧", font=font, fill="#172134")
            sheet.paste(Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)), (22, y+34))
            sheet.paste(Image.fromarray(last).convert("RGB"), (762, y+34))
        name = f"motion_{window:03d}.jpg"
        sheet.save(args.out / name, quality=92)
        links.append(f"<h2>第{window}窗</h2><img src='{name}'>")
    report = read_json(args.out / "report.json") if (args.out / "report.json").exists() else None
    table = "<p>尚未拟合评估；当前页只显示特征预处理。</p>"
    if report:
        table = "<table><tr><th>方法</th><th>准确率</th><th>平衡准确率</th><th>左任务BA</th><th>右任务BA</th></tr>"
        for method in METHODS:
            r = report["methods"][method]
            values = [r["pooled"]["accuracy"], r["pooled"]["balanced_accuracy"], r["by_task"]["left"]["balanced_accuracy"], r["by_task"]["right"]["balanced_accuracy"]]
            table += "<tr><td>" + html.escape(NAMES[method]) + "</td>" + "".join(f"<td>{v:.2%}</td>" for v in values) + "</tr>"
        table += "</table><p>同45窗、10个整轨迹留一折；只做开发诊断，不是独立最终测试。评分不是校准概率。</p>"
    page = "<!doctype html><meta charset='utf-8'><title>局部运动诊断</title><style>body{font:18px sans-serif;margin:24px;max-width:1500px;background:#f4f6fa;color:#172134}img{max-width:100%}td,th{padding:12px;border:1px solid #aaa}table{border-collapse:collapse}</style><h1>相对血管参照的局部运动诊断</h1>"
    page += "<p>只读；不修改人工标注、不接入策略、不代表碰壁判断。暗线可能包含血管纹理；未验证可靠导丝跟踪。</p>" + table + "".join(links)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print({"page": str(args.out / "index.html"), "images": 5, "visual_status": "not_viewed"}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("check", "prepare", "evaluate", "render"), required=True)
    p.add_argument("--reference", type=Path, default=Path("simulation_output/real10_response_matched_roi_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--source-root", type=Path, default=Path("collected_data"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_response_local_motion_v1"))
    args = p.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "check":
        check()
    else:
        {"prepare": prepare, "evaluate": evaluate, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
