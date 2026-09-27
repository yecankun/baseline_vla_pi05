"""Forecast frozen local-motion proxies from unchanged pre-request inputs.

Prepare numerical supervision locally from original RGB; fit on project4090.
No new mask, manual label, PI05 change, contact input or hardware operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_local_motion import crop_original, pair_motion, PARAMETERS, GRID, VIEWS
from run_real10_response_baseline import _ridge_standardized_features
from run_real10_elite_request_forecast import ALPHA, CLIP, BLOCKS, FLAGS, win_counts


SCHEMA = "real10_elite_local_response_forecast_v1"
PROTOCOL = Path("docs/algorithm-real10-local-response-forecast-protocol-20260921.md")
ARMS = ("persistence", "train_mean", "observation_only", "observation_plus_request")
FIELDS = ("dx_px_s", "dy_px_s", "speed_mean_px_s")
TARGET_NAMES = [f"{view}_mean_r{r}c{c}_{field}" for view in VIEWS
                for r in range(GRID[0]) for c in range(GRID[1]) for field in FIELDS]
METRICS = ("vector_epe_px_s", "component_mae_px_s", "component_rmse_px_s",
           "speed_mae_px_s", "speed_rmse_px_s", "event_macro_epe_px_s")


def prepare(args):
    started = time.perf_counter()
    if args.targets.exists():
        raise FileExistsError("preserve prior target packs; choose a fresh --targets")
    source = read_json(args.source / "manifest.json")
    old = read_json(args.motion_reference / "protocol.json")
    if source["schema"] != "real10_elite_macro_request_response_v1" or old["parameters"] != PARAMETERS:
        raise ValueError("source contract or frozen motion extractor changed")
    observations = read_jsonl(args.source / "observations.jsonl")
    targets = read_jsonl(args.source / "response_targets.jsonl")
    groups = read_jsonl(args.source / "sample_groups.jsonl")
    folds = read_json(args.source / "folds.json")
    if not len(observations) == len(targets) == len(groups) == 175 or len(folds) != 10:
        raise ValueError("fixed 175 events/10 episodes required")
    for i, (obs, target, group) in enumerate(zip(observations, targets, groups)):
        if (obs["sample_index"] != i or target["sample_index"] != i or group["sample_index"] != i
                or not obs["sample_id"] == target["sample_id"] == group["sample_id"]
                or target["joint_motion_response"] is not None):
            raise ValueError("row identity/order or unlabeled target contract changed")

    args.targets.mkdir(parents=True)
    shutil.copyfile(PROTOCOL, args.targets / "fixed_protocol.md")
    shutil.copyfile(__file__, args.targets / "entrypoint_snapshot.py")
    for name in ("manifest.json", "observations.jsonl", "requests.jsonl", "response_targets.jsonl",
                 "sample_groups.jsonl", "folds.json", "timing_audit.jsonl"):
        shutil.copyfile(args.source / name, args.targets / name)
    protocol = {"schema": SCHEMA, "source": args.source.as_posix(), "motion_reference": args.motion_reference.as_posix(),
        "fixed_protocol": PROTOCOL.as_posix(), "parameters": PARAMETERS, "target_names": TARGET_NAMES,
        "calibration_frame": old["calibration_frame"], "new_candidate_region_applied": False,
        "target_validity": "all adjacent pairs valid in this cell; no missing-target imputation",
        "target_units": "resized ROI pixels/second; not mm, axial wire advance or contact",
        "inputs": "unchanged prior 2065-d pre-request caches; future motion/quality never as inputs",
        "arms": list(ARMS), "ridge_alpha": ALPHA, "standardized_clip": CLIP, "feature_blocks": BLOCKS,
        "labels_read_or_changed": False, "source_images_changed": False, **FLAGS}
    write_json(args.targets / "protocol.json", protocol)

    with np.load(args.motion_reference / "preview_arrays.npz", allow_pickle=False) as saved:
        masks = {view: (saved[f"interior_{view}"].copy(), saved[f"rim_{view}"].copy()) for view in VIEWS}
        calibration = {f"calibration_{view}": saved[f"calibration_{view}"].copy() for view in VIEWS}
    preview_indices = []
    for fold in folds:
        indices = [i for i, row in enumerate(groups) if row["source_episode"] == fold["heldout_episode"]]
        preview_indices.append(indices[len(indices) // 2])
    preview = {**calibration, **{f"interior_{v}": masks[v][0] for v in VIEWS},
               **{f"rim_{v}": masks[v][1] for v in VIEWS}}

    def raw_path(reference, episode, view):
        path = Path(reference["path"])
        if path.parent.name != view or path.parents[1].name != episode:
            raise ValueError("image path identity does not match event group")
        return args.raw_root / episode / "frames" / view / path.name

    @lru_cache(maxsize=64)
    def gray(path, view):
        return cv2.cvtColor(crop_original(cv2.imread(str(path)), view), cv2.COLOR_BGR2GRAY)

    values, fractions, quality, endpoint_times = [], [], [], []
    pair_cache = {}
    for i, (obs, target, group) in enumerate(zip(observations, targets, groups)):
        frames = [obs["model_observation"]["history"][-1], *target["future_observations"]]
        parts, coverage, by_view, spans = [], [], {}, {}
        for view in VIEWS:
            refs = [frame["images"][view] for frame in frames]
            paths = [raw_path(ref, group["source_episode"], view) for ref in refs]
            stats, durations, details = [], [], []
            for a, b, pa, pb in zip(refs, refs[1:], paths, paths[1:]):
                dt = b["relative_time_s"] - a["relative_time_s"]
                key = (view, str(pa), str(pb), dt)
                if key not in pair_cache:
                    matrix, diagnostic, _ = pair_motion(gray(pa, view), gray(pb, view), *masks[view], dt)
                    pair_cache[key] = (matrix[:, :3], diagnostic)
                matrix, diagnostic = pair_cache[key]
                stats.append(matrix)
                durations.append(dt)
                details.append({"from_frame": pa.name, "to_frame": pb.name, **diagnostic})
            a = np.stack(stats)
            weight = np.asarray(durations)[:, None, None]
            finite = np.isfinite(a)
            full = np.all(finite, axis=0)
            mean = np.sum(np.where(finite, a, 0) * weight, axis=0) / np.sum(weight)
            mean[~full] = np.nan
            parts.append(mean.ravel())
            coverage.append((np.sum(finite * weight, axis=0) / np.sum(weight)).ravel())
            by_view[view] = details
            spans[view] = {"anchor_relative_time_s": refs[0]["relative_time_s"],
                           "last_relative_time_s": refs[-1]["relative_time_s"],
                           "observed_span_s": sum(durations)}
            if i in preview_indices:
                preview[f"s{i:03d}_{view}_anchor"] = gray(paths[0], view)
                preview[f"s{i:03d}_{view}_last"] = gray(paths[-1], view)
        values.append(np.concatenate(parts))
        fractions.append(np.concatenate(coverage))
        quality.append({"sample_index": i, "sample_id": group["sample_id"], "by_view": by_view})
        endpoint_times.append({"sample_index": i, "sample_id": group["sample_id"], "by_view": spans})
        if (i + 1) % 10 == 0 or i + 1 == len(groups):
            print(f"local supervision {i + 1}/{len(groups)} ({time.perf_counter() - started:.1f}s)", flush=True)
    y, coverage = np.stack(values), np.stack(fractions)
    if y.shape != (175, 72) or not np.isfinite(y).any():
        raise ValueError("no usable local targets or wrong layout")
    np.savez_compressed(args.targets / "local_targets.npz", targets=y, valid=np.isfinite(y), valid_time_fraction=coverage)
    np.savez_compressed(args.targets / "preview_arrays.npz", **preview)
    write_json(args.targets / "preview_indices.json", preview_indices)
    write_jsonl(args.targets / "motion_quality.jsonl", quality)
    write_jsonl(args.targets / "observed_times.jsonl", endpoint_times)
    pairs = [d for row in quality for view in VIEWS for d in row["by_view"][view]]
    valid = np.isfinite(y).reshape(175, 2, 12, 3).all(axis=-1)
    prepared = {"schema": SCHEMA, "samples": 175, "episodes": 10, "target_shape": list(y.shape),
        "frame_pair_instances": len(pairs), "unique_pairs_computed": len(pair_cache),
        "valid_reference_pair_instances": sum(d["valid"] for d in pairs),
        "valid_cells": int(valid.sum()), "possible_cells": int(valid.size),
        "samples_with_any_target": int(valid.any(axis=(1, 2)).sum()),
        "by_view": {v: {"valid_cells": int(valid[:, j].sum()),
                        "samples_with_any_target": int(valid[:, j].any(axis=1).sum()),
                        "valid_windows_by_grid_cell": valid[:, j].sum(axis=0).reshape(2, 6).tolist()}
                    for j, v in enumerate(VIEWS)},
        "fully_missing_dimensions": [TARGET_NAMES[j] for j in range(72) if not np.isfinite(y[:, j]).any()],
        "partly_observed_cell_dimensions_not_filled": int(((coverage > 0) & ~np.isfinite(y)).sum()),
        "prepare_seconds": time.perf_counter() - started,
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "ridge_fit_executed": False, "visual_status": "not_viewed", **FLAGS}
    write_json(args.targets / "prepared.json", prepared)
    render_previews(args.targets)
    print(prepared, flush=True)


def transform(x, normalizer):
    return (_ridge_standardized_features(x, normalizer["mean"], normalizer["std"],
                                         normalizer["standardized_clip"]) / normalizer["feature_scale"])


def fit_masked(z, y):
    """One independent output loss per column; target NaNs are NEVER imputed."""
    coef, intercept = np.full((z.shape[1], y.shape[1]), np.nan), np.full(y.shape[1], np.nan)
    masks, group_ids = np.unique(np.isfinite(y).T, axis=0, return_inverse=True)
    log = []
    for group_id, mask in enumerate(masks):
        columns = np.flatnonzero(group_ids == group_id)
        if mask.sum() < 2:
            continue
        x0, y0 = z[mask], y[np.ix_(mask, columns)]
        center, target_center = x0.mean(axis=0), y0.mean(axis=0)
        design = x0 - center
        kernel = design @ design.T + ALPHA * np.eye(len(y0))
        dual = np.linalg.solve(kernel, y0 - target_center)
        residual = float(np.max(np.abs(kernel @ dual - (y0 - target_center))))
        group_coef = design.T @ dual
        if not np.isfinite(group_coef).all() or residual > 1e-7 * (1 + np.max(np.abs(y0))):
            raise ValueError("masked ridge solve failed")
        coef[:, columns] = group_coef
        intercept[columns] = target_center - center @ group_coef
        log.append({"target_columns": columns.tolist(), "valid_training_rows": int(mask.sum()),
                    "max_solve_residual": residual})
    return coef, intercept, log


def metric(y, prediction):
    y, prediction = y.reshape(len(y), -1, 3), prediction.reshape(len(y), -1, 3)
    valid = np.isfinite(y).all(axis=-1)
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("a supervised test cell lacks supported finite predictions")
    error = prediction - y
    epe = np.linalg.norm(error[..., :2], axis=-1)
    counts = valid.sum(axis=1)
    per_event = np.divide(np.where(valid, epe, 0).sum(axis=1), counts,
                          out=np.full(len(y), np.nan), where=counts > 0)
    result = {"n": len(y), "samples_with_valid_motion": int((counts > 0).sum()),
              "valid_cells": int(valid.sum()), "possible_cells": int(valid.size)}
    if not valid.any():
        return {**result, **{key: None for key in METRICS}}
    return {**result, "vector_epe_px_s": float(epe[valid].mean()),
            "component_mae_px_s": float(np.abs(error[..., :2][valid]).mean()),
            "component_rmse_px_s": float(np.sqrt(np.mean(error[..., :2][valid] ** 2))),
            "speed_mae_px_s": float(np.abs(error[..., 2][valid]).mean()),
            "speed_rmse_px_s": float(np.sqrt(np.mean(error[..., 2][valid] ** 2))),
            "event_macro_epe_px_s": float(per_event[counts > 0].mean())}


def grouped(y, prediction, labels):
    labels = np.asarray(labels)
    return {v: metric(y[labels == v], prediction[labels == v]) for v in sorted(set(labels))}


def delta(new, reference):
    return {k: {"absolute_change": new[k] - reference[k] if new[k] is not None and reference[k] is not None else None,
                "relative_change_percent": 100 * (new[k] - reference[k]) / reference[k]
                if new[k] is not None and reference[k] not in (None, 0) else None} for k in METRICS}


def evaluate(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve existing evaluations; choose a fresh --out")
    protocol = read_json(args.targets / "protocol.json")
    prepared = read_json(args.targets / "prepared.json")
    if protocol["schema"] != SCHEMA or protocol["parameters"] != PARAMETERS:
        raise ValueError("unexpected frozen target protocol")
    folds = read_json(args.targets / "folds.json")
    groups = read_jsonl(args.targets / "sample_groups.jsonl")
    if folds != read_json(args.reference / "folds.json") or groups != read_jsonl(args.reference / "sample_groups.jsonl"):
        raise ValueError("prior sample identities or whole-episode folds changed")
    for name in ("observation", "request"):
        if read_jsonl(args.targets / f"{name}s.jsonl") != read_jsonl(args.reference / f"{name}_snapshot.jsonl"):
            raise ValueError("cannot use prior feature caches for changed model inputs")
    target_rows = read_jsonl(args.targets / "response_targets.jsonl")
    for target, old in zip(target_rows, read_jsonl(args.reference / "endpoint_target_snapshot.jsonl")):
        if (target["sample_id"] != old["sample_id"] or target["future_observations"][-1] != old["endpoint_target_only"]
                or len(target["future_observations"]) != old["source_future_record_count"]):
            raise ValueError("future window endpoints changed")
    with np.load(args.targets / "local_targets.npz") as saved:
        y = saved["targets"]
        np.testing.assert_array_equal(saved["valid"], np.isfinite(y))
    with np.load(args.reference / "features.npz") as saved:
        x = {arm: saved[arm] for arm in ARMS[2:]}
    if y.shape != (175, 72) or any(a.shape != (175, 2065) for a in x.values()):
        raise ValueError("fixed feature/target dimensions changed")
    np.testing.assert_array_equal(x[ARMS[2]][:, :-3], x[ARMS[3]][:, :-3])
    args.out.mkdir(parents=True)
    (args.out / "models").mkdir()
    for name in ("protocol.json", "prepared.json", "folds.json", "sample_groups.jsonl", "fixed_protocol.md"):
        shutil.copyfile(args.targets / name, args.out / name)
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    predictions = {arm: np.full_like(y, np.nan) for arm in ARMS}
    predictions[ARMS[0]][:] = 0
    logs, coverage = [], np.zeros(175, dtype=int)
    solve_seconds, reloaded = 0.0, 0
    normalizer_keys = ("mean", "std", "feature_scale", "standardized_clip")
    for fold_index, fold in enumerate(folds):
        train, test = np.asarray(fold["train_indices"]), np.asarray(fold["test_indices"])
        coverage[test] += 1
        observed = np.isfinite(y[train])
        counts = observed.sum(axis=0)
        if np.any(np.isfinite(y[test]) & (counts < 2)):
            raise ValueError(f"fold {fold_index} has test targets with <2 valid training targets; do not drop them")
        train_mean = np.divide(np.where(observed, y[train], 0).sum(axis=0), counts,
                               out=np.full(72, np.nan), where=counts > 0)
        predictions[ARMS[1]][test] = train_mean
        previous = None
        for arm in ARMS[2:]:
            # Whitelist only X-derived normalization, never the old learned outputs.
            with np.load(args.reference / "models" / f"fold_{fold_index:02d}_{arm}.npz") as old:
                np.testing.assert_array_equal(old["train_indices"], train)
                np.testing.assert_array_equal(old["test_indices"], test)
                if float(old["alpha"]) != ALPHA or float(old["standardized_clip"]) != CLIP:
                    raise ValueError("prior frozen fit settings changed")
                normalizer = {key: old[key] for key in normalizer_keys}
            z = transform(x[arm], normalizer)
            common = np.r_[np.arange(2062), np.arange(2065, 4127)]
            if previous is not None:
                np.testing.assert_array_equal(z[:, common], previous[:, common])
            previous = z
            t0 = time.perf_counter()
            coef, intercept, systems = fit_masked(z[train], y[train])
            path = args.out / "models" / f"fold_{fold_index:02d}_{arm}.npz"
            np.savez_compressed(path, **normalizer, coef=coef, intercept=intercept,
                                train_indices=train, test_indices=test, target_train_counts=counts, alpha=ALPHA)
            elapsed = time.perf_counter() - t0
            solve_seconds += elapsed
            with np.load(path) as saved:
                predictions[arm][test] = transform(x[arm][test], saved) @ saved["coef"] + saved["intercept"]
            reloaded += 1
            logs.append({"fold": fold_index, "arm": arm, "heldout_episode": fold["heldout_episode"],
                         "systems": systems, "solve_save_seconds": elapsed})
        print(f"local forecast fold {fold_index + 1}/10 complete", flush=True)
    if not np.all(coverage == 1):
        raise ValueError("incomplete or duplicated OOF coverage")
    np.savez_compressed(args.out / "oof_predictions.npz", **predictions, targets=y)
    write_json(args.out / "fold_log.json", logs)

    # Post-prediction strata are deliberately absent from all predictors.
    audits = read_jsonl(args.targets / "timing_audit.jsonl")
    if [a["sample_id"] for a in audits] != [g["sample_id"] for g in groups]:
        raise ValueError("audit join mismatch")
    strata = {
        "later_request": ["later_request_logged" if
            (a["source_command_pair"]["future_response_audit_only"]["later_requests_before_last_image"]
             or a["source_command_pair"]["future_response_audit_only"]["request_brackets_overlap_last_image"])
            else "no_later_request_logged" for a in audits],
        "anchor_history": ["available" if a["source_command_pair"]["anchor_observation"]
                           ["prior_controller_history_valid"] else "missing" for a in audits]}
    episodes = [g["source_episode"] for g in groups]
    tasks = [g["task"] for g in groups]
    methods = {}
    for arm, p in predictions.items():
        by_episode = grouped(y, p, episodes)
        macro = {k: float(np.mean([m[k] for m in by_episode.values() if m[k] is not None])) for k in METRICS}
        valid = np.isfinite(y).reshape(175, 24, 3).all(axis=-1)
        methods[arm] = {"pooled_oof": metric(y, p), "episode_macro": macro, "by_episode": by_episode,
            "by_view": {v: metric(y[:, j * 36:(j + 1) * 36], p[:, j * 36:(j + 1) * 36]) for j, v in enumerate(VIEWS)},
            "by_task": grouped(y, p, tasks),
            "post_prediction_audit": {key: grouped(y, p, values) for key, values in strata.items()},
            "negative_predicted_speed_cells": int((p.reshape(175, 24, 3)[..., 2][valid] < 0).sum())}
    comparisons = {}
    for reference in ARMS[:3]:
        new, old = methods[ARMS[3]], methods[reference]
        by_episode = {ep: delta(new["by_episode"][ep], old["by_episode"][ep]) for ep in new["by_episode"]}
        comparisons[f"request_minus_{reference}"] = {"pooled_oof": delta(new["pooled_oof"], old["pooled_oof"]),
            "episode_macro": delta(new["episode_macro"], old["episode_macro"]), "by_episode": by_episode,
            "episode_win_tie_loss": {k: win_counts([d[k]["absolute_change"] for d in by_episode.values()
                                                    if d[k]["absolute_change"] is not None]) for k in METRICS}}
    events = []
    for i, group in enumerate(groups):
        metrics = {arm: metric(y[i:i + 1], p[i:i + 1]) for arm, p in predictions.items()}
        events.append({**group, "metrics": metrics,
                       "request_minus_observation": delta(metrics[ARMS[3]], metrics[ARMS[2]]),
                       "audit_only": {k: v[i] for k, v in strata.items()}})
    write_jsonl(args.out / "oof_event_metrics.jsonl", events)
    positive = (all(methods[ARMS[3]][scope]["vector_epe_px_s"] < methods[a][scope]["vector_epe_px_s"]
                    for scope in ("pooled_oof", "episode_macro") for a in ARMS[:3])
                and all(methods[ARMS[3]][scope]["speed_mae_px_s"] <= methods[ARMS[2]][scope]["speed_mae_px_s"]
                        for scope in ("pooled_oof", "episode_macro")))
    report = {"schema": SCHEMA, "status": "complete", "reference": args.reference.as_posix(),
        "targets": args.targets.as_posix(), "counts": {"samples": 175, "episodes": 10, "by_task": dict(Counter(tasks)),
        "fold_arm_fits": len(logs), "linear_system_solves": sum(len(x["systems"]) for x in logs)},
        "target_coverage": prepared, "methods": methods, "paired_comparisons": comparisons,
        "verification": {"prior_inputs_and_fold_snapshots_match": True, "shared_transformed_features_match": True,
            "per_event_oof_count": [1], "saved_models_reloaded": reloaded,
            "target_imputation": False, "test_valid_targets_without_train_support": 0,
            "audit_read_after_prediction": True, "new_image_encodings": 0},
        "timing_seconds": {"local_target_extraction": prepared["prepare_seconds"], "ridge_solve_save": solve_seconds,
                           "remote_evaluate_before_report_write": time.perf_counter() - started},
        "decision": {"positive_local_proxy_signal": positive, "scorer_adopted": False,
                     "scope": "local heuristic motion prediction only; not wire advance, contact or policy performance",
                     "repeat_tuning": False}, "visual_status": "not_viewed", **FLAGS}
    write_json(args.out / "report.json", report)
    print({"seconds": report["timing_seconds"], "decision": report["decision"],
           "pooled": {a: m["pooled_oof"] for a, m in methods.items()}}, flush=True)


def render_previews(targets):
    """Fixed middle event per episode; observed arrows, NOT wire trajectories."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title_font = ImageFont.truetype(font_path, 29)
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 16)
    groups = read_jsonl(targets / "sample_groups.jsonl")
    indices = read_json(targets / "preview_indices.json")
    with np.load(targets / "local_targets.npz") as z:
        y = z["targets"].reshape(175, 2, 12, 3)
    figures = targets / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "data-manifest.md").write_text(
        "# 局部监督预览数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"| 左/右监督预览 | {targets.as_posix()}/preview_arrays.npz + local_targets.npz | real diagnostic | "
        "每个episode居中事件；原图ROI和观测局部速度 | tools/run_real10_elite_local_forecast.py:render_previews | "
        "local_target_preview_left.png / local_target_preview_right.png |\n\n"
        "蓝标放在网格中心，不是检测到的导丝点。显示缩放不影响监督或拟合。\n", encoding="utf-8")
    with np.load(targets / "preview_arrays.npz") as z:
        for task, task_zh in (("left", "左"), ("right", "右")):
            selected = [i for i in indices if groups[i]["task"] == task]
            canvas = Image.new("RGB", (1600, 140 + 185 * len(selected)), "#FFFFFF")
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 15), f"{task_zh}任务：每条轨迹居中请求的局部运动监督", font=title_font, fill="#192B3C")
            draw.text((24, 59), "原冻结区域，未启用新血管包络；光流/暗线可能含器械和纹理，不是导丝真值。", font=font, fill="#4B5563")
            for c, text_ in enumerate(("Side 锚点", "Side 末帧＋观测方向", "Top 锚点", "Top 末帧＋观测方向")):
                draw.text((140 + c * 360, 98), text_, font=font, fill="#192B3C")
            for row, i in enumerate(selected):
                top = 133 + row * 185
                label = task_zh + groups[i]["source_episode"].rsplit("_", 1)[1]
                draw.text((20, top + 34), label, font=font, fill="#192B3C")
                draw.text((20, top + 62), f"事件 {i + 1}", font=small, fill="#4B5563")
                for j, view in enumerate(VIEWS):
                    for endpoint, offset in (("anchor", 0), ("last", 1)):
                        original = z[f"s{i:03d}_{view}_{endpoint}"]
                        h = round(original.shape[0] * 350 / original.shape[1])
                        tile = Image.fromarray(original).convert("RGB").resize((350, h))
                        painter = ImageDraw.Draw(tile)
                        if endpoint == "last":
                            for cell in range(12):
                                r, c = divmod(cell, 6)
                                cx, cy = (c + .5) * 350 / 6, (r + .5) * h / 2
                                vector = y[i, j, cell, :2]
                                if not np.isfinite(vector).all():
                                    painter.text((cx - 3, cy - 5), "×", font=small, fill="#929AA5")
                                    continue
                                d = vector * 8
                                length = np.linalg.norm(d)
                                if length > 28:
                                    d *= 28 / length
                                end = (cx + d[0], cy + d[1])
                                painter.line((cx, cy, *end), fill="#0077BB", width=2)
                                painter.ellipse((end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2), fill="#0077BB")
                        canvas.paste(tile, (140 + (j * 2 + offset) * 360, top))
                draw.text((140, top + 132), "蓝标位于网格中心，非尖端/跟踪点；观测向量×8、限28px，仅显示；×为监督缺失。", font=small, fill="#4B5563")
            canvas.save(figures / f"local_target_preview_{task}.png", dpi=(450, 450))


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    report = read_json(args.out / "report.json")
    if report["schema"] != SCHEMA or report["status"] != "complete":
        raise ValueError("completed report required")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "path"})
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "data-manifest.md").write_text(
        "# 图表数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"| 局部响应预测 | {args.out.as_posix()}/report.json | real diagnostic | 175请求固定LOEO | "
        "tools/run_real10_elite_local_forecast.py --stage render | local_forecast.png / .svg |\n\n"
        "同一输入缓存、新的局部启发式目标；不是导丝推进/碰壁/独立泛化指标。\n", encoding="utf-8")
    labels = ["零运动", "训练均值", "仅观测", "观测＋请求"]
    colors = ["#929AA5", "#A7BBC7", "#0077BB", "#EE7733"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, key, title in ((axes[0], "vector_epe_px_s", "A  局部二维速度向量误差"),
                           (axes[1], "speed_mae_px_s", "B  平均速度模长误差")):
        values = [report["methods"][a]["pooled_oof"][key] for a in ARMS]
        ax.bar(range(4), values, color=colors, width=.65)
        ax.set_xticks(range(4), labels, rotation=15)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_ylabel("缩放 ROI 像素/秒 ↓")
        ax.set_title(title, loc="left", pad=14, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        for j, value in enumerate(values):
            ax.text(j, value + max(values) * .025, f"{value:.4f}", ha="center", fontsize=9)
    pairs = report["paired_comparisons"]["request_minus_observation_only"]["by_episode"]
    deltas = [v["vector_epe_px_s"]["absolute_change"] for v in pairs.values()]
    names = [("左" if "_left_" in ep else "右") + ep.rsplit("_", 1)[1] for ep in pairs]
    axes[2].barh(range(len(deltas)), deltas, color=["#009988" if d < 0 else "#EE7733" for d in deltas])
    axes[2].set_yticks(range(len(deltas)), names)
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="#929AA5", lw=1)
    axes[2].set_xlabel("加请求 − 仅观测：向量误差\n负值表示改善")
    axes[2].set_title("C  逐轨迹配对差异", loc="left", pad=14, fontweight="bold")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[2].grid(axis="x", alpha=.15)
    axes[2].set_axisbelow(True)
    fig.suptitle("固定输入下，Elite 请求能否改善局部运动代理预测？", fontsize=16, fontweight="bold", y=.99)
    valid, total = report["target_coverage"]["valid_cells"], report["target_coverage"]["possible_cells"]
    fig.text(.5, .03, f"175 事件 / 10 条相关轨迹 · 有效监督 {valid}/{total} 个时窗网格 · 原 ROI/掩膜与输入缓存不变\n"
             "目标是带参照平移补偿的局部光流代理，不是导丝轴向推进、碰壁或策略成功率。",
             ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.25, wspace=.42)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"local_forecast.{extension}", dpi=450, bbox_inches="tight")
    plt.close(fig)
    if not (figures / "visual_review.json").exists():
        write_json(figures / "visual_review.json", {"visual_status": "not_viewed", "user_acceptance": False,
                                                   "artifacts": ["local_forecast.png", "local_forecast.svg"]})
    print({"rendered": str(figures), "fit_executed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_elite_request_response_pack_v1"))
    parser.add_argument("--targets", type=Path, default=Path("simulation_output/real10_elite_local_response_targets_v1"))
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_elite_request_forecast_v1"))
    parser.add_argument("--motion-reference", type=Path, default=Path("simulation_output/real10_response_local_motion_v1"))
    parser.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_elite_local_forecast_v1"))
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "render": render}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
