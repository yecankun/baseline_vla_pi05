"""Fixed Elite macro-request forecast diagnostic; no PI05 or hardware calls.

Compare zero-change persistence, pre-request observations, and those same
observations plus the logged absolute target xyz. Future images are targets,
not conditioning. All folds/settings are fixed before looking at results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np

from prepare_real10_elite_request_pack import load_model_inputs, REQUEST_KIND
from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_baseline import _ridge_standardized_features


SCHEMA = "real10_elite_request_forecast_v1"
PROTOCOL = Path("docs/algorithm-real10-elite-request-forecast-protocol-20260921.md")
ARMS = ("persistence", "observation_only", "observation_plus_request")
VIEWS = ("side", "top")
BLOCKS = [14, 2048, 3]
ALPHA, CLIP, SEED = 1.0, 5.0, 20260921
FLAGS = {"policy_input_allowed": False, "policy_training_ready": False,
         "formal_data_allowed": False, "real_system_validated": False,
         "canonical_action9_compatible": False, "pi05_training_executed": False,
         "hardware_executed": False}
STATE_NAMES = ([f"anchor_elite_{a}_mm" for a in "xyz"]
               + [f"past_elite_change_{a}_mm" for a in "xyz"]
               + ["task_left", "task_right", "previous_busy", "past_busy_mean",
                  "past_event_count_change_not_mm", "past_observation_span_s",
                  "anchor_controller_history_valid", "both_endpoints_history_valid"])


def encode_images(input_paths, target_paths):
    """Frozen per-image inference; future paths are explicitly target-only."""
    import PIL
    from PIL import Image
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights

    if not torch.cuda.is_available():
        raise RuntimeError("run the encoder on the remote CUDA environment")
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cached weights required; no auto-download: {checkpoint}")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval().to("cuda")
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    keys = sorted(input_paths | target_paths)
    embeddings = {}
    with torch.inference_mode():
        for offset in range(0, len(keys), 32):
            batch = keys[offset:offset + 32]
            pixels = []
            for path in batch:
                with Image.open(path) as image:
                    if image.size != (224, 224):
                        raise ValueError("use the existing 224px images without implicit resize/crop")
                    pixels.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
            x = torch.stack(pixels).to(device="cuda", dtype=torch.float32) / 255.0
            z = encoder((x - mean) / std).cpu().numpy()
            if z.shape != (len(batch), 512) or not np.isfinite(z).all():
                raise ValueError("invalid frozen visual features")
            embeddings.update(zip(batch, z))
            if offset % 128 == 0 or offset + 32 >= len(keys):
                print(f"frozen images {min(offset + 32, len(keys))}/{len(keys)}", flush=True)
    return embeddings, {
        "encoder": "torchvision_resnet18_IMAGENET1K_V1_frozen_eval",
        "weights": str(checkpoint), "real10_finetuning": False, "batchnorm_updates": False,
        "input_image_count": len(input_paths), "endpoint_target_image_count": len(target_paths),
        "input_target_path_overlap": len(input_paths & target_paths), "images_encoded": len(keys),
        "future_features_in_forecast_inputs": False,
        "preprocessing": "existing 224px RGB /255 -> ImageNet mean/std; no ROI/augmentation",
        "seconds_including_model_load": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "Pillow": PIL.__version__},
    }


def observation_features(observation, input_embeddings):
    """Accept only the observation whitelist and its past-image embeddings."""
    history = observation["history"]
    s = np.array([h["state_32"] for h in history], dtype=np.float64)
    valid = np.array([h["state_valid_32"] for h in history], dtype=bool)
    s[~valid] = np.nan
    times = []
    for h in history:
        pose_time = h["elite_pose_query_start_relative_time_s"]
        ts = [pose_time, *[h["images"][v]["relative_time_s"] for v in VIEWS]]
        if h["controller_snapshot_relative_time_s"] is not None:
            ts.append(h["controller_snapshot_relative_time_s"])
        if not all(np.isfinite(t) and t <= 0 for t in ts):
            raise ValueError("post-request information in input history")
        times.append(pose_time)
    if any(b <= a for a, b in zip(times, times[1:])) or observation["nominal_horizon_s"] != 1.5:
        raise ValueError("history order or fixed horizon changed")
    busy = s[:, 16][np.isfinite(s[:, 16])]
    state = np.concatenate((s[-1, :3], s[-1, :3] - s[0, :3], s[-1, 14:16],
        [s[-1, 16], float(busy.mean()) if len(busy) else np.nan,
         s[-1, 6] - s[0, 6], times[-1] - times[0], s[-1, 27],
         float(s[-1, 27] == 1 and s[0, 27] == 1)]))
    visual = []
    for view in VIEWS:
        first = input_embeddings[history[0]["images"][view]["path"]]
        anchor = input_embeddings[history[-1]["images"][view]["path"]]
        visual.extend((anchor, anchor - first))
    if state.shape != (14,):
        raise ValueError("fixed state feature layout changed")
    return np.concatenate((state, *visual))


def request_features(request):
    if (request["motion_mode"] != REQUEST_KIND or not all(request["elite_target_valid_6d"])
            or request["piper_intent_id"] != 1 or request["piper_burst_count"] != 1
            or request["elite_path_speed_sdk_parameter"] != 20):
        raise ValueError("not the fixed absolute-target/hold diagnostic")
    target = np.asarray(request["elite_target_tcp_pose_6d_xyz_mm_rpy_rad"], dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("invalid requested target")
    return target[:3]


def fit_regression(x, y):
    """Uniform multi-output ridge, unpenalized intercept, train-fold stats only."""
    finite = np.isfinite(x)
    count = np.maximum(finite.sum(axis=0), 1)
    mean = np.where(finite, x, 0).sum(axis=0) / count
    std = np.sqrt((np.where(finite, x - mean, 0) ** 2).sum(axis=0) / count)
    std[std < 1e-6] = 1.0
    scale = np.tile(np.sqrt(np.repeat(2 * np.asarray(BLOCKS), BLOCKS)), 2)
    z = _ridge_standardized_features(x, mean, std, CLIP) / scale
    center, target_center = z.mean(axis=0), y.mean(axis=0)
    design = z - center
    kernel = design @ design.T + ALPHA * np.eye(len(y))
    rhs = y - target_center
    dual = np.linalg.solve(kernel, rhs)
    residual = float(np.max(np.abs(kernel @ dual - rhs)))
    if not np.isfinite(dual).all() or residual > 1e-7 * (1 + np.max(np.abs(rhs))):
        raise ValueError("ridge solve failed the numerical residual check")
    return {"mean": mean, "std": std, "feature_scale": scale, "center": center,
            "design": design, "dual": dual, "target_center": target_center,
            "alpha": np.array(ALPHA), "standardized_clip": np.array(CLIP)}, residual


def predict(model, x):
    z = _ridge_standardized_features(x, model["mean"], model["std"], model["standardized_clip"])
    z /= model["feature_scale"]
    return ((z - model["center"]) @ model["design"].T) @ model["dual"] + model["target_center"]


def error_metrics(y, prediction):
    error = prediction - y
    return {"n": len(y), "elite_translation_mae_mm": float(np.abs(error[:, :3]).mean()),
            "elite_translation_rmse_mm": float(np.sqrt(np.mean(error[:, :3] ** 2))),
            **{f"elite_{a}_mae_mm": float(np.abs(error[:, j]).mean()) for j, a in enumerate("xyz")},
            "visual_feature_delta_rmse": float(np.sqrt(np.mean(error[:, 3:] ** 2))),
            "side_feature_delta_rmse": float(np.sqrt(np.mean(error[:, 3:515] ** 2))),
            "top_feature_delta_rmse": float(np.sqrt(np.mean(error[:, 515:] ** 2)))}


def grouped_metrics(y, prediction, values):
    values = np.asarray(values)
    return {str(v): error_metrics(y[values == v], prediction[values == v]) for v in sorted(set(values))}


def macro_metrics(by_episode):
    return {"episodes": len(by_episode), **{
        k: float(np.mean([m[k] for m in by_episode.values()]))
        for k in next(iter(by_episode.values())) if k != "n"}}


def differences(new, reference):
    return {k: {"absolute_change": new[k] - reference[k],
                "relative_change_percent": 100 * (new[k] - reference[k]) / reference[k]
                if reference[k] != 0 else None}
            for k in new if k not in ("n", "episodes")}


def win_counts(deltas):
    deltas = np.asarray(deltas)
    return {"improved": int(np.sum(deltas < -1e-12)), "tied": int(np.sum(np.abs(deltas) <= 1e-12)),
            "worsened": int(np.sum(deltas > 1e-12))}


def summary_range(values):
    return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve prior artifacts; choose a fresh --out")
    manifest = read_json(args.source / "manifest.json")
    if any(manifest[k] is not False for k in FLAGS if k in manifest):
        raise ValueError("source diagnostic flags changed")
    inputs = load_model_inputs(args.source)
    targets = read_jsonl(args.source / "response_targets.jsonl")
    groups = read_jsonl(args.source / "sample_groups.jsonl")
    observations = read_jsonl(args.source / "observations.jsonl")
    requests = read_jsonl(args.source / "requests.jsonl")
    folds = read_json(args.source / "folds.json")
    if not len(inputs) == len(targets) == len(groups) == 175 or len(folds) != 10:
        raise ValueError("fixed 175 samples/10 episode folds required")
    for i, group in enumerate(groups):
        for rows in (targets, observations, requests):
            if rows[i]["sample_index"] != i or rows[i]["sample_id"] != group["sample_id"]:
                raise ValueError("source row identity or order changed")
        if group["sample_index"] != i or group["original_source_split"] != "train":
            raise ValueError("original all-train source identity changed")
    episodes = [g["source_episode"] for g in groups]
    if [f["heldout_episode"] for f in folds] != manifest["source_episodes"]:
        raise ValueError("original whole-episode fold ordering changed")
    coverage = np.zeros(len(inputs), dtype=int)
    for fold in folds:
        test = [i for i, episode in enumerate(episodes) if episode == fold["heldout_episode"]]
        train = [i for i, episode in enumerate(episodes) if episode != fold["heldout_episode"]]
        if fold["test_indices"] != test or fold["train_indices"] != train or not test or not train:
            raise ValueError("not the original full-episode LOEO split")
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ValueError("each sample must be held out exactly once")

    endpoints = [row["future_observations"][-1] for row in targets]
    for target, endpoint in zip(targets, endpoints):
        if (target["joint_motion_response"] is not None or target["joint_motion_response_valid"]
                or not all(endpoint["measured_pose_valid_6d"][:3])
                or not all(endpoint["images"][v]["relative_time_s"] > 0 for v in VIEWS)):
            raise ValueError("fixed unlabeled, valid measured endpoint required; do not filter rows")
    args.out.mkdir(parents=True)
    (args.out / "models").mkdir()
    shutil.copyfile(PROTOCOL, args.out / "fixed_protocol.md")
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    write_json(args.out / "source_manifest.json", manifest)
    write_json(args.out / "folds.json", folds)
    write_jsonl(args.out / "observation_snapshot.jsonl", observations)
    write_jsonl(args.out / "request_snapshot.jsonl", requests)
    write_jsonl(args.out / "sample_groups.jsonl", groups)
    write_jsonl(args.out / "endpoint_target_snapshot.jsonl", [
        {"sample_index": i, "sample_id": groups[i]["sample_id"], "endpoint_target_only": endpoint,
         "source_future_record_count": len(targets[i]["future_observations"])}
        for i, endpoint in enumerate(endpoints)])
    write_json(args.out / "protocol.json", {
        "schema": SCHEMA, "source": args.source.as_posix(), "fixed_protocol": str(PROTOCOL),
        "source_split": manifest["source_split"], "arms": list(ARMS), "raw_feature_blocks": BLOCKS,
        "state_feature_names": STATE_NAMES, "visual_features": "per view: anchor512, anchor-minus-first512",
        "request_features": "absolute target xyz in mm; zero slots in observation-only arm",
        "target_dimensions": {"elite_translation_mm": 3, "side_delta": 512, "top_delta": 512},
        "target_endpoint": "last recorded observation in original nominal 1.5s window; not exact t+1.5s",
        "alpha": ALPHA, "standardized_clip": CLIP, "seed": SEED, "sample_weights": "uniform",
        "normalization": "training-fold mean/std only; mean fill plus missing indicators; per-block scale",
        "optimizer_steps": 0, "closed_form_fits": 20, "model_selection": False,
        "manual_labels_used": False, "diagnostic_audit_as_input": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), **FLAGS})

    input_paths = {h["images"][v]["path"] for x in inputs
                   for h in (x["observation"]["history"][0], x["observation"]["history"][-1]) for v in VIEWS}
    target_paths = {endpoint["images"][v]["path"] for endpoint in endpoints for v in VIEWS}
    embeddings, encoding = encode_images(input_paths, target_paths)
    # Input construction gets only input-image lookups, never future rows.
    input_embeddings = {path: embeddings[path] for path in input_paths}
    common = np.stack([observation_features(x["observation"], input_embeddings) for x in inputs])
    request = np.stack([request_features(x["request"]) for x in inputs])
    arrays = {ARMS[1]: np.column_stack((common, np.zeros_like(request))),
              ARMS[2]: np.column_stack((common, request))}
    target_elite = np.asarray([e["measured_elite_translation_from_anchor_mm"] for e in endpoints], dtype=float)
    target_visual = np.stack([np.concatenate([
        embeddings[endpoint["images"][v]["path"]]
        - input_embeddings[x["observation"]["history"][-1]["images"][v]["path"]] for v in VIEWS])
        for x, endpoint in zip(inputs, endpoints)])
    y = np.column_stack((target_elite, target_visual)).astype(np.float64)
    if (common.shape != (175, 2062) or y.shape != (175, 1027) or not np.isfinite(y).all()
            or not np.array_equal(arrays[ARMS[1]][:, :-3], arrays[ARMS[2]][:, :-3], equal_nan=True)):
        raise ValueError("fixed feature/target layout or shared inputs changed")
    np.savez_compressed(args.out / "features.npz", **arrays, targets=y)
    np.savez_compressed(args.out / "frozen_embeddings.npz", paths=np.asarray(sorted(embeddings)),
                        embeddings=np.stack([embeddings[k] for k in sorted(embeddings)]))
    predictions = {ARMS[0]: np.zeros_like(y), **{arm: np.full_like(y, np.nan) for arm in ARMS[1:]}}
    fold_log = []
    fit_seconds = prediction_seconds = 0.0
    common_norm_ok = 0
    for fold_index, fold in enumerate(folds):
        train, test = np.asarray(fold["train_indices"]), np.asarray(fold["test_indices"])
        previous = None
        for arm in ARMS[1:]:
            fit_started = time.perf_counter()
            model, residual = fit_regression(arrays[arm][train], y[train])
            if previous is not None:
                indices = np.r_[np.arange(2062), np.arange(2065, 4127)]
                if (not all(np.array_equal(model[k][:-3], previous[k][:-3]) for k in ("mean", "std"))
                        or not np.array_equal(model["design"][:, indices], previous["design"][:, indices])):
                    raise ValueError("common observation normalization differs across arms")
                common_norm_ok += 1
            previous = model
            model_path = args.out / "models" / f"fold_{fold_index:02d}_{arm}.npz"
            np.savez_compressed(model_path, **model, train_indices=train, test_indices=test)
            fit_elapsed = time.perf_counter() - fit_started
            fit_seconds += fit_elapsed
            predict_started = time.perf_counter()
            # Reopen the delivered fold artifact for the actual OOF predictions.
            with np.load(model_path) as saved:
                predictions[arm][test] = predict(saved, arrays[arm][test])
            prediction_elapsed = time.perf_counter() - predict_started
            prediction_seconds += prediction_elapsed
            fold_log.append({"fold_index": fold_index, "heldout_episode": fold["heldout_episode"],
                "arm": arm, "train_samples": len(train), "test_samples": len(test),
                "solve_and_save_seconds": fit_elapsed, "reload_and_predict_seconds": prediction_elapsed,
                "solve_residual_max_abs": residual, "model": model_path.relative_to(args.out).as_posix()})
        print(f"LOEO fold {fold_index + 1}/{len(folds)} complete", flush=True)
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("incomplete/nonfinite OOF predictions")
    np.savez_compressed(args.out / "oof_predictions.npz", **predictions, targets=y)
    write_json(args.out / "fold_log.json", fold_log)

    # Only now read post-request audit information; it never affects fitting.
    audits = read_jsonl(args.source / "timing_audit.jsonl")
    if len(audits) != len(groups) or any(a["sample_id"] != g["sample_id"] for a, g in zip(audits, groups)):
        raise ValueError("post-prediction audit join mismatch")
    strata = {
        "later_request": ["later_request_logged" if
            (a["source_command_pair"]["future_response_audit_only"]["later_requests_before_last_image"]
             or a["source_command_pair"]["future_response_audit_only"]["request_brackets_overlap_last_image"])
            else "no_later_request_logged" for a in audits],
        "anchor_history": ["available" if a["source_command_pair"]["anchor_observation"]
                           ["prior_controller_history_valid"] else "missing" for a in audits],
        "history_window": ["truncated" if a["history_boundary_truncated"] else "full" for a in audits],
    }
    tasks = [g["task"] for g in groups]
    methods = {}
    for arm, prediction in predictions.items():
        by_episode = grouped_metrics(y, prediction, episodes)
        methods[arm] = {"pooled_oof": error_metrics(y, prediction), "by_episode": by_episode,
                        "episode_macro": macro_metrics(by_episode),
                        "by_task": grouped_metrics(y, prediction, tasks),
                        "post_prediction_audit": {name: grouped_metrics(y, prediction, values)
                                                  for name, values in strata.items()}}
    paired = {}
    for reference in ARMS[:2]:
        changed, baseline = methods[ARMS[2]], methods[reference]
        per_episode = {ep: differences(changed["by_episode"][ep], baseline["by_episode"][ep])
                       for ep in sorted(set(episodes))}
        paired[f"request_minus_{reference}"] = {
            "pooled_oof": differences(changed["pooled_oof"], baseline["pooled_oof"]),
            "episode_macro": differences(changed["episode_macro"], baseline["episode_macro"]),
            "by_episode": per_episode,
            "episode_win_tie_loss": {metric: win_counts([values[metric]["absolute_change"]
                for values in per_episode.values()]) for metric in changed["pooled_oof"] if metric != "n"}}
    event_rows = []
    for i, group in enumerate(groups):
        event_metrics = {arm: error_metrics(y[i:i + 1], p[i:i + 1]) for arm, p in predictions.items()}
        event_rows.append({**group, "target_elite_translation_mm": target_elite[i].tolist(),
            "predicted_elite_translation_mm": {arm: p[i, :3].tolist() for arm, p in predictions.items()},
            "metrics": event_metrics,
            "request_minus_observation": differences(event_metrics[ARMS[2]], event_metrics[ARMS[1]]),
            "request_minus_persistence": differences(event_metrics[ARMS[2]], event_metrics[ARMS[0]]),
            "audit_only": {name: values[i] for name, values in strata.items()}})
    write_jsonl(args.out / "oof_event_metrics.jsonl", event_rows)
    visual_positive = all(methods[ARMS[2]][scope]["visual_feature_delta_rmse"]
                          < methods[ref][scope]["visual_feature_delta_rmse"]
                          for scope in ("pooled_oof", "episode_macro") for ref in ARMS[:2])
    elite_positive = all(methods[ARMS[2]][scope]["elite_translation_mae_mm"]
                         < methods[ref][scope]["elite_translation_mae_mm"]
                         for scope in ("pooled_oof", "episode_macro") for ref in ARMS[:2])
    report = {"schema": SCHEMA, "status": "complete", "source": args.source.as_posix(),
        "counts": {"samples": len(inputs), "episodes": len(set(episodes)), "by_task": dict(Counter(tasks)),
                   "folds": len(folds), "ridge_fits": len(fold_log), "per_sample_oof_coverage": [1],
                   "unique_target_xyz": len(np.unique(request, axis=0))},
        "encoding": encoding, "methods": methods, "paired_comparisons": paired,
        "endpoint_actual_time_s_target_only": {
            **{view: summary_range([e["images"][view]["relative_time_s"] for e in endpoints]) for view in VIEWS},
            "elite_pose_query_start": summary_range([e["pose_query_start_relative_time_s"] for e in endpoints]),
            "top_minus_side": summary_range([e["images"]["top"]["relative_time_s"]
                                             - e["images"]["side"]["relative_time_s"] for e in endpoints])},
        "verification": {"complete_episode_oof": True, "all_predictions_finite": True,
            "common_feature_normalization_equal_folds": common_norm_ok, "fold_models_reloaded": len(fold_log),
            "max_solve_residual": max(row["solve_residual_max_abs"] for row in fold_log),
            "future_images_are_targets_only": True, "post_request_audit_read_after_prediction": True,
            "manual_labels_used": False, "rows_filtered": 0,
            "missing_state_feature_counts": dict(zip(STATE_NAMES, np.sum(~np.isfinite(common[:, :14]), axis=0).tolist()))},
        "timing_seconds": {"encoding_including_model_load": encoding["seconds_including_model_load"],
            "ridge_solve_and_save": fit_seconds, "reload_and_prediction": prediction_seconds,
            "run_before_final_report_write": time.perf_counter() - started},
        "decision": {"visual_positive_diagnostic_signal": visual_positive,
            "elite_positive_diagnostic_signal": elite_positive,
            "outcome": ("positive_visual_diagnostic_only" if visual_positive else
                        "elite_command_correspondence_only" if elite_positive else "no_positive_request_signal"),
            "scorer_adopted": False, "repeat_tuning_on_175_events": False,
            "scope": "observational related-episode diagnostic; no wire/contact/causal-ranking/policy-success claim"},
        "regression_fitted": True, "neural_network_training_executed": False,
        "visual_status": "not_viewed", **FLAGS}
    write_json(args.out / "report.json", report)
    print({"status": report["status"], "seconds": report["timing_seconds"],
           "decision": report["decision"], "pooled": {arm: methods[arm]["pooled_oof"] for arm in ARMS}}, flush=True)
    return report


def render(out):
    """Render actual saved metrics locally, with Chinese labels and no fitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    report = read_json(out / "report.json")
    if report["schema"] != SCHEMA or report["status"] != "complete":
        raise ValueError("completed forecast report required")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if not font.exists():
        raise FileNotFoundError("render locally with the Chinese Microsoft YaHei font")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "path", "savefig.dpi": 450})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    source = out / "report.json"
    (figures / "data-manifest.md").write_text(
        "# 图表数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"| Elite请求未来预测 | {source.as_posix()} | real diagnostic | 175事件、10轨迹固定LOEO | "
        "tools/run_real10_elite_request_forecast.py --stage render | forecast_comparison.png / .svg |\n\n"
        "未来是原名义1.5秒窗口的已有末帧；没有绘制生成的未来图像。视觉单位是冻结特征，不是导丝位移。\n",
        encoding="utf-8")
    methods = report["methods"]
    labels = ["保持不变", "仅观测", "观测＋请求"]
    colors = ["#929AA5", "#0077BB", "#EE7733"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, metric, title, ylabel, precision in (
            (axes[0], "elite_translation_mae_mm", "A  机械臂平移预测", "OOF 平移 MAE（mm）↓", 3),
            (axes[1], "visual_feature_delta_rmse", "B  双视角视觉变化预测", "OOF 特征变化 RMSE（任意单位）↓", 4)):
        values = [methods[a]["pooled_oof"][metric] for a in ARMS]
        ax.bar(range(3), values, color=colors, width=.64)
        ax.set_xticks(range(3), labels, rotation=12)
        ax.set_ylim(0, max(values) * 1.23)
        ax.set_title(title, loc="left", pad=16, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        for j, value in enumerate(values):
            ax.text(j, value + max(values) * .03, f"{value:.{precision}f}", ha="center", fontsize=11)
    by_episode = report["paired_comparisons"]["request_minus_observation_only"]["by_episode"]
    episode_labels = [("左" if "_left_" in ep else "右") + ep.rsplit("_", 1)[1] for ep in by_episode]
    delta = [v["visual_feature_delta_rmse"]["absolute_change"] for v in by_episode.values()]
    axes[2].barh(range(len(delta)), delta, color=["#009988" if d < 0 else "#EE7733" for d in delta])
    axes[2].set_yticks(range(len(delta)), episode_labels)
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="#929AA5", lw=1)
    axes[2].set_xlabel("加请求 − 仅观测：视觉 RMSE\n负值表示误差降低")
    axes[2].set_title("C  整轨迹留一：逐轨迹差异", loc="left", pad=16, fontweight="bold")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[2].grid(axis="x", alpha=.15)
    axes[2].set_axisbelow(True)
    fig.suptitle("Elite 目标请求是否提供额外的未来预测信息？", fontsize=17, fontweight="bold", y=.99)
    fig.text(.5, .035, "175 个事件 / 10 条相关轨迹 · 固定 10 折留一 · 冻结 ResNet18 + 线性岭回归\n"
             "预测名义 1.5 秒窗口的已有末帧；不是导丝推进、碰壁、策略成功率或独立泛化验证。",
             ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.25, wspace=.42)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"forecast_comparison.{extension}", dpi=450, bbox_inches="tight")
    plt.close(fig)
    review = figures / "visual_review.json"
    if not review.exists():
        write_json(review, {"visual_status": "not_viewed", "user_acceptance": False,
                            "artifact": "forecast_comparison.png", "source": source.as_posix()})
    print({"rendered": str(figures), "source": str(source), "fit_executed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "render"), default="run")
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_elite_request_response_pack_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_elite_request_forecast_v1"))
    args = parser.parse_args()
    if args.stage == "render":
        render(args.out)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
