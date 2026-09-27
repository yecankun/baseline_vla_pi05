"""Fold-isolated, frozen-localization response comparison. Never a policy.

profile discards its temporary weights; train uses all fixed folds/seeds and
final steps only. Frozen response inputs are captured from the original forward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np

from fit_real10_head_region import (
    distance_to_geometry, make_records, read_json, read_jsonl, write_json, write_jsonl)

SCHEMA = "real10_frozen_head_response_pair_v1"
SEEDS = (20261020, 20261120, 20261220)
ARMS = ("tip_frozen", "head_frozen")
VIEWS = ("side", "top")
CONFIG = {"base_seeds": list(SEEDS), "arms": list(ARMS),
          "fit_seed": "base_seed + zero_based_fold_index",
          "localization_steps": 2000, "response_steps": 200,
          "optimizer": "separate AdamW per stage", "lr": .001, "weight_decay": .01,
          "gradient_clip": 1., "localization_scale": .1,
          "tip_target": "old sigma1 grid Gaussian CE/log(56^2)",
          "head_target": "old bbox/one-cell line support set-mass NLL/log(56^2)",
          "localization_weighting": "frame mean then labeled training-window macro",
          "response": "same balanced BCE and mean pooling; training-only class weight",
          "freeze": "response frozen during localization; project/location frozen during response",
          "matched_geometry_pool": "same 25 frames with both valid tip and head annotation",
          "threshold": .5, "selection": "fixed final steps; all 3 seeds and 10 folds",
          "precision": "FP32 deterministic no AMP/TF32"}
DESIGN = Path("docs/algorithm-real10-head-response-pair-protocol-20260923.md")
DEPENDENCIES = {"runner_snapshot.py": Path(__file__),
                "model_snapshot.py": Path("tools/real10_spatial_response.py"),
                "geometry_adapter_snapshot.py": Path("tools/fit_real10_head_region.py")}


def runtime():
    # The old runner sets CUBLAS_WORKSPACE_CONFIG before importing torch.
    import train_real10_spatial_aux_pair as old
    import torch
    return old, torch


def fold_geometry_audit(records, folds):
    audits = []
    for fold_id, fold in enumerate(folds):
        train = [i for i, r in enumerate(records) if r["sample_index"] in fold["train_indices"]]
        test = [i for i, r in enumerate(records) if r["sample_index"] in fold["test_indices"]]
        episodes = sorted({records[i]["source_episode"] for i in train})
        if not train or fold["heldout_episode"] in episodes or set(train) & set(test):
            raise ValueError("localization supervision crosses the heldout episode")
        audits.append({"fold_index": fold_id, "heldout_episode": fold["heldout_episode"],
                       "train_geometry_indices": train, "test_geometry_indices": test,
                       "train_geometry_episodes": episodes, "train_geometry_frames": len(train),
                       "test_geometry_frames": len(test), "train_windows": len(fold["train_indices"]),
                       "test_windows": len(fold["test_indices"])})
    return audits


def initialize(args):
    old, _ = runtime()
    tick = time.perf_counter()
    inputs, targets, rows, folds = old.check_frozen(args.source)
    geometry, support, _, roi, grid = make_records(args, inputs, rows, folds)
    audits = fold_geometry_audit(geometry, folds)
    if args.out.exists():
        protocol = read_json(args.out / "protocol.json")
        if protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix():
            raise ValueError("existing output uses another protocol")
        for name, current in DEPENDENCIES.items():
            if (args.out / name).read_bytes() != current.read_bytes():
                raise ValueError(f"{name} changed; preserve existing experiment")
        for name in ("report.json", "source_manifest.json", "source_head_annotations.jsonl"):
            if (args.out / "head_source" / name).read_bytes() != (args.labels / name).read_bytes():
                raise ValueError("head annotation source changed")
        for name in ("model_inputs.jsonl", "supervision.jsonl", "annotation_snapshot.jsonl",
                     "folds.json", "point_audit.jsonl", "protocol.json"):
            if (args.out / "response_source" / name).read_bytes() != (args.source / "source" / name).read_bytes():
                raise ValueError("response source changed")
    else:
        args.out.mkdir(parents=True)
        (args.out / "head_source").mkdir()
        (args.out / "response_source").mkdir()
        (args.out / "checkpoints").mkdir()
        for name, path in DEPENDENCIES.items():
            shutil.copy2(path, args.out / name)
        shutil.copy2(DESIGN, args.out / "design_protocol.md")
        for name in ("report.json", "source_manifest.json", "source_head_annotations.jsonl"):
            shutil.copy2(args.labels / name, args.out / "head_source" / name)
        for name in ("model_inputs.jsonl", "supervision.jsonl", "annotation_snapshot.jsonl",
                     "folds.json", "point_audit.jsonl", "protocol.json"):
            shutil.copy2(args.source / "source" / name, args.out / "response_source" / name)
        write_jsonl(args.out / "matched_geometry.jsonl", geometry)
        write_json(args.out / "fold_geometry_audit.json", audits)
        write_json(args.out / "protocol.json", {
            "schema": SCHEMA, "config": CONFIG, "source": args.source.as_posix(),
            "labels": args.labels.as_posix(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "windows": 45, "source_episodes": 10, "folds": folds, "fits": 60,
            "response_counts": {"stationary": 20, "advance": 25},
            "matched_geometry_frames": 25, "roi_xyxy_original": roi,
            "case4_head_geometry_excluded_for_both_arms": True,
            "original_case4_response_included": True,
            "whole_episode_isolation_for_both_stages": True,
            "full_data_deployment_fit": False, "prepare_s": time.perf_counter()-tick,
            "test_scope": "reused development LOEO, not independent testing",
            "input": "original ROI pixels/frozen maps, task and observed dt only",
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
    return inputs, targets, rows, folds, geometry, support, roi, grid, audits


def load_initial(args, seed, fold_id, fold):
    old, torch = runtime()
    path = old.checkpoint_path(args.source, seed, fold_id, "spatial_window_tip_aux")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    old.validate_checkpoint(saved, seed, fold_id, fold, "spatial_window_tip_aux")
    return saved["initial_state"]


def cached_response(model, features, valid):
    logits = model.response(features).squeeze(-1)
    return logits.masked_fill(~valid, 0).sum(1) / valid.sum(1)


def fit_one(tensors, geometry, support, fold, audit, initial, fit_seed, arm, loc_steps, resp_steps):
    old, torch = runtime()
    from real10_spatial_response import SpatialResponse, point_distribution
    from torch.nn import functional as F
    old.seed_all(fit_seed)
    model = SpatialResponse().cuda().eval()
    model.load_state_dict(initial)
    model.response.requires_grad_(False)
    ids = audit["train_geometry_indices"]
    indices = torch.tensor([[geometry[i]["sample_index"], geometry[i]["frame_index"],
                             VIEWS.index(geometry[i]["view"])] for i in ids], device="cuda")
    maps = tensors["maps"][indices[:, 0], indices[:, 1], indices[:, 2]]
    xy = tensors["xy_uv"][indices[:, 0], indices[:, 1], indices[:, 2]]
    if not tensors["tip_valid"][indices[:, 0], indices[:, 1], indices[:, 2]].all():
        raise ValueError("matched frames must have valid old tip targets")
    weights_by_window = defaultdict(int)
    for i in ids:
        weights_by_window[geometry[i]["window_id"]] += 1
    weights = torch.tensor([1/(len(weights_by_window)*weights_by_window[geometry[i]["window_id"]])
                            for i in ids], device="cuda")
    mask = torch.from_numpy(support[ids]).cuda()
    q = point_distribution(xy, torch.ones(len(ids), device="cuda", dtype=torch.bool)).flatten(1)
    log_cells = float(np.log(56**2))

    def localization():
        logits = model.location(model.project(maps)).flatten(1)
        if arm == "tip_frozen":
            per_frame = -(q*logits.log_softmax(-1)).sum(-1)
        else:
            per_frame = logits.logsumexp(-1)-logits.masked_fill(~mask, -torch.inf).logsumexp(-1)
        return (per_frame*weights).sum()/log_cells

    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 529:
        raise ValueError("unexpected localization parameter count")
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    with torch.no_grad():
        loc_curve = [{"step": 0, "loss_unscaled": float(localization())}]
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(1, loc_steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = .1*localization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 200 == 0 or step == loc_steps:
            with torch.no_grad():
                loc_curve.append({"step": step, "loss_unscaled": float(localization())})
    torch.cuda.synchronize()
    localization_s = time.perf_counter()-tick
    if any(not torch.equal(p.cpu(), initial[k]) for k, p in model.state_dict().items() if k.startswith("response.")):
        raise ValueError("response parameters changed during localization")
    del optimizer, maps, parameters
    shared_after_loc = {k: p.cpu().clone() for k, p in model.state_dict().items() if not k.startswith("response.")}
    model.project.requires_grad_(False)
    model.location.requires_grad_(False)
    model.response.requires_grad_(True)
    parameters = list(model.response.parameters())
    if sum(p.numel() for p in parameters) != 1025:
        raise ValueError("unexpected response parameter count")
    train = old.subset(tensors, fold["train_indices"])
    captured = []
    hook = model.response.register_forward_pre_hook(lambda module, values: captured.append(values[0].detach()))
    torch.cuda.synchronize()
    tick = time.perf_counter()
    try:
        with torch.no_grad():
            original = old.forward(model, train)
    finally:
        hook.remove()
    features = captured[0]
    with torch.no_grad():
        cache_diff = float((cached_response(model, features, train["transition_valid"])-original["window_logit"]).abs().max())
    if cache_diff > 1e-6:
        raise ValueError("frozen feature cache differs from original forward")
    torch.cuda.synchronize()
    cache_s = time.perf_counter()-tick
    del original, captured
    positives = train["response_y"].sum()
    pos_weight = (len(train["response_y"])-positives)/positives
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)

    def response_loss():
        return F.binary_cross_entropy_with_logits(cached_response(model, features, train["transition_valid"]),
                                                  train["response_y"], pos_weight=pos_weight)

    with torch.no_grad():
        resp_curve = [{"step": 0, "loss": float(response_loss())}]
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(1, resp_steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = response_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == resp_steps:
            with torch.no_grad():
                resp_curve.append({"step": step, "loss": float(response_loss())})
    torch.cuda.synchronize()
    response_s = time.perf_counter()-tick
    if any(not torch.equal(model.state_dict()[k].cpu(), value) for k, value in shared_after_loc.items()):
        raise ValueError("localization changed during response fitting")
    with torch.no_grad():
        full = old.forward(model, train)["window_logit"]
        final_cache_diff = float((full-cached_response(model, features, train["transition_valid"])).abs().max())
    if final_cache_diff > 1e-6:
        raise ValueError("final cached classifier and complete forward differ")
    info = {"localization_steps": loc_steps, "response_steps": resp_steps,
            "localization_s": localization_s, "response_s": response_s, "feature_cache_s": cache_s,
            "train_windows": len(fold["train_indices"]), "train_geometry_frames": len(ids),
            "train_geometry_windows": len(weights_by_window), "train_pos_weight": float(pos_weight),
            "localization_parameters": 529, "response_parameters": 1025,
            "localization_curve": loc_curve, "response_curve": resp_curve,
            "response_unchanged_during_localization": True, "localization_unchanged_during_response": True,
            "cached_forward_initial_max_abs": cache_diff, "cached_forward_final_max_abs": final_cache_diff}
    return model, info


def profile(args):
    old, torch = runtime()
    old.configure()
    inputs, targets, rows, folds, geometry, support, roi, grid, audits = initialize(args)
    if (args.out / "profile.json").exists():
        raise FileExistsError("profile already exists; inspect instead of repeating")
    with old.writer_lock(args.out):
        tensors = old.load_tensors(args.source, inputs, targets)
        fold_id = max(range(len(folds)), key=lambda i: (len(folds[i]["train_indices"]), audits[i]["train_geometry_frames"]))
        initial = load_initial(args, SEEDS[0], fold_id, folds[fold_id])
        records = {}
        torch.cuda.reset_peak_memory_stats()
        for arm in ARMS:
            model, info = fit_one(tensors, geometry, support, folds[fold_id], audits[fold_id], initial,
                                  SEEDS[0]+fold_id, arm, 200, 20)
            records[arm] = info
            del model
        point_ratio = max(a["train_geometry_frames"] for a in audits)/audits[fold_id]["train_geometry_frames"]
        raw_s = 30*sum(r["localization_s"]*10*point_ratio+r["response_s"]*10+r["feature_cache_s"] for r in records.values())
        estimate = raw_s*1.25+60+read_json(args.out / "protocol.json")["prepare_s"]
        result = {"status": "profile_only_weights_discarded", "fold_index": fold_id,
                  "arms": records, "temporary_optimizer_steps": 440,
                  "heldout_inference_executed": False, "formal_fits_completed": 0,
                  "same_initial_state": True, "extrapolated_fit_s": raw_s,
                  "complete_job_estimate_s": estimate,
                  "localization_max_point_ratio": point_ratio,
                  "estimate_formula": "30 pairs * (loc time*10*max-point-ratio + response time*10 + feature cache) *1.25 +60s +prepare",
                  "default_execution": "user_run" if estimate > 300 else "agent_short_job_allowed",
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}
        write_json(args.out / "profile.json", result)
        print(f"440 temporary updates discarded; whole job estimate={estimate:.1f}s; {result['default_execution']}", flush=True)


def checkpoint_path(out, seed, fold_id, arm):
    return out / "checkpoints" / f"seed{seed}_fold{fold_id:02d}_{arm}.pt"


def validate(saved, seed, fold_id, fold, arm, audit):
    expected = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "fold_index": fold_id,
                "arm": arm, "heldout_episode": fold["heldout_episode"],
                "test_indices": fold["test_indices"], "geometry_audit": audit}
    if any(saved.get(k) != v for k, v in expected.items()) or saved["fit"]["localization_steps"] != 2000 or saved["fit"]["response_steps"] != 200:
        raise ValueError("incomplete or different checkpoint")


def train(args):
    old, torch = runtime()
    old.configure()
    inputs, targets, rows, folds, geometry, support, roi, grid, audits = initialize(args)
    if (args.out / "report.json").exists():
        if args.resume:
            print("Already complete; no optimization or overwrite performed.", flush=True)
            return
        raise FileExistsError("experiment already complete")
    if any((args.out / "checkpoints").glob("*.pt")) and not args.resume:
        raise FileExistsError("partial run exists; --resume preserves completed fits")
    with old.writer_lock(args.out):
        tensors = old.load_tensors(args.source, inputs, targets)
        cache_path = args.source / "feature_cache.pt"
        cache_stat = (cache_path.stat().st_size, cache_path.stat().st_mtime_ns)
        torch.cuda.reset_peak_memory_stats()
        started, new, skipped = time.perf_counter(), 0, 0
        for seed in SEEDS:
            for fold_id, fold in enumerate(folds):
                initial = load_initial(args, seed, fold_id, fold)
                for arm in ARMS:
                    path = checkpoint_path(args.out, seed, fold_id, arm)
                    if path.exists():
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        validate(saved, seed, fold_id, fold, arm, audits[fold_id])
                        skipped += 1
                    else:
                        model, info = fit_one(tensors, geometry, support, fold, audits[fold_id], initial,
                                              seed+fold_id, arm, 2000, 200)
                        test = old.subset(tensors, fold["test_indices"])
                        with torch.no_grad():
                            output = old.forward(model, test)
                        predictions = []
                        for j, i in enumerate(fold["test_indices"]):
                            logit = float(output["window_logit"][j])
                            predictions.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                "sample_index": i, "window_id": rows[i]["window_id"], "ui_index": rows[i]["ui_index"],
                                "source_episode": rows[i]["source_episode"], "task": rows[i]["task"],
                                "response_y": targets[i]["response_y"], "logit": logit,
                                "probability_advance": float(torch.sigmoid(output["window_logit"][j])),
                                "prediction": int(logit >= 0)})
                        locations, maps = [], {}
                        for i in audits[fold_id]["test_geometry_indices"]:
                            meta = geometry[i]
                            j, v, f = fold["test_indices"].index(meta["sample_index"]), VIEWS.index(meta["view"]), meta["frame_index"]
                            p = output["heatmap_logits"][j, f, v].flatten().softmax(-1).cpu().numpy()
                            wh = np.subtract(roi[meta["view"]][2:], roi[meta["view"]][:2])
                            origin = np.asarray(roi[meta["view"]][:2])-.5
                            peak, mean = grid[int(p.argmax())]*wh+origin, (p @ grid)*wh+origin
                            key = f"geometry_{i:02d}"
                            maps[key] = torch.from_numpy(p.reshape(56, 56))
                            locations.append({**meta, "base_seed": seed, "fold_index": fold_id, "arm": arm,
                                "phase": "heldout_episode", "local_array_key": key,
                                "support_mass": float(p[support[i]].sum()),
                                "support_area_fraction": float(support[i].mean()),
                                "peak_in_support": float(support[i, int(p.argmax())]),
                                "peak_distance_px": float(distance_to_geometry(peak, meta["geometry"])),
                                "mean_distance_px": float(distance_to_geometry(mean, meta["geometry"])),
                                "peak_xy_original": peak.tolist(), "mean_xy_original": mean.tolist()})
                        saved = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed,
                            "fold_index": fold_id, "fit_seed": seed+fold_id, "arm": arm,
                            "heldout_episode": fold["heldout_episode"], "test_indices": fold["test_indices"],
                            "geometry_audit": audits[fold_id], "fit": info, "model_state": old.cpu_state(model),
                            "initial_state": initial, "predictions": predictions,
                            "locations": locations, "heatmaps": maps,
                            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False}
                        partial = path.with_suffix(".pt.partial")
                        torch.save(saved, partial)
                        restored = torch.load(partial, map_location="cpu", weights_only=True)
                        from real10_spatial_response import SpatialResponse
                        replay = SpatialResponse().cuda().eval()
                        replay.load_state_dict(restored["model_state"])
                        with torch.no_grad():
                            repeat = old.forward(replay, test)
                        errors = {k: float((output[k]-repeat[k]).abs().max())
                                  for k in ("window_logit", "heatmap_logits", "location_uv")}
                        if any(not np.isfinite(value) or value > 1e-6 for value in errors.values()):
                            raise ValueError("saved checkpoint replay differs")
                        saved["replay_max_abs"] = errors
                        old.atomic_torch_save(saved, path)
                        partial.unlink(missing_ok=True)
                        del model, replay, repeat, output, test
                        new += 1
                    if any(not torch.equal(saved["initial_state"][k], initial[k]) for k in initial):
                        raise ValueError("paired initial state differs from the original fold")
                    print(f"{new+skipped}/60 seed={seed} fold={fold_id} {arm} loc=2000 response=200", flush=True)
        if cache_stat != (cache_path.stat().st_size, cache_path.stat().st_mtime_ns):
            raise ValueError("old feature cache changed")
        write_json(args.out / "train_timing.json", {"new_fits": new, "skipped_complete_fits": skipped,
            "elapsed_s": time.perf_counter()-started, "old_cache_stat_unchanged": True,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
        summarize(args)


def metrics(records):
    matrix = np.zeros((2, 2), dtype=int)
    for row in records:
        matrix[row["response_y"], row["prediction"]] += 1
    support = matrix.sum(1)
    recall = [float(matrix[j, j]/support[j]) if support[j] else None for j in range(2)]
    episodes = sorted({r["source_episode"] for r in records})
    return {"n": len(records), "accuracy": float(matrix.trace()/len(records)),
            "balanced_accuracy": float(np.mean(recall)) if all(support) else None,
            "recall_stationary": recall[0], "recall_advance": recall[1],
            "episode_macro_accuracy": float(np.mean([np.mean([r["prediction"] == r["response_y"]
                for r in records if r["source_episode"] == ep]) for ep in episodes])),
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist()}


def summarize(args):
    _, torch = runtime()
    _, targets, rows, folds, geometry, support, roi, grid, audits = initialize(args)
    predictions, locations, fits, arrays = [], [], [], {}
    for seed in SEEDS:
        for fold_id, fold in enumerate(folds):
            pair_initial = None
            for arm in ARMS:
                saved = torch.load(checkpoint_path(args.out, seed, fold_id, arm),
                                   map_location="cpu", weights_only=True)
                validate(saved, seed, fold_id, fold, arm, audits[fold_id])
                if pair_initial is not None and any(not torch.equal(saved["initial_state"][k], pair_initial[k]) for k in pair_initial):
                    raise ValueError("summary rejects unmatched initial states")
                pair_initial = saved["initial_state"]
                predictions.extend(saved["predictions"])
                fits.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                             "geometry_audit": saved["geometry_audit"], "replay_max_abs": saved["replay_max_abs"],
                             **saved["fit"]})
                for row in saved["locations"]:
                    key = f"map_{len(arrays):03d}"
                    arrays[key] = saved["heatmaps"][row["local_array_key"]].numpy()
                    locations.append({**row, "array_key": key})
    if len(predictions) != 270 or len(locations) != 150:
        raise ValueError("OOF exports incomplete")
    seed_reports, deltas = [], []
    for seed in SEEDS:
        paired = {}
        for arm in ARMS:
            group = [r for r in predictions if r["base_seed"] == seed and r["arm"] == arm]
            if sorted(r["sample_index"] for r in group) != list(range(45)):
                raise ValueError("each window needs exactly one heldout prediction per seed/arm")
            result = metrics(group)
            paired[arm] = result
            seed_reports.append({"base_seed": seed, "arm": arm, **result})
        deltas.append({"base_seed": seed, **{key: paired[ARMS[1]][key]-paired[ARMS[0]][key]
                        for key in ("accuracy", "balanced_accuracy", "episode_macro_accuracy")}})
    means = {arm: {key: {"mean": float(np.mean([r[key] for r in seed_reports if r["arm"] == arm])),
                        "sample_std_ddof1": float(np.std([r[key] for r in seed_reports if r["arm"] == arm], ddof=1))}
                   for key in ("accuracy", "balanced_accuracy", "episode_macro_accuracy",
                               "recall_stationary", "recall_advance")} for arm in ARMS}
    localization_reports = []
    for seed in SEEDS:
        for arm in ARMS:
            for view in VIEWS:
                group = [r for r in locations if (r["base_seed"], r["arm"], r["view"]) == (seed, arm, view)]
                windows = sorted({r["window_id"] for r in group})
                values = {key: float(np.mean([np.mean([r[key] for r in group if r["window_id"] == w])
                                               for w in windows]))
                          for key in ("support_mass", "support_area_fraction", "peak_in_support",
                                      "peak_distance_px", "mean_distance_px")}
                localization_reports.append({"base_seed": seed, "arm": arm, "view": view, **values})
    consistency = all(d["balanced_accuracy"] > 0 for d in deltas) and np.mean([d["episode_macro_accuracy"] for d in deltas]) >= 0
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_jsonl(args.out / "heldout_localization.jsonl", locations)
    write_jsonl(args.out / "fit_records.jsonl", fits)
    np.savez_compressed(args.out / "heldout_heatmaps.npz", **arrays)
    write_json(args.out / "report.json", {
        "schema": SCHEMA, "status": "complete_developmental_loeo", "config": CONFIG,
        "seed_reports": seed_reports, "means": means, "paired_deltas_head_minus_tip": deltas,
        "all_three_ba_deltas_positive_and_mean_episode_delta_nonnegative": bool(consistency),
        "consistency_is_significance_or_deployment_gate": False,
        "localization_reports": localization_reports, "completed_fits": 60,
        "oof_rows": 270, "distinct_windows": 45, "distinct_source_episodes": 10,
        "heldout_geometry_rows": 150, "matched_geometry_frames": 25,
        "confusion_order": ["stationary", "advance"], "test_scope": "reused development folds",
        "all_initial_states_paired": True, "whole_episode_isolation_both_stages": True,
        "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
    print("Complete: 60 fixed fits, 270 OOF responses, 150 heldout geometry records.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train", "summarize"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--labels", type=Path, default=Path("simulation_output/real10_head_segment_annotation_audit_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_head_response_pair_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"profile": profile, "train": train, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
