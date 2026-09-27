"""Fixed matched spatial-response experiment; not a policy or world model.

prepare caches frozen features; profile discards its temporary optimizer updates.
train uses final-step checkpoints only; --resume skips completed fold/arm fits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from real10_spatial_response import ARMS, SpatialResponse, response_losses
from run_real10_response_baseline import folds_for, metrics


SCHEMA = "real10_spatial_aux_fixed_pair_v1"
SEEDS = (20261020, 20261120, 20261220)
STEPS = 200
PROFILE_STEPS = 20
VIEWS = ("side", "top")
SNAPSHOTS = ("model_inputs.jsonl", "supervision.jsonl", "annotation_snapshot.jsonl",
             "folds.json", "point_audit.jsonl", "protocol.json", "report.json",
             "preflight.json")
CONFIG = {"base_seeds": list(SEEDS), "fold_seed": "base_seed + zero_based_fold_index",
          "arms": ARMS, "optimizer": "AdamW", "learning_rate": .001,
          "weight_decay": .01, "optimizer_steps_per_fit": STEPS,
          "batch": "all training windows, no heldout graph", "gradient_clip_norm": 1.,
          "threshold": .5, "selection": "final step only", "precision": "FP32, no AMP/TF32",
          "deterministic_algorithms": True, "spatial_map": [64, 56, 56],
          "encoder": "frozen ImageNet ResNet18 IMAGENET1K_V1 through layer1, eval",
          "tip_sigma_grid_cells": 1., "class_weight": "n_negative/n_positive, TRAIN only",
          "consistency_description": "all three paired BA deltas > 0 and mean episode-macro delta >= 0",
          "consistency_is_significance_or_deployment_gate": False}


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("run this fixed experiment in project2026-pi on the remote 4090")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_torch_save(value, path):
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


@contextmanager
def writer_lock(out):
    # POSIX advisory lock releases even after a killed SSH process; no stale lock cleanup.
    import fcntl
    with (out / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is writing this experiment") from exc
        yield


def load_rows(source):
    inputs = read_jsonl(source / "model_inputs.jsonl")
    targets = read_jsonl(source / "supervision.jsonl")
    rows = read_jsonl(source / "annotation_snapshot.jsonl")
    folds = read_json(source / "folds.json")
    if len(inputs) != 45 or len(targets) != 45 or len(rows) != 45 or folds != folds_for(rows):
        raise ValueError("expected the fixed 45 windows and original ten episode folds")
    for i, (inp, target, row) in enumerate(zip(inputs, targets, rows)):
        if inp["sample_index"] != i or target["sample_index"] != i or not (
                inp["window_id"] == target["window_id"] == row["window_id"]):
            raise ValueError("input/target/window ordering differs")
        mi = inp["model_input"]
        if set(mi) != {"sequence", "dt_s", "task_right"}:
            raise ValueError("only ROI pixels, task and observed time may enter forward")
        n = len(mi["sequence"])
        if n < 2 or len(mi["dt_s"]) != n-1 or len(target["tip_valid"]) != n:
            raise ValueError("inconsistent observed sequence lengths")
        if target["response_label"] != row["human_annotation"]["joint_motion_response"]:
            raise ValueError("response snapshot mismatch")
        if target["response_y"] != int(target["response_label"] == "advance"):
            raise ValueError("response label encoding mismatch")
        for coords, valid in zip(target["tip_xy_uv"], target["tip_valid"]):
            for xy, known in zip(coords, valid):
                if known and (xy is None or len(xy) != 2 or not np.isfinite(xy).all()):
                    raise ValueError("known tip needs a finite coordinate")
                if not known and xy is not None:
                    raise ValueError("unknown point must remain null, not pseudo-supervision")
    if sum(t["response_y"] for t in targets) != 25 or sum(
            sum(sum(v) for v in t["tip_valid"]) for t in targets) != 25:
        raise ValueError("fixed response/point counts changed")
    return inputs, targets, rows, folds


def prepare(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve this output; prepare requires a fresh --out")
    inputs, _, rows, _ = load_rows(args.source)
    latest = {r["window_id"]: r for r in read_jsonl(
        Path("simulation_output/real10_event_windows_v1/annotations_joint_v2.jsonl"))}
    if latest != {r["window_id"]: r["human_annotation"] for r in rows}:
        raise ValueError("human labels changed; refresh the preparation explicitly")
    if (args.source / "model_snapshot.py").read_bytes() != Path(
            "tools/real10_spatial_response.py").read_bytes():
        raise ValueError("spatial model differs from the zero-step preparation")
    from PIL import Image
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not weight_path.is_file():
        raise FileNotFoundError(f"no auto-download; cached encoder required: {weight_path}")
    cnn = resnet18(weights=None)
    cnn.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    encoder = torch.nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1)
    encoder.requires_grad_(False).eval().cuda()
    image_paths = sorted({frame["images"][view] for inp in inputs
                          for frame in inp["model_input"]["sequence"] for view in VIEWS})
    if len(image_paths) != 602:
        raise ValueError("expected the original 602 ROI image references")
    args.out.mkdir(parents=True)
    source = args.out / "source"
    source.mkdir()
    for name in SNAPSHOTS:
        shutil.copy2(args.source / name, source / name)
    shutil.copy2(__file__, args.out / "runner_snapshot.py")
    shutil.copy2("tools/real10_spatial_response.py", args.out / "model_snapshot.py")
    shutil.copy2("docs/algorithm-real10-spatial-aux-protocol-20260922.md",
                 args.out / "design_protocol.md")
    write_json(args.out / "run_protocol.json", {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source.as_posix(), "config": CONFIG,
        "windows": 45, "folds": 10, "fits": 60, "total_formal_optimizer_steps": 12000,
        "recognition_not_prediction": True, "policy_input_allowed": False,
        "formal_data_allowed": False, "deployable": False,
        "test_scope": "reused developmental whole-episode LOEO, not independent testing",
        "point_scope": "25 existing tip points in 5 selected windows; no new labels",
        "snapshot_identity": "byte comparison of runner/model; no routine file hashing",
        "software": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "cuda": str(torch.version.cuda), "gpu": torch.cuda.get_device_name(0)},
        "weight_path": str(weight_path), "large_cache_transfer": "keep on 4090; >100 MB user-only"})
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    cached = []
    encoded_at = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(image_paths), 32):
            images = []
            for path in image_paths[offset:offset+32]:
                with Image.open(path) as img:
                    if img.size != (224, 224):
                        raise ValueError("reuse ROI224 without resizing or recropping")
                    images.append(np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)/255.)
            pixels = torch.from_numpy(np.stack(images)).cuda()
            cached.append(encoder((pixels-mean)/std).cpu())
    feature_maps = torch.cat(cached)
    if feature_maps.shape != (602, 64, 56, 56) or not torch.isfinite(feature_maps).all():
        raise ValueError("invalid frozen layer1 cache")
    encoded_s = time.perf_counter() - encoded_at
    atomic_torch_save({"image_paths": image_paths, "maps": feature_maps}, args.out / "feature_cache.pt")
    write_json(args.out / "prepare.json", {"status": "prepared", "formal_training_executed": False,
        "unique_images": 602, "feature_shape": list(feature_maps.shape), "feature_dtype": "float32",
        "encoding_s": encoded_s, "elapsed_s": time.perf_counter()-started,
        "cache_bytes": (args.out / "feature_cache.pt").stat().st_size,
        "encoder_frozen_eval": True, "optimizer_steps": 0})
    print(f"Prepared 602 frozen maps in {time.perf_counter()-started:.2f}s", flush=True)


def check_frozen(out):
    protocol = read_json(out / "run_protocol.json")
    if protocol["schema"] != SCHEMA or protocol["config"] != CONFIG:
        raise ValueError("run configuration changed; do not mix experiments")
    for frozen, current in (("runner_snapshot.py", Path(__file__)),
                            ("model_snapshot.py", Path("tools/real10_spatial_response.py"))):
        if (out / frozen).read_bytes() != current.read_bytes():
            raise ValueError(f"{frozen} differs; inspect before resuming, never silently replace")
    return load_rows(out / "source")


def load_tensors(out, inputs, targets):
    cache = torch.load(out / "feature_cache.pt", map_location="cpu", weights_only=True)
    lookup = {path: i for i, path in enumerate(cache["image_paths"])}
    n, t = len(inputs), max(len(r["model_input"]["sequence"]) for r in inputs)
    indices = torch.zeros((n, t, 2), dtype=torch.long)
    dt = torch.zeros((n, t-1))
    transitions = torch.zeros((n, t-1), dtype=torch.bool)
    xy = torch.zeros((n, t, 2, 2))  # masked placeholders; not model inputs or negative labels
    known = torch.zeros((n, t, 2), dtype=torch.bool)
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        mi = inp["model_input"]
        length = len(mi["sequence"])
        for j, frame in enumerate(mi["sequence"]):
            indices[i, j] = torch.tensor([lookup[frame["images"][v]] for v in VIEWS])
            for v in range(2):
                if target["tip_valid"][j][v]:
                    xy[i, j, v] = torch.tensor(target["tip_xy_uv"][j][v])
                    known[i, j, v] = True
        indices[i, length:] = indices[i, length-1]  # padding only; never a valid transition/target
        dt[i, :length-1] = torch.tensor(mi["dt_s"])
        transitions[i, :length-1] = True
    maps = cache["maps"].cuda()[indices.cuda()]
    return {"maps": maps, "dt_s": dt.cuda(), "transition_valid": transitions.cuda(),
            "task_right": torch.tensor([r["model_input"]["task_right"] for r in inputs], device="cuda"),
            "response_y": torch.tensor([r["response_y"] for r in targets], dtype=torch.float32, device="cuda"),
            "xy_uv": xy.cuda(), "tip_valid": known.cuda()}


def subset(tensors, indices):
    idx = torch.tensor(indices, device="cuda")
    return {k: v[idx] for k, v in tensors.items()}


def forward(model, batch):
    return model(batch["maps"], batch["dt_s"], batch["task_right"], batch["transition_valid"])


def fit(train, seed, arm, steps):
    seed_all(seed)
    model = SpatialResponse().cuda()
    initial = cpu_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    positives = train["response_y"].sum()
    pos_weight = (len(train["response_y"])-positives)/positives
    train_window = torch.ones_like(train["response_y"], dtype=torch.bool)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = response_losses(forward(model, train), train["response_y"], train["xy_uv"],
                               train["tip_valid"], train_window, pos_weight, arm)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter()-started
    final_loss = {k: float(loss[k].detach()) for k in ("total", "response", "localization")}
    return model, initial, {"optimizer_steps": steps, "fit_s": elapsed_s,
                           "last_pre_update_loss": final_loss,
                           "train_pos_weight": float(pos_weight),
                           "train_windows": len(train_window),
                           "train_tip_points": int(train["tip_valid"].sum()),
                           "trainable_parameters": sum(p.numel() for p in model.parameters())}


def profile(args):
    if (args.out / "profile.json").exists():
        raise FileExistsError("profile already completed; read it rather than rerunning")
    inputs, targets, _, folds = check_frozen(args.out)
    tensors = load_tensors(args.out, inputs, targets)
    fold_id = max(range(len(folds)), key=lambda i: len(folds[i]["train_indices"]))
    train = subset(tensors, folds[fold_id]["train_indices"])
    records, initial_states = {}, []
    torch.cuda.reset_peak_memory_stats()
    for arm in ARMS:
        model, initial, record = fit(train, SEEDS[0]+fold_id, arm, PROFILE_STEPS)
        initial_states.append(initial)
        records[arm] = record
        del model
    equal = all(torch.equal(initial_states[0][k], initial_states[1][k]) for k in initial_states[0])
    if not equal:
        raise ValueError("matched arms must start identically")
    seconds_per_pair = sum(r["fit_s"] for r in records.values()) / PROFILE_STEPS * STEPS
    raw_s = seconds_per_pair * len(SEEDS) * len(folds)
    # Largest fold timing + 25% headroom + loading/checkpoint/replay/aggregation overhead.
    estimate_s = raw_s * 1.25 + 60 + read_json(args.out / "prepare.json")["elapsed_s"]
    write_json(args.out / "profile.json", {"status": "profile_complete_not_a_result",
        "fold_index": fold_id, "records": records, "same_initialization": equal,
        "temporary_optimizer_steps": PROFILE_STEPS*len(ARMS), "profile_weights_discarded": True,
        "formal_fits_completed": 0, "heldout_metrics_computed": False,
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "extrapolated_formal_training_s": raw_s, "conservative_complete_job_estimate_s": estimate_s,
        "estimate_formula": "largest-fold pair * 30 * 200/20 * 1.25 + 60s + preparation",
        "default_execution": "user_run" if estimate_s > 300 else "agent_short_job_allowed"})
    print(f"40 temporary updates discarded; full-job estimate {estimate_s/60:.1f} minutes", flush=True)


def checkpoint_path(out, seed, fold_id, arm):
    return out / "checkpoints" / f"seed{seed}_fold{fold_id:02d}_{arm}.pt"


def validate_checkpoint(ckpt, seed, fold_id, fold, arm):
    expected = {"schema": SCHEMA, "base_seed": seed, "fold_index": fold_id,
                "fit_seed": seed+fold_id, "arm": arm, "config": CONFIG,
                "heldout_episode": fold["heldout_episode"], "test_indices": fold["test_indices"]}
    if any(ckpt.get(k) != v for k, v in expected.items()) or ckpt["fit"]["optimizer_steps"] != STEPS:
        raise ValueError("incomplete or mismatched checkpoint; do not silently skip/refit")


def train(args):
    inputs, targets, _, folds = check_frozen(args.out)
    directory = args.out / "checkpoints"
    if directory.exists() and any(directory.glob("*.pt")) and not args.resume:
        raise FileExistsError("completed fits exist; use --resume to preserve them")
    directory.mkdir(exist_ok=True)
    tensors = load_tensors(args.out, inputs, targets)
    torch.cuda.reset_peak_memory_stats()
    started, new_fits, skipped = time.perf_counter(), 0, 0
    for seed in SEEDS:
        for fold_id, fold in enumerate(folds):
            train_batch = subset(tensors, fold["train_indices"])
            test_batch = subset(tensors, fold["test_indices"])
            pair_initial = None
            for arm in ARMS:
                path = checkpoint_path(args.out, seed, fold_id, arm)
                if path.is_file():
                    ckpt = torch.load(path, map_location="cpu", weights_only=True)
                    validate_checkpoint(ckpt, seed, fold_id, fold, arm)
                    skipped += 1
                else:
                    model, initial, timing = fit(train_batch, seed+fold_id, arm, STEPS)
                    model.eval()
                    with torch.no_grad():
                        prediction = forward(model, test_batch)
                    ckpt = {"schema": SCHEMA, "base_seed": seed, "fold_index": fold_id,
                        "fit_seed": seed+fold_id, "arm": arm, "config": CONFIG,
                        "heldout_episode": fold["heldout_episode"], "test_indices": fold["test_indices"],
                        "model_state": cpu_state(model), "initial_state": initial, "fit": timing,
                        "heldout_window_logit": prediction["window_logit"].cpu(),
                        "heldout_location_uv": prediction["location_uv"].cpu()}
                    # Persist and replay before treating this fit as complete.
                    partial = path.with_suffix(".pt.partial")
                    torch.save(ckpt, partial)
                    restored = torch.load(partial, map_location="cpu", weights_only=True)
                    replay_model = SpatialResponse().cuda().eval()
                    replay_model.load_state_dict(restored["model_state"])
                    with torch.no_grad():
                        replay = forward(replay_model, test_batch)
                    errors = {key: float((replay[key]-prediction[key]).abs().max())
                              for key in ("window_logit", "location_uv")}
                    if any(not np.isfinite(v) or v > 1e-6 for v in errors.values()):
                        raise ValueError("saved model replay differs from its heldout export")
                    ckpt["saved_replay_max_abs_error"] = errors
                    atomic_torch_save(ckpt, path)
                    del model, replay_model, prediction, replay
                    new_fits += 1
                if pair_initial is not None and any(not torch.equal(pair_initial[k], ckpt["initial_state"][k])
                                                     for k in pair_initial):
                    raise ValueError("pair initialization mismatch")
                pair_initial = ckpt["initial_state"]
                print(f"{new_fits+skipped}/60 seed={seed} fold={fold_id} {arm} final={STEPS}", flush=True)
            del train_batch, test_batch
    write_json(args.out / "last_train_invocation.json", {"new_fits": new_fits, "skipped_complete_fits": skipped,
        "elapsed_s": time.perf_counter()-started, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
    summarize(args)


def localization_metrics(records):
    result = {}
    for view in VIEWS:
        values = [r for r in records if r["view"] == view]
        windows = sorted({r["window_id"] for r in values})
        per_window = {w: float(np.mean([r["error_px"] for r in values if r["window_id"] == w]))
                      for w in windows}
        result[view] = {"points": len(values), "windows": len(windows),
            "window_macro_error_px": float(np.mean(list(per_window.values()))) if windows else None,
            "point_mean_error_px": float(np.mean([r["error_px"] for r in values])) if values else None,
            "per_window_error_px": per_window}
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"n_seeds": len(a), "mean": float(a.mean()),
            "sample_std_ddof1": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max())}


def summarize(args):
    _, targets, rows, folds = check_frozen(args.out)
    roi = read_json(args.out / "source/protocol.json")["roi_xyxy_original"]
    audit = {(r["window_id"], r["frame_index"], r["view"]): r
             for r in read_jsonl(args.out / "source/point_audit.jsonl") if r["tip_target_valid"]}
    predictions, locations, fit_records, seed_reports = [], [], [], []
    for seed in SEEDS:
        arms = {}
        for arm in ARMS:
            records, point_records, fold_reports = [], [], []
            for fold_id, fold in enumerate(folds):
                path = checkpoint_path(args.out, seed, fold_id, arm)
                if not path.is_file():
                    raise FileNotFoundError(f"experiment incomplete; resume training: {path}")
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                validate_checkpoint(ckpt, seed, fold_id, fold, arm)
                counterpart = torch.load(checkpoint_path(args.out, seed, fold_id, "spatial_window"),
                                         map_location="cpu", weights_only=True)
                if any(not torch.equal(ckpt["initial_state"][k], counterpart["initial_state"][k])
                       for k in ckpt["initial_state"]):
                    raise ValueError("summary cannot accept unmatched initialization")
                fit_records.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                    **ckpt["fit"], "saved_replay_max_abs_error": ckpt["saved_replay_max_abs_error"]})
                fold_points = []
                for j, index in enumerate(fold["test_indices"]):
                    row, target = rows[index], targets[index]
                    logit = float(ckpt["heldout_window_logit"][j])
                    if not np.isfinite(logit):
                        raise ValueError("nonfinite heldout score")
                    record = {"base_seed": seed, "fold_index": fold_id, "arm": arm,
                        "sample_index": index, "window_id": row["window_id"], "ui_index": row["ui_index"],
                        "source_episode": row["source_episode"], "task": row["task"],
                        "response_y": target["response_y"], "logit": logit,
                        "probability_advance": float(torch.sigmoid(torch.tensor(logit))),
                        "prediction": int(logit >= 0)}
                    records.append(record)
                    for frame_index, mask in enumerate(target["tip_valid"]):
                        for v, valid in enumerate(mask):
                            if not valid:
                                continue
                            view = VIEWS[v]
                            x0, y0, x1, y1 = roi[view]
                            predicted_uv = ckpt["heldout_location_uv"][j, frame_index, v].numpy()
                            target_uv = np.asarray(target["tip_xy_uv"][frame_index][v])
                            predicted_xy = predicted_uv*np.array([x1-x0, y1-y0])+[x0-.5, y0-.5]
                            label = audit[row["window_id"], frame_index, view]
                            error = float(np.linalg.norm((predicted_uv-target_uv)*[x1-x0, y1-y0]))
                            if not np.isfinite(error):
                                raise ValueError("nonfinite heldout location")
                            fold_points.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                "window_id": row["window_id"], "ui_index": row["ui_index"],
                                "frame_index": frame_index, "frame_id": label["frame_id"], "view": view,
                                "target_xy_original": label["wire"]["xy"],
                                "predicted_xy_original": predicted_xy.tolist(), "error_px": error})
                point_records.extend(fold_points)
                fold_reports.append({"heldout_episode": fold["heldout_episode"],
                                     "localization": localization_metrics(fold_points)})
            records.sort(key=lambda r: r["sample_index"])
            if [r["sample_index"] for r in records] != list(range(45)) or len(point_records) != 25:
                raise ValueError("OOF window or heldout point coverage incomplete/duplicated")
            by_episode = {}
            for episode in sorted({r["source_episode"] for r in records}):
                subset_rows = [r for r in records if r["source_episode"] == episode]
                by_episode[episode] = metrics(np.array([r["response_y"] for r in subset_rows]),
                                              np.array([r["prediction"] for r in subset_rows]))
            arms[arm] = {**metrics(np.array([r["response_y"] for r in records]),
                                   np.array([r["prediction"] for r in records])),
                "episode_macro_accuracy": float(np.mean([r["accuracy"] for r in by_episode.values()])),
                "by_episode": by_episode, "localization": localization_metrics(point_records),
                "fold_localization": fold_reports}
            predictions.extend(records)
            locations.extend(point_records)
        baseline, auxiliary = (arms[a] for a in ARMS)
        deltas = {key: auxiliary[key]-baseline[key] for key in
                  ("accuracy", "balanced_accuracy", "episode_macro_accuracy")}
        seed_reports.append({"base_seed": seed, "arms": arms, "paired_aux_minus_window": deltas})
    aggregate = {arm: {key: describe([s["arms"][arm][key] for s in seed_reports]) for key in
                      ("accuracy", "balanced_accuracy", "episode_macro_accuracy")} for arm in ARMS}
    paired = {key: describe([s["paired_aux_minus_window"][key] for s in seed_reports]) for key in
              ("accuracy", "balanced_accuracy", "episode_macro_accuracy")}
    consistent = all(s["paired_aux_minus_window"]["balanced_accuracy"] > 0 for s in seed_reports) and (
        paired["episode_macro_accuracy"]["mean"] >= 0)
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_jsonl(args.out / "oof_tip_locations.jsonl", locations)
    write_jsonl(args.out / "fit_records.jsonl", fit_records)
    write_json(args.out / "report.json", {"schema": SCHEMA, "status": "complete_fixed_pair",
        "formal_fits_completed": 60, "optimizer_steps": 12000, "independent_windows": 45,
        "source_episodes": 10, "prediction_rows": len(predictions), "point_prediction_rows": len(locations),
        "per_seed": seed_reports, "across_seeds": aggregate, "paired_aux_minus_window": paired,
        "descriptive_consistency_condition_met": consistent,
        "total_fit_s": sum(r["fit_s"] for r in fit_records),
        "saved_replay_max_abs_error": max(max(r["saved_replay_max_abs_error"].values()) for r in fit_records),
        "checkpoint_selection": "final-step only, no best fold/seed selection or deployment checkpoint",
        "visual_status": "not_viewed", "policy_input_allowed": False, "deployable": False,
        "independent_generalization_validated": False,
        "limitations": ["reused developmental LOEO; all source episodes used by original PI05",
                        "25 tip points in five selected windows, left 1/right 4; not independent localization validation",
                        "three seeds are repeated fits, not 135 independent samples",
                        "post-action recognition only, not contact, action-effect causality or policy success"]})
    print(f"Complete: 60 final fits, paired mean BA delta={paired['balanced_accuracy']['mean']*100:+.2f} pp", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "profile", "train", "summarize"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        configure()
        prepare(args)
    else:
        if args.stage != "summarize":
            configure()
        with writer_lock(args.out):
            {"profile": profile, "train": train, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
