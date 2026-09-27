"""One fixed paired experiment: retain regional moments vs broadcast a global summary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

import train_real10_spatial_aux_pair as old
import train_real10_head_response_pair as previous
from fit_real10_head_region import read_json, read_jsonl, write_json, write_jsonl
from real10_region_response import (
    ARMS, INPUT_DIM, STD_FLOOR, RegionResponse, select_representation, temporal_features)
from real10_spatial_response import spatial_grid

SCHEMA = "real10_region_response_pair_v1"
SEEDS = previous.SEEDS
CASES = (4, 39, 62, 65, 77, 100)
STEPS = 200
CONFIG = {"base_seeds": list(SEEDS), "arms": list(ARMS), "steps": STEPS,
          "fit_seed": "base_seed + fold_index", "region_grid": [4, 4],
          "token_channels": 11, "response_input": INPUT_DIM, "hidden": 8,
          "trainable_parameters_per_arm": 8481, "frozen_localizer_parameters": 529,
          "normalization": "per feature, training valid transitions only, population std",
          "std_floor": STD_FLOOR, "optimizer": "AdamW", "lr": .001, "weight_decay": .01,
          "gradient_clip": 1., "loss": "training-class-weighted BCE, window-only",
          "temporal_input": "anchor + anchor delta + adjacent delta + task + observed dt/1.5",
          "time_pooling": "unchanged mean valid local logits", "threshold": .5,
          "selection": "fixed final step, all seeds/folds; no tuning",
          "localizer": "matching seed/fold previous head_frozen project/location only",
          "new_localization_updates": 0, "precision": "deterministic FP32, no AMP/TF32"}
DESIGN = Path("docs/algorithm-real10-region-response-protocol-20260923.md")
DEPENDENCIES = {
    "runner_snapshot.py": Path(__file__),
    "region_model_snapshot.py": Path("tools/real10_region_response.py"),
    "original_model_snapshot.py": Path("tools/real10_spatial_response.py"),
    "previous_runner_snapshot.py": Path("tools/train_real10_head_response_pair.py"),
    "cache_runner_snapshot.py": Path("tools/train_real10_spatial_aux_pair.py"),
    "io_geometry_snapshot.py": Path("tools/fit_real10_head_region.py"),
    "design_protocol.md": DESIGN}


def file_stat(path):
    s = path.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}


def initialize(args):
    inputs, targets, rows, folds = old.check_frozen(args.source)
    source_protocol = read_json(args.localizers / "protocol.json")
    source_report = read_json(args.localizers / "report.json")
    audits = read_json(args.localizers / "fold_geometry_audit.json")
    geometry = read_jsonl(args.localizers / "matched_geometry.jsonl")
    if (source_protocol["config"] != previous.CONFIG or source_protocol["folds"] != folds or
            source_report["completed_fits"] != 60 or len(geometry) != 25):
        raise ValueError("require the completed original head pair and identical episode folds")
    if audits != previous.fold_geometry_audit(geometry, folds):
        raise ValueError("source localizer geometry assignment differs from current folds")
    sources = {"source/"+p.name: p for p in (args.localizers / "response_source").iterdir()}
    for dest, path in sources.items():
        if path.read_bytes() != (args.source / "source" / path.name).read_bytes():
            raise ValueError(f"localizer and response source differ: {dest}")
    sources.update({"localizer_protocol.json": args.localizers / "protocol.json",
                    "localizer_geometry_audit.json": args.localizers / "fold_geometry_audit.json",
                    "localizer_matched_geometry.jsonl": args.localizers / "matched_geometry.jsonl"})
    protected = {str(args.source / "feature_cache.pt"): file_stat(args.source / "feature_cache.pt")}
    for seed in SEEDS:
        for fold_id in range(10):
            path = previous.checkpoint_path(args.localizers, seed, fold_id, "head_frozen")
            protected[str(path)] = file_stat(path)
    if args.out.exists():
        protocol = read_json(args.out / "protocol.json")
        if (protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix() or
                protocol["localizers"] != args.localizers.as_posix() or protocol["protected_files"] != protected):
            raise ValueError("existing protocol or protected cache/localizers changed")
        for name, path in {**DEPENDENCIES, **sources}.items():
            if (args.out / name).read_bytes() != path.read_bytes():
                raise ValueError(f"frozen dependency changed: {name}")
    else:
        (args.out / "checkpoints").mkdir(parents=True)
        (args.out / "source").mkdir()
        for name, path in {**DEPENDENCIES, **sources}.items():
            shutil.copy2(path, args.out / name)
        write_json(args.out / "protocol.json", {
            "schema": SCHEMA, "config": CONFIG, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": args.source.as_posix(), "localizers": args.localizers.as_posix(),
            "folds": folds, "protected_files": protected, "windows": 45, "source_episodes": 10,
            "class_counts": {"stationary": 20, "advance": 25}, "shared_geometry_pool": 25,
            "formal_fits": 60, "new_optimizer_updates": 12000,
            "test_scope": "reused developmental whole-episode LOEO",
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
    return inputs, targets, rows, folds, audits


def load_pair(args, seed, fold_id, fold, audit):
    path = previous.checkpoint_path(args.localizers, seed, fold_id, "head_frozen")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    previous.validate(saved, seed, fold_id, fold, "head_frozen", audit)
    if not saved["fit"]["localization_unchanged_during_response"]:
        raise ValueError("the source localization branch was not frozen")
    old.seed_all(seed+fold_id)
    model = RegionResponse().cuda().eval()
    model.load_localizer(saved["model_state"])
    initial = old.cpu_state(model.response)
    localizer = {k: v.clone() for k, v in saved["model_state"].items()
                 if k.startswith(("project.", "location."))}
    return model, initial, localizer, str(path)


def extract(model, batch):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.no_grad():
        tokens = model.region_tokens(batch["maps"])
        features = {arm: temporal_features(tokens, batch["dt_s"], batch["task_right"], arm) for arm in ARMS}
    torch.cuda.synchronize()
    return tokens, features, time.perf_counter()-tick


def full_forward(model, batch, arm):
    return model(batch["maps"], batch["dt_s"], batch["task_right"], batch["transition_valid"], arm)


def fit(model, initial, localizer, features, train, arm, steps):
    model.response.load_state_dict(initial)
    model.fit_normalizer(features, train["transition_valid"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 8481:
        raise ValueError("unexpected paired parameter budget")
    positives = train["response_y"].sum()
    pos_weight = (len(train["response_y"])-positives)/positives
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)

    def loss():
        return F.binary_cross_entropy_with_logits(
            model.readout(features, train["transition_valid"])["window_logit"],
            train["response_y"], pos_weight=pos_weight)

    with torch.no_grad():
        curve = [{"step": 0, "loss": float(loss())}]
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == steps:
            with torch.no_grad():
                curve.append({"step": step, "loss": float(loss())})
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-tick
    if any(not torch.equal(model.state_dict()[k].cpu(), value) for k, value in localizer.items()):
        raise ValueError("frozen shared localizer changed")
    with torch.no_grad():
        cached = model.readout(features, train["transition_valid"])["window_logit"]
        full = full_forward(model, train, arm)["window_logit"]
        cache_error = float((cached-full).abs().max())
        if cache_error > 1e-6 or not torch.isfinite(cached).all():
            raise ValueError("cached and end-to-end response differ")
        pred = cached >= 0
        target = train["response_y"].bool()
        train_ba = float(((pred[~target] == 0).float().mean()+(pred[target] == 1).float().mean())/2)
    return {"optimizer_steps": steps, "fit_s": elapsed, "curve": curve,
            "train_windows": len(train["response_y"]), "normalization_transitions": int(train["transition_valid"].sum()),
            "train_pos_weight": float(pos_weight), "train_accuracy": float((pred == target).float().mean()),
            "train_balanced_accuracy": train_ba, "trainable_parameters": 8481,
            "frozen_localizer_parameters": 529, "localizer_unchanged": True,
            "cached_full_forward_max_abs": cache_error}


def profile(args):
    tick = time.perf_counter()
    inputs, targets, _, folds, audits = initialize(args)
    if (args.out / "profile.json").exists():
        raise FileExistsError("profile is already recorded; do not rerun")
    with old.writer_lock(args.out):
        tensors = old.load_tensors(args.source, inputs, targets)
        fold_id = max(range(10), key=lambda i: len(folds[i]["train_indices"]))
        fold = folds[fold_id]
        train = old.subset(tensors, fold["train_indices"])
        model, initial, localizer, _ = load_pair(args, SEEDS[0], fold_id, fold, audits[fold_id])
        torch.cuda.reset_peak_memory_stats()
        tokens, features, extract_s = extract(model, train)
        with torch.no_grad():
            one = train["maps"][:1]
            n,t,v,c,h,w = one.shape
            z = model.project(one.reshape(n*t*v,c,h,w))
            p = model.location(z).flatten(2).softmax(-1).reshape(-1,1,h,w)
            xy = spatial_grid(z.device,z.dtype).permute(2,0,1).unsqueeze(0)
            direct = torch.cat(((p*z).sum((2,3)),(p*xy).sum((2,3)),p.sum((2,3))),1).reshape(n,t,v,11)
            conservation = float((tokens[:1].sum(-2)-direct).abs().max())
            collapsed = select_representation(tokens, ARMS[0])
            broadcast_error = float((collapsed-collapsed[...,:1,:]).abs().max())
            global_sum_error = float((collapsed.sum(-2)-tokens.sum(-2)).abs().max())
            regional_spread = float((tokens-tokens.mean(-2,keepdim=True)).square().mean().sqrt())
        if conservation > 1e-5 or global_sum_error > 1e-5 or broadcast_error != 0 or regional_spread <= 0:
            raise ValueError("the representation ablation is not behaving as specified")
        prepared_s = time.perf_counter()-tick
        records = {arm: fit(model, initial, localizer, features[arm], train, arm, 20) for arm in ARMS}
        estimate = (30*(sum(r["fit_s"]*10 for r in records.values())+2*extract_s))*1.25+60+prepared_s
        write_json(args.out / "profile.json", {
            "status": "profile_only_weights_discarded", "fold_index": fold_id, "records": records,
            "temporary_optimizer_updates": 40, "formal_fits_completed": 0, "heldout_inference_executed": False,
            "shared_initial_state": True, "same_localizer": True, "moment_conservation_max_abs": conservation,
            "global_sum_max_abs": global_sum_error, "broadcast_slot_difference": broadcast_error,
            "regional_slot_rms_spread": regional_spread, "input_shape": list(features[ARMS[0]].shape),
            "extract_pair_s": extract_s, "prepare_s": prepared_s, "complete_job_estimate_s": estimate,
            "estimate_formula": "30 pairs * (two 20-step timings * 10 + 2*extraction) *1.25 +60s +prepare",
            "default_execution": "user_run" if estimate > 300 else "agent_short_job_allowed",
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
        print(f"Profile passed; weights discarded. Whole job estimate={estimate:.1f}s.", flush=True)


def checkpoint_path(out, seed, fold_id, arm):
    return out / "checkpoints" / f"seed{seed}_fold{fold_id:02d}_{arm}.pt"


def validate_checkpoint(saved, seed, fold_id, fold, arm, source_path):
    expected = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "fold_index": fold_id,
                "fit_seed": seed+fold_id, "heldout_episode": fold["heldout_episode"],
                "arm": arm, "test_indices": fold["test_indices"], "localizer_source": source_path}
    if any(saved.get(k) != v for k,v in expected.items()) or saved["fit"]["optimizer_steps"] != STEPS:
        raise ValueError("incomplete or incompatible response checkpoint")


def train(args):
    inputs, targets, rows, folds, audits = initialize(args)
    if (args.out / "report.json").exists():
        if args.resume:
            print("Already complete; no optimization or overwrite.", flush=True)
            return
        raise FileExistsError("preserve the completed experiment")
    if any((args.out / "checkpoints").glob("*.pt")) and not args.resume:
        raise FileExistsError("use --resume to keep completed fits")
    with old.writer_lock(args.out):
        tick = time.perf_counter()
        tensors = old.load_tensors(args.source, inputs, targets)
        torch.cuda.reset_peak_memory_stats()
        new, skipped = 0, 0
        for seed in SEEDS:
            for fold_id,fold in enumerate(folds):
                model, initial, localizer, source_path = load_pair(args, seed, fold_id, fold, audits[fold_id])
                train_batch = old.subset(tensors, fold["train_indices"])
                _, features, extract_s = extract(model, train_batch)
                for arm in ARMS:
                    path = checkpoint_path(args.out, seed, fold_id, arm)
                    if path.exists():
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        validate_checkpoint(saved, seed, fold_id, fold, arm, source_path)
                        skipped += 1
                    else:
                        info = fit(model, initial, localizer, features[arm], train_batch, arm, STEPS)
                        info["shared_training_extraction_s"] = extract_s
                        test = old.subset(tensors, fold["test_indices"])
                        with torch.no_grad():
                            output = full_forward(model, test, arm)
                        predictions, cases = [], []
                        for j,i in enumerate(fold["test_indices"]):
                            length = len(inputs[i]["model_input"]["sequence"])
                            logit = float(output["window_logit"][j])
                            predictions.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                "sample_index": i, "window_id": rows[i]["window_id"], "ui_index": rows[i]["ui_index"],
                                "source_episode": rows[i]["source_episode"], "response_y": targets[i]["response_y"],
                                "task": rows[i]["task"], "logit": logit, "prediction": int(logit >= 0),
                                "probability_advance": float(torch.sigmoid(output["window_logit"][j])),
                                "local_logits": output["local_logits"][j,:length-1].cpu().tolist()})
                            if rows[i]["ui_index"] in CASES:
                                mass = select_representation(output["region_tokens"][j:j+1,:length], arm)[0,...,-1]
                                cases.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                    "ui_index": rows[i]["ui_index"], "source_episode": rows[i]["source_episode"],
                                    "response_y": targets[i]["response_y"], "frames": inputs[i]["model_input"]["sequence"],
                                    "region_mass": mass.cpu().tolist(), "probability_is_wire_confidence": False})
                        saved = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "fold_index": fold_id,
                            "fit_seed": seed+fold_id, "arm": arm, "heldout_episode": fold["heldout_episode"],
                            "test_indices": fold["test_indices"], "train_indices": fold["train_indices"],
                            "geometry_audit": audits[fold_id], "localizer_source": source_path,
                            "model_state": old.cpu_state(model), "response_initial": initial, "fit": info,
                            "predictions": predictions, "case_records": cases,
                            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False}
                        temporary = path.with_suffix(".pt.partial")
                        torch.save(saved, temporary)
                        reload = torch.load(temporary, map_location="cpu", weights_only=True)
                        replay = RegionResponse().cuda().eval()
                        replay.load_state_dict(reload["model_state"])
                        with torch.no_grad():
                            repeat = full_forward(replay, test, arm)
                        errors = {key: float((output[key]-repeat[key]).abs().max())
                                  for key in ("window_logit", "local_logits", "region_tokens")}
                        if any(not np.isfinite(e) or e > 1e-6 for e in errors.values()):
                            raise ValueError("checkpoint replay differs")
                        saved["replay_max_abs"] = errors
                        old.atomic_torch_save(saved, path)
                        # atomic_torch_save uses and renames this same .pt.partial path.
                        new += 1
                        del replay, reload, repeat, output, test
                    if any(not torch.equal(saved["response_initial"][k], v) for k,v in initial.items()):
                        raise ValueError("paired response initial state changed")
                    print(f"{new+skipped}/60 seed={seed} fold={fold_id} {arm} response_steps=200", flush=True)
                del model, train_batch, features
        protected = read_json(args.out / "protocol.json")["protected_files"]
        if any(file_stat(Path(name)) != value for name,value in protected.items()):
            raise ValueError("old cache or source checkpoint changed")
        write_json(args.out / "train_timing.json", {"new_fits": new, "skipped_complete_fits": skipped,
            "elapsed_s": time.perf_counter()-tick, "protected_files_unchanged": True,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
        summarize(args)


def summarize(args):
    _, _, _, folds, _ = initialize(args)
    predictions, fits, cases = [], [], []
    for seed in SEEDS:
        for fold_id,fold in enumerate(folds):
            initial, localizer = None, None
            source_path = str(previous.checkpoint_path(args.localizers, seed, fold_id, "head_frozen"))
            for arm in ARMS:
                saved = torch.load(checkpoint_path(args.out,seed,fold_id,arm), map_location="cpu", weights_only=True)
                validate_checkpoint(saved,seed,fold_id,fold,arm,source_path)
                current = {k:v for k,v in saved["model_state"].items() if k.startswith(("project.","location."))}
                if initial is not None and (any(not torch.equal(saved["response_initial"][k],v) for k,v in initial.items()) or
                                             any(not torch.equal(current[k],v) for k,v in localizer.items())):
                    raise ValueError("summary rejects unmatched initial states/localizers")
                initial, localizer = saved["response_initial"], current
                predictions.extend(saved["predictions"])
                cases.extend(saved["case_records"])
                fits.append({"base_seed":seed,"fold_index":fold_id,"arm":arm,
                             "geometry_audit":saved["geometry_audit"],"replay_max_abs":saved["replay_max_abs"],**saved["fit"]})
    if len(predictions) != 270 or len(cases) != 36:
        raise ValueError("incomplete fixed paired exports")
    metrics = ("accuracy","balanced_accuracy","episode_macro_accuracy","recall_stationary","recall_advance")
    seed_reports, deltas = [], []
    for seed in SEEDS:
        pair = {}
        for arm in ARMS:
            group = [r for r in predictions if (r["base_seed"],r["arm"]) == (seed,arm)]
            if sorted(r["sample_index"] for r in group) != list(range(45)):
                raise ValueError("require exactly one heldout prediction per window")
            pair[arm] = previous.metrics(group)
            seed_reports.append({"base_seed":seed,"arm":arm,**pair[arm]})
        deltas.append({"base_seed":seed,**{k:pair[ARMS[1]][k]-pair[ARMS[0]][k] for k in metrics}})
    means = {arm:{k:{"mean":float(np.mean([r[k] for r in seed_reports if r["arm"]==arm])),
                         "sample_std_ddof1":float(np.std([r[k] for r in seed_reports if r["arm"]==arm],ddof=1))}
                  for k in metrics} for arm in ARMS}
    consistent = (all(r["balanced_accuracy"] > 0 for r in deltas) and
                  all(np.mean([r[k] for r in deltas]) >= 0 for k in ("episode_macro_accuracy","recall_stationary")))
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_jsonl(args.out / "fit_records.jsonl", fits)
    write_jsonl(args.out / "case_regions.jsonl", cases)
    write_json(args.out / "report.json", {"schema":SCHEMA,"status":"complete_developmental_loeo",
        "config":CONFIG,"completed_fits":60,"optimizer_updates":12000,"oof_rows":270,"case_records":36,
        "distinct_windows":45,"distinct_source_episodes":10,"seed_reports":seed_reports,"means":means,
        "paired_deltas_region_minus_global":deltas,"descriptive_consistency_condition_met":bool(consistent),
        "consistency_is_significance_or_deployment_gate":False,"same_parameter_counts_and_initial_states":True,
        "same_fold_isolated_localizers":True,"confusion_order":["stationary","advance"],
        "test_scope":"reused developmental whole-episode LOEO",
        "policy_input_allowed":False,"formal_data_allowed":False,"deployable":False})
    print("Complete: 60 fixed response fits, 270 OOF responses, 36 hardcase region records.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("profile","train","summarize"),required=True)
    parser.add_argument("--source",type=Path,default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--localizers",type=Path,default=Path("simulation_output/real10_head_response_pair_v1"))
    parser.add_argument("--out",type=Path,default=Path("simulation_output/real10_region_response_pair_v1"))
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    old.configure()
    {"profile":profile,"train":train,"summarize":summarize}[args.stage](args)


if __name__ == "__main__":
    main()
