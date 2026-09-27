"""Fixed12 paired action ablation preparation and random step0 evaluation only.

No optimizer import/construction, backward, training switch, feature extraction,
source decoding or rollout. The old model, loss and normalization remain intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from pi05_libero_action_ablation import ACTION_ABLATION_MODES, action_ablation_inputs
from pi05_libero_action_study_sampling import build_draw_manifest, verify_draw_manifest
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, read_json, sha256_file
from pi05_libero_world_model_objectives import NativeWorldModelMetrics, collate_window_targets, native_world_model_loss
from pi05_libero_world_model_persistence import persistence_predictions, persistence_scoring_targets

PLAN_PATH = "docs/libero-action-study-step0-plan-v1.json"
PACK_PATH = "simulation_output/pi05_libero_action_study_features_v1"
OUT_PATH = "simulation_output/pi05_libero_action_study_step0_v1"
STUDY_SHA = "210ff8bbb75e3316620590cda9e5067f7447125acddb53ae7fe751bea01810ac"
FEATURE_SHA = "9e0537b5d520f5e242a02be1f94c454febfec62854a8c43a517f7acdce05ac56"
SEEDS = (20260912, 20260913, 20260914)
COUNTS = {"train": 1086, "validation": 500}
ARMS = ACTION_ABLATION_MODES
REQUIRED_INPUTS = {
    "docs/libero-action-ablation-study-plan-v1.json",
    "docs/libero-action-study-feature-plan-v1.json",
    "docs/libero-native-world-model-learning-protocol-v1.json",
    PACK_PATH + "/report.json", PACK_PATH + "/status.json",
    PACK_PATH + "/manifest.json", PACK_PATH + "/split.json",
    PACK_PATH + "/preparation/normalization.json",
    "tools/run_pi05_libero_action_study_step0.py",
    "tools/pi05_libero_action_study_sampling.py",
    "tools/pi05_libero_action_ablation.py", "tools/pi05_libero_world_model.py",
    "tools/pi05_libero_world_model_adapter.py", "tools/pi05_libero_world_model_objectives.py",
    "tools/pi05_libero_world_model_persistence.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_hash(value):
    require(value.device.type == "cpu", "CPU tensor required")
    h = hashlib.sha256(canonical({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprints(values):
    return {key: tensor_hash(value) if isinstance(value, torch.Tensor) else canonical(value)
            for key, value in sorted(values.items())}


def parameter_hash(model):
    return hashlib.sha256(canonical(fingerprints(dict(model.state_dict()))).encode()).hexdigest()


def paired_inputs(inputs):
    """Owned ordinary tensors; prove the action transform is the only change."""
    before = fingerprints(inputs)
    pair = {arm: action_ablation_inputs(inputs, arm) for arm in ARMS}
    require(fingerprints(inputs) == before, "source inputs mutated")
    require(fingerprints(pair[ARMS[0]]) == before, "observed inputs differ")
    for key in inputs:
        if key != "candidate_actions":
            require(fingerprints({key: pair[ARMS[0]][key]}) == fingerprints({key: pair[ARMS[1]][key]}),
                    f"non-action input differs: {key}")
    require(bool((pair[ARMS[1]]["candidate_actions"] == 0).all()), "zero arm not exactly normalized zero")
    return pair


def initialized_pair(seed, registry):
    require(type(seed) is int and seed in SEEDS, "fixed initialization seed required")
    models = {}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = LiberoWorldModel(LiberoWorldModelConfig(), registry)
        models[arm] = model.cpu().eval().requires_grad_(False)
    require(parameter_hash(models[ARMS[0]]) == parameter_hash(models[ARMS[1]]), "paired initialization mismatch")
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(models[ARMS[0]].parameters(), models[ARMS[1]].parameters())),
            "paired models share parameter storage")
    return models


class Totals:
    def __init__(self, stats):
        self.metrics = NativeWorldModelMetrics(stats["state_std"])
        self.loss = dict(visual_sum=0.0, state_sum=0.0, visual_count=0, state_count=0)

    def update(self, predictions, targets):
        self.metrics.update(predictions, targets)
        _, details = native_world_model_loss(predictions, targets)
        for key in self.loss:
            self.loss[key] += float(details[key]) if key.endswith("sum") else details[key]

    def summary(self):
        result = self.metrics.summary()
        v = self.loss["visual_sum"] / max(1, self.loss["visual_count"])
        s = self.loss["state_sum"] / max(1, self.loss["state_count"])
        result["masked_objective"] = {**self.loss, "visual_loss": v, "state_loss": s, "total": v + 0.25 * s}
        return result


def scalar_metrics(summary):
    return {
        "normalized_state_mae": summary["state_normalized"]["aggregate"]["mae"],
        "normalized_state_rmse": summary["state_normalized"]["aggregate"]["rmse"],
        "visual_mae": summary["visual"]["aggregate"]["mae"],
        "visual_rmse": summary["visual"]["aggregate"]["rmse"],
        "masked_objective": summary["masked_objective"]["total"],
    }


def macro_metrics(episodes):
    rows = [scalar_metrics(value) for value in episodes.values()]
    require(bool(rows), "empty episode metrics")
    # In this explicitly named episode_macro, RMSE means mean of episode RMSEs,
    # not pooled RMSE; the latter remains in micro with sufficient sums/counts.
    return {key: statistics.mean(row[key] for row in rows) if all(row[key] is not None for row in rows) else None
            for key in rows[0]}


def changes(candidate, reference):
    return {key: {"absolute": None if candidate[key] is None or value is None else candidate[key] - value,
                  "relative_percent": None if candidate[key] is None or value in (None, 0) else 100 * (candidate[key] - value) / value}
            for key, value in reference.items()}


def evaluate_pair(models, dataset, stats, batch_size=16, trace=None, partition="validation", seed=None,
                  require_full_support=False):
    """Every window once, episode-homogeneous batches; six inputs only to models."""
    require(type(batch_size) is int and 1 <= batch_size <= 16, "batch size must be 1..16")
    require(set(models) == set(ARMS), "exact paired arms required")
    initial = {arm: parameter_hash(model) for arm, model in models.items()}
    require(len(set(initial.values())) == 1, "step0 requires identical initial model bytes")
    versions = {arm: [p._version for p in model.parameters()] for arm, model in models.items()}
    for model in models.values():
        require(all(not m.training for m in model.modules()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                "all modules eval, parameters frozen and gradients absent required")
    names = (*ARMS, "persistence")
    micro = {name: Totals(stats) for name in names}
    episodes = {name: {} for name in names}
    indices = {}
    for i, window in enumerate(dataset.windows):
        indices.setdefault(window.episode_index, []).append(i)
    timings = {name: 0.0 for name in names}
    counts = {"windows": 0, "batches": 0, "action_values_changed": 0, "original_state_scalars": 0,
              "common_state_scalars": 0, "original_visual_views": 0, "common_visual_views": 0}
    for episode, rows in indices.items():
        accumulators = {name: Totals(stats) for name in names}
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            items = [dataset[i] for i in selected]
            inputs = collate_window_inputs([item["inputs"] for item in items])
            targets = collate_window_targets([item["targets"] for item in items])
            source_before, target_before = fingerprints(inputs), fingerprints(targets)
            pair = paired_inputs(inputs)  # Must be OUTSIDE inference_mode.
            pair_before = {arm: fingerprints(value) for arm, value in pair.items()}
            with torch.inference_mode():
                common = persistence_scoring_targets(inputs, targets)
                common_before = fingerprints(common)
                if require_full_support:
                    require(common_before == target_before, "fixed all-valid pack scoring support changed")
                prediction_hashes = {}
                for name in names:
                    began = time.monotonic()
                    predictions = persistence_predictions(inputs) if name == "persistence" else models[name](**pair[name])
                    require(all(not value.requires_grad and bool(torch.isfinite(value).all()) for value in predictions.values()),
                            "prediction must be finite and gradient-free")
                    prediction_hashes[name] = fingerprints(predictions)
                    micro[name].update(predictions, common)
                    accumulators[name].update(predictions, common)
                    timings[name] += time.monotonic() - began
                require(fingerprints(common) == common_before, "scoring targets changed")
            require(fingerprints(inputs) == source_before and fingerprints(targets) == target_before, "source/targets mutated")
            require({arm: fingerprints(value) for arm, value in pair.items()} == pair_before, "arm inputs mutated")
            counts["windows"] += len(items)
            counts["batches"] += 1
            counts["action_values_changed"] += int((pair[ARMS[0]]["candidate_actions"] != pair[ARMS[1]]["candidate_actions"]).sum())
            for scope, target in (("original", targets), ("common", common)):
                counts[f"{scope}_state_scalars"] += int(target["state_target_valid"].sum())
                counts[f"{scope}_visual_views"] += int(target["future_visual_valid"].sum())
            if trace is not None:
                trace.write(canonical({"seed": seed, "partition": partition, "episode_index": episode,
                                       "window_indices": selected, "source_input_sha256": source_before,
                                       "arm_input_sha256": pair_before, "original_target_sha256": target_before,
                                       "common_target_sha256": common_before, "predictions_sha256": prediction_hashes}) + "\n")
        for name in names:
            episodes[name][str(episode)] = accumulators[name].summary()
    require(counts["windows"] == len(dataset), "incomplete evaluation")
    for arm, model in models.items():
        require(parameter_hash(model) == initial[arm] and [p._version for p in model.parameters()] == versions[arm]
                and all(p.grad is None and not p.requires_grad for p in model.parameters())
                and all(not module.training for module in model.modules()), "model changed during step0")
    results = {name: {"per_episode": episodes[name], "episode_macro": macro_metrics(episodes[name]),
                      "micro": micro[name].summary(), "forward_and_scoring_seconds": timings[name]} for name in names}
    for result in results.values():
        require(result["micro"]["state_normalized"]["aggregate"]["count"] == counts["common_state_scalars"]
                and result["micro"]["visual"]["aggregate"]["count"] == counts["common_visual_views"] * 2048,
                "metric support differs from shared scoring targets")
    comparisons = {"observed_minus_zero": changes(results[ARMS[0]]["episode_macro"], results[ARMS[1]]["episode_macro"])}
    for arm in ARMS:
        comparisons[f"{arm}_minus_persistence"] = changes(results[arm]["episode_macro"], results["persistence"]["episode_macro"])
    return {"arms": results, "episode_macro_changes": comparisons, "coverage": counts,
            "parameter_sha256": initial, "parameters_and_versions_unchanged": True,
            "only_action_input_changed": True, "common_target_support_shared": True,
            "optimizer_steps": 0, "backward_calls": 0}


def checked(root, relative, digest):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")), "safe relative path required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"missing/escaped file: {relative}")
    require(type(digest) is str and len(digest) == 64 and sha256_file(path) == digest, f"SHA mismatch: {relative}")
    return path


def validate(root, plan_path, plan_sha):
    root = Path(root).resolve()
    require(Path(plan_path).resolve() == (root / PLAN_PATH).resolve(), "fixed step0 plan path required")
    checked(root, PLAN_PATH, plan_sha)
    plan = read_json(plan_path)
    require(plan["schema"] == "libero_action_study_step0_plan_v1", "wrong step0 schema")
    require(plan["authorization"] == {"fixed_draws": True, "paired_initialization": True, "step0_evaluation": True,
            "optimizer_construction": False, "optimization": False, "feature_extraction": False, "rollout": False}, "scope changed")
    require(plan["feature_pack"] == PACK_PATH and plan["output"] == OUT_PATH, "fixed input/output paths required")
    require(plan["seeds"] == list(SEEDS) and plan["windows"] == COUNTS, "fixed seeds/window counts required")
    require(plan["model_config"] == asdict(LiberoWorldModelConfig()), "unchanged model required")
    require(set(plan["input_sha256"]) == REQUIRED_INPUTS, "complete fixed input/code inventory required")
    require(plan["input_sha256"]["docs/libero-action-ablation-study-plan-v1.json"] == STUDY_SHA
            and plan["input_sha256"][PACK_PATH + "/report.json"] == FEATURE_SHA, "fixed study/feature report required")
    for name, digest in plan["input_sha256"].items():
        checked(root, name, digest)
    study = read_json(root / "docs/libero-action-ablation-study-plan-v1.json")
    require(plan["selected_split"] == {"train_episode_indices": [1633, 1674, 1419, 1312, 1518, 1520, 1531, 1690],
            "validation_episode_indices": [1530, 1476, 1458, 1566]}, "fixed selected split required")
    require(plan["prospective_batches_per_seed"] == study["future_execution_design"]["optimizer_updates_per_arm_seed"] == 200
            and plan["batch_size"] == study["future_execution_design"]["batch_size"] == 16, "prespecified draw budget required")
    report = read_json(root / PACK_PATH / "report.json")
    require(report["status"] == "passed_frozen_feature_cache_only", "completed cache required")
    hashes = dict(plan["input_sha256"])
    for name, digest in report["output_sha256"].items():
        relative = PACK_PATH + "/" + name
        checked(root, relative, digest)
        hashes[relative] = digest
    require(torch.__version__ == "2.10.0+cu128", "pinned remote torch runtime required")
    return {"plan": plan, "input_sha256": hashes, "plan_sha256": plan_sha,
            "status": "passed_preflight_no_model_or_optimizer", "optimizer_steps": 0}


def run_step0(root, plan_path, plan_sha, execute=False):
    require(execute is True, "explicit --execute-step0 required; no training mode exists")
    root = Path(root).resolve()
    require(not (root / OUT_PATH).exists(), "output exists; preserve it, no overwrite/resume/retry")
    started = time.monotonic()
    preflight = validate(root, plan_path, plan_sha)
    out = root / OUT_PATH
    out.mkdir()  # Atomic claim after read-only preflight.
    write_json(out / "preflight.json", preflight)
    write_json(out / "started.json", {"stage": "step0_only", "optimizer_steps": 0,
                                    "pid": os.getpid(), "started_unix_seconds": time.time()})
    try:
        require(not torch.cuda.is_initialized(), "CPU-only process required; CUDA already initialized")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "runtime settings failed")
        with LiberoFeaturePack(root / PACK_PATH) as pack:
            split = pack.load_split(root / PACK_PATH / "split.json")
            require(all(split[key] == value for key, value in preflight["plan"]["selected_split"].items()), "fixed split changed")
            stats = read_json(root / PACK_PATH / "preparation/normalization.json")
            # Existing constructor independently recomputes train statistics for
            # exact equality validation only; saved statistics are never refit/replaced.
            datasets = {part: LiberoWindowDataset(pack, split, partition=part, normalization=stats) for part in COUNTS}
            windows = {}
            for part, dataset in datasets.items():
                with (root / PACK_PATH / f"preparation/{part}_windows.jsonl").open(encoding="utf-8") as stream:
                    windows[part] = [json.loads(line) for line in stream]
                require(len(dataset) == COUNTS[part] and canonical([asdict(w) for w in dataset.windows]) == canonical(windows[part]),
                        f"fixed complete windows differ: {part}")
            groups = {}
            for eid in split["train_episode_indices"]:
                groups.setdefault(pack.episodes[eid]["leakage_group_id"], []).append(eid)
            draws = {}
            for seed in SEEDS:
                manifest = build_draw_manifest(windows["train"], groups, seed)
                verify_draw_manifest(manifest, windows["train"], groups)
                require(len(manifest["draws"]) == 3200, "exactly 200 prospective batches x16 draws required")
                name = f"draws_seed{seed}.json"
                write_json(out / name, manifest)
                draws[str(seed)] = {"path": name, "sha256": sha256_file(out / name), "draw_count": len(manifest["draws"]),
                                    "both_arms_reference_this_same_file": True, "optimizer_updates_executed": 0}
            # All manifests are sealed before any model is constructed.
            write_json(out / "sampling_index.json", {"groups": groups, "seeds": draws, "prospective_batches_per_seed": 200,
                       "batch_size": 16, "normalization_sha256": sha256_file(root / PACK_PATH / "preparation/normalization.json"),
                       "execution_authorized": False})
            runs = {}
            with (out / "evaluation_trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
                for seed in SEEDS:
                    print(f"step0 seed={seed}: paired random initialization; optimizer updates=0", flush=True)
                    models = initialized_pair(seed, pack.manifest["task_registry"])
                    runs[str(seed)] = {part: evaluate_pair(models, dataset, stats, trace=trace, partition=part, seed=seed,
                                                           require_full_support=True)
                                       for part, dataset in datasets.items()}
                    del models
        for name, digest in preflight["input_sha256"].items():
            checked(root, name, digest)
        checked(root, PLAN_PATH, plan_sha)
        require(not torch.cuda.is_initialized(), "CPU-only run initialized CUDA")
        for seed, entry in draws.items():
            require(sha256_file(out / entry["path"]) == entry["sha256"], "sealed draws changed")
        primary = [runs[str(seed)]["validation"]["episode_macro_changes"]["observed_minus_zero"]["normalized_state_mae"]["absolute"] for seed in SEEDS]
        report = {"schema": "libero_action_study_step0_result_v1", "status": "passed_fixed_draws_paired_random_step0_only",
                  "plan_sha256": plan_sha, "input_sha256": preflight["input_sha256"], "sampling": draws,
                  "runs": runs, "validation_random_observed_minus_zero_mae": {"by_seed": dict(zip(map(str, SEEDS), primary)),
                  "mean": statistics.mean(primary), "descriptive_population_std": statistics.pstdev(primary)},
                  "runtime": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "amp": False, "tf32": False,
                  "cuda_initialized": torch.cuda.is_initialized()},
                  "input_files_unchanged": True, "optimizer_constructed": False, "optimizer_steps": 0, "backward_calls": 0,
                  "training_started": False, "feature_extractions": 0, "policy_actions": 0, "simulation_steps": 0,
                  "trained_checkpoint_written": False, "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
                  "training_ready": False, "formal_data_allowed": False, "real_system_validated": False,
                  "decision": "engineering_step0_pass_only; no learned action benefit, causal, robustness or policy conclusion; training requires separate approval",
                  "episode_macro_semantics": "equal mean over source episodes; macro RMSE is mean episode RMSE, pooled RMSE is micro; three seeds are not new data units",
                  "wall_seconds": time.monotonic() - started,
                  "output_sha256": {path.name: sha256_file(path) for path in sorted(out.iterdir()) if path.is_file()}}
        write_json(out / "report.json", report)
        write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json"), "optimizer_steps": 0})
        return report
    except BaseException as error:
        write_json(out / "failure.json", {"status": "failed_preserve_no_automatic_retry", "error": str(error),
                    "traceback": traceback.format_exc(), "optimizer_steps": 0, "wall_seconds": time.monotonic() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "step0"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute-step0", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.stage == "preflight":
        require(not args.execute_step0, "execution flag invalid for preflight")
        result = validate(root, root / PLAN_PATH, args.plan_sha256)
        print(canonical({"status": result["status"], "input_hashes_checked": len(result["input_sha256"]),
                         "output_exists": (root / OUT_PATH).exists(), "optimizer_steps": 0}))
    else:
        result = run_step0(root, root / PLAN_PATH, args.plan_sha256, args.execute_step0)
        print(canonical({"status": result["status"], "wall_seconds": result["wall_seconds"], "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
