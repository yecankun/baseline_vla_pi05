"""Guarded fixed12 training implementation; real execution needs separate approval.

The current implementation plan permits preflight and first-batch gradient-only
checks. It does NOT permit public-data optimizer updates. The train stage also
requires a separate, hash-bound positive authorization file not created here.
Old step0/model/loss/sampler/checkpoint primitives are never modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

import run_pi05_libero_action_study_step0 as base
import pi05_libero_action_study_training as training
from audit_pi05_libero_action_study_step0 import audit as audit_step0, REPORT_SHA as STEP0_SHA
from pi05_libero_action_ablation import action_ablation_inputs
from pi05_libero_action_study_sampling import verify_draw_manifest
from pi05_libero_world_model import collate_window_inputs
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, read_json, sha256_file
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_world_model_persistence import persistence_scoring_targets
from smoke_pi05_libero_world_model import parameter_sha as checkpoint_parameter_sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-action-study-training-plan-v1.json"
AUTH = "docs/libero-action-study-training-authorization-v1.json"
STEP0_ROOT = base.OUT_PATH
GRAD_ROOT = "simulation_output/pi05_libero_action_study_gradient_check_v1"
TRAIN_ROOT = "simulation_output/pi05_libero_action_study_training_v1"
STEP0_PLAN_SHA = "2716481652a40b783b55eaec4cd7ec54f6e7de845f0a249d54355b6ffedd21c9"
REQUIRED_PINS = {
    base.PLAN_PATH, STEP0_ROOT + "/report.json", STEP0_ROOT + "/status.json",
    STEP0_ROOT + "/sampling_index.json", STEP0_ROOT + "/evaluation_trace.jsonl",
    *{STEP0_ROOT + f"/draws_seed{s}.json" for s in base.SEEDS},
    "tools/run_pi05_libero_action_study_training.py", "tools/pi05_libero_action_study_training.py",
    "tools/pi05_libero_world_model_training.py", "tools/smoke_pi05_libero_world_model.py",
    "tools/audit_pi05_libero_action_study_step0.py",
}
AUTH_KEYS = {"schema", "training_execution_allowed", "training_plan_sha256", "gradient_check_report_sha256",
             "optimizer_updates_per_arm_seed", "total_optimizer_updates", "output"}
ALLOWED = {"implementation": True, "synthetic_optimizer_checkpoint_tests": True,
           "public_gradient_only_check": True, "public_optimizer_updates": False,
           "new_features": False, "rollout": False}


def load_authorization(root, plan_sha, auth_sha, execute):
    """Fail before model/optimizer/output creation; never generate authorization."""
    base.require(execute is True and auth_sha is not None, "separate authorization SHA and --execute-training required")
    auth = read_json(base.checked(root, AUTH, auth_sha))
    base.require(set(auth) == AUTH_KEYS and auth["schema"] == "libero_action_study_training_authorization_v1"
                 and auth["training_execution_allowed"] is True, "explicit positive fixed-study training authorization required")
    base.require(auth["training_plan_sha256"] == plan_sha and auth["output"] == TRAIN_ROOT,
                 "training authorization plan/output differs")
    for key, value in (("optimizer_updates_per_arm_seed", 200), ("total_optimizer_updates", 1200)):
        base.require(type(auth[key]) is int and auth[key] == value, "fixed authorized update budget required")
    gradient = read_json(base.checked(root, GRAD_ROOT + "/report.json", auth["gradient_check_report_sha256"]))
    base.require(gradient["status"] == "passed_six_first_batch_gradient_checks_no_optimizer"
                 and gradient["training_plan_sha256"] == plan_sha and gradient["optimizer_constructed"] is False
                 and gradient["public_optimizer_steps"] == 0 and gradient["backward_calls"] == 6,
                 "completed matching no-update gradient check required")
    base.require(set(gradient["results"]) == set(map(str, base.SEEDS))
                 and all(set(row) == set(base.ARMS) for row in gradient["results"].values()), "six-arm gradient coverage required")
    for name, digest in gradient["input_sha256"].items():
        base.checked(root, name, digest)
    return auth


def validate(root, plan_sha, *, stage="preflight", auth_sha=None, execute=False):
    root = Path(root).resolve()
    base.require(stage in ("preflight", "gradient_check", "train"), "unknown execution stage")
    plan = read_json(base.checked(root, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_action_study_training_plan_v1" and plan["allowed"] == ALLOWED,
                 "implementation/gradient-only scope changed")
    base.require(set(plan["input_sha256"]) == REQUIRED_PINS, "complete new and reused implementation inventory required")
    base.require(plan["input_sha256"][STEP0_ROOT + "/report.json"] == STEP0_SHA
                 and plan["input_sha256"][base.PLAN_PATH] == STEP0_PLAN_SHA, "frozen step0 reference required")
    base.require(plan["seeds"] == list(base.SEEDS) and plan["arms"] == list(base.ARMS)
                 and plan["gradient_output"] == GRAD_ROOT and plan["training_output"] == TRAIN_ROOT,
                 "fixed arms/seeds/outputs required")
    for name, digest in plan["input_sha256"].items():
        base.checked(root, name, digest)
    auth = None
    if stage == "train":
        auth = load_authorization(root, plan_sha, auth_sha, execute)
    elif stage == "gradient_check":
        base.require(execute is True and auth_sha is None, "explicit gradient-only flag required; no training authorization here")
    else:
        base.require(execute is False and auth_sha is None, "preflight does not accept execution flags")
    old = base.validate(root, root / base.PLAN_PATH, STEP0_PLAN_SHA)
    study = read_json(root / "docs/libero-action-ablation-study-plan-v1.json")
    base.require(plan["future_design"] == study["future_execution_design"], "prespecified future design changed")
    readback = audit_step0(root, STEP0_SHA, replay=True)
    hashes = {**old["input_sha256"], **plan["input_sha256"], PLAN: plan_sha}
    if auth is not None:
        hashes[AUTH] = auth_sha
        hashes[GRAD_ROOT + "/report.json"] = auth["gradient_check_report_sha256"]
    base.require(not torch.cuda.is_initialized(), "CPU-only process required")
    return {"status": "passed_preflight_no_model_or_optimizer", "stage": stage,
            "plan": plan, "training_plan_sha256": plan_sha, "authorization": auth,
            "authorization_sha256": auth_sha, "input_sha256": hashes, "step0_readback": readback}


def runtime_settings():
    base.require(torch.__version__ == "2.10.0+cu128" and not torch.cuda.is_initialized(), "pinned CPU-only runtime required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "runtime setup failed")


def runtime_evidence():
    return {"torch": str(torch.__version__), "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "amp": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32,
            "cuda_initialized": torch.cuda.is_initialized()}


def load_inputs(root, pack):
    split = pack.load_split(root / base.PACK_PATH / "split.json")
    stats = read_json(root / base.PACK_PATH / "preparation/normalization.json")
    datasets = {p: LiberoWindowDataset(pack, split, partition=p, normalization=stats) for p in base.COUNTS}
    windows = {}
    for part, dataset in datasets.items():
        with (root / base.PACK_PATH / f"preparation/{part}_windows.jsonl").open(encoding="utf-8") as stream:
            windows[part] = [json.loads(line) for line in stream]
        base.require(len(dataset) == base.COUNTS[part]
                     and base.canonical(windows[part]) == base.canonical([asdict(w) for w in dataset.windows]), "fixed windows differ")
    groups = {}
    for eid in split["train_episode_indices"]:
        groups.setdefault(pack.episodes[eid]["leakage_group_id"], []).append(eid)
    draws = {}
    for seed in base.SEEDS:
        manifest = read_json(root / STEP0_ROOT / f"draws_seed{seed}.json")
        verify_draw_manifest(manifest, windows["train"], groups)
        base.require(manifest["batches"] == 200 and manifest["batch_size"] == 16, "fixed full draw manifest required")
        draws[seed] = manifest
    report = read_json(root / STEP0_ROOT / "report.json")
    with (root / STEP0_ROOT / "evaluation_trace.jsonl").open(encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    references = {(item["seed"], item["partition"], tuple(item["window_indices"])): item for item in trace}
    base.require(len(references) == len(trace) == 312, "unique complete reference batches required")
    return datasets, stats, draws, report, references


def draw_batch(manifest, dataset, step):
    base.require(type(step) is int and 1 <= step <= 200, "step must be1..200")
    base.require(manifest["batches"] == 200 and manifest["batch_size"] == 16 and len(manifest["draws"]) == 3200,
                 "fixed3200 shared draws required")
    selected = manifest["draws"][(step-1)*16:step*16]
    items = []
    for offset, draw in enumerate(selected):
        base.require(draw["batch_index"] == step-1 and draw["draw_in_batch"] == offset,
                     "sealed batch schedule changed")
        index = draw["window_index"]
        base.require(type(index) is int and 0 <= index < len(dataset)
                     and dataset.windows[index].episode_index == draw["episode_index"], "draw window/episode mismatch")
        items.append(dataset[index])
    inputs = collate_window_inputs([item["inputs"] for item in items])
    targets = collate_window_targets([item["targets"] for item in items])
    return inputs, targets, [row["window_index"] for row in selected]


def numeric_equal(actual, expected):
    """Frozen same-runtime sufficient statistics are deterministic, not tuned."""
    base.require(base.canonical(actual) == base.canonical(expected), "frozen step0 statistics differ")


def evaluate_arm(model, dataset, stats, arm, seed, partition, references, *, verify_step0=False, trace=None):
    """Independent arm evaluation; unlike step0, final parameter bytes may differ."""
    base.require(arm in base.ARMS, "unknown arm")
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    before = base.parameter_hash(model)
    versions = [p._version for p in model.parameters()]
    totals, episodes, ids = base.Totals(stats), {}, {}
    for index, window in enumerate(dataset.windows):
        ids.setdefault(window.episode_index, []).append(index)
    count, started = 0, time.monotonic()
    for episode, indices in ids.items():
        episode_totals = base.Totals(stats)
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[index] for index in selected]
            raw = collate_window_inputs([item["inputs"] for item in items])
            targets = collate_window_targets([item["targets"] for item in items])
            inputs = action_ablation_inputs(raw, arm)
            ref = references[(seed, partition, tuple(selected))]
            source_fp, target_fp, arm_fp = base.fingerprints(raw), base.fingerprints(targets), base.fingerprints(inputs)
            base.require(ref["source_input_sha256"] == source_fp and ref["arm_input_sha256"][arm] == arm_fp
                         and ref["original_target_sha256"] == target_fp, "frozen evaluation inputs/targets changed")
            with torch.inference_mode():
                common = persistence_scoring_targets(raw, targets)
                base.require(base.fingerprints(common) == target_fp == ref["common_target_sha256"], "fixed scoring support changed")
                predictions = model(**inputs)
                prediction_fp = base.fingerprints(predictions)
                if verify_step0:
                    base.require(prediction_fp == ref["predictions_sha256"][arm], "frozen random step0 predictions differ")
                totals.update(predictions, common)
                episode_totals.update(predictions, common)
                base.require(base.fingerprints(common) == target_fp, "scoring targets mutated")
            base.require(base.fingerprints(raw) == source_fp and base.fingerprints(inputs) == arm_fp
                         and base.fingerprints(targets) == target_fp, "evaluation input/target mutation")
            count += len(items)
            if trace is not None:
                trace.write(base.canonical({"seed": seed, "arm": arm, "partition": partition, "step": 0 if verify_step0 else 200,
                            "window_indices": selected, "input_sha256": arm_fp, "target_sha256": target_fp,
                            "prediction_sha256": prediction_fp}) + "\n")
        episodes[str(episode)] = episode_totals.summary()
    base.require(count == len(dataset) and base.parameter_hash(model) == before
                 and versions == [p._version for p in model.parameters()]
                 and all(p.grad is None and not p.requires_grad for p in model.parameters())
                 and all(not module.training for module in model.modules()), "evaluation changed model/coverage")
    return {"per_episode": episodes, "episode_macro": base.macro_metrics(episodes), "micro": totals.summary(),
            "forward_and_scoring_seconds": time.monotonic()-started}


def compare_initial(actual, reference):
    for key in ("per_episode", "episode_macro", "micro"):
        numeric_equal(actual[key], reference[key])


def claim_output(root, relative, preflight):
    out = root / relative
    base.require(not out.exists(), "output exists; no overwrite/resume/retry")
    out.mkdir()
    base.write_json(out / "preflight.json", preflight)
    base.write_json(out / "started.json", {"pid": os.getpid(), "unix_seconds": time.time(), "stage": preflight["stage"]})
    return out


def rehash(root, pins):
    for name, digest in pins.items():
        base.checked(root, name, digest)
    base.require(not torch.cuda.is_initialized(), "unexpected CUDA initialization")


def gradient_check(root, plan_sha, execute=False):
    root = Path(root).resolve()
    base.require(not (root / GRAD_ROOT).exists(), "output exists; no repeat gradient probe")
    preflight = validate(root, plan_sha, stage="gradient_check", execute=execute)
    runtime_settings()
    out = claim_output(root, GRAD_ROOT, preflight)
    began = time.monotonic()
    try:
        results = {}
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            datasets, stats, draws, reference, references = load_inputs(root, pack)
            for seed in base.SEEDS:
                models = base.initialized_pair(seed, pack.manifest["task_registry"])
                raw, targets, indices = draw_batch(draws[seed], datasets["train"], 1)
                pair = base.paired_inputs(raw)
                results[str(seed)] = {}
                for arm, model in models.items():
                    initial_sha = base.parameter_hash(model)
                    base.require(initial_sha == reference["runs"][str(seed)]["train"]["parameter_sha256"][arm], "step0 initial bytes differ")
                    print(f"frozen evaluator replay seed={seed} arm={arm}; optimizer updates=0", flush=True)
                    replay = {}
                    for part, dataset in datasets.items():
                        frozen = evaluate_arm(model, dataset, stats, arm, seed, part, references, verify_step0=True)
                        compare_initial(frozen, reference["runs"][str(seed)][part]["arms"][arm])
                        replay[part] = {"windows": frozen["micro"]["window_count"], "batches": frozen["micro"]["update_count"],
                                        "prediction_and_statistics_match": True,
                                        "forward_and_scoring_seconds": frozen["forward_and_scoring_seconds"]}
                    print(f"gradient-only seed={seed} arm={arm}; optimizer updates=0", flush=True)
                    model.train().requires_grad_(True)
                    probe = training.gradient_probe(model, pair[arm], targets, arm)
                    base.require(base.parameter_hash(model) == initial_sha and all(p.grad is None for p in model.parameters()),
                                 "gradient-only probe changed weights or left gradients")
                    results[str(seed)][arm] = {"probe": probe, "frozen_evaluation_replay": replay,
                        "window_indices": indices, "input_sha256": base.fingerprints(pair[arm]),
                        "target_sha256": base.fingerprints(targets), "step0_parameter_sha256": initial_sha,
                        "checkpoint_parameter_sha256": checkpoint_parameter_sha(model)}
                del models
        rehash(root, preflight["input_sha256"])
        report = {"schema": "libero_action_study_gradient_check_v1", "status": "passed_six_first_batch_gradient_checks_no_optimizer",
                  "training_plan_sha256": plan_sha, "input_sha256": preflight["input_sha256"], "results": results,
                  "optimizer_constructed": False, "public_optimizer_steps": 0, "backward_calls": 6,
                  "public_training_started": False, "parameters_unchanged": True, "gradients_cleared": True,
                  "batch_source": "first16 frozen shared train draws per seed only; no validation backward",
                  "checkpoint_written": False, "wall_seconds": time.monotonic()-began,
                  "runtime": runtime_evidence(),
                  "training_ready": False, "formal_data_allowed": False, "real_system_validated": False}
        base.write_json(out / "report.json", report)
        base.write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json"), "public_optimizer_steps": 0})
        return report
    except BaseException as error:
        base.write_json(out / "failure.json", {"status": "failed_preserve_no_retry", "error": str(error),
                        "traceback": traceback.format_exc(), "public_optimizer_steps": 0})
        raise


def train_arm(model, dataset, manifest, arm, trace):
    model.train().requires_grad_(True)
    optimizer = training.build_optimizer(model, training.OPTIMIZER_CONFIG)
    records, began = [], time.monotonic()
    for step in range(1, 201):
        raw, targets, indices = draw_batch(manifest, dataset, step)
        inputs = action_ablation_inputs(raw, arm)
        before_input, before_target = base.fingerprints(inputs), base.fingerprints(targets)
        details = training.train_step(model, optimizer, inputs, targets, arm=arm, expected_step=step)
        base.require(details["step"] == step and base.fingerprints(inputs) == before_input
                     and base.fingerprints(targets) == before_target, "training counter or input/target mutation")
        record = {"seed": manifest["seed"], "arm": arm, "step": step, "window_indices": indices,
                  "input_sha256": before_input, "target_sha256": before_target, "details": details}
        trace.write(base.canonical(record) + "\n")
        trace.flush()
        records.append(details)
    base.require(len(records) == 200 and [r["step"] for r in records] == list(range(1, 201)), "exactly200 updates required")
    return optimizer, {"optimizer_steps": len(records), "wall_seconds": time.monotonic()-began,
                       "first_step": records[0], "last_step": records[-1]}


def train(root, plan_sha, auth_sha, execute=False):
    """Future authorized job. This turn never calls this function with permission."""
    root = Path(root).resolve()
    base.require(not (root / TRAIN_ROOT).exists(), "output exists; no overwrite/resume/retry")
    preflight = validate(root, plan_sha, stage="train", auth_sha=auth_sha, execute=execute)
    runtime_settings()
    out = claim_output(root, TRAIN_ROOT, preflight)
    started, completed = time.monotonic(), {}
    try:
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            datasets, stats, draws, reference, references = load_inputs(root, pack)
            models = {seed: base.initialized_pair(seed, pack.manifest["task_registry"]) for seed in base.SEEDS}
            # Validate all six initial models and complete evaluation supports
            # before the first optimizer is constructed anywhere in this job.
            with (out / "evaluation_step0.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
                for seed, pair in models.items():
                    for arm, model in pair.items():
                        base.require(base.parameter_hash(model) == reference["runs"][str(seed)]["train"]["parameter_sha256"][arm],
                                     "frozen step0 initialization differs")
                        for part, dataset in datasets.items():
                            initial = evaluate_arm(model, dataset, stats, arm, seed, part, references, verify_step0=True, trace=trace)
                            compare_initial(initial, reference["runs"][str(seed)][part]["arms"][arm])
            with (out / "training_trace.jsonl").open("x", encoding="utf-8", newline="\n") as log, \
                 (out / "evaluation_final.jsonl").open("x", encoding="utf-8", newline="\n") as final_trace:
                for seed, pair in models.items():
                    completed[str(seed)] = {}
                    for arm, model in pair.items():
                        initial_sha = checkpoint_parameter_sha(model)
                        print(f"fixed training seed={seed} arm={arm} updates=200", flush=True)
                        optimizer, timing = train_arm(model, datasets["train"], draws[seed], arm, log)
                        model.zero_grad(set_to_none=True)
                        model.eval().requires_grad_(False)
                        final_sha = checkpoint_parameter_sha(model)
                        base.require(final_sha != initial_sha, "no learned parameter change")
                        draw_sha = reference["sampling"][str(seed)]["sha256"]
                        binding = training.make_checkpoint_binding(seed=seed, arm=arm, study_plan_sha256=base.STUDY_SHA,
                            training_plan_sha256=plan_sha, training_authorization_sha256=auth_sha, sampling_plan_sha256=draw_sha,
                            initial_parameter_sha256=initial_sha, final_parameter_sha256=final_sha)
                        metadata = {"protocol_sha256": base.STUDY_SHA, "authorization_sha256": auth_sha,
                            "source_report_sha256": pack.manifest["source"]["metadata_sha256"], "feature_report_sha256": base.FEATURE_SHA,
                            "manifest_sha256": pack.manifest_sha256, "split_sha256": stats["split_sha256"],
                            "normalization_sha256": sha256_file(root / base.PACK_PATH / "preparation/normalization.json"),
                            "sampling_plan_sha256": draw_sha, "initial_parameter_sha256": initial_sha, "final_parameter_sha256": final_sha,
                            "model_config": asdict(model.config), "task_registry": pack.manifest["task_registry"],
                            "step": 200, "checkpoint_kind": "final_diagnostic_no_resume"}
                        ckpt = out / f"seed{seed}_{arm}_final200.pt"
                        checkpoint_start = time.monotonic()
                        envelope = training.save_final_checkpoint(ckpt, model, optimizer, metadata, binding)
                        base.write_json(ckpt.with_suffix(".json"), envelope)
                        reload_model = base.initialized_pair(seed, pack.manifest["task_registry"])[arm]
                        training.load_final_checkpoint(ckpt, reload_model, envelope, binding, expected_metadata=metadata)
                        raw, _, _ = draw_batch(draws[seed], datasets["train"], 200)
                        check_inputs = action_ablation_inputs(raw, arm)
                        with torch.inference_mode():
                            base.require(base.fingerprints(model(**check_inputs)) == base.fingerprints(reload_model(**check_inputs)),
                                         "checkpoint reload prediction bytes differ")
                        checkpoint_seconds = time.monotonic()-checkpoint_start
                        final = {part: evaluate_arm(reload_model, dataset, stats, arm, seed, part, references, trace=final_trace)
                                 for part, dataset in datasets.items()}
                        row = {"training": timing, "checkpoint": {"path": ckpt.name, "envelope": envelope,
                               "save_reload_verify_seconds": checkpoint_seconds, "prediction_bytes_match": True}, "final": final}
                        completed[str(seed)][arm] = row
                        base.write_json(out / f"seed{seed}_{arm}_result.json", row)
                        del optimizer, reload_model
        rehash(root, preflight["input_sha256"])
        differences, comparisons = [], {}
        for seed in base.SEEDS:
            arms = completed[str(seed)]
            macros = {arm: arms[arm]["final"]["validation"]["episode_macro"] for arm in base.ARMS}
            comparison = {"observed_minus_zero": base.changes(macros[base.ARMS[0]], macros[base.ARMS[1]])}
            for arm in base.ARMS:
                comparison[arm + "_minus_random"] = base.changes(macros[arm], reference["runs"][str(seed)]["validation"]["arms"][arm]["episode_macro"])
                comparison[arm + "_minus_persistence"] = base.changes(macros[arm], reference["runs"][str(seed)]["validation"]["arms"]["persistence"]["episode_macro"])
            comparison["per_validation_episode"] = {}
            for eid in arms[base.ARMS[0]]["final"]["validation"]["per_episode"]:
                values = {arm: base.scalar_metrics(arms[arm]["final"]["validation"]["per_episode"][eid]) for arm in base.ARMS}
                episode_changes = {"observed_minus_zero": base.changes(values[base.ARMS[0]], values[base.ARMS[1]])}
                for arm in base.ARMS:
                    episode_changes[arm + "_minus_random"] = base.changes(values[arm], base.scalar_metrics(reference["runs"][str(seed)]["validation"]["arms"][arm]["per_episode"][eid]))
                    episode_changes[arm + "_minus_persistence"] = base.changes(values[arm], base.scalar_metrics(reference["runs"][str(seed)]["validation"]["arms"]["persistence"]["per_episode"][eid]))
                comparison["per_validation_episode"][eid] = episode_changes
            comparisons[str(seed)] = comparison
            differences.append(comparison["observed_minus_zero"]["normalized_state_mae"]["absolute"])
        benefit = all(value < 0 for value in differences) and all(comparisons[str(seed)][base.ARMS[0]+"_minus_persistence"]["normalized_state_mae"]["absolute"] < 0 for seed in base.SEEDS)
        total = sum(arm["training"]["optimizer_steps"] for row in completed.values() for arm in row.values())
        base.require(total == 1200, "fixed study requires exactly1200 total updates")
        report = {"schema": "libero_action_study_training_result_v1", "status": "completed_fixed_budget_diagnostic",
            "training_plan_sha256": plan_sha, "training_authorization_sha256": auth_sha, "input_sha256": preflight["input_sha256"],
            "runs": completed, "comparisons": comparisons, "total_optimizer_steps": total,
            "primary_paired_macro": {"by_seed": dict(zip(map(str, base.SEEDS), differences)), "mean": statistics.mean(differences),
                                     "descriptive_population_std": statistics.pstdev(differences)},
            "decision": "descriptive_observed_action_predictive_association_only" if benefit else "inconclusive_or_no_stable_action_benefit",
            "visual_regression": {str(seed): comparisons[str(seed)]["observed_minus_zero"]["visual_mae"]["absolute"] > 0 for seed in base.SEEDS},
            "wall_seconds": time.monotonic()-started, "input_files_unchanged": True, "family_independence_verified": False,
            "runtime": runtime_evidence(),
            "checkpoint_training_overlap_unknown": True, "training_ready": False, "formal_data_allowed": False,
            "real_system_validated": False, "causal_or_policy_benefit_claim": False,
            "output_sha256": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}}
        base.write_json(out / "report.json", report)
        base.write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json"), "optimizer_steps": total})
        return report
    except BaseException as error:
        base.write_json(out / "failure.json", {"status": "failed_preserve_no_retry", "error": str(error), "traceback": traceback.format_exc(),
            "completed_arms": completed, "partial_step_evidence": "training_trace.jsonl and any final checkpoints; do not resume or infer zero updates"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gradient_check", "train"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--execute-gradient-check", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    base.require(not (args.execute_gradient_check and args.execute_training), "execution flags are mutually exclusive")
    if args.stage == "preflight":
        result = validate(ROOT, args.plan_sha256, execute=args.execute_gradient_check or args.execute_training, auth_sha=args.authorization_sha256)
        print(base.canonical({"status": result["status"], "hashes_checked": len(result["input_sha256"]), "public_optimizer_steps": 0}))
    elif args.stage == "gradient_check":
        base.require(not args.execute_training and args.authorization_sha256 is None, "gradient-only stage cannot train")
        result = gradient_check(ROOT, args.plan_sha256, args.execute_gradient_check)
        print(base.canonical({"status": result["status"], "wall_seconds": result["wall_seconds"], "public_optimizer_steps": 0}))
    else:
        base.require(not args.execute_gradient_check, "train stage requires separate training authorization")
        result = train(ROOT, args.plan_sha256, args.authorization_sha256, args.execute_training)
        print(base.canonical({"status": result["status"], "total_optimizer_steps": result["total_optimizer_steps"]}))


if __name__ == "__main__":
    main()
