"""Fixed clean/low-contrast observation evaluation, no optimization or rollout.

GPU feature extraction and CPU checkpoint evaluation run in separate processes.
Every output is exclusive. Missing/drifted references fail closed; no auto retry.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

import run_pi05_libero_action_study_step0 as base
import run_pi05_libero_action_study_training as previous
import pi05_libero_action_study_training as checkpoint_io
from pi05_libero_action_study_low_contrast import CONDITIONS, VALIDATION_COUNTS, evaluate_condition, summarize, validate_feature_rows
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-action-study-low-contrast-plan-v1.json"
FEATURE_OUT = "simulation_output/pi05_libero_action_study_low_contrast_features_v1"
EVAL_OUT = "simulation_output/pi05_libero_action_study_low_contrast_eval_v1"
TRAIN_REPORT = previous.TRAIN_ROOT + "/report.json"
TRAIN_SHA = "837cb7b8df45a3b7230e979f9378d91b0177242e98449f6a60ed1608ad2fe5bf"
CORRUPTION = {"name": "low_contrast", "severity": 2, "alpha": 0.45, "seed": 20260911,
              "boundary": "stored_uint8_RGB_before_unchanged_native_preprocessing",
              "targets": "original_clean_future_visual_and_state_targets",
              "formula": "per_channel_spatial_mean + alpha * (pixel - per_channel_spatial_mean)",
              "rounding": "float32_unit_interval_then_numpy_rint_uint8"}
ALLOW = {"validation_feature_extraction": True, "frozen_checkpoint_evaluation": True,
         "optimizer_updates": False, "normalization_refit": False, "checkpoint_selection": False,
         "policy_rollout": False, "project_data_inspection": False}
NEW_CODE = {
    "tools/run_pi05_libero_action_study_low_contrast.py",
    "tools/pi05_libero_action_study_low_contrast.py",
    "tools/pi05_libero_action_study_low_contrast_features.py",
    "tools/build_pi05_libero_action_study_features.py",
    "tools/pi05_libero_action_study_feature_source.py",
    "tools/vla_benchmark_contract.py", "tools/vla_visual_noise_wrapper.py",
}


def validate(repo, plan_sha):
    plan = read_json(base.checked(repo, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_action_study_low_contrast_plan_v1"
                 and plan["allowed"] == ALLOW and plan["corruption"] == CORRUPTION,
                 "fixed evaluation authorization/condition required")
    base.require(plan["seeds"] == list(base.SEEDS) and plan["arms"] == list(base.ARMS)
                 and plan["conditions"] == list(CONDITIONS)
                 and plan["validation_episode_counts"] == {str(k): v for k, v in VALIDATION_COUNTS.items()}
                 and plan["validation_windows"] == 500 and plan["feature_rows"] == 524
                 and plan["feature_output"] == FEATURE_OUT and plan["evaluation_output"] == EVAL_OUT,
                 "fixed complete evaluation scope changed")
    base.require(set(plan["input_sha256"]) == NEW_CODE | {TRAIN_REPORT}, "complete explicit new input inventory required")
    base.require(plan["input_sha256"][TRAIN_REPORT] == TRAIN_SHA, "completed final200 reference required")
    for path, digest in plan["input_sha256"].items():
        base.checked(repo, path, digest)
    report = read_json(repo / TRAIN_REPORT)
    base.require(report["status"] == "completed_fixed_budget_diagnostic"
                 and report["total_optimizer_steps"] == 1200
                 and report["input_files_unchanged"] is True, "complete fixed training reference required")
    pins = {**report["input_sha256"], **plan["input_sha256"], PLAN: plan_sha}
    pins.update({previous.TRAIN_ROOT + "/" + path: digest for path, digest in report["output_sha256"].items()})
    status_path = repo / previous.TRAIN_ROOT / "status.json"
    status = read_json(status_path)
    base.require(status == {"status": "completed", "report_sha256": TRAIN_SHA, "optimizer_steps": 1200},
                 "training status drifted")
    base.require(not (repo / previous.TRAIN_ROOT / "failure.json").exists(), "training failure artifact present")
    for path, digest in pins.items():
        base.checked(repo, path, digest)
    return {"status": "passed_frozen_checkpoint_and_input_preflight", "plan_sha256": plan_sha,
            "input_sha256": pins, "plan": plan, "new_optimizer_steps": 0}


def rehash(repo, pins):
    for path, digest in pins.items():
        base.checked(repo, path, digest)


def claim(repo, relative, preflight):
    out = repo / relative
    out.mkdir(exist_ok=False)
    base.write_json(out / "preflight.json", preflight)
    base.write_json(out / "started.json", {"pid": os.getpid(), "unix_seconds": time.time(), "optimizer_steps": 0})
    return out


def feature_stage(repo, plan_sha, execute):
    base.require(execute is True and not (repo / FEATURE_OUT).exists(), "explicit new feature run required; no retry")
    preflight = validate(repo, plan_sha)
    # Load the old, immutable extractor contract, not a modified train/val pack.
    import build_pi05_libero_action_study_features as old
    from pi05_libero_action_study_low_contrast_features import extract_validation_features
    from probe_pi05_libero_public_features import Tee, load_policy
    old_plan, source, runtime, source_preflight = old.preflight(repo)
    try:
        out = claim(repo, FEATURE_OUT, {**preflight, "source_and_extractor_preflight": source_preflight})
        started = time.monotonic()
        with (out / "run.log").open("x", encoding="utf-8") as log, \
             redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                policy, checkpoint = load_policy(Path(runtime["checkpoint"]), Path(runtime["protocol"]), "cuda")
                report = extract_validation_features(source, policy, checkpoint, old_plan, runtime,
                    out, repo / base.PACK_PATH,
                    progress=lambda count, eid, frame, condition: print(
                        f"PAIRED_FEATURE_PROGRESS rows={count + (524 if condition == 'low_contrast' else 0)}/1048 condition={condition} episode={eid} frame={frame}", flush=True))
                rehash(repo, preflight["input_sha256"])
                report.update(plan_sha256=plan_sha, input_sha256=preflight["input_sha256"],
                              runtime_seconds_excluding_preflight=time.monotonic()-started)
                report["output_sha256"] = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                    if p.is_file() and p.relative_to(out).as_posix() not in {"report.json", "status.json", "run.log"}}
                base.write_json(out / "report.json", report)
                base.write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json"),
                    "optimizer_steps": 0})
                print(base.canonical({"status": "completed_paired_validation_features", "report_sha256": sha256_file(out / "report.json"),
                    "wall_seconds": time.monotonic()-started, "optimizer_steps": 0}), flush=True)
                return report
            except BaseException as error:
                base.write_json(out / "failure.json", {"status": "failed_preserve_no_retry", "error": str(error),
                    "traceback": traceback.format_exc(), "optimizer_steps": 0})
                raise
    finally:
        old.close_source(source)


def checked_features(repo, feature_sha, plan_sha):
    base.require(feature_sha is not None, "external completed feature report SHA required")
    folder = repo / FEATURE_OUT
    report = read_json(base.checked(repo, FEATURE_OUT + "/report.json", feature_sha))
    base.require(report["schema"] == "pi05_libero_action_study_low_contrast_features_v1"
                 and report["status"] == "passed_validation_only_frozen_clean_replay_and_low_contrast"
                 and report["rows_per_condition"] == 524 and report["total_encoded_rows"] == 1048
                 and report["total_real_view_embeddings"] == 2096 and report["training_rows_encoded"] == 0
                 and report["clean_replay_matches_cached_latent_bytes"] is True
                 and report["target_tensors_modified"] is False and report["normalization_fitted"] is False
                 and report["optimizer_steps"] == 0
                 and {"row_indices.npy", "degraded_visual_latent.npy"}.issubset(report["output_sha256"]),
                 "complete hash-bound clean-replay/degraded validation arrays required")
    base.require(report["plan_sha256"] == plan_sha and not (folder / "failure.json").exists(),
                 "feature provenance or completion differs")
    base.require(read_json(folder / "status.json") == {"status": "completed", "report_sha256": feature_sha, "optimizer_steps": 0},
                 "feature status differs")
    pins = {FEATURE_OUT + "/report.json": feature_sha,
            **{FEATURE_OUT + "/" + path: digest for path, digest in report["output_sha256"].items()}}
    rehash(repo, pins)
    return report, pins


def evaluate_stage(repo, plan_sha, feature_sha, execute):
    base.require(execute is True and not (repo / EVAL_OUT).exists(), "explicit fresh evaluation required; no retry")
    preflight = validate(repo, plan_sha)
    feature_report, feature_pins = checked_features(repo, feature_sha, plan_sha)
    previous.runtime_settings()  # CPU-only separate process, frozen torch2.10.
    train = read_json(repo / TRAIN_REPORT)
    step0 = read_json(repo / base.OUT_PATH / "report.json")
    out = claim(repo, EVAL_OUT, {**preflight, "feature_report_sha256": feature_sha})
    started = time.monotonic()
    try:
        final_trace = [json.loads(line) for line in (repo / previous.TRAIN_ROOT / "evaluation_final.jsonl").read_text().splitlines()]
        initial_trace = [json.loads(line) for line in (repo / base.OUT_PATH / "evaluation_trace.jsonl").read_text().splitlines()]
        rows = np.load(repo / FEATURE_OUT / "row_indices.npy", allow_pickle=False)
        latents = np.load(repo / FEATURE_OUT / "degraded_visual_latent.npy", allow_pickle=False)
        models, runs, persistence, checkpoint_checks = {}, {}, {}, {}
        with LiberoFeaturePack(repo / base.PACK_PATH) as pack:
            split = pack.load_split(repo / base.PACK_PATH / "split.json")
            stats = read_json(repo / base.PACK_PATH / "preparation/normalization.json")
            dataset = LiberoWindowDataset(pack, split, partition="validation", normalization=stats)
            base.require(len(dataset) == 500 and {eid: len(pack.indices[eid]) for eid in VALIDATION_COUNTS} == VALIDATION_COUNTS,
                         "fixed whole-episode validation coverage differs")
            lookup = validate_feature_rows(pack, rows, latents)
            for seed in base.SEEDS:
                models[seed] = base.initialized_pair(seed, pack.manifest["task_registry"])
                runs[str(seed)] = {arm: {} for arm in base.ARMS}
                checkpoint_checks[str(seed)] = {}
                for arm, model in models[seed].items():
                    record = train["runs"][str(seed)][arm]["checkpoint"]
                    envelope = record["envelope"]
                    binding = deepcopy(envelope["binding"])
                    base.require(binding["seed"] == seed and binding["arm"] == arm
                                 and binding["training_authorization_sha256"] == train["training_authorization_sha256"],
                                 "saved checkpoint seed/arm/auth differs")
                    metadata = deepcopy(envelope["legacy_metadata"])
                    metadata["authorization_sha256"] = train["training_authorization_sha256"]
                    checkpoint_io.load_final_checkpoint(repo / previous.TRAIN_ROOT / record["path"], model,
                        envelope, binding, expected_metadata=metadata)
                    checkpoint_checks[str(seed)][arm] = {"checkpoint_sha256": envelope["checkpoint_sha256"],
                        "parameter_sha256_before": base.parameter_hash(model), "step": 200}
            with (out / "evaluation_trace.jsonl").open("x", encoding="utf-8") as trace:
                # All six clean models must pass the saved final200 reference
                # before any degraded-condition prediction is scored.
                for condition in CONDITIONS:
                    refs = {tuple(row["window_indices"]): row for row in initial_trace
                            if row["seed"] == base.SEEDS[0] and row["partition"] == "validation"}
                    base.require(len(refs) == 33, "complete persistence reference required")
                    persistence[condition] = evaluate_condition(None, dataset, stats, arm="persistence", seed=None,
                        condition=condition, row_lookup=lookup, latents=latents, references=refs, trace=trace)
                    if condition == "clean":
                        previous.compare_initial(persistence[condition], step0["runs"][str(base.SEEDS[0])]["validation"]["arms"]["persistence"])
                    for seed, pair in models.items():
                        for arm, model in pair.items():
                            refs = {tuple(row["window_indices"]): row for row in final_trace
                                if row["seed"] == seed and row["arm"] == arm and row["partition"] == "validation"}
                            base.require(len(refs) == 33, "complete final200 reference required")
                            print(f"FROZEN_EVAL seed={seed} arm={arm} condition={condition}; optimizer_steps=0", flush=True)
                            result = evaluate_condition(model, dataset, stats, arm=arm, seed=seed, condition=condition,
                                row_lookup=lookup, latents=latents, references=refs, trace=trace)
                            if condition == "clean":
                                previous.compare_initial(result, train["runs"][str(seed)][arm]["final"]["validation"])
                            runs[str(seed)][arm][condition] = result
            for seed, pair in models.items():
                for arm, model in pair.items():
                    check = checkpoint_checks[str(seed)][arm]
                    check["parameter_sha256_after"] = base.parameter_hash(model)
                    base.require(check["parameter_sha256_after"] == check["parameter_sha256_before"], "checkpoint parameters changed")
        pins = {**preflight["input_sha256"], **feature_pins}
        rehash(repo, pins)
        base.require(not torch.cuda.is_initialized(), "CPU evaluator initialized CUDA")
        model_batches = sum(row["micro"]["update_count"] for arms in runs.values()
                            for conditions in arms.values() for row in conditions.values())
        persistence_batches = sum(row["micro"]["update_count"] for row in persistence.values())
        base.require(model_batches == 396 and persistence_batches == 66, "complete scoring batch counts required")
        report = {"schema": "libero_action_study_low_contrast_result_v1", "status": "completed_frozen_paired_evaluation",
            "plan_sha256": plan_sha, "feature_report_sha256": feature_sha, "training_report_sha256": TRAIN_SHA,
            "input_sha256": pins, "input_files_unchanged": True, "runs": runs, "persistence": persistence,
            "checkpoint_checks": checkpoint_checks, **summarize(runs, persistence),
            "wall_seconds": time.monotonic()-started, "runtime": previous.runtime_evidence(),
            "optimizer_steps": 0, "training_started": False, "normalization_refitted": False,
            "clean_reference_prediction_and_statistics_replayed": True, "future_targets_remain_clean": True,
            "validation_episode_count": 4, "validation_windows": 500,
            "model_prediction_batches": model_batches, "scored_persistence_prediction_batches": persistence_batches,
            "persistence_helper_calls_including_support_checks": model_batches + 2*persistence_batches,
            "target_scope": "same clean future visual latents and normalized8D state residuals",
            "limitations": preflight["plan"]["limitations"], "policy_performance_evaluated": False,
            "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
            "training_ready": False, "formal_data_allowed": False, "real_system_validated": False,
            "output_sha256": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}}
        base.write_json(out / "report.json", report)
        base.write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json"), "optimizer_steps": 0})
        print(base.canonical({"status": report["status"], "decision": report["decision"], "wall_seconds": report["wall_seconds"],
            "report_sha256": sha256_file(out / "report.json"), "optimizer_steps": 0}), flush=True)
        return report
    except BaseException as error:
        base.write_json(out / "failure.json", {"status": "failed_preserve_no_retry", "error": str(error),
            "traceback": traceback.format_exc(), "optimizer_steps": 0})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "features", "evaluate"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--feature-report-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    base.require(args.execute == (args.stage != "preflight"), "explicit execution flag required only for actual stages")
    base.require((args.feature_report_sha256 is not None) == (args.stage == "evaluate"), "feature SHA only for evaluation")
    if args.stage == "preflight":
        result = validate(ROOT, args.plan_sha256)
        print(base.canonical({"status": result["status"], "hashes_checked": len(result["input_sha256"]), "optimizer_steps": 0}))
    elif args.stage == "features":
        feature_stage(ROOT, args.plan_sha256, args.execute)
    else:
        evaluate_stage(ROOT, args.plan_sha256, args.feature_report_sha256, args.execute)


if __name__ == "__main__":
    main()
