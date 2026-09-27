"""One explicitly authorized, fixed 200-update native LIBERO CPU diagnostic.

Not a policy trainer or a quality benchmark. No PI0.5 load, new data, model/loss
change, parameter search, partial resume, rollout or action selection.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from pi05_libero_world_model_adapter import (
    LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization, read_json, sha256_file,
)
from smoke_pi05_libero_world_model import (
    BATCH_SIZE, COUNTS, SEED, SPLIT, SOURCE_SHA, Tee, contained, input_fingerprints,
    parameter_sha, prediction_check, tensor_sha, validate_artifacts, write_json,
)
from train_pi05_libero_world_model import ObjectiveTotals
from eval_pi05_libero_world_model_persistence import (
    PROTOCOL_SHA, assert_matching_batch, change, compare_stats, load_protocol,
    sampler_plan_hash, validate_reference,
)

SCHEMA = "pi05_libero_learning_diagnostic_v1"
AUTHORIZATION_SHA = "8bfe78322c44f7e4959287a9f53a920b0591c3df975b85c3b8b02d4f4f209511"
OWN_FILES = ("run_pi05_libero_world_model_learning_diagnostic.py", "pi05_libero_world_model_training.py")


def load_authorization(path: Path, expected_sha: str, *, execute: bool) -> dict:
    if execute is not True:
        raise ValueError("explicit --execute-authorized-diagnostic required")
    if expected_sha != AUTHORIZATION_SHA or sha256_file(path) != AUTHORIZATION_SHA:
        raise ValueError("frozen diagnostic authorization SHA mismatch")
    record = read_json(path)
    if record["design_protocol_sha256"] != PROTOCOL_SHA or record["execution_authorized"] is not True:
        raise ValueError("authorization must bind the unchanged design protocol")
    return record


def validate_persistence(root: Path, authorization: dict, verified: dict, reference_hashes: dict) -> tuple[dict, dict]:
    digest = authorization["persistence_report_sha256"]
    if sha256_file(contained(root, "report.json")) != digest:
        raise ValueError("persistence reference SHA mismatch")
    report = read_json(root / "report.json")
    expected = {"schema": "pi05_libero_persistence_evaluation_v1", "status": "passed",
                "protocol_sha256": PROTOCOL_SHA, "split": SPLIT, "seed": SEED,
                "batch_size": BATCH_SIZE, "device": "cpu", "optimizer_steps": 0,
                "verified_feature_artifacts": verified, "verified_reference_artifacts": reference_hashes}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report.get(key) != value:
            raise ValueError(f"persistence contract mismatch: {key}")
    outputs = {"report.json": digest, **report["output_sha256"]}
    for name, value in outputs.items():
        if sha256_file(contained(root, name)) != value:
            raise ValueError(f"persistence artifact mismatch: {name}")
    for name, value in report["implementation_sha256"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != value:
            raise ValueError(f"persistence implementation mismatch: {name}")
    return report, outputs


def numeric_tree_close(actual, expected, *, path="root") -> None:
    if isinstance(expected, dict):
        if type(actual) is not dict or set(actual) != set(expected):
            raise ValueError(f"metric keys differ at {path}")
        for key in expected:
            numeric_tree_close(actual[key], expected[key], path=path + "." + key)
    elif isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"metric axis differs at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            numeric_tree_close(a, b, path=f"{path}[{i}]")
    elif type(expected) is float:
        if type(actual) not in (float, int) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"initial metric value differs at {path}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"metric metadata/count differs at {path}")


def ordinary_tensors(batch: dict) -> None:
    import torch
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and (torch.is_inference(value) or value.requires_grad):
            raise ValueError(f"ordinary detached input/target tensors required: {key}")


def runtime_gate(protocol: dict) -> None:
    import torch
    spec = protocol["future_learning_diagnostic"]
    if str(torch.__version__) != spec["torch_version"] or torch.is_inference_mode_enabled():
        raise ValueError("frozen Torch runtime and ordinary outer context required")
    torch.set_num_threads(spec["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_initialized():
        raise ValueError("CPU diagnostic must not initialize CUDA")


def evaluate(model, batches: dict, stats: dict, out: Path, step: int, *, reference_traces=None) -> dict:
    """Use the frozen reference forward path; restore modes and gradient flags."""
    import torch
    from pi05_libero_world_model_objectives import NativeWorldModelMetrics, native_world_model_loss
    modes = {module: module.training for module in model.modules()}
    gradient_flags = {parameter: parameter.requires_grad for parameter in model.parameters()}
    before = parameter_sha(model)
    versions = [p._version for p in model.parameters()]
    parts = {}
    try:
        # inference_mode alone does not reproduce the historical frozen-parameter
        # CPU path bitwise. Freeze only during evaluation, never the train cache.
        model.eval().requires_grad_(False)
        with (out / f"evaluation_step{step:03d}_trace.jsonl").open("x", encoding="utf-8") as stream:
            with torch.inference_mode():
                for part, rows in batches.items():
                    totals, metrics = ObjectiveTotals(), NativeWorldModelMetrics(stats["state_std"])
                    for start, inputs, targets in rows:
                        predictions = model(**inputs)
                        prediction_check(predictions, len(inputs["task_instruction"]), 1)
                        hashes = {k: tensor_sha(v) for k, v in predictions.items()}
                        if reference_traces is not None and hashes != reference_traces[(part, start)]["predictions"]:
                            raise ValueError("step0 prediction SHA differs; strict preregistered gate, no optimizer started")
                        loss, details = native_world_model_loss(predictions, targets)
                        totals.update(details)
                        metrics.update(predictions, targets)
                        stream.write(json.dumps({"step": step, "partition": part, "window_start": start,
                            "batch_size": len(inputs["task_instruction"]), "predictions": hashes,
                            "inputs": input_fingerprints(inputs), "targets": {k: tensor_sha(v) for k, v in targets.items()},
                            "loss": float(loss), "objective_sufficient_statistics": {
                                k: float(v) if k.endswith("_sum") else v for k, v in details.items()
                                if k.endswith("_sum") or k.endswith("_count")}}, allow_nan=False) + "\n")
                        if step == 200 and start == 0:
                            with (out / f"final_{part}_audit.npz").open("xb") as archive:
                                np.savez_compressed(archive, **{k: v.detach().cpu().numpy() for k, v in {**predictions, **targets}.items()})
                    summary = metrics.summary()
                    if (summary["window_count"] != COUNTS[part]
                            or summary["visual"]["aggregate"]["count"] != totals.counts["visual"]
                            or summary["state_normalized"]["aggregate"]["count"] != totals.counts["state"]):
                        raise ValueError("full evaluation coverage/count mismatch")
                    parts[part] = {"objective": totals.summary(), "metrics": summary}
    finally:
        for parameter, flag in gradient_flags.items():
            parameter.requires_grad_(flag)
        for module, mode in modes.items():
            module.training = mode
    if before != parameter_sha(model) or versions != [p._version for p in model.parameters()]:
        raise ValueError("evaluation modified parameters")
    write_json(out / f"evaluation_step{step:03d}.json", parts)
    return parts


def report_comparison(final: dict, baseline: dict, metrics_key: str) -> dict:
    return {part: {"objective": change(final[part]["objective"]["objective"], baseline[part]["objective"]["objective"]),
                  **{key: compare_stats(final[part]["metrics"][key], baseline[part][metrics_key][key])
                     for key in ("visual", "state_normalized", "state_native")}} for part in COUNTS}


def run(args, progress: dict) -> dict:
    authorization = load_authorization(args.authorization, args.authorization_sha256, execute=args.execute_authorized_diagnostic)
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    if args.out.name != authorization["output_directory_name"]:
        raise ValueError("only the named one-attempt diagnostic output is authorized")
    _, verified = validate_artifacts(args.feature_pack, protocol["data"]["feature_report_sha256"])
    for key, name in (("manifest_sha256", "manifest.json"), ("split_sha256", "split.json"),
                      ("normalization_sha256", "preparation/normalization.json")):
        if verified[name] != protocol["data"][key]:
            raise ValueError("protocol feature/split/normalization linkage differs")
    random, traces, random_hashes = validate_reference(args.random_reference, protocol, verified)
    persistence, persistence_hashes = validate_persistence(args.persistence_reference, authorization, verified, random_hashes)
    implementations = {**persistence["implementation_sha256"],
                       **{name: sha256_file(Path(__file__).with_name(name)) for name in OWN_FILES}}
    print("AUTHORIZATION_PROTOCOL_FEATURE_BASELINES_VERIFIED", flush=True)

    import torch
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
    from pi05_libero_world_model_objectives import collate_window_targets
    from pi05_libero_world_model_persistence import persistence_scoring_targets
    from pi05_libero_world_model_training import build_optimizer, train_step, save_final_checkpoint, load_final_checkpoint
    runtime_gate(protocol)
    plan = protocol["future_learning_diagnostic"]
    generator = torch.Generator(device="cpu").manual_seed(plan["sampling_seed"])
    draws = torch.randint(COUNTS["train"], (200, BATCH_SIZE), generator=generator).tolist()
    draw_sha = hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode("ascii")).hexdigest()
    if draw_sha != sampler_plan_hash(protocol):
        raise ValueError("actual draw plan differs")
    write_json(args.out / "sampling_plan.json", {"schema": "native_fixed_200x16_draws_v1", "indices": draws, "indices_sha256": draw_sha})
    timings = {}
    with LiberoFeaturePack(args.feature_pack) as pack:
        split = pack.load_split(args.feature_pack / "split.json")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(args.feature_pack / "preparation" / "normalization.json"):
            raise ValueError("train-only normalization differs")
        datasets = {part: LiberoWindowDataset(pack, split, partition=part, normalization=stats) for part in COUNTS}
        if {part: len(d) for part, d in datasets.items()} != COUNTS:
            raise ValueError("fixed native window counts differ")
        items = {part: [dataset[i] for i in range(len(dataset))] for part, dataset in datasets.items()}
        batches = {}
        for part in COUNTS:
            batches[part] = []
            for start in range(0, COUNTS[part], BATCH_SIZE):
                rows = items[part][start:start+BATCH_SIZE]
                inputs = collate_window_inputs([i["inputs"] for i in rows], device="cpu")
                targets = collate_window_targets([i["targets"] for i in rows], device="cpu")
                ordinary_tensors(inputs)
                ordinary_tensors(targets)
                common = persistence_scoring_targets(inputs, targets)
                assert_matching_batch(inputs, targets, common, rows, traces[(part, start)])
                batches[part].append((start, inputs, targets))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = LiberoWorldModel(LiberoWorldModelConfig(**protocol["model"]["config"]), pack.manifest["task_registry"])
        initial_sha = parameter_sha(model)
        if initial_sha != protocol["baselines"]["random_reference_initial_parameter_sha256"]:
            raise ValueError("initial parameter SHA differs; training forbidden")
        start = time.monotonic()
        initial = evaluate(model, batches, stats, args.out, 0, reference_traces=traces)
        for part in COUNTS:
            numeric_tree_close(initial[part]["metrics"], random["partitions"][part]["diagnostic_untrained_metrics"])
            numeric_tree_close(initial[part]["objective"], random["partitions"][part]["objective"])
        timings["initial_evaluation_seconds"] = time.monotonic() - start
        print("STEP0_EXACT_PREDICTIONS_AND_METRICS_VERIFIED", flush=True)
        optimizer = build_optimizer(model, plan["optimizer"])
        model.train().requires_grad_(True)
        start = time.monotonic()
        progress["training_started"] = True
        with (args.out / "training_trace.jsonl").open("x", encoding="utf-8") as stream:
            for step, indices in enumerate(draws, 1):
                progress["last_attempted_step"] = step
                rows = [items["train"][i] for i in indices]
                inputs = collate_window_inputs([i["inputs"] for i in rows], device="cpu")
                targets = collate_window_targets([i["targets"] for i in rows], device="cpu")
                ordinary_tensors(inputs)
                ordinary_tensors(targets)
                before_i, before_t = input_fingerprints(inputs), {k: tensor_sha(v) for k, v in targets.items()}
                record = train_step(model, optimizer, inputs, targets, clip_norm=1.0, expected_step=step)
                if (input_fingerprints(inputs) != before_i or {k: tensor_sha(v) for k, v in targets.items()} != before_t
                        or any(v.grad is not None for v in inputs.values() if isinstance(v, torch.Tensor))):
                    raise ValueError("training modified/propagated gradients into detached features or targets")
                progress["completed_updates"] = step
                record = {**record, "completed_updates": step, "sampled_train_window_indices": indices,
                          "inputs": before_i, "targets": before_t}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if step == 1 or step % 25 == 0:
                    elapsed = time.monotonic() - start
                    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "pid": os.getpid(),
                               **progress, "training_elapsed_seconds": elapsed})
                    print(f"NATIVE_TRAIN updates={step}/200 loss={record['loss']:.8f} elapsed={elapsed:.2f}s", flush=True)
        timings["training_seconds"] = time.monotonic() - start
        optimizer.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
        final_sha = parameter_sha(model)
        if final_sha == initial_sha or progress["completed_updates"] != 200:
            raise ValueError("parameters unchanged or not exactly200 updates")
        metadata = {"protocol_sha256": PROTOCOL_SHA, "authorization_sha256": AUTHORIZATION_SHA,
            "source_report_sha256": SOURCE_SHA, "feature_report_sha256": protocol["data"]["feature_report_sha256"],
            "manifest_sha256": protocol["data"]["manifest_sha256"], "split_sha256": protocol["data"]["split_sha256"],
            "normalization_sha256": protocol["data"]["normalization_sha256"], "sampling_plan_sha256": draw_sha,
            "model_config": protocol["model"]["config"], "task_registry": pack.manifest["task_registry"],
            "step": 200, "checkpoint_kind": "final_diagnostic_no_resume",
            "initial_parameter_sha256": initial_sha, "final_parameter_sha256": final_sha}
        start = time.monotonic()
        checkpoint = args.out / "final_step200.pt"
        checkpoint_sha = save_final_checkpoint(checkpoint, model, optimizer, metadata)
        with torch.random.fork_rng(devices=[]):
            reloaded = LiberoWorldModel(LiberoWorldModelConfig(**protocol["model"]["config"]), pack.manifest["task_registry"])
        load_final_checkpoint(checkpoint, reloaded, metadata, checkpoint_sha)
        reloaded.eval().requires_grad_(False)
        if parameter_sha(reloaded) != final_sha:
            raise ValueError("reloaded state-dict SHA differs")
        with torch.inference_mode():
            for part in COUNTS:
                inputs = batches[part][0][1]
                before, after = model(**inputs), reloaded(**inputs)
                for key in before:
                    torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        timings["checkpoint_save_load_verification_seconds"] = time.monotonic() - start
        start = time.monotonic()
        final = evaluate(reloaded, batches, stats, args.out, 200)
        timings["final_evaluation_seconds"] = time.monotonic() - start
        # Recheck cached evaluation inputs/targets after all forward and update work.
        for part, part_batches in batches.items():
            for start_index, inputs, targets in part_batches:
                rows = items[part][start_index:start_index+BATCH_SIZE]
                assert_matching_batch(inputs, targets, targets, rows, traces[(part, start_index)])
        parameter_count = sum(p.numel() for p in reloaded.parameters())
    for root, hashes in ((args.feature_pack, verified), (args.random_reference, random_hashes),
                         (args.persistence_reference, persistence_hashes)):
        for name, digest in hashes.items():
            if sha256_file(contained(root, name)) != digest:
                raise ValueError(f"source/reference changed during training: {name}")
    for name, digest in implementations.items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"implementation changed during training: {name}")
    load_protocol(args.protocol, args.protocol_sha256)
    load_authorization(args.authorization, args.authorization_sha256, execute=args.execute_authorized_diagnostic)
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA initialization")
    return {"schema": SCHEMA, "status": "passed", "scope": "one_task_two_episode_learning_diagnostic_only",
        "protocol_sha256": PROTOCOL_SHA, "authorization_sha256": AUTHORIZATION_SHA,
        "authorization": authorization, "protocol_design_unchanged": True,
        "implementation_sha256": implementations, "verified_feature_artifacts": verified,
        "verified_random_reference_artifacts": random_hashes, "verified_persistence_artifacts": persistence_hashes,
        "split": SPLIT, "windows": COUNTS, "seed": SEED, "batch_size": BATCH_SIZE,
        "device": "cpu", "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "sampling_plan_sha256": draw_sha,
        "initial_parameter_sha256": initial_sha, "final_parameter_sha256": final_sha,
        "parameter_count": parameter_count, "parameters_changed": True,
        "initial_reference_prediction_hashes_exact": True, "initial_metric_rtol": 1e-12, "initial_metric_atol": 1e-10,
        "training_started": True, "optimizer_steps": 200, "backward_calls": 200,
        "all_update_gradient_and_optimizer_guards_passed": True, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metadata": metadata, "checkpoint_reloaded_weights_only": True,
        "checkpoint_reload_parameter_hash_exact": True, "checkpoint_reload_first_two_batches_prediction_max_difference": 0.0,
        "final_evaluation_uses_reloaded_checkpoint": True, "all_bound_inputs_unchanged": True,
        "timings": timings, "evaluations": {"step0": initial, "step200": final},
        "comparison_to_random_untrained": report_comparison(final, random["partitions"], "diagnostic_untrained_metrics"),
        "comparison_to_persistence": report_comparison(final, persistence["partitions"], "persistence_metrics"),
        "model_architecture_or_loss_changed": False, "extractor_loaded": False, "pi05_loaded": False,
        "feature_input_gradients": False, "policy_actions_generated": 0, "simulation_steps": 0,
        "training_ready": False, "formal_data_allowed": False, "family_independence_verified": False,
        "checkpoint_training_overlap_unknown": True, "generalization_evaluated": False, "real_system_validated": False,
        "checkpoint_selection": "fixed final200 only; no validation selection or tuning",
        "visual_review_required": False, "visual_scope": "numerical future latent/state prediction only; no pixel decoder or policy rollout",
        "output_sha256": {p.name: sha256_file(p) for p in args.out.iterdir()
                          if p.is_file() and p.name not in {"report.json", "status.json", "run.log"}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-pack", "random-reference", "persistence-reference", "protocol", "authorization", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--execute-authorized-diagnostic", action="store_true", required=True)
    args = parser.parse_args(argv)
    authorization = load_authorization(args.authorization, args.authorization_sha256, execute=args.execute_authorized_diagnostic)
    load_protocol(args.protocol, args.protocol_sha256)
    if args.out.name != authorization["output_directory_name"]:
        raise ValueError("fixed one-attempt output name required")
    args.out.mkdir(parents=True, exist_ok=False)
    progress = {"training_started": False, "completed_updates": 0, "last_attempted_step": 0}
    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "pid": os.getpid(), **progress})
    started = time.monotonic()
    with (args.out / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                report = run(args, progress)
                report["runtime_seconds"] = time.monotonic() - started
                write_json(args.out / "report.json", report)
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "completed", **progress,
                    "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 200})
                print("NATIVE_LEARNING_DIAGNOSTIC_PASSED", report["runtime_seconds"], flush=True)
            except BaseException as error:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "failed", **progress,
                    "error_type": type(error).__name__, "error": str(error),
                    "last_attempt_update_outcome_uncertain": progress["last_attempted_step"] > progress["completed_updates"],
                    "automatic_retry_allowed": False})
                raise


if __name__ == "__main__":
    main()
