"""Frozen twenty-source candidate test, not training or a selector closed loop.

prepare binds existing checkpoints and checks saved validation inputs only.
run alone creates native test trajectories; summarize only reads saved output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import collect_pusht_action_effect_pairs as collection
    from . import train_pusht_action_effect_contrast as training
else:
    import collect_pusht_action_effect_pairs as collection
    import train_pusht_action_effect_contrast as training

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_output/pusht_action_effect_fresh_test_v1"
SUPPLEMENT = ROOT / "docs/pusht-action-effect-test-protocol-v1.json"
SCHEMA = "pusht_action_effect_fresh_test_v1"
read, rows, write = collection.read, collection.rows, collection.write
common, baseline, inference = collection.common, collection.baseline, collection.inference
replay, vision = collection.replay, collection.vision
ARMS = training.pair.ARMS


def settings():
    original, test = read(training.PROTOCOL), read(SUPPLEMENT)
    if (test["schema"] != SCHEMA or test["test_source_seeds"] != list(range(430000, 430020))
            or original["future_source_seed_ranges"]["test"] != [430000, 430019]
            or test["training_seeds"] != original["paired_training_seeds"]
            or any(test[k] != original[k] for k in ("anchors", "nominal_cap", "horizon",
                                                   "coverage_tie_epsilon", "prediction_tie_epsilon"))
            or test["maximum_environment_steps_single_pass"] != 20 * collection.MAX_STEPS_PER_SOURCE):
        raise ValueError("only the locked, reserved twenty-source test is supported")
    return original, test


def model_key(seed, arm):
    return f"{seed}__{arm}"


def checkpoint_identities(test):
    identities = {}
    for seed in test["training_seeds"]:
        for arm in ARMS:
            path = training.TRAIN_ROOT / str(seed) / arm / "final.pt"
            stat = path.stat()
            identities[model_key(seed, arm)] = {"path": str(path.resolve()), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "training_seed": seed, "arm": arm, "step": 2000}
    return identities


def source_inventory(original, out):
    # Earlier train/validation source IDs are now used, so check only the test reservation.
    only_test = {**original, "future_source_seed_ranges": {"test": original["future_source_seed_ranges"]["test"]}}
    inventory = collection.prep.reserved_seed_inventory(only_test, out)
    reset_ids = collection.reset_ids_from_metadata(inventory)
    paired_sources = []
    for split in ("train", "validation"):
        low, high = original["future_source_seed_ranges"][split]
        for seed in range(low, high + 1):
            done = read(training.DATA_ROOT / split / str(seed) / "done.json")
            if done["status"] != "completed_source" or done["seed"] != seed:
                raise ValueError("completed train/validation reset provenance missing")
            paired_sources.append(done["reset_observation_sha256"])
    if len(set(paired_sources)) != 40:
        raise ValueError("expected forty distinct prior paired-data reset observations")
    reset_ids.update(paired_sources)
    return {**inventory, "paired_data_resets_added": len(paired_sources),
            "prior_reset_observation_ids": sorted(reset_ids)}


def configure_torch():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_models(identities, contract):
    models, goal = {}, None
    for key, identity in identities.items():
        model, payload = training.load_final(Path(identity["path"]))
        if (payload["training_seed"] != identity["training_seed"] or payload["arm"] != identity["arm"]
                or payload["contract"] != contract or sum(p.numel() for p in model.parameters()) != training.PARAMETERS):
            raise ValueError("checkpoint arm/seed/contract/architecture mismatch")
        if goal is not None and not torch.equal(model.goal, goal):
            raise ValueError("all six models must retain the same training-only goal")
        goal = model.goal
        models[key] = model
    return models


@torch.inference_mode()
def score_inputs(models, inputs):
    scores, seconds = {}, {}
    for key, model in models.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores[key] = model.predict_effect(*inputs).cpu().numpy().astype(np.float64)
        torch.cuda.synchronize()
        seconds[key] = time.perf_counter() - start
        if not np.isfinite(scores[key]).all():
            raise ValueError("nonfinite frozen scorer prediction")
    return scores, seconds


def cluster_interval(values, bootstrap, confidence):
    """Rows are paired training repeats; columns are independent source clusters."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("finite [training repeat, source] effects required")
    per_repeat, per_source = values.mean(1), values.mean(0)
    bounds = None
    if len(per_source) >= 2:
        indices = np.random.default_rng(bootstrap["seed"]).integers(
            len(per_source), size=(bootstrap["replicates"], len(per_source)))
        distribution = per_source[indices].mean(1)
        alpha = 1 - confidence
        bounds = np.quantile(distribution, [alpha / 2, 1 - alpha / 2]).tolist()
    return {"mean": float(per_source.mean()), "confidence_level": confidence, "percentile_interval": bounds,
            "source_clusters": len(per_source), "paired_training_repeats": len(per_repeat),
            "per_training_repeat_mean": per_repeat.tolist(),
            "training_repeat_sample_std": float(per_repeat.std(ddof=1)) if len(per_repeat) > 1 else None,
            "per_source_training_mean": per_source.tolist(),
            "conditional_on_fixed_training_repeats": True}


def prepare(out):
    if out.exists():
        raise FileExistsError("preserve the prepared protocol; use run/resume, not another preparation")
    original, test = settings()
    report = read(training.TRAIN_ROOT / "report.json")
    if (report["status"] != "completed_fixed_paired_training_and_validation"
            or report["total_optimizer_steps"] != 12000 or report["contract"]["protocol"] != original
            or report["contract"]["cache_identity"] != collection.cache_identity()):
        raise ValueError("requires the six completed matched final2000 checkpoints")
    used, identities = source_inventory(original, out), checkpoint_identities(test)
    configure_torch()
    models = load_models(identities, report["contract"])
    validation = training.pair.PairPack(training.DATA_ROOT / "validation/pack")
    scores, _ = score_inputs(models, validation.inputs(slice(None), device="cuda"))
    errors = {}
    for seed in test["training_seeds"]:
        with np.load(training.TRAIN_ROOT / str(seed) / "validation_predictions.npz", allow_pickle=False) as saved:
            expected_keys = np.array([(r["source_seed"], r["anchor_step"]) for r in validation.contexts])
            if not np.array_equal(saved["context_keys"], expected_keys):
                raise ValueError("saved validation row order changed")
            for arm in ARMS:
                key = model_key(seed, arm)
                errors[key] = float(np.abs(scores[key] - saved[arm + "_scores"]).max())
                if errors[key] > 2e-6:
                    raise ValueError("frozen checkpoint forward does not reproduce saved validation scores")
        # Optional fixed controls must leave all old metric slots unchanged.
        metrics = training.selection_metrics(validation.targets,
            {arm: scores[model_key(seed, arm)] for arm in ARMS}, expected_keys[:, 0], original, fixed_candidates=True)
        old = report["validation"][str(seed)]["metrics"]
        for name, method in old["methods"].items():
            if metrics["methods"][name] != method:
                raise ValueError("adding constant-index controls changed an existing metric")
    # A mechanical zero-effect check, not mock experiment results or test data.
    if cluster_interval(np.zeros((3, 2)), test["bootstrap"], .95)["percentile_interval"] != [0., 0.]:
        raise AssertionError("paired source bootstrap must preserve zero effects")
    if checkpoint_identities(test) != identities:
        raise ValueError("a checkpoint changed while preparing")
    spec = {"schema": SCHEMA, "training_protocol": original, "test_protocol": test,
            "checkpoint_identity": identities, "training_contract": report["contract"],
            "reference_runtime_binding": read(training.DATA_ROOT / "runtime_binding.json"),
            "reference_environment_parity": read(training.DATA_ROOT / "environment_parity.json"),
            "torch_version": torch.__version__, "created_utc": datetime.now(timezone.utc).isoformat()}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", spec)
    write(out / "seed_inventory.json", used)
    prepared = {"status": "prepared_frozen_checkpoints_existing_validation_forward_only",
                "frozen_models": len(models), "checked_validation_contexts": len(validation.contexts),
                "scorer_forward_calls": len(models), "validation_max_abs_error": errors,
                "old_metric_slots_unchanged": True, "zero_effect_cluster_check": True,
                "test_sources_executed": 0, "environment_steps": 0, "optimizer_steps": 0, "hardware_actions": 0,
                "maximum_single_pass_steps": test["maximum_environment_steps_single_pass"],
                "full_test_and_interrupt_recovery_verified": False}
    write(out / "preparation_report.json", prepared)
    write(out / "status.json", prepared)
    print(json.dumps(prepared, indent=2), flush=True)
    return 0


def contract(out):
    spec = read(out / "protocol.json")
    original, test = settings()
    if (spec["schema"] != SCHEMA or spec["test_protocol"] != test or spec["training_protocol"] != original
            or spec["checkpoint_identity"] != checkpoint_identities(test)
            or spec["training_contract"] != read(training.TRAIN_ROOT / "report.json")["contract"]
            or spec["training_contract"]["cache_identity"] != collection.cache_identity()
            or spec["torch_version"] != torch.__version__):
        raise ValueError("frozen checkpoint/protocol/runtime binding changed; no silent test restart")
    return spec


def completed(out, test):
    found = []
    for seed in test["test_source_seeds"]:
        path = out / "sources" / str(seed) / "done.json"
        if path.exists():
            entry = read(path)
            if entry["status"] != "completed_source" or entry["seed"] != seed:
                raise ValueError("source completion identity mismatch")
            found.append(entry)
    return found


def all_attempt_steps(out):
    return sum(read(path).get("environment_steps", 0) for path in (out / "sources").glob("*/attempt_*/report.json"))


def run(out, *, resume, stop_after_sources):
    spec = contract(out)
    test, original = spec["test_protocol"], spec["training_protocol"]
    done = completed(out, test)
    if len(done) == len(test["test_source_seeds"]):
        if (out / "report.json").exists():
            print("Test already completed; no repeated inference or environment execution.", flush=True)
            return 0
        return summarize(out)
    if (out / "sources").exists() and not resume:
        raise ValueError("prior source attempt exists; use --resume")
    used = source_inventory(original, out)
    reset_ids = set(used["prior_reset_observation_ids"])
    for entry in done:
        if entry["reset_observation_sha256"] in reset_ids:
            raise ValueError("completed test source duplicates a prior source reset")
        reset_ids.add(entry["reset_observation_sha256"])
    configure_torch()
    state = {"status": "running", "optimizer_steps": 0, "hardware_actions": 0}
    vector, invocation_steps, new_sources = None, 0, 0
    start, handlers = time.monotonic(), {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}: preserve partial attempt; resume at this source's start")

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupt)
    try:
        models = load_models(spec["checkpoint_identity"], spec["training_contract"])
        runtime = baseline.imports()
        sources, reference = baseline.source_binding(runtime)
        act, binding = inference.load_final("act")
        if {"ACT_binding": binding, "source_binding": sources} != spec["reference_runtime_binding"]:
            raise ValueError("ACT/native execution binding differs from the paired-data collection")
        prior = replay.FrozenACTPrior(act, binding)
        goal = next(iter(models.values())).goal.cpu().numpy()
        config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        env = vector.envs[0]
        common.validate_env_kwargs(config.gym_kwargs)
        parity = {"environment_gym_kwargs": config.gym_kwargs,
                  "environment_step_source_sha256": common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
                  "package_versions": runtime["_package_versions"]()}
        common.verify_environment_parity(reference, parity)
        if parity != spec["reference_environment_parity"]:
            raise ValueError("environment differs from the original paired-data execution")
        pre, post = runtime["make_env_pre_post_processors"](config, act.policy.config)
        if pre.steps or post.steps:
            raise ValueError("native environment processors must stay identity")
        write(out / "runtime_binding.json", {"ACT_binding": binding, "source_binding": sources, "environment": parity})
        for seed in test["test_source_seeds"]:
            root = out / "sources" / str(seed)
            if (root / "done.json").exists():
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
            attempt.mkdir()
            (attempt / "observations").mkdir()
            counter = {"nominal": 0, "replay": 0, "candidate": 0}
            entry = {"status": "running", "seed": seed, "attempt": str(attempt.relative_to(out)),
                     "environment_steps_by_kind": counter, "started_utc": datetime.now(timezone.utc).isoformat()}
            begin = time.monotonic()
            write(attempt / "report.json", entry)

            def on_context(context, seen, candidates):
                eligible = bool(seen.valid and all(context["valid"]))
                scores, seconds = {}, {}
                if eligible:
                    inputs = (torch.as_tensor(seen.features.values, dtype=torch.float32, device="cuda"),
                              torch.as_tensor([context["agent_xy"]], dtype=torch.float32, device="cuda"), candidates.actions)
                    scores, seconds = score_inputs(models, inputs)
                row = {**context, "actions": context["actions"].tolist(), "eligible_input": eligible,
                       "scores": {k: scores[k][0].tolist() if eligible else [None] * 5 for k in models},
                       "selected": {k: int(scores[k][0].argmax()) if eligible else 0 for k in models},
                       "scoring_seconds": seconds, "scores_saved_before_this_anchor_future": True}
                common._append(attempt / "predictions.jsonl", row)

            try:
                nominal = collection.nominal_prefix(env, runtime, act, prior, seed, attempt, counter,
                                                    original, reset_ids, on_context=on_context)
                for context in nominal["contexts"]:
                    for candidate, valid in enumerate(context["valid"]):
                        if valid:
                            collection.previous.branch(env, runtime, seed, context, nominal, candidate, goal, attempt, counter)
                if len(nominal["contexts"]) + len(nominal["skipped"]) != 3 or sum(counter.values()) > collection.MAX_STEPS_PER_SOURCE:
                    raise ValueError("fixed source context/step budget exceeded")
                entry.update({"status": "completed_source", "contexts": len(nominal["contexts"]),
                    "skipped_anchors": nominal["skipped"], "reset_observation_sha256": nominal["reset_id"],
                    "all_replayed_observations_exact": True, "ACT_reference_continuations_exact": True,
                    "all_six_predictions_before_future": True})
            except BaseException as exc:
                entry.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              "error": str(exc), "traceback": traceback.format_exc()})
                raise
            finally:
                entry.update({"environment_steps": sum(counter.values()), "runtime_seconds": time.monotonic() - begin})
                invocation_steps += sum(counter.values())
                write(attempt / "report.json", entry)
            write(root / "done.json", entry)
            reset_ids.add(entry["reset_observation_sha256"])
            new_sources += 1
            state.update({"completed_sources": len(completed(out, test)), "last_seed": seed,
                          "invocation_environment_steps": invocation_steps})
            write(out / "status.json", state)
            print(json.dumps(state), flush=True)
            if stop_after_sources is not None and new_sources >= stop_after_sources:
                break
        state["status"] = "sources_completed" if len(completed(out, test)) == 20 else "paused_at_source_boundary"
    except BaseException as exc:
        state.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if vector is not None:
            vector.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        state.update({"completed_sources": len(completed(out, test)), "runtime_seconds": time.monotonic() - start,
                      "invocation_environment_steps": invocation_steps, "all_recorded_attempt_environment_steps": all_attempt_steps(out)})
        common._append(out / "invocations.jsonl", state)
        write(out / "status.json", state)
    if state["status"] == "sources_completed":
        return summarize(out)
    print(json.dumps(state, indent=2), flush=True)
    return 1 if state["status"] in ("failed", "interrupted") else 0


def summarize(out):
    spec = contract(out)
    test, original = spec["test_protocol"], spec["training_protocol"]
    entries = completed(out, test)
    if len(entries) != 20:
        raise ValueError("finish all twenty planned sources before a test summary; no favorable partial-cohort report")
    keys, target, contexts, exclusions, skipped = [], [], [], [], []
    scores = {key: [] for key in spec["checkpoint_identity"]}
    scoring_seconds = {key: [] for key in scores}
    actual_candidates = 0
    for entry in entries:
        if not all(entry[k] for k in ("all_replayed_observations_exact", "ACT_reference_continuations_exact", "all_six_predictions_before_future")):
            raise ValueError("source replay/prediction provenance incomplete")
        folder = out / entry["attempt"]
        skipped.extend(entry["skipped_anchors"])
        predicted, actual = rows(folder / "predictions.jsonl"), rows(folder / "outcomes.jsonl")
        source_contexts = rows(folder / "contexts.jsonl")
        expected = {(entry["seed"], r["anchor_step"]) for r in source_contexts}
        if (len(predicted) != len(expected) or {(r["seed"], r["anchor_step"]) for r in predicted} != expected):
            raise ValueError("six-arm prediction/context alignment changed")
        keyed = {(r["anchor_step"], r["candidate"]): r for r in actual}
        if len(keyed) != len(actual) or any(r["seed"] != entry["seed"] for r in actual):
            raise ValueError("duplicate or mismatched candidate outcome")
        actual_candidates += len(actual)
        for prediction in predicted:
            futures = [keyed.get((prediction["anchor_step"], k)) for k in range(5)]
            reasons = []
            if not prediction["current_valid"]:
                reasons.append("invalid_current_object")
            if not all(prediction["valid"]):
                reasons.append("not_all_five_candidates_valid")
            if any(r is None or not r["full_horizon"] or r["steps"] != 8 for r in futures):
                reasons.append("not_all_five_complete8step_futures")
            if reasons:
                exclusions.append({"seed": prediction["seed"], "anchor_step": prediction["anchor_step"], "reasons": reasons})
                continue
            keys.append((prediction["seed"], prediction["anchor_step"]))
            target.append([r["terminal_coverage"] for r in futures])
            contexts.append({"source_seed": prediction["seed"], "anchor_step": prediction["anchor_step"],
                             "source_attempt": entry["attempt"], "current_rgb": prediction["current_rgb"]})
            for key in scores:
                value = np.asarray(prediction["scores"][key], dtype=float)
                if value.shape != (5,) or not np.isfinite(value).all() or int(value.argmax()) != prediction["selected"][key]:
                    raise ValueError("saved score or choice changed")
                scores[key].append(value)
                scoring_seconds[key].append(prediction["scoring_seconds"][key])
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 5)
    scores = {k: np.asarray(v, dtype=np.float64).reshape(-1, 5) for k, v in scores.items()}
    if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("invalid actual terminal coverage")
    training.save_arrays(out / "predictions.npz", context_keys=keys, **scores)
    training.save_arrays(out / "targets.npz", terminal_coverage=target)
    write(out / "contexts.json", contexts)
    write(out / "manifest.json", {"schema": SCHEMA, "data_role": "test", "training_allowed": False,
        "planned_sources": 20, "eligible_sources": len(np.unique(keys[:, 0])), "contexts": len(keys),
        "source_seeds": sorted(set(keys[:, 0].tolist())), "future_targets_are_observations": False})
    metrics, primary, controls, mean_methods, harm_difference = {}, None, {}, {}, None
    decision = "no_eligible_evidence_no_replacement_or_promotion"
    if len(keys):
        sources = np.unique(keys[:, 0])
        for seed in test["training_seeds"]:
            metrics[str(seed)] = training.selection_metrics(target,
                {arm: scores[model_key(seed, arm)] for arm in ARMS}, keys[:, 0], original, fixed_candidates=True)

        def source_matrix(name, metric):
            return np.array([[metrics[str(seed)]["methods"][name]["per_source_seed"][str(int(s))][metric]
                              for s in sources] for seed in test["training_seeds"]])

        boot = test["bootstrap"]
        effect = source_matrix(ARMS[1], "coverage")
        primary = cluster_interval(effect - source_matrix(ARMS[0], "coverage"), boot, boot["primary_confidence_level"])
        primary["source_seed_order"] = sources.tolist()
        for k in range(5):
            controls[str(k)] = cluster_interval(effect - source_matrix(f"fixed_candidate_{k}", "coverage"),
                                                boot, boot["control_individual_confidence_level"])
        for name in metrics[str(test["training_seeds"][0])]["methods"]:
            method = metrics[str(test["training_seeds"][0])]["methods"][name]
            mean_methods[name] = {metric: float(source_matrix(name, metric).mean()) for metric in method["source_seed_macro"]}
        harm_difference = float((source_matrix(ARMS[1], "harm_rate") - source_matrix(ARMS[0], "harm_rate")).mean())
        reliable = primary["percentile_interval"] is not None and primary["percentile_interval"][0] > 0
        beats_constants = all(r["percentile_interval"] is not None and r["percentile_interval"][0] > 0 for r in controls.values())
        repeats_positive = all(v > 0 for v in primary["per_training_repeat_mean"])
        if primary["mean"] <= 0 or harm_difference > 0:
            decision = "no_mean_gain_or_increased_harm_stop_scope_expansion_no_promotion"
        elif reliable and beats_constants and repeats_positive:
            decision = "supports_further_selection_research_not_automatic_policy_promotion"
        else:
            decision = "uncertain_or_fixed_direction_confound_unresolved_no_tuning_or_promotion"
    first_source_sheet(out, entries, spec)
    frozen_training = read(training.TRAIN_ROOT / "report.json")
    report = {"schema": SCHEMA, "status": "completed_frozen_twenty_source_candidate_test",
        "protocol": test, "checkpoint_identity": spec["checkpoint_identity"],
        "planned_source_seeds": test["test_source_seeds"], "executed_sources": len(entries),
        "eligible_sources": len(np.unique(keys[:, 0])), "eligible_contexts": len(keys),
        "eligible_candidates": int(target.size), "actually_executed_candidates": actual_candidates,
        "exclusions": exclusions, "unreached_anchors": skipped,
        "mean_source_macro_metrics_across_training_repeats": mean_methods, "per_training_seed": metrics,
        "primary_paired_effect": primary, "effect_difference_vs_all_fixed_indices": controls,
        "fixed_control_intervals": "five_99percent_percentile_intervals_Bonferroni_nominal_family95percent_approximation",
        "mean_harm_rate_difference_effect_minus_pointwise": harm_difference,
        "selected_candidate_histograms": {k: np.bincount(v.argmax(1), minlength=5).tolist() for k, v in scores.items()},
        "efficiency": {"parameter_count_each": training.PARAMETERS,
            "training_seconds_from_frozen_run": {str(s): frozen_training["training"][str(s)]["training_seconds_by_arm"] for s in test["training_seeds"]},
            "saved_validation_warm_ms": {str(s): frozen_training["validation"][str(s)]["warm_scoring_ms_one_context_five_candidates"] for s in test["training_seeds"]},
            "test_scoring_wall_ms_including_D2H_excluding_ACT_and_feature_extraction":
                {k: float(np.mean(v) * 1000) if v else None for k, v in scoring_seconds.items()}},
        "environment_steps_completed_sources": sum(r["environment_steps"] for r in entries),
        "environment_steps_all_recorded_attempts": all_attempt_steps(out),
        "optimizer_steps": 0, "hardware_actions": 0, "selector_closed_loop": False, "training_allowed": False,
        "test_used_for_fitting_or_selection": False, "visual_status": "not_viewed",
        "visual_selection": test["visual_cases"], "decision": decision, "policy_promotion": False,
        "freshness_limit": "unused_recorded_project_sources_not_unknown_original_dataset_states_or_broad_behavior_diversity",
        "inference_limit": "small_source_cluster_bootstrap_conditional_on_three_fixed_training_repeats_not_closed_loop_success"}
    write(out / "report.json", report)
    write(out / "status.json", {"status": report["status"], "completed_sources": len(entries),
        "environment_steps": report["environment_steps_all_recorded_attempts"], "optimizer_steps": 0})
    print(json.dumps({"status": report["status"], "eligible_sources": report["eligible_sources"],
                      "primary": primary, "decision": decision, "out": str(out)}, indent=2), flush=True)
    return 0


def first_source_sheet(out, entries, spec):
    """Fixed first source and first training seed, including missing/invalid anchors."""
    test = spec["test_protocol"]
    entry = next(r for r in entries if r["seed"] == test["test_source_seeds"][0])
    folder = out / entry["attempt"]
    predicted = {r["anchor_step"]: r for r in rows(folder / "predictions.jsonl")}
    actual = {(r["anchor_step"], r["candidate"]): r for r in rows(folder / "outcomes.jsonl")}
    font_path = next(p for p in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                                Path("C:/Windows/Fonts/msyh.ttc")) if p.is_file())
    title, font, small = (ImageFont.truetype(str(font_path), n) for n in (25, 19, 16))
    image = Image.new("RGB", (1510, 1140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), "冻结评分器测试：预先固定的首个测试来源", font=title, fill="black")
    draw.text((16, 47), f"来源430000全部起点；蓝=普通、红=效应对比；仅标首个训练seed20260918，完整三组结果见报告。", font=small, fill="#444444")
    for row, anchor in enumerate(test["anchors"]):
        top, p = 87 + 337 * row, predicted.get(anchor)
        draw.text((16, top), f"起点 {anchor}；输入有效={p['eligible_input'] if p else '未到达'}", font=font, fill="black")
        if p is None:
            draw.text((16, top + 45), "原 ACT 提前结束；不补选起点。", font=font, fill="black")
            continue
        for col in range(6):
            left, k = 16 + 250 * col, col - 1
            outcome = actual.get((anchor, k)) if col else None
            path = p["current_rgb"] if col == 0 else outcome["terminal_rgb"] if outcome else None
            draw.text((left, top + 31), "当前观测" if col == 0 else f"候选 {k}" + (" / ACT" if k == 0 else ""), font=small, fill="black")
            if path:
                with Image.open(folder / path) as tile:
                    image.paste(tile.convert("RGB").resize((164, 164)), (left, top + 59))
            if outcome:
                draw.text((left, top + 232), f"实际 {outcome['terminal_coverage']:.6f} / {outcome['steps']}步", font=small, fill="black")
                for j, (arm, name, color) in enumerate(((ARMS[0], "普通", "#2455bb"), (ARMS[1], "对比", "#bb3333"))):
                    key = model_key(test["training_seeds"][0], arm)
                    score = p["scores"][key][k]
                    draw.text((left, top + 257 + j * 24), f"{name} {score:+.6f}" if score is not None else f"{name} 未评分", font=small, fill=color)
                    if p["eligible_input"] and p["selected"][key] == k:
                        pad = 2 + j * 3
                        draw.rectangle((left - pad, top + 59 - pad, left + 163 + pad, top + 222 + pad), outline=color, width=2)
    draw.text((16, 1106), "均为保存的原生执行图像；预测先于未来落盘。不是模型生成后继，也不是选择器闭环/实机成功率。", font=small, fill="black")
    image.save(out / "first_test_source_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), default="prepare")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int, help="pause after N newly completed sources, never change the cohort")
    args = parser.parse_args()
    if args.stop_after_sources is not None and args.stop_after_sources < 1:
        parser.error("--stop-after-sources must be positive")
    if args.stage == "prepare":
        return prepare(args.out)
    if args.stage == "summarize":
        return summarize(args.out)
    return run(args.out, resume=args.resume, stop_after_sources=args.stop_after_sources)


if __name__ == "__main__":
    raise SystemExit(main())
