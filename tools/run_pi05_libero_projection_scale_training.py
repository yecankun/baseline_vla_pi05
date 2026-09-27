"""Fixed matched-budget projection-scale diagnostic; no policy/backbone training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import traceback

import torch

import run_pi05_libero_projection_scale_control as gate
from pi05_libero_projection_scale_checkpoint import save_reload
from pi05_libero_world_model_adapter import LiberoFeaturePack, read_json

previous, base = gate.previous, gate.base
training = previous.previous
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-projection-scale-training-plan-v1.json"
OUT = "simulation_output/pi05_libero_projection_scale_training_v1"
SMOKE_PLAN_SHA = "8976c32a2761c512220d1b6d102f0340411617cf213b9c4d1a48a9545b4aebe8"
SMOKE_REPORT_SHA = "16a2d07a686bb16b248d84ebaca1797a97eeb9886c5fbded8a3e85b5dcb25d66"
REFERENCE_REPORT_SHA = "a06f6957a6575502782e646eb76e5b69f75d8a4d57c1ab9ad9277008c5817520"
VARIANTS = gate.VARIANTS
CODE = {"tools/run_pi05_libero_projection_scale_training.py", "tools/test_run_pi05_libero_projection_scale_training.py",
        "tools/pi05_libero_projection_scale_checkpoint.py", "tools/test_pi05_libero_projection_scale_checkpoint.py"}
DESIGN = dict(seeds=list(base.SEEDS), arms=list(base.ARMS), variants=list(VARIANTS),
    steps_per_model=200, total_optimizer_steps=2400, batch_size=16, train_windows=1086,
    validation_windows=500, validation_episodes=[1458, 1476, 1530, 1566],
    train_episodes=[1312, 1419, 1518, 1520, 1531, 1633, 1674, 1690],
    history=4, horizon=3, hidden_dim=128, parameter_elements=1321352,
    optimizer=training.training.OPTIMIZER_CONFIG, clip_norm=1.0,
    loss="unchanged visual SmoothL1(beta=1) + 0.25 normalized-state SmoothL1(beta=1)",
    draws="original_shared_200x16_per_seed", normalization_refit=False,
    formula=gate.FORMULA, projection_scale_compute="float64_intermediate_float32_output",
    evaluation_phases=["step0", "final200"], evaluation_conditions=["clean"],
    evaluation_batching="original_episode_homogeneous_33_batches", evaluation_forwards=792,
    checkpoint_saves=12, checkpoint_loads=12, checkpoint_selection=False,
    training_execution_allowed=True, primary_arm="observed_action", secondary_arm="normalized_zero_action",
    reference_replay="all1200_update_records_final_parameter_bytes_all396_clean_predictions_and_scores",
    primary_gain="every_seed_objective_delta_negative_and_every_mean_global_and_horizon_visual_state_MAE_delta_nonpositive",
    original_raw_reference="saved_matching_final200_metrics_only_not_new_training",
    aggregation="episode_macro_then_equal_seed_mean; paired_population_std_descriptive_only",
    robustness_evaluation=False, validation_is_reused_development_set=True,
    backbone_training=False, policy_rollout=False, hyperparameter_search=False)


def validate(root, plan_sha):
    plan = read_json(base.checked(root, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_projection_scale_training_plan_v1" and plan["design"] == DESIGN
        and plan["output"] == OUT and plan["smoke_plan_sha256"] == SMOKE_PLAN_SHA
        and plan["smoke_report_sha256"] == SMOKE_REPORT_SHA
        and plan["reference_report_sha256"] == REFERENCE_REPORT_SHA
        and set(plan["input_sha256"]) == CODE, "fixed matched training protocol differs")
    prior, _, visual_stats = gate.validate(root, SMOKE_PLAN_SHA)
    smoke = read_json(base.checked(root, gate.OUT + "/report.json", SMOKE_REPORT_SHA))
    base.require(smoke["status"] == "passed_projection_scale_control_zero_update_gate"
        and smoke["plan_sha256"] == SMOKE_PLAN_SHA and smoke["optimizer_steps"] == 0
        and smoke["backward_calls"] == 12 and smoke["model_forwards"] == 24
        and read_json(root / gate.OUT / "status.json") == dict(status="completed", report_sha256=SMOKE_REPORT_SHA, optimizer_steps=0)
        and not (root / gate.OUT / "failure.json").exists(), "completed zero-update gate required")
    old = read_json(base.checked(root, previous.OUT + "/report.json", REFERENCE_REPORT_SHA))
    base.require(old["status"] == "completed_fixed_budget_normalization_comparison"
        and old["optimizer_steps"] == 2400 and not (root / previous.OUT / "failure.json").exists()
        and read_json(root / previous.OUT / "status.json") == dict(status="completed", report_sha256=REFERENCE_REPORT_SHA, optimizer_steps=2400),
        "completed normalization comparison required")
    pins = {**prior["input_sha256"], **smoke["input_sha256"], **plan["input_sha256"], PLAN: plan_sha,
        gate.OUT + "/report.json": SMOKE_REPORT_SHA,
        **{gate.OUT + "/" + k: v for k,v in smoke["output_sha256"].items()},
        previous.OUT + "/report.json": REFERENCE_REPORT_SHA,
        **{previous.OUT + "/" + k: v for k,v in old["output_sha256"].items()}}
    training.rehash(root, pins)
    return dict(status="passed_projection_scale_training_preflight", plan=plan,
                plan_sha256=plan_sha, input_sha256=pins), smoke, old, visual_stats


def reference_maps(root):
    evaluation, updates = {}, {}
    with (root / previous.OUT / "evaluation_trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["variant"] == "train_visual_normalized" and row["condition"] == "clean":
                key = (row["seed"], row["arm"], row["phase"], tuple(row["window_indices"]))
                base.require(key not in evaluation, "duplicate reference evaluation")
                evaluation[key] = row
    with (root / previous.OUT / "training_trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.pop("variant") == "train_visual_normalized":
                key = (row["seed"], row["arm"], row["step"])
                base.require(key not in updates, "duplicate reference update")
                updates[key] = row
    base.require(len(evaluation) == 396 and len(updates) == 1200, "complete original normalized references required")
    return evaluation, updates


def metric_vector(episodes):
    """Equal episode weighting, including horizon RMSE = mean episode RMSE."""
    result = base.macro_metrics(episodes)
    for field, label in (("visual", "visual"), ("state_normalized", "normalized_state")):
        for h in range(3):
            for metric in ("mae", "rmse"):
                result[f"{label}_h{h+1}_{metric}"] = statistics.mean(r[field]["per_horizon"][h][metric] for r in episodes.values())
    for name in ("visual_loss", "state_loss"):
        result[name] = statistics.mean(r["masked_objective"][name] for r in episodes.values())
    return result


class ActivationTotals:
    def __init__(self):
        self.native = previous.ViewTotals()
        self.rms = {str(v): {field: [] for field in ("raw_affine", "pre")} for v in range(2)}

    def update(self, values):
        self.native.update(values)
        for view in self.rms:
            for field in self.rms[view]:
                self.rms[view][field].extend(values[view][field].double().square().mean(-1).sqrt().flatten().tolist())

    def summary(self):
        native = self.native.summary()
        rms = {v: {f: dict(count=len(xs), min=min(xs), mean=statistics.mean(xs), max=max(xs))
                   for f,xs in fields.items()} for v,fields in self.rms.items()}
        return native, rms


def evaluate(model, dataset, stats, *, seed, arm, variant, phase, references, trace=None):
    base.require(variant in VARIANTS and phase in ("step0", "final200"), "unknown evaluation variant/phase")
    model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
    before = base.parameter_hash(model)
    groups, episodes, activations, rms = {}, {}, {}, {}
    for i,w in enumerate(dataset.windows): groups.setdefault(w.episode_index, []).append(i)
    all_score = base.Totals(stats)
    count = forwards = 0
    for eid, indices in groups.items():
        score, views = base.Totals(stats), ActivationTotals()
        for start in range(0, len(indices), 16):
            ids = indices[start:start+16]
            items = [dataset[i] for i in ids]
            raw = previous.collate_window_inputs([x["inputs"] for x in items])
            targets = previous.collate_window_targets([x["targets"] for x in items])
            inputs = previous.action_ablation_inputs(raw, arm)
            fp, tfp = base.fingerprints(inputs), base.fingerprints(targets)
            ref = references[(seed, arm, phase, tuple(ids))]
            base.require(fp == ref["input_sha256"] and tfp == ref["target_sha256"]
                and eid == ref["episode_index"], "fixed evaluation input/target/episode differs")
            captured, values = gate.capture_views(model, inputs)
            views.update(values)
            batch = base.Totals(stats)
            for total in (score, all_score, batch): total.update(captured["predictions"], targets)
            pred_fp = base.fingerprints(captured["predictions"])
            if variant == VARIANTS[0]:
                base.require(pred_fp == ref["prediction_sha256"] and batch.summary() == ref["score"],
                    "normalized reference prediction/score did not replay exactly")
            base.require(base.fingerprints(inputs) == fp and base.fingerprints(targets) == tfp,
                "evaluation mutated input/target")
            if trace is not None:
                trace.write(base.canonical(dict(seed=seed, arm=arm, variant=variant, phase=phase, condition="clean",
                    episode_index=eid, window_indices=ids, input_sha256=fp, target_sha256=tfp,
                    prediction_sha256=pred_fp, score=batch.summary())) + "\n")
                trace.flush()
            count += len(ids); forwards += 1
        episodes[str(eid)] = score.summary()
        activations[str(eid)], rms[str(eid)] = views.summary()
    base.require(count == len(dataset) and base.parameter_hash(model) == before, "evaluation coverage/model differs")
    return dict(per_episode=episodes, episode_macro=metric_vector(episodes), micro=all_score.summary(),
        activation=dict(per_episode=activations, episode_macro=previous.view_macro(activations), vector_rms=rms),
        windows=count, model_forwards=forwards)


class PairedTrace:
    """Trace metadata cannot change old update arithmetic; replay every reference row."""
    def __init__(self, stream, variant, references):
        self.stream, self.variant, self.references = stream, variant, references
        self.steps = 0

    def write(self, line):
        row = json.loads(line)
        key = (row["seed"], row["arm"], row["step"])
        ref = self.references[key]
        base.require(row["step"] == self.steps+1, "update trace sequence differs")
        for name in ("window_indices", "input_sha256", "target_sha256"):
            base.require(row[name] == ref[name], "shared training draw/input/target differs")
        if self.variant == VARIANTS[0]:
            base.require(row == ref, "normalized reference update did not replay exactly")
        self.steps += 1
        self.stream.write(base.canonical(dict(row, variant=self.variant)) + "\n")

    def flush(self): self.stream.flush()


def summarize(runs, old):
    means, changes, paired, raw_means = {}, {}, {}, {}
    for arm in base.ARMS:
        vectors = {v: {str(s): runs[str(s)][arm][v]["final"]["episode_macro"] for s in base.SEEDS} for v in VARIANTS}
        means[arm] = {v: {m: statistics.mean(r[m] for r in vectors[v].values()) for m in next(iter(vectors[v].values()))} for v in VARIANTS}
        paired[arm] = {str(s): base.changes(vectors[VARIANTS[1]][str(s)], vectors[VARIANTS[0]][str(s)]) for s in base.SEEDS}
        changes[arm] = base.changes(means[arm][VARIANTS[1]], means[arm][VARIANTS[0]])
        raw = [metric_vector(old["runs"][str(s)][arm]["baseline"]["final"]["conditions"]["clean"]["per_episode"]) for s in base.SEEDS]
        raw_means[arm] = {m: statistics.mean(r[m] for r in raw) for m in raw[0]}
    primary = base.ARMS[0]
    guard_keys = [m for m in changes[primary] if m.endswith("mae")]
    objective_guard = all(paired[primary][str(s)]["masked_objective"]["absolute"] < 0 for s in base.SEEDS)
    mae_guards = {m: changes[primary][m]["absolute"] <= 0 for m in guard_keys}
    gain = objective_guard and all(mae_guards.values())
    desaturation = all(runs[str(s)][primary][VARIANTS[1]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"]
        < runs[str(s)][primary][VARIANTS[0]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"]
        for s in base.SEEDS for v in ("0", "1"))
    return dict(clean_seed_means=means, clean_changes_of_seed_means=changes, paired_seed_changes=paired,
        paired_delta_population_std={a: {m: statistics.pstdev(paired[a][str(s)][m]["absolute"] for s in base.SEEDS)
            for m in changes[a]} for a in base.ARMS},
        saved_raw_baseline_seed_means=raw_means,
        changes_vs_saved_raw_baseline={a: {v: base.changes(means[a][v], raw_means[a]) for v in VARIANTS} for a in base.ARMS},
        primary_all_seed_objective_guard=objective_guard, primary_mean_MAE_guards=mae_guards,
        primary_desaturation_all_seed_view=desaturation,
        primary_fixed_validation_prediction_gain=gain,
        decision="descriptive_fixed_validation_prediction_gain" if gain else "mixed_or_no_stable_overall_prediction_gain",
        automatic_checkpoint_promotion=False, validation_is_reused_development_set=True,
        independent_test_generalization=False, policy_benefit=False, robustness_claim=False,
        architecture_innovation_claim=False)


def train(root, plan_sha, execute):
    base.require(execute is True, "explicit matched training execution flag required")
    preflight, smoke, old, visual_stats = validate(root, plan_sha)
    training.runtime_settings()
    eval_refs, train_refs = reference_maps(root)
    out = previous.claim(root, OUT, preflight)
    began, runs, updates = time.monotonic(), {}, 0
    try:
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            datasets, stats, draws, step0, _ = training.load_inputs(root, pack)
            split = pack.load_split(root / base.PACK_PATH / "split.json")
            base.require(sorted(split["train_episode_indices"]) == DESIGN["train_episodes"]
                and sorted(split["validation_episode_indices"]) == DESIGN["validation_episodes"]
                and visual_stats["feature_pack_manifest_sha256"] == pack.manifest_sha256
                and visual_stats["split_sha256"] == split["split_sha256"], "fixed split/normalization differs")
            registry = pack.manifest["task_registry"]
            with (out / "training_trace.jsonl").open("x", encoding="utf-8") as train_log, \
                 (out / "evaluation_trace.jsonl").open("x", encoding="utf-8") as eval_log:
                for seed in base.SEEDS:
                    runs[str(seed)] = {}
                    for arm in base.ARMS:
                        runs[str(seed)][arm] = {}
                        for variant in VARIANTS:
                            print(f"scale training seed={seed} arm={arm} variant={variant} budget=200", flush=True)
                            model = gate.build_model(seed, registry, variant, visual_stats)
                            init_sha = previous.named_parameter_sha(model)
                            base.require(init_sha == smoke["results"][str(seed)][arm]["models"][variant]["named_parameter_sha256"]
                                and sum(p.numel() for p in model.parameters()) == DESIGN["parameter_elements"], "fixed initialization/capacity differs")
                            initial = evaluate(model, datasets["validation"], stats, seed=seed, arm=arm, variant=variant,
                                phase="step0", references=eval_refs, trace=eval_log)
                            buffers = base.fingerprints(dict(model.named_buffers()))
                            trace = PairedTrace(train_log, variant, train_refs)
                            optimizer, timing = training.train_arm(model, datasets["train"], draws[seed], arm, trace)
                            updates += timing["optimizer_steps"]
                            base.require(trace.steps == 200 and base.fingerprints(dict(model.named_buffers())) == buffers,
                                "training budget or fixed buffers changed")
                            model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
                            if variant == VARIANTS[0]:
                                old_binding = old["runs"][str(seed)][arm]["train_visual_normalized"]["checkpoint"]["binding"]
                                base.require(previous.named_parameter_sha(model) == old_binding["final_named_parameter_sha256"]
                                    and base.parameter_hash(model) == old_binding["final_state_sha256"], "old normalized final parameters differ")
                            binding = dict(seed=seed, arm=arm, variant=variant, step=200, plan_sha256=plan_sha,
                                smoke_report_sha256=SMOKE_REPORT_SHA, normalization_sha256=smoke["normalization_sha256"],
                                normalization_applied=True, projection_scale_control_applied=variant == VARIANTS[1],
                                draw_sha256=step0["sampling"][str(seed)]["sha256"], initial_named_parameter_sha256=init_sha,
                                final_named_parameter_sha256=previous.named_parameter_sha(model), final_state_sha256=base.parameter_hash(model))
                            fresh, checkpoint = save_reload(out / f"seed{seed}_{arm}_{variant}_final200.pt", model, optimizer,
                                registry=registry, visual_stats=visual_stats, binding=binding)
                            final = evaluate(fresh, datasets["validation"], stats, seed=seed, arm=arm, variant=variant,
                                phase="final200", references=eval_refs, trace=eval_log)
                            for result in (initial, final):
                                base.require(result["windows"] == 500 and result["model_forwards"] == 33, "fixed evaluation coverage differs")
                            if variant == VARIANTS[0]:
                                for phase,new in (("initial", initial), ("final", final)):
                                    ref = old["runs"][str(seed)][arm]["train_visual_normalized"][phase]
                                    base.require(new["per_episode"] == ref["conditions"]["clean"]["per_episode"]
                                        and new["micro"] == ref["conditions"]["clean"]["micro"]
                                        and base.macro_metrics(new["per_episode"]) == ref["conditions"]["clean"]["episode_macro"]
                                        and new["activation"]["per_episode"] == ref["activation"]["per_episode"], "old normalized episode/activation replay differs")
                            row = dict(initial=initial, training=timing, final=final, checkpoint=checkpoint)
                            runs[str(seed)][arm][variant] = row
                            base.write_json(out / f"seed{seed}_{arm}_{variant}_result.json", row)
                            del model, optimizer, fresh
        forwards = sum(v[p]["model_forwards"] for r in runs.values() for a in r.values() for v in a.values() for p in ("initial", "final"))
        base.require(updates == 2400 and forwards == 792, "incomplete fixed budget")
        training.rehash(root, preflight["input_sha256"])
        previous.finish(out, dict(schema="libero_projection_scale_training_result_v1",
            status="completed_fixed_budget_projection_scale_comparison", plan_sha256=plan_sha,
            smoke_report_sha256=SMOKE_REPORT_SHA, reference_report_sha256=REFERENCE_REPORT_SHA,
            input_sha256=preflight["input_sha256"], normalization_sha256=smoke["normalization_sha256"],
            runs=runs, summary=summarize(runs, old), optimizer_steps=updates, backward_calls=updates,
            evaluation_forwards=forwards, checkpoint_saves=12, checkpoint_loads=12,
            reference_update_records_replayed=1200, reference_evaluation_predictions_replayed=396,
            reference_final_models_replayed=6, old_inputs_unchanged=True,
            runtime=training.runtime_evidence(), wall_seconds=time.monotonic()-began,
            feature_encodings=0, rollout_steps=0, formal_training_ready=False))
    except BaseException as exc:
        base.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
            completed_arm_optimizer_steps=updates, partial_current_arm_may_have_steps=True, preserve_no_retry=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        base.require(not args.execute, "preflight has no execution flag")
        preflight, _, _, _ = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=preflight["status"], input_pins=len(preflight["input_sha256"]))))
    else: train(ROOT, args.plan_sha256, args.execute)


if __name__ == "__main__": main()
