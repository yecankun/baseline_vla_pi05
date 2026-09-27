"""One-factor train-only visual normalization study; old artifacts stay immutable.

Explicit smoke and matched-training stages. No PI0.5 loading or fine-tuning,
source decoding, new split, hyperparameter search, or policy rollout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

import run_pi05_libero_activation_inspection as inspection
import run_pi05_libero_visual_dependence as dependence
from pi05_libero_activation_inspection import MomentTotals, TanhTotals
from pi05_libero_visual_dependence import capture_prediction, transform_inputs, DriftTotals, _hook_state
from pi05_libero_visual_normalization import NormalizedLiberoWorldModel, fit_visual_statistics, validate_statistics
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
from pi05_libero_world_model_adapter import LiberoFeaturePack, read_json, sha256_file
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_action_ablation import action_ablation_inputs
import pi05_libero_world_model_training as native

base, previous = dependence.base, dependence.previous
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-visual-normalization-plan-v1.json"
SMOKE = "simulation_output/pi05_libero_visual_normalization_smoke_v1"
OUT = "simulation_output/pi05_libero_visual_normalization_v1"
OLD_PLAN_SHA = "3a0c79f69dffc998394795057ceb1ce2e1e8841cd55f46cefc8fba00f3a6c1ea"
OLD_REPORT_SHA = "963dad4c9450cf6c34e4be3b1a3be04e06861502c099f0e6188864aaaba8e980"
VARIANTS = ("baseline", "train_visual_normalized")
CONDITIONS = dependence.CONDITIONS
CODE = {"tools/run_pi05_libero_visual_normalization.py", "tools/pi05_libero_visual_normalization.py",
        "tools/test_pi05_libero_visual_normalization.py", "tools/test_run_pi05_libero_visual_normalization.py"}
DESIGN = dict(seeds=list(base.SEEDS), arms=list(base.ARMS), variants=list(VARIANTS),
    steps_per_model=200, total_optimizer_steps=2400, batch_size=16,
    train_windows=1086, validation_windows=500, train_unique_rows=1134,
    scale_floor=0.1, fit_partition="train_unique_complete_episode_rows_only",
    fit_dtype="float64", apply_dtype="float32", conditions=list(CONDITIONS),
    optimizer=native.OPTIMIZER_CONFIG, clip_norm=1.0,
    training_execution_allowed=True, backbone_training=False, policy_rollout=False,
    checkpoint_selection=False, hyperparameter_search=False,
    primary_arm="observed_action",
    decision="all_three_primary_objective_deltas_negative_and_both_mean_MAEs_nonpositive",
    responsiveness="descriptive_drift_and_intervened_target_error_not_automatic_benefit")


def named_parameter_sha(model):
    return hashlib.sha256(base.canonical(base.fingerprints(dict(model.named_parameters()))).encode()).hexdigest()


def build_model(seed, registry, variant, visual_stats, config=None):
    base.require(variant in VARIANTS, "unknown normalization variant")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        cfg = config or LiberoWorldModelConfig()
        model = (LiberoWorldModel(cfg, registry) if variant == "baseline"
                 else NormalizedLiberoWorldModel(cfg, registry, visual_stats))
    return model.cpu().eval().requires_grad_(False)


def validate(repo, plan_sha):
    plan = read_json(base.checked(repo, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_visual_normalization_plan_v1"
                 and plan["design"] == DESIGN and plan["output"] == OUT and plan["smoke_output"] == SMOKE
                 and plan["prior_plan_sha256"] == OLD_PLAN_SHA and plan["prior_report_sha256"] == OLD_REPORT_SHA
                 and set(plan["input_sha256"]) == CODE, "fixed authorized normalization protocol differs")
    old = inspection.validate(repo, OLD_PLAN_SHA)
    report = read_json(base.checked(repo, inspection.OUT + "/report.json", OLD_REPORT_SHA))
    base.require(report["status"] == "completed_frozen_activation_inspection"
                 and read_json(repo / inspection.OUT / "status.json")["report_sha256"] == OLD_REPORT_SHA
                 and not (repo / inspection.OUT / "failure.json").exists(), "completed saturation evidence required")
    pins = {**old["input_sha256"], **plan["input_sha256"], PLAN: plan_sha,
            inspection.OUT + "/report.json": OLD_REPORT_SHA,
            **{inspection.OUT + "/" + name: sha for name, sha in report["output_sha256"].items()}}
    previous.rehash(repo, pins)
    return dict(status="passed_normalization_preflight", plan=plan, plan_sha256=plan_sha, input_sha256=pins)


def capture_views(model, inputs):
    """Actual downstream tanh values + actual residual, in exactly one forward."""
    hooks, before = _hook_state(model), base.parameter_hash(model)
    saved, handles = {}, []
    def save_projection(view):
        def hook(_module, args, result):
            base.require(view not in saved, "visual projection ran more than once")
            saved[view] = (args[0].detach().clone(), result.detach().clone())
        return hook
    def save_history(_module, args, _result):
        base.require("history" not in saved, "history projection ran more than once")
        saved["history"] = args[0].detach().clone()
    try:
        for view, layer in enumerate(model.view_projections):
            handles.append(layer.register_forward_hook(save_projection(view)))
        handles.append(model.history_projection.register_forward_hook(save_history))
        captured = capture_prediction(model, inputs)
        base.require(set(saved) == {0, 1, "history"}, "missing activation capture")
        h = model.config.hidden_dim
        views = {}
        for view in range(2):
            actual_input, pre = (value.clone() for value in saved[view])
            expected = inputs["history_visual_latent"][:, :, view]
            if isinstance(model, NormalizedLiberoWorldModel):
                expected = (expected - model.visual_mean[view]) / model.visual_scale[view]
            base.require(torch.equal(actual_input, expected), "actual preprojection transform differs")
            post = saved["history"][..., view*h:(view+1)*h].clone()
            base.require(torch.equal(torch.tanh(pre), post), "actual downstream tanh differs")
            views[str(view)] = dict(pre=pre, post=post, projected_input=actual_input)
        return captured, views
    finally:
        for handle in handles:
            handle.remove()
        base.require(_hook_state(model) == hooks and base.parameter_hash(model) == before,
                     "capture mutated model parameters/buffers/hooks")


class ViewTotals:
    def __init__(self):
        self.tanh = {str(v): TanhTotals() for v in range(2)}
        self.post = {}

    def update(self, values):
        for view, tensors in values.items():
            self.tanh[view].update(tensors["pre"], tensors["post"])
            if view not in self.post:
                self.post[view] = MomentTotals(feature_width=tensors["post"].shape[-1])
            self.post[view].update(tensors["post"])

    def summary(self):
        return {view: dict(tanh=self.tanh[view].summary(), post=self.post[view].summary()) for view in self.post}


def view_macro(episodes):
    return {v: dict(saturated_fraction=statistics.mean(r[v]["tanh"]["post_abs_ge"]["0.99"]["fraction"] for r in episodes.values()),
                   exact_one_fraction=statistics.mean(r[v]["tanh"]["post_abs_exact_1_fraction"] for r in episodes.values()),
                   mean_slope=statistics.mean(r[v]["tanh"]["mean_slope"] for r in episodes.values()),
                   coordinate_std_rms=statistics.mean(r[v]["post"]["coordinate_std_rms"] for r in episodes.values()))
            for v in ("0", "1")}


def evaluate(model, dataset, stats, *, seed, arm, variant, phase, anchor, references, trace=None):
    conditions = ("clean",) if phase == "step0" else CONDITIONS
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    before = base.parameter_hash(model)
    groups = {}
    for i, w in enumerate(dataset.windows):
        groups.setdefault(w.episode_index, []).append(i)
    all_score = {c: base.Totals(stats) for c in conditions}
    all_drift = {c: DriftTotals() for c in conditions}
    episodes, drift_episodes, activations = {c: {} for c in conditions}, {c: {} for c in conditions}, {}
    count = batches = 0
    for eid, indices in groups.items():
        sub = {c: base.Totals(stats) for c in conditions}
        ds = {c: DriftTotals() for c in conditions}
        vs = ViewTotals()
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[i] for i in selected]
            raw = collate_window_inputs([x["inputs"] for x in items])
            targets = collate_window_targets([x["targets"] for x in items])
            source_fp, target_fp = base.fingerprints(raw), base.fingerprints(targets)
            ref = references[(seed, arm, tuple(selected))]
            base.require(ref["target_sha256"] == target_fp, "fixed target differs")
            clean_capture = clean_inputs = None
            for c in conditions:
                inputs = transform_inputs(raw, arm, c, anchor)
                fp = base.fingerprints(inputs)
                base.require(all(fp[k] == value for k, value in ref["input_sha256"].items()
                                 if c == "clean" or k != "history_visual_latent"), "input protocol differs")
                if c == "clean":
                    captured, values = capture_views(model, inputs)
                    vs.update(values)
                    clean_capture, clean_inputs = captured, inputs
                    if variant == "baseline":
                        base.require(base.fingerprints(captured["predictions"]) == ref["prediction_sha256"],
                                     "fresh baseline did not reproduce old prediction bytes")
                else:
                    captured = capture_prediction(model, inputs)
                batch_score, drift = base.Totals(stats), DriftTotals()
                for accumulator in (all_score[c], sub[c], batch_score):
                    accumulator.update(captured["predictions"], targets)
                for accumulator in (all_drift[c], ds[c], drift):
                    accumulator.update(clean_capture, captured, clean_inputs, inputs)
                base.require(base.fingerprints(inputs) == fp and base.fingerprints(targets) == target_fp,
                             "evaluation changed inputs/targets")
                if trace is not None:
                    trace.write(base.canonical(dict(seed=seed, arm=arm, variant=variant, phase=phase,
                        condition=c, episode_index=eid, window_indices=selected, input_sha256=fp,
                        target_sha256=target_fp, prediction_sha256=base.fingerprints(captured["predictions"]),
                        score=batch_score.summary(), drift=drift.summary())) + "\n")
                batches += 1
            base.require(base.fingerprints(raw) == source_fp, "raw evaluation input mutated")
            count += len(items)
        for c in conditions:
            episodes[c][str(eid)], drift_episodes[c][str(eid)] = sub[c].summary(), ds[c].summary()
        activations[str(eid)] = vs.summary()
    base.require(count == len(dataset) and base.parameter_hash(model) == before, "evaluation coverage/model differs")
    return dict(conditions={c: dict(per_episode=episodes[c], episode_macro=base.macro_metrics(episodes[c]),
                 micro=all_score[c].summary(), drift=dict(per_episode=drift_episodes[c],
                 episode_macro=dependence.drift_macro(drift_episodes[c]), micro=all_drift[c].summary())) for c in conditions},
                 activation=dict(per_episode=activations, episode_macro=view_macro(activations)),
                 windows=count, model_forwards=batches)


def reference_maps(root):
    initial, final = {}, {}
    for row in (json.loads(x) for x in (root / base.OUT_PATH / "evaluation_trace.jsonl").read_text().splitlines()):
        if row["partition"] != "validation":
            continue
        for arm in base.ARMS:
            initial[(row["seed"], arm, tuple(row["window_indices"]))] = dict(
                input_sha256=row["arm_input_sha256"][arm], target_sha256=row["original_target_sha256"],
                prediction_sha256=row["predictions_sha256"][arm])
    for row in (json.loads(x) for x in (root / previous.TRAIN_ROOT / "evaluation_final.jsonl").read_text().splitlines()):
        if row["partition"] == "validation":
            final[(row["seed"], row["arm"], tuple(row["window_indices"]))] = row
    base.require(len(initial) == len(final) == 198, "complete reference groups required")
    return initial, final


def claim(root, relative, preflight):
    out = root / relative
    base.require(not out.exists(), "output exists; preserve it, no overwrite or automatic retry")
    out.mkdir()
    base.write_json(out / "preflight.json", preflight)
    base.write_json(out / "started.json", dict(pid=os.getpid(), unix_seconds=time.time()))
    return out


def finish(out, report):
    report["output_sha256"] = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    base.write_json(out / "report.json", report)
    base.write_json(out / "status.json", dict(status="completed", report_sha256=sha256_file(out / "report.json"),
                                               optimizer_steps=report["optimizer_steps"]))


def smoke(root, plan_sha, execute):
    base.require(execute is True, "explicit smoke execution flag required")
    preflight = validate(root, plan_sha)
    previous.runtime_settings()
    out = claim(root, SMOKE, preflight)
    began = time.monotonic()
    try:
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            split = pack.load_split(root / base.PACK_PATH / "split.json")
            visual_stats = fit_visual_statistics(pack, split, scale_floor=DESIGN["scale_floor"])
            base.require(visual_stats["train_record_count"] == DESIGN["train_unique_rows"], "fixed unique train rows differ")
            base.write_json(out / "visual_normalization.json", visual_stats)
            datasets, stats, draws, old, _ = previous.load_inputs(root, pack)
            results = {}
            for seed in base.SEEDS:
                results[str(seed)] = {}
                raw, targets, indices = previous.draw_batch(draws[seed], datasets["train"], 1)
                for arm in base.ARMS:
                    results[str(seed)][arm] = {}
                    models = {v: build_model(seed, pack.manifest["task_registry"], v, visual_stats) for v in VARIANTS}
                    base.require(len({named_parameter_sha(m) for m in models.values()}) == 1,
                                 "paired trainable parameter initialization differs")
                    base.require(base.parameter_hash(models["baseline"]) == old["runs"][str(seed)]["train"]["parameter_sha256"][arm],
                                 "old initialization differs")
                    inputs = action_ablation_inputs(raw, arm)
                    for variant, model in models.items():
                        capture, values = capture_views(model, inputs)
                        vtot = ViewTotals(); vtot.update(values)
                        model.train().requires_grad_(True)
                        grad = previous.training.gradient_probe(model, inputs, targets, arm)
                        results[str(seed)][arm][variant] = dict(initial_named_parameter_sha256=named_parameter_sha(model),
                            window_indices=indices, input_sha256=base.fingerprints(inputs), target_sha256=base.fingerprints(targets),
                            prediction_sha256=base.fingerprints(capture["predictions"]), activation=vtot.summary(), gradient=grad)
        previous.rehash(root, preflight["input_sha256"])
        finish(out, dict(schema="libero_visual_normalization_smoke_v1", status="passed_twelve_no_update_gradient_checks",
            plan_sha256=plan_sha, input_sha256=preflight["input_sha256"], results=results,
            normalization_sha256=sha256_file(out / "visual_normalization.json"), optimizer_steps=0,
            backward_calls=12, model_forwards=24, wall_seconds=time.monotonic()-began, runtime=previous.runtime_evidence()))
    except BaseException as exc:
        base.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(), preserve_no_retry=True))
        raise


def checked_smoke(root, plan_sha, smoke_sha):
    report = read_json(base.checked(root, SMOKE + "/report.json", smoke_sha))
    base.require(report["plan_sha256"] == plan_sha and report["status"] == "passed_twelve_no_update_gradient_checks"
                 and report["optimizer_steps"] == 0 and report["backward_calls"] == 12
                 and not (root / SMOKE / "failure.json").exists(), "matching completed no-update smoke required")
    status = read_json(root / SMOKE / "status.json")
    base.require(status == dict(status="completed", report_sha256=smoke_sha, optimizer_steps=0), "smoke status differs")
    pins = {SMOKE + "/report.json": smoke_sha,
            **{SMOKE + "/" + name: sha for name, sha in report["output_sha256"].items()}}
    previous.rehash(root, {**report["input_sha256"], **pins})
    normal = read_json(base.checked(root, SMOKE + "/visual_normalization.json", report["normalization_sha256"]))
    validate_statistics(normal, 2, 2048)
    return report, normal, pins


def save_reload(path, model, optimizer, *, registry, visual_stats, binding):
    """Explicit new diagnostic checkpoint, not an old native parameter-only file."""
    keys = {"seed", "arm", "variant", "step", "plan_sha256", "smoke_report_sha256", "normalization_sha256",
            "normalization_applied", "draw_sha256", "initial_named_parameter_sha256",
            "final_named_parameter_sha256", "final_state_sha256"}
    base.require(type(binding) is dict and set(binding) == keys, "exact checkpoint binding required")
    base.require(type(binding["seed"]) is int and binding["seed"] in base.SEEDS
                 and binding["arm"] in base.ARMS and binding["variant"] in VARIANTS
                 and type(binding["step"]) is int and binding["step"] == 200,
                 "checkpoint seed/arm/variant/step differs")
    for key in keys:
        if key.endswith("sha256"):
            native._digest(binding[key])
    normalized = binding["variant"] != "baseline"
    base.require(binding["normalization_applied"] is normalized
                 and type(model) is (NormalizedLiberoWorldModel if normalized else LiberoWorldModel),
                 "checkpoint variant/model mismatch")
    validate_statistics(visual_stats, model.config.views, model.config.visual_dim)
    stats_file_bytes = (json.dumps(visual_stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    base.require(hashlib.sha256(stats_file_bytes).hexdigest() == binding["normalization_sha256"],
                 "checkpoint normalization provenance differs")
    base.require(named_parameter_sha(model) == binding["final_named_parameter_sha256"]
                 and base.parameter_hash(model) == binding["final_state_sha256"], "checkpoint bound state differs")
    native._optimizer(optimizer, native._model_parameters(model, trainable=False), 200)
    base.require(not Path(path).exists(), "checkpoint overwrite forbidden")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    payload = dict(schema="libero_visual_normalization_final_checkpoint_v1", binding=binding,
        config=asdict(model.config), registry=deepcopy(registry), model_state=state,
        normalization=deepcopy(visual_stats) if binding["variant"] != "baseline" else None,
        optimizer_state_saved=False, resumable=False)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    base.require(set(loaded) == set(payload) and loaded["schema"] == payload["schema"]
                 and loaded["binding"] == binding and loaded["config"] == payload["config"]
                 and loaded["registry"] == payload["registry"] and loaded["normalization"] == payload["normalization"]
                 and loaded["optimizer_state_saved"] is False and loaded["resumable"] is False,
                 "checkpoint normalization/provenance/schema differs")
    fresh = build_model(binding["seed"], registry, binding["variant"], visual_stats, LiberoWorldModelConfig(**loaded["config"]))
    expected = fresh.state_dict()
    base.require(set(loaded["model_state"]) == set(expected), "checkpoint state keys differ")
    for name, value in loaded["model_state"].items():
        base.require(isinstance(value, torch.Tensor) and value.layout == torch.strided and value.device.type == "cpu"
                     and not value.requires_grad and value.dtype == expected[name].dtype and value.shape == expected[name].shape
                     and bool(torch.isfinite(value).all()), "checkpoint tensor contract differs")
    fresh.load_state_dict(loaded["model_state"], strict=True)
    base.require(base.parameter_hash(fresh) == base.parameter_hash(model) == binding["final_state_sha256"]
                 and named_parameter_sha(fresh) == binding["final_named_parameter_sha256"]
                 and sha256_file(path) == digest, "checkpoint load/bytes changed")
    return fresh, dict(path=Path(path).name, sha256=digest, binding=binding, optimizer_state_saved=False, resumable=False)


def summarize(runs):
    means, changes, seed_changes = {}, {}, {}
    for arm in base.ARMS:
        means[arm] = {v: {metric: statistics.mean(runs[str(s)][arm][v]["final"]["conditions"]["clean"]["episode_macro"][metric]
                         for s in base.SEEDS) for metric in ("normalized_state_mae", "visual_mae", "masked_objective")}
                         for v in VARIANTS}
        changes[arm] = base.changes(means[arm][VARIANTS[1]], means[arm][VARIANTS[0]])
        seed_changes[arm] = {str(s): base.changes(runs[str(s)][arm][VARIANTS[1]]["final"]["conditions"]["clean"]["episode_macro"],
                                               runs[str(s)][arm][VARIANTS[0]]["final"]["conditions"]["clean"]["episode_macro"])
                             for s in base.SEEDS}
    primary = base.ARMS[0]
    gain = (all(seed_changes[primary][str(s)]["masked_objective"]["absolute"] < 0 for s in base.SEEDS)
            and all(changes[primary][m]["absolute"] <= 0 for m in ("normalized_state_mae", "visual_mae")))
    desaturated = all(runs[str(s)][primary][VARIANTS[1]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"]
                     < runs[str(s)][primary][VARIANTS[0]]["final"]["activation"]["episode_macro"][v]["saturated_fraction"]
                     for s in base.SEEDS for v in ("0", "1"))
    return dict(clean_seed_means=means, clean_changes_of_seed_means=changes, paired_seed_changes=seed_changes,
        paired_delta_population_std={a: {m: statistics.pstdev(seed_changes[a][str(s)][m]["absolute"] for s in base.SEEDS)
                                         for m in means[a][VARIANTS[0]]} for a in base.ARMS},
        primary_desaturation_all_seed_view=desaturated, primary_fixed_validation_prediction_gain=gain,
        decision="descriptive_fixed_validation_prediction_gain" if gain else "mixed_or_no_stable_overall_prediction_gain",
        intervention_drift_is_not_benefit=True, validation_is_reused_development_set=True,
        independent_test_generalization=False, policy_benefit=False, architecture_innovation_claim=False)


def train(root, plan_sha, smoke_sha, execute):
    base.require(execute is True and smoke_sha is not None, "explicit training and completed smoke SHA required")
    preflight = validate(root, plan_sha)
    smoke_report, visual_stats, smoke_pins = checked_smoke(root, plan_sha, smoke_sha)
    preflight["input_sha256"].update(smoke_pins)
    previous.runtime_settings()
    out = claim(root, OUT, preflight)
    began, runs, updates = time.monotonic(), {}, 0
    try:
        old = read_json(root / previous.TRAIN_ROOT / "report.json")
        initial_refs, final_refs = reference_maps(root)
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            datasets, stats, draws, step0, _ = previous.load_inputs(root, pack)
            split = pack.load_split(root / base.PACK_PATH / "split.json")
            base.require(visual_stats["feature_pack_manifest_sha256"] == pack.manifest_sha256
                         and visual_stats["split_sha256"] == split["split_sha256"]
                         and visual_stats["train_episode_indices"] == sorted(split["train_episode_indices"])
                         and visual_stats["train_record_count"] == DESIGN["train_unique_rows"],
                         "normalization source/split/unique train count differs")
            anchor, anchor_evidence = dependence.select_anchor(pack, split)
            registry = pack.manifest["task_registry"]
            with (out / "training_trace.jsonl").open("x", encoding="utf-8") as train_log, \
                 (out / "evaluation_trace.jsonl").open("x", encoding="utf-8") as eval_log:
                for seed in base.SEEDS:
                    runs[str(seed)] = {}
                    for arm in base.ARMS:
                        runs[str(seed)][arm] = {}
                        initial_parameter = None
                        for variant in VARIANTS:
                            print(f"normalization seed={seed} arm={arm} variant={variant} budget=200", flush=True)
                            model = build_model(seed, registry, variant, visual_stats)
                            init_sha = named_parameter_sha(model)
                            if initial_parameter is None:
                                initial_parameter = init_sha
                            base.require(init_sha == initial_parameter == smoke_report["results"][str(seed)][arm][variant]["initial_named_parameter_sha256"],
                                         "paired/smoke initialization differs")
                            initial = evaluate(model, datasets["validation"], stats, seed=seed, arm=arm, variant=variant,
                                phase="step0", anchor=anchor, references=initial_refs, trace=eval_log)
                            buffers_before = base.fingerprints(dict(model.named_buffers()))
                            # Wrapper adds only reporting metadata; the old update function remains unchanged.
                            class LabelTrace:
                                def write(self, line):
                                    row = json.loads(line); row["variant"] = variant
                                    train_log.write(base.canonical(row) + "\n")
                                def flush(self):
                                    train_log.flush()
                            optimizer, timing = previous.train_arm(model, datasets["train"], draws[seed], arm, LabelTrace())
                            updates += timing["optimizer_steps"]
                            base.require(base.fingerprints(dict(model.named_buffers())) == buffers_before,
                                         "normalization buffers changed during training")
                            model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
                            if variant == "baseline":
                                base.require(previous.checkpoint_parameter_sha(model) == old["runs"][str(seed)][arm]["checkpoint"]["envelope"]["binding"]["final_parameter_sha256"],
                                             "fresh baseline final parameters failed old replay")
                            binding = dict(seed=seed, arm=arm, variant=variant, step=200, plan_sha256=plan_sha,
                                smoke_report_sha256=smoke_sha, normalization_sha256=smoke_report["normalization_sha256"],
                                normalization_applied=variant != "baseline", draw_sha256=step0["sampling"][str(seed)]["sha256"],
                                initial_named_parameter_sha256=init_sha, final_named_parameter_sha256=named_parameter_sha(model),
                                final_state_sha256=base.parameter_hash(model))
                            reloaded, checkpoint = save_reload(out / f"seed{seed}_{arm}_{variant}_final200.pt", model, optimizer,
                                registry=registry, visual_stats=visual_stats, binding=binding)
                            final = evaluate(reloaded, datasets["validation"], stats, seed=seed, arm=arm, variant=variant,
                                phase="final200", anchor=anchor, references=final_refs, trace=eval_log)
                            if variant == "baseline":
                                previous.numeric_equal(final["conditions"]["clean"]["episode_macro"],
                                    old["runs"][str(seed)][arm]["final"]["validation"]["episode_macro"])
                            row = dict(initial=initial, training=timing, final=final, checkpoint=checkpoint)
                            runs[str(seed)][arm][variant] = row
                            base.write_json(out / f"seed{seed}_{arm}_{variant}_result.json", row)
                            del optimizer, model, reloaded
        base.require(updates == DESIGN["total_optimizer_steps"], "incomplete optimizer budget")
        previous.rehash(root, preflight["input_sha256"])
        finish(out, dict(schema="libero_visual_normalization_result_v1", status="completed_fixed_budget_normalization_comparison",
            plan_sha256=plan_sha, smoke_report_sha256=smoke_sha, normalization_sha256=smoke_report["normalization_sha256"],
            input_sha256=preflight["input_sha256"], runs=runs, summary=summarize(runs), anchor=anchor_evidence,
            optimizer_steps=updates, backward_calls=updates, evaluation_forwards=sum(v[p]["model_forwards"] for r in runs.values()
                for a in r.values() for v in a.values() for p in ("initial", "final")),
            old_baseline_replayed_bit_exact=True, old_inputs_unchanged=True, runtime=previous.runtime_evidence(),
            wall_seconds=time.monotonic()-began, feature_encodings=0, rollout_steps=0, formal_training_ready=False))
    except BaseException as exc:
        base.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
            completed_arm_optimizer_steps=updates, partial_current_arm_may_have_steps=True, preserve_no_retry=True))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "smoke", "train"), required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--smoke-report-sha256")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.stage == "preflight":
        base.require(not args.execute and args.smoke_report_sha256 is None, "preflight has no execution flags")
        evidence = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=evidence["status"], input_pins=len(evidence["input_sha256"]))))
    elif args.stage == "smoke":
        base.require(args.smoke_report_sha256 is None, "smoke takes no existing smoke SHA")
        smoke(ROOT, args.plan_sha256, args.execute)
    else:
        train(ROOT, args.plan_sha256, args.smoke_report_sha256, args.execute)


if __name__ == "__main__":
    main()
