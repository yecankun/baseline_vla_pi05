"""Projection-output scale-control interface/numerical gate; no training stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

import run_pi05_libero_normalization_diagnosis as diagnosis
import run_pi05_libero_visual_normalization as previous
from pi05_libero_projection_scale_control import ProjectionScaledLiberoWorldModel, smooth_projection_scale, FORMULA, SCHEMA
from pi05_libero_world_model_adapter import LiberoFeaturePack, read_json

base = previous.base
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-projection-scale-control-smoke-plan-v1.json"
OUT = "simulation_output/pi05_libero_projection_scale_control_smoke_v1"
DIAG_PLAN_SHA = "9c2e938a9655c89116b0479c4d659d3b984c05e934208fe601bd48619aa1ec37"
DIAG_REPORT_SHA = "a840d53abfbaf8574cdb53bb2e9ce567984ecd9f85cf68677e06f6e5b3282427"
NORMALIZATION_SMOKE_SHA = "ffbf8903ba2cd48c2a7a0646b8fe38f2b73e471974df0d0fb2ff9c8d0a55cf56"
VARIANTS = ("input_normalized", "input_normalized_projection_scaled")
CODE = {"tools/pi05_libero_projection_scale_control.py", "tools/test_pi05_libero_projection_scale_control.py",
        "tools/run_pi05_libero_projection_scale_control.py", "tools/test_run_pi05_libero_projection_scale_control.py"}
DESIGN = dict(seeds=list(base.SEEDS), arms=list(base.ARMS), variants=list(VARIANTS),
    batch_source="original_shared_train_draw_step1", batch_size=16, train_windows=1086,
    model_initialization="same_original_seed_step0_named_parameters", gradient_checks=12,
    model_forwards=24, backward_calls=12, optimizer_updates=0, optimizer_constructed=False,
    validation_forwards=0, normalization_refit=False, checkpoint_loads=0, checkpoint_saves=0,
    feature_encodings=0, rollout_steps=0, original_loss_unchanged=True, added_trainable_parameters=0,
    formula=FORMULA, projection_scale_compute="float64_intermediate_float32_output",
    rms_rounding_tolerance=1e-6, scalar_saturation_elimination_claim=False,
    model_quality_comparison=False, training_execution_allowed=False)


def validate(root, plan_sha):
    plan = read_json(base.checked(root, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_projection_scale_control_smoke_plan_v1" and plan["design"] == DESIGN
        and plan["prior_report_sha256"] == DIAG_REPORT_SHA and plan["output"] == OUT
        and set(plan["input_sha256"]) == CODE, "fixed interface-only scale-control protocol differs")
    prior, _ = diagnosis.validate(root, DIAG_PLAN_SHA)
    diag = read_json(base.checked(root, diagnosis.OUT + "/report.json", DIAG_REPORT_SHA))
    base.require(diag["status"] == "completed_readonly_normalization_diagnosis"
        and read_json(root / diagnosis.OUT / "status.json") == dict(status="completed", report_sha256=DIAG_REPORT_SHA, optimizer_steps=0)
        and not (root / diagnosis.OUT / "failure.json").exists(), "completed diagnostic evidence required")
    old_smoke, visual_stats, norm_pins = previous.checked_smoke(root,
        "18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5", NORMALIZATION_SMOKE_SHA)
    pins = {**prior["input_sha256"], **diag["input_sha256"], **norm_pins,
        diagnosis.OUT + "/report.json": DIAG_REPORT_SHA,
        **{diagnosis.OUT + "/" + name: digest for name, digest in diag["output_sha256"].items()},
        **plan["input_sha256"], PLAN: plan_sha}
    previous.previous.rehash(root, pins)
    return dict(status="passed_scale_control_smoke_preflight", plan_sha256=plan_sha,
                plan=plan, input_sha256=pins), old_smoke, visual_stats


def build_model(seed, registry, variant, visual_stats, config=None):
    base.require(variant in VARIANTS, "unknown scale-control variant")
    if variant == VARIANTS[0]:
        return previous.build_model(seed, registry, "train_visual_normalized", visual_stats, config)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ProjectionScaledLiberoWorldModel(config or previous.LiberoWorldModelConfig(), registry, visual_stats)
    return model.cpu().eval().requires_grad_(False)


def capture_views(model, inputs):
    """Capture actual effective pre-tanh values; recompute raw affine for identity checks."""
    hooks, before = previous._hook_state(model), base.parameter_hash(model)
    saved, handles = {}, []
    scaled = isinstance(model, ProjectionScaledLiberoWorldModel)
    def view_hook(view):
        def hook(layer, args, output):
            base.require(view not in saved, "projection executed more than once")
            raw = F.linear(args[0], layer.weight, layer.bias)
            expected = smooth_projection_scale(raw) if scaled else raw
            base.require(torch.equal(output, expected), "actual projection differs from fixed affine/scale formula")
            saved[view] = dict(projected_input=args[0].detach().clone(), raw_affine=raw.detach().clone(), pre=output.detach().clone())
        return hook
    def history_hook(_module, args, _output):
        base.require("history" not in saved, "history projection executed more than once")
        saved["history"] = args[0].detach().clone()
    try:
        for view, layer in enumerate(model.view_projections): handles.append(layer.register_forward_hook(view_hook(view)))
        handles.append(model.history_projection.register_forward_hook(history_hook))
        captured = previous.capture_prediction(model, inputs)
        base.require(set(saved) == {0, 1, "history"}, "complete two-view capture required")
        views = {}
        for view in range(2):
            row = {k: v.clone() for k, v in saved[view].items()}
            row["post"] = saved["history"][..., view*model.config.hidden_dim:(view+1)*model.config.hidden_dim].clone()
            expected_input = torch.where(inputs["history_visual_valid"][..., view, None],
                (torch.where(inputs["history_visual_valid"][..., view, None], inputs["history_visual_latent"][..., view, :], 0.)
                 - model.visual_mean[view])/model.visual_scale[view], 0.)
            base.require(torch.equal(row["projected_input"], expected_input), "train input normalization or remask changed")
            base.require(torch.equal(torch.tanh(row["pre"]), row["post"]), "actual original downstream tanh changed")
            if scaled:
                base.require(float(row["pre"].double().square().mean(-1).max()) <= (1+1e-6)**2, "effective vector RMS bound failed")
            views[str(view)] = row
        return captured, views
    finally:
        for handle in handles: handle.remove()
        base.require(previous._hook_state(model) == hooks and base.parameter_hash(model) == before,
                     "capture changed model state or leaked hooks")


def activation_summary(views):
    totals = previous.ViewTotals(); totals.update(views)
    result = totals.summary()
    for name, tensors in views.items():
        result[name]["vector_rms"] = {}
        for field in ("raw_affine", "pre"):
            rms = tensors[field].double().square().mean(-1).sqrt()
            result[name]["vector_rms"][field] = dict(count=rms.numel(), min=float(rms.min()), mean=float(rms.mean()), max=float(rms.max()))
        result[name]["raw_affine_sha256"] = base.tensor_hash(tensors["raw_affine"])
        result[name]["projection_input_sha256"] = base.tensor_hash(tensors["projected_input"])
    return result


def check_pair(models, raw, targets, seed, arm, indices, reference):
    inputs = previous.action_ablation_inputs(raw, arm)
    fp, tfp = base.fingerprints(inputs), base.fingerprints(targets)
    base.require(fp == reference["input_sha256"] and tfp == reference["target_sha256"]
                 and indices == reference["window_indices"], "original shared training batch differs")
    base.require(all(bool(inputs[k].all()) for k in ("history_visual_valid", "history_state_valid"))
        and all(bool(targets[k].all()) for k in ("future_visual_valid", "state_target_valid")), "original complete mask support required")
    params = {v: previous.named_parameter_sha(m) for v, m in models.items()}
    base.require(set(models) == set(VARIANTS) and len(set(params.values())) == 1
        and params[VARIANTS[0]] == reference["initial_named_parameter_sha256"], "paired fresh parameters differ")
    base.require(all(a.data_ptr() != b.data_ptr() for a,b in zip(models[VARIANTS[0]].parameters(), models[VARIANTS[1]].parameters())),
                 "paired models share parameter storage")
    before = {v: base.parameter_hash(m) for v,m in models.items()}
    results, baseline_views = {}, None
    for variant in VARIANTS:
        model = models[variant]
        captured, views = capture_views(model, inputs)
        if variant == VARIANTS[0]:
            baseline_views = views
            base.require(base.fingerprints(captured["predictions"]) == reference["prediction_sha256"], "old input-normalized step0 predictions differ")
            old_totals = previous.ViewTotals(); old_totals.update(views)
            base.require(old_totals.summary() == reference["activation"], "old step0 activation did not replay")
        else:
            base.require(all(torch.equal(views[v][key], baseline_views[v][key]) for v in ("0","1")
                         for key in ("projected_input", "raw_affine")), "pair differs before output scale control")
        model.train().requires_grad_(True)
        try:
            gradient = previous.previous.training.gradient_probe(model, inputs, targets, arm)
        finally:
            model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
        if variant == VARIANTS[0]:
            base.require(gradient == reference["gradient"], "old input-normalized no-update gradients did not replay")
        base.require(base.parameter_hash(model) == before[variant], "no-update check changed model state")
        results[variant] = dict(metadata=model.metadata(), named_parameter_sha256=params[variant],
            state_sha256=before[variant], trainable_parameter_tensors=len(list(model.parameters())),
            parameter_elements=sum(p.numel() for p in model.parameters()),
            prediction_sha256=base.fingerprints(captured["predictions"]), activation=activation_summary(views), gradient=gradient)
    base.require(base.fingerprints(inputs) == fp and base.fingerprints(targets) == tfp, "smoke mutated input/target bytes")
    return dict(seed=seed, arm=arm, window_indices=indices, input_sha256=fp, target_sha256=tfp,
                models=results, model_forwards=4, backward_calls=2, optimizer_steps=0,
                old_normalized_reference_replayed=True, parameter_storage_disjoint=True)


def smoke(root, plan_sha, execute):
    base.require(execute is True, "explicit zero-update smoke flag required")
    preflight, old_smoke, visual_stats = validate(root, plan_sha)
    previous.previous.runtime_settings()
    out = previous.claim(root, OUT, preflight)
    began = time.monotonic()
    try:
        results = {}
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            datasets, _, draws, _, _ = previous.previous.load_inputs(root, pack)
            base.require(len(datasets["train"]) == 1086, "original train partition differs")
            for seed in base.SEEDS:
                results[str(seed)] = {}
                raw, targets, indices = previous.previous.draw_batch(draws[seed], datasets["train"], 1)
                for arm in base.ARMS:
                    print(f"zero-update scale-control smoke seed={seed} arm={arm}", flush=True)
                    models = {v: build_model(seed, pack.manifest["task_registry"], v, visual_stats) for v in VARIANTS}
                    ref = old_smoke["results"][str(seed)][arm]["train_visual_normalized"]
                    results[str(seed)][arm] = check_pair(models, raw, targets, seed, arm, indices, ref)
        previous.previous.rehash(root, preflight["input_sha256"])
        previous.finish(out, dict(schema="libero_projection_scale_control_smoke_result_v1",
            status="passed_projection_scale_control_zero_update_gate", plan_sha256=plan_sha,
            input_sha256=preflight["input_sha256"], normalization_sha256=old_smoke["normalization_sha256"],
            model_schema=SCHEMA, results=results, optimizer_steps=0, optimizer_constructed=False,
            backward_calls=12, model_forwards=24, validation_forwards=0, checkpoint_loads=0, checkpoint_saves=0,
            normalization_refit=False, feature_encodings=0, rollout_steps=0, input_files_unchanged=True,
            runtime=previous.previous.runtime_evidence(), wall_seconds=time.monotonic()-began,
            decision="interface_and_numerics_only_training_comparison_not_started",
            limitations=["fresh initial parameters and first fixed train batch only, not a quality evaluation",
                "vector RMS control does not guarantee no saturated coordinates or nonvanishing gradients",
                "magnitude information is compressed, not proven beneficial",
                "no trained scale-control checkpoint or promotion to original checkpoint schema",
                "no policy robustness, capacity, architecture innovation, formal or real-system claim"]))
    except BaseException as exc:
        base.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
                        optimizer_steps=0, preserve_no_retry=True))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "smoke"), required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.stage == "preflight":
        base.require(not args.execute, "preflight cannot execute smoke")
        preflight, _, _ = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=preflight["status"], input_pins=len(preflight["input_sha256"]))))
    else: smoke(ROOT, args.plan_sha256, args.execute)


if __name__ == "__main__": main()
