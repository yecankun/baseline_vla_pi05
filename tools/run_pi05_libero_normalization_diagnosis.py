"""Read-only endpoint/trace/error diagnosis; no optimization or model intervention."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import run_pi05_libero_visual_normalization as previous
import audit_pi05_libero_visual_normalization as prior_audit
from pi05_libero_normalization_trace_diagnosis import summarize_training_trace, summarize_saved_errors
from pi05_libero_projection_geometry import projection_geometry
from pi05_libero_world_model import LiberoWorldModelConfig, collate_window_inputs
from pi05_libero_world_model_adapter import LiberoFeaturePack, read_json, sha256_file
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_action_ablation import action_ablation_inputs

base = previous.base
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-normalization-diagnosis-plan-v1.json"
OUT = "simulation_output/pi05_libero_normalization_diagnosis_v1"
PRIOR_SHA = "a06f6957a6575502782e646eb76e5b69f75d8a4d57c1ab9ad9277008c5817520"
AUDIT_SHA = "a1764475325244a135013e27c9d3af5b5d2c6224f2b18725bfe1ee75f8127166"
EDGES = (0.01, 0.05, 0.1, 0.5, 1.0)
FIELDS = ("baseline_abs_sum", "normalized_abs_sum", "baseline_squared_sum", "normalized_squared_sum",
          "baseline_smooth_l1_sum", "normalized_smooth_l1_sum")
CODE = {"tools/run_pi05_libero_normalization_diagnosis.py", "tools/test_run_pi05_libero_normalization_diagnosis.py",
        "tools/pi05_libero_normalization_trace_diagnosis.py", "tools/test_pi05_libero_normalization_trace_diagnosis.py",
        "tools/pi05_libero_projection_geometry.py", "tools/test_pi05_libero_projection_geometry.py"}
SCOPE = dict(optimizer_updates=0, backward_calls=0, feature_encodings=0, rollout_steps=0,
    model_forwards=396, validation_windows=500, unique_train_rows=1134, unique_validation_rows=524,
    input_interventions=False, model_weight_interventions=False, retraining=False,
    normalization_refit=False, diagnostic_centered_moments=True, covariance_matrix_or_svd=False,
    projection_geometry_float64_pairs=48, baseline_absolute_error_bin_edges=list(EDGES),
    step_intervals=[[1, 20], [21, 50], [51, 100], [101, 150], [151, 200]])


def require_close(x, y, name):
    base.require(math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12), name)


class ErrorBuckets:
    """Baseline-conditioned shared scalar bins, not a new evaluation weighting."""
    def __init__(self):
        self.count = np.zeros(6, dtype=np.int64)
        self.sums = {k: np.zeros(6, dtype=np.float64) for k in FIELDS}
        self.improved = np.zeros(6, dtype=np.int64)
        self.worsened = np.zeros(6, dtype=np.int64)
        self.windows = self.batches = 0

    def update(self, baseline_pred, normalized_pred, target):
        shape = baseline_pred.shape
        base.require(baseline_pred.dtype == normalized_pred.dtype == target.dtype == torch.float32
                     and baseline_pred.device.type == normalized_pred.device.type == target.device.type == "cpu"
                     and normalized_pred.shape == target.shape == shape and len(shape) == 4
                     and shape[0] > 0 and shape[1:3] == (3, 2), "finite native visual tensors required")
        for value in (baseline_pred, normalized_pred, target):
            base.require(not value.requires_grad and bool(torch.isfinite(value).all()), "detached finite visual values required")
        eb = (baseline_pred.double() - target.double()).abs().numpy().reshape(-1)
        en = (normalized_pred.double() - target.double()).abs().numpy().reshape(-1)
        lb = F.smooth_l1_loss(baseline_pred, target, beta=1., reduction="none").double().numpy().reshape(-1)
        ln = F.smooth_l1_loss(normalized_pred, target, beta=1., reduction="none").double().numpy().reshape(-1)
        groups = np.searchsorted(np.asarray(EDGES), eb, side="right")
        self.count += np.bincount(groups, minlength=6)
        self.improved += np.bincount(groups, weights=(en < eb).astype(np.int64), minlength=6).astype(np.int64)
        self.worsened += np.bincount(groups, weights=(en > eb).astype(np.int64), minlength=6).astype(np.int64)
        for name, values in zip(FIELDS, (eb, en, eb*eb, en*en, lb, ln)):
            self.sums[name] += np.bincount(groups, weights=values, minlength=6)
        self.windows += shape[0]; self.batches += 1

    def summary(self):
        n = int(self.count.sum())
        base.require(n > 0, "empty bucket accumulator")
        bins = []
        for i, count in enumerate(self.count):
            row = dict(lower=0. if i == 0 else EDGES[i-1], upper=EDGES[i] if i < 5 else None,
                       interval="lower_inclusive_upper_exclusive", count=int(count),
                       improved_count=int(self.improved[i]), worsened_count=int(self.worsened[i]),
                       unchanged_count=int(count-self.improved[i]-self.worsened[i]))
            row.update({k: float(v[i]) for k, v in self.sums.items()})
            for prefix in ("baseline", "normalized"):
                row[prefix+"_mae"] = row[prefix+"_abs_sum"] / count if count else None
                row[prefix+"_smooth_l1"] = row[prefix+"_smooth_l1_sum"] / count if count else None
            row["mae_delta_contribution_to_full_support"] = (row["normalized_abs_sum"]-row["baseline_abs_sum"])/n
            row["smooth_l1_delta_contribution_to_full_support"] = (row["normalized_smooth_l1_sum"]-row["baseline_smooth_l1_sum"])/n
            bins.append(row)
        return dict(count=n, windows=self.windows, batches=self.batches, bins=bins,
            totals={k: float(v.sum()) for k, v in self.sums.items()},
            semantics="shared bins selected by same-seed/arm baseline error; conditional regression-to-mean caveat; no denoising claim")


def bucket_macro(episodes):
    base.require(bool(episodes), "episodes required")
    return [{key: statistics.mean(row["bins"][i][key] for row in episodes.values())
             for key in ("mae_delta_contribution_to_full_support", "smooth_l1_delta_contribution_to_full_support")}
            for i in range(6)]


def validate(root, plan_sha):
    plan = read_json(base.checked(root, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_normalization_diagnosis_plan_v1" and plan["scope"] == SCOPE
                 and plan["prior_report_sha256"] == PRIOR_SHA and plan["prior_audit_sha256"] == AUDIT_SHA
                 and plan["seeds"] == list(base.SEEDS) and plan["arms"] == list(base.ARMS)
                 and plan["variants"] == list(previous.VARIANTS) and plan["output"] == OUT
                 and set(plan["input_sha256"]) == CODE, "fixed read-only diagnosis scope differs")
    base.checked(root, "tools/audit_pi05_libero_visual_normalization.py", AUDIT_SHA)
    report = read_json(base.checked(root, previous.OUT + "/report.json", PRIOR_SHA))
    audited = prior_audit.audit(root, previous.OUT + "/report.json", PRIOR_SHA)
    pins = {**report["input_sha256"], **plan["input_sha256"], PLAN: plan_sha,
        "tools/audit_pi05_libero_visual_normalization.py": AUDIT_SHA,
        previous.OUT + "/report.json": PRIOR_SHA,
        **{previous.OUT + "/" + name: sha for name, sha in report["output_sha256"].items()}}
    previous.previous.rehash(root, pins)
    return dict(status="passed_readonly_diagnosis_preflight", plan_sha256=plan_sha,
                plan=plan, input_sha256=pins, previous_audit=audited), report


def load_frozen(root, report, seed, arm, variant, visual_stats, registry):
    record = report["runs"][str(seed)][arm][variant]["checkpoint"]
    binding = record["binding"]
    base.require(binding["seed"] == seed and binding["arm"] == arm and binding["variant"] == variant
                 and binding["step"] == 200 and binding["plan_sha256"] == report["plan_sha256"]
                 and binding["normalization_sha256"] == report["normalization_sha256"]
                 and binding["normalization_applied"] is (variant != "baseline"), "checkpoint binding differs")
    path = base.checked(root, previous.OUT + "/" + record["path"], record["sha256"])
    data = path.read_bytes()
    base.require(hashlib.sha256(data).hexdigest() == record["sha256"], "checkpoint bytes changed before load")
    loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    base.require(set(loaded) == {"schema", "binding", "config", "registry", "model_state", "normalization",
                               "optimizer_state_saved", "resumable"}
                 and loaded["schema"] == "libero_visual_normalization_final_checkpoint_v1"
                 and loaded["binding"] == binding and loaded["config"] == asdict(LiberoWorldModelConfig())
                 and loaded["registry"] == registry and loaded["optimizer_state_saved"] is False
                 and loaded["resumable"] is False
                 and loaded["normalization"] == (visual_stats if variant != "baseline" else None),
                 "frozen checkpoint schema/config/normalization differs")
    model = previous.build_model(seed, registry, variant, visual_stats)
    base.require(previous.named_parameter_sha(model) == binding["initial_named_parameter_sha256"], "initial parameter bytes differ")
    initial = {name: value.detach().numpy().copy() for name, value in model.named_parameters() if name.startswith("view_projections.")}
    expected = model.state_dict()
    base.require(set(loaded["model_state"]) == set(expected), "checkpoint state keys differ")
    for name, value in loaded["model_state"].items():
        base.require(isinstance(value, torch.Tensor) and value.dtype == expected[name].dtype and value.shape == expected[name].shape
                     and value.device.type == "cpu" and value.layout == torch.strided and not value.requires_grad
                     and bool(torch.isfinite(value).all()), "invalid checkpoint tensor")
    model.load_state_dict(loaded["model_state"], strict=True)
    base.require(base.parameter_hash(model) == binding["final_state_sha256"]
                 and previous.named_parameter_sha(model) == binding["final_named_parameter_sha256"], "final checkpoint state differs")
    return model, initial


def geometry(model, initial, pack, split):
    result = {}
    for part, count in (("train", 1134), ("validation", 524)):
        eids = sorted(split[part + "_episode_indices"])
        rows = np.concatenate([pack.indices[e] for e in eids])
        raw = np.array(pack.arrays["visual_latent"][rows], copy=True)
        base.require(len(rows) == len(np.unique(rows)) == count and raw.shape == (count, 2, 2048)
                     and raw.dtype == np.float32 and bool(pack.arrays["visual_valid"][rows].all()), "unique full raw rows differ")
        x = raw
        if isinstance(model, previous.NormalizedLiberoWorldModel):
            # Same float32 subtraction/division as the model, before float64 diagnostics.
            x = ((torch.from_numpy(raw) - model.visual_mean) / model.visual_scale).numpy()
        views = {}
        for v in range(2):
            layer = model.view_projections[v]
            views[str(v)] = projection_geometry(x[:, v], initial[f"view_projections.{v}.weight"],
                initial[f"view_projections.{v}.bias"], layer.weight.detach().numpy(), layer.bias.detach().numpy())
        result[part] = dict(episode_indices=eids, unique_rows=count, views=views,
            raw_array_sha256=base.tensor_hash(torch.from_numpy(raw)),
            actual_projection_input_sha256=base.tensor_hash(torch.from_numpy(x)),
            normalization_refit=False, statistics_used_to_change_inputs=False)
    return result


def diagnose_errors(models, dataset, stats, seed, arm, references, trace):
    grouped = {}
    for i, window in enumerate(dataset.windows): grouped.setdefault(window.episode_index, []).append(i)
    all_buckets, episodes = ErrorBuckets(), {}
    summaries = {v: {} for v in previous.VARIANTS}
    before = {v: base.parameter_hash(model) for v, model in models.items()}
    forwards = 0
    for eid, indices in grouped.items():
        buckets = ErrorBuckets()
        scores = {v: base.Totals(stats) for v in previous.VARIANTS}
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[i] for i in selected]
            raw = collate_window_inputs([x["inputs"] for x in items])
            targets = collate_window_targets([x["targets"] for x in items])
            base.require(bool(targets["future_visual_valid"].all()), "shared scalar bins require all visual targets valid")
            base.require(bool(raw["history_visual_valid"].all()) and bool(raw["history_state_valid"].all())
                         and bool(targets["state_target_valid"].all()), "all original history/state support must be valid")
            inputs = action_ablation_inputs(raw, arm)
            fp, tfp = base.fingerprints(inputs), base.fingerprints(targets)
            predictions = {}
            for variant, model in models.items():
                ref = references[(seed, arm, variant, tuple(selected))]
                base.require(ref["input_sha256"] == fp and ref["target_sha256"] == tfp, "clean inputs/targets differ")
                with torch.inference_mode(): pred = model(**inputs)
                predictions[variant] = {k: v.detach().clone() for k, v in pred.items()}
                base.require(base.fingerprints(pred) == ref["prediction_sha256"], "frozen clean predictions did not replay")
                scores[variant].update(predictions[variant], targets)
                forwards += 1
            batch = ErrorBuckets()
            for accumulator in (all_buckets, buckets, batch):
                accumulator.update(predictions["baseline"]["pred_future_visual_latent"][:, 0],
                    predictions["train_visual_normalized"]["pred_future_visual_latent"][:, 0], targets["future_visual_latent"])
            base.require(base.fingerprints(inputs) == fp and base.fingerprints(targets) == tfp, "prediction changed inputs/targets")
            trace.write(base.canonical(dict(seed=seed, arm=arm, episode_index=eid, window_indices=selected,
                input_sha256=fp, target_sha256=tfp, prediction_sha256={v: base.fingerprints(pred) for v, pred in predictions.items()},
                buckets=batch.summary())) + "\n")
        episodes[str(eid)] = buckets.summary()
        for variant in previous.VARIANTS:
            summaries[variant][str(eid)] = scores[variant].summary()
            prefix = "baseline" if variant == "baseline" else "normalized"
            sums = episodes[str(eid)]["totals"]; score = summaries[variant][str(eid)]
            require_close(sums[prefix+"_abs_sum"], score["visual"]["aggregate"]["absolute_error_sum"], "bucket MAE sum differs")
            require_close(sums[prefix+"_smooth_l1_sum"], score["masked_objective"]["visual_sum"], "bucket loss sum differs")
    base.require({v: base.parameter_hash(m) for v, m in models.items()} == before, "frozen model state changed")
    base.require(all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters()), "gradients unexpectedly enabled")
    return dict(per_episode=episodes, micro=all_buckets.summary(), episode_macro_contributions=bucket_macro(episodes),
                scored_per_episode=summaries, model_forwards=forwards)


def run(root, plan_sha, execute):
    base.require(execute is True, "explicit read-only diagnosis flag required")
    preflight, report = validate(root, plan_sha)
    previous.previous.runtime_settings()
    out = previous.claim(root, OUT, preflight)
    began = time.monotonic()
    try:
        rows = [json.loads(x) for x in (root / previous.OUT / "training_trace.jsonl").read_text().splitlines()]
        trace_summary = summarize_training_trace(rows)
        saved_errors = summarize_saved_errors(report)
        norm = read_json(base.checked(root, previous.SMOKE + "/visual_normalization.json", report["normalization_sha256"]))
        refs = {(x["seed"], x["arm"], x["variant"], tuple(x["window_indices"])): x for x in
                (json.loads(line) for line in (root / previous.OUT / "evaluation_trace.jsonl").read_text().splitlines())
                if x["phase"] == "final200" and x["condition"] == "clean"}
        base.require(len(refs) == 396, "complete original clean references required")
        geometries, errors = {}, {}
        with LiberoFeaturePack(root / base.PACK_PATH) as pack:
            split = pack.load_split(root / base.PACK_PATH / "split.json")
            stats = read_json(root / base.PACK_PATH / "preparation/normalization.json")
            dataset = previous.previous.LiberoWindowDataset(pack, split, partition="validation", normalization=stats)
            base.require(len(dataset) == 500, "fixed full validation set required")
            with (out / "error_trace.jsonl").open("x", encoding="utf-8") as trace:
                for seed in base.SEEDS:
                    geometries[str(seed)], errors[str(seed)] = {}, {}
                    for arm in base.ARMS:
                        print(f"frozen diagnosis seed={seed} arm={arm}", flush=True)
                        models = {}; geometries[str(seed)][arm] = {}
                        for variant in previous.VARIANTS:
                            model, initial = load_frozen(root, report, seed, arm, variant, norm, pack.manifest["task_registry"])
                            geometries[str(seed)][arm][variant] = geometry(model, initial, pack, split)
                            models[variant] = model
                        error = diagnose_errors(models, dataset, stats, seed, arm, refs, trace)
                        for variant in previous.VARIANTS:
                            base.require(error["scored_per_episode"][variant] == report["runs"][str(seed)][arm][variant]["final"]["conditions"]["clean"]["per_episode"],
                                         "replayed original per-episode score differs")
                        errors[str(seed)][arm] = error
        previous.previous.rehash(root, preflight["input_sha256"])
        forwards = sum(r["model_forwards"] for s in errors.values() for r in s.values())
        base.require(forwards == 396, "exact frozen forward budget required")
        previous.finish(out, dict(schema="libero_normalization_diagnosis_result_v1", status="completed_readonly_normalization_diagnosis",
            plan_sha256=plan_sha, prior_report_sha256=PRIOR_SHA, input_sha256=preflight["input_sha256"],
            training_trace_analysis=trace_summary, saved_error_analysis=saved_errors, projection_geometry=geometries,
            visual_error_buckets=errors, optimizer_steps=0, backward_calls=0, model_forwards=forwards,
            float64_projection_geometry_pairs=48, normalization_refit=False, feature_encodings=0, rollout_steps=0,
            input_files_unchanged=True, frozen_predictions_replayed=True, runtime=previous.previous.runtime_evidence(),
            wall_seconds=time.monotonic()-began,
            limitations=["step0/final endpoints cannot locate onset or establish unique causation",
                "trace gradient/loss bins use different sampled batches; not fixed-set learning curves",
                "covariance decomposition is descriptive, no input or model intervention",
                "baseline-error conditioning permits regression-to-mean; no denoising claim",
                "reused four validation episodes in one task, not independent test or policy benefit"]))
    except BaseException as exc:
        base.write_json(out / "failure.json", dict(error=str(exc), traceback=traceback.format_exc(),
                        optimizer_steps=0, preserve_no_retry=True))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "diagnose"), required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.stage == "preflight":
        base.require(not args.execute, "preflight does not execute a diagnosis")
        preflight, _ = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=preflight["status"], input_pins=len(preflight["input_sha256"]))))
    else:
        run(ROOT, args.plan_sha256, args.execute)


if __name__ == "__main__":
    main()
