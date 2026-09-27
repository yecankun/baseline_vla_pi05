"""Frozen clean-input activation/scale inspection, with no training or intervention."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import run_pi05_libero_visual_dependence as prior
from pi05_libero_activation_inspection import MomentTotals, TanhTotals, capture_activations
from pi05_libero_action_ablation import action_ablation_inputs
from pi05_libero_world_model import collate_window_inputs
from pi05_libero_world_model_objectives import collate_window_targets
from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, read_json, sha256_file

base, previous, frozen = prior.base, prior.previous, prior.frozen
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/libero-activation-inspection-plan-v1.json"
OUT = "simulation_output/pi05_libero_activation_inspection_v1"
PRIOR_PLAN_SHA = "b81ed7f66ee10625d66a9e34e582283690930833c91e0f2c7bdb77905f6d7aea"
PRIOR_REPORT_SHA = "1849406c3a434b3dd43ed1d7859c0d401f69022acd4f9f1200a8af3951b25e41"
PHASES = ("step0", "final200")
PARTITION_ROWS = {"train": 1134, "validation": 524}
ALLOW = dict(clean_frozen_activation_inspection=True, optimizer_updates=False, backward=False,
             input_intervention=False, normalization_refit=False, feature_extraction=False,
             architecture_or_loss_change=False, policy_rollout=False, project_data_inspection=False)
NEW_CODE = {"tools/run_pi05_libero_activation_inspection.py", "tools/pi05_libero_activation_inspection.py"}
TANH_PAIRS = {"view0": ("view0_pre", "view0_post"), "view1": ("view1_pre", "view1_post"),
              "history": ("history_pre", "history_post"), "context_task": ("context_task_pre", "initial_post"),
              "action": ("action_pre", "action_post")}


def validate(repo, plan_sha):
    plan = read_json(base.checked(repo, PLAN, plan_sha))
    base.require(plan["schema"] == "libero_activation_inspection_plan_v1" and plan["allowed"] == ALLOW
                 and plan["phases"] == list(PHASES) and plan["seeds"] == list(base.SEEDS)
                 and plan["arms"] == list(base.ARMS) and plan["validation_windows"] == 500
                 and plan["model_forwards"] == 396 and plan["output"] == OUT
                 and plan["partition_unique_rows"] == PARTITION_ROWS
                 and plan["previous_plan_sha256"] == PRIOR_PLAN_SHA
                 and plan["previous_report_sha256"] == PRIOR_REPORT_SHA
                 and set(plan["input_sha256"]) == NEW_CODE, "fixed activation scope/authorization differs")
    old = prior.validate(repo, PRIOR_PLAN_SHA)
    report = read_json(base.checked(repo, prior.OUT + "/report.json", PRIOR_REPORT_SHA))
    base.require(report["status"] == "completed_frozen_visual_dependence"
                 and report["input_files_unchanged"] and report["optimizer_steps"] == 0
                 and read_json(repo / prior.OUT / "status.json") == dict(status="completed", report_sha256=PRIOR_REPORT_SHA, optimizer_steps=0)
                 and not (repo / prior.OUT / "failure.json").exists(), "prior complete reference differs")
    pins = {**old["input_sha256"], **plan["input_sha256"], PLAN: plan_sha,
            prior.OUT + "/report.json": PRIOR_REPORT_SHA,
            **{prior.OUT + "/" + name: sha for name, sha in report["output_sha256"].items()}}
    frozen.rehash(repo, pins)
    return dict(status="passed_activation_inspection_preflight", plan=plan, plan_sha256=plan_sha,
                input_sha256=pins, optimizer_updates=0)


def affine_parts(layer, actual_concat, groups, actual_pre):
    """Descriptive affine components in float64, never causal attribution shares."""
    base.require(isinstance(layer, torch.nn.Linear) and layer.weight.device.type == "cpu"
                 and layer.weight.dtype == torch.float32 and not layer.weight.requires_grad
                 and not layer.training, "frozen CPU linear required")
    base.require(actual_concat.dtype == torch.float32 and actual_pre.dtype == torch.float32
                 and actual_concat.device.type == actual_pre.device.type == "cpu"
                 and actual_concat.shape[-1] == layer.in_features
                 and actual_pre.shape == (*actual_concat.shape[:-1], layer.out_features)
                 and bool(torch.isfinite(actual_concat).all()) and bool(torch.isfinite(actual_pre).all()),
                 "actual finite affine input/output shapes differ")
    base.require(isinstance(groups, dict) and "bias" not in groups and bool(groups), "named input groups required")
    columns = []
    for name, (start, stop) in groups.items():
        base.require(type(name) is str and type(start) is type(stop) is int and 0 <= start < stop <= layer.in_features,
                     "invalid affine group")
        columns.extend(range(start, stop))
    base.require(sorted(columns) == list(range(layer.in_features)), "groups must cover each column exactly once")
    with torch.no_grad():
        x, weight = actual_concat.double(), layer.weight.double()
        bias = layer.bias.double() if layer.bias is not None else torch.zeros(layer.out_features, dtype=torch.float64)
        parts = {name: F.linear(x[..., start:stop], weight[:, start:stop]).clone()
                 for name, (start, stop) in groups.items()}
        parts["bias"] = bias.expand_as(actual_pre).clone()
        full = F.linear(x, weight, bias)
        reconstructed = torch.stack(list(parts.values())).sum(0)
        error = float((reconstructed - full).abs().max())
        base.require(error <= 1e-10 * (1 + float(full.abs().max())), "float64 block decomposition differs")
        return dict(parts=parts, max_abs_float64_reconstruction_error=error,
                    max_abs_float32_vs_float64_error=float((actual_pre.double() - full).abs().max()))


def cache_scales(pack, split):
    result = {}
    for partition, count in PARTITION_ROWS.items():
        ids = split[partition + "_episode_indices"]
        indices = np.concatenate([pack.indices[e] for e in ids])
        base.require(len(indices) == len(np.unique(indices)) == count, "unique complete raw rows differ")
        values = np.asarray(pack.arrays["visual_latent"][indices])
        base.require(values.dtype == np.float32 and values.shape == (count, 2, 2048)
                     and np.isfinite(values).all() and bool(pack.arrays["visual_valid"][indices].all()),
                     "native raw feature rows/masks differ")
        views = {}
        for view in range(2):
            actual = torch.from_numpy(values[:, view].copy())
            accumulator = MomentTotals(feature_width=2048)
            accumulator.update(actual)
            views[str(view)] = accumulator.summary()
        result[partition] = dict(episode_indices=ids, unique_rows=count, views=views,
                                  input_sha256=base.tensor_hash(torch.from_numpy(values.copy())))
    return result


class SummarySet:
    def __init__(self):
        self.moments, self.tanh = {}, {key: TanhTotals() for key in TANH_PAIRS}

    def update(self, tensors, inputs, decompositions):
        extra = dict(raw_view0=inputs["history_visual_latent"][:, :, 0], raw_view1=inputs["history_visual_latent"][:, :, 1],
                     normalized_state=inputs["history_state"], normalized_action=inputs["candidate_actions"])
        extra.update({prefix + "/" + name: value for prefix, parts in decompositions.items() for name, value in parts["parts"].items()})
        for name, tensor in {**tensors, **extra}.items():
            if name not in self.moments:
                self.moments[name] = MomentTotals(feature_width=tensor.shape[-1])
            self.moments[name].update(tensor)
        for name, (pre, post) in TANH_PAIRS.items():
            actual_post = tensors[post][:, 0] if name == "context_task" else tensors[post]
            self.tanh[name].update(tensors[pre], actual_post)

    def summary(self):
        return dict(moments={k: v.summary() for k, v in self.moments.items()},
                    tanh={k: v.summary() for k, v in self.tanh.items()})


def scalar_summary(summary):
    result = {}
    for name, row in summary["moments"].items():
        for metric in ("mean", "mean_abs", "rms", "std", "coordinate_mean_rms", "coordinate_std_rms"):
            result["moments/" + name + "/" + metric] = row[metric]
    for name, row in summary["tanh"].items():
        result["tanh/" + name + "/mean_slope"] = row["mean_slope"]
        result["tanh/" + name + "/exact_1_fraction"] = row["post_abs_exact_1_fraction"]
        for threshold, data in row["post_abs_ge"].items():
            result[f"tanh/{name}/abs_ge_{threshold}_fraction"] = data["fraction"]
        for threshold, data in row["local_slope_le"].items():
            result[f"tanh/{name}/slope_le_{threshold}_fraction"] = data["fraction"]
    return result


def macros(episodes):
    values = [scalar_summary(row) for row in episodes.values()]
    base.require(bool(values) and all(set(v) == set(values[0]) for v in values), "complete matching episode summaries required")
    return {k: statistics.mean(v[k] for v in values) for k in values[0]}


def inspect_model(model, dataset, *, phase, seed, arm, references, trace=None):
    base.require(phase in PHASES and arm in base.ARMS, "fixed phase/arm required")
    before = base.parameter_hash(model)
    grouped = {}
    for i, window in enumerate(dataset.windows):
        grouped.setdefault(window.episode_index, []).append(i)
    total, episodes = SummarySet(), {}
    counts, batches, began = 0, 0, time.monotonic()
    recon = {name: dict(max_abs_float64_reconstruction_error=0., max_abs_float32_vs_float64_error=0.)
             for name in ("history_fusion", "context_task_fusion")}
    for eid, indices in grouped.items():
        subtotal = SummarySet()
        for start in range(0, len(indices), 16):
            selected = indices[start:start+16]
            items = [dataset[i] for i in selected]
            raw = collate_window_inputs([x["inputs"] for x in items])
            targets = collate_window_targets([x["targets"] for x in items])
            source_fp, target_fp = base.fingerprints(raw), base.fingerprints(targets)
            inputs = action_ablation_inputs(raw, arm)
            input_fp = base.fingerprints(inputs)
            base.require(tuple(selected) in references, "missing saved batch reference")
            ref = references[tuple(selected)]
            ri = ref["arm_input_sha256"][arm] if phase == "step0" else ref["input_sha256"]
            rt = ref["common_target_sha256"] if phase == "step0" else ref["target_sha256"]
            rp = ref["predictions_sha256"][arm] if phase == "step0" else ref["prediction_sha256"]
            base.require(input_fp == ri and target_fp == rt, "original input/target reference differs")
            captured = capture_activations(model, inputs)
            base.require(base.fingerprints(captured["predictions"]) == rp, "saved prediction replay differs")
            tensors, h = captured["tensors"], model.config.hidden_dim
            decompositions = {
                "history_fusion": affine_parts(model.history_projection, tensors["history_concat"],
                    dict(view0=(0,h), view1=(h,2*h), visual_mask=(2*h,2*h+2),
                         state=(2*h+2,2*h+10), state_mask=(2*h+10,2*h+18)), tensors["history_pre"]),
                "context_task_fusion": affine_parts(model.context_task_projection, tensors["context_task_concat"],
                    dict(context=(0,h), task=(h,2*h)), tensors["context_task_pre"])}
            batch = SummarySet()
            for accumulator in (total, subtotal, batch):
                accumulator.update(tensors, inputs, decompositions)
            for name, value in decompositions.items():
                for key in recon[name]:
                    recon[name][key] = max(recon[name][key], value[key])
            base.require(base.fingerprints(raw) == source_fp and base.fingerprints(targets) == target_fp
                         and base.fingerprints(inputs) == input_fp, "inspection mutated inputs/targets")
            if trace is not None:
                trace.write(base.canonical(dict(seed=seed, arm=arm, phase=phase, episode_index=eid,
                    window_indices=selected, input_sha256=input_fp, target_sha256=target_fp,
                    prediction_sha256=rp, activation_sha256=base.fingerprints(tensors), summary=batch.summary(),
                    affine_reconstruction={name: {k: value[k] for k in recon[name]} for name, value in decompositions.items()})) + "\n")
            counts += len(selected)
            batches += 1
        episodes[str(eid)] = subtotal.summary()
    base.require(counts == len(dataset) and base.parameter_hash(model) == before, "incomplete inspection or changed weights")
    return dict(per_episode=episodes, micro=total.summary(), episode_macro=macros(episodes),
                windows=counts, batches=batches, affine_reconstruction_max=recon, seconds=time.monotonic()-began)


def parameter_scales(model):
    result = {}
    for name, value in model.named_parameters():
        accum = MomentTotals()
        accum.update(value.detach().reshape(1, -1))
        result[name] = accum.summary()
    return result


def summarize(runs):
    base.require(set(runs) == set(map(str,base.SEEDS)) and all(set(v) == set(base.ARMS) for v in runs.values()), "six models required")
    base.require(all(set(phases) == set(PHASES) for pair in runs.values() for phases in pair.values()), "exact two phases required")
    metric_keys = set(runs[str(base.SEEDS[0])][base.ARMS[0]][PHASES[0]]["episode_macro"])
    base.require(bool(metric_keys) and all(set(row["episode_macro"]) == metric_keys for pair in runs.values()
                                         for phases in pair.values() for row in phases.values()), "matching macro metric keys required")
    result, differences = {}, {}
    for phase in PHASES:
        result[phase] = {}
        for arm in base.ARMS:
            rows = [runs[str(s)][arm][phase]["episode_macro"] for s in base.SEEDS]
            result[phase][arm] = {k: dict(mean=statistics.mean(r[k] for r in rows),
                descriptive_population_std=statistics.pstdev(r[k] for r in rows)) for k in rows[0]}
    for arm in base.ARMS:
        differences[arm] = {}
        for key in result["step0"][arm]:
            values = {str(s): runs[str(s)][arm]["final200"]["episode_macro"][key] - runs[str(s)][arm]["step0"]["episode_macro"][key] for s in base.SEEDS}
            differences[arm][key] = dict(by_seed=values, mean=statistics.mean(values.values()),
                descriptive_population_std=statistics.pstdev(values.values()))
    return dict(seed_summary=result, paired_final_minus_step0=differences)


def run(repo, plan_sha, execute):
    base.require(execute is True and not (repo / OUT).exists(), "explicit fresh inspection required; no overwrite/retry")
    preflight = validate(repo, plan_sha)
    previous.runtime_settings()
    out = frozen.claim(repo, OUT, preflight)
    started = time.monotonic()
    try:
        train = read_json(repo / frozen.TRAIN_REPORT)
        step0 = read_json(repo / base.OUT_PATH / "report.json")
        initial_refs = [json.loads(s) for s in (repo / base.OUT_PATH / "evaluation_trace.jsonl").read_text().splitlines()]
        final_refs = [json.loads(s) for s in (repo / previous.TRAIN_ROOT / "evaluation_final.jsonl").read_text().splitlines()]
        runs, checks = {}, {}
        with LiberoFeaturePack(repo / base.PACK_PATH) as pack:
            split = pack.load_split(repo / base.PACK_PATH / "split.json")
            scales = cache_scales(pack, split)
            stats = read_json(repo / base.PACK_PATH / "preparation/normalization.json")
            dataset = LiberoWindowDataset(pack, split, partition="validation", normalization=stats)
            base.require(len(dataset) == 500 and {e: len(pack.indices[e]) for e in frozen.VALIDATION_COUNTS} == frozen.VALIDATION_COUNTS,
                         "original complete validation coverage required")
            with (out / "activation_trace.jsonl").open("x", encoding="utf-8") as trace:
                for seed in base.SEEDS:
                    pair = base.initialized_pair(seed, pack.manifest["task_registry"])
                    runs[str(seed)], checks[str(seed)] = {}, {}
                    for arm, model in pair.items():
                        runs[str(seed)][arm], checks[str(seed)][arm] = {}, {}
                        for phase in PHASES:
                            checkpoint_sha = None
                            if phase == "final200":
                                record = train["runs"][str(seed)][arm]["checkpoint"]
                                envelope, binding = record["envelope"], deepcopy(record["envelope"]["binding"])
                                base.require(binding["seed"] == seed and binding["arm"] == arm
                                             and binding["training_authorization_sha256"] == train["training_authorization_sha256"], "original checkpoint binding differs")
                                metadata = deepcopy(envelope["legacy_metadata"])
                                metadata["authorization_sha256"] = train["training_authorization_sha256"]
                                prior.checkpoint_io.load_final_checkpoint(repo / previous.TRAIN_ROOT / record["path"], model,
                                    envelope, binding, expected_metadata=metadata)
                                checkpoint_sha = envelope["checkpoint_sha256"]
                            before = base.parameter_hash(model)
                            if phase == "step0":
                                base.require(before == step0["runs"][str(seed)]["validation"]["parameter_sha256"][arm],
                                             "original step0 parameter bytes differ")
                            records = initial_refs if phase == "step0" else final_refs
                            refs = {tuple(r["window_indices"]): r for r in records if r["seed"] == seed and r["partition"] == "validation"
                                    and (phase == "step0" or r["arm"] == arm)}
                            base.require(len(refs) == 33, "all33 original clean batches required")
                            print(f"ACTIVATION_INSPECTION seed={seed} arm={arm} phase={phase}; updates=0", flush=True)
                            result = inspect_model(model, dataset, phase=phase, seed=seed, arm=arm, references=refs, trace=trace)
                            after = base.parameter_hash(model)
                            base.require(after == before, "inspection changed parameters")
                            runs[str(seed)][arm][phase] = result
                            checks[str(seed)][arm][phase] = dict(parameter_sha256_before=before, parameter_sha256_after=after,
                                checkpoint_sha256=checkpoint_sha, parameter_scales=parameter_scales(model))
        frozen.rehash(repo, preflight["input_sha256"])
        forwards = sum(r["batches"] for pair in runs.values() for phases in pair.values() for r in phases.values())
        base.require(forwards == 396 and not torch.cuda.is_initialized(), "fixed CPU forward budget differs")
        report = dict(schema="libero_activation_inspection_result_v1", status="completed_frozen_activation_inspection",
            plan_sha256=plan_sha, prior_report_sha256=PRIOR_REPORT_SHA, input_sha256=preflight["input_sha256"],
            input_files_unchanged=True, raw_unique_cache_scales=scales, runs=runs, parameter_checks=checks, **summarize(runs),
            optimizer_updates=0, backward_calls=0, feature_encodings=0, simulation_steps=0, model_forwards=forwards,
            original_step0_and_final_prediction_hashes_replayed=True, input_interventions=0,
            normalization_refitted=False, architecture_changed=False, training_started=False,
            robustness_certified=False, policy_performance_evaluated=False, real_system_validated=False,
            formal_data_allowed=False, training_ready=False, limitations=preflight["plan"]["limitations"],
            runtime=previous.runtime_evidence(), wall_seconds=time.monotonic()-started,
            output_sha256={p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
        base.write_json(out / "report.json", report)
        base.write_json(out / "status.json", dict(status="completed", report_sha256=sha256_file(out / "report.json"), optimizer_updates=0))
        print(base.canonical(dict(status=report["status"], report_sha256=sha256_file(out / "report.json"), wall_seconds=report["wall_seconds"], optimizer_updates=0)), flush=True)
        return report
    except BaseException as error:
        base.write_json(out / "failure.json", dict(status="failed_preserve_no_retry", error=str(error), traceback=traceback.format_exc(), optimizer_updates=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "inspect"), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    base.require(args.execute == (args.stage == "inspect"), "execution flag required only for inspection")
    if args.stage == "preflight":
        value = validate(ROOT, args.plan_sha256)
        print(base.canonical(dict(status=value["status"], input_pins=len(value["input_sha256"]), optimizer_updates=0)))
    else:
        run(ROOT, args.plan_sha256, args.execute)


if __name__ == "__main__":
    main()
