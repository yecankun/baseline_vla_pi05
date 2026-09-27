"""Calibrate and run one predeclared, train-only matched Real10 comparison.

No hardware; original checkpoints are read-only. Both arms resume atomically
from a completed paired step. The fixed 30-context panel is NOT validation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_real10_pi05_joint_temporal import (
    Real10TemporalView, check, context, flow_loss, grad_summary, predict,
)
from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_joint_temporal_world import (
    JointTemporalConfig, JointTemporalWorld, TemporalPrefixInjection, visual_target,
)
from probe_pi05_lerobot_adapter import set_seed
from real10_pi05_policy import Real10PI05Policy


VERSION = "real10_joint_prefix_matched_v1"
ARMS = ("action_only", "joint")


def panel_indices(data):
    panel = []
    episodes = sorted({r["source"]["episode"] for r in data.rows})
    for episode in episodes:
        available = sorted(
            (i for i, r in enumerate(data.rows)
             if r["source"]["episode"] == episode and r["source"]["step"] >= 2),
            key=lambda i: data.rows[i]["source"]["step"],
        )
        hold = [i for i in available if data.intents[i] == 1]
        feed = [i for i in available if data.intents[i] == 2]
        check(len(hold) >= 3 and feed, f"insufficient calibration contexts in {episode}")
        chosen = [hold[int(.25 * (len(hold) - 1))], hold[int(.75 * (len(hold) - 1))],
                  feed[int(.50 * (len(feed) - 1))]]
        check(len(set(chosen)) == 3, "calibration contexts must be distinct")
        panel.extend(chosen)
    return panel


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max()), "mean": float(a.mean())}


def future_errors(model, prefix, case):
    prediction = model.predict_future(prefix, case["visual"][:, -1], case["world_action"])
    return (prediction.float() - visual_target(case["future_visual"]).detach()).square().mean((0, 2))


def noise_for(seed, ordinal, device):
    generator = torch.Generator(device=device).manual_seed(seed + 100_000 + ordinal)
    return torch.randn(1, 1, 32, generator=generator, device=device)


def calibrate(base, model, injection, cases, panel, data, seed):
    increments = torch.stack([
        (visual_target(cases[i]["future_visual"]) - visual_target(cases[i]["visual"][:, -1]))
        .square().mean((0, 2)) for i in panel
    ])
    scales = increments.mean(0).clamp_min(1e-6)
    shared = list(model.temporal.parameters())
    rows = []
    for ordinal, index in enumerate(panel):
        case = cases[index]
        prefix = context(model, case)
        errors = future_errors(model, prefix, case)
        raw, scaled = errors.mean(), (errors / scales).mean()
        with injection.condition(prefix):
            action = flow_loss(base, case, noise_for(seed, ordinal, prefix.device))
            ga = torch.autograd.grad(action, shared, retain_graph=True)
        gr = torch.autograd.grad(raw, shared, retain_graph=True)
        gs = torch.autograd.grad(scaled, shared)
        a, r, s = grad_summary(ga), grad_summary(gr), grad_summary(gs)
        check(all(x["finite"] and x["l2"] > 0 for x in (a, r, s)), "invalid calibration gradient")
        dot = sum(float((x.detach().float() * y.detach().float()).sum()) for x, y in zip(ga, gs))
        row = {"pack_index": index, "sample_id": data.rows[index]["sample_id"],
               "episode": data.rows[index]["source"]["episode"], "task": data.rows[index]["task"],
               "piper_target": int(data.intents[index]), "flow_time": .5,
               "action_loss": float(action.detach()), "future_raw_loss": float(raw.detach()),
               "future_scaled_loss": float(scaled.detach()),
               "action_grad_l2": a["l2"], "future_raw_grad_l2": r["l2"],
               "future_scaled_grad_l2": s["l2"],
               "old_weighted_ratio": .1 * r["l2"] / a["l2"],
               "scaled_unweighted_ratio": s["l2"] / a["l2"],
               "gradient_cosine": dot / (a["l2"] * s["l2"]),
               "persistence_mse_per_view": increments[ordinal].cpu().tolist()}
        rows.append(row)
    weight = .1 / float(np.median([r["scaled_unweighted_ratio"] for r in rows]))
    for row in rows:
        row["new_weighted_ratio"] = weight * row["scaled_unweighted_ratio"]
    return {"schema": VERSION, "contexts": rows, "views": ["side", "top"],
            "future_scale_per_view": scales.cpu().tolist(), "scale_floor": 1e-6,
            "persistence_mse_per_view": increments.mean(0).cpu().tolist(),
            "lambda": weight, "target_initial_median_ratio": .1,
            "rule": "0.1 / median(scaled future gradient L2 / action gradient L2)",
            "gradient_ratio_before": distribution([r["old_weighted_ratio"] for r in rows]),
            "gradient_ratio_after": distribution([r["new_weighted_ratio"] for r in rows]),
            "gradient_cosine": distribution([r["gradient_cosine"] for r in rows]),
            "training_diagnostic_only": True}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_pair(path, models, optimizers, initial, calibration, protocol, logs):
    payload = {"version": VERSION, "step": len(logs), "protocol": protocol,
               "initial_state": initial, "calibration": calibration, "logs": logs,
               "models": {k: cpu_state(v) for k, v in models.items()},
               "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
               "deployable": False}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate(base, models, initial_model, injection, cases, panel, data, calibration, seed):
    variants = {"original_pi05": None, "initialized_adapter": initial_model, **models}
    rows = []
    device = base.action_std.device
    scales = torch.tensor(calibration["future_scale_per_view"], device=device)
    with torch.no_grad():
        for ordinal, index in enumerate(panel):
            case = cases[index]
            noise = noise_for(seed, ordinal, device)
            target = data.actions[index, :3].astype(float)
            row = {"pack_index": index, "sample_id": data.rows[index]["sample_id"],
                   "episode": data.rows[index]["source"]["episode"], "task": data.rows[index]["task"],
                   "target_translation_mm": target.tolist(), "piper_target": int(data.intents[index]),
                   "next_record_dt_s": data.rows[index]["timing"]["observation_dt_s"],
                   "variants": {}}
            for name, model in variants.items():
                prefix = None if model is None else context(model, case)
                _, packet = predict(base, case, injection, prefix, noise)
                with injection.condition(prefix):
                    action = flow_loss(base, case, noise)
                error = np.abs(np.asarray(packet["elite_tcp_delta_6d"][:3]) - target)
                entry = {"prediction": packet, "translation_abs_error_xyz_mm": error.tolist(),
                         "translation_mae_mm": float(error.mean()), "action_flow_loss": float(action)}
                if model is not None:
                    future = future_errors(model, prefix, case)
                    entry.update(future_mse_per_view=future.cpu().tolist(),
                                 future_scaled_loss=float((future / scales).mean()))
                row["variants"][name] = entry
            reference = row["variants"]["original_pi05"]["prediction"]
            check(all(v["prediction"]["piper_probabilities"] == reference["piper_probabilities"]
                      for v in row["variants"].values()), "frozen Piper outputs changed")
            row["persistence_mse_per_view"] = (
                visual_target(case["future_visual"]) - visual_target(case["visual"][:, -1])
            ).square().mean((0, 2)).cpu().tolist()
            rows.append(row)
    metrics = {}
    for name in variants:
        entries = [r["variants"][name] for r in rows]
        by_episode = {ep: float(np.mean([r["variants"][name]["translation_mae_mm"] for r in rows
                                        if r["episode"] == ep])) for ep in sorted({r["episode"] for r in rows})}
        item = {"translation_mae_mm": float(np.mean([e["translation_mae_mm"] for e in entries])),
                "translation_mae_xyz_mm": np.mean([e["translation_abs_error_xyz_mm"] for e in entries], 0).tolist(),
                "episode_macro_mae_mm": float(np.mean(list(by_episode.values()))),
                "per_episode_mae_mm": by_episode,
                "task_mae_mm": {t: float(np.mean([r["variants"][name]["translation_mae_mm"] for r in rows
                                                  if r["task"] == t])) for t in ("left", "right")},
                "action_flow_loss": float(np.mean([e["action_flow_loss"] for e in entries]))}
        if name != "original_pi05":
            future = np.mean([e["future_mse_per_view"] for e in entries], 0)
            persistence = np.mean([r["persistence_mse_per_view"] for r in rows], 0)
            item.update(future_mse_per_view=future.tolist(), future_mse=float(future.mean()),
                        future_to_persistence_ratio=float(future.mean() / persistence.mean()))
        metrics[name] = item
    comparisons = {}
    for metric in ("translation_mae_mm", "episode_macro_mae_mm", "action_flow_loss", "future_mse"):
        a, b = metrics["action_only"][metric], metrics["joint"][metric]
        comparisons[metric] = {"action_only": a, "joint": b, "joint_minus_action_only": b - a,
                               "relative_change_percent": 100 * (b - a) / a if a else None,
                               "lower_is_better": True}
    return {"rows": rows, "metrics": metrics, "paired_comparison": comparisons,
            "persistence_mse_per_view": np.mean([r["persistence_mse_per_view"] for r in rows], 0).tolist(),
            "piper_probabilities_exactly_unchanged": True, "heldout": False,
            "panel": "2 hold + 1 feed per training episode; enriched, NOT prevalence-weighted"}


def run(args):
    started = time.perf_counter()
    data = Real10TemporalView(args.pack)
    audit = data.audit_index()
    panel = panel_indices(data)
    order = np.random.default_rng(args.seed).permutation(len(data.rows))[:args.steps].tolist()
    check(len(order) == args.steps and args.steps > 0, "short budget must be within one full-index permutation")
    protocol = {"version": VERSION, "pack": str(args.pack.resolve()),
                "training_root": str(args.training_root.resolve()), "seed": args.seed,
                "steps_per_arm": args.steps, "batch_size": 1, "learning_rate": 1e-4, "weight_decay": 0.,
                "temporal_clip_l2": 1., "world_clip_l2": 1., "target_initial_gradient_ratio": .1,
                "calibration_flow_time": .5, "panel_indices": panel,
                "panel_sample_ids": [data.rows[i]["sample_id"] for i in panel],
                "training_order": order, "training_sample_ids": [data.rows[i]["sample_id"] for i in order],
                "index_audit": audit, "heldout": False, "hardware_executed": False,
                "design": "docs/algorithm-real10-joint-prefix-protocol-20260921.md"}
    if args.resume:
        check(not (args.out / "report.json").exists(), "completed comparison must not be rerun")
        check(read_json(args.out / "protocol.json") == protocol, "resume protocol mismatch")
        saved = torch.load(args.out / "paired_checkpoint.pt", weights_only=True, map_location="cpu")
        check(saved["version"] == VERSION and saved["protocol"] == protocol, "checkpoint protocol mismatch")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "protocol.json", protocol)
        saved = None
    set_seed(args.seed)
    base = Real10PI05Policy(args.training_root / "elite/final_policy.pt",
                            args.training_root / "piper/mixed_head_policy.pt", seed=args.seed, device=args.device)
    print(f"Frozen base loaded in {base.metadata['load_seconds']:.2f}s", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cases = {i: data.encode_case(base, i) for i in panel}
    set_seed(args.seed)  # Initialization independent of model-loading / feature-extraction RNG.
    dimension = cases[panel[0]]["visual"].shape[-1]
    initial_model = JointTemporalWorld(JointTemporalConfig(visual_dim=dimension, prefix_dim=dimension)).to(args.device).eval()
    initial = cpu_state(initial_model) if saved is None else saved["initial_state"]
    initial_model.load_state_dict(initial, strict=True)
    injection = TemporalPrefixInjection(base.policy)
    try:
        if saved is None:
            calibration = calibrate(base, initial_model, injection, cases, panel, data, args.seed)
            write_json(args.out / "calibration.json", calibration)
        else:
            calibration = saved["calibration"]
        models = {name: copy.deepcopy(initial_model) for name in ARMS}
        optimizers = {name: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
                      for name, model in models.items()}
        for a, b in zip(models["action_only"].parameters(), models["joint"].parameters()):
            check(torch.equal(a, b) and a.data_ptr() != b.data_ptr(), "arms must start equal with independent storage")
        logs = []
        if saved is not None:
            for name in ARMS:
                models[name].load_state_dict(saved["models"][name], strict=True)
                optimizers[name].load_state_dict(saved["optimizers"][name])
            logs = saved["logs"]
            check(saved["step"] == len(logs), "incomplete paired checkpoint")
        else:
            save_pair(args.out / "paired_checkpoint.pt", models, optimizers, initial, calibration, protocol, logs)
        print(json.dumps({"phase": "calibrated", "lambda": calibration["lambda"],
                          "ratio_before": calibration["gradient_ratio_before"],
                          "ratio_after": calibration["gradient_ratio_after"],
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if args.calibrate_only:
            print("Calibration saved; no updates. Resume with --resume and the same parameters.", flush=True)
            return
        scales = torch.tensor(calibration["future_scale_per_view"], device=args.device)
        train_started = time.perf_counter()
        for step in range(len(logs), args.steps):
            index = order[step]
            if index not in cases:
                cases[index] = data.encode_case(base, index)
            case = cases[index]
            # Stateless per-step seed permits exact paired-step recovery.
            set_seed(args.seed + 200_000 + step)
            noise = base.policy.model.sample_noise(case["action_target"].shape, args.device)
            flow_time = base.policy.model.sample_time(1, args.device)
            log = {"step": step + 1, "sample_id": data.rows[index]["sample_id"],
                   "pack_index": index, "flow_time": float(flow_time[0]), "arms": {}}
            for name in ARMS:
                model, optimizer = models[name], optimizers[name]
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prefix = context(model, case)
                with injection.condition(prefix):
                    action_loss = flow_loss(base, case, noise.clone(), flow_time=flow_time.clone())
                    if name == "joint":
                        future = future_errors(model, prefix, case)
                        scaled = (future / scales).mean()
                        total = action_loss + calibration["lambda"] * scaled
                    else:
                        future, scaled, total = None, None, action_loss
                    total.backward()
                temporal_norm = torch.nn.utils.clip_grad_norm_(model.temporal.parameters(), 1., error_if_nonfinite=True)
                world_norm = torch.nn.utils.clip_grad_norm_(model.world.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                entry = {"action_loss": float(action_loss.detach()), "total_loss": float(total.detach()),
                         "temporal_gradient_l2_before_clip": float(temporal_norm),
                         "world_gradient_l2_before_clip": float(world_norm),
                         "seconds": time.perf_counter() - tick}
                if future is not None:
                    entry.update(future_raw_loss=float(future.detach().mean()), future_scaled_loss=float(scaled.detach()))
                log["arms"][name] = entry
            if step == 0:
                check(log["arms"]["action_only"]["action_loss"] == log["arms"]["joint"]["action_loss"],
                      "first paired forward differs despite matched initialization and draws")
            logs.append(log)
            if (step + 1) % 25 == 0 or step + 1 == args.steps:
                save_pair(args.out / "paired_checkpoint.pt", models, optimizers, initial, calibration, protocol, logs)
                print(json.dumps({"phase": "training", "step": step + 1, "arms": log["arms"],
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
        train_seconds = time.perf_counter() - train_started
        check(all(not p.requires_grad and p.grad is None for p in base.policy.parameters()), "base policy was modified")
        check(all(not p.requires_grad and p.grad is None for p in base.head.parameters()), "Piper head was modified")
        for k, v in models["action_only"].state_dict().items():
            if k.startswith("world."):
                torch.testing.assert_close(v.cpu(), initial[k], rtol=0, atol=0)
        evaluation = evaluate(base, models, initial_model, injection, cases, panel, data, calibration, args.seed)
        check(injection.active_prefix is None and base.policy.model._project2026_state_context is None,
              "conditioning context leaked")
        # One direct disabled-hook comparison complements the architectural freeze.
        case = cases[panel[0]]
        noise = noise_for(args.seed, 0, args.device)
        disabled = predict(base, case, injection, None, noise)[0]
        injection.close()
        torch.testing.assert_close(disabled, predict(base, case, None, None, noise)[0], rtol=0, atol=0)
        for name, model in models.items():
            torch.save({"model_state": cpu_state(model), "metadata": model.metadata(), "arm": name,
                        "protocol": protocol, "calibration": calibration, "steps": args.steps,
                        "deployable": False}, args.out / f"{name}_adapter.pt")
        write_json(args.out / "training_log.json", logs)
        write_json(args.out / "panel_predictions.json", evaluation.pop("rows"))
        coverage = {ep: sum(data.rows[i]["source"]["episode"] == ep for i in order)
                    for ep in sorted({r["source"]["episode"] for r in data.rows})}
        report = {"schema": VERSION, "status": "completed", "scope": "train-only matched short optimization",
                  "protocol": protocol, "calibration": calibration, "evaluation": evaluation,
                  "base_checkpoint": base.metadata, "adapter": initial_model.metadata(),
                  "trainable_parameters": sum(p.numel() for p in initial_model.parameters()),
                  "training_coverage_per_episode": coverage,
                  "training_intent_counts": {str(k): sum(int(data.intents[i]) == k for i in order) for k in (1, 2)},
                  "panel_training_order_overlap": len(set(panel) & set(order)),
                  "checks": {"equal_initial_values_separate_storage": True, "matched_order_noise_time_steps": True,
                             "world_action_only_unchanged": True, "base_and_piper_frozen": True,
                             "disabled_hook_exact": True, "conditioning_cleared": True},
                  "timing": {"this_invocation_seconds": time.perf_counter() - started,
                             "this_invocation_training_seconds": train_seconds,
                             "all_logged_update_seconds": {name: sum(r["arms"][name]["seconds"] for r in logs) for name in ARMS}},
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                  "hardware_executed": False, "heldout_evaluation": False, "deployable": False,
                  "visual_status": "not_viewed"}
        write_json(args.out / "report.json", report)
        print(json.dumps({"status": report["status"], "paired_comparison": evaluation["paired_comparison"],
                          "timing": report["timing"], "checks": report["checks"]}, indent=2), flush=True)
    finally:
        injection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--training-root", type=Path, default=Path("simulation_output/real10_pi05_train_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_joint_comparison_v1"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if (args.out / "protocol.json").exists() and not (args.out / "report.json").exists():
            failure = args.out / f"failure_{time.time_ns()}.json"
            write_json(failure, {"error": repr(error), "hardware_executed": False})
        raise


if __name__ == "__main__":
    main()
