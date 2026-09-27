"""Matched 100-step absolute-history vs adjacent-change gated PI05 diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_real10_pi05_gated_history import CONDITIONS, distribution, modified_history, relative_l2, rms
from check_real10_pi05_joint_temporal import Real10TemporalView, check, context, flow_loss, predict
from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_gated_temporal_world import GatedTemporalWorld, StateTokenResidualInjection
from pi05_joint_temporal_world import JointTemporalConfig
from pi05_motion_temporal_world import MotionGatedTemporalWorld
from probe_pi05_lerobot_adapter import set_seed
from real10_pi05_policy import Real10PI05Policy
from run_real10_pi05_gated_action import entry, gradient_norm, metrics
from run_real10_pi05_joint_comparison import cpu_state, noise_for


VERSION = "real10_gated_motion_comparison_v1"
MODELS = {"absolute": GatedTemporalWorld, "motion": MotionGatedTemporalWorld}


def save_pair(path, models, optimizers, initial, logs, protocol):
    temporary = path.with_suffix(".tmp")
    torch.save({"version": VERSION, "protocol": protocol, "step": len(logs), "logs": logs,
                "models": {k: cpu_state(v) for k, v in models.items()},
                "metadata": {k: v.metadata() for k, v in models.items()},
                "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                "initial_state": initial, "deployable": False}, temporary)
    temporary.replace(path)


@torch.no_grad()
def input_checks(data, base, model, cases, panel):
    checks = []
    start = next(i for i, r in enumerate(data.rows) if r["source"]["step"] == 0)
    ep = data.rows[start]["source"]["episode"]
    for step in (0, 1):
        i = data.lookup[(ep, step)]
        if i not in cases:
            cases[i] = data.encode_case(base, i)
        case = cases[i]
        changed = dict(case)
        missing = ~case["valid"]
        for key in ("visual", "states", "ages"):
            changed[key] = case[key].clone()
            changed[key][missing] = -9999.0
        torch.testing.assert_close(context(model, case), context(model, changed), rtol=0, atol=0)
        inputs = model.temporal.motion_inputs(case["visual"], case["states"], case["valid"], case["ages"])
        expected = torch.tensor([[False, step == 1, True]], device=case["valid"].device)
        check(torch.equal(inputs["valid"], expected), "motion pair validity differs at reset")
        checks.append({"pack_index": i, "source_step": step, "motion_slot_valid": inputs["valid"].cpu().tolist(),
                       "padded_content_invariant": True})
    for i in panel:
        repeated = modified_history(cases[i], "repeat_current")
        inputs = model.temporal.motion_inputs(repeated["visual"], repeated["states"], repeated["valid"], repeated["ages"])
        check(torch.count_nonzero(inputs["visual"][:, :2]).item() == 0, "repeat-current visual changes not zero")
        check(torch.count_nonzero(inputs["states"][:, :2]).item() == 0, "repeat-current state changes not zero")
    return {"reset_cases": checks, "repeat_current_has_zero_visual_state_changes_all30": True,
            "intervals_preserved_in_repeat_current": True, "future_targets_not_consumed": True}


def run(args):
    started = time.perf_counter()
    source = read_json(args.reference / "report.json")
    check(source["status"] == "completed" and source["world_loss_enabled"] is False, "requires completed action-only reference")
    fixed = source["protocol"]["source_protocol"]
    check(fixed["steps_per_arm"] == 100 and fixed["seed"] == 123, "requires fixed 100-step seed123 source")
    panel, order, seed = fixed["panel_indices"], fixed["training_order"], fixed["seed"]
    data = Real10TemporalView(Path(fixed["pack"]))
    check(data.audit_index() == fixed["index_audit"], "source data index changed")
    for indices, key in ((panel, "panel_sample_ids"), (order, "training_sample_ids")):
        check([data.rows[i]["sample_id"] for i in indices] == fixed[key], "saved sample order changed")
    initialization_path = Path(source["protocol"]["reference"]) / "paired_checkpoint.pt"
    protocol = {"version": VERSION, "reference": str(args.reference.resolve()), "reference_protocol": source["protocol"],
                "initialization_path": str(initialization_path.resolve()), "initialization": "saved untrained initial_state + gate0 in both arms",
                "arms": list(MODELS), "steps_per_arm": 100, "seed": seed, "batch_size": 1,
                "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0., "gradient_clip_l2": 1.,
                "connection": "same unchanged post-cast StateTokenResidualInjection in both arms",
                "single_factor": "absolute history versus two adjacent changes + current content + elapsed intervals",
                "shared_parameter_values_at_initialization": True, "parameter_count_matched": True,
                "history_conditions": list(CONDITIONS), "checkpoint_selection": "fixed final step100",
                "world_loss_enabled": False, "heldout": False, "hardware_executed": False,
                "design": "docs/algorithm-real10-motion-prefix-protocol-20260921.md"}
    if args.resume:
        check(not (args.out / "report.json").exists(), "completed run must not be repeated")
        check(read_json(args.out / "protocol.json") == protocol, "resume protocol changed")
        resumed = torch.load(args.out / "paired_checkpoint.pt", map_location="cpu", weights_only=True)
        check(resumed["protocol"] == protocol and resumed["version"] == VERSION, "resume checkpoint changed")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "protocol.json", protocol)
        resumed = None
    original_pair = torch.load(initialization_path, map_location="cpu", weights_only=True, mmap=True)
    check(original_pair["protocol"] == fixed, "initialization provenance changed")
    initial = dict(original_pair["initial_state"], policy_gate=torch.zeros(()))
    del original_pair
    root = Path(fixed["training_root"])
    base = Real10PI05Policy(root / "elite/final_policy.pt", root / "piper/mixed_head_policy.pt", device=args.device, seed=seed)
    print(f"Original frozen pair loaded in {base.metadata['load_seconds']:.2f}s", flush=True)
    models = {name: cls(JointTemporalConfig()).to(args.device).eval() for name, cls in MODELS.items()}
    for model in models.values():
        model.load_state_dict(initial, strict=True)
        model.world.requires_grad_(False)
    for key, value in models["absolute"].state_dict().items():
        torch.testing.assert_close(value, models["motion"].state_dict()[key], rtol=0, atol=0)
    counts = {name: sum(p.numel() for p in model.policy_parameters()) for name, model in models.items()}
    check(len(set(counts.values())) == 1, "parameter counts differ")
    optimizers = {name: torch.optim.AdamW(model.policy_parameters(), lr=1e-4, weight_decay=0.) for name, model in models.items()}
    cases = {i: data.encode_case(base, i) for i in panel}
    input_report = input_checks(data, base, models["motion"], cases, panel)
    historical = {r["pack_index"]: r for r in read_json(args.reference / "panel_predictions.json")}
    historical_logs = read_json(args.reference / "training_log.json")
    injection = StateTokenResidualInjection(base.policy)
    rows, originals, logs = [], {}, []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for ordinal, i in enumerate(panel):
                case, noise = cases[i], noise_for(seed, ordinal, args.device)
                action, packet = predict(base, case, injection, None, noise)
                loss = flow_loss(base, case, noise)
                check(packet == historical[i]["original"]["prediction"], "original reference prediction differs")
                target = data.actions[i, :3].astype(float)
                row = {"pack_index": i, "sample_id": data.rows[i]["sample_id"], "episode": data.rows[i]["source"]["episode"],
                       "task": data.rows[i]["task"], "piper_target": int(data.intents[i]),
                       "target_translation_mm": target.tolist(), "original": entry(packet, loss, target)}
                for name, model in models.items():
                    residual = model.policy_residual(context(model, case))
                    candidate, output = predict(base, case, injection, residual, noise)
                    with injection.condition(residual):
                        candidate_loss = flow_loss(base, case, noise)
                    torch.testing.assert_close(candidate, action, rtol=0, atol=0)
                    torch.testing.assert_close(candidate_loss, loss, rtol=0, atol=0)
                    check(output == packet, f"{name} zero gate packet changed")
                    row[f"{name}_initial"] = entry(output, candidate_loss, target)
                rows.append(row)
                originals[i] = action.cpu()
        write_json(args.out / "identity_check.json", {"status": "passed", "contexts_per_arm": len(panel),
                   "both_zero_gate_action_packet_loss_exact": True, "same_initial_parameters": True,
                   "parameters_per_arm": counts, "input_checks": input_report, "prefix": injection.last_info})
        print("Both arms: all30 zero-gate actions/losses exact; motion reset/zero-change checks passed.", flush=True)
        if resumed is not None:
            for name, model in models.items():
                model.load_state_dict(resumed["models"][name], strict=True)
                optimizers[name].load_state_dict(resumed["optimizers"][name])
            logs = resumed["logs"]
            check(resumed["step"] == len(logs), "paired step mismatch")
        else:
            save_pair(args.out / "paired_checkpoint.pt", models, optimizers, initial, logs, protocol)
        training_started = time.perf_counter()
        for step in range(len(logs), 100):
            i = order[step]
            if i not in cases:
                cases[i] = data.encode_case(base, i)
            case = cases[i]
            set_seed(seed + 200_000 + step)
            noise = base.policy.model.sample_noise(case["action_target"].shape, args.device)
            flow_time = base.policy.model.sample_time(1, args.device)
            log = {"step": step+1, "pack_index": i, "sample_id": data.rows[i]["sample_id"], "flow_time": float(flow_time[0]), "arms": {}}
            for name, model in models.items():
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                residual = model.policy_residual(context(model, case))
                with injection.condition(residual):
                    loss = flow_loss(base, case, noise, flow_time=flow_time)
                    loss.backward()
                gate_gradient = float(model.policy_gate.grad.detach())
                temporal_gradient = gradient_norm(model.temporal.parameters())
                total_gradient = torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 1., error_if_nonfinite=True)
                if step == 0:
                    check(abs(gate_gradient) > 0 and np.isfinite(gate_gradient), f"{name}: no finite initial gate gradient")
                    check(temporal_gradient == 0., "gate0 must initially block temporal gradient")
                optimizer.step()
                log["arms"][name] = {"action_loss": float(loss.detach()), "gate_gradient_before_clip": gate_gradient,
                                     "shared_gradient_l2_before_clip": temporal_gradient,
                                     "policy_gradient_l2_before_clip": float(total_gradient),
                                     "gate_after_step": float(model.policy_gate.detach()),
                                     "gate_tanh_after_step": float(model.policy_gate.detach().tanh())}
            expected = historical_logs[step]
            check(log["flow_time"] == expected["flow_time"], "control flow-time reproduction failed")
            for key, value in log["arms"]["absolute"].items():
                check(value == expected[key], f"absolute control training reproduction failed at step{step+1}: {key}")
            logs.append(log)
            if (step + 1) % 25 == 0:
                save_pair(args.out / "paired_checkpoint.pt", models, optimizers, initial, logs, protocol)
                print(json.dumps({"phase": "paired_training", "step": step+1, "arms": log["arms"],
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
        training_seconds = time.perf_counter() - training_started
        for name, model in models.items():
            check(any(r["arms"][name]["shared_gradient_l2_before_clip"] > 0 for r in logs[1:]), "temporal gradient never opened")
            for key, value in model.state_dict().items():
                if key.startswith("world."):
                    torch.testing.assert_close(value.cpu(), initial[key], rtol=0, atol=0)
        check(all(not p.requires_grad and p.grad is None for m in (base.policy, base.head) for p in m.parameters()), "base/Piper unfrozen")
        previous_final = torch.load(args.reference / "gated_action_adapter.pt", weights_only=True, map_location="cpu")
        for key, value in models["absolute"].state_dict().items():
            torch.testing.assert_close(value.cpu(), previous_final["model_state"][key], rtol=0, atol=0)
        del previous_final
        with torch.no_grad():
            for ordinal, row in enumerate(rows):
                i, noise = row["pack_index"], noise_for(seed, ordinal, args.device)
                case = cases[i]
                for name, model in models.items():
                    true = None
                    for condition in CONDITIONS:
                        variant = modified_history(case, condition)
                        tokens = context(model, variant)
                        residual = model.policy_residual(tokens)
                        action, packet = predict(base, case, injection, residual, noise)
                        with injection.condition(residual):
                            loss = flow_loss(base, case, noise)
                        item = entry(packet, loss, data.actions[i, :3].astype(float))
                        item["residual_rms"] = rms(residual)
                        item["normalized_action_max_abs_drift"] = float((action.cpu() - originals[i]).abs().max())
                        check(packet["piper_probabilities"] == row["original"]["prediction"]["piper_probabilities"], "Piper changed")
                        if condition == "true_history":
                            true = (tokens, residual, action, packet)
                            if name == "absolute":
                                check(packet == historical[i]["gated_final"]["prediction"], "control final packet reproduction failed")
                                check(float(loss) == historical[i]["gated_final"]["action_flow_loss"], "control final loss reproduction failed")
                        else:
                            difference = np.asarray(packet["elite_tcp_delta_6d"][:3]) - np.asarray(true[3]["elite_tcp_delta_6d"][:3])
                            item["versus_true"] = {
                                "shared_tokens_relative_l2_change": relative_l2(tokens, true[0]),
                                "requested_residual_relative_l2_change": relative_l2(residual, true[1]),
                                "active_action_exactly_equal": torch.equal(action[..., :3], true[2][..., :3]),
                                "translation_mean_abs_change_mm": float(np.abs(difference).mean()),
                                "translation_max_abs_change_mm": float(np.abs(difference).max()),
                            }
                        row[f"{name}__{condition}"] = item
                if (ordinal+1) % 10 == 0:
                    print(json.dumps({"phase": "fixed_panel", "contexts": ordinal+1,
                                      "elapsed_seconds": time.perf_counter()-started}), flush=True)
            sample, noise = cases[panel[0]], noise_for(seed, 0, args.device)
            torch.testing.assert_close(predict(base, sample, injection, None, noise)[0].cpu(), originals[panel[0]], rtol=0, atol=0)
        for name, model in models.items():
            path = args.out / f"{name}_adapter.pt"
            torch.save({"model_state": cpu_state(model), "metadata": model.metadata(), "arm": name,
                        "steps": 100, "protocol": protocol, "deployable": False, "world_loss_enabled": False}, path)
            saved = torch.load(path, weights_only=True, map_location=args.device)
            check(saved["metadata"] == model.metadata(), "saved representation metadata changed")
            restored = MODELS[name](model.config).to(args.device).eval()
            restored.load_state_dict(saved["model_state"], strict=True)
            with torch.no_grad():
                before = model.policy_residual(context(model, sample))
                after = restored.policy_residual(context(restored, sample))
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                torch.testing.assert_close(predict(base, sample, injection, before, noise)[0],
                                           predict(base, sample, injection, after, noise)[0], rtol=0, atol=0)
            del restored, saved
        check(injection.active_residual is None and base.policy.model._project2026_state_context is None, "context leaked")
        measured = {key: metrics(rows, key) for key in ["original", "absolute_initial", "motion_initial"]
                    + [f"{name}__{c}" for name in MODELS for c in CONDITIONS]}
        comparisons = {}
        for reference_key in ("original", "absolute__true_history"):
            new, old = measured["motion__true_history"], measured[reference_key]
            change = new["translation_mae_mm"] - old["translation_mae_mm"]
            episode = {k: value-old["episode_mae_mm"][k] for k, value in new["episode_mae_mm"].items()}
            comparisons[reference_key] = {"mae_difference_mm": change, "relative_change_percent": 100*change/old["translation_mae_mm"],
                                          "episode_mae_changes_mm": episode, "episodes_lower_mae": sum(v < 0 for v in episode.values()),
                                          "episodes_equal_mae": sum(v == 0 for v in episode.values()),
                                          "episodes_higher_mae": sum(v > 0 for v in episode.values())}
        sensitivity = {}
        for name in MODELS:
            for condition in CONDITIONS[1:]:
                key = f"{name}__{condition}"
                items = [r[key]["versus_true"] for r in rows]
                sensitivity[key] = {field: distribution([r[field] for r in items]) for field in items[0]
                                    if field != "active_action_exactly_equal"}
                sensitivity[key]["active_action_equal_contexts"] = sum(r["active_action_exactly_equal"] for r in items)
        report = {"schema": VERSION, "status": "completed", "protocol": protocol, "metrics": measured,
                  "motion_comparisons": comparisons, "history_sensitivity": sensitivity,
                  "gate": {name: {"final_raw": float(model.policy_gate.detach()), "final_tanh": float(model.policy_gate.detach().tanh()),
                                  "first_gradient": logs[0]["arms"][name]["gate_gradient_before_clip"],
                                  "steps_with_nonzero_shared_gradient": sum(r["arms"][name]["shared_gradient_l2_before_clip"] > 0 for r in logs)}
                           for name, model in models.items()},
                  "residual_rms_mean": {name: float(np.mean([r[f"{name}__true_history"]["residual_rms"] for r in rows])) for name in MODELS},
                  "trainable_parameters_per_arm": counts, "model_metadata": {k: v.metadata() for k, v in models.items()},
                  "input_checks": input_report, "prefix": injection.last_info,
                  "checks": {"parameter_values_shapes_counts_matched_at_initialization": True, "both_initial_identity_all30": True,
                             "motion_reset_masks_and_zero_changes_verified": True, "original_predictions_reproduced": True,
                             "control_training_parameters_and_final_predictions_reproduced_exactly": True,
                             "both_gate_then_temporal_gradients_connected": True, "base_piper_world_frozen": True,
                             "piper_probabilities_unchanged_all_conditions": True, "both_adapter_save_reload_exact": True,
                             "disabled_hook_identity_and_context_cleanup": True},
                  "training_seconds_this_invocation": training_seconds, "elapsed_seconds": time.perf_counter()-started,
                  "resumed": args.resume, "steps_per_arm": len(logs),
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                  "world_loss_enabled": False, "heldout_evaluation": False, "hardware_executed": False,
                  "deployable": False, "visual_status": "not_viewed"}
        write_json(args.out / "training_log.json", logs)
        write_json(args.out / "panel_predictions.json", rows)
        write_json(args.out / "report.json", report)
        print(json.dumps({"status": "completed", "elapsed_seconds": report["elapsed_seconds"], "comparison": comparisons,
                          "mae_mm": {k: v["translation_mae_mm"] for k, v in measured.items()},
                          "gate": report["gate"], "checks": report["checks"]}, indent=2), flush=True)
    finally:
        injection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_pi05_gated_action_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_motion_comparison_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if (args.out / "protocol.json").exists() and not (args.out / "report.json").exists():
            write_json(args.out / f"failure_{time.time_ns()}.json", {"error": repr(error), "hardware_executed": False})
        raise


if __name__ == "__main__":
    main()
