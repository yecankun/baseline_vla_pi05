"""Fixed Real10 gated-state residual: identity check plus action-only 100 steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_real10_pi05_joint_temporal import Real10TemporalView, check, context, flow_loss, predict
from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_gated_temporal_world import GatedTemporalWorld, StateTokenResidualInjection
from pi05_joint_temporal_world import JointTemporalConfig
from probe_pi05_lerobot_adapter import set_seed
from real10_pi05_policy import Real10PI05Policy
from run_real10_pi05_joint_comparison import cpu_state, noise_for


VERSION = "real10_gated_temporal_action_v1"


def entry(packet, flow, target):
    errors = np.abs(np.asarray(packet["elite_tcp_delta_6d"][:3]) - target)
    return {"prediction": packet, "translation_abs_error_xyz_mm": errors.tolist(),
            "translation_mae_mm": float(errors.mean()), "action_flow_loss": float(flow)}


def metrics(rows, key):
    values = [r[key] for r in rows]
    return {"translation_mae_mm": float(np.mean([v["translation_mae_mm"] for v in values])),
            "translation_mae_xyz_mm": np.mean([v["translation_abs_error_xyz_mm"] for v in values], 0).tolist(),
            "action_flow_loss": float(np.mean([v["action_flow_loss"] for v in values])),
            "task_mae_mm": {task: float(np.mean([r[key]["translation_mae_mm"] for r in rows if r["task"] == task]))
                            for task in ("left", "right")},
            "episode_mae_mm": {ep: float(np.mean([r[key]["translation_mae_mm"] for r in rows if r["episode"] == ep]))
                               for ep in sorted({r["episode"] for r in rows})}}


def gradient_norm(parameters):
    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def save_checkpoint(path, model, optimizer, logs, protocol):
    temporary = path.with_suffix(".tmp")
    torch.save({"version": VERSION, "model_state": cpu_state(model), "optimizer": optimizer.state_dict(),
                "logs": logs, "step": len(logs), "protocol": protocol, "metadata": model.metadata(),
                "deployable": False}, temporary)
    temporary.replace(path)


def run(args):
    started = time.perf_counter()
    reference = read_json(args.reference / "report.json")
    previous = reference["protocol"]
    check(reference["status"] == "completed" and previous["steps_per_arm"] == 100 and previous["seed"] == 123,
          "requires the completed, fixed 100-step seed123 reference")
    data = Real10TemporalView(Path(previous["pack"]))
    check(data.audit_index() == previous["index_audit"], "source index changed")
    panel, order, seed = previous["panel_indices"], previous["training_order"], previous["seed"]
    check([data.rows[i]["sample_id"] for i in panel] == previous["panel_sample_ids"], "panel changed")
    check([data.rows[i]["sample_id"] for i in order] == previous["training_sample_ids"], "training sequence changed")
    protocol = {"version": VERSION, "reference": str(args.reference.resolve()), "source_protocol": previous,
                "connection": "zero signed-tanh scalar residual on existing state token; no new positions",
                "initialization": "saved untrained reference initial_state + zero gate",
                "world_loss_enabled": False, "world_predictor_frozen": True, "steps": 100,
                "lr": 1e-4, "weight_decay": 0., "policy_gradient_clip_l2": 1.,
                "heldout": False, "hardware_executed": False,
                "design": "docs/algorithm-real10-gated-prefix-protocol-20260921.md"}
    if args.resume:
        check(not (args.out / "report.json").exists(), "completed run must not be repeated")
        check(read_json(args.out / "protocol.json") == protocol, "resume protocol differs")
        resumed = torch.load(args.out / "checkpoint.pt", map_location="cpu", weights_only=True)
        check(resumed["protocol"] == protocol and resumed["version"] == VERSION, "checkpoint protocol differs")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "protocol.json", protocol)
        resumed = None
    old_pair = torch.load(args.reference / "paired_checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    check(old_pair["protocol"] == previous, "reference initialization provenance differs")
    initial = dict(old_pair["initial_state"], policy_gate=torch.zeros(()))
    del old_pair
    root = Path(previous["training_root"])
    set_seed(seed)
    base = Real10PI05Policy(root / "elite/final_policy.pt", root / "piper/mixed_head_policy.pt", seed=seed, device=args.device)
    print(f"Frozen original model loaded in {base.metadata['load_seconds']:.2f}s", flush=True)
    cases = {i: data.encode_case(base, i) for i in panel}
    dim = cases[panel[0]]["visual"].shape[-1]
    model = GatedTemporalWorld(JointTemporalConfig(visual_dim=dim, prefix_dim=dim)).to(args.device).eval()
    model.load_state_dict(initial, strict=True)
    model.world.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.policy_parameters(), lr=1e-4, weight_decay=0.)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    historical = {r["pack_index"]: r for r in read_json(args.reference / "panel_predictions.json")}
    injection = StateTokenResidualInjection(base.policy)
    rows, originals = [], {}
    try:
        with torch.no_grad():
            for ordinal, i in enumerate(panel):
                case = cases[i]
                noise = noise_for(seed, ordinal, args.device)
                original, packet = predict(base, case, injection, None, noise)
                original_flow = flow_loss(base, case, noise)
                residual = model.policy_residual(context(model, case))
                action, gated_packet = predict(base, case, injection, residual, noise)
                with injection.condition(residual):
                    gated_flow = flow_loss(base, case, noise)
                torch.testing.assert_close(action, original, rtol=0, atol=0)
                torch.testing.assert_close(gated_flow, original_flow, rtol=0, atol=0)
                check(packet == gated_packet, "gate-zero output packet differs")
                check(packet == historical[i]["variants"]["original_pi05"]["prediction"],
                      "original outputs do not reproduce the historical reference")
                target = data.actions[i, :3].astype(float)
                rows.append({"pack_index": i, "sample_id": data.rows[i]["sample_id"],
                             "episode": data.rows[i]["source"]["episode"], "task": data.rows[i]["task"],
                             "target_translation_mm": target.tolist(), "piper_target": int(data.intents[i]),
                             "original": entry(packet, original_flow, target),
                             "gated_initial": entry(gated_packet, gated_flow, target)})
                originals[i] = original.cpu()
        write_json(args.out / "identity_check.json", {"status": "passed", "contexts": len(panel),
                   "normalized_action_exact": True, "flow_loss_exact": True, "piper_exact": True,
                   "original_matches_historical_exact": True, "prefix": injection.last_info,
                   "gate": float(model.policy_gate.detach()), "hardware_executed": False})
        print("All30 gate-zero actions/losses exactly equal original; original matches saved reference.", flush=True)
        logs = []
        if resumed is not None:
            model.load_state_dict(resumed["model_state"], strict=True)
            optimizer.load_state_dict(resumed["optimizer"])
            logs = resumed["logs"]
            check(resumed["step"] == len(logs), "checkpoint step mismatch")
        else:
            save_checkpoint(args.out / "checkpoint.pt", model, optimizer, logs, protocol)
        training_started = time.perf_counter()
        for step in range(len(logs), 100):
            i = order[step]
            if i not in cases:
                cases[i] = data.encode_case(base, i)
            case = cases[i]
            set_seed(seed + 200_000 + step)
            noise = base.policy.model.sample_noise(case["action_target"].shape, args.device)
            flow_time = base.policy.model.sample_time(1, args.device)
            optimizer.zero_grad(set_to_none=True)
            residual = model.policy_residual(context(model, case))
            with injection.condition(residual):
                loss = flow_loss(base, case, noise, flow_time=flow_time)
                loss.backward()
            gate_gradient = float(model.policy_gate.grad.detach())
            shared_gradient = gradient_norm(model.temporal.parameters())
            total_gradient = torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 1., error_if_nonfinite=True)
            if step == 0:
                check(abs(gate_gradient) > 0 and np.isfinite(gate_gradient), "initial gate gradient absent/nonfinite")
                check(shared_gradient == 0., "zero gate must block temporal action gradient initially")
            optimizer.step()
            logs.append({"step": step + 1, "pack_index": i, "sample_id": data.rows[i]["sample_id"],
                         "flow_time": float(flow_time[0]), "action_loss": float(loss.detach()),
                         "gate_gradient_before_clip": gate_gradient, "shared_gradient_l2_before_clip": shared_gradient,
                         "policy_gradient_l2_before_clip": float(total_gradient),
                         "gate_after_step": float(model.policy_gate.detach()),
                         "gate_tanh_after_step": float(model.policy_gate.detach().tanh())})
            if (step + 1) % 25 == 0:
                save_checkpoint(args.out / "checkpoint.pt", model, optimizer, logs, protocol)
                print(json.dumps({"phase": "training", **logs[-1], "elapsed_seconds": time.perf_counter() - started}), flush=True)
        training_seconds = time.perf_counter() - training_started
        check(any(r["shared_gradient_l2_before_clip"] > 0 for r in logs[1:]), "temporal gradient never opens")
        check(all(not p.requires_grad and p.grad is None for p in base.policy.parameters()), "base policy not frozen")
        check(all(not p.requires_grad and p.grad is None for p in base.head.parameters()), "Piper not frozen")
        for k, v in model.state_dict().items():
            if k.startswith("world."):
                torch.testing.assert_close(v.cpu(), initial[k], rtol=0, atol=0)
        with torch.no_grad():
            for ordinal, row in enumerate(rows):
                i = row["pack_index"]
                residual = model.policy_residual(context(model, cases[i]))
                noise = noise_for(seed, ordinal, args.device)
                action, packet = predict(base, cases[i], injection, residual, noise)
                with injection.condition(residual):
                    action_loss = flow_loss(base, cases[i], noise)
                row["gated_final"] = entry(packet, action_loss, data.actions[i, :3].astype(float))
                row["residual_rms"] = float(residual.float().square().mean().sqrt())
                row["normalized_action_max_abs_drift"] = float((action.cpu() - originals[i]).abs().max())
                check(packet["piper_probabilities"] == row["original"]["prediction"]["piper_probabilities"],
                      "Piper changed")
            sample = cases[panel[0]]
            noise = noise_for(seed, 0, args.device)
            disabled = predict(base, sample, injection, None, noise)[0]
            torch.testing.assert_close(disabled.cpu(), originals[panel[0]], rtol=0, atol=0)
        check(injection.active_residual is None and base.policy.model._project2026_state_context is None, "leaked context")
        final = {"model_state": cpu_state(model), "metadata": model.metadata(), "protocol": protocol,
                 "steps": 100, "deployable": False, "world_loss_enabled": False}
        torch.save(final, args.out / "gated_action_adapter.pt")
        restored = GatedTemporalWorld(model.config).to(args.device).eval()
        restored.load_state_dict(torch.load(args.out / "gated_action_adapter.pt", map_location=args.device, weights_only=True)["model_state"], strict=True)
        restored.world.requires_grad_(False)
        with torch.no_grad():
            residual = model.policy_residual(context(model, sample))
            reloaded = restored.policy_residual(context(restored, sample))
            torch.testing.assert_close(residual, reloaded, rtol=0, atol=0)
            before = predict(base, sample, injection, residual, noise)[0]
            after = predict(base, sample, injection, reloaded, noise)[0]
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        measured = {name: metrics(rows, name) for name in ("original", "gated_initial", "gated_final")}
        a, b = measured["original"]["translation_mae_mm"], measured["gated_final"]["translation_mae_mm"]
        episode_changes = {e: measured["gated_final"]["episode_mae_mm"][e] - v
                           for e, v in measured["original"]["episode_mae_mm"].items()}
        report = {"schema": VERSION, "status": "completed", "protocol": protocol,
                  "metrics": measured, "historical_append_only_action_metrics": reference["evaluation"]["metrics"]["action_only"],
                  "comparison": {"gated_minus_original_mae_mm": b-a, "relative_change_percent": 100*(b-a)/a,
                                 "episode_mae_changes_mm": episode_changes,
                                 "episodes_lower_mae": sum(v < 0 for v in episode_changes.values()),
                                 "episodes_equal_mae": sum(v == 0 for v in episode_changes.values()),
                                 "episodes_higher_mae": sum(v > 0 for v in episode_changes.values())},
                  "gate": {"initial": 0., "final_raw": float(model.policy_gate.detach()),
                           "final_tanh": float(model.policy_gate.detach().tanh()),
                           "first_gradient": logs[0]["gate_gradient_before_clip"],
                           "first_shared_gradient": logs[0]["shared_gradient_l2_before_clip"],
                           "steps_with_nonzero_shared_gradient": sum(r["shared_gradient_l2_before_clip"] > 0 for r in logs)},
                  "residual_rms_mean": float(np.mean([r["residual_rms"] for r in rows])),
                  "max_normalized_action_drift": max(r["normalized_action_max_abs_drift"] for r in rows),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "adapter": model.metadata(), "base_checkpoint": base.metadata, "prefix": injection.last_info,
                  "checks": {"initial_identity_all30": True, "original_reference_exact": True,
                             "action_reaches_gate_then_temporal": True, "base_piper_world_frozen": True,
                             "disabled_action_exact": True, "save_reload_action_exact": True, "contexts_cleared": True},
                  "elapsed_seconds": time.perf_counter() - started, "training_seconds": training_seconds,
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                  "world_loss_enabled": False, "heldout_evaluation": False, "hardware_executed": False,
                  "deployable": False, "visual_status": "not_viewed"}
        write_json(args.out / "training_log.json", logs)
        write_json(args.out / "panel_predictions.json", rows)
        write_json(args.out / "report.json", report)
        print(json.dumps({k: report[k] for k in ("status", "comparison", "gate", "checks", "elapsed_seconds")}, indent=2), flush=True)
    finally:
        injection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_pi05_joint_comparison_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_gated_action_v1"))
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
