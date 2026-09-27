"""Frozen-weight precision/history audit; no optimizer, training, or hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from check_real10_pi05_joint_temporal import Real10TemporalView, check, context, flow_loss, predict
from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_gated_temporal_world import GatedTemporalWorld, StateTokenResidualInjection
from pi05_joint_temporal_world import JointTemporalConfig
from real10_pi05_policy import Real10PI05Policy
from run_real10_pi05_gated_action import entry, metrics
from run_real10_pi05_joint_comparison import noise_for


VERSION = "real10_gated_precision_history_audit_v1"
CONDITIONS = ("true_history", "repeat_current", "swap_past")


def rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def relative_l2(value, reference):
    denominator = float(reference.float().norm())
    return float((value.float() - reference.float()).norm()) / denominator if denominator else None


def distribution(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    return {"count": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())} if len(a) else {"count": 0}


class PrefixCapture:
    """Observe the unmodified production hook both before and after its addition."""
    def __init__(self, injection):
        self.injection = injection
        self.original_before = injection.original
        self.original_after = injection.model.embed_prefix
        self.records = []
        self.before = None
        owner = self

        def capture_before(*args, **kwargs):
            embeddings, pad, att = owner.original_before(*args, **kwargs)
            if injection.active_residual is not None:
                owner.before = (embeddings[:, -1].detach().clone(), pad, att)
            return embeddings, pad, att

        def capture_after(model, *args, **kwargs):
            result = owner.original_after(*args, **kwargs)
            residual = injection.active_residual
            if residual is not None:
                state, pad, att = owner.before
                embeddings, new_pad, new_att = result
                check(pad is new_pad and att is new_att, "instrumented hook changed mask objects")
                check(state.dtype == torch.bfloat16, "expected the recorded BF16 prefix")
                requested = residual.detach()
                cast = requested.to(dtype=state.dtype)
                updated = embeddings[:, -1].detach()
                torch.testing.assert_close(updated, state + cast, rtol=0, atol=0)
                owner.records.append({"state": state.float().cpu(), "requested": requested.float().cpu(),
                                      "cast": cast.float().cpu(), "effective": (updated.float() - state.float()).cpu()})
            return result  # No replacement tensors are fed back into the policy.

        injection.original = capture_before
        injection.model.embed_prefix = types.MethodType(capture_after, injection.model)

    def close(self):
        self.injection.original = self.original_before
        self.injection.model.embed_prefix = self.original_after


def quantization_stats(record):
    r, q, e, state = (record[k].flatten() for k in ("requested", "cast", "effective", "state"))
    nonzero = r != 0
    check(bool(nonzero.any()), "this frozen nonzero-gate audit requires a nonzero requested residual")
    rn, en = float(r.norm()), float(e.norm())
    return {"dimensions": r.numel(), "state_rms": rms(state), "requested_rms": rms(r),
            "cast_rms": rms(q), "effective_rms": rms(e),
            "cast_relative_l2_error": relative_l2(q, r),
            "cast_nonzero_fraction": float((q != 0).float().mean()),
            "effective_changed_fraction": float((e != 0).float().mean()),
            "swallowed_nonzero_fraction": float((e[nonzero] == 0).float().mean()),
            "effective_to_requested_l2_ratio": en/rn,
            "effective_relative_l2_error": relative_l2(e, r),
            "effective_cosine_with_requested": float(torch.dot(e, r))/(en*rn) if en else None}


def modified_history(case, condition):
    if condition == "true_history":
        return case
    changed = dict(case)
    for key in ("visual", "states"):
        values = case[key].clone()
        if condition == "repeat_current":
            values[:, :-1] = case[key][:, -1:].expand_as(values[:, :-1])
        elif condition == "swap_past":
            values[:, :2] = case[key][:, [1, 0]]
        else:
            raise ValueError(condition)
        changed[key] = values
        torch.testing.assert_close(changed[key][:, -1], case[key][:, -1], rtol=0, atol=0)
    check(changed["processed"] is case["processed"] and changed["ages"] is case["ages"]
          and changed["valid"] is case["valid"], "current observation / timing / masks changed")
    return changed


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    source = read_json(args.source / "report.json")
    check(source["status"] == "completed" and source["world_loss_enabled"] is False, "requires completed action-only source")
    source_protocol = source["protocol"]["source_protocol"]
    panel, seed = source_protocol["panel_indices"], source_protocol["seed"]
    data = Real10TemporalView(Path(source_protocol["pack"]))
    check(data.audit_index() == source_protocol["index_audit"], "data index changed")
    check([data.rows[i]["sample_id"] for i in panel] == source_protocol["panel_sample_ids"], "panel changed")
    protocol = {"version": VERSION, "source": str(args.source.resolve()), "source_protocol": source["protocol"],
                "conditions": list(CONDITIONS), "panel_indices": panel, "seed": seed,
                "fixed": "current observation, timestamps, valid masks, weights, per-context noise",
                "intervention": "past visual/state content only; deterministic two-past-slot swap",
                "optimizer_steps": 0, "precision_changed": False, "hardware_executed": False,
                "heldout": False, "design": "docs/algorithm-real10-gated-history-audit-protocol-20260921.md"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    checkpoint = torch.load(args.source / "gated_action_adapter.pt", map_location="cpu", weights_only=True)
    check(checkpoint["protocol"] == source["protocol"] and checkpoint["steps"] == 100, "checkpoint source differs")
    root = Path(source_protocol["training_root"])
    base = Real10PI05Policy(root / "elite/final_policy.pt", root / "piper/mixed_head_policy.pt", seed=seed, device=args.device)
    print(f"Frozen original pair loaded in {base.metadata['load_seconds']:.2f}s", flush=True)
    adapter = GatedTemporalWorld(JointTemporalConfig(**checkpoint["metadata"]["config"])).to(args.device)
    adapter.load_state_dict(checkpoint["model_state"], strict=True)
    adapter.requires_grad_(False).eval()
    reference = {r["pack_index"]: r for r in read_json(args.source / "panel_predictions.json")}
    injection = StateTokenResidualInjection(base.policy)
    capture = PrefixCapture(injection)
    rows, arrays = [], {}
    try:
        for ordinal, index in enumerate(panel):
            case = data.encode_case(base, index)
            check(bool(case["valid"].all()) and case["visual"].shape[1] == 3, "requires two valid past slots")
            noise = noise_for(seed, ordinal, args.device)
            original, packet = predict(base, case, injection, None, noise)
            check(packet == reference[index]["original"]["prediction"], "original prediction reproduction failed")
            row = {"pack_index": index, "sample_id": data.rows[index]["sample_id"],
                   "episode": data.rows[index]["source"]["episode"], "task": data.rows[index]["task"],
                   "original": entry(packet, flow_loss(base, case, noise), data.actions[index, :3]),
                   "history_indices": case["history_indices"], "relative_times_s": case["ages"].cpu().tolist()}
            cache = {}
            for condition in CONDITIONS:
                variant = modified_history(case, condition)
                tokens = context(adapter, variant)
                residual = adapter.policy_residual(tokens)
                capture.records = []
                action, output = predict(base, case, injection, residual, noise)
                check(len(capture.records) == 1, "expected one observed prefix per sampled action")
                record = capture.records[0]
                with injection.condition(residual):
                    loss = flow_loss(base, case, noise)
                item = entry(output, loss, data.actions[index, :3])
                item["quantization"] = quantization_stats(record)
                item["input_change"] = {"past_visual_rms": rms(variant["visual"][:, :-1] - case["visual"][:, :-1]),
                                        "past_state_rms": rms(variant["states"][:, :-1] - case["states"][:, :-1])}
                check(output["piper_probabilities"] == packet["piper_probabilities"], "Piper output changed")
                if condition == "true_history":
                    check(output == reference[index]["gated_final"]["prediction"], "frozen gated prediction reproduction failed")
                else:
                    true = cache["true_history"]
                    torch.testing.assert_close(record["state"], true["record"]["state"], rtol=0, atol=0)
                    difference = np.asarray(output["elite_tcp_delta_6d"][:3]) - np.asarray(row["true_history"]["prediction"]["elite_tcp_delta_6d"][:3])
                    item["versus_true"] = {
                        "tokens_relative_l2_change": relative_l2(tokens, true["tokens"]),
                        "requested_residual_relative_l2_change": relative_l2(residual, true["residual"]),
                        "effective_token_equal": torch.equal(record["effective"], true["record"]["effective"]),
                        "effective_changed_coordinate_fraction": float((record["effective"] != true["record"]["effective"]).float().mean()),
                        "active_action_exactly_equal": torch.equal(action[..., :3], true["action"][..., :3]),
                        "normalized_translation_max_abs_change": float((action[..., :3] - true["action"][..., :3]).abs().max()),
                        "translation_abs_change_xyz_mm": np.abs(difference).tolist(),
                        "translation_mean_abs_change_mm": float(np.abs(difference).mean()),
                        "translation_max_abs_change_mm": float(np.abs(difference).max()),
                    }
                cache[condition] = {"tokens": tokens, "residual": residual, "action": action, "record": record}
                row[condition] = item
                for name, tensor in {**record, "shared_tokens": tokens}.items():
                    arrays.setdefault(f"{condition}__{name}", []).append(tensor.detach().float().cpu().numpy()[0])
            rows.append(row)
            if (ordinal + 1) % 10 == 0:
                write_json(args.out / "partial_measurements.json", rows)
                print(json.dumps({"contexts_done": ordinal+1, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        summary = {}
        for condition in CONDITIONS:
            fields = rows[0][condition]["quantization"]
            quantization = {key: distribution([r[condition]["quantization"][key] for r in rows]) for key in fields if key != "dimensions"}
            summary[condition] = {"metrics": metrics(rows, condition), "quantization": quantization}
            if condition != "true_history":
                compare = [r[condition]["versus_true"] for r in rows]
                summary[condition]["versus_true"] = {
                    key: distribution([r[key] for r in compare])
                    for key in ("tokens_relative_l2_change", "requested_residual_relative_l2_change",
                                "effective_changed_coordinate_fraction", "normalized_translation_max_abs_change",
                                "translation_mean_abs_change_mm", "translation_max_abs_change_mm")}
                summary[condition]["versus_true"].update(
                    effective_token_equal_contexts=sum(r["effective_token_equal"] for r in compare),
                    active_action_equal_contexts=sum(r["active_action_exactly_equal"] for r in compare))
                summary[condition]["input_change"] = {
                    key: distribution([r[condition]["input_change"][key] for r in rows])
                    for key in ("past_visual_rms", "past_state_rms")}
        for key, value in adapter.state_dict().items():
            torch.testing.assert_close(value.cpu(), checkpoint["model_state"][key], rtol=0, atol=0)
        check(all(not p.requires_grad and p.grad is None for model in (base.policy, base.head, adapter)
                  for p in model.parameters()), "an audited model is trainable or has gradients")
        check(injection.active_residual is None and base.policy.model._project2026_state_context is None, "leaked context")
        np.savez_compressed(args.out / "captured_vectors.npz", **{key: np.stack(value) for key, value in arrays.items()})
        write_json(args.out / "measurements.json", rows)
        report = {"schema": VERSION, "status": "completed", "protocol": protocol, "contexts": len(rows),
                  "original_metrics": metrics(rows, "original"), "conditions": summary,
                  "gate_tanh": float(adapter.policy_gate.tanh()), "prefix": injection.last_info,
                  "checks": {"original_and_gated_predictions_reproduced_all30": True,
                             "actual_native_addition_and_masks_verified": True,
                             "current_inputs_timing_noise_fixed": True, "adapter_weights_exactly_unchanged": True,
                             "all_models_frozen_no_grad": True, "piper_unchanged": True, "contexts_cleared": True},
                  "captured_vectors": "captured_vectors.npz; float32 copies of actual BF16 state/cast/effective deltas",
                  "optimizer_steps": 0, "world_loss_enabled": False, "precision_changed": False,
                  "hardware_executed": False, "heldout_evaluation": False, "visual_status": "not_viewed",
                  "elapsed_seconds": time.perf_counter()-started}
        write_json(args.out / "report.json", report)
        compact = {"status": "completed", "true_history_quantization": summary["true_history"]["quantization"],
                   "mae_mm": {name: summary[name]["metrics"]["translation_mae_mm"] for name in CONDITIONS},
                   "counterfactuals": {name: summary[name]["versus_true"] for name in CONDITIONS[1:]},
                   "elapsed_seconds": report["elapsed_seconds"]}
        print(json.dumps(compact, indent=2), flush=True)
    finally:
        capture.close()
        injection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_pi05_gated_action_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_gated_history_audit_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if (args.out / "protocol.json").exists() and not (args.out / "report.json").exists():
            write_json(args.out / f"failure_{time.time_ns()}.json", {"error": repr(error), "optimizer_steps": 0})
        raise


if __name__ == "__main__":
    main()
