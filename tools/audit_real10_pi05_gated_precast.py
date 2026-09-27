"""Frozen single-factor pre/post-BF16 residual comparison; no training/hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from audit_real10_pi05_gated_history import CONDITIONS, distribution, modified_history, quantization_stats, relative_l2, rms
from check_real10_pi05_joint_temporal import Real10TemporalView, check, context, flow_loss, predict
from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_gated_temporal_world import GatedTemporalWorld, PrecastStateTokenResidualInjection, StateTokenResidualInjection
from pi05_joint_temporal_world import JointTemporalConfig
from real10_pi05_policy import Real10PI05Policy
from run_real10_pi05_gated_action import entry, metrics
from run_real10_pi05_joint_comparison import noise_for


VERSION = "real10_gated_precast_audit_v1"
PATHS = ("postcast", "precast")


class NumericalCapture:
    """Transparent observation; install before the pre-cast mutation hook."""
    def __init__(self, policy):
        self.model = policy.model
        self.original = self.model.embed_prefix
        self.mode, self.residual, self.record, self.baseline = "original", None, None, None
        self.calls = 0
        self.last_info = None
        owner = self

        def projection(module, inputs, output):
            check(output.dtype == torch.float32, "native state projection is not FP32")
            owner.projected = output.detach().clone()
            return output

        self.handle = self.model.state_proj.register_forward_hook(projection)

        def prefix(model, *args, **kwargs):
            embeddings, pad, att = owner.original(*args, **kwargs)
            check(embeddings.dtype == torch.bfloat16, "expected unchanged BF16 prefix")
            s = owner.projected
            r = torch.zeros_like(s) if owner.residual is None else owner.residual.detach()
            check(r.dtype == torch.float32, "expected frozen FP32 requested residual")
            baseline = s.to(dtype=embeddings.dtype)
            ideal = s + r
            expected = (ideal.to(embeddings.dtype) if owner.mode == "precast"
                        else baseline + r.to(embeddings.dtype))
            actual = embeddings[:, -1].detach()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if owner.baseline is None:
                check(owner.mode == "original", "capture each context's baseline first")
                owner.baseline = (embeddings.detach().clone(), pad.detach().clone(), att.detach().clone())
            else:
                prior, prior_pad, prior_att = owner.baseline
                check(embeddings.shape == prior.shape, "prefix shape changed")
                check(torch.equal(embeddings[:, :-1], prior[:, :-1]), "non-state prefix changed")
                check(torch.equal(pad, prior_pad) and torch.equal(att, prior_att), "prefix masks changed")
            owner.record = {"projected": s.cpu(), "state": baseline.float().cpu(),
                            "requested": r.float().cpu(), "cast": r.to(embeddings.dtype).float().cpu(),
                            "effective": (actual.float() - baseline.float()).cpu(),
                            "updated": actual.float().cpu(), "ideal": ideal.float().cpu()}
            owner.calls += 1
            owner.last_info = {"projected_dtype": str(s.dtype), "prefix_dtype": str(embeddings.dtype),
                               "prefix_tokens": int(embeddings.shape[1]), "masks_unchanged": True,
                               "non_state_prefix_exactly_unchanged": True}
            return embeddings, pad, att

        self.model.embed_prefix = types.MethodType(prefix, self.model)

    def reset(self, mode, residual):
        self.mode, self.residual, self.calls, self.record = mode, residual, 0, None

    def close(self):
        self.handle.remove()
        self.model.embed_prefix = self.original


def action_difference(action, packet, other_action, other_packet):
    difference = np.asarray(packet["elite_tcp_delta_6d"][:3]) - np.asarray(other_packet["elite_tcp_delta_6d"][:3])
    return {"active_action_exactly_equal": torch.equal(action[..., :3], other_action[..., :3]),
            "normalized_translation_max_abs_change": float((action[..., :3] - other_action[..., :3]).abs().max()),
            "translation_mean_abs_change_mm": float(np.abs(difference).mean()),
            "translation_max_abs_change_mm": float(np.abs(difference).max())}


def summarize_comparisons(items):
    result = {key: distribution([r[key] for r in items]) for key in items[0]
              if not isinstance(items[0][key], bool)}
    result.update({f"{key}_contexts": sum(r[key] for r in items) for key in items[0]
                   if isinstance(items[0][key], bool)})
    return result


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    source = read_json(args.source / "report.json")
    prior = read_json(args.reference / "report.json")
    check(source["status"] == "completed" and source["world_loss_enabled"] is False, "requires completed action-only source")
    check(prior["status"] == "completed" and prior["protocol"]["source_protocol"] == source["protocol"], "audit source differs")
    fixed = source["protocol"]["source_protocol"]
    panel, seed = fixed["panel_indices"], fixed["seed"]
    data = Real10TemporalView(Path(fixed["pack"]))
    check(data.audit_index() == fixed["index_audit"], "dataset index changed")
    check([data.rows[i]["sample_id"] for i in panel] == fixed["panel_sample_ids"], "panel changed")
    protocol = {"version": VERSION, "source": str(args.source.resolve()), "reference": str(args.reference.resolve()),
                "source_protocol": source["protocol"], "panel_indices": panel, "seed": seed,
                "paths": list(PATHS), "history_conditions": list(CONDITIONS),
                "single_factor": "FP32 residual before state-token BF16 cast versus unchanged post-cast addition",
                "fixed": "all weights, gate, current inputs, history diagnostics, slot times, masks and noise",
                "optimizer_steps": 0, "heldout": False, "hardware_executed": False,
                "design": "docs/algorithm-real10-gated-precast-protocol-20260921.md"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", protocol)
    checkpoint = torch.load(args.source / "gated_action_adapter.pt", map_location="cpu", weights_only=True)
    check(checkpoint["protocol"] == source["protocol"] and checkpoint["steps"] == 100, "fixed checkpoint differs")
    root = Path(fixed["training_root"])
    base = Real10PI05Policy(root / "elite/final_policy.pt", root / "piper/mixed_head_policy.pt", seed=seed, device=args.device)
    print(f"Frozen original pair loaded in {base.metadata['load_seconds']:.2f}s", flush=True)
    adapter = GatedTemporalWorld(JointTemporalConfig(**checkpoint["metadata"]["config"])).to(args.device)
    adapter.load_state_dict(checkpoint["model_state"], strict=True)
    adapter.requires_grad_(False).eval()
    references = {r["pack_index"]: r for r in read_json(args.reference / "measurements.json")}
    reference_vectors = np.load(args.reference / "captured_vectors.npz", allow_pickle=False)
    post = StateTokenResidualInjection(base.policy)
    capture = NumericalCapture(base.policy)  # Registered first: capture native, unmodified state_proj output.
    pre = PrecastStateTokenResidualInjection(base.policy)
    injections = {"postcast": post, "precast": pre}
    rows, arrays, last_case = [], {}, None
    try:
        for ordinal, index in enumerate(panel):
            case = data.encode_case(base, index)
            check(bool(case["valid"].all()), "panel requires two valid past slots")
            noise = noise_for(seed, ordinal, args.device)
            capture.baseline = None
            capture.reset("original", None)
            original_action, original_packet = predict(base, case, None, None, noise)
            original_record = capture.record
            original_loss = flow_loss(base, case, noise)
            reference = references[index]
            check(original_packet == reference["original"]["prediction"], "original packet reproduction failed")
            check(float(original_loss) == reference["original"]["action_flow_loss"], "original flow reproduction failed")
            row = {"pack_index": index, "sample_id": data.rows[index]["sample_id"],
                   "episode": data.rows[index]["source"]["episode"], "task": data.rows[index]["task"],
                   "original": entry(original_packet, original_loss, data.actions[index, :3])}
            tokens = {name: context(adapter, modified_history(case, name)) for name in CONDITIONS}
            residuals = {name: adapter.policy_residual(value) for name, value in tokens.items()}
            zero = 0.0 * tokens["true_history"].mean(dim=1)
            capture.reset("precast", zero)
            zero_action, zero_packet = predict(base, case, pre, zero, noise)
            with pre.condition(zero):
                zero_loss = flow_loss(base, case, noise)
            torch.testing.assert_close(zero_action, original_action, rtol=0, atol=0)
            check(zero_packet == original_packet and torch.equal(zero_loss, original_loss), "zero-gate identity failed")
            row["zero_gate_identity"] = True
            cache = {}
            for path in PATHS:
                for condition in CONDITIONS:
                    key = f"{path}__{condition}"
                    residual = residuals[condition]
                    injection = injections[path]
                    capture.reset(path, residual)
                    action, packet = predict(base, case, injection, residual, noise)
                    check(capture.calls == 1, "expected one prefix per sampled action")
                    record = capture.record
                    with injection.condition(residual):
                        loss = flow_loss(base, case, noise)
                    torch.testing.assert_close(record["projected"], original_record["projected"], rtol=0, atol=0)
                    check(packet["piper_probabilities"] == original_packet["piper_probabilities"], "Piper changed")
                    item = entry(packet, loss, data.actions[index, :3])
                    item["quantization"] = quantization_stats(record)
                    item["quantization"].update(
                        combined_token_rms_error=rms(record["updated"] - record["ideal"]),
                        combined_token_relative_l2_error=relative_l2(record["updated"], record["ideal"]),
                        original_token_rms_error=rms(record["state"] - record["projected"]))
                    item["versus_original"] = action_difference(action, packet, original_action, original_packet)
                    if path == "postcast":
                        check(packet == reference[condition]["prediction"], "post-cast history packet reproduction failed")
                        check(float(loss) == reference[condition]["action_flow_loss"], "post-cast flow reproduction failed")
                        for name in ("state", "requested", "cast", "effective"):
                            expected = torch.from_numpy(reference_vectors[f"{condition}__{name}"][ordinal]).unsqueeze(0)
                            torch.testing.assert_close(record[name], expected, rtol=0, atol=0)
                    else:
                        old = cache[f"postcast__{condition}"]
                        item["versus_postcast"] = action_difference(action, packet, old["action"], old["packet"])
                        item["versus_postcast"]["effective_token_exactly_equal"] = torch.equal(record["effective"], old["record"]["effective"])
                        item["versus_postcast"]["mae_difference_mm"] = item["translation_mae_mm"] - row[f"postcast__{condition}"]["translation_mae_mm"]
                    if condition != "true_history":
                        true = cache[f"{path}__true_history"]
                        item["versus_true"] = action_difference(action, packet, true["action"], true["packet"])
                        item["versus_true"].update(
                            requested_residual_relative_l2_change=relative_l2(residual, residuals["true_history"]),
                            effective_token_exactly_equal=torch.equal(record["effective"], true["record"]["effective"]),
                            effective_changed_coordinate_fraction=float((record["effective"] != true["record"]["effective"]).float().mean()))
                    cache[key] = {"action": action, "packet": packet, "record": record}
                    row[key] = item
                    for name, value in {**record, "normalized_action": action.reshape(1, -1)}.items():
                        arrays.setdefault(f"{key}__{name}", []).append(value.detach().float().cpu().numpy()[0])
            rows.append(row)
            last_case = (case, noise, original_action, original_packet)
            if (ordinal + 1) % 10 == 0:
                write_json(args.out / "partial_measurements.json", rows)
                print(json.dumps({"contexts_done": ordinal+1, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        for key, value in adapter.state_dict().items():
            torch.testing.assert_close(value.cpu(), checkpoint["model_state"][key], rtol=0, atol=0)
        check(all(not p.requires_grad and p.grad is None for model in (base.policy, base.head, adapter)
                  for p in model.parameters()), "model is trainable or received gradients")
        check(post.active_residual is None and pre.active_residual is None
              and base.policy.model._project2026_state_context is None, "leaked context")
        prefix_info, pre_info = capture.last_info, pre.last_info
    finally:
        pre.close()
        capture.close()
        post.close()
        reference_vectors.close()
    restored_action, restored_packet = predict(base, last_case[0], None, None, last_case[1])
    torch.testing.assert_close(restored_action, last_case[2], rtol=0, atol=0)
    check(restored_packet == last_case[3], "closing hooks did not restore inference")
    summaries = {}
    for path in PATHS:
        for condition in CONDITIONS:
            key = f"{path}__{condition}"
            summary = {"metrics": metrics(rows, key), "quantization": {
                field: distribution([row[key]["quantization"][field] for row in rows])
                for field in rows[0][key]["quantization"] if field != "dimensions"}}
            for compare in ("versus_original", "versus_postcast", "versus_true"):
                if compare in rows[0][key]:
                    summary[compare] = summarize_comparisons([row[key][compare] for row in rows])
            summaries[key] = summary
    np.savez_compressed(args.out / "captured_vectors.npz", **{k: np.stack(v) for k, v in arrays.items()})
    write_json(args.out / "measurements.json", rows)
    report = {"schema": VERSION, "status": "completed", "protocol": protocol, "contexts": len(rows),
              "original_metrics": metrics(rows, "original"), "conditions": summaries,
              "gate_tanh": float(adapter.policy_gate.tanh()), "prefix": prefix_info, "precast_hook": pre_info,
              "checks": {"original_and_all_postcast_conditions_reproduced": True,
                         "postcast_vectors_reproduced_exactly": True, "zero_gate_full_action_packet_loss_exact_all30": True,
                         "native_fp32_projection_and_actual_formulas_verified": True, "only_state_token_changed": True,
                         "all_weights_frozen_no_grad_adapter_exactly_unchanged": True, "piper_probabilities_unchanged": True,
                         "contexts_cleared_and_hook_close_restores_original": True},
              "optimizer_steps": 0, "world_loss_enabled": False, "model_wide_precision_changed": False,
              "hardware_executed": False, "heldout_evaluation": False, "deployable": False,
              "visual_status": "not_viewed", "elapsed_seconds": time.perf_counter()-started}
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": "completed", "elapsed_seconds": report["elapsed_seconds"], "checks": report["checks"],
                      "mae_mm": {"original": report["original_metrics"]["translation_mae_mm"],
                                 **{k: v["metrics"]["translation_mae_mm"] for k, v in summaries.items()}},
                      "true_history": {p: summaries[f"{p}__true_history"]["quantization"] for p in PATHS}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_pi05_gated_action_v1"))
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_pi05_gated_history_audit_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_gated_precast_audit_v1"))
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
