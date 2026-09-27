"""Bounded real-checkpoint joint-prefix test; no full training or actuation.

Runs four predeclared train-set contexts, one disposable future-only update
(restored), and one joint diagnostic update. Never writes the base checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from export_openpi_compat_to_lerobot import read_json, read_jsonl, write_json
from pi05_action_effect_world_model import extract_pi05_multiview_visual_latent
from pi05_joint_temporal_world import JointTemporalConfig, JointTemporalWorld, TemporalPrefixInjection, visual_target
from probe_pi05_lerobot_adapter import set_seed
from real10_pi05_policy import Real10PI05Policy


class Real10TemporalView:
    """Read-only complete transition index; no episode truncation or new split."""
    def __init__(self, root, history=3):
        self.root, self.history = root, history
        self.manifest = read_json(root / "manifest.json")
        if self.manifest.get("adapter_version") != "real10_pi05_observable_history_v1":
            raise ValueError("use the real10 compact pack, not an E2 state layout")
        self.rows = read_jsonl(root / "index.jsonl")
        with np.load(root / "openpi_arrays.npz") as data:
            self.states = data["state_32"].copy()
            self.actions = data["action_32"].copy()
            self.intents = data["piper_intent_id"].copy()
        self.lookup = {(r["source"]["episode"], r["source"]["step"]): i for i, r in enumerate(self.rows)}
        if len(self.lookup) != len(self.rows) or len(self.states) != len(self.rows):
            raise ValueError("transition index is duplicated or differs from arrays")

    def history_indices(self, index):
        row = self.rows[index]
        episode, step = row["source"]["episode"], row["source"]["step"]
        indices = []
        for source_step in range(step - self.history + 1, step + 1):
            if source_step < 0:
                indices.append(None)
            else:
                indices.append(self.lookup[(episode, source_step)])
        return indices

    def audit_index(self):
        starts, terminal = 0, 0
        for i, row in enumerate(self.rows):
            if row["split"] != "train" or row["source"]["next_step"] != row["source"]["step"] + 1:
                raise ValueError("unexpected split or non-adjacent target")
            indices = self.history_indices(i)
            starts += int(indices[-2] is None)
            current_time = row["timing"]["timestamp"]
            for j in indices:
                if j is not None and self.rows[j]["timing"]["timestamp"] > current_time:
                    raise ValueError("future observation entered context")
            terminal += int((row["source"]["episode"], row["source"]["next_step"]) not in self.lookup)
        return {"records_indexed": len(self.rows), "episodes": len(self.manifest["source_episodes"]),
                "episode_reset_contexts": starts, "terminal_image_targets": terminal,
                "history_episode_crossings": 0, "future_rows_in_context": 0,
                "split": "unchanged all-ten-episode train-only prototype"}

    def selected_indices(self):
        left = next(r["source"]["episode"] for r in self.rows if r["task"] == "left")
        right = next(r["source"]["episode"] for r in self.rows if r["task"] == "right")
        first_left = self.lookup[(left, 0)]
        feed_left = next(i for i, r in enumerate(self.rows) if r["source"]["episode"] == left and self.intents[i] == 2 and r["source"]["step"] >= 2)
        feed_right = next(i for i, r in enumerate(self.rows) if r["source"]["episode"] == right and self.intents[i] == 2 and r["source"]["step"] >= 2)
        last_left = max(i for i, r in enumerate(self.rows) if r["source"]["episode"] == left)
        return [first_left, feed_left, feed_right, last_left]

    def observation(self, index, *, future_images=False):
        row = self.rows[index]
        obs = {"observation.state": torch.from_numpy(self.states[index].copy()),
               "task": row["language_instruction"]}
        for view in ("side", "top"):
            path = Path(row[f"observation.images.{view}"])
            if future_images:
                path = path.with_name(f"{row['source']['next_step']:06d}.png")
            path = self.root / "images" / path
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"missing compact-pack image: {path}")
            if bgr.shape != (224, 224, 3):
                raise ValueError("reuse the original square224 preprocessing, without a new crop")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs[f"observation.images.{view}"] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return obs

    def encode_case(self, base, index):
        history = self.history_indices(index)
        visual, states, valid, ages = [], [], [], []
        reference = self.rows[index]["timing"]["timestamp"]
        processed_current = None
        for j in history:
            if j is None:
                visual.append(None)
                states.append(None)
                valid.append(False)
                ages.append(0.0)
                continue
            processed = base.preprocessor(self.observation(j))
            images, _ = base.policy._preprocess_images(processed)
            # The helper is inference_mode: clone OUTSIDE it to create ordinary
            # detached tensors which trainable Linear layers may save backward.
            latent = extract_pi05_multiview_visual_latent(base.policy, images).clone()
            visual.append(latent)
            states.append(processed["observation.state"].clone())
            valid.append(True)
            ages.append(float(self.rows[j]["timing"]["timestamp"] - reference))
            if j == index:
                processed_current = processed
        visual = [torch.zeros_like(visual[-1]) if x is None else x for x in visual]
        states = [torch.zeros_like(states[-1]) if x is None else x for x in states]
        # Only the future IMAGE goes to the frozen image encoder. The copied
        # current-state/task fields are not passed to that encoder or to policy.
        future_obs = self.observation(index, future_images=True)
        future_images = {k: v.unsqueeze(0) for k, v in future_obs.items() if k.startswith("observation.images.")}
        future_images, _ = base.policy._preprocess_images(future_images)
        future_visual = extract_pi05_multiview_visual_latent(base.policy, future_images).clone()
        device = visual[-1].device
        raw_action = torch.from_numpy(self.actions[index].copy()).to(device)
        target_batch = base.preprocessor({
            "observation.state": torch.from_numpy(self.states[index].copy()),
            "task": self.rows[index]["language_instruction"],
            "action": torch.from_numpy(self.actions[index].copy()).unsqueeze(0),
        })
        # The single-observation processor preserves action [1,32]; the
        # dataset trainer supplies [B,1,32]. Both contain one action here.
        if target_batch["action"].numel() != 32:
            raise ValueError("expected exactly one normalized 32D action")
        normalized = target_batch["action"].reshape(32)
        world_action = torch.cat([normalized[:6], raw_action[6:9]])[None]
        return {"index": index, "processed": processed_current,
                "visual": torch.stack(visual, dim=1), "states": torch.stack(states, dim=1),
                "valid": torch.tensor([valid], dtype=torch.bool, device=device),
                "ages": torch.tensor([ages], dtype=torch.float32, device=device),
                "future_visual": future_visual, "action_target": normalized[None, None],
                "world_action": world_action, "history_indices": history}


def context(model, case):
    return model.encode_context(case["visual"], case["states"], case["valid"], case["ages"])


def flow_loss(base, case, noise, flow_time=None):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    processed = case["processed"]
    images, masks = base.policy._preprocess_images(processed)
    losses = base.policy.model.forward(
        images, masks, processed[OBS_LANGUAGE_TOKENS], processed[OBS_LANGUAGE_ATTENTION_MASK],
        case["action_target"], noise=noise,
        time=torch.full((1,), 0.5, device=noise.device) if flow_time is None else flow_time,
        state=processed["observation.state"],
    )
    return losses[..., :3].mean()


@torch.no_grad()
def predict(base, case, injection, prefix, noise):
    base.policy.reset()
    with injection.condition(prefix) if injection is not None else nullcontext():
        action = base.policy.predict_action_chunk(case["processed"], noise=noise.clone())
    translation = (action * base.action_std + base.action_mean)[0, 0, :3]
    probabilities = base.head(None, None, case["processed"]["observation.state"]).float().softmax(-1)[0]
    if not torch.isfinite(translation).all() or not torch.isfinite(probabilities).all():
        raise ValueError("non-finite inference output")
    packet = {"elite_tcp_delta_6d": translation.cpu().tolist() + [0.0, 0.0, 0.0],
              "piper_intent_id": int(probabilities.argmax()),
              "piper_probabilities": probabilities.cpu().tolist(), "hardware_executed": False}
    return action.clone(), packet


def grad_summary(grads):
    active = [g.detach().float() for g in grads if g is not None]
    finite = all(torch.isfinite(g).all().item() for g in active)
    return {"l2": float(torch.stack([g.square().sum() for g in active]).sum().sqrt()) if active else 0.0,
            "nonzero_parameter_tensors": sum(bool(torch.count_nonzero(g)) for g in active),
            "parameter_tensors": len(grads), "finite": finite}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    data = Real10TemporalView(args.pack)
    index_audit = data.audit_index()
    base = Real10PI05Policy(args.training_root / "elite/final_policy.pt",
                            args.training_root / "piper/mixed_head_policy.pt", seed=args.seed, device=args.device)
    print(f"loaded frozen real10 pair in {base.metadata['load_seconds']:.2f}s", flush=True)
    selected = data.selected_indices()
    cases = [data.encode_case(base, i) for i in selected]
    dimension = cases[0]["visual"].shape[-1]
    model = JointTemporalWorld(JointTemporalConfig(visual_dim=dimension, prefix_dim=dimension)).to(args.device).eval()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    noise = torch.randn(1, 1, 32, device=args.device)
    baseline = [predict(base, case, None, None, noise) for case in cases]
    injection = TemporalPrefixInjection(base.policy)
    try:
        disabled = [predict(base, case, injection, None, noise) for case in cases]
        for (before, _), (after, _) in zip(baseline, disabled):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        print("disabled adapter reproduces original actions exactly", flush=True)
        before_predictions = []
        with torch.no_grad():
            for case in cases:
                before_predictions.append(predict(base, case, injection, context(model, case), noise))

        # Episode reset: padded history must be ignored, not borrowed from the
        # previous episode. This also checks masks without another test suite.
        reset_case = cases[0]
        perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in reset_case.items()}
        missing = ~perturbed["valid"]
        perturbed["visual"][missing] = 1000.0
        perturbed["states"][missing] = -1000.0
        perturbed["ages"][missing] = -1000.0
        torch.testing.assert_close(context(model, reset_case), context(model, perturbed), rtol=0, atol=0)

        # Loss targets are absent from the context function. Corrupting a future
        # target must not change either shared prefix or policy prediction.
        case = cases[1]
        future_changed = dict(case, future_visual=case["future_visual"] + torch.randn_like(case["future_visual"]))
        torch.testing.assert_close(context(model, case), context(model, future_changed), rtol=0, atol=0)
        shared = list(model.temporal.parameters())
        prefix = context(model, case)
        prediction = model.predict_future(prefix, case["visual"][:, -1], case["world_action"])
        target_probe = case["future_visual"].detach().clone().requires_grad_(True)
        future_loss = model.future_loss(prediction, target_probe)
        changed_loss = model.future_loss(prediction, future_changed["future_visual"])
        check(float((changed_loss - future_loss).detach().abs()) > 0, "future target does not affect its supervised loss")
        with injection.condition(prefix):
            action_loss = flow_loss(base, case, noise)
            action_grads = torch.autograd.grad(action_loss, shared, retain_graph=True, allow_unused=True)
        future_grads = torch.autograd.grad(future_loss, shared, retain_graph=True, allow_unused=True)
        target_gradient = torch.autograd.grad(future_loss, target_probe, retain_graph=True, allow_unused=True)[0]
        check(target_gradient is None, "future target encoder must not receive gradient")
        action_grad, future_grad = grad_summary(action_grads), grad_summary(future_grads)
        check(action_grad["finite"] and action_grad["l2"] > 0, "PI05 action loss is disconnected from shared temporal parameters")
        check(future_grad["finite"] and future_grad["l2"] > 0, "future loss is disconnected from shared temporal parameters")
        print(f"shared gradient L2: action={action_grad['l2']:.6g}, future={future_grad['l2']:.6g}", flush=True)

        # Check executed-action conditioning without claiming calibrated effects.
        with torch.no_grad():
            alternate = case["world_action"].clone()
            alternate[:, :3] += 1.0
            alternate[:, 6:9] = torch.tensor([0., 1., 0.], device=args.device)
            action_dependence = float((model.predict_future(prefix, case["visual"][:, -1], alternate) - prediction).abs().max())
        check(action_dependence > 0, "world predictor ignores the action input")

        # A disposable FUTURE-ONLY update to shared parameters proves that this
        # auxiliary objective can change what PI05 actually samples. Then reset.
        prefix_before = prefix.detach().clone()
        policy_before = before_predictions[1][0]
        auxiliary_optimizer = torch.optim.AdamW(shared, lr=1e-4, weight_decay=0.0)
        auxiliary_optimizer.zero_grad(set_to_none=True)
        future_loss.backward()
        auxiliary_optimizer.step()
        with torch.no_grad():
            auxiliary_prefix = context(model, case)
            auxiliary_action, _ = predict(base, case, injection, auxiliary_prefix, noise)
        auxiliary_prefix_delta = float((auxiliary_prefix - prefix_before).abs().max())
        auxiliary_action_delta = float((auxiliary_action[..., :3] - policy_before[..., :3]).abs().max())
        check(auxiliary_prefix_delta > 0, "future-only update did not change the shared prefix")
        check(auxiliary_action_delta > 0, "future-only update did not change PI05 translation with fixed noise")
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        del action_loss, future_loss, prediction, prefix, action_grads, future_grads, target_probe

        # One JOINT step in a disposable adapter, not a policy training run.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        prefix = context(model, case)
        with injection.condition(prefix):
            action_loss = flow_loss(base, case, noise)
            prediction = model.predict_future(prefix, case["visual"][:, -1], case["world_action"])
            future_loss = model.future_loss(prediction, case["future_visual"])
            loss = action_loss + 0.1 * future_loss
            loss.backward()
        joint_grad = grad_summary([p.grad for p in model.parameters()])
        check(joint_grad["finite"] and joint_grad["l2"] > 0, "non-finite/zero joint gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        shared_change = max(float((v - initial[k]).abs().max()) for k, v in model.state_dict().items() if k.startswith("temporal."))
        world_change = max(float((v - initial[k]).abs().max()) for k, v in model.state_dict().items() if k.startswith("world."))
        check(shared_change > 0 and world_change > 0, "joint optimizer failed to update both components")
        check(all(not p.requires_grad and p.grad is None for p in base.policy.parameters()), "base policy was unfrozen or received parameter gradients")
        check(all(not p.requires_grad and p.grad is None for p in base.head.parameters()), "Piper head changed trainability")

        examples = []
        with torch.no_grad():
            for n, sample in enumerate(cases):
                tokens = context(model, sample)
                action, packet = predict(base, sample, injection, tokens, noise)
                check(packet["piper_intent_id"] == baseline[n][1]["piper_intent_id"], "Piper intent unexpectedly changed")
                examples.append({"pack_index": sample["index"], "sample_id": data.rows[sample["index"]]["sample_id"],
                                 "task": data.rows[sample["index"]]["task"], "history_indices": sample["history_indices"],
                                 "history_valid": sample["valid"].cpu().tolist(),
                                 "relative_observation_times_s": sample["ages"].cpu().tolist(),
                                 "baseline": baseline[n][1], "initialized_adapter": before_predictions[n][1],
                                 "one_joint_step_adapter": packet,
                                 "recorded_target": data.actions[sample["index"], :6].tolist()})
            disabled_after = predict(base, cases[1], injection, None, noise)[0]
            torch.testing.assert_close(disabled_after, baseline[1][0], rtol=0, atol=0)
        check(injection.active_prefix is None and base.policy.model._project2026_state_context is None,
              "conditioning context leaked after inference")

        checkpoint = {"metadata": model.metadata(), "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "seed": args.seed, "joint_diagnostic_updates": 1, "deployable": False,
                      "base_elite_checkpoint": str((args.training_root / "elite/final_policy.pt").resolve()),
                      "base_piper_checkpoint": str((args.training_root / "piper/mixed_head_policy.pt").resolve())}
        torch.save(checkpoint, args.out / "diagnostic_adapter.pt")
        restored = JointTemporalWorld(model.config).to(args.device).eval()
        restored.load_state_dict(torch.load(args.out / "diagnostic_adapter.pt", weights_only=True, map_location=args.device)["model_state"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(context(restored, case), context(model, case), rtol=0, atol=0)

        report = {"schema": "project2026_real10_joint_temporal_check_v1", "status": "passed",
                  "scope": "four train-set contexts; gradient/interface check, NOT model-effect evaluation",
                  "adapter": model.metadata(), "base_checkpoint": base.metadata, "index_audit": index_audit,
                  "sample_selection": "first left observation, first left feed after step2, first right feed after step2, last left transition",
                  "trainable_parameters": sum(p.numel() for p in model.parameters()),
                  "shared_trainable_parameters": sum(p.numel() for p in model.temporal.parameters()),
                  "world_trainable_parameters": sum(p.numel() for p in model.world.parameters()),
                  "history_shape": list(case["visual"].shape), "shared_prefix_shape": list(tokens.shape),
                  "normalization": "reuse checkpoint state/action stats; per-view LayerNorm for frozen visual features",
                  "feature_image_range": "native PI05 _preprocess_images: RGB [0,1] -> [-1,1]",
                  "losses_before_joint_step": {"action": float(action_loss.detach()), "future": float(future_loss.detach()),
                                               "future_weight": 0.1, "joint": float(loss.detach())},
                  "shared_action_gradient": action_grad, "shared_future_gradient": future_grad, "joint_gradient": joint_grad,
                  "future_only_update": {"optimizer": "AdamW shared temporal params only", "lr": 1e-4,
                                         "restored_before_joint_step": True, "prefix_max_abs_change": auxiliary_prefix_delta,
                                         "normalized_translation_max_abs_change_fixed_noise": auxiliary_action_delta},
                  "one_joint_step_parameter_max_change": {"shared": shared_change, "world": world_change},
                  "action_conditioning_max_prediction_change": action_dependence,
                  "persistence_future_mse_selected_case": float(F.mse_loss(visual_target(case["visual"][:, -1]), visual_target(case["future_visual"]))),
                  "checks": {"disabled_policy_exactly_preserved": True, "base_policy_frozen_no_grad": True,
                             "piper_head_preserved": True, "masked_history_invariant": True,
                             "future_target_not_in_policy_context": True, "future_target_stop_gradient": True,
                             "both_losses_reach_shared_parameters": True, "future_only_update_changes_policy_output": True,
                             "inference_consumes_temporal_tokens": injection.last_extra_tokens == model.config.prefix_tokens,
                             "contexts_cleared": True, "adapter_save_reload_exact": True},
                  "optimizer_steps": {"temporary_future_only_then_restored": 1, "joint_diagnostic_only": 1,
                                      "base_policy": 0, "piper": 0},
                  "sample_predictions": examples, "hardware_executed": False, "full_training_executed": False,
                  "heldout_evaluation": False, "visual_status": "not_viewed",
                  "elapsed_seconds": time.perf_counter() - started}
        check(all(report["checks"].values()), "one or more final checks failed")
        write_json(args.out / "report.json", report)
        print(json.dumps({k: report[k] for k in ("status", "trainable_parameters", "shared_action_gradient", "shared_future_gradient", "future_only_update", "checks", "elapsed_seconds")}, indent=2), flush=True)
        return report
    finally:
        injection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--training-root", type=Path, default=Path("simulation_output/real10_pi05_train_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_joint_temporal_check_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        # Preserve evidence of a failed attempt; do not silently retry/overwrite.
        if args.out.is_dir() and not (args.out / "report.json").exists():
            failure = args.out / "failure.json"
            if not failure.exists():
                write_json(failure, {"status": "failed", "error": repr(error), "hardware_executed": False})
        raise


if __name__ == "__main__":
    main()
