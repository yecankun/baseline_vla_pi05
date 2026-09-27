"""Prepare/check an RGB-object predictor, or explicitly train it later.

prepare-check never constructs an optimizer or changes model parameters.
The separate train stage is fixed10000 updates and final-only evaluation.
No ACT/old-WM checkpoint, environment, hardware or development coverage input.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_object_dynamics as wm
    from . import pusht_object_goal as vision
else:
    import pusht_object_dynamics as wm
    import pusht_object_goal as vision

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_OUT = ROOT / "simulation_output/pusht_object_dynamics_pretrain_v1"
OLD_CACHE = Path("/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_act_wm_v1")


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(cache):
    # Reuse the existing pinned image/state/action loader and its provenance
    # checks. Do not add new per-file hashes or read diagnostic/coverage columns.
    if __package__:
        from . import pusht_bc_act_training_data as dm
    else:
        import pusht_bc_act_training_data as dm
    cache.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(cache / "started.json", {"plan": wm.PLAN, "status": "preparing", "optimizer_steps": 0})
    data = dm.load_training_data()
    old = json.loads((OLD_CACHE / "manifest.json").read_text())
    if data.split != old["split"] or data.binding["revision"] != old["source_revision"]:
        raise ValueError("source/split differs from fixed BC/ACT/old-WM protocol")
    goal_index = old["goal"]["global_index"]
    if (old["goal"]["episode_index"], old["goal"]["frame_index"], goal_index, old["goal"]["split"]) != (1, 117, 278, "train"):
        raise ValueError("goal source changed")
    with Image.open(OLD_CACHE / "goal_rgb.png") as image:
        if not np.array_equal(data._images[goal_index], np.asarray(image.convert("RGB"))):
            raise ValueError("fixed goal RGB differs from pinned source decode")
    print(f"Loaded {len(data)} pinned RGB frames; no checkpoint loaded", flush=True)
    counts = np.zeros((len(data), 24, 24), dtype=np.uint8)
    valid, pixels = np.zeros(len(data), dtype=bool), np.zeros(len(data), dtype=np.int32)
    for index, image in enumerate(data._images):
        observed = vision.observe_rgb(image)
        counts[index] = np.rint(observed.features.values[0] * 16).astype(np.uint8)
        valid[index], pixels[index] = observed.valid, observed.visible_pixels
    if not valid[goal_index]:
        raise ValueError("fixed training goal not visible under frozen parser")
    ep, fr = data.episode_indices, data.frame_indices
    anchors, window_counts = wm.full_windows(ep, fr, valid, data.split)
    if window_counts["train"]["full_windows"] != 22000 or window_counts["val"]["full_windows"] != 2002:
        raise ValueError("original complete-window schedule changed")
    split_counts = {}
    for name in ("train", "val"):
        rows = np.isin(ep, data.split[name + "_episodes"])
        split_counts[name] = {"frames": int(rows.sum()), "valid_frames": int(valid[rows].sum()),
                             "invalid_frames": int((~valid[rows]).sum()),
                             "episodes": {str(int(e)): {"frames": int((ep == e).sum()),
                                 "valid_frames": int(valid[ep == e].sum()),
                                 "eligible_windows": int((ep[anchors[name]] == e).sum())} for e in data.split[name + "_episodes"]}}
    normalization = {f"{kind}_{key}": data.stats[source][key] for kind, source in (("state", "observation.state"), ("action", "action")) for key in ("mean", "std")}
    np.save(cache / "grid_counts.npy", counts, allow_pickle=False)
    np.savez(cache / "arrays.npz", states=data._states, actions=data._actions, episodes=ep,
             frames=fr, valid=valid, visible_pixels=pixels)
    Image.fromarray(data._images[goal_index]).save(cache / "goal_rgb.png")
    with (cache / "invalid_frames.jsonl").open("w", encoding="utf-8") as stream:
        for index in np.flatnonzero(~valid):
            stream.write(json.dumps({"global_index": int(index), "episode_index": int(ep[index]),
                "frame_index": int(fr[index]), "valid": False, "reason": "frozen_RGB_component_rule_failed_no_truth_fallback"}) + "\n")
    manifest = {"schema": wm.SCHEMA, "status": "prepared", "plan": wm.PLAN,
        "representation": vision.PLAN, "source_revision": data.binding["revision"],
        "source_root": data.binding["source_root"], "split": data.split,
        "normalization": normalization, "normalization_refit_after_filter": False,
        "goal": old["goal"], "window_counts": window_counts, "frame_counts": split_counts,
        "cache_layout": "uint8_N24x24_counts0..16_convert_to_float32_div16",
        "grid_count_bytes": counts.nbytes, "RGB_frames": len(data), "valid_RGB_frames": int(valid.sum()),
        "metadata_in_forward": False, "numeric_columns_read": list(dm.NUMERIC_COLUMNS),
        "policy_inputs": ["current_RGB_derived_grid", "observable_agentXY", "candidate_native_XY_actions"],
        "supervision": "subsequent_RGB_derived_grids_only", "fps": 10, "horizon_seconds": .8,
        "optimizer_steps": 0, "environment_steps": 0, "trained_checkpoint_loads": 0,
        "elapsed_seconds": time.monotonic() - started}
    # Predetermined lowest-ID training episode's middle eligible window; no
    # search by model loss or development outcomes. This visual is targets only.
    selected = anchors["train"][ep[anchors["train"]] == min(data.split["train_episodes"])]
    if not len(selected):
        raise ValueError("no eligible window in the fixed visual-review training episode")
    anchor = int(selected[len(selected) // 2])
    sheet = Image.new("RGB", (1220, 570), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19)
    draw.text((12, 10), "物体网格训练接口：固定训练片段；只有当前观测进入模型，后续帧仅作监督", font=font, fill="black")
    for column, offset in enumerate((0, 1, 4, 8)):
        index, left = anchor + offset, column * 305 + 12
        label = "当前输入" if offset == 0 else f"未来 +{offset} 步：监督目标"
        draw.text((left, 49), label, font=font, fill="black")
        draw.text((left, 78), f"episode={ep[index]} / frame={fr[index]}", font=font, fill="black")
        sheet.paste(Image.fromarray(data._images[index]).resize((208, 208)), (left, 112))
        image = Image.fromarray(np.uint8(255 - counts[index].astype(np.float32) * 255 / 16))
        sheet.paste(image.resize((208, 208), Image.Resampling.NEAREST), (left, 328))
    draw.text((12, 542), "下排是 RGB 规则提取的可见物体，不是仿真真值、模型预测或实机结果。", font=font, fill="black")
    sheet.save(cache / "training_window_targets_zh.png")
    manifest["visual_source"] = {"episode_index": int(ep[anchor]), "anchor_frame": int(fr[anchor]),
                                 "global_index": anchor, "offsets": [0, 1, 4, 8], "visual_status": "not_viewed"}
    write(cache / "manifest.json", manifest)
    print(json.dumps({"window_counts": window_counts, "valid_RGB_frames": int(valid.sum()),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def check(cache, out):
    if __package__:
        from .pusht_world_model_adapter import make_candidates
    else:
        from pusht_world_model_adapter import make_candidates
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(out / "started.json", {"plan": wm.PLAN, "optimizer_steps": 0, "status": "checking"})
    torch.manual_seed(wm.PLAN["seed"])
    data = wm.ObjectData(cache)
    model = wm.ObjectGridDynamics(data.stats).cuda()
    z, state, actions, target = data.batch(data.train[:wm.PLAN["batch_size"]])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    predicted = model(z, state, actions)
    assert predicted.shape == target.shape == (64, 1, 8, 24, 24)
    loss = model.loss(predicted, target)
    loss.backward()
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    assert np.isfinite(list(gradients.values())).all()
    assert gradients["action_embed.weight"] > 0 and gradients["encode.0.weight"] > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        altered = actions.clone()
        altered[:, :, 4:] += torch.where(altered[:, :, 4:] > 256, -1., 1.)
        different = model(z, state, altered)
        assert torch.equal(predicted[:, :, :4], different[:, :, :4])
        action_effect = float((predicted[:, :, 4:] - different[:, :, 4:]).abs().max())
        assert action_effect > 0
    torch.cuda.synchronize()
    measured = time.monotonic()
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        model.loss(model(z, state, actions), target).backward()
    torch.cuda.synchronize()
    seconds_per_backward = (time.monotonic() - measured) / 5
    model.zero_grad(set_to_none=True)
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    # No optimizer/checkpoint is created. Roundtrip the initial state in RAM.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = wm.ObjectGridDynamics(data.stats).cuda().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(z[:2], state[:2], actions[:2]), restored(z[:2], state[:2], actions[:2]))
    # One real training input plus an explicitly unavailable observation fixture.
    current = z[:2].cpu().numpy().copy()
    current[1] = 0
    goal = vision.ObjectGridFeatures(data.goal[None].repeat(2, 1, 1).cpu().numpy())
    candidates = make_candidates(actions[:2, 0], offset_xy=8)
    advice = wm.predict_and_score(model, vision.ObjectGridFeatures(current), np.array([True, False]), state[:2], candidates, goal)
    assert advice.selected_index[1] == 0 and np.isnan(advice.predicted.values[1]).all() and np.isnan(advice.costs[1]).all()
    assert np.isfinite(advice.costs[0][candidates.valid[0].cpu().numpy()]).all()
    edge = make_candidates(torch.full_like(actions[:1, 0], 511), offset_xy=8)
    edge_advice = wm.predict_and_score(model, vision.ObjectGridFeatures(current[:1]), np.array([True]), state[:1], edge,
                                     vision.ObjectGridFeatures(goal.values[:1]))
    assert np.isnan(edge_advice.predicted.values[~edge.valid.cpu().numpy()]).all()
    assert np.isinf(edge_advice.costs[~edge.valid.cpu().numpy()]).all()
    actual_terminal = target[:, 0, -1].detach()
    goal_batch = data.goal[None].expand(len(z), -1, -1)
    fixture = np.broadcast_to(actual_terminal[:, None, None].cpu().numpy(), (len(z), 5, 8, 24, 24))
    numpy_cost = vision.score_candidate_grids(vision.ObjectGridFeatures(fixture),
        vision.ObjectGridFeatures(goal_batch.cpu().numpy()), np.ones((len(z), 5), dtype=bool)).costs[:, 0]
    torch_cost = wm.dice_cost(actual_terminal, goal_batch).cpu().numpy()
    assert np.allclose(numpy_cost, torch_cost, atol=1e-7, rtol=1e-6)
    report = {"schema": wm.SCHEMA, "status": "passed_pretraining_object_grid_check", "plan": wm.PLAN,
        "cache": str(cache), "window_counts": data.manifest["window_counts"],
        "parameter_count": sum(p.numel() for p in model.parameters()), "real_batch_shape": list(target.shape),
        "initial_training_batch_loss_not_quality": float(loss.detach()), "gradients_finite": True,
        "action_gradient_norm": gradients["action_embed.weight"], "grid_encoder_gradient_norm": gradients["encode.0.weight"],
        "causal_action_prefix_exact": True, "action_effect_at_random_init_not_learned_use": action_effect,
        "parameters_unchanged": True, "initial_state_roundtrip_exact": True, "checkpoint_created": False,
        "invalid_current_retains_reference_without_forecast": True, "invalid_candidate_mask_passed": True,
        "torch_numpy_goal_score_match": True, "validation_model_inference": False,
        "seconds_per_forward_backward": seconds_per_backward,
        "estimated_train_seconds_50percent_overhead": seconds_per_backward * wm.PLAN["steps"] * 1.5,
        "optimizer_created": False, "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
        "old_model_checkpoint_loads": 0, "policy_benefit_evaluated": False,
        "visual_artifact": str(cache / "training_window_targets_zh.png"), "visual_status": "not_viewed",
        "elapsed_seconds": time.monotonic() - started, "decision": "implementation_ready_training_not_started"}
    write(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def train(cache, out, resume, *, model_api=wm):
    """Explicit future stage; safe interruption saves optimizer and sampler state."""
    # The default keeps the original model; an explicit variant reuses this
    # exact optimizer/sampler/checkpoint loop with its own versioned model API.
    plan = model_api.PLAN
    data = model_api.ObjectData(cache)
    torch.manual_seed(plan["seed"])
    model = model_api.ObjectGridDynamics(data.stats).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
    sampler = torch.Generator().manual_seed(plan["seed"])
    start_step, previous_seconds = 0, 0.
    if resume:
        if (out / "report.json").exists():
            raise ValueError("completed training must not be resumed")
        payload = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if payload["plan"] != plan or payload["cache_identity"] != data.identity:
            raise ValueError("resume plan/cache mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        sampler.set_state(payload["sampler_rng"].cpu())
        start_step, previous_seconds = payload["step"], payload["elapsed_seconds"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / "run.json", {"plan": plan, "cache_identity": data.identity, "stage": "explicit_train",
                                  "checkpoint_selection": "final10000_only"})
    started, stop = time.monotonic(), False

    def save(step, kind="resume"):
        path = out / ("final.pt" if kind == "final" else "last.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"schema": model_api.SCHEMA, "kind": kind, "plan": plan, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
            "cache_identity": data.identity, "elapsed_seconds": previous_seconds + time.monotonic() - started}, temporary)
        os.replace(temporary, path)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Stop requested; finish current update and save last.pt", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not resume:
        save(0)
    model.train()
    loss_sum, interval_count = 0., 0
    for step in range(start_step + 1, plan["steps"] + 1):
        indices = data.train[torch.randint(len(data.train), (plan["batch_size"],), generator=sampler).numpy()]
        z, state, actions, target = data.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(z, state, actions), target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}; preceding last.pt retained")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        loss_sum, interval_count = loss_sum + float(loss.detach()), interval_count + 1
        if step % plan["log_every"] == 0:
            row = {"step": step, "loss_interval_mean": loss_sum / interval_count, "grad_norm": float(grad),
                   "session_seconds": time.monotonic() - started}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            loss_sum, interval_count = 0., 0
        if step % plan["checkpoint_every"] == 0 or stop or step == plan["steps"]:
            save(step)
            write(out / "status.json", {"status": "interrupted" if stop else "training", "step": step})
        if stop:
            return
    save(plan["steps"], "final")
    validation = model_api.evaluate(model, data)
    loaded = model_api.load_final(out / "final.pt")
    z, state, actions, _ = data.batch(data.val[:2])
    with torch.no_grad():
        assert torch.equal(model(z, state, actions), loaded(z, state, actions))
    observed = validation["metrics"]["observed_actions"]["window_mean"]
    reference = validation["metrics"]["persistence"]["window_mean"]
    report = {"schema": model_api.SCHEMA, "status": "completed_fixed_object_dynamics_training", "plan": plan,
        "cache_identity": data.identity, "validation": validation, "final_checkpoint": str(out / "final.pt"),
        "observed_minus_persistence": {key: {"absolute": observed[key] - reference[key],
            "relative_percent": 100 * (observed[key] - reference[key]) / reference[key] if reference[key] else None} for key in observed},
        "checkpoint_reload_exact": True, "optimizer_steps": plan["steps"], "environment_steps": 0,
        "hardware_actions": 0, "goal_used_to_train_or_select_model": False, "policy_benefit_evaluated": False,
        "elapsed_seconds": previous_seconds + time.monotonic() - started,
        "decision": "inspect_prediction_and_candidate_ranking_before_any_policy_switch"}
    write(out / "report.json", report)
    write(out / "status.json", {"status": "completed", "step": plan["steps"]})
    print(json.dumps({"status": report["status"], "metrics": observed, "out": str(out)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare-check", "check", "train"), required=True)
    parser.add_argument("--cache", type=Path, default=wm.CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.stage != "train":
        parser.error("--resume is only for explicitly requested training")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    out = args.out or (wm.TRAIN_ROOT if args.stage == "train" else PRETRAIN_OUT)
    try:
        if args.stage == "prepare-check":
            prepare(args.cache)
        if args.stage in ("prepare-check", "check"):
            check(args.cache, out)
        else:
            train(args.cache, out, args.resume)
    except BaseException:
        # Do not overwrite an existing successful artifact or auto-retry.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
