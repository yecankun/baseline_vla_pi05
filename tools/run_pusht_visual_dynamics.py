"""Prepare frozen ACT features/goal, check one real batch, or train visual dynamics.

Preparation/check: zero optimizer updates and environment steps. Train is a
separate explicit stage, fixed10000 updates, final-only validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

if __package__:
    from . import pusht_visual_dynamics as wm
else:
    import pusht_visual_dynamics as wm


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(cache: Path):
    if __package__:
        from . import pusht_bc_act_training_data as dm
        from . import pusht_bc_act_inference as inference
        from .pusht_world_model_adapter import FrozenACTPrior
    else:
        import pusht_bc_act_training_data as dm
        import pusht_bc_act_inference as inference
        from pusht_world_model_adapter import FrozenACTPrior
    from PIL import Image, ImageDraw, ImageFont

    cache.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = dm.load_training_data()
    print(f"Loaded {len(data)} pinned frames in {time.monotonic() - started:.1f}s", flush=True)
    policy, binding = inference.load_final("act")
    prior = FrozenACTPrior(policy, binding)
    features = np.empty((len(data), 9, 512), dtype=np.float32)
    for start in range(0, len(data), 256):
        indices = np.arange(start, min(start + 256, len(data)))
        observation = data.batch(indices)["observation"]
        features[indices] = prior.encode_image(observation["observation.image"]).values.cpu().numpy()
        if start % (256 * 20) == 0:
            print(f"Encoded {indices[-1] + 1}/{len(data)} frozen image features", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite frozen features")
    episodes, frames = data.episode_indices, data.frame_indices
    train, val = wm.temporal_anchors(episodes, data.split)
    assert len(train) == 22000 and len(val) == 2002
    for anchors in (train, val):
        assert np.all(episodes[anchors, None] == episodes[anchors[:, None] + np.arange(9)[None]])
        assert np.all(frames[anchors + 8] == frames[anchors] + 8)
    train_features = features[data.train_indices].astype(np.float64)
    visual_mean = train_features.mean(axis=(0, 1))
    raw_std = train_features.std(axis=(0, 1))
    visual_std = np.maximum(raw_std, 0.001)
    del train_features
    normalization = {"visual_mean": visual_mean.tolist(), "visual_std": visual_std.tolist(),
                     "state_mean": data.stats["observation.state"]["mean"], "state_std": data.stats["observation.state"]["std"],
                     "action_mean": data.stats["action"]["mean"], "action_std": data.stats["action"]["std"]}
    # Deterministic, declared before viewing any goals/validation: episode1's last record.
    goal_episode = min(data.split["train_episodes"])
    goal_indices = np.flatnonzero(episodes == goal_episode)
    goal_index = int(goal_indices[-1])
    goal_rgb = data._images[goal_index]
    Image.fromarray(goal_rgb).save(cache / "goal_rgb.png")
    selected = [int(goal_indices[0]), int(goal_indices[len(goal_indices) // 2]), goal_index]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
    sheet = Image.new("RGB", (1200, 518), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"训练集 episode {goal_episode}：固定目标来源；未读取验证集来选择目标", fill="black", font=font)
    for column, index in enumerate(selected):
        label = ("首帧", "中间帧", "末帧：目标候选")[column]
        draw.text((column * 400 + 12, 50), f"{label}｜frame={frames[index]}", fill="black", font=font)
        sheet.paste(Image.fromarray(data._images[index]).resize((384, 384)), (column * 400 + 8, 86))
    draw.text((12, 480), "图像来源：原始训练视频；不是环境真值渲染，也不是新策略执行结果。", fill="black", font=font)
    sheet.save(cache / "goal_source_zh.png")
    np.save(cache / "features.npy", features, allow_pickle=False)
    np.savez(cache / "arrays.npz", states=data._states, actions=data._actions, episodes=episodes, frames=frames)
    manifest = {
        "schema": wm.SCHEMA, "status": "prepared", "plan": wm.PLAN,
        "source_revision": data.binding["revision"], "source_root": data.binding["source_root"],
        "split_sha256": data.binding["split_sha256"], "split": data.split,
        "feature_space_id": prior.space_id, "act_checkpoint_path": binding["checkpoint_path"],
        "feature_shape": list(features.shape), "feature_bytes": features.nbytes,
        "normalization": normalization, "normalization_source": "all23488_training_frames_only",
        "visual_std_floor": 0.001, "channels_below_std_floor": int((raw_std < 0.001).sum()),
        "train_windows": len(train), "val_windows": len(val), "temporal_mapping": "z[t], s[t], a[t:t+8] -> z[t+1:t+9]",
        "fps": 10, "horizon_seconds": 0.8, "cross_episode_windows": 0, "padding": "none; incomplete windows excluded",
        "numeric_columns_read": list(dm.NUMERIC_COLUMNS), "goal": {"episode_index": goal_episode,
            "frame_index": int(frames[goal_index]), "global_index": goal_index, "split": "train",
            "source": "last_recorded_rgb_of_lowest_training_episode", "filename": "goal_rgb.png",
            "validation_goal_selection": False, "privileged_metadata_used": False, "visual_status": "not_viewed"},
        "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0, "environment_steps": 0,
        "user_visual_acceptance": False,
    }
    write_json(cache / "manifest.json", manifest)
    print(json.dumps({"cache": str(cache), "train_windows": len(train), "val_windows": len(val),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def check(cache: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(wm.PLAN["seed"])
    data = wm.FeatureData(cache)
    model = wm.SpatialDynamics(data.stats, data.space_id, wm.PLAN["hidden_dim"]).cuda()
    z, state, actions, target = data.batch(data.train[:wm.PLAN["batch_size"]])
    initial = model(z, state, actions)
    assert initial.shape == target.shape == (64, 1, 8, 9, 512)
    loss = model.loss(initial, target)
    loss.backward()
    gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()}
    assert np.isfinite(list(gradients.values())).all() and gradients["action_embed.weight"] > 0
    model.zero_grad(set_to_none=True)
    # Future action must not affect earlier predictions; use real inputs, no synthetic targets.
    with torch.no_grad():
        changed = actions.clone()
        changed[:, :, 4:] += 1
        prediction = model(z, state, changed)
        assert torch.equal(initial[:, :, :4], prediction[:, :, :4])
        action_effect = float((initial[:, :, 4:] - prediction[:, :, 4:]).abs().max())
        assert action_effect > 0
    # Real forward/backward timing only. No optimizer is created or stepped here.
    torch.cuda.synchronize()
    measured = time.monotonic()
    for _ in range(20):
        model.zero_grad(set_to_none=True)
        model.loss(model(z, state, actions), target).backward()
    torch.cuda.synchronize()
    seconds_per_forward_backward = (time.monotonic() - measured) / 20
    model.zero_grad(set_to_none=True)
    # Candidate scoring connects to this actual (untrained) model; no action is dispatched.
    if __package__:
        from .pusht_world_model_adapter import VisualFeatures, make_candidates, score_visual_goal
    else:
        from pusht_world_model_adapter import VisualFeatures, make_candidates, score_visual_goal
    candidates = make_candidates(actions[:1, 0], offset_xy=8)
    model.eval()
    predicted = model.predict_candidates(VisualFeatures(z[:1], data.space_id), state[:1], candidates)
    score = score_visual_goal(candidates, predicted, VisualFeatures(data.goal[None], data.space_id))
    edge = make_candidates(torch.full_like(actions[:1, 0], 511), offset_xy=8)
    edge_prediction = model.predict_candidates(VisualFeatures(z[:1], data.space_id), state[:1], edge)
    assert torch.isnan(edge_prediction.values[~edge.valid]).all()
    edge_score = score_visual_goal(edge, edge_prediction, VisualFeatures(data.goal[None], data.space_id))
    assert torch.isfinite(edge_score.costs[edge.valid]).all() and torch.isinf(edge_score.costs[~edge.valid]).all()
    torch.save({"model": model.state_dict(), "cache_identity": data.identity}, out / "initial_reload_check.pt")
    restored = wm.SpatialDynamics(data.stats, data.space_id, wm.PLAN["hidden_dim"]).cuda().eval()
    restored.load_state_dict(torch.load(out / "initial_reload_check.pt", weights_only=True)["model"])
    with torch.no_grad():
        assert torch.equal(model(z, state, actions), restored(z, state, actions))
    report = {"schema": wm.SCHEMA, "status": "passed_pretraining_real_batch_check", "plan": wm.PLAN,
        "cache": str(cache), "parameter_count": sum(p.numel() for p in model.parameters()),
        "real_batch_shape": list(target.shape), "initial_training_batch_loss_not_quality_result": float(loss.detach()),
        "action_gradient_norm": gradients["action_embed.weight"], "causal_prefix_exact": True,
        "action_input_effect_max_at_random_init_not_learned_use": action_effect,
        "checkpoint_reload_exact": True, "future_prediction_goal_score_connected": True,
        "untrained_candidate_costs_not_action_recommendation": [None if not np.isfinite(x) else x for x in score.costs[0].tolist()],
        "seconds_per_forward_backward": seconds_per_forward_backward,
        "estimated_train_seconds_with_50percent_overhead": seconds_per_forward_backward * wm.PLAN["steps"] * 1.5,
        "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0, "environment_steps": 0,
        "policy_benefit_evaluated": False, "trained_checkpoint_created": False}
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def train(cache: Path, out: Path, resume: bool):
    plan = wm.PLAN
    torch.manual_seed(plan["seed"])
    torch.backends.cudnn.benchmark = False
    data = wm.FeatureData(cache)
    model = wm.SpatialDynamics(data.stats, data.space_id, plan["hidden_dim"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
    generator = torch.Generator().manual_seed(plan["seed"])
    start_step, previous_seconds = 0, 0.0
    if resume:
        if (out / "report.json").exists():
            raise ValueError("run already completed; do not retrain or select another checkpoint")
        run = json.loads((out / "run.json").read_text())
        payload = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if run["plan"] != plan or payload["plan"] != plan or payload["cache_identity"] != data.identity:
            raise ValueError("resume plan/cache does not match")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["sampler_rng"].cpu())
        start_step, previous_seconds = payload["step"], payload["elapsed_seconds"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run.json", {"plan": plan, "cache": str(cache), "cache_identity": data.identity,
                                     "checkpoint_selection": "final10000_only", "execution_stage": "train"})
    started = time.monotonic()

    def save(step, kind="resume"):
        payload = {"schema": wm.SCHEMA, "kind": kind, "plan": plan, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler_rng": generator.get_state(),
            "cache_identity": data.identity, "elapsed_seconds": previous_seconds + time.monotonic() - started}
        destination = out / ("final.pt" if kind == "final" else "last.pt")
        torch.save(payload, destination.with_suffix(".pt.tmp"))
        os.replace(destination.with_suffix(".pt.tmp"), destination)

    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Stop requested; finish current update and save last.pt", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not resume:
        save(0)
    model.train()
    accumulated = 0.0
    for step in range(start_step + 1, plan["steps"] + 1):
        indices = data.train[torch.randint(len(data.train), (plan["batch_size"],), generator=generator).numpy()]
        z, state, actions, target = data.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(z, state, actions), target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}; retain preceding last.pt")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        accumulated += float(loss.detach())
        if step % plan["log_every"] == 0:
            row = {"step": step, "loss_last_interval": accumulated / min(plan["log_every"], step - start_step),
                   "grad_norm": float(grad_norm), "session_seconds": time.monotonic() - started}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            accumulated = 0.0
        if step % plan["checkpoint_every"] == 0 or stop or step == plan["steps"]:
            save(step)
            write_json(out / "status.json", {"status": "interrupted" if stop else "training", "step": step})
        if stop:
            return
    save(plan["steps"], kind="final")
    validation = wm.evaluate(model, data)
    loaded = wm.load_final(out / "final.pt")
    z, state, actions, _ = data.batch(data.val[:2])
    with torch.no_grad():
        assert torch.equal(model(z, state, actions), loaded(z, state, actions))
    observed = validation["metrics"]["observed_actions"]["window_mean"]
    persistence = validation["metrics"]["persistence"]["window_mean"]
    report = {"schema": wm.SCHEMA, "status": "completed_fixed_visual_dynamics_training", "plan": plan,
        "cache_identity": data.identity, "validation": validation,
        "observed_minus_persistence": {key: {"absolute": observed[key] - persistence[key],
            "relative_percent": 100 * (observed[key] - persistence[key]) / persistence[key] if persistence[key] else None} for key in observed},
        "final_checkpoint": str(out / "final.pt"), "checkpoint_reload_exact": True,
        "elapsed_seconds": previous_seconds + time.monotonic() - started, "optimizer_steps": plan["steps"],
        "environment_steps": 0, "policy_benefit_evaluated": False, "goal_used_to_train_or_select_model": False,
        "decision": "inspect_prediction_and_action_ablation_before_any_policy_rollout"}
    write_json(out / "report.json", report)
    write_json(out / "status.json", {"status": "completed", "step": plan["steps"]})
    print(json.dumps({"status": report["status"], "metrics": observed, "out": str(out)}, indent=2), flush=True)


def inspect_prediction(cache: Path, checkpoint: Path, out: Path):
    """One predetermined held-out window; original RGB plus spatial feature error.

    No RGB prediction is fabricated and no policy rollout/optimization occurs.
    """
    import av
    from PIL import Image, ImageDraw, ImageFont
    if __package__:
        from .pusht_bc_act_training_data import check_video_frame
    else:
        from pusht_bc_act_training_data import check_video_frame
    out.mkdir(parents=True, exist_ok=False)
    data = wm.FeatureData(cache)
    model = wm.load_final(checkpoint)
    if model.space_id != data.space_id:
        raise ValueError("inspection cache and checkpoint feature spaces differ")
    episode = int(min(data.manifest["split"]["val_episodes"]))
    anchors = data.val[data.episodes[data.val] == episode]
    index = int(anchors[len(anchors) // 2])
    z, state, actions, target = data.batch([index])
    with torch.inference_mode():
        prediction = model(z, state, actions)[:, 0, -1]
        errors = {"model": (prediction - target[:, 0, -1]).square().mean(-1)[0].cpu().numpy(),
                  "persistence": (z - target[:, 0, -1]).square().mean(-1)[0].cpu().numpy()}
    video = Path(data.manifest["source_root"]) / "videos/observation.image/chunk-000/file-000.mp4"
    selected = {index, index + 8}
    images = {}
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in selected:
                images[i] = check_video_frame(frame, i, len(data.episodes))
            if len(images) == 2:
                break
    if set(images) != selected:
        raise ValueError("selected source RGB frames are missing")
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 18)
    sheet = Image.new("RGB", (1440, 498), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"固定验证样本：episode {episode} 中点窗口｜训练完成后的视觉特征预测检查", fill="black", font=font)
    titles = (f"当前观测：frame {data.frames[index]}", f"8步后的真实观测：{data.frames[index + 8]}",
              "模型：末帧逐格特征 MSE", "保持当前特征：逐格 MSE")
    for column, title in enumerate(titles):
        draw.text((column * 360 + 12, 48), title, fill="black", font=small)
    for column, i in enumerate((index, index + 8)):
        sheet.paste(Image.fromarray(images[i]).resize((324, 324)), (column * 360 + 12, 82))
    maximum = max(float(x.max()) for x in errors.values())
    for column, values in enumerate(errors.values(), start=2):
        for patch, value in enumerate(values):
            x, y = column * 360 + 12 + (patch % 3) * 108, 82 + (patch // 3) * 108
            ratio = float(value) / max(maximum, 1e-12)
            color = (255, int(250 - 145 * ratio), int(235 - 180 * ratio))
            draw.rectangle((x, y, x + 108, y + 108), fill=color, outline="white", width=2)
            draw.text((x + 10, y + 43), f"{value:.4f}", fill="black", font=small)
        draw.text((column * 360 + 12, 414), f"该窗口均值：{values.mean():.6f}", fill="black", font=small)
    draw.text((12, 446), "两张误差图共用量程；颜色越深误差越大。右侧不是预测图像，也不是像素误差。", fill="black", font=small)
    draw.text((12, 472), "固定单窗口仅用于检查，不代替完整20个 episode 的评估；环境步数与新增优化步数均为0。", fill="black", font=small)
    sheet.save(out / "prediction_inspection_zh.png")
    report = {"schema": wm.SCHEMA, "status": "fixed_window_inspected", "checkpoint": str(checkpoint),
        "selection": "middle_full_window_of_lowest_ID_validation_episode", "episode": episode,
        "global_index": index, "current_frame": int(data.frames[index]), "future_frame": int(data.frames[index + 8]),
        "terminal_per_patch_raw_feature_mse": {k: v.tolist() for k, v in errors.items()},
        "shared_color_scale_max": maximum, "future_validation_rgb_used_as_task_goal": False,
        "rgb_prediction_decoder_available": False, "optimizer_steps": 0, "environment_steps": 0,
        "visual_status": "not_viewed"}
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare-check", "check", "train", "inspect"), required=True)
    parser.add_argument("--cache", type=Path, default=wm.CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=wm.TRAIN_ROOT / "final.pt")
    args = parser.parse_args()
    if args.resume and args.stage != "train":
        parser.error("--resume is only for training")
    default_out = {"train": wm.TRAIN_ROOT, "inspect": Path("simulation_output/pusht_visual_dynamics_inspection_v1")}
    out = args.out or default_out.get(args.stage, Path("simulation_output/pusht_visual_dynamics_pretrain_v1"))
    if args.stage == "prepare-check":
        prepare(args.cache)
    if args.stage in ("prepare-check", "check"):
        check(args.cache, out)
    elif args.stage == "inspect":
        inspect_prediction(args.cache, args.checkpoint, out)
    else:
        train(args.cache, out, args.resume)


if __name__ == "__main__":
    main()
