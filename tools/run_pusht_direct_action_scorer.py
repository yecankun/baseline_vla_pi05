"""DS0 real-batch preparation, or a separately authorized fixed training run.

--stage check: no optimizer, checkpoint file, validation inference or environment.
--stage train: explicit fixed10000 updates; --resume only for an interrupted run.
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
    from . import pusht_direct_action_scorer as ds
    from .run_pusht_object_dynamics import write
    from .pusht_world_model_adapter import make_candidates
else:
    import pusht_direct_action_scorer as ds
    from run_pusht_object_dynamics import write
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_OUT = ROOT / "simulation_output/pusht_direct_scorer_pretrain_v1"


def target_sheet(data, out):
    """Same preselected training window as the old cache; targets, not forecasts."""
    i = data.manifest["visual_source"]["global_index"]
    if i not in data.train:
        raise ValueError("visual example must be an unchanged eligible training window")
    current, state, actions, future = data.batch([i])
    y = ds.observed_effect(current, future, data.goal)
    d0 = float(ds.base.dice_cost(current[0], data.goal))
    d8 = float(ds.base.dice_cost(future[0, 0, -1], data.goal))
    sheet = Image.new("RGB", (1120, 480), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19)
    draw.text((16, 12), "DS0 监督对齐：固定训练片段，不是模型预测或策略效果", font=font, fill="black")
    for column, (label, grid) in enumerate((("当前观测 t：模型输入", current[0]),
                                           ("固定训练 goal：模型输入", data.goal),
                                           ("真实 t+8：仅用于监督", future[0, 0, -1]))):
        x = 16 + column * 368
        draw.text((x, 57), label, font=font, fill="black")
        image = Image.fromarray(np.rint(255 * (1 - grid.cpu().numpy())).astype(np.uint8))
        sheet.paste(image.resize((264, 264), Image.Resampling.NEAREST), (x, 94))
    draw.text((16, 374), f"episode={data.episodes[i]} / frame={data.frames[i]}；动作 t..t+7 对应后继 t+8", font=font, fill="black")
    draw.text((16, 410), f"当前代价 {d0:.6f} - 后继代价 {d8:.6f} = 监督改善量 {float(y):+.6f}", font=font, fill="black")
    draw.text((16, 444), "正值更接近视觉目标；不是 coverage、成功概率或导丝引导能力。", font=font, fill="black")
    sheet.save(out / "supervision_alignment_zh.png")
    np.savez_compressed(out / "target_example.npz", current=current.cpu().numpy(), state=state.cpu().numpy(),
                        actions=actions.cpu().numpy(), terminal=future[:, 0, -1].cpu().numpy(),
                        goal=data.goal.cpu().numpy(), target_effect=y.cpu().numpy())
    return {"episode": int(data.episodes[i]), "frame": int(data.frames[i]), "global_index": int(i),
            "target_effect": float(y), "current_cost": d0, "terminal_cost": d8}


def check(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(out / "started.json", {"plan": ds.PLAN, "stage": "check", "optimizer_steps": 0})
    data = ds.base.ObjectData(cache)
    stats = ds.training_target_stats(data)
    torch.manual_seed(ds.PLAN["seed"])
    model = ds.DirectActionScorer(data.stats, stats, data.goal).cuda()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    indices = data.train[:ds.PLAN["batch_size"]]
    current, state, actions, future = data.batch(indices)
    y = ds.observed_effect(current, future, data.goal)
    assert y.shape == (64, 1)
    np.testing.assert_array_equal(future[:, 0, -1].cpu().numpy(), data.grids[indices + ds.HORIZON].cpu().numpy())
    # Independent NumPy calculation uses the existing scorer, not the new target helper.
    now, terminal = current.cpu().numpy(), future[:, 0, -1].cpu().numpy()
    # Duplicated terminal placeholders satisfy the old five-by-eight API;
    # these are score-formula fixtures, not fabricated rollout labels.
    endpoints = np.stack((now, terminal, now, now, now), axis=1)
    grids = np.broadcast_to(endpoints[:, :, None], (64, 5, 8, 24, 24))
    goals = np.broadcast_to(data.goal.cpu().numpy(), (64, 24, 24)).copy()
    costs = ds.vision.score_candidate_grids(ds.vision.ObjectGridFeatures(grids),
        ds.vision.ObjectGridFeatures(goals), np.ones((64, 5), dtype=bool)).costs
    np.testing.assert_allclose(y[:, 0].cpu().numpy(), costs[:, 0] - costs[:, 1], atol=1e-7, rtol=1e-6)
    a = actions.detach().clone().requires_grad_(True)
    z = current.detach().clone().requires_grad_(True)
    xy = state.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    measured = time.monotonic()
    predicted = model(z, xy, a)
    loss = model.loss(predicted, y)
    loss.backward()
    torch.cuda.synchronize()
    batch_seconds = time.monotonic() - measured
    gradient_norms = {key: float(p.grad.norm()) for key, p in model.named_parameters()}
    assert np.isfinite(list(gradient_norms.values())).all()
    assert all(value > 0 for value in gradient_norms.values())
    action_step_norms = a.grad.abs().sum((0, 1, 3)).cpu().tolist()
    assert all(value > 0 for value in action_step_norms) and float(z.grad.norm()) > 0 and float(xy.grad.norm()) > 0
    model.zero_grad(set_to_none=True)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    model.eval()
    candidates = make_candidates(actions[:2, 0], offset_xy=8)
    advice = ds.predict_and_score(model, ds.vision.ObjectGridFeatures(current[:2].cpu().numpy()),
                                 np.array([True, False]), state[:2], candidates)
    assert advice.selected_index[1] == 0 and np.isnan(advice.scores[1]).all()
    assert torch.equal(advice.selected_chunk[1], actions[1, 0])
    # Bounds/tie fixture exercises selection only, never a new policy rollout.
    edge = make_candidates(torch.full_like(actions[:1, 0], 511), offset_xy=8)
    tied = ds.select_candidates(np.zeros((1, 5)), edge, np.array([True]))
    assert tied.selected_index[0] == 0 and np.isneginf(tied.scores[~edge.valid.cpu().numpy()]).all()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = ds.DirectActionScorer(data.stats, stats, data.goal).cuda().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(model(current[:2], state[:2], actions[:2]), restored(current[:2], state[:2], actions[:2]))
    perfect = ds.summarize_effects(y[:, 0].cpu().numpy(), y[:, 0].cpu().numpy(), data.episodes[indices])
    assert perfect["all"]["window_mean"]["effect_mae"] == 0
    example = target_sheet(data, out)
    report = {"schema": ds.SCHEMA, "status": "passed_direct_scorer_pretraining_check", "plan": ds.PLAN,
        "cache": str(cache), "cache_identity": data.identity, "target_stats": stats,
        "parameter_count": sum(p.numel() for p in model.parameters()), "real_batch_shape": list(predicted.shape),
        "initial_training_batch_loss_not_quality": float(loss.detach()), "initialization": "fresh_random",
        "gradients_finite_nonzero": True, "grid_gradient_norm": float(z.grad.norm()),
        "state_gradient_norm": float(xy.grad.norm()), "eight_action_step_gradient_l1": action_step_norms,
        "numpy_target_match": True, "terminal_index_exact": True, "parameters_and_buffers_unchanged": True,
        "in_memory_roundtrip_exact": True, "invalid_current_retains_ACT": True,
        "bounds_mask_and_exact_tie_fixture_passed": True, "metric_perfect_prediction_fixture_passed": True,
        "target_example": example, "visual_artifact": str(out / "supervision_alignment_zh.png"),
        "visual_status": "not_viewed", "one_cold_forward_backward_seconds": batch_seconds,
        "runtime_is_not_steady_state_training_or_policy_latency": True,
        "optimizer_created": False, "optimizer_steps": 0, "disk_checkpoint_created": False,
        "validation_model_inference": False, "saved_development_candidates_read": False,
        "source_dataset_decode": False, "environment_steps": 0, "hardware_actions": 0,
        "elapsed_seconds": time.monotonic() - started, "decision": "implementation_ready_training_not_started"}
    write(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def train(cache, out, resume):
    """Fresh output only, or safe resume from this run's last optimizer state."""
    if (out / "report.json").exists() or (out.exists() and not resume):
        raise FileExistsError(f"preserve existing run: {out}")
    prepared = json.loads((PRETRAIN_OUT / "report.json").read_text())
    data = ds.base.ObjectData(cache)
    stats = ds.training_target_stats(data)
    if (prepared["status"] != "passed_direct_scorer_pretraining_check" or prepared["plan"] != ds.PLAN
            or prepared["cache_identity"] != data.identity or prepared["target_stats"] != stats):
        raise ValueError("matching completed real-batch preparation is required")
    torch.manual_seed(ds.PLAN["seed"])
    model = ds.DirectActionScorer(data.stats, stats, data.goal).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ds.PLAN["lr"], weight_decay=ds.PLAN["weight_decay"])
    sampler = torch.Generator().manual_seed(ds.PLAN["seed"])
    first, previous_seconds = 0, 0.
    if resume:
        payload = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if (payload["schema"] != ds.SCHEMA or payload["plan"] != ds.PLAN
                or payload["cache_identity"] != data.identity or payload["target_stats"] != stats):
            raise ValueError("resume plan/cache/target statistics mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        sampler.set_state(payload["sampler_rng"].cpu())
        first, previous_seconds = payload["step"], payload["elapsed_seconds"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / "run.json", {"plan": ds.PLAN, "cache_identity": data.identity, "target_stats": stats,
                                  "stage": "explicit_train", "checkpoint_selection": "final10000_only"})
    started, stop = time.monotonic(), False

    def save(step, kind="resume"):
        path = out / ("final.pt" if kind == "final" else "last.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"schema": ds.SCHEMA, "kind": kind, "plan": ds.PLAN, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
            "cache_identity": data.identity, "target_stats": stats,
            "elapsed_seconds": previous_seconds + time.monotonic() - started}, temporary)
        os.replace(temporary, path)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Stop requested; finish this update and save last.pt", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not resume:
        save(0)
    model.train()
    loss_sum, count = 0., 0
    for step in range(first + 1, ds.PLAN["steps"] + 1):
        indices = data.train[torch.randint(len(data.train), (ds.PLAN["batch_size"],), generator=sampler).numpy()]
        current, state, actions, future = data.batch(indices)
        target = ds.observed_effect(current, future, data.goal)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(current, state, actions), target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}; previous last.pt retained")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), ds.PLAN["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        loss_sum, count = loss_sum + float(loss.detach()), count + 1
        if step % ds.PLAN["log_every"] == 0:
            row = {"step": step, "loss_interval_mean": loss_sum / count, "grad_norm": float(grad),
                   "resume_from_step": first, "session_seconds": time.monotonic() - started}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            loss_sum, count = 0., 0
        if step % ds.PLAN["checkpoint_every"] == 0 or stop or step == ds.PLAN["steps"]:
            save(step)
            write(out / "status.json", {"status": "interrupted" if stop else "training", "step": step})
        if stop:
            return
    save(ds.PLAN["steps"], "final")
    validation = ds.evaluate(model, data)
    loaded = ds.load_final(out / "final.pt")
    z, state, actions, _ = data.batch(data.val[:2])
    with torch.no_grad():
        assert torch.equal(model(z, state, actions), loaded(z, state, actions))
    report = {"schema": ds.SCHEMA, "status": "completed_fixed_direct_scorer_training", "plan": ds.PLAN,
        "cache_identity": data.identity, "target_stats": stats, "validation": validation,
        "parameter_count": sum(p.numel() for p in model.parameters()), "final_checkpoint": str(out / "final.pt"),
        "checkpoint_reload_exact": True, "optimizer_steps": ds.PLAN["steps"], "environment_steps": 0,
        "hardware_actions": 0, "goal_used_to_train_model": True, "validation_for_checkpoint_selection": False,
        "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False,
        "elapsed_seconds": previous_seconds + time.monotonic() - started,
        "decision": "prediction_diagnostic_only_keep_ACT_pending_separate_selection_evidence"}
    write(out / "report.json", report)
    write(out / "status.json", {"status": "completed", "step": ds.PLAN["steps"]})
    print(json.dumps({"status": report["status"], "out": str(out),
                      "validation": validation["metrics"]["direct_scorer"]["all"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "train"), required=True)
    parser.add_argument("--cache", type=Path, default=ds.base.CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.stage != "train":
        parser.error("--resume requires --stage train")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    out = args.out or (ds.TRAIN_ROOT if args.stage == "train" else PRETRAIN_OUT)
    try:
        if args.stage == "check":
            check(args.cache, out)
        else:
            train(args.cache, out, args.resume)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
