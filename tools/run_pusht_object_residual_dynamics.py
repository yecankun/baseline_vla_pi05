"""Check the residual variant without fitting; train only via the explicit stage.

Reuses the existing cache and optimizer/evaluation loop. A nonzero-decoder
fixture checks the action path; it is untrained, discarded and never a checkpoint.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_object_residual_dynamics as wm
    from . import pusht_object_goal as vision
    from . import run_pusht_object_dynamics as runner
    from .pusht_world_model_adapter import make_candidates
else:
    import pusht_object_residual_dynamics as wm
    import pusht_object_goal as vision
    import run_pusht_object_dynamics as runner
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_OUT = ROOT / "simulation_output/pusht_object_residual_pretrain_v1"


def initial_visual(out, data, model):
    # Reuse the original cache's predetermined TRAIN example, never a failure
    # chosen from the development cases or a validation window.
    selected = data.manifest["visual_source"]
    index = selected["global_index"]
    if index not in data.train:
        raise ValueError("initialization display must be the cached training example")
    current, state, actions, target = data.batch([index])
    with torch.no_grad():
        predicted = model(current, state, actions)
    if not torch.equal(predicted, current[:, None, None].expand_as(predicted)):
        raise ValueError("display lost exact zero-head identity")
    p, y, z = predicted.cpu().numpy()[0, 0], target.cpu().numpy()[0, 0], current.cpu().numpy()[0]
    np.savez_compressed(out / "initial_identity_example.npz", current=z, predicted=p, observed_future=y)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (22, 18))
    sheet = Image.new("RGB", (1170, 750), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "残差预测器训练前检查：零初始化精确复制当前网格，不是学习收益", font=title, fill="black")
    draw.text((12, 47), f"固定原训练片段：episode={selected['episode_index']} / frame={selected['anchor_frame']}；没有参数更新。", font=font, fill="black")
    draw.text((12, 78), "网格白0 / 黑1；未来网格仅为现有监督目标，不进入前向模型。", font=font, fill="black")
    for column, offset in enumerate((0, 1, 4, 8)):
        left = 170 + column * 245
        draw.text((left, 121), "当前" if not offset else f"未来 +{offset}步", font=title, fill="black")
        for row, grid in enumerate((z if not offset else p[offset - 1], z if not offset else y[offset - 1])):
            tile = Image.fromarray(np.rint(255 * (1 - grid)).astype(np.uint8)).convert("RGB").resize((208, 208), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, 165 + row * 265))
    draw.text((12, 237), "零初始化输出", font=font, fill="black")
    draw.text((12, 270), "= 保持当前", font=font, fill="black")
    draw.text((12, 502), "RGB 观测网格", font=font, fill="black")
    draw.text((12, 535), "= 监督目标", font=font, fill="black")
    draw.text((12, 675), "初始输出对动作不敏感是零输出层的设计结果；动作通路另用未训练的非零输出层夹具检查。", font=font, fill="black")
    draw.text((12, 705), "本轮没有训练、验证集预测、策略评估或环境/实机执行，不能据此宣称运动预测改善。", font=font, fill="black")
    sheet.save(out / "initial_identity_zh.png")
    return {**selected, "visual_status": "not_viewed", "model_is_untrained": True}


def check(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    runner.write(out / "started.json", {"plan": wm.PLAN, "status": "checking", "optimizer_steps": 0})
    started = time.monotonic()
    data = wm.ObjectData(cache)
    old_check = json.loads((runner.PRETRAIN_OUT / "report.json").read_text())
    if old_check["status"] != "passed_pretraining_object_grid_check" or old_check["window_counts"] != data.manifest["window_counts"]:
        raise ValueError("requires the original checked cache/window scope")
    torch.manual_seed(wm.PLAN["seed"])
    original_init = wm.base.ObjectGridDynamics(data.stats).cuda()
    torch.manual_seed(wm.PLAN["seed"])
    model = wm.ObjectGridDynamics(data.stats).cuda()
    trunk_same = all(torch.equal(p, dict(original_init.named_parameters())[name]) for name, p in model.named_parameters() if not name.startswith("decode."))
    parameter_count = sum(p.numel() for p in model.parameters())
    assert trunk_same and parameter_count == old_check["parameter_count"]
    del original_init
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    z, state, actions, target = data.batch(data.train[:wm.PLAN["batch_size"]])
    predicted = model(z, state, actions)
    assert predicted.shape == target.shape == (64, 1, 8, 24, 24)
    assert torch.equal(predicted, z[:, None, None].expand_as(predicted))
    loss = model.loss(predicted, target)
    loss.backward()
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    assert np.isfinite(list(gradients.values())).all()
    assert gradients["decode.weight"] > 0 and gradients["decode.bias"] > 0
    # With W_decode=0, upstream gradients on the FIRST backward are correctly
    # zero. Do not misreport that as missing conditioning or trained behavior.
    assert all(value == 0 for name, value in gradients.items() if not name.startswith("decode."))
    model.zero_grad(set_to_none=True)
    boundary_residual = torch.zeros(2, device="cuda", requires_grad=True)
    boundary_current = torch.tensor([0., 1.], device="cuda")
    boundary_target = 1 - boundary_current
    boundary_loss = ((boundary_current + boundary_residual).clamp(0, 1) - boundary_target).square().sum()
    boundary_loss.backward()
    assert torch.equal(boundary_residual.grad, torch.tensor([-2., 2.], device="cuda"))

    # Test fixture only, never an update to the actual initialized model.
    # A zero output layer cannot demonstrate action sensitivity by definition.
    probe = wm.ObjectGridDynamics(data.stats).cuda()
    probe.load_state_dict(model.state_dict())
    generator = torch.Generator(device="cuda").manual_seed(wm.PLAN["seed"] + 1)
    with torch.no_grad():
        probe.decode.weight.copy_(torch.randn(probe.decode.weight.shape, device="cuda", generator=generator) * .001)
    probe_prediction = probe(z, state, actions)
    probe.loss(probe_prediction, target).backward()
    probe_gradients = {name: float(p.grad.norm()) for name, p in probe.named_parameters()}
    assert np.isfinite(list(probe_gradients.values())).all()
    assert probe_gradients["action_embed.weight"] > 0 and probe_gradients["encode.0.weight"] > 0
    with torch.no_grad():
        altered = actions.clone()
        altered[:, :, 4:] += torch.where(altered[:, :, 4:] > 256, -1., 1.)
        different = probe(z, state, altered)
        assert torch.equal(probe_prediction[:, :, :4], different[:, :, :4])
        action_effect = float((probe_prediction[:, :, 4:] - different[:, :, 4:]).abs().max())
        assert action_effect > 0
    del probe
    # In-memory state roundtrip only; no fake final/step0 checkpoint is written.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = wm.ObjectGridDynamics(data.stats).cuda()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    model.eval()
    current = z[:2].cpu().numpy().copy()
    current[1] = 0
    goal = vision.ObjectGridFeatures(data.goal[None].repeat(2, 1, 1).cpu().numpy())
    candidates = make_candidates(actions[:2, 0], offset_xy=8)
    advice = wm.predict_and_score(model, vision.ObjectGridFeatures(current), np.array([True, False]), state[:2], candidates, goal)
    assert advice.selected_index.tolist() == [0, 0]
    assert np.isnan(advice.predicted.values[1]).all() and np.isnan(advice.costs[1]).all()
    allowed = candidates.valid[0].cpu().numpy()
    assert np.all(advice.costs[0, allowed] == advice.costs[0, 0])
    visual = initial_visual(out, data, model)
    report = {"schema": wm.SCHEMA, "status": "passed_pretraining_residual_identity_check", "plan": wm.PLAN,
        "cache": str(cache), "cache_identity": data.identity, "window_counts": data.manifest["window_counts"],
        "parameter_count": parameter_count, "same_trunk_initialization_as_base": trunk_same,
        "real_batch_shape": list(target.shape), "initial_batch_loss_is_persistence_not_quality": float(loss.detach()),
        "initial_prediction_equals_current_bitwise": True, "zero_head_decoder_gradients": {k: v for k, v in gradients.items() if k.startswith("decode.")},
        "first_backward_trunk_gradients_zero_as_designed": True, "clamp_boundary_residual_gradients": boundary_residual.grad.tolist(),
        "nonzero_head_fixture": {"untrained_not_optimizer_updated_not_saved": True,
            "decoder_normal_std": .001, "seed": wm.PLAN["seed"] + 1,
            "action_gradient_norm": probe_gradients["action_embed.weight"], "grid_gradient_norm": probe_gradients["encode.0.weight"],
            "causal_prefix_exact": True, "post_prefix_action_effect_max": action_effect},
        "actual_model_parameters_unchanged": True, "in_memory_state_roundtrip_exact": True,
        "initial_candidate_ties_keep_ACT": True, "invalid_current_retains_ACT_without_forecast": True,
        "cache_schema_unchanged": data.identity["schema"], "checkpoint_schema_distinct_from_base": wm.SCHEMA != wm.base.SCHEMA,
        "validation_model_inference": False, "development_model_inference": False,
        "optimizer_created": False, "optimizer_steps": 0, "checkpoint_created": False,
        "trained_checkpoint_loads": 0, "environment_steps": 0, "hardware_actions": 0,
        "visual_source": visual, "visual_artifact": "initial_identity_zh.png", "visual_status": "not_viewed",
        "elapsed_seconds": time.monotonic() - started, "decision": "variant_ready_training_not_started"}
    runner.write(out / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "parameter_count", "window_counts", "initial_prediction_equals_current_bitwise",
        "zero_head_decoder_gradients", "first_backward_trunk_gradients_zero_as_designed", "nonzero_head_fixture", "optimizer_steps", "elapsed_seconds")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "train"), required=True)
    parser.add_argument("--cache", type=Path, default=wm.CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.stage != "train":
        parser.error("--resume requires explicit training")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    out = args.out or (wm.TRAIN_ROOT if args.stage == "train" else PRETRAIN_OUT)
    try:
        if args.stage == "check":
            check(args.cache, out)
        else:
            checked = json.loads((PRETRAIN_OUT / "report.json").read_text())
            manifest = json.loads((args.cache / "manifest.json").read_text())
            if checked["status"] != "passed_pretraining_residual_identity_check" or checked["plan"] != wm.PLAN:
                raise ValueError("matching residual pretraining check required")
            if any(manifest[key] != value for key, value in checked["cache_identity"].items()):
                raise ValueError("cache changed since the residual pretraining check")
            runner.train(args.cache, out, args.resume, model_api=wm)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
