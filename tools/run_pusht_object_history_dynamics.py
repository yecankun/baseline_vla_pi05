"""Prepare two-frame observable history; fitting requires the explicit train stage.

The check uses cached metadata and training examples only. No optimizer,
validation/development inference, checkpoint or environment is created.
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_object_history_dynamics as wm
    from . import run_pusht_object_dynamics as runner
else:
    import pusht_object_history_dynamics as wm
    import run_pusht_object_dynamics as runner

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_OUT = ROOT / "simulation_output/pusht_object_history_pretrain_v1"


def alignment_summary(data):
    result = {}
    for name, indices in (("train", data.train), ("val", data.val)):
        previous, valid = data.previous_indices[indices], data.history_valid[indices]
        assert np.all(previous[valid] == indices[valid] - 1)
        assert np.all(data.episodes[previous] == data.episodes[indices])
        assert np.all(data.frames[previous[valid]] + 1 == data.frames[indices[valid]])
        assert np.all(data.valid[previous[valid]])
        assert np.array_equal(previous[~valid], indices[~valid])
        device = data.grids.device
        i, p = (torch.as_tensor(x, device=device) for x in (indices, previous))
        changed_grid = (data.grids[i] != data.grids[p]).any(dim=(-2, -1))
        changed_xy = (data.states[i] != data.states[p]).any(dim=-1)
        result[name] = {"windows_unchanged": len(indices), "history_available": int(valid.sum()),
            "missing_reason_counts": {label: int((data.history_reasons[indices] == reason).sum()) for reason, label in enumerate(wm.MISSING_REASONS)},
            "nonzero_observed_grid_delta_windows": int(changed_grid.sum()),
            "nonzero_observed_XY_delta_windows": int(changed_xy.sum()),
            "same_episode_consecutive_past_only": True, "missing_indices_point_to_current": True}
    # Small metadata fixture covers a gap, invalid predecessor and episode reset
    # even if a particular real split contains no examples of one case.
    previous, valid, reasons = wm.history_indices(np.array([0, 0, 0, 0, 1, 1]),
        np.array([0, 1, 3, 4, 0, 1]), np.array([True, True, False, True, True, True]))
    assert previous.tolist() == [0, 0, 2, 3, 4, 4]
    assert valid.tolist() == [False, True, False, False, False, True]
    assert reasons.tolist() == [1, 0, 2, 3, 1, 0]
    return result


def render_training_examples(out, data, model):
    selected = [int(data.train[0]), int(data.manifest["visual_source"]["global_index"])]
    assert all(i in data.train for i in selected)
    z, history, actions, target = data.batch(selected)
    with torch.no_grad():
        predicted = model(z, history, actions)
    assert torch.equal(predicted, z[:, None, None].expand_as(predicted))
    np.savez_compressed(out / "history_examples.npz", indices=np.array(selected),
        previous_indices=data.previous_indices[selected], current=z.cpu().numpy(),
        current_xy=history.current_xy.cpu().numpy(), previous_grid=history.previous_grid.cpu().numpy(),
        previous_xy=history.previous_xy.cpu().numpy(), history_valid=history.valid.cpu().numpy(),
        actions=actions.cpu().numpy(), initial_prediction=predicted.cpu().numpy(), observed_future=target.cpu().numpy())
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (22, 18))
    sheet = Image.new("RGB", (1200, 850), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "双帧历史适配：训练前时序核对；没有优化器更新", font=title, fill="black")
    draw.text((12, 45), "上一帧与当前帧是输入；未来只作监督。初始输出等于当前网格，不是预测能力证据。", font=font, fill="black")
    examples = []
    for row, index in enumerate(selected):
        top = 90 + 340 * row
        available = bool(history.valid[row])
        draw.text((12, top), f"episode={data.episodes[index]}，当前 frame={data.frames[index]}，history_valid={int(available)}", font=title, fill="black")
        labels = ["上一帧 t-1" if available else "缺失历史：复制当前", "当前帧 t", "未训练输出 +8步", "实际 +8步（仅目标）"]
        grids = [history.previous_grid[row], z[row], predicted[row, 0, -1], target[row, 0, -1]]
        for column, (label, grid) in enumerate(zip(labels, grids)):
            left = 12 + 295 * column
            draw.text((left, top + 40), label, font=font, fill="black")
            values = grid.cpu().numpy()
            tile = Image.fromarray(np.rint(255 * (1 - values)).astype(np.uint8)).convert("RGB").resize((192, 192), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, top + 70))
        xy, previous_xy = history.current_xy[row].tolist(), history.previous_xy[row].tolist()
        draw.text((12, top + 276), f"上一 XY=({previous_xy[0]:.2f}, {previous_xy[1]:.2f})；当前 XY=({xy[0]:.2f}, {xy[1]:.2f})。原生坐标差，不解释为物理速度。", font=font, fill="black")
        examples.append({"global_index": index, "episode": int(data.episodes[index]), "frame": int(data.frames[index]),
            "previous_source_index": int(data.previous_indices[index]), "history_valid": available})
    draw.text((12, 795), "网格统一白0 / 黑1；相邻帧缺失、跨 episode 或上一目标不可见时，历史分支关闭。", font=font, fill="black")
    sheet.save(out / "history_alignment_zh.png")
    return examples


def check(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    runner.write(out / "started.json", {"plan": wm.PLAN, "status": "checking", "optimizer_steps": 0})
    started = time.monotonic()
    data = wm.ObjectData(cache)
    alignment = alignment_summary(data)
    old = json.loads((ROOT / "simulation_output/pusht_object_residual_pretrain_v1/report.json").read_text())
    assert old["cache_identity"] == data.identity and old["window_counts"] == data.manifest["window_counts"]
    np.savez_compressed(out / "history_index.npz", previous_indices=data.previous_indices,
        history_valid=data.history_valid, missing_reason=data.history_reasons)
    torch.manual_seed(wm.PLAN["seed"])
    reference = wm.residual.ObjectGridDynamics(data.stats).cuda()
    torch.manual_seed(wm.PLAN["seed"])
    model = wm.ObjectGridDynamics(data.stats).cuda()
    common_initialization_equal = all(torch.equal(value, model.state_dict()[name]) for name, value in reference.state_dict().items())
    count, base_count = (sum(p.numel() for p in x.parameters()) for x in (model, reference))
    assert common_initialization_equal and count - base_count == wm.PLAN["history_projection_parameters"]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    indices = data.train[:64]
    z, history, actions, target = data.batch(indices)
    bz, xy, ba, by = wm.base.ObjectData.batch(data, indices)
    assert torch.equal(z, bz) and torch.equal(history.current_xy, xy) and torch.equal(actions, ba) and torch.equal(target, by)
    features = model.history_features(z, history)
    assert torch.equal(features[~history.valid], torch.zeros_like(features[~history.valid]))
    assert torch.count_nonzero(features[history.valid, :-1]) > 0
    prediction = model(z, history, actions)
    assert prediction.shape == target.shape == (64, 1, 8, 24, 24)
    assert torch.equal(prediction, z[:, None, None].expand_as(prediction))
    loss = model.loss(prediction, target)
    loss.backward()
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    assert np.isfinite(list(gradients.values())).all() and gradients["decode.weight"] > 0
    assert all(value == 0 for name, value in gradients.items() if not name.startswith("decode."))
    model.zero_grad(set_to_none=True)

    # An untrained, unsaved nonzero-decoder fixture exposes the new history path;
    # it is never an optimizer step, checkpoint or alternative initialization.
    probe = wm.ObjectGridDynamics(data.stats).cuda()
    probe.load_state_dict(model.state_dict())
    generator = torch.Generator(device="cuda").manual_seed(wm.PLAN["seed"] + 1)
    with torch.no_grad():
        probe.decode.weight.copy_(torch.randn(probe.decode.weight.shape, device="cuda", generator=generator) * .001)
    pg = history.previous_grid.detach().clone().requires_grad_(True)
    px = history.previous_xy.detach().clone().requires_grad_(True)
    probe_history = wm.make_history(z, xy, previous_grid=pg, previous_xy=px, valid=history.valid)
    p = probe(z, probe_history, actions)
    probe.loss(p, target).backward()
    probe_grad = {"history_projection": float(probe.history_embed.weight.grad.norm()),
        "previous_grid_input": float(pg.grad.norm()), "previous_XY_input": float(px.grad.norm()),
        "action_projection": float(probe.action_embed.weight.grad.norm())}
    assert all(np.isfinite(v) and v > 0 for v in probe_grad.values())
    assert not torch.count_nonzero(pg.grad[~history.valid]) and not torch.count_nonzero(px.grad[~history.valid])
    with torch.no_grad():
        repeated = wm.make_history(z, xy, previous_grid=z, previous_xy=xy, valid=history.valid)
        history_effect = float((p - probe(z, repeated, actions)).abs().max())
        assert history_effect > 0  # same validity bits, only past observation changed
        altered = actions.clone()
        altered[:, :, 4:] += torch.where(altered[:, :, 4:] > 256, -1., 1.)
        changed = probe(z, history, altered)
        assert torch.equal(p[:, :, :4], changed[:, :, :4])
        action_effect = float((p[:, :, 4:] - changed[:, :, 4:]).abs().max())
        assert action_effect > 0
        missing = wm.make_history(z, xy)
        nan_missing = wm.make_history(z, xy, previous_grid=torch.full_like(z, float("nan")),
            previous_xy=torch.full_like(xy, float("nan")), valid=torch.zeros_like(history.valid))
        reference.load_state_dict({k: v for k, v in probe.state_dict().items() if not k.startswith("history_embed.")}, strict=True)
        assert torch.equal(probe(z, missing, actions), reference(z, xy, actions))
        assert torch.equal(probe(z, missing, actions), probe(z, nan_missing, actions))
        h2 = wm.make_history(z[:2], xy[:2], previous_grid=history.previous_grid[:2], previous_xy=history.previous_xy[:2], valid=history.valid[:2])
        assert model(z[:2], h2, actions[:2].expand(-1, 5, -1, -1)).shape == (2, 5, 8, 24, 24)
    del probe, reference
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = wm.ObjectGridDynamics(data.stats).cuda()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    examples = render_training_examples(out, data, model.eval())
    report = {"schema": wm.SCHEMA, "status": "passed_two_frame_history_preparation", "plan": wm.PLAN,
        "cache_identity": data.identity, "window_counts": data.manifest["window_counts"], "alignment": alignment,
        "parameter_count": count, "single_frame_residual_parameter_count": base_count,
        "common_initialization_equal": common_initialization_equal, "original_current_actions_targets_exact": True,
        "history_features_shape": list(features.shape), "initial_forecast_exact_persistence": True,
        "initial_loss_not_quality": float(loss.detach()), "first_backward_decoder_weight_norm": gradients["decode.weight"],
        "first_backward_upstream_zero_by_design": True,
        "untrained_nonzero_decoder_fixture": {"std": .001, "seed": wm.PLAN["seed"] + 1,
            "gradient_norms": probe_grad, "past_input_change_with_same_validity_effect_max": history_effect,
            "masked_past_input_gradients_exact_zero": True, "action_prefix_causal_exact": True,
            "later_action_change_effect_max": action_effect, "history_off_exactly_matches_same_weights_single_frame": True,
            "unavailable_NaN_history_ignored_exactly": True, "optimized": False, "checkpointed": False},
        "in_memory_state_roundtrip_exact": True, "actual_initial_model_parameters_unchanged": True,
        "optimizer_created": False, "optimizer_steps": 0, "checkpoint_created": False,
        "trained_checkpoint_loads": 0, "validation_model_inference": False, "development_model_inference": False,
        "environment_steps": 0, "hardware_actions": 0, "source_cache_modified": False,
        "visual_examples": examples, "visual_artifact": "history_alignment_zh.png", "visual_status": "not_viewed",
        "runtime_seconds": time.monotonic() - started,
        "decision": "history_adapter_ready_training_not_started_no_performance_claim"}
    runner.write(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "parameter_count", "alignment", "untrained_nonzero_decoder_fixture", "optimizer_steps", "runtime_seconds")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "train"), required=True)
    parser.add_argument("--cache", type=Path, default=wm.CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.stage != "train":
        parser.error("--resume is only for explicitly approved training")
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
            if checked["status"] != "passed_two_frame_history_preparation" or checked["plan"] != wm.PLAN:
                raise ValueError("matching successful history preparation required")
            if any(manifest[k] != v for k, v in checked["cache_identity"].items()):
                raise ValueError("source cache changed since preparation")
            runner.train(args.cache, out, args.resume, model_api=wm)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
