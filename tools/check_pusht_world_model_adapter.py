"""One short ACT/visual-goal interface check; zero training and environment steps.

Loads the existing ACT final checkpoint. Scoring uses explicitly synthetic
future features and an identity-image goal, NOT a trained world model/task goal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

if __package__:
    from . import pusht_bc_act_inference as inference
    from . import pusht_bc_act_models as models
    from . import pusht_world_model_adapter as adapter
    from .smoke_lerobot_public_envs import _make_environment
else:
    import pusht_bc_act_inference as inference
    import pusht_bc_act_models as models
    import pusht_world_model_adapter as adapter
    from smoke_lerobot_public_envs import _make_environment


def must_reject(call, reason):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(reason)


def scoring_fixtures(candidates, goal):
    b, k, h, _ = candidates.actions.shape
    assert b == 1 and bool(candidates.valid.all()), "reset fixture requires all five valid candidates"
    g = goal.values
    # Fabricated offsets produce analytically known costs 4,1,9,0.25,16.
    delta = g.new_tensor((2, 1, 3, 0.5, 4)).reshape(1, k, 1, 1, 1)
    predicted = adapter.VisualFeatures(g[:, None, None].expand(b, k, h, *g.shape[1:]) + delta, goal.space_id)
    result = adapter.score_visual_goal(candidates, predicted, goal)
    torch.testing.assert_close(result.costs, g.new_tensor([[4, 1, 9, 0.25, 16]]), rtol=1e-5, atol=1e-6)
    assert result.selected_index.item() == 3
    assert torch.equal(result.selected_chunk, candidates.actions[:, 3])
    assert torch.equal(result.first_action, candidates.actions[:, 3, 0])
    identity = adapter.VisualFeatures(g[:, None, None].expand_as(predicted.values).clone(), goal.space_id)
    assert adapter.score_visual_goal(candidates, identity, goal).selected_index.item() == 0
    # Only the terminal frame counts; preceding predictions do not alter this objective.
    earlier_changed = predicted.values.clone()
    earlier_changed[:, :, :-1] += 100
    assert torch.equal(result.costs, adapter.score_visual_goal(
        candidates, adapter.VisualFeatures(earlier_changed, goal.space_id), goal).costs)
    weights = torch.zeros_like(g[:, :, 0])
    weights[:, 0] = 1
    masked = predicted.values.clone()
    masked[:, :, -1, 1:] += 100
    torch.testing.assert_close(result.costs, adapter.score_visual_goal(
        candidates, adapter.VisualFeatures(masked, goal.space_id), goal, goal_patch_weights=weights).costs)
    edge = adapter.make_candidates(torch.full_like(candidates.actions[:, 0], 511), offset_xy=8)
    assert edge.valid.tolist() == [[True, False, True, False, True]]
    invalid_winner = identity.values.clone()
    invalid_winner[:, [0, 2, 4]] += 2
    invalid_winner[:, 2] -= 1
    edge_result = adapter.score_visual_goal(edge, adapter.VisualFeatures(invalid_winner, goal.space_id), goal)
    assert edge_result.selected_index.item() == 2 and bool(torch.isinf(edge_result.costs[:, [1, 3]]).all())
    must_reject(lambda: adapter.score_visual_goal(candidates, predicted,
                adapter.VisualFeatures(g, "incompatible_dino_or_libero_space")), "feature mismatch accepted")
    must_reject(lambda: adapter.make_candidates(torch.full_like(candidates.actions[:, 0], 513), offset_xy=8),
                "invalid reference accepted")
    must_reject(lambda: adapter.score_visual_goal(candidates, predicted, goal, goal_patch_weights=weights * 0),
                "empty goal mask accepted")
    return {"source": "synthetic_feature_offsets_not_world_model_predictions", "costs": result.costs.tolist(),
            "selected_index": result.selected_index.tolist(), "expected_selected_index": [3],
            "reference_tie_break": True, "terminal_only": True, "goal_patch_mask": True,
            "out_of_bounds_candidates_rejected_without_clipping": True,
            "feature_space_mismatch_rejected": True, "no_candidate_executed": True}


def draw_candidates(raw, candidates, out):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 20)
    title = ImageFont.truetype(str(font_path), 25)
    canvas = Image.new("RGB", (1240, 552), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), "冻结 ACT → 候选动作接口：只画目标坐标，不执行动作", font=title, fill="black")
    draw.text((16, 46), "同一初始观测｜候选 0 保持原样；其余四组逐步偏移，末步偏移 8 个坐标单位", font=font, fill="black")
    frame = Image.fromarray(raw["pixels"]).resize((384, 384))
    canvas.paste(frame, (16, 114))
    canvas.paste(frame, (420, 114))
    draw.text((16, 83), "原始 RGB（身份目标夹具也是此图）", font=font, fill="black")
    draw.text((420, 83), "五组动作目标；不是预测运动轨迹", font=font, fill="black")
    colors = ("#111827", "#dc2626", "#2563eb", "#16a34a", "#9333ea")
    labels = ("0：ACT 原始候选", "1：X 正向偏移", "2：X 负向偏移", "3：Y 正向偏移", "4：Y 负向偏移")
    for index, chunk in enumerate(candidates.actions[0].cpu().tolist()):
        points = [(420 + x * 384 / 512, 114 + y * 384 / 512) for x, y in chunk]
        draw.line(points, fill=colors[index], width=2)
        x, y = points[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[index])
        draw.text((825, 128 + index * 35), labels[index], font=font, fill=colors[index])
    draw.text((825, 329), "评分检查：使用人工特征夹具", font=font, fill="black")
    draw.text((825, 362), "不代表世界模型预测或任务目标", font=font, fill="black")
    draw.text((825, 395), "环境步数：0；优化步数：0", font=font, fill="black")
    draw.text((16, 512), "仅验证接口：尚未获得匹配的视觉动力学模型，也没有策略成功率增益结论。", font=font, fill="black")
    path = out / "candidate_interface_zh.png"
    canvas.save(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulation_output/pusht_act_wm_interface_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    from lerobot.envs.utils import preprocess_observation

    policy, binding = inference.load_final("act")
    prior = adapter.FrozenACTPrior(policy, binding)
    config, vector = _make_environment("pusht", argparse.Namespace(episode_length=300))
    try:
        raw, _ = vector.envs[0].reset(seed=100000)  # info and simulator internals are not consumed.
        obs = inference.observation_from_native(raw, preprocess_observation)
        request = prior.prepare(obs, offset_xy=8)
        assert policy.action_calls == 0 and len(policy.policy._action_queue) == 0
        candidates, current = request["candidates"], request["current_visual"]
        assert torch.equal(request["agent_pos"], obs[models.STATE])
        # Verify actual ACT preprocessing equals the independent image-only goal encoder path.
        processed = policy.pre(models.act_batch(obs))[models.IMAGE]
        torch.testing.assert_close(processed, models.imagenet_normalize(obs[models.IMAGE]), rtol=1e-6, atol=1e-6)
        # Queue preservation with a partially consumed queue (the adapter never resets it).
        emitted = [policy.select_action(obs).clone()]
        queue_before = [a.clone() for a in policy.policy._action_queue]
        again = prior.candidates(obs, offset_xy=8)
        assert policy.action_calls == 1 and len(policy.policy._action_queue) == 7
        assert all(torch.equal(a, b) for a, b in zip(queue_before, policy.policy._action_queue))
        assert torch.equal(again.actions, candidates.actions)
        emitted += [policy.select_action(obs).clone() for _ in range(7)]
        torch.testing.assert_close(torch.stack(emitted, dim=1), candidates.actions[:, 0], rtol=0, atol=0)
        # The only goal here is a declared identity-image fixture; not a task-success image.
        goal = prior.encode_image(obs[models.IMAGE])
        assert torch.equal(current.values, goal.values) and bool(torch.isfinite(current.values).all())
        fixtures = scoring_fixtures(candidates, goal)
        must_reject(lambda: prior.prepare(dict(obs, coverage=1), offset_xy=8), "oracle extra accepted")
        assert all(not p.requires_grad and p.grad is None for p in policy.policy.parameters())
        figure = draw_candidates(raw, candidates, args.out)
        report = {
            "schema": adapter.SCHEMA, "status": "passed_interface_only", "elapsed_seconds": time.monotonic() - start,
            "seed": 100000, "checkpoint_path": binding["checkpoint_path"],
            "model_state_sha256": binding["model_state_sha256"], "feature_space_id": prior.space_id,
            "current_feature_shape": list(current.values.shape), "expected_prediction_shape": [1, 5, 8, *current.values.shape[1:]],
            "policy_observation_keys": list(obs), "world_model_input_keys": list(request),
            "action_units": "absolute native XY targets, inclusive 0..512; not deltas or velocity", "offset_xy": 8,
            "candidate_shape": list(candidates.actions.shape), "candidate_valid": candidates.valid.tolist(),
            "candidate_labels": list(candidates.labels), "candidate_actions": candidates.actions.tolist(),
            "reference_first8_matches_baseline_queue_exactly": True, "partial_baseline_queue_unchanged": True,
            "goal_encoder_matches_act_visual_preprocessing": True, "goal_source": "current_rgb_identity_fixture_only",
            "visual_feature_source": "frozen_ACT_ResNet18_spatial_map_not_DINO", "scoring_fixtures": fixtures,
            "trained_world_model_loaded": False, "world_model_training_steps": 0, "optimizer_steps": 0,
            "environment_steps": 0, "hardware_actions": 0, "policy_benefit_evaluated": False,
            "figure": figure, "visual_status": "not_viewed",
        }
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "current_feature_shape", "candidate_shape", "environment_steps")}, indent=2))
    finally:
        vector.close()


if __name__ == "__main__":
    main()
