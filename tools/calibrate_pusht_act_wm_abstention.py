"""Training-only residual gate calibration and one offline check; no new rollout.

No threshold sweep or checkpoint selection. Known20/100 benchmark logs are not
opened. Existing loaders materialize a shared train/val cache, but only declared
training indices reach prediction, calibration and the fixed-window diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

if __package__:
    from . import pusht_act_wm_abstention as gate
    from . import pusht_visual_dynamics as wm
    from . import pusht_bc_act_inference as inference
    from . import pusht_bc_act_training_data as dm
    from .pusht_world_model_adapter import CandidateScores, FrozenACTPrior, VisualFeatures, make_candidates, score_visual_goal
else:
    import pusht_act_wm_abstention as gate
    import pusht_visual_dynamics as wm
    import pusht_bc_act_inference as inference
    import pusht_bc_act_training_data as dm
    from pusht_world_model_adapter import CandidateScores, FrozenACTPrior, VisualFeatures, make_candidates, score_visual_goal


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()), "min": float(values.min()),
            "median": float(np.median(values)), "q95_higher": float(np.quantile(values, .95, method="higher")),
            "max": float(values.max())}


@torch.inference_mode()
def calibrate(data, model, out):
    if len(data.train) != 22000 or len(np.unique(data.episodes[data.train])) != 186:
        raise ValueError("training-window population differs from the declared plan")
    if np.intersect1d(data.train, data.val).size:
        raise ValueError("calibration includes validation windows")
    predicted_costs, recorded_costs = [], []
    started = time.monotonic()
    for start in range(0, len(data.train), 64):
        z, state, actions, target = data.batch(data.train[start:start + 64])
        predicted = model.predict(VisualFeatures(z, data.space_id), state, actions).values
        predicted_costs.append((predicted[:, 0, -1] - data.goal).square().mean((-2, -1)).cpu())
        recorded_costs.append((target[:, 0, -1] - data.goal).square().mean((-2, -1)).cpu())
    predicted, recorded = (torch.cat(values).numpy() for values in (predicted_costs, recorded_costs))
    residual = np.abs(predicted.astype(np.float64) - recorded.astype(np.float64))
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite training residuals")
    q95 = float(np.quantile(residual, gate.PLAN["quantile"], method=gate.PLAN["quantile_method"]))
    episodes = data.episodes[data.train]
    np.savez(out / "training_residuals.npz", indices=data.train, episodes=episodes,
             predicted_goal_cost=predicted, recorded_goal_cost=recorded, absolute_error=residual)
    artifact = {
        "status": "completed_training_residual_calibration", "plan": gate.PLAN,
        "source_identity": {key: data.manifest[key] for key in ("source_revision", "split_sha256", "feature_space_id", "goal")},
        "wm_checkpoint": str(wm.TRAIN_ROOT / "final.pt"), "residual_q95": q95,
        "threshold": gate.PLAN["multiplier"] * q95, "windows": len(residual), "episodes": 186,
        "residual_stats": stats(residual),
        "per_episode": {str(int(ep)): stats(residual[episodes == ep]) for ep in np.unique(episodes)},
        "aggregation": "one_equal_weight_per_overlapping_window_not_independent_samples",
        "calibration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
        "independent_calibration": False, "counterfactual_error_bound_validated": False,
    }
    # Freeze this artifact before any ACT candidate diagnostic; it is never retuned.
    write_json(out / "calibration.json", artifact)
    return artifact


def gate_fixtures():
    """One small logic check. Fabricated scores are NOT measured model results."""
    candidates = make_candidates(torch.full((4, 8, 2), 256.0), offset_xy=8)
    costs = torch.tensor([[1., 1., 1., 1., 1.], [1., .875, 2., 2., 2.],
                          [1., .5, 2., 2., 2.], [1., .25, 2., 2., 2.]])
    best = costs.argmin(1)
    scored = CandidateScores(costs, best, candidates.actions[torch.arange(4), best])
    result = gate.retain_reference(candidates, scored, .5)
    assert result.scores.selected_index.tolist() == [0, 0, 0, 1]
    assert torch.equal(result.scores.selected_chunk[:3], candidates.actions[:3, 0])
    assert torch.equal(result.scores.selected_chunk[3], candidates.actions[3, 1])
    return {"source": "fabricated_cost_logic_only_not_model_scores",
            "tie_below_and_equal_threshold_keep_reference": True, "strictly_above_accepts_original_candidate": True}


@torch.inference_mode()
def inspect_training_windows(data, raw, selector, out):
    if raw.split != data.manifest["split"] or raw.binding["revision"] != data.manifest["source_revision"]:
        raise ValueError("raw observation source differs from feature cache")
    chosen = []
    for ep in sorted(data.manifest["split"]["train_episodes"]):
        anchors = data.train[data.episodes[data.train] == ep]
        chosen.append(int(anchors[len(anchors) // 2]))
    if len(chosen) != 186 or np.intersect1d(chosen, raw.val_indices).size:
        raise ValueError("offline observation selection is not training-only")
    rows, first_example = [], None
    started = time.monotonic()
    # Single observations match the actual executor's inference batch/precision.
    for index in chosen:
        obs = raw.batch(np.array([index], dtype=np.int64))["observation"]
        selector.reset()
        action = selector.select_action(obs)
        record = {"global_index": index, "episode": int(data.episodes[index]), "frame": int(data.frames[index]),
                  **selector.last_decision, "native_action": action[0].tolist(),
                  "max_abs_reference_offset": selector.last_trace["max_abs_reference_offset"]}
        assert record["accepted"] == (record["greedy_index"] != 0 and record["predicted_gain"] > selector.threshold)
        if not record["accepted"]:
            assert record["selected_chunk"] == record["reference_chunk"]
        rows.append(record)
        if first_example is None:
            first_example = (raw._images[index].copy(), record, obs)
    with (out / "training_decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    selector.reset()
    return rows, first_example, time.monotonic() - started


@torch.inference_mode()
def queue_check(selector, obs):
    """Real frozen-model17 calls on one fixed training observation, no env step."""
    act = selector.prior.policy
    act.reset()
    expected = [act.select_action(obs).clone() for _ in range(17)]
    selector.reset()
    actual, decisions = [], []
    for _ in range(17):
        action = selector.select_action(obs)
        actual.append(action.clone())
        if selector.last_decision is not None:
            decisions.append(selector.last_decision)
        chunk_action = decisions[-1]["selected_chunk"][(selector.action_calls - 1) % 8]
        assert action[0].tolist() == chunk_action
    all_reference = all(row["selected_index"] == 0 for row in decisions)
    exact = all(torch.equal(a, b) for a, b in zip(expected, actual))
    if all_reference:
        assert exact
    assert selector.decisions == 3 and len(selector.queue) == 7
    selector.reset()
    assert len(selector.queue) == 0 and len(selector.reference_queue) == 0 and selector.action_calls == 0
    return {"action_calls": 17, "decisions": 3, "selected_indices": [r["selected_index"] for r in decisions],
            "selected_chunk_queue_exact": True, "all_reference": all_reference,
            "original_ACT_first17_exact": exact, "partial_queue_reset": True, "environment_steps": 0}


def draw_example(pixels, record, threshold, out):
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, n) for n in (24, 20))
    canvas = Image.new("RGB", (1224, 610), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "ACT 保留规则：固定首条训练轨迹的中间窗口（零环境步）", font=title, fill="black")
    draw.text((12, 46), f'episode={record["episode"]} / frame={record["frame"]}；坐标线是动作目标，不是物体预测轨迹', font=font, fill="black")
    prior = torch.tensor(record["reference_chunk"], dtype=torch.float32)[None]
    candidates = make_candidates(prior, offset_xy=gate.PLAN["offset_xy"])
    variants = [("原始训练 RGB", None, "black"),
                (f'原贪心选择：候选 {record["greedy_index"]}', candidates.actions[0, record["greedy_index"]].tolist(), "#dc2626"),
                (f'护栏选择：候选 {record["selected_index"]}', record["selected_chunk"], "#2563eb")]
    for column, (label, chunk, color) in enumerate(variants):
        left, top = column * 408 + 12, 113
        draw.text((left, 82), label, font=font, fill=color)
        canvas.paste(Image.fromarray(pixels).resize((384, 384)), (left, top))
        if chunk is not None:
            points = [(left + x * 384 / 512, top + y * 384 / 512) for x, y in chunk]
            draw.line(points, fill=color, width=3)
            x, y = points[-1]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text((12, 516), f'预测收益 {record["predicted_gain"]:.7g}；训练误差尺度阈值 {threshold:.7g}；'
              + ("允许改选" if record["accepted"] else "保留 ACT"), font=font, fill="black")
    draw.text((12, 552), "训练内诊断，不是置信保证或成功率结果；未执行新策略，未使用测试场景调阈值。", font=font, fill="black")
    canvas.save(out / "training_gate_example_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulation_output/pusht_act_wm_abstention_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "plan": gate.PLAN, "optimizer_steps": 0, "environment_steps": 0,
              "hardware_actions": 0, "benchmark_logs_opened": False, "validation_predictions": 0}
    write_json(args.out / "started.json", report)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(20260915)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = wm.FeatureData()
        model = wm.load_final(wm.TRAIN_ROOT / "final.pt")
        if model.space_id != data.space_id:
            raise ValueError("world model/cache feature space mismatch")
        artifact = calibrate(data, model, args.out)
        print(json.dumps({"stage": "threshold_frozen_before_candidate_inspection",
                          "residual_q95": artifact["residual_q95"], "threshold": artifact["threshold"]}), flush=True)
        del model
        report["gate_fixtures"] = gate_fixtures()
        act, binding = inference.load_final("act")
        prior = FrozenACTPrior(act, binding)
        selector = gate.load_selector(prior, args.out / "calibration.json")
        raw = dm.load_training_data()
        rows, example, seconds = inspect_training_windows(data, raw, selector, args.out)
        report["queue_check"] = queue_check(selector, example[2])
        accepted = sum(row["accepted"] for row in rows)
        report.update({"status": "completed_training_only_gate_check", "calibration": {
            key: artifact[key] for key in ("residual_q95", "threshold", "windows", "episodes", "residual_stats", "calibration_seconds")},
            "training_window_diagnostic": {"windows": len(rows), "episodes": len(rows), "accepted": accepted,
                "retained_reference": len(rows) - accepted,
                "greedy_selected_counts": np.bincount([r["greedy_index"] for r in rows], minlength=5).tolist(),
                "gated_selected_counts": np.bincount([r["selected_index"] for r in rows], minlength=5).tolist(),
                "predicted_gain": stats([r["predicted_gain"] for r in rows]),
                "max_gain_over_threshold": max(r["predicted_gain"] for r in rows) / artifact["threshold"] if artifact["threshold"] else None,
                "seconds": seconds, "reference_preserved_exact_when_abstaining": True},
            "decision": "retains_ACT_on_all_inspected_training_windows_no_policy_gain_claim" if accepted == 0 else "nonzero_acceptance_needs_independent_ranking_evidence_not_policy_gain",
            "act_checkpoint": binding["checkpoint_path"], "act_feature_space": prior.space_id,
            "loaded_existing_train_val_container": True, "only_train_indices_used_for_inference_and_calibration": True,
            "independent_calibration": False, "counterfactual_ranking_validated": False,
            "policy_benefit_evaluated": False, "automatic_rollout_or_training": False,
            "visual_artifact": "training_gate_example_zh.png", "visual_status": "not_viewed"})
        draw_example(example[0], example[1], artifact["threshold"], args.out)
        if any(p.requires_grad or p.grad is not None for module in (act.policy, selector.world_model) for p in module.parameters()):
            raise ValueError("frozen inference parameters changed gradient status")
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    report["runtime_seconds"] = time.monotonic() - started
    write_json(args.out / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "calibration", "training_window_diagnostic", "queue_check", "runtime_seconds", "error")}, indent=2), flush=True)
    return 0 if report["status"] == "completed_training_only_gate_check" else 1


if __name__ == "__main__":
    raise SystemExit(main())
