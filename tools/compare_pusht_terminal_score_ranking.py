"""Compare the fixed terminal-score pair on already saved development futures.

No training, ACT inference, environment creation/replay or policy promotion.
Save both arms' predictions before joining any saved future target or old score.
"""
from __future__ import annotations

import argparse
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
    from . import run_pusht_terminal_score_pair as pair
    from . import run_pusht_fresh_scorer_pair as previous
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import make_candidates
else:
    import run_pusht_terminal_score_pair as pair
    import run_pusht_fresh_scorer_pair as previous
    import pusht_object_goal as vision
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_fresh_scorer_pair20_v1"
OUT = ROOT / "simulation_output/pusht_terminal_score_ranking_dev20_v1"
NAMES = pair.VARIANTS
PLAN = {
    "schema": "pusht_terminal_score_ranking_dev20_v1", "source": str(SOURCE),
    "training_root": str(pair.TRAIN_ROOT), "arms": list(NAMES),
    "checkpoint_selection": "both_fixed_final10000_no_selection_or_retraining",
    "population": "same58_saved_contexts_57_eligible_20_previously_inspected_seeds",
    "candidates": "same_ACT_plus4_native8step_ramps_offset8_no_clipping",
    "input_fields": ["current_RGB_grid", "observable_agent_XY", "saved_candidate_actions", "fixed_training_goal"],
    "selection": "argmax_raw_terminal_score_exact_tie_first_index_no_gate_or_clipping",
    "prediction_isolation": "persist_both_arms_before_future_target_and_historical_score_join",
    "primary": "paired_seed_macro_coverage_gain_reward_terminal_minus_visual_terminal",
    "secondary": "same_context_gain_regret_harm_worst_top1_pair_ordering_and_object_metrics",
    "coverage_tie_epsilon": 1e-6, "object_and_prediction_tie_epsilon": 1e-7,
    "aggregation": "reuse_original_analysis_context_and_seed_macros_no_independent_pair_claim",
    "uniform": "analytical_candidate_expectation_not_random_rollout",
    "oracles": "actual_saved_outcome_only_offline_not_policy_inputs",
    "visual_cases": [[300007, 80], [300005, 160], [300010, 160]],
    "visual_selection": "previous_positive_case_previous_worst_DS0_harm_previous_proxy_harm_fixed_before_new_inference",
    "reused_development_not_fresh_test": True, "training_seeds": 1,
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "ACT_inference": False, "new_candidate_continuations": 0, "policy_promotion": False,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def rename_methods(value):
    """Only evaluator slot names change; the original metric math is reused."""
    aliases = {"DS0": "reward_terminal", "residual": "visual_terminal"}
    if isinstance(value, dict):
        return {aliases.get(k, k): rename_methods(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_methods(v) for v in value]
    return value


def render(out, predictions, outcomes, analysis):
    title, font, small = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n)
                          for n in (26, 20, 17))
    image = Image.new("RGB", (1540, 1415), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "末端监督配对：同一批候选，比较实际动作选择后果", font=title, fill="black")
    draw.text((16, 51), "57 个有效上下文 / 20 个已看 seed；均为保存的实际后继 RGB，不是模型生成图像或新 rollout。", font=font, fill="black")
    for x, text in ((16, "方法"), (310, "seed宏平均覆盖率增益"), (620, "距最好候选的损失"), (970, "有害选择 / 57"), (1220, "可分辨场景最优命中")):
        draw.text((x, 86), text, font=small, fill="black")
    for i, (name, label) in enumerate((("ACT", "ACT 原动作"), (NAMES[0], "视觉末端评分"),
                                     (NAMES[1], "任务 reward 评分"), ("oracle", "实际覆盖率最优（事后）"))):
        m = analysis["coverage"]["methods"][name]
        for x, text in ((16, label), (310, f"{m['seed_macro']['gain']:+.7f}"), (620, f"{m['seed_macro']['regret']:.7f}"),
                        (970, f"{m['better_tie_harm_count_or_uniform_expectation']['harm']:.0f} / 57"),
                        (1220, f"{m['top1_informative_only'] * 17:.0f} / 17")):
            draw.text((x, 116 + i * 27), text, font=small, fill="black")
    keyed = {(r["seed"], r["anchor_step"]): r for r in predictions}
    targets = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcomes}
    for i, key in enumerate(PLAN["visual_cases"]):
        p = keyed[tuple(key)]
        top = 243 + i * 365
        v, r = (p["selected"][n] for n in NAMES)
        draw.text((16, top), f"固定旧案例 {key[0]} / {key[1]}：视觉组选 {v}（蓝），任务组选 {r}（红）；ACT=0", font=font, fill="black")
        for col in range(6):
            left, k = 16 + col * 253, col - 1
            target = targets[(*key, k)] if col else None
            path = target["terminal_rgb"] if target else p["current_rgb"]
            draw.text((left, top + 35), "当前输入" if not col else f"候选 {k}" + (" / ACT" if k == 0 else ""), font=small, fill="black")
            image.paste(Image.fromarray(rgb(SOURCE / path)).resize((176, 176)), (left, top + 63))
            if col:
                if k == v:
                    draw.rectangle((left - 2, top + 61, left + 177, top + 240), outline="#2455bb", width=2)
                if k == r:
                    draw.rectangle((left - 5, top + 58, left + 180, top + 243), outline="#bb3333", width=2)
                draw.text((left, top + 249), f"实际覆盖率 {target['terminal_coverage']:.6f}", font=small, fill="black")
                draw.text((left, top + 278), f"视觉预测 {p['scores'][NAMES[0]][k]:+.6f}", font=small, fill="#2455bb")
                draw.text((left, top + 307), f"任务预测 {p['scores'][NAMES[1]][k]:+.6f}", font=small, fill="#bb3333")
    draw.text((16, 1356), "案例在本轮推理前固定；两组评分单位不同，不能直接比数值。未来真值只用于事后评价。", font=font, fill="black")
    draw.text((16, 1385), "单训练种子、重复使用的开发场景；不是闭环成功率或真实系统收益，不自动替换 ACT / real10。", font=small, fill="black")
    image.save(out / "terminal_score_candidates_zh.png")


@torch.inference_mode()
def run(out):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    training = read(pair.TRAIN_ROOT / "report.json")
    if (training["status"] != "completed_matched_terminal_training"
            or training["optimizer_steps_per_arm"] != 10000
            or not training["same_initial_trainable_parameters"]
            or not training["same_complete_ordered_sampled_rows"]):
        raise ValueError("requires the completed fixed paired training")
    # Whitelist current-only fields; do not use historical scores as inputs.
    fields = ("seed", "anchor_step", "current_valid", "valid", "actions", "agent_xy", "current_rgb")
    inputs = [{k: row[k] for k in fields} for row in lines(SOURCE / "predictions.jsonl")]
    keys = np.array([(r["seed"], r["anchor_step"]) for r in inputs])
    expected = np.array([(s, a) for s in range(300000, 300020) for a in (0, 80, 160)
                         if (s, a) not in ((300011, 80), (300011, 160))])
    seen = [vision.observe_rgb(rgb(SOURCE / r["current_rgb"])) for r in inputs]
    flags = np.array([r.valid for r in seen])
    if (not np.array_equal(keys, expected) or not np.array_equal(flags, [r["current_valid"] for r in inputs])
            or flags.sum() != 57 or keys[~flags].tolist() != [[300006, 0]]):
        raise ValueError("original cohort/current-observation eligibility changed")
    current = np.concatenate([r.features.values for r in seen])
    xy = np.asarray([r["agent_xy"] for r in inputs], dtype=np.float32)
    actions = np.asarray([r["actions"] for r in inputs], dtype=np.float32)
    candidates = make_candidates(torch.from_numpy(actions[:, 0]).cuda(), offset_xy=8.)
    if (not np.array_equal(actions, candidates.actions.cpu().numpy())
            or not np.array_equal(candidates.valid.cpu().numpy(), [r["valid"] for r in inputs])
            or not bool(candidates.valid.all())):
        raise ValueError("saved candidate action or validity changed")
    scores, selected, checkpoint_identity, goal = {}, {}, {}, None
    for name in NAMES:
        path = pair.TRAIN_ROOT / name / "final.pt"
        model, payload = pair.load_final(path)
        if payload["variant"] != name or payload["contract"] != training["contract"]:
            raise ValueError("checkpoint target/contract mismatch")
        observed_goal = model.goal.cpu().numpy()
        if goal is not None and not np.array_equal(goal, observed_goal):
            raise ValueError("paired fixed training goal differs")
        goal = observed_goal.copy()
        advice = pair.ds.predict_and_score(model, vision.ObjectGridFeatures(current), flags,
                                           torch.from_numpy(xy).cuda(), candidates)
        scores[name], selected[name] = advice.scores, advice.selected_index
        if not np.array_equal(advice.selected_index[~flags], [0]):
            raise ValueError("invalid current observation must retain ACT")
        stat = path.stat()
        checkpoint_identity[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                     "schema": payload["schema"], "step": payload["step"]}
        del model, payload
    predictions = [{"seed": int(k[0]), "anchor_step": int(k[1]), "current_valid": bool(flags[i]),
                    "current_rgb": inputs[i]["current_rgb"],
                    "selected": {n: int(selected[n][i]) for n in NAMES},
                    "scores": {n: [float(v) if np.isfinite(v) else None for v in scores[n][i]] for n in NAMES}}
                   for i, k in enumerate(keys)]
    np.savez_compressed(out / "predictions.npz", context_keys=keys, current=current, current_valid=flags,
                        agent_xy=xy, actions=actions, goal=goal,
                        **{n + "_scores": scores[n] for n in NAMES},
                        **{n + "_selected": selected[n] for n in NAMES})
    write_rows(out / "predictions.jsonl", predictions)
    write(out / "status.json", {"status": "both_predictions_saved_before_target_join"})

    # First future-target access. Reuse and exactly reproduce the original analysis.
    outcomes = lines(SOURCE / "outcomes.jsonl")
    old_inputs, old_report = lines(SOURCE / "predictions.jsonl"), read(SOURCE / "report.json")
    if (old_report["status"] != "completed_fresh_seed_candidate_comparison"
            or previous.analyze(old_inputs, outcomes) != old_report["analysis"]):
        raise ValueError("historical metrics do not reproduce on unchanged saved outcomes")
    # analyze()'s historical DS0 slot accepts any higher-is-better score; its
    # residual slot accepts a cost. No delta interpretation enters metric math.
    adapted = [{**row, "selected": {"DS0": int(selected[NAMES[1]][i]), "residual": int(selected[NAMES[0]][i])},
                "DS0_effect": scores[NAMES[1]][i].tolist(), "residual_costs": (-scores[NAMES[0]][i]).tolist()}
               for i, row in enumerate(inputs)]
    analysis = rename_methods(previous.analyze(adapted, outcomes))
    analysis.pop("new_seed_generalization_not_closed_loop_success")
    analysis["reused_development_not_fresh_test"] = True
    for metric, epsilon in (("coverage", 1e-6), ("object", 1e-7)):
        a = analysis[metric]
        if a["eligible_contexts"] != old_report["analysis"][metric]["eligible_contexts"]:
            raise ValueError("outcome cohort changed")
        a["paired_seed_delta"]["direction"] = "reward_terminal_minus_visual_terminal"
        delta = np.array([r["strategies"][NAMES[1]]["gain"] - r["strategies"][NAMES[0]]["gain"] for r in a["per_context"]])
        a["paired_context_delta"] = {"direction": "reward_terminal_minus_visual_terminal", "mean": float(delta.mean()),
                                     "better_tie_worse": [int((delta > epsilon).sum()), int((np.abs(delta) <= epsilon).sum()), int((delta < -epsilon).sum())]}
    targets = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcomes}
    details = []
    for i in np.flatnonzero(flags):
        coverage = [targets[(*keys[i], k)]["terminal_coverage"] for k in range(5)]
        details.append({**predictions[i], "actual_coverages": coverage,
                        "actual_object_costs": [targets[(*keys[i], k)]["terminal_object_cost"] for k in range(5)],
                        "coverage_gain_vs_ACT": {n: float(coverage[selected[n][i]] - coverage[0]) for n in NAMES}})
    write_rows(out / "per_context.jsonl", details)
    coverage = np.array([r["actual_coverages"] for r in details])
    render(out, predictions, outcomes, analysis)
    return {"status": "completed_saved_terminal_score_ranking", "checkpoint_identity": checkpoint_identity,
            "analysis": analysis, "historical_metric_reproduction_exact": True,
            "historical_references_not_matched_terminal_arms": {
                m: {n: old_report["analysis"][m]["methods"][n] for n in ("DS0", "residual")} for m in ("coverage", "object")},
            "same_selection_contexts": int((selected[NAMES[0]][flags] == selected[NAMES[1]][flags]).sum()),
            "actual_reward_clipping": {"coverage_at_or_above_0_95_candidates": int((coverage >= .95).sum()),
                                       "eligible_candidates": int(coverage.size), "maximum_coverage": float(coverage.max())},
            "predictions_saved_before_target_join": True, "saved_future_candidates_reused": len(outcomes),
            "eligible_contexts": 57, "eligible_seeds": 20, "new_model_forward_calls": 2,
            "new_candidate_scores_per_arm": 285, "candidate_rejections": 0,
            "invalid_current_retains_ACT_both": True, "new_environment_resets": 0,
            "validation_reruns": 0, "dataset_video_decodes": 0,
            "visual_artifact": "terminal_score_candidates_zh.png", "visual_status": "not_viewed",
            "policy_success_rate_measured": False,
            "decision": "development_diagnostic_only_no_automatic_tuning_extension_or_policy_promotion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--render-only", action="store_true", help="Redraw completed evidence, no inference or metric change.")
    args = parser.parse_args()
    if args.render_only:
        report = read(args.out / "report.json")
        if report["status"] != "completed_saved_terminal_score_ranking":
            raise ValueError("render-only requires a completed report")
        render(args.out, lines(args.out / "predictions.jsonl"), lines(SOURCE / "outcomes.jsonl"), report["analysis"])
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": PLAN, "optimizer_steps": 0, "environment_steps": 0,
              "hardware_actions": 0, "new_candidate_continuations": 0}
    write(args.out / "started.json", report)
    started = time.monotonic()
    try:
        report.update(run(args.out))
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    report["runtime_seconds"] = time.monotonic() - started
    write(args.out / "report.json", report)
    write(args.out / "status.json", {"status": report["status"], "environment_steps": 0})
    print(json.dumps({k: report[k] for k in ("status", "runtime_seconds", "error", "same_selection_contexts") if k in report}, indent=2), flush=True)
    if report["status"] == "failed":
        print(report["traceback"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
