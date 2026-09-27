"""Score the unchanged saved development candidates with fixed-final DS0.

No training, ACT inference, environment replay, history recovery or policy switch.
Persist predictions before opening actual future targets or comparator analyses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import check_pusht_object_action_ranking as ranking
    from . import pusht_direct_action_scorer as ds
    from . import pusht_object_goal as vision
    from .compare_pusht_object_residual_ranking import summarize, write_rows
    from .inspect_pusht_object_dynamics import write
    from .pusht_world_model_adapter import make_candidates
else:
    import check_pusht_object_action_ranking as ranking
    import pusht_direct_action_scorer as ds
    import pusht_object_goal as vision
    from compare_pusht_object_residual_ranking import summarize, write_rows
    from inspect_pusht_object_dynamics import write
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_object_action_ranking_dev10_v1"
RESIDUAL = ROOT / "simulation_output/pusht_object_residual_ranking_dev10_v1"
METHODS = ("ACT", "uniform", "residual", "DS0", "actual_object_oracle", "actual_coverage_oracle")
METRICS = ("object_gain_vs_ACT", "object_regret", "coverage_gain_vs_ACT", "coverage_regret")
PLAN = {
    "schema": "pusht_direct_scorer_saved_ranking_v1",
    "source": str(SOURCE), "checkpoint": str(ds.TRAIN_ROOT / "final.pt"),
    "checkpoint_selection": "fixed_final10000_only",
    "scope": "same28_saved_contexts_27_eligible_140_saved_futures_135_scored_candidates",
    "candidates": "unchanged_ACT_plus_four_native8step_ramps_offset8_no_clipping",
    "observable_inputs": ["current_RGB_grid", "recovered_current_agent_XY", "same_recorded_candidate_actions", "fixed_training_goal"],
    "selection": "argmax_raw_effect_exact_ties_first_index_no_gate_or_score_clipping",
    "analysis_cost": "observed_current_goal_cost_minus_predicted_effect",
    "cost_tie_epsilon": ranking.EPS, "coverage_tie_epsilon": ranking.CEPS,
    "comparators": list(METHODS), "uniform": "analytical_expectation_not_random_rollout",
    "oracles": "post_prediction_only_never_policy_input",
    "visual_cases": [[200000, 80], [200006, 160]],
    "visual_selection": "prior_positive_and_harm_cases_fixed_before_DS0_predictions",
    "development_reused_not_fresh_test": True, "training_seeds": 1,
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "ACT_inference": False, "new_policy_rollout": False,
}


def dice(a, b):
    return np.square(a - b).sum(axis=(-2, -1)) / (
        np.square(a).sum(axis=(-2, -1)) + np.square(b).sum(axis=(-2, -1)))


def compare_methods(rows, residual_rows):
    """Expected uniform harm counts reflect the candidate distribution, not mean gain sign."""
    eligible = [i for i, row in enumerate(rows) if row["eligible"]]
    details = []
    for i in eligible:
        row, old = rows[i], residual_rows[i]
        costs, coverage = (np.asarray(row[k]) for k in ("actual_object_costs", "actual_coverages"))
        weights = {"ACT": np.eye(5)[0], "uniform": np.full(5, .2),
            "residual": np.eye(5)[old["selected_index"]], "DS0": np.eye(5)[row["selected_index"]],
            "actual_object_oracle": np.eye(5)[costs.argmin()], "actual_coverage_oracle": np.eye(5)[coverage.argmax()]}
        item = {"seed": row["seed"], "anchor_step": row["anchor_step"],
            "informative_object": row["informative_object"], "coverage_varies": row["coverage_varies"],
            "DS0_selected_index": row["selected_index"], "residual_selected_index": old["selected_index"],
            "object_headroom": float(costs[0] - costs.min()),
            "coverage_headroom": float(coverage.max() - coverage[0]), "strategies": {}}
        for name, w in weights.items():
            m = {"object_gain_vs_ACT": float(costs[0] - w @ costs), "object_regret": float(w @ costs - costs.min()),
                "coverage_gain_vs_ACT": float(w @ coverage - coverage[0]), "coverage_regret": float(coverage.max() - w @ coverage),
                "object_top1": float(w @ (costs <= costs.min() + ranking.EPS)),
                "coverage_top1": float(w @ (coverage >= coverage.max() - ranking.CEPS))}
            for prefix, gains, eps in (("object", costs[0] - costs, ranking.EPS), ("coverage", coverage - coverage[0], ranking.CEPS)):
                m.update({prefix + "_better_probability": float(w @ (gains > eps)),
                    prefix + "_tie_probability": float(w @ (np.abs(gains) <= eps)),
                    prefix + "_harm_probability": float(w @ (gains < -eps)),
                    prefix + "_worst_supported_gain": float(gains[w > 0].min())})
            item["strategies"][name] = m
            # Preserve the existing metric definitions rather than silently replacing them.
            if name in ("ACT", "uniform", "actual_object_oracle", "DS0"):
                original = row["strategies"]["model" if name == "DS0" else name]
                if any(m[k] != original[k] for k in original):
                    raise ValueError("existing strategy metric changed")
        details.append(item)
    per_seed = {str(seed): {name: {metric: float(np.mean([r["strategies"][name][metric] for r in details if r["seed"] == seed]))
        for metric in METRICS} for name in METHODS} for seed in sorted({r["seed"] for r in details})}
    methods = {}
    for name in METHODS:
        method_rows = [r["strategies"][name] for r in details]
        item = {"context_mean": {metric: float(np.mean([r[metric] for r in method_rows])) for metric in METRICS},
            "seed_macro": {metric: float(np.mean([s[name][metric] for s in per_seed.values()])) for metric in METRICS}}
        for prefix in ("object", "coverage"):
            group = [r for r in details if r["informative_object" if prefix == "object" else "coverage_varies"]]
            counts = {label: float(sum(r[prefix + "_" + label + "_probability"] for r in method_rows)) for label in ("better", "tie", "harm")}
            item[prefix] = {"better_tie_harm_count_or_uniform_expectation": counts,
                "harm_rate_or_uniform_expected_rate": counts["harm"] / len(details),
                "worst_supported_gain": min(r[prefix + "_worst_supported_gain"] for r in method_rows),
                "informative_contexts": len(group),
                "top1_informative_only": float(np.mean([r["strategies"][name][prefix + "_top1"] for r in group])) if group else None}
        methods[name] = item
    paired = {}
    for prefix, eps in (("object", ranking.EPS), ("coverage", ranking.CEPS)):
        deltas = np.array([r["strategies"]["DS0"][prefix + "_gain_vs_ACT"] - r["strategies"]["residual"][prefix + "_gain_vs_ACT"] for r in details])
        paired[prefix] = {"DS0_minus_residual_mean_gain": float(deltas.mean()),
            "better_tie_worse": [int((deltas > eps).sum()), int((np.abs(deltas) <= eps).sum()), int((deltas < -eps).sum())]}
    return {"methods": methods, "per_seed": per_seed, "DS0_vs_residual": paired}, details


def render(out, keys, current, goal, actual_grids, scores, rows, residual_rows, comparison):
    title, font = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (24, 18))
    sheet = Image.new("RGB", (1280, 1320), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), "DS0 离线候选选择：复用原动作与后果，零新增环境步", font=title, fill="black")
    draw.text((16, 50), "27 个有效上下文 / 10 个开发 seed；仅 7 处覆盖率可分辨；不是新测试集或成功率。", font=font, fill="black")
    for x, label in ((16, "方法"), (282, "平均覆盖率增益"), (584, "距最好候选的损失"), (929, "低于 ACT 的比例")):
        draw.text((x, 85), label, font=font, fill="black")
    for n, (name, label) in enumerate((("ACT", "ACT"), ("uniform", "均匀选择期望"), ("residual", "残差预测后评分"), ("DS0", "DS0 直接评分"))):
        m = comparison["methods"][name]
        y = 121 + 31 * n
        for x, text in ((16, label), (282, f"{m['context_mean']['coverage_gain_vs_ACT']:+.6f}"),
                        (584, f"{m['context_mean']['coverage_regret']:.6f}"), (929, f"{100*m['coverage']['harm_rate_or_uniform_expected_rate']:.2f}%")):
            draw.text((x, y), text, font=font, fill="black")
    for n, wanted in enumerate(PLAN["visual_cases"]):
        i = next(i for i, k in enumerate(keys) if k.tolist() == wanted)
        row, old = rows[i], residual_rows[i]
        top = 276 + 480 * n
        draw.text((16, top), f"固定旧案例 {wanted[0]}/{wanted[1]}：DS0 选 {row['selected_index']}；残差选 {old['selected_index']}；ACT=0", font=title, fill="black")
        draw.text((16, top + 39), "下图为已保存的各候选 +8 步实际网格，白0 / 黑1；不是 DS0 生成图像。", font=font, fill="black")
        for c, label in enumerate(("0：ACT", "1：+X", "2：-X", "3：+Y", "4：-Y")):
            left = 215 + 210 * c
            chosen = c == row["selected_index"]
            draw.text((left, top + 78), label + (" ←DS0" if chosen else ""), font=font, fill="#bd2929" if chosen else "black")
            tile = Image.fromarray(np.rint(255 * (1 - actual_grids[i, c])).astype(np.uint8)).convert("RGB").resize((160, 160), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, top + 110))
            if chosen:
                draw.rectangle((left - 3, top + 106, left + 162, top + 272), outline="#bd2929", width=2)
        observed = dice(current[i].astype(float), goal.astype(float)) - np.asarray(row["actual_object_costs"])
        for j, (label, values) in enumerate((("DS0 预测改善（高优）", scores[i]), ("实际目标改善（高优）", observed),
                                           ("实际覆盖率（高优）", row["actual_coverages"]))):
            y = top + 295 + 33 * j
            draw.text((16, y), label, font=font, fill="black")
            for c, value in enumerate(values):
                draw.text((215 + 210 * c, y), f"{value:+.6f}", font=font, fill="black")
        ds_gain, old_gain = (x["strategies"]["model"]["coverage_gain_vs_ACT"] for x in (row, old))
        draw.text((16, top + 409), f"相对 ACT 的实际覆盖率增益：DS0 {ds_gain:+.6f}；残差 {old_gain:+.6f}。未来真值不进入模型。", font=font, fill="black")
    draw.text((16, 1250), "两个案例在 DS0 评分前已固定；全部上下文及逐 seed 结果在报告中保留。无调参、门控或策略替换。", font=font, fill="black")
    sheet.save(out / "saved_candidate_direct_scorer_zh.png")


def run(out):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    train = ranking.read_json(ds.TRAIN_ROOT / "report.json")
    if train["plan"] != ds.PLAN or train["status"] != "completed_fixed_direct_scorer_training":
        raise ValueError("requires the frozen completed DS0 run")
    input_names = ("context_keys", "current", "current_valid", "current_agent_xy", "actions", "candidate_valid", "goal")
    with np.load(SOURCE / "predictions.npz", allow_pickle=False) as a:
        saved = {name: a[name].copy() for name in input_names}
    keys, current, flags = (saved[k] for k in ("context_keys", "current", "current_valid"))
    expected = np.array([(s, a) for s in range(200000, 200010) for a in (0, 80, 160) if (s, a) not in ((200005, 80), (200005, 160))])
    if not np.array_equal(keys, expected) or flags.sum() != 27 or keys[~flags].tolist() != [[200004, 0]]:
        raise ValueError("saved context schedule/eligibility changed")
    candidates = make_candidates(torch.from_numpy(saved["actions"][:, 0]).cuda(), offset_xy=8.)
    if not np.array_equal(candidates.actions.cpu().numpy(), saved["actions"]) or not np.array_equal(candidates.valid.cpu().numpy(), saved["candidate_valid"]) or not bool(candidates.valid.all()):
        raise ValueError("unchanged candidate actions/validity required")
    model = ds.load_final(ds.TRAIN_ROOT / "final.pt")
    if not np.array_equal(model.goal.cpu().numpy(), saved["goal"][0]):
        raise ValueError("fixed training goal changed")
    advice = ds.predict_and_score(model, vision.ObjectGridFeatures(current), flags,
        torch.from_numpy(saved["current_agent_xy"]).cuda(), candidates)
    current_cost = dice(current.astype(float), saved["goal"][0].astype(float))
    predicted_costs = current_cost[:, None] - advice.scores  # No clipping; same per-context constant.
    if not np.array_equal(predicted_costs[flags].argmin(1), advice.selected_index[flags]) or int(advice.selected_index[~flags][0]) != 0:
        raise ValueError("score/cost ranking or invalid-current fallback changed")
    np.savez_compressed(out / "predictions.npz", context_keys=keys, current_valid=flags,
        predicted_effect=advice.scores, predicted_costs=predicted_costs, selected_index=advice.selected_index)
    write_rows(out / "predictions.jsonl", [{"seed": int(k[0]), "anchor_step": int(k[1]), "eligible": bool(flags[i]),
        "selected_index": int(advice.selected_index[i]), "predicted_effect": [float(v) if np.isfinite(v) else None for v in advice.scores[i]]} for i, k in enumerate(keys)])
    write(out / "status.json", {"status": "predictions_saved_before_target_join"})

    # First access to saved future targets and residual decisions, after DS0 output is durable.
    with np.load(SOURCE / "offline_targets.npz", allow_pickle=False) as a:
        if not np.array_equal(keys, a["context_keys"]):
            raise ValueError("observation/target context alignment changed")
        actual, costs, coverage = (a[k].copy() for k in ("terminal_grids", "object_costs", "coverages"))
    if not np.allclose(dice(actual.astype(float), saved["goal"][0].astype(float)), costs, atol=1e-12, rtol=0):
        raise ValueError("saved target goal-cost calculation changed")
    reasons = tuple("DS0_direct_effect_no_calibrated_gate" if f else "invalid_current_object_retain_ACT" for f in flags)
    legacy_advice = SimpleNamespace(costs=predicted_costs, selected_index=advice.selected_index, eligible_observation=flags, reasons=reasons)
    rows = [ranking.analyze({"seed": int(k[0]), "anchor_step": int(k[1])}, legacy_advice, i, costs[i], coverage[i]) for i, k in enumerate(keys)]
    old_rows = [r for r in ranking.read_rows(RESIDUAL / "analysis.jsonl") if r["variant"] == "residual"]
    old_report = ranking.read_json(RESIDUAL / "report.json")
    if [ranking.key(r) for r in old_rows] != [tuple(k) for k in keys]:
        raise ValueError("residual comparator context order changed")
    for i, r in enumerate(old_rows):
        if r["eligible"] != bool(flags[i]) or not np.array_equal(r["actual_object_costs"], costs[i]) or not np.array_equal(r["actual_coverages"], coverage[i]):
            raise ValueError("residual comparator has different targets/eligibility")
    if ranking.aggregate(old_rows) != old_report["summaries"]["residual"]["summary"]:
        raise ValueError("residual summary does not reproduce")
    new_summary = summarize(rows)
    comparison, details = compare_methods(rows, old_rows)
    write_rows(out / "analysis.jsonl", rows)
    write_rows(out / "per_context.jsonl", details)
    render(out, keys, current, saved["goal"][0], actual, advice.scores, rows, old_rows, comparison)
    return {"status": "completed_saved_direct_scorer_ranking", "DS0": new_summary, "comparison": comparison,
        "all_informative_contexts": [r for r in details if r["informative_object"] or r["coverage_varies"]],
        "residual_summary_reproduced": True, "predictions_saved_before_target_join": True,
        "observations_actions_goal_targets_unchanged": True, "saved_future_candidates_reused": 140,
        "eligible_contexts": 27, "new_candidate_scores": 135, "new_model_forward_calls": 1,
        "candidate_rejections": int((~saved["candidate_valid"]).sum()), "invalid_current_contexts": 1,
        "invalid_current_retains_ACT": True, "validation_reruns": 0, "source_dataset_decodes": 0,
        "visual_artifact": "saved_candidate_direct_scorer_zh.png", "visual_status": "not_viewed",
        "policy_success_rate_measured": False, "model_promotion": False,
        "decision": "development_diagnostic_only_retain_ACT_no_automatic_tuning_or_extension"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_direct_scorer_ranking_dev10_v1")
    parser.add_argument("--render-only", action="store_true", help="Redraw a completed report without model inference or changing metrics.")
    args = parser.parse_args()
    if args.render_only:
        report = ranking.read_json(args.out / "report.json")
        if report["status"] != "completed_saved_direct_scorer_ranking":
            raise ValueError("render-only requires the completed report")
        with np.load(SOURCE / "predictions.npz", allow_pickle=False) as inputs, np.load(SOURCE / "offline_targets.npz", allow_pickle=False) as targets, np.load(args.out / "predictions.npz", allow_pickle=False) as predictions:
            render(args.out, inputs["context_keys"], inputs["current"], inputs["goal"][0], targets["terminal_grids"],
                predictions["predicted_effect"], ranking.read_rows(args.out / "analysis.jsonl"),
                [r for r in ranking.read_rows(RESIDUAL / "analysis.jsonl") if r["variant"] == "residual"], report["comparison"])
        print("Regenerated figure only; model forwards, optimizer and environment steps: 0.", flush=True)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": PLAN, "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0}
    write(args.out / "started.json", report)
    started = time.monotonic()
    try:
        report.update(run(args.out))
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    report["runtime_seconds"] = time.monotonic() - started
    write(args.out / "report.json", report)
    write(args.out / "status.json", {"status": report["status"], "environment_steps": 0})
    print(json.dumps({k: report[k] for k in ("status", "runtime_seconds", "error", "comparison") if k in report}, indent=2), flush=True)
    if report["status"] == "failed":
        print(report["traceback"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
