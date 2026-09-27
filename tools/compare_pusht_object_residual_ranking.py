"""One residual-model inference on saved Push-T candidates; no environment run.

Reuse the original ranking and endpoint-error calculations. Future targets are
opened only after the new forecasts are persisted; no fitting or gate tuning.
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
    from . import audit_pusht_object_prediction_errors as errors
    from . import check_pusht_object_action_ranking as ranking
    from . import pusht_object_dynamics as base
    from . import pusht_object_residual_dynamics as wm
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import make_candidates
else:
    import audit_pusht_object_prediction_errors as errors
    import check_pusht_object_action_ranking as ranking
    import pusht_object_dynamics as base
    import pusht_object_residual_dynamics as wm
    import pusht_object_goal as vision
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = errors.SOURCE
PLAN = {"schema": "pusht_object_residual_saved_ranking_v1", "source": str(SOURCE),
    "scope": "same28_saved_contexts_27_eligible_135_candidate_sequences",
    "checkpoint": str(wm.TRAIN_ROOT / "final.pt"), "checkpoint_selection": "fixed_final10000",
    "comparators": ["saved_base_forecast", "current_grid_persistence", "ACT", "uniform_analytical_expectation", "actual_object_oracle"],
    "observable_inputs": ["current_RGB_grid", "recovered_current_agent_XY", "same_recorded_native_candidate_actions"],
    "goal": "same_training_episode1_frame117_global278_for_scoring_not_model_forward",
    "score": vision.PLAN["score"], "cost_tie_epsilon": ranking.EPS,
    "coverage_tie_epsilon": ranking.CEPS, "selection": "argmin_exact_ties_first_index_no_new_gate",
    "cases": errors.PLAN["cases"], "visual_selection": "same_three_previously_identified_cases_not_newly_selected",
    "endpoint_group": errors.PLAN["endpoint_unchanged"],
    "endpoint_equality_not_full_horizon_stationarity_or_contact": True,
    "development_reused_not_new_heldout_test": True, "training_seeds": 1,
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "new_policy_rollout": False, "ACT_inference": False, "gate_calibration": False}


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def advice_from_arrays(predicted, costs, selected, valid):
    reasons = tuple("object_forecast_score_no_calibrated_gate" if flag else "invalid_current_object_retain_ACT" for flag in valid)
    return base.ObjectCandidateAdvice(vision.ObjectGridFeatures(predicted), costs, selected, valid, reasons)


def summarize(rows):
    by_seed = {str(s): ranking.aggregate([r for r in rows if r["seed"] == s]) for s in range(200000, 200010)}
    macro = {}
    for metric in ("object_gain_vs_ACT", "coverage_gain_vs_ACT", "object_regret", "coverage_regret"):
        macro[metric] = float(np.mean([r["strategies_same_eligible_contexts"]["model"][metric] for r in by_seed.values()]))
    for metric in ("predicted_vs_actual_object_pairs", "predicted_vs_coverage_pairs"):
        values = [r[metric]["accuracy"] for r in by_seed.values() if r[metric]["accuracy"] is not None]
        macro[metric] = {"accuracy": float(np.mean(values)), "informative_seeds": len(values)}
    return {"summary": ranking.aggregate(rows), "per_seed": by_seed, "seed_macro": macro}


def endpoint_metrics(predicted, current, actual):
    terminal = predicted[:, :, -1]
    return {"model_terminal_dice": errors.dice(terminal, actual),
        "persistence_terminal_dice": errors.dice(current[:, None], actual),
        "model_centroid_error_L2_px": np.linalg.norm(errors.centroid(terminal) - errors.centroid(actual), axis=-1),
        "model_visible_mass_absolute_relative_error": np.abs(terminal.sum((-2, -1)) / actual.sum((-2, -1)) - 1),
        "model_visible_mass_signed_relative_error": terminal.sum((-2, -1)) / actual.sum((-2, -1)) - 1,
        "predicted_softness": errors.softness(terminal), "observed_softness": errors.softness(actual)}


def draw_comparison(out, keys, current, goal, actual, advice, rows, cases):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (22, 17))
    sheet = Image.new("RGB", (1280, 1460), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "残差世界模型：旧开发案例的预测与五候选排序对照", font=title, fill="black")
    draw.text((12, 45), "复用全部动作与实际未来；无环境执行。下图网格统一白0 / 黑1，均非生成 RGB。", font=font, fill="black")
    names = ["实际端点未变", "之前排序反转", "之前正确排序对照"]
    for row, wanted in enumerate(PLAN["cases"]):
        i = next(i for i, key in enumerate(keys) if list(key) == wanted)
        top = 90 + 440 * row
        b, r = rows["base"][i], rows["residual"][i]
        draw.text((12, top), f"{names[row]}：{wanted[0]}/{wanted[1]}；原模型选 {b['selected_index']}，残差模型选 {r['selected_index']}，ACT=0", font=title, fill="black")
        grids = [current[i], goal, actual[i, 0], advice["base"].predicted.values[i, 0, -1], advice["residual"].predicted.values[i, 0, -1]]
        labels = ["当前观测网格", "固定训练目标网格", "ACT候选0实际 +8步", "原模型候选0预测", "残差模型候选0预测"]
        for c, (grid, label) in enumerate(zip(grids, labels)):
            left = 12 + 250 * c
            draw.text((left, top + 37), label, font=font, fill="black")
            tile = Image.fromarray(np.rint(255 * (1 - np.clip(grid, 0, 1))).astype(np.uint8)).convert("RGB").resize((176, 176), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, top + 62))
        for c, label in enumerate(("0：ACT", "1：+X", "2：-X", "3：+Y", "4：-Y")):
            draw.text((205 + 204 * c, top + 248), label, font=font, fill="black")
        for j, (label, values) in enumerate((("原模型目标代价", b["predicted_costs"]), ("残差模型目标代价", r["predicted_costs"]),
                ("实际目标代价（低优）", r["actual_object_costs"]), ("实际覆盖率（高优）", r["actual_coverages"]))):
            draw.text((12, top + 274 + 24 * j), label, font=font, fill="black")
            for c, value in enumerate(values):
                selected = (j == 0 and c == b["selected_index"]) or (j == 1 and c == r["selected_index"])
                draw.text((205 + 204 * c, top + 274 + 24 * j), f"{value:.6f}" + (" ←选择" if selected else ""), font=font, fill="#c62828" if selected else "black")
        bc, rc = cases["base"][row], cases["residual"][row]
        draw.text((12, top + 379), f"候选0质心位移（图像像素）：实际 {rc['reference_actual_displacement_L2_px']:.3f} / 原模型 {bc['reference_predicted_displacement_L2_px']:.3f} / 残差 {rc['reference_predicted_displacement_L2_px']:.3f}", font=font, fill="black")
    draw.text((12, 1420), "固定旧案例，不是新测试集；端点未变不等于全程静止。覆盖率变化不是成功率提升。", font=font, fill="black")
    sheet.save(out / "saved_candidate_residual_comparison_zh.png")


def run(out):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    train = ranking.read_json(wm.TRAIN_ROOT / "report.json")
    previous = ranking.read_json(SOURCE / "report.json")
    if train["status"] != "completed_fixed_object_dynamics_training" or train["plan"] != wm.PLAN:
        raise ValueError("requires the completed fixed residual training")
    if previous["status"] != "completed_fixed_object_candidate_ranking" or previous["goal"] != train["cache_identity"]["goal"]:
        raise ValueError("completed source/fixed training goal changed")
    with np.load(SOURCE / "predictions.npz", allow_pickle=False) as a:
        saved = {name: a[name].copy() for name in a.files}
    keys, current, flags = saved["context_keys"], saved["current"], saved["current_valid"]
    expected = np.array([(s, a) for s in range(200000, 200010) for a in (0, 80, 160) if (s, a) not in ((200005, 80), (200005, 160))])
    if not np.array_equal(keys, expected) or flags.shape != (28,) or flags.sum() != 27 or keys[~flags].tolist() != [[200004, 0]]:
        raise ValueError("saved schedule/eligibility changed")
    candidates = make_candidates(torch.from_numpy(saved["actions"][:, 0]).cuda(), offset_xy=8.)
    if not np.array_equal(candidates.actions.cpu().numpy(), saved["actions"]) or not np.array_equal(candidates.valid.cpu().numpy(), saved["candidate_valid"]) or not bool(candidates.valid.all()):
        raise ValueError("saved candidate actions/validity changed")
    goals = np.repeat(saved["goal"], len(keys), axis=0)
    model = wm.load_final(wm.TRAIN_ROOT / "final.pt")
    new = wm.predict_and_score(model, vision.ObjectGridFeatures(current), flags,
        torch.from_numpy(saved["current_agent_xy"]), candidates, vision.ObjectGridFeatures(goals))
    np.savez_compressed(out / "predictions.npz", predicted=new.predicted.values, predicted_costs=new.costs,
        selected_index=new.selected_index, current_valid=flags, context_keys=keys)
    write_rows(out / "predictions.jsonl", [{"seed": int(k[0]), "anchor_step": int(k[1]),
        "eligible": bool(flags[i]), "selected_index": int(new.selected_index[i]), "reason": new.reasons[i],
        "predicted_costs": [float(v) if np.isfinite(v) else None for v in new.costs[i]]} for i, k in enumerate(keys)])
    errors.write(out / "status.json", {"status": "predictions_saved_before_target_join"})
    # Only now open offline target arrays. They never enter model/scorer selection.
    with np.load(SOURCE / "offline_targets.npz", allow_pickle=False) as target:
        if not np.array_equal(keys, target["context_keys"]):
            raise ValueError("source observation/target alignment differs")
        actual, costs, coverage = target["terminal_grids"].astype(float), target["object_costs"], target["coverages"]
    goal = saved["goal"][0].astype(float)
    if not np.allclose(errors.dice(actual, goal), costs, rtol=0, atol=1e-12):
        raise ValueError("saved actual-object costs not reproducible")
    persistence = np.broadcast_to(current[:, None, None], new.predicted.values.shape).copy()
    persistence[~flags] = np.nan
    p_scores = vision.score_candidate_grids(vision.ObjectGridFeatures(persistence[flags]), vision.ObjectGridFeatures(goals[flags]), saved["candidate_valid"][flags])
    p_costs = np.full((28, 5), np.nan)
    p_costs[flags] = p_scores.costs
    if not np.all(p_scores.selected_index == 0) or int(new.selected_index[~flags][0]) != 0 or not np.isnan(new.predicted.values[~flags]).all():
        raise ValueError("persistence tie or invalid-observation ACT fallback changed")
    advices = {"base": advice_from_arrays(saved["predicted"], saved["predicted_costs"], saved["selected_index"], flags),
        "residual": new, "persistence": advice_from_arrays(persistence, p_costs, np.zeros(28, dtype=np.int64), flags)}
    analyses, summaries, metrics, groups, cases = {}, {}, {}, {}, {}
    eligible_keys, z, y = keys[flags], current[flags].astype(float), actual[flags]
    unchanged = (y == z[:, None]).all((-2, -1))
    masks = {"all_eligible": np.ones_like(unchanged), "terminal_grid_unchanged": unchanged, "terminal_grid_changed": ~unchanged}
    for name, advice in advices.items():
        if not np.allclose(errors.dice(advice.predicted.values[flags, :, -1].astype(float), goal), advice.costs[flags], rtol=0, atol=1e-12):
            raise ValueError("forecast/goal-score mismatch")
        rows = [ranking.analyze({"seed": int(k[0]), "anchor_step": int(k[1])}, advice, i, costs[i], coverage[i]) for i, k in enumerate(keys)]
        analyses[name], summaries[name] = rows, summarize(rows)
        p = advice.predicted.values[flags].astype(float)
        metrics[name] = endpoint_metrics(p, z, y)
        groups[name] = {group: errors.group_summary(mask, metrics[name], eligible_keys) for group, mask in masks.items()}
        by_key = {(r["seed"], r["anchor_step"]): r for r in rows}
        cases[name] = [errors.case_result(next(i for i, k in enumerate(eligible_keys) if list(k) == wanted),
            eligible_keys, z, p, y, goal, metrics[name], by_key) for wanted in PLAN["cases"]]
    if summaries["base"]["summary"] != previous["summary"] or summaries["base"]["per_seed"] != previous["per_seed"]:
        raise ValueError("base ranking aggregates do not reproduce the previous report")
    old_groups = ranking.read_json(ROOT / "simulation_output/pusht_object_error_analysis_v1/report.json")["groups"]
    if groups["base"] != old_groups:
        raise ValueError("base endpoint diagnostics differ from previous analysis")
    write_rows(out / "analysis.jsonl", [{"variant": name, **row} for name, rows in analyses.items() for row in rows])
    write_rows(out / "per_candidate.jsonl", [{"seed": int(k[0]), "anchor_step": int(k[1]), "candidate": c,
        "terminal_grid_unchanged": bool(unchanged[i, c]),
        **{name: {metric: float(values[i, c]) for metric, values in measured.items()} for name, measured in metrics.items()}}
        for i, k in enumerate(eligible_keys) for c in range(5)])
    paired = {}
    for name, epsilon in (("coverage_gain_vs_ACT", ranking.CEPS), ("object_gain_vs_ACT", ranking.EPS)):
        gains = {v: np.array([r["strategies"]["model"][name] for r in analyses[v] if r["eligible"]]) for v in ("base", "residual")}
        delta = gains["residual"] - gains["base"]
        paired[name] = {"residual_minus_base_mean": float(delta.mean()),
            "better_tie_worse": [int((delta > epsilon).sum()), int((np.abs(delta) <= epsilon).sum()), int((delta < -epsilon).sum())],
            "residual_context_gains": [{"seed": int(k[0]), "anchor_step": int(k[1]), "gain": float(g)} for k, g in zip(eligible_keys, gains["residual"])]}
    errors.write(out / "cases.json", cases)
    draw_comparison(out, keys, current, goal, actual, advices, analyses, cases)
    return {"status": "completed_saved_residual_ranking_comparison", "summaries": summaries,
        "endpoint_groups": groups, "paired_selection": paired, "cases": cases,
        "base_ranking_and_error_reports_reproduced": True, "predictions_saved_before_target_join": True,
        "saved_source_observations_actions_goal_unchanged": True, "saved_future_candidates_reused": 140,
        "eligible_contexts": 27, "candidate_sequence_forecasts": 135, "checkpoint_loads": 1,
        "new_model_forward_calls": 1, "validation_reruns": 0, "source_dataset_decodes": 0,
        "invalid_current_retains_ACT_with_NaN_forecasts": True, "persistence_ties_retain_ACT": True,
        "visual_artifact": "saved_candidate_residual_comparison_zh.png", "visual_status": "not_viewed",
        "policy_success_rate_measured": False, "model_promotion": False,
        "decision": "keep_ACT_no_automatic_training_gate_or_policy_rollout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_residual_ranking_dev10_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": PLAN, "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0}
    errors.write(args.out / "started.json", report)
    started = time.monotonic()
    try:
        report.update(run(args.out))
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    report["runtime_seconds"] = time.monotonic() - started
    errors.write(args.out / "report.json", report)
    errors.write(args.out / "status.json", {"status": report["status"], "environment_steps": 0})
    print(json.dumps({k: report[k] for k in ("status", "runtime_seconds", "error") if k in report}, indent=2), flush=True)
    if report["status"] == "failed":
        print(report["traceback"], flush=True)
        return 1
    print(json.dumps({"summaries": {k: v["summary"] for k, v in report["summaries"].items()},
        "endpoint_candidate_means": {v: {g: r["candidate_means"] for g, r in groups.items()} for v, groups in report["endpoint_groups"].items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
