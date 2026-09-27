"""Recover observable XY by recorded-prefix replay, then rank saved candidates.

No ACT inference, new candidate continuations, training, tuning or policy rollout.
Future RGB and coverage are joined only after object-model predictions are saved.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import check_pusht_wm_action_ranking as replay
    from . import pusht_object_dynamics as wm
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import make_candidates
else:
    import check_pusht_wm_action_ranking as replay
    import pusht_object_dynamics as wm
    import pusht_object_goal as vision
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_wm_action_ranking_dev10_v1"
OBJECT_SOURCE = ROOT / "simulation_output/pusht_object_goal_dev_v1"
common, baseline = replay.common, replay.baseline
EPS, CEPS = 1e-7, 1e-6
PLAN = {
    "schema": "pusht_object_action_ranking_dev10_v1",
    "development_seeds": list(range(200000, 200010)), "anchors": [0, 80, 160],
    "available_contexts": 28, "skipped": [[200005, 80], [200005, 160]],
    "prefix_replay_steps": 1440, "new_candidate_continuation_steps": 0,
    "replay": "same_native_reset_and_recorded_ACT_actions_through_last_available_anchor",
    "XY_source": "native_observation.agent_pos_at_anchor_not_action_or_hidden_object_state",
    "replay_acceptance": "exact_saved_current_RGB_and_saved_nominal_coverage_atol1e-12",
    "candidates": "same_five_native8_step_chunks_offset8_no_clipping",
    "checkpoint": "object_dynamics_fixed_final10000_only",
    "goal": "same_training_RGB_episode1_frame117_global278",
    "score": vision.PLAN["score"], "cost_tie_epsilon": EPS, "coverage_tie_epsilon": CEPS,
    "selection": "argmin_exact_ties_first_index_no_new_gate",
    "invalid_current": "no_model_prediction_retain_ACT_exclude_from_model_ranking",
    "primary": "predicted_vs_observed_future_object_goal_cost_ranking",
    "secondary": "candidate_ranking_vs_saved_terminal_coverage",
    "comparators": ["ACT", "uniform_analytical_expectation", "actual_object_oracle"],
    "aggregation": "same_eligible_contexts_and_seed_macros_ties_explicit",
    "visual_selection": "seed200000_all_three_anchors_all_five_candidates",
    "optimizer_steps": 0, "hardware_actions": 0, "new_policy_rollout": False,
    "new_weights_or_threshold": False, "future_targets_used_as_model_input": False,
    "development_reused_not_new_heldout_test": True,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def key(row):
    return row["seed"], row["anchor_step"]


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def recover_xy(contexts, nominal, source_report, out, report):
    runtime = baseline.imports()
    sources, reference = baseline.source_binding(runtime)
    if sources != source_report["source_binding"]:
        raise ValueError("saved development environment/helper binding changed")
    config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
    recovered = {}
    try:
        env = vector.envs[0]
        common.validate_env_kwargs(config.gym_kwargs)
        parity = {"environment_gym_kwargs": config.gym_kwargs,
            "environment_step_source_sha256": common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
            "package_versions": runtime["_package_versions"]()}
        common.verify_environment_parity(reference, parity)
        if any(parity[k] != source_report[k] for k in ("environment_gym_kwargs", "package_versions")):
            raise ValueError("environment differs from saved development run")
        report.update({"environment_parity": parity, "source_binding": sources})
        for seed in PLAN["development_seeds"]:
            anchors = {c["anchor_step"]: c for c in contexts if c["seed"] == seed}
            actions = {r["step"]: r for r in nominal if r["seed"] == seed}
            raw = replay.reset(env, seed, runtime)
            for t in range(max(anchors) + 1):
                if t in anchors:
                    context = anchors[t]
                    observation = replay.inference.observation_from_native(raw, runtime["preprocess_observation"], "cpu")
                    if not np.array_equal(raw["pixels"], rgb(SOURCE / context["current_rgb"])):
                        raise ValueError(f"saved current RGB mismatch: {seed}/{t}")
                    xy = observation["observation.state"].numpy()
                    if xy.shape != (1, 2) or xy.dtype != np.float32 or not np.array_equal(xy[0], raw["agent_pos"].astype(np.float32)):
                        raise ValueError("native observable XY preprocessing changed")
                    reference_actions = np.array([actions[t + h + 1]["action"] for h in range(8)], dtype=np.float32)
                    if not np.array_equal(reference_actions, np.array(context["actions"][0], dtype=np.float32)):
                        raise ValueError("saved candidate0 differs from original ACT continuation")
                    recovered[seed, t] = xy[0].copy()
                    common._append(out / "recovered_observations.jsonl", {
                        "seed": seed, "anchor_step": t, "agent_pos": raw["agent_pos"].tolist(),
                        "native_dtype": str(raw["agent_pos"].dtype), "model_agent_xy_float32": xy[0].tolist(),
                        "source": "native_observation.agent_pos", "timing": "before_action_at_anchor",
                        "confidence": "direct_observable_not_estimated", "valid": True,
                        "model_input_allowed_in_public_PushT": True, "current_rgb_exact": True,
                        "historical_XY_byte_match": "unavailable_original_XY_not_persisted",
                        "current_rgb": context["current_rgb"], "reference_actions_exact": True})
                if t == max(anchors):
                    break
                recorded = actions[t + 1]
                raw, measured = replay.step(env, np.array(recorded["action"], dtype=np.float32)[None],
                    report["environment_steps_by_kind"], "prefix_replay")
                if (measured["terminated"] != recorded["terminated"] or measured["truncated"] != recorded["truncated"]
                        or abs(measured["coverage"] - recorded["coverage"]) > 1e-12):
                    raise ValueError(f"saved nominal prefix outcome mismatch: {seed}/{t+1}")
                if measured["terminated"] or measured["truncated"]:
                    raise ValueError("prefix ended before the saved anchor")
            report["replayed_seeds"].append(seed)
            common._write(out / "status.json", {"status": "recovering_observable_XY",
                "contexts_recovered": len(recovered), "environment_steps_by_kind": report["environment_steps_by_kind"]})
            print(json.dumps({"seed": seed, "recovered": len(anchors), "replay_steps": max(anchors)}), flush=True)
    finally:
        vector.close()
    if len(recovered) != 28 or report["environment_steps_by_kind"]["prefix_replay"] != 1440:
        raise ValueError("fixed replay accounting changed")
    return recovered


def analyze(context, advice, index, actual, coverage):
    eligible = bool(advice.eligible_observation[index])
    selected = int(advice.selected_index[index])
    row = {"seed": context["seed"], "anchor_step": context["anchor_step"], "eligible": eligible,
        "reason": advice.reasons[index], "selected_index": selected,
        "informative_object": bool(np.ptp(actual) > EPS), "coverage_varies": bool(np.ptp(coverage) > CEPS),
        "actual_object_costs": actual.tolist(), "actual_coverages": coverage.tolist(),
        "actual_object_best_index": int(actual.argmin()), "actual_coverage_best_index": int(coverage.argmax())}
    strategies = {"model": np.eye(5)[selected], "ACT": np.eye(5)[0],
        "uniform": np.full(5, .2), "actual_object_oracle": np.eye(5)[actual.argmin()]}
    row["strategies"] = {name: {
        "object_top1": float(weights @ (actual <= actual.min() + EPS)),
        "object_gain_vs_ACT": float(actual[0] - weights @ actual),
        "object_regret": float(weights @ actual - actual.min()),
        "coverage_top1": float(weights @ (coverage >= coverage.max() - CEPS)),
        "coverage_gain_vs_ACT": float(weights @ coverage - coverage[0]),
        "coverage_regret": float(coverage.max() - weights @ coverage)} for name, weights in strategies.items()}
    if eligible:
        predicted = advice.costs[index]
        row.update({"predicted_costs": predicted.tolist(),
            "predicted_vs_actual_object_pairs": replay.pair_agreement(predicted, actual, EPS, EPS),
            "predicted_vs_coverage_pairs": replay.pair_agreement(predicted, -coverage, EPS, CEPS),
            "actual_object_vs_coverage_pairs": replay.pair_agreement(actual, -coverage, EPS, CEPS),
            "goal_cost_MAE": float(np.abs(predicted - actual).mean()),
            "reference_relative_goal_gain_MAE": float(np.abs((predicted[0] - predicted[1:]) - (actual[0] - actual[1:])).mean())})
    return row


def aggregate(rows):
    eligible = [r for r in rows if r["eligible"]]
    informative = [r for r in eligible if r["informative_object"]]
    coverage_varying = [r for r in eligible if r["coverage_varies"]]
    mean = lambda values: float(np.mean(values)) if values else None
    result = {"contexts": len(rows), "eligible_contexts": len(eligible), "informative_object_contexts": len(informative),
        "coverage_varying_contexts": len(coverage_varying), "coverage_varying_seeds": len({r["seed"] for r in coverage_varying}),
        "excluded_current_contexts": [{"seed": r["seed"], "anchor_step": r["anchor_step"], "reason": r["reason"]} for r in rows if not r["eligible"]]}
    for name in ("predicted_vs_actual_object_pairs", "predicted_vs_coverage_pairs", "actual_object_vs_coverage_pairs"):
        counts = {k: sum(r[name][k] for r in eligible) for k in ("comparable", "correct", "wrong", "estimated_tie", "target_tie")}
        result[name] = {**counts, "accuracy": counts["correct"] / counts["comparable"] if counts["comparable"] else None}
    for name in ("goal_cost_MAE", "reference_relative_goal_gain_MAE"):
        result[name] = mean([r[name] for r in eligible])
    result["strategies_same_eligible_contexts"] = {}
    for name in ("model", "ACT", "uniform", "actual_object_oracle"):
        values = {metric: mean([r["strategies"][name][metric] for r in eligible]) for metric in
            ("object_gain_vs_ACT", "object_regret", "coverage_gain_vs_ACT", "coverage_regret")}
        for metric, group in (("object_top1", informative), ("coverage_top1", coverage_varying)):
            values[metric + "_accuracy_informative_only"] = mean([r["strategies"][name][metric] for r in group])
            values[metric + "_count_or_uniform_expected_count"] = sum(r["strategies"][name][metric] for r in group)
        result["strategies_same_eligible_contexts"][name] = values
    for metric, epsilon in (("object_gain_vs_ACT", EPS), ("coverage_gain_vs_ACT", CEPS)):
        gains = np.array([r["strategies"]["model"][metric] for r in eligible])
        result["model_" + metric + "_counts"] = {"better": int((gains > epsilon).sum()),
            "tie": int((np.abs(gains) <= epsilon).sum()), "worse": int((gains < -epsilon).sum())}
    result["all_context_model_or_invalid_ACT_fallback_coverage_gain"] = mean([r["strategies"]["model"]["coverage_gain_vs_ACT"] for r in rows])
    return result


def draw_first_seed(out, contexts, advice, actual_grids, outcomes, analyses, goal):
    # Fixed-seed montage, not a selected favorable example or synthesized RGB.
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (23, 17))
    sheet = Image.new("RGB", (1536, 1780), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "物体世界模型候选排序：固定首个开发 seed=200000，全部决策点", font=title, fill="black")
    draw.text((12, 46), "当前 RGB + 可观测 XY + 相同候选动作 → 预测网格；实际未来与覆盖率仅用于预测后的离线核对。", font=font, fill="black")
    draw.text((12, 72), "网格统一白0 / 黑1；预测不是生成 RGB。红框为模型选择，仅离线建议，本轮没有执行候选分支。", font=font, fill="black")
    labels = ["0：ACT", "1：X 正向", "2：X 负向", "3：Y 正向", "4：Y 负向"]
    grid_rgb = lambda grid: Image.fromarray(np.rint(255 * (1 - np.clip(grid, 0, 1))).astype(np.uint8)).convert("RGB")
    for row_index, anchor in enumerate(PLAN["anchors"]):
        index = next(i for i, c in enumerate(contexts) if key(c) == (200000, anchor))
        context, analysis = contexts[index], analyses[index]
        top = 112 + row_index * 536
        draw.text((12, top), f"ACT 前缀 {anchor} 步", font=title, fill="black")
        draw.text((12, top + 36), "当前输入 RGB", font=font, fill="black")
        sheet.paste(Image.fromarray(rgb(SOURCE / context["current_rgb"])).resize((210, 210)), (12, top + 66))
        draw.text((12, top + 290), "固定训练目标网格", font=font, fill="black")
        sheet.paste(grid_rgb(goal).resize((150, 150), Image.Resampling.NEAREST), (12, top + 322))
        for candidate in range(5):
            left = 246 + candidate * 256
            selected = candidate == analysis["selected_index"]
            draw.text((left, top), labels[candidate] + (" / 模型选择" if selected else ""), font=font, fill="#c62828" if selected else "black")
            draw.text((left, top + 30), "已保存的实际 +8步 RGB", font=font, fill="black")
            outcome = outcomes[200000, anchor, candidate]
            sheet.paste(Image.fromarray(rgb(SOURCE / outcome["terminal_rgb"])).resize((192, 192)), (left, top + 60))
            draw.text((left, top + 263), "预测网格      实际网格", font=font, fill="black")
            for column, grid in enumerate((advice.predicted.values[index, candidate, -1], actual_grids[index, candidate])):
                sheet.paste(grid_rgb(grid).resize((112, 112), Image.Resampling.NEAREST), (left + column * 120, top + 293))
            draw.text((left, top + 416), f"目标代价：预测 {advice.costs[index,candidate]:.5f}", font=font, fill="black")
            draw.text((left, top + 444), f"实际 {analysis['actual_object_costs'][candidate]:.5f}（低优）", font=font, fill="black")
            draw.text((left, top + 472), f"实际覆盖率 {analysis['actual_coverages'][candidate]:.5f}", font=font, fill="black")
            if selected:
                draw.rectangle((left - 4, top - 4, left + 235, top + 502), outline="#c62828", width=2)
    draw.text((12, 1724), "开发诊断；目标平局单独计数。覆盖率不是成功率；本图不支持策略提升或实机有效性的结论。", font=font, fill="black")
    sheet.save(out / "first_seed_object_prediction_ranking_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_action_ranking_dev10_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": PLAN, "replayed_seeds": [],
        "environment_steps_by_kind": {"prefix_replay": 0, "candidate_continuation": 0},
        "optimizer_steps": 0, "hardware_actions": 0, "source_directory": str(SOURCE)}
    common._write(args.out / "started.json", report)
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        source_report = read_json(SOURCE / "report.json")
        if (source_report["status"] != "completed_fixed_counterfactual_ranking_diagnostic"
                or source_report["plan"] != replay.PLAN or not source_report["all_replayed_observations_exact"]
                or not source_report["reference_actions_and_continuations_match_ACT"]):
            raise ValueError("requires the unchanged complete fixed development diagnostic")
        contexts = read_rows(SOURCE / "predictions.jsonl")
        expected = [(s, a) for s in PLAN["development_seeds"] for a in PLAN["anchors"] if [s, a] not in PLAN["skipped"]]
        if [key(c) for c in contexts] != expected or not all(all(c["valid"]) for c in contexts):
            raise ValueError("saved context/candidate schedule changed")
        recorded = np.array([c["actions"] for c in contexts], dtype=np.float32)
        candidates = make_candidates(torch.from_numpy(recorded[:, 0]).to("cuda"), offset_xy=8.0)
        if not np.array_equal(candidates.actions.cpu().numpy(), recorded) or not bool(candidates.valid.all()):
            raise ValueError("candidate chunks differ from original native actions")
        recovered = recover_xy(contexts, read_rows(SOURCE / "nominal_prefix.jsonl"), source_report, args.out, report)
        observed = [vision.observe_rgb(rgb(SOURCE / c["current_rgb"])) for c in contexts]
        current = np.stack([o.features.values[0] for o in observed])
        flags = np.array([o.valid for o in observed], dtype=bool)
        states = np.stack([recovered[key(c)] for c in contexts])
        manifest = read_json(wm.CACHE_ROOT / "manifest.json")
        if manifest["plan"] != wm.PLAN or [manifest["goal"][k] for k in ("episode_index", "frame_index", "global_index")] != [1, 117, 278]:
            raise ValueError("trained representation or fixed training goal changed")
        goal_observed = vision.observe_rgb(rgb(wm.CACHE_ROOT / "goal_rgb.png"))
        if not goal_observed.valid:
            raise ValueError("fixed training goal not visible")
        goal = np.repeat(goal_observed.features.values, len(contexts), axis=0)
        model = wm.load_final(wm.TRAIN_ROOT / "final.pt")
        advice = wm.predict_and_score(model, vision.ObjectGridFeatures(current), flags, torch.from_numpy(states),
            candidates, vision.ObjectGridFeatures(goal))
        # Durable model output BEFORE any saved candidate-future target is opened.
        np.savez_compressed(args.out / "predictions.npz", current=current, current_agent_xy=states,
            current_valid=flags, actions=recorded, candidate_valid=candidates.valid.cpu().numpy(),
            goal=goal_observed.features.values, predicted=advice.predicted.values, predicted_costs=advice.costs,
            selected_index=advice.selected_index, context_keys=np.array(expected))
        for i, c in enumerate(contexts):
            common._append(args.out / "predictions.jsonl", {"seed": c["seed"], "anchor_step": c["anchor_step"],
                "eligible": bool(flags[i]), "reason": advice.reasons[i], "selected_index": int(advice.selected_index[i]),
                "predicted_costs": [float(v) if np.isfinite(v) else None for v in advice.costs[i]],
                "current_rgb": c["current_rgb"], "model_agent_xy": states[i].tolist()})
        common._write(args.out / "status.json", {"status": "predictions_saved_before_target_join", "contexts": len(contexts)})

        # Coverage and observed FUTURE grids stay in this post-prediction analysis.
        outcome_rows = read_rows(SOURCE / "outcomes.jsonl")
        outcomes = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcome_rows}
        if len(outcomes) != 140 or not all(r["full_horizon"] and r["steps"] == 8 for r in outcome_rows):
            raise ValueError("requires the same140 complete saved candidate futures")
        previous = {key(r): r for r in read_rows(OBJECT_SOURCE / "rgb_only_scores.jsonl")}
        with np.load(OBJECT_SOURCE / "object_grids.npz", allow_pickle=False) as saved:
            if not np.array_equal(goal_observed.features.values, saved["goal_grid"]):
                raise ValueError("goal differs from previous actual-object scoring")
        actual_grids, analyses = [], []
        for i, c in enumerate(contexts):
            targets = [outcomes[key(c) + (k,)] for k in range(5)]
            parsed = [vision.observe_rgb(rgb(SOURCE / r["terminal_rgb"])) for r in targets]
            if not all(p.valid for p in parsed):
                raise ValueError("saved terminal object became invalid")
            grids = np.stack([p.features.values[0] for p in parsed])
            # Scoring is terminal-only; repeat terminal target solely to use the existing score API.
            costs = vision.score_candidate_grids(vision.ObjectGridFeatures(np.repeat(grids[None, :, None], 8, axis=2)),
                goal_observed.features, np.ones((1, 5), dtype=bool)).costs[0]
            if not np.array_equal(costs, np.array(previous[key(c)]["costs"])) or bool(flags[i]) != previous[key(c)]["current_object_valid"]:
                raise ValueError("same saved RGB scorer/current validity differs from prior report")
            analysis = analyze(c, advice, i, costs, np.array([r["terminal_coverage"] for r in targets]))
            actual_grids.append(grids)
            analyses.append(analysis)
            common._append(args.out / "analysis.jsonl", analysis)
        actual_grids = np.stack(actual_grids)
        np.savez_compressed(args.out / "offline_targets.npz", terminal_grids=actual_grids,
            object_costs=np.array([r["actual_object_costs"] for r in analyses]),
            coverages=np.array([r["actual_coverages"] for r in analyses]), context_keys=np.array(expected))
        report["summary"] = aggregate(analyses)
        report["per_seed"] = {str(s): aggregate([r for r in analyses if r["seed"] == s]) for s in PLAN["development_seeds"]}
        report["seed_macro"] = {}
        for metric in ("object_gain_vs_ACT", "coverage_gain_vs_ACT", "object_regret", "coverage_regret"):
            values = [r["strategies_same_eligible_contexts"]["model"][metric] for r in report["per_seed"].values()]
            report["seed_macro"][metric] = float(np.mean([v for v in values if v is not None]))
        for metric in ("predicted_vs_actual_object_pairs", "predicted_vs_coverage_pairs"):
            values = [r[metric]["accuracy"] for r in report["per_seed"].values() if r[metric]["accuracy"] is not None]
            report["seed_macro"][metric] = {"accuracy": float(np.mean(values)) if values else None, "informative_seeds": len(values)}
        draw_first_seed(args.out, contexts, advice, actual_grids, outcomes, analyses, goal[0])
        report.update({"status": "completed_fixed_object_candidate_ranking", "checkpoint": str(wm.TRAIN_ROOT / "final.pt"),
            "goal": manifest["goal"], "recovered_observable_contexts": len(recovered), "saved_future_candidates_reused": 140,
            "current_RGB_matches": len(contexts), "nominal_coverage_and_done_flags_match": True,
            "historical_XY_direct_comparison_available": False, "predictions_saved_before_target_join": True,
            "candidate_actions_and_actual_object_scores_unchanged": True,
            "policy_success_rate_measured": False, "model_promotion": False, "gate_calibrated": False,
            "visual_artifact": "first_seed_object_prediction_ranking_zh.png", "visual_status": "not_viewed",
            "decision": "inspect_ranking_only_keep_ACT_no_automatic_training_gate_or_rollout"})
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        report["environment_steps"] = sum(report["environment_steps_by_kind"].values())
        common._write(args.out / "report.json", report)
        common._write(args.out / "status.json", {"status": report["status"], "environment_steps": report["environment_steps"]})
    print(json.dumps({k: report.get(k) for k in ("status", "summary", "seed_macro", "environment_steps", "runtime_seconds", "error")}, indent=2), flush=True)
    return 0 if report["status"] == "completed_fixed_object_candidate_ranking" else 1


if __name__ == "__main__":
    raise SystemExit(main())
