"""Fixed public Push-T counterfactual ranking diagnostic, not a success benchmark.

Freeze predictions before executing five candidate continuations. Reconstruct
each context by native reset + identical ACT-prefix replay, without privileged
state injection. Coverage is a diagnostic target only. No training or tuning.
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
import torch

if __package__:
    from . import run_pusht_bc_act_baseline as baseline
    from . import pusht_bc_act_inference as inference
    from . import pusht_act_wm_abstention as gate
    from .pusht_world_model_adapter import FrozenACTPrior, score_visual_goal
else:
    import run_pusht_bc_act_baseline as baseline
    import pusht_bc_act_inference as inference
    import pusht_act_wm_abstention as gate
    from pusht_world_model_adapter import FrozenACTPrior, score_visual_goal

common = baseline.common
ROOT = Path(__file__).resolve().parents[1]
PLAN = {
    "schema": "pusht_wm_action_ranking_dev10_v1",
    "development_seeds": list(range(200000, 200010)), "anchor_steps": [0, 80, 160],
    "selection": "fixed_seed_order_and_nominal_ACT_prefix_steps_not_outcomes",
    "max_contexts": 30, "nominal_prefix_cap": 168, "native_episode_limit": 300,
    "candidates": 5, "horizon": 8, "offset_xy": 8.0,
    "reference": "unchanged_ACT_chunk16_execute8_final100000",
    "dynamics": "unchanged_final10000", "goal": "cached_train_episode1_frame117_index278",
    "score": "terminal_raw_visual_feature_MSE_uniform_patches",
    "gate": "unchanged_training_residual_q95_times2_not_recalibrated",
    "branch_rule": "native_reset_same_seed_then_exact_recorded_prefix_then_fixed_candidate",
    "candidate_order": "0_1_2_3_4", "state_injection": False,
    "primary_eligibility": "at_least_two_valid_candidates_all_complete8_native_steps",
    "early_done": "stop_immediately_exclude_entire_context_from_primary_no_replacement",
    "cost_reporting_tie_epsilon": 1e-7, "coverage_reporting_tie_epsilon": 1e-6,
    "replay_coverage_atol": 1e-12,
    "primary": "predicted_vs_actual_visual_cost_ranking_and_reference_relative_gain_error",
    "secondary": "predicted_or_actual_visual_ranking_vs_actual_terminal_coverage",
    "aggregation": "contexts_and_seed_macros_not_independent_candidate_samples",
    "visual_selection": "first_development_seed_all_planned_anchors_even_if_unfavorable_or_missing",
    "optimizer_steps": 0, "hardware_actions": 0, "policy_success_benchmark": False,
    "counterfactual_targets_policy_input": False, "automatic_extension_or_tuning": False,
}


def same_observation(actual, expected, label):
    if set(actual) != {"pixels", "agent_pos"} or set(expected) != set(actual):
        raise ValueError(f"{label}: unexpected observation keys")
    for key in actual:
        if not np.array_equal(actual[key], expected[key]):
            difference = float(np.max(np.abs(actual[key].astype(float) - expected[key].astype(float))))
            raise ValueError(f"{label}: {key} replay differs, max_abs={difference}")


def clone_observation(raw):
    return {key: value.copy() for key, value in raw.items()}


def reset(env, seed, runtime):
    common._seed(seed, runtime)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    raw, _info = env.reset(seed=seed)
    return raw


def step(env, native, counter, kind):
    action, outside = common.validate_native_action(native, env.action_space.low, env.action_space.high)
    if outside:
        raise ValueError("invalid native action; no clipping or execution")
    raw, _reward, term, trunc, info = env.step(action[0])
    counter[kind] += 1
    term, trunc = bool(term), bool(trunc)
    metrics = common.info_metrics(info, terminal=term or trunc)
    return raw, {"terminated": term, "truncated": trunc, **metrics}


def save_rgb(out, name, raw):
    from PIL import Image
    relative = f"observations/{name}.png"
    Image.fromarray(raw["pixels"]).save(out / relative)
    return relative


@torch.inference_mode()
def nominal_prefix(env, runtime, act, prior, selector, seed, out, counter, old_reset_ids):
    act.reset()
    raw = reset(env, seed, runtime)
    reset_id = baseline.observation_hash(raw)  # Ten reset identities only; not per-step hashes.
    if reset_id in old_reset_ids:
        raise ValueError("new development reset duplicates a previous benchmark reset")
    raws, actions, metrics, contexts = [clone_observation(raw)], [], [], []
    for t in range(PLAN["nominal_prefix_cap"]):
        obs = inference.observation_from_native(raw, runtime["preprocess_observation"])
        if t in PLAN["anchor_steps"]:
            if len(act.policy._action_queue) != 0 or act.action_calls != t:
                raise ValueError("anchor is not on the original ACT queue boundary")
            request = prior.prepare(obs, offset_xy=PLAN["offset_xy"])
            candidates = request["candidates"]
            predicted = selector.world_model.predict_candidates(request["current_visual"], request["agent_pos"], candidates)
            scored = score_visual_goal(candidates, predicted, selector.goal)
            gated = gate.retain_reference(candidates, scored, selector.threshold)
            row = {"seed": seed, "anchor_step": t, "valid": candidates.valid[0].tolist(),
                "actions": candidates.actions[0].cpu().numpy(),
                "predicted_costs": [float(x) if np.isfinite(x) else None for x in scored.costs[0].tolist()],
                "greedy_index": int(scored.selected_index.item()), "gated_index": int(gated.scores.selected_index.item()),
                "predicted_best_gain": float(gated.predicted_gain.item()), "gate_threshold": selector.threshold,
                "current_rgb": save_rgb(out, f"{seed}_{t}_current", raw)}
            contexts.append(row)
            # Candidate scores/choices are durable BEFORE any continuation from this anchor.
            common._append(out / "predictions.jsonl", {**row, "actions": row["actions"].tolist()})
        native = act.select_action(obs)
        action = native.detach().cpu().numpy().copy()
        if contexts and t - contexts[-1]["anchor_step"] < PLAN["horizon"]:
            expected = contexts[-1]["actions"][0, t - contexts[-1]["anchor_step"]]
            if not np.array_equal(action[0], expected):
                raise ValueError("candidate0 differs from the official ACT queued action")
        raw, measured = step(env, action, counter, "nominal")
        actions.append(action.copy())
        metrics.append(measured)
        raws.append(clone_observation(raw))
        common._append(out / "nominal_prefix.jsonl", {"seed": seed, "step": t + 1,
            "action": action[0].tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    available = {row["anchor_step"] for row in contexts}
    skipped = [{"seed": seed, "anchor_step": t, "reason": "nominal_ended_before_anchor",
                "nominal_steps": len(actions)} for t in PLAN["anchor_steps"] if t not in available]
    return {"raws": raws, "actions": actions, "metrics": metrics, "contexts": contexts,
            "skipped": skipped, "reset_observation_sha256": reset_id}


@torch.inference_mode()
def counterfactual(env, runtime, prior, goal, seed, context, nominal, candidate, out, counter):
    anchor = context["anchor_step"]
    raw = reset(env, seed, runtime)
    same_observation(raw, nominal["raws"][0], "reset")
    for t, action in enumerate(nominal["actions"][:anchor]):
        raw, measured = step(env, action, counter, "replay")
        same_observation(raw, nominal["raws"][t + 1], f"prefix {seed}/{anchor}/{candidate}/{t + 1}")
        if measured["terminated"] or measured["truncated"]:
            raise ValueError("replay terminated before its nonterminal decision point")
    observations, measurements = [], []
    for h, action in enumerate(context["actions"][candidate]):
        raw, measured = step(env, action[None], counter, "candidate")
        if candidate == 0:
            # Original nominal trajectory supplies an independent repeat of candidate0.
            expected = nominal["metrics"][anchor + h]
            same_observation(raw, nominal["raws"][anchor + h + 1], f"reference continuation {seed}/{anchor}/{h}")
            if (measured["terminated"] != expected["terminated"] or measured["truncated"] != expected["truncated"]
                    or abs(measured["coverage"] - expected["coverage"]) > PLAN["replay_coverage_atol"]):
                raise ValueError("reference continuation outcome differs from original ACT prefix")
        measurements.append(measured)
        observations.append(clone_observation(raw))
        common._append(out / "candidate_steps.jsonl", {"seed": seed, "anchor_step": anchor,
            "candidate": candidate, "horizon_step": h + 1, "action": action.tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    # Real future RGB becomes an OFFLINE target, never a new planning input.
    obs = inference.observation_from_native(raw, runtime["preprocess_observation"])
    actual_features = prior.encode_image(obs["observation.image"])
    cost = float((actual_features.values - goal.values).square().mean().item())
    row = {"seed": seed, "anchor_step": anchor, "candidate": candidate,
           "steps": len(measurements), "full_horizon": len(measurements) == PLAN["horizon"],
           "actual_visual_cost_at_actual_horizon": cost, "terminal_coverage": measurements[-1]["coverage"],
           "terminated": measurements[-1]["terminated"], "truncated": measurements[-1]["truncated"],
           "prefix_observations_exact": True, "reference_continuation_exact": True if candidate == 0 else None,
           "terminal_rgb": save_rgb(out, f"{seed}_{anchor}_candidate{candidate}", raw)}
    common._append(out / "outcomes.jsonl", row)
    return row


def pair_agreement(estimated_cost, target_cost, estimated_epsilon, target_epsilon):
    counts = {"comparable": 0, "correct": 0, "wrong": 0, "estimated_tie": 0, "target_tie": 0}
    for i in range(len(target_cost)):
        for j in range(i + 1, len(target_cost)):
            truth, prediction = target_cost[i] - target_cost[j], estimated_cost[i] - estimated_cost[j]
            if abs(truth) <= target_epsilon:
                counts["target_tie"] += 1
                continue
            counts["comparable"] += 1
            if abs(prediction) <= estimated_epsilon:
                counts["estimated_tie"] += 1
            elif np.sign(truth) == np.sign(prediction):
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
    return {**counts, "accuracy": counts["correct"] / counts["comparable"] if counts["comparable"] else None}


def analyze(context, outcomes):
    row = {"seed": context["seed"], "anchor_step": context["anchor_step"],
           "valid_candidates": len(outcomes), "candidate_ids": [o["candidate"] for o in outcomes]}
    if len(outcomes) < 2 or not all(o["full_horizon"] for o in outcomes):
        return {**row, "eligible": False, "reason": "early_done_or_no_valid_alternative"}
    ids = row["candidate_ids"]
    predicted = np.array([context["predicted_costs"][i] for i in ids])
    actual = np.array([o["actual_visual_cost_at_actual_horizon"] for o in outcomes])
    coverage = np.array([o["terminal_coverage"] for o in outcomes])
    greedy, gated = ids.index(context["greedy_index"]), ids.index(context["gated_index"])
    vbest, cbest = int(actual.argmin()), int(coverage.argmax())
    eps, ceps = PLAN["cost_reporting_tie_epsilon"], PLAN["coverage_reporting_tie_epsilon"]
    informative = float(np.ptp(actual)) > eps
    row.update({"eligible": True, "informative_visual_ranking": informative,
        "coverage_varies": bool(np.ptp(coverage) > ceps),
        "predicted_costs": predicted.tolist(), "actual_visual_costs": actual.tolist(), "actual_coverages": coverage.tolist(),
        "greedy_index": ids[greedy], "gated_index": ids[gated],
        "actual_visual_best_index": ids[vbest], "actual_coverage_best_index": ids[cbest],
        "visual_top1_correct": bool(actual[greedy] <= actual.min() + eps),
        "uniform_visual_top1_chance": float(np.mean(actual <= actual.min() + eps)),
        "reference_visual_top1_correct": bool(actual[0] <= actual.min() + eps),
        "predicted_vs_actual_visual_pairs": pair_agreement(predicted, actual, eps, eps),
        "predicted_vs_coverage_pairs": pair_agreement(predicted, -coverage, eps, ceps),
        "actual_visual_vs_coverage_pairs": pair_agreement(actual, -coverage, eps, ceps),
        "goal_cost_absolute_error_mean": float(np.abs(predicted - actual).mean()),
        "reference_relative_goal_gain_mae": float(np.abs((predicted[0] - predicted[1:]) - (actual[0] - actual[1:])).mean()),
        "predicted_selected_gain": float(predicted[0] - predicted[greedy]),
        "actual_selected_visual_gain": float(actual[0] - actual[greedy]),
        "actual_visual_regret": float(actual[greedy] - actual.min()),
        "uniform_expected_actual_visual_regret": float(actual.mean() - actual.min()),
        "reference_actual_visual_regret": float(actual[0] - actual.min()),
        "actual_selected_coverage_gain": float(coverage[greedy] - coverage[0]),
        "actual_gated_coverage_gain": float(coverage[gated] - coverage[0]),
        "actual_visual_best_coverage_gain": float(coverage[vbest] - coverage[0]),
        "coverage_oracle_gain_within_candidates": float(coverage[cbest] - coverage[0]),
        "uniform_expected_coverage_gain": float(coverage.mean() - coverage[0]),
    })
    return row


def aggregate(rows):
    eligible = [r for r in rows if r["eligible"]]
    informative = [r for r in eligible if r["informative_visual_ranking"]]
    result = {"contexts": len(rows), "eligible8_step_contexts": len(eligible),
              "informative_visual_contexts": len(informative),
              "coverage_varying_contexts": sum(r["coverage_varies"] for r in eligible)}
    if not eligible:
        return result
    mean = lambda name, group=eligible: float(np.mean([r[name] for r in group])) if group else None
    result.update({"visual_top1_accuracy": mean("visual_top1_correct", informative),
                   "visual_top1_correct_count": sum(r["visual_top1_correct"] for r in informative),
                   "uniform_top1_chance_with_actual_ties": mean("uniform_visual_top1_chance", informative),
                   "reference_top1_accuracy": mean("reference_visual_top1_correct", informative)})
    for key in ("predicted_vs_actual_visual_pairs", "predicted_vs_coverage_pairs", "actual_visual_vs_coverage_pairs"):
        counts = {name: sum(r[key][name] for r in eligible) for name in ("comparable", "correct", "wrong", "estimated_tie", "target_tie")}
        result[key] = {**counts, "accuracy": counts["correct"] / counts["comparable"] if counts["comparable"] else None}
    for key in ("goal_cost_absolute_error_mean", "reference_relative_goal_gain_mae", "predicted_selected_gain",
                "actual_selected_visual_gain", "actual_visual_regret", "uniform_expected_actual_visual_regret",
                "reference_actual_visual_regret", "actual_selected_coverage_gain", "actual_gated_coverage_gain",
                "actual_visual_best_coverage_gain", "coverage_oracle_gain_within_candidates", "uniform_expected_coverage_gain"):
        result[key] = mean(key)
    for key, epsilon in (("actual_selected_visual_gain", PLAN["cost_reporting_tie_epsilon"]),
                         ("actual_selected_coverage_gain", PLAN["coverage_reporting_tie_epsilon"]),
                         ("actual_visual_best_coverage_gain", PLAN["coverage_reporting_tie_epsilon"])):
        values = np.array([r[key] for r in eligible])
        result[key + "_counts"] = {"better": int((values > epsilon).sum()), "tie": int((np.abs(values) <= epsilon).sum()),
                                  "worse": int((values < -epsilon).sum())}
    return result


def draw_first_seed(out, contexts, outcomes):
    from PIL import Image, ImageDraw, ImageFont
    title, font = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (23, 17))
    canvas = Image.new("RGB", (1536, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "世界模型动作排序诊断：首个开发 seed=200000，全部预定决策点", font=title, fill="black")
    draw.text((12, 44), "同一 seed 重放相同 ACT 前缀；各候选最多执行8步。画面均为实际 RGB，不是世界模型生成图。", font=font, fill="black")
    labels = ["0：ACT 原始", "1：X 正向", "2：X 负向", "3：Y 正向", "4：Y 负向"]
    for r, anchor in enumerate(PLAN["anchor_steps"]):
        top = 90 + r * 365
        context = next((c for c in contexts if c["anchor_step"] == anchor), None)
        draw.text((12, top), f"ACT 前缀 {anchor} 步", font=title, fill="black")
        if context is None:
            draw.text((12, top + 45), "原轨迹已提前结束；不替换、不补挑其他场景。", font=font, fill="black")
            continue
        draw.text((12, top + 35), "决策前原始观测", font=font, fill="black")
        with Image.open(out / context["current_rgb"]) as im:
            canvas.paste(im.resize((244, 244)), (6, top + 68))
        for k in range(5):
            left = (k + 1) * 256 + 6
            outcome = next((o for o in outcomes if o["anchor_step"] == anchor and o["candidate"] == k), None)
            marker = " / WM选择" if k == context["greedy_index"] else ""
            draw.text((left, top + 35), labels[k] + marker, font=font, fill="#dc2626" if marker else "black")
            if outcome is None:
                draw.text((left, top + 75), "越界候选未执行", font=font, fill="black")
                continue
            with Image.open(out / outcome["terminal_rgb"]) as im:
                canvas.paste(im.resize((244, 244)), (left, top + 68))
            draw.text((left, top + 315), f'J预测 {context["predicted_costs"][k]:.5f} / 实际 {outcome["actual_visual_cost_at_actual_horizon"]:.5f}', font=font, fill="black")
            draw.text((left, top + 339), f'覆盖 {outcome["terminal_coverage"]:.4f} / 执行 {outcome["steps"]}步', font=font, fill="black")
    draw.text((12, 1195), "覆盖率仅为离线诊断目标；提前结束组不进入8步主比较。此图不代表策略成功率或实机有效性。", font=font, fill="black")
    canvas.save(out / "first_seed_counterfactual_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_wm_action_ranking_dev10_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "observations").mkdir()
    counter = {"nominal": 0, "replay": 0, "candidate": 0}
    report = {"status": "running", "plan": PLAN, "optimizer_steps": 0, "hardware_actions": 0,
              "environment_steps_by_kind": counter, "skipped_anchors": [], "analyses": [], "seeds_completed": []}
    common._write(args.out / "started.json", report)
    started, vector = time.monotonic(), None
    try:
        torch.set_num_threads(1)
        runtime = baseline.imports()
        sources, reference = baseline.source_binding(runtime)
        if set(PLAN["development_seeds"]) & (set(baseline.PROTOCOL["benchmark_seeds"]) | set(baseline.PROTOCOL["smoke_seeds"])):
            raise ValueError("development seed IDs overlap existing benchmark/smoke schedule")
        act, binding = inference.load_final("act")
        prior = FrozenACTPrior(act, binding)
        calibration = ROOT / "simulation_output/pusht_act_wm_abstention_v1/calibration.json"
        selector = gate.load_selector(prior, calibration)
        config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        env = vector.envs[0]
        common.validate_env_kwargs(config.gym_kwargs)
        common.verify_environment_parity(reference, {"environment_gym_kwargs": config.gym_kwargs,
            "environment_step_source_sha256": common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
            "package_versions": runtime["_package_versions"]()})
        env_pre, env_post = runtime["make_env_pre_post_processors"](config, act.policy.config)
        if env_pre.steps or env_post.steps:
            raise ValueError("environment processors must remain identity")
        old_reset_ids = {e["reset_observation_sha256"] for e in reference["episodes"]}
        report.update({"act_checkpoint": binding["checkpoint_path"], "feature_space_id": prior.space_id,
            "wm_checkpoint": str(gate.wm.TRAIN_ROOT / "final.pt"), "gate_threshold": selector.threshold,
            "source_binding": sources, "environment_gym_kwargs": config.gym_kwargs,
            "package_versions": runtime["_package_versions"](),
            "existing_benchmark_report_use": "source_environment_parity_and_reset_disjointness_only",
            "no_old_outcome_based_selection_or_tuning": True})
        first_contexts, first_outcomes = [], []
        for seed in PLAN["development_seeds"]:
            nominal = nominal_prefix(env, runtime, act, prior, selector, seed, args.out, counter, old_reset_ids)
            report["skipped_anchors"].extend(nominal["skipped"])
            seed_outcomes = []
            for context in nominal["contexts"]:
                outcomes = [counterfactual(env, runtime, prior, selector.goal, seed, context, nominal, k, args.out, counter)
                            for k, valid in enumerate(context["valid"]) if valid]
                seed_outcomes.extend(outcomes)
                analysis = analyze(context, outcomes)
                report["analyses"].append(analysis)
                common._append(args.out / "analysis.jsonl", analysis)
            if seed == PLAN["development_seeds"][0]:
                first_contexts, first_outcomes = nominal["contexts"], seed_outcomes
            report["seeds_completed"].append(seed)
            common._write(args.out / "status.json", {"status": "running", "seeds_completed": report["seeds_completed"],
                "contexts": len(report["analyses"]), "environment_steps_by_kind": counter})
            print(json.dumps({"seed": seed, "contexts": len(nominal["contexts"]), "skipped": len(nominal["skipped"]),
                              "environment_steps": sum(counter.values())}), flush=True)
        report["summary"] = aggregate(report["analyses"])
        report["per_seed"] = {str(seed): aggregate([r for r in report["analyses"] if r["seed"] == seed]) for seed in PLAN["development_seeds"]}
        macro_keys = ("visual_top1_accuracy", "reference_relative_goal_gain_mae", "actual_selected_visual_gain",
                      "actual_selected_coverage_gain", "actual_visual_best_coverage_gain")
        report["seed_macro"] = {key: float(np.mean([r[key] for r in report["per_seed"].values() if r.get(key) is not None]))
            if any(r.get(key) is not None for r in report["per_seed"].values()) else None for key in macro_keys}
        if len(report["analyses"]) + len(report["skipped_anchors"]) != PLAN["max_contexts"]:
            raise ValueError("planned anchor accounting mismatch")
        if any(p.requires_grad or p.grad is not None for module in (act.policy, selector.world_model) for p in module.parameters()):
            raise ValueError("inference gradients enabled unexpectedly")
        draw_first_seed(args.out, first_contexts, first_outcomes)
        report.update({"status": "completed_fixed_counterfactual_ranking_diagnostic", "environment_steps": sum(counter.values()),
            "all_replayed_observations_exact": True, "reference_actions_and_continuations_match_ACT": True,
            "early_terminated_contexts_not_replaced": True, "benchmark_or_policy_success_rate_measured": False,
            "new_weights_or_threshold_written": False, "model_promotion": False,
            "visual_artifact": "first_seed_counterfactual_zh.png", "visual_status": "not_viewed",
            "decision": "inspect_ranking_vs_goal_proxy_separately_no_automatic_training_or_tuning"})
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    finally:
        if vector is not None:
            vector.close()
        report["runtime_seconds"] = time.monotonic() - started
        report["environment_steps"] = sum(counter.values())
        common._write(args.out / "report.json", report)
        common._write(args.out / "status.json", {"status": report["status"], "seeds_completed": report["seeds_completed"],
                                                "environment_steps_by_kind": counter})
    print(json.dumps({key: report.get(key) for key in ("status", "summary", "seed_macro", "runtime_seconds", "environment_steps_by_kind", "error")}, indent=2), flush=True)
    return 0 if report["status"] == "completed_fixed_counterfactual_ranking_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
