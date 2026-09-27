"""One frozen RGB-object scoring check on already-seen development artifacts.

No model is loaded, no training/environment/hardware step, no new data split.
Compute all image-only scores before joining the saved coverage sidecar. These
are retrospective scores of observed futures, NOT a new world-model policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

if __package__:
    from . import pusht_object_goal as adapter
    from .check_pusht_wm_action_ranking import pair_agreement
else:
    import pusht_object_goal as adapter
    from check_pusht_wm_action_ranking import pair_agreement

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_wm_action_ranking_dev10_v1"
GOAL_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_act_wm_v1")
CHECK_PLAN = {"schema": "pusht_object_goal_saved_dev_check_v1", "representation": adapter.PLAN,
    "source": "completed_dev10_counterfactual_28_contexts_140_futures_already_seen",
    "current_observations": 28, "future_observations": 140,
    "no_parameter_sweep_or_goal_reselection": True,
    "actual_future_RGB_is_offline_target_not_policy_observation": True,
    "comparison": "same_actual_futures_old_full_image_feature_cost_vs_RGB_object_cost_vs_ACT",
    "score_tie_epsilon": 1e-7, "coverage_tie_epsilon": 1e-6,
    "future_interface_check": "only_terminal_RGB_saved_broadcast_for_shape_fixture_not_a_predicted_sequence",
    "new_environment_steps": 0, "optimizer_steps": 0, "hardware_actions": 0}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lines(path):
    return [json.loads(row) for row in Path(path).read_text(encoding="utf-8").splitlines()]


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def extract_all(goal, out):
    predictions = lines(SOURCE / "predictions.jsonl")
    if len(predictions) != 28:
        raise ValueError("saved development context count changed")
    records, grids, scores, by_id = [], [], [], {}
    for context in predictions:
        seed, anchor = context["seed"], context["anchor_step"]
        for k in [-1] + [i for i, valid in enumerate(context["valid"]) if valid]:
            path = context["current_rgb"] if k == -1 else f"observations/{seed}_{anchor}_candidate{k}.png"
            observation = adapter.observe_rgb(rgb(SOURCE / path))
            index = len(records)
            row = {"seed": seed, "anchor_step": anchor, "candidate": k, "grid_index": index,
                   "rgb_path": path, "valid": observation.valid, "visible_pixels": observation.visible_pixels,
                   "centroid_xy": observation.centroid_xy.tolist() if observation.valid else None,
                   "relative_to_goal": observation.relative_to(goal),
                   "source": "observed_RGB_gray_component_not_simulator_truth",
                   "confidence": "not_calibrated", "policy_role": "current_observation" if k == -1 else "offline_future_target"}
            records.append(row)
            grids.append(observation.features.values[0])
            by_id[(seed, anchor, k)] = (row, observation)
        valid = np.array(context["valid"], dtype=bool)
        valid_observations = all(by_id[(seed, anchor, k)][0]["valid"] for k in range(5) if valid[k])
        result = {"seed": seed, "anchor_step": anchor, "eligible": valid_observations,
                  "current_object_valid": by_id[(seed, anchor, -1)][0]["valid"]}
        if valid_observations:
            terminal = np.full((1, 5, 24, 24), np.nan, dtype=np.float32)
            for k in range(5):
                if valid[k]:
                    terminal[0, k] = by_id[(seed, anchor, k)][1].features.values[0]
            # ONLY the final grid is measured. Broadcast is an explicit terminal-
            # scorer interface fixture, not fabricated intermediate observations.
            fixture = adapter.ObjectGridFeatures(np.repeat(terminal[:, :, None], 8, axis=2))
            measured = adapter.score_candidate_grids(fixture, goal.features, valid[None])
            result.update({"costs": [float(v) if np.isfinite(v) else None for v in measured.costs[0]],
                           "selected_index": int(measured.selected_index[0]),
                           "valid": valid.tolist(), "uses_observed_future_not_prediction": True})
        else:
            result["reason"] = "missing_visible_object_in_at_least_one_valid_candidate_no_fallback"
        scores.append(result)
    if len(records) != 168:
        raise ValueError("expected28 current and140 actual future RGB frames")
    np.savez_compressed(out / "object_grids.npz", grids=np.stack(grids), goal_grid=goal.features.values,
                        goal_mask96=goal.mask96)
    for name, values in (("image_observations.jsonl", records), ("rgb_only_scores.jsonl", scores)):
        with (out / name).open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    # All image-only outputs are persisted before the coverage sidecar is opened.
    return records, scores, by_id


def compare(scores, by_id):
    outcomes = lines(SOURCE / "outcomes.jsonl")
    previous = {(r["seed"], r["anchor_step"]): r for r in lines(SOURCE / "analysis.jsonl")}
    result = []
    eps, ceps = CHECK_PLAN["score_tie_epsilon"], CHECK_PLAN["coverage_tie_epsilon"]
    for score in scores:
        key = score["seed"], score["anchor_step"]
        if not score["eligible"]:
            result.append(score)
            continue
        group = sorted([o for o in outcomes if (o["seed"], o["anchor_step"]) == key], key=lambda o: o["candidate"])
        if len(group) != 5 or not all(o["full_horizon"] for o in group):
            raise ValueError("comparison requires the same five full8-step measured futures")
        coverage = np.array([o["terminal_coverage"] for o in group])
        costs = np.array(score["costs"])
        old = np.array([o["actual_visual_cost_at_actual_horizon"] for o in group])
        chosen, old_chosen = score["selected_index"], int(old.argmin())
        grids = np.stack([by_id[(*key, i)][1].features.values[0] for i in range(5)])
        matches = []
        for i in range(5):
            for j in range(i + 1, 5):
                if np.array_equal(grids[i], grids[j]):
                    matches.append({"i": i, "j": j, "new_cost_delta": abs(float(costs[i] - costs[j])),
                                    "old_cost_delta": abs(float(old[i] - old[j]))})
        result.append({"seed": key[0], "anchor_step": key[1], "eligible": True,
            "coverage_varies": bool(np.ptp(coverage) > ceps), "coverage": coverage.tolist(),
            "object_costs": costs.tolist(), "old_actual_visual_costs": old.tolist(),
            "selected_index": chosen, "old_actual_visual_selected_index": old_chosen,
            "new_pairs": pair_agreement(costs, -coverage, eps, ceps),
            "old_pairs": pair_agreement(old, -coverage, eps, ceps),
            "chosen_best_coverage": bool(coverage[chosen] >= coverage.max() - ceps),
            "old_chosen_best_coverage": bool(coverage[old_chosen] >= coverage.max() - ceps),
            "reference_best_coverage": bool(coverage[0] >= coverage.max() - ceps),
            "uniform_best_coverage_chance": float(np.mean(coverage >= coverage.max() - ceps)),
            "new_coverage_gain": float(coverage[chosen] - coverage[0]),
            "old_coverage_gain": float(coverage[old_chosen] - coverage[0]),
            "new_coverage_regret": float(coverage.max() - coverage[chosen]),
            "old_coverage_regret": float(coverage.max() - coverage[old_chosen]),
            "object_grid_all_futures_identical": bool(np.all(grids == grids[0])),
            "same_grid_pair_deltas": matches,
            "old_analysis_matches": bool(old_chosen == previous[key]["actual_visual_best_index"]),
        })
    return result


def summary(rows):
    good = [r for r in rows if r["eligible"]]
    varied = [r for r in good if r["coverage_varies"]]
    result = {"contexts": len(rows), "eligible": len(good), "coverage_varying_contexts": len(varied)}
    if not good:
        return result
    for key in ("new_pairs", "old_pairs"):
        counts = {k: sum(r[key][k] for r in good) for k in ("comparable", "correct", "wrong", "estimated_tie", "target_tie")}
        result[key] = {**counts, "accuracy": counts["correct"] / counts["comparable"] if counts["comparable"] else None}
    result["coverage_top1_on_varying_contexts"] = {
        k: sum(r[k] for r in varied) / len(varied) if varied else None
        for k in ("chosen_best_coverage", "old_chosen_best_coverage", "reference_best_coverage", "uniform_best_coverage_chance")}
    for name in ("new_coverage_gain", "old_coverage_gain", "new_coverage_regret", "old_coverage_regret"):
        values = np.array([r[name] for r in good])
        result[name + "_mean"] = float(values.mean())
        if "gain" in name:
            e = CHECK_PLAN["coverage_tie_epsilon"]
            result[name + "_counts"] = {"better": int((values > e).sum()), "tie": int((abs(values) <= e).sum()),
                                       "worse": int((values < -e).sum())}
    same_pairs = [p for r in good for p in r["same_grid_pair_deltas"]]
    result["same_object_grid"] = {"contexts_all5_identical": sum(r["object_grid_all_futures_identical"] for r in good),
        "candidate_pairs": len(same_pairs), "new_cost_max_delta": max((p["new_cost_delta"] for p in same_pairs), default=None),
        "old_cost_max_delta": max((p["old_cost_delta"] for p in same_pairs), default=None)}
    return result


def grid_picture(observed, goal):
    # Red=observed object, green=goal, overlap=dark; soft grid coverage is retained.
    o, g = observed.features.values[0], goal.features.values[0]
    image = np.ones((24, 24, 3), dtype=np.float32)
    image -= g[..., None] * np.array([.65, .12, .65], dtype=np.float32)
    image -= o[..., None] * np.array([.12, .60, .60], dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize((176, 176), Image.Resampling.NEAREST)


def figures(goal, goal_pixels, scores, by_id, comparison, out):
    from PIL import ImageDraw, ImageFont
    title, font = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (23, 17))
    template = Image.new("RGB", (1120, 475), "white")
    d = ImageDraw.Draw(template)
    d.text((12, 10), "同一训练目标图：只提取灰色 T 形物体；不读取环境目标位置", font=title, fill="black")
    d.text((12, 50), "原始目标 RGB", font=font, fill="black")
    template.paste(Image.fromarray(goal_pixels).resize((336, 336)), (12, 85))
    d.text((380, 50), f"可见物体掩码：{goal.visible_pixels} 像素", font=font, fill="black")
    template.paste(Image.fromarray(np.uint8(goal.mask96) * 255).resize((336, 336), Image.Resampling.NEAREST), (380, 85))
    d.text((745, 100), "表征：24×24 占据网格", font=font, fill="black")
    d.text((745, 145), "位置和朝向保留；不做居中对齐", font=font, fill="black")
    d.text((745, 190), "蓝色执行器、绿色标记、背景排除", font=font, fill="black")
    d.text((745, 235), "目标是示范图里的物体位姿", font=font, fill="black")
    d.text((745, 280), "不等同于精确环境目标几何", font=font, fill="black")
    d.text((12, 438), "任务专用 RGB 规则，不是通用视觉模型；遮挡或换渲染风格可能失效。", font=font, fill="black")
    template.save(out / "goal_object_template_zh.png")

    canvas = Image.new("RGB", (1536, 1470), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "物体—目标评分：对已保存的真实后续 RGB 做离线检查，未运行新策略", font=title, fill="black")
    draw.text((12, 45), "固定首个开发 seed=200000；上：实际 RGB，下：红色物体网格 / 绿色训练目标网格。", font=font, fill="black")
    for row, anchor in enumerate((0, 80, 160)):
        top = 88 + row * 442
        item = next(s for s in scores if s["seed"] == 200000 and s["anchor_step"] == anchor)
        measured = next(s for s in comparison if s["seed"] == 200000 and s["anchor_step"] == anchor)
        draw.text((12, top), f"前缀 {anchor} 步", font=title, fill="black")
        for column, k in enumerate((-1, 0, 1, 2, 3, 4)):
            left = column * 256 + 8
            record, observation = by_id[(200000, anchor, k)]
            label = "决策前观测" if k == -1 else f"候选 {k}" + ("（物体评分选择）" if item.get("selected_index") == k else "")
            draw.text((left, top + 35), label, font=font, fill="black")
            canvas.paste(Image.fromarray(rgb(SOURCE / record["rgb_path"])).resize((176, 176)), (left, top + 61))
            canvas.paste(grid_picture(observation, goal), (left, top + 241))
            if k != -1 and item["eligible"]:
                draw.text((left, top + 417), f'损失 {item["costs"][k]:.4f} / 覆盖 {measured["coverage"][k]:.4f}', font=font, fill="black")
    draw.text((12, 1430), "这些网格来自已观测未来，不是世界模型预测；不能用其回顾性选择结果宣称策略成功率提高。", font=font, fill="black")
    canvas.save(out / "first_seed_object_goal_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_goal_dev_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "plan": CHECK_PLAN, "optimizer_steps": 0, "environment_steps": 0,
              "hardware_actions": 0, "checkpoint_loads": 0, "new_policy_benefit_evaluated": False}
    write(args.out / "started.json", report)
    try:
        source = json.loads((SOURCE / "report.json").read_text())
        if source["status"] != "completed_fixed_counterfactual_ranking_diagnostic" or source["summary"]["eligible8_step_contexts"] != 28:
            raise ValueError("requires the completed, previously seen28-context diagnostic")
        manifest = json.loads((GOAL_ROOT / "manifest.json").read_text())
        if (manifest["goal"]["episode_index"], manifest["goal"]["frame_index"], manifest["goal"]["global_index"], manifest["goal"]["split"]) != (1, 117, 278, "train"):
            raise ValueError("training goal changed")
        goal_pixels = rgb(GOAL_ROOT / "goal_rgb.png")
        goal = adapter.observe_rgb(goal_pixels)
        if not goal.valid:
            raise ValueError("fixed goal object is not visible under the declared rule")
        write(args.out / "goal.json", {"plan": adapter.PLAN, "source": manifest["goal"],
            "rgb_path": str(GOAL_ROOT / "goal_rgb.png"), "visible_pixels": goal.visible_pixels,
            "centroid_xy": goal.centroid_xy.tolist(), "confidence": "not_calibrated", "validity": "visible_component_only"})
        # Reuse the real goal as an identity fixture; no synthetic quality scores.
        identity = adapter.ObjectGridFeatures(np.broadcast_to(goal.features.values[:, None, None], (1, 5, 8, 24, 24)))
        check = adapter.score_candidate_grids(identity, goal.features, np.ones((1, 5), dtype=bool))
        assert np.array_equal(check.costs, np.zeros((1, 5))) and check.selected_index.tolist() == [0]
        rejected = False
        try:
            adapter.score_candidate_grids(adapter.ObjectGridFeatures(np.zeros((1, 5, 8, 9, 512), np.float32), "ACT_ResNet_space"),
                                          goal.features, np.ones((1, 5), dtype=bool))
        except ValueError:
            rejected = True
        assert rejected
        measured = time.monotonic()
        records, scores, by_id = extract_all(goal, args.out)
        extraction_seconds = time.monotonic() - measured
        comparison = compare(scores, by_id)
        with (args.out / "comparison.jsonl").open("w", encoding="utf-8") as stream:
            for row in comparison:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        figures(goal, goal_pixels, scores, by_id, comparison, args.out)
        per_seed = {str(seed): summary([r for r in comparison if r["seed"] == seed]) for seed in sorted({r["seed"] for r in comparison})}
        report.update({"status": "completed_RGB_object_goal_interface_and_saved_dev_check", "summary": summary(comparison),
            "goal_visible_pixels": goal.visible_pixels, "RGB_frames": len(records), "valid_RGB_frames": sum(r["valid"] for r in records),
            "extraction_and_score_seconds": extraction_seconds, "per_seed": per_seed,
            "identity_zero_cost_and_reference_tie": True, "old_ResNet_space_rejected": True,
            "image_scores_saved_before_coverage_join": True, "learned_dynamics_predicts_object_grids": False,
            "actual_future_scores_are_retrospective_only": True, "heldout_or_cross_task_validation": False,
            "visual_artifacts": ["goal_object_template_zh.png", "first_seed_object_goal_zh.png"], "visual_status": "not_viewed",
            "decision": "inspect_task_relevance_before_any_new_dynamics_training_no_automatic_policy_switch"})
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    report["runtime_seconds"] = time.monotonic() - started
    write(args.out / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "summary", "RGB_frames", "valid_RGB_frames", "goal_visible_pixels", "runtime_seconds", "error")}, indent=2), flush=True)
    return 0 if report["status"] == "completed_RGB_object_goal_interface_and_saved_dev_check" else 1


if __name__ == "__main__":
    raise SystemExit(main())
