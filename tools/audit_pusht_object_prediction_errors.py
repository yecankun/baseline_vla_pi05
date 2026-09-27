"""Read-only analysis of saved object predictions: drift versus ranking reversal.

NumPy/Pillow only: no checkpoint load, model forward, training or environment.
Endpoint equality is a visible-grid diagnostic, not a contact/stationarity label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_object_action_ranking_dev10_v1"
RGB_SOURCE = ROOT / "simulation_output/pusht_wm_action_ranking_dev10_v1"
PLAN = {
    "schema": "pusht_saved_object_error_analysis_v1", "source": str(SOURCE),
    "selection": "all27_previously_eligible_contexts_all135_candidates_no_new_exclusions",
    "endpoint_unchanged": "exact_equality_of_observed_terminal_and_current_24x24_grids",
    "endpoint_not_whole_horizon_stationarity_or_no_contact": True,
    "cases": [[200000, 0], [200006, 160], [200000, 80]],
    "case_selection": "previously_identified_stationary_endpoint_failure_reversed_ranking_and_positive_control",
    "posthoc_descriptive_not_new_validation": True,
    "centroid_units": "24x24_grid_cell_centers_in_96x96_image_pixels",
    "axis_response": "centroid_or_grid_plus_candidate_minus_minus_candidate_at_saved_terminal",
    "softness": "1-sum(grid_squared)/sum(grid);pooled_observations_also_have_fractional_boundaries",
    "comparators": ["same_saved_final10000_prediction", "repeat_current_grid"],
    "optimizer_steps": 0, "model_forward_calls": 0, "environment_steps": 0, "hardware_actions": 0,
    "no_threshold_candidate_or_model_change": True,
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def dice(a, b):
    return ((a - b) ** 2).sum((-2, -1)) / (a * a + b * b).sum((-2, -1))


def centroid(grid):
    xy = (np.arange(24, dtype=np.float64) + .5) * 4
    return np.stack(((grid.sum(-2) * xy).sum(-1), (grid.sum(-1) * xy).sum(-1)), -1) / grid.sum((-2, -1))[..., None]


def softness(grid):
    return 1 - (grid * grid).sum((-2, -1)) / grid.sum((-2, -1))


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def mean(values):
    return float(np.mean(values)) if len(values) else None


def group_summary(mask, metrics, keys):
    owners = mask.any(1)
    result = {"candidate_sequences": int(mask.sum()), "contexts": int(owners.sum()),
        "seeds": len(np.unique(keys[owners, 0])), "candidate_means": {}, "context_macro": {}, "seed_macro": {}}
    for name, values in metrics.items():
        result["candidate_means"][name] = mean(values[mask])
        by_context = [mean(values[i, mask[i]]) for i in np.where(owners)[0]]
        by_seed = [mean(values[mask & (keys[:, 0] == s)[:, None]]) for s in np.unique(keys[owners, 0])]
        result["context_macro"][name] = mean(by_context)
        result["seed_macro"][name] = mean(by_seed)
    delta = (metrics["model_terminal_dice"] - metrics["persistence_terminal_dice"])[mask]
    result["model_dice_vs_persistence_counts"] = {"better": int((delta < -1e-7).sum()),
        "tie": int((np.abs(delta) <= 1e-7).sum()), "worse": int((delta > 1e-7).sum())}
    return result


def case_result(index, keys, current, predicted, actual, goal, metrics, source_rows):
    p, y, z = predicted[index], actual[index], current[index]
    cp, cy, cz = centroid(p[:, -1]), centroid(y), centroid(z)
    pd, yd = cp[0] - cz, cy[0] - cz
    base = source_rows[tuple(keys[index])]
    result = {"seed": int(keys[index, 0]), "anchor_step": int(keys[index, 1]),
        "goal_centroid_px": centroid(goal).tolist(), "current_centroid_px": cz.tolist(),
        "predicted_terminal_centroids_px": cp.tolist(), "actual_terminal_centroids_px": cy.tolist(),
        "reference_predicted_displacement_px": pd.tolist(), "reference_actual_displacement_px": yd.tolist(),
        "reference_predicted_displacement_L2_px": float(np.linalg.norm(pd)),
        "reference_actual_displacement_L2_px": float(np.linalg.norm(yd)),
        "reference_displacement_magnitude_ratio": float(np.linalg.norm(pd) / np.linalg.norm(yd)) if np.linalg.norm(yd) > 0 else None,
        "reference_predicted_displacements_per_step_px": (centroid(p[0]) - cz).tolist(),
        "reference_terminal_metrics": {name: float(value[index, 0]) for name, value in metrics.items()},
        "selected_index": base["selected_index"], "actual_object_best_index": base["actual_object_best_index"],
        "actual_coverage_best_index": base["actual_coverage_best_index"],
        "predicted_costs": base["predicted_costs"], "actual_object_costs": base["actual_object_costs"],
        "actual_coverages": base["actual_coverages"], "axis_responses": {}}
    for name, plus, minus in (("X", 1, 2), ("Y", 3, 4)):
        pred_delta, actual_delta = cp[plus] - cp[minus], cy[plus] - cy[minus]
        result["axis_responses"][name] = {
            "predicted_plus_minus_centroid_px": pred_delta.tolist(), "actual_plus_minus_centroid_px": actual_delta.tolist(),
            "centroid_direction_cosine": cosine(pred_delta, actual_delta),
            "centroid_response_magnitude_ratio": float(np.linalg.norm(pred_delta) / np.linalg.norm(actual_delta)) if np.linalg.norm(actual_delta) > 0 else None,
            "full_grid_response_cosine": cosine(p[plus, -1] - p[minus, -1], y[plus] - y[minus]),
            "predicted_plus_minus_goal_cost": base["predicted_costs"][plus] - base["predicted_costs"][minus],
            "actual_plus_minus_goal_cost": base["actual_object_costs"][plus] - base["actual_object_costs"][minus]}
    return result


def draw_cases(out, keys, current, predicted, actual, metadata, cases):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (22, 17))
    sheet = Image.new("RGB", (1536, 1350), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "已保存预测的失败分析：末端保持失败、排序反转，以及正确排序对照", font=title, fill="black")
    draw.text((12, 44), "固定旧案例，均为 ACT 原始候选0；网格白0 / 黑1。中间列仅为旧模型预测，本轮没有重新推断。", font=font, fill="black")
    draw.text((12, 70), "实际只保存了当前与+8步端点，不能据此判定全程静止、接触状态或连续运动方向。", font=font, fill="black")
    labels = ["当前实际 RGB", "当前物体网格", "预测 +1步", "预测 +4步", "预测 +8步", "实际 +8步网格", "实际 +8步 RGB"]
    names = ["末端网格保持失败", "目标评分排序反转", "正确排序对照"]
    for row, case in enumerate(cases):
        wanted = (case["seed"], case["anchor_step"])
        index = next(i for i, value in enumerate(keys) if tuple(value) == wanted)
        top = 110 + row * 395
        draw.text((12, top), f"{names[row]}：seed={wanted[0]}，anchor={wanted[1]}", font=title, fill="black")
        raw_paths = {0: RGB_SOURCE / metadata[wanted]["current_rgb"],
            6: RGB_SOURCE / f"observations/{wanted[0]}_{wanted[1]}_candidate0.png"}
        grids = {1: current[index], 2: predicted[index, 0, 0], 3: predicted[index, 0, 3],
            4: predicted[index, 0, -1], 5: actual[index, 0]}
        for column, label in enumerate(labels):
            left = 12 + column * 218
            draw.text((left, top + 38), label, font=font, fill="black")
            if column in raw_paths:
                with Image.open(raw_paths[column]) as im:
                    tile = im.convert("RGB").resize((192, 192))
            else:
                tile = Image.fromarray(np.rint(255 * (1 - np.clip(grids[column], 0, 1))).astype(np.uint8)).convert("RGB").resize((192, 192), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, top + 68))
        draw.text((12, top + 275), f"原始候选末端质心位移：预测 {case['reference_predicted_displacement_L2_px']:.3f} 像素，实际 {case['reference_actual_displacement_L2_px']:.3f} 像素。", font=font, fill="black")
        m = case["reference_terminal_metrics"]
        draw.text((12, top + 303), f"末端网格 Dice 损失（低优）：模型 {m['model_terminal_dice']:.4f}，保持当前网格 {m['persistence_terminal_dice']:.4f}。", font=font, fill="black")
        if row == 1:
            x, y = case["axis_responses"]["X"], case["axis_responses"]["Y"]
            caption = f"±X / ±Y 质心响应方向余弦：{x['centroid_direction_cosine']:.3f} / {y['centroid_direction_cosine']:.3f}；响应幅度仅为实际的 {x['centroid_response_magnitude_ratio']*100:.1f}% / {y['centroid_response_magnitude_ratio']*100:.1f}%。"
        elif row == 0:
            caption = "五个候选的实际末端网格都与当前相同；预测网格仍位移、变模糊。端点保持不等于全程没有运动。"
        else:
            caption = "这个固定对照的两轴响应方向也大致正确，且物体位置预测更准；原有五候选排序在此正确。"
        draw.text((12, top + 331), caption, font=font, fill="black")
    draw.text((12, 1305), "仅描述已见开发案例，不是新验证集或策略收益；不改模型、门控、动作、训练或实机接口。", font=font, fill="black")
    sheet.save(out / "saved_object_error_cases_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_error_analysis_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "started.json", PLAN)
    started = time.monotonic()
    source_report = json.loads((SOURCE / "report.json").read_text())
    if source_report["status"] != "completed_fixed_object_candidate_ranking":
        raise ValueError("requires the completed saved ranking check")
    rows = {(r["seed"], r["anchor_step"]): r for r in read_rows(SOURCE / "analysis.jsonl")}
    metadata = {(r["seed"], r["anchor_step"]): r for r in read_rows(SOURCE / "predictions.jsonl")}
    with np.load(SOURCE / "predictions.npz", allow_pickle=False) as a, np.load(SOURCE / "offline_targets.npz", allow_pickle=False) as b:
        if not np.array_equal(a["context_keys"], b["context_keys"]):
            raise ValueError("prediction/target keys differ")
        valid = a["current_valid"]
        if valid.shape != (28,) or valid.sum() != 27 or a["predicted"].shape != (28, 5, 8, 24, 24):
            raise ValueError("fixed saved scope changed")
        keys = a["context_keys"][valid].copy()
        current, predicted, actual, goal = a["current"][valid].astype(float), a["predicted"][valid].astype(float), b["terminal_grids"][valid].astype(float), a["goal"][0].astype(float)
        saved_costs = a["predicted_costs"][valid].copy()
    terminal = predicted[:, :, -1]
    if not np.allclose(dice(terminal, goal), saved_costs, rtol=0, atol=1e-12):
        raise ValueError("saved prediction goal costs not reproducible")
    endpoint_equal = (actual == current[:, None]).all((-2, -1))
    invariant_candidates = (actual == actual[:, 0, None]).all((1, 2, 3))
    metrics = {"model_terminal_dice": dice(terminal, actual), "persistence_terminal_dice": dice(current[:, None], actual),
        "model_centroid_error_L2_px": np.linalg.norm(centroid(terminal) - centroid(actual), axis=-1),
        "model_visible_mass_absolute_relative_error": np.abs(terminal.sum((-2, -1)) / actual.sum((-2, -1)) - 1),
        "model_visible_mass_signed_relative_error": terminal.sum((-2, -1)) / actual.sum((-2, -1)) - 1,
        "predicted_softness": softness(terminal), "observed_softness": softness(actual)}
    groups = {name: group_summary(mask, metrics, keys) for name, mask in
        (("all_eligible", np.ones_like(endpoint_equal)), ("terminal_grid_unchanged", endpoint_equal), ("terminal_grid_changed", ~endpoint_equal))}
    with (args.out / "per_candidate.jsonl").open("w", encoding="utf-8") as stream:
        for i, (seed, anchor) in enumerate(keys):
            for k in range(5):
                stream.write(json.dumps({"seed": int(seed), "anchor_step": int(anchor), "candidate": k,
                    "terminal_grid_unchanged": bool(endpoint_equal[i, k]), **{name: float(v[i, k]) for name, v in metrics.items()}}, allow_nan=False) + "\n")
    cases = [case_result(next(i for i, key in enumerate(keys) if list(key) == wanted), keys, current, predicted, actual, goal, metrics, rows) for wanted in PLAN["cases"]]
    write(args.out / "cases.json", cases)
    draw_cases(args.out, keys, current, predicted, actual, metadata, cases)
    report = {"status": "completed_saved_object_error_analysis", "plan": PLAN, "groups": groups, "cases": cases,
        "all5_terminal_grids_equal_current_contexts": keys[endpoint_equal.all(1)].tolist(),
        "all5_terminal_grids_identical_contexts": keys[invariant_candidates].tolist(),
        "endpoint_unchanged_is_diagnostic_target_only": True, "runtime_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "model_forward_calls": 0, "environment_steps": 0, "hardware_actions": 0,
        "visual_artifact": "saved_object_error_cases_zh.png", "visual_status": "not_viewed",
        "conclusion_scope": "posthoc_examples_and_descriptive_counts_not_unique_cause_or_policy_improvement",
        "decision": "keep_ACT_no_model_or_gate_change_in_this_analysis"}
    write(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "groups": groups,
        "case_reference_displacement": [{"seed": c["seed"], "anchor": c["anchor_step"],
            "predicted_px": c["reference_predicted_displacement_L2_px"], "actual_px": c["reference_actual_displacement_L2_px"],
            "axis_responses": c["axis_responses"]} for c in cases], "runtime_seconds": report["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
