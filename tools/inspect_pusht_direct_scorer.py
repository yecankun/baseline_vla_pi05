"""Read the DS0 final run and one already-fixed validation example.

No fitting, full validation rerun, candidate evaluation or environment steps.
The scalar scorer does not predict a future image or occupancy grid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_direct_action_scorer as ds
    from . import pusht_object_dynamics as base
    from .compare_pusht_object_residual import changes
    from .inspect_pusht_object_dynamics import grid_image, write
else:
    import pusht_direct_action_scorer as ds
    import pusht_object_dynamics as base
    from compare_pusht_object_residual import changes
    from inspect_pusht_object_dynamics import grid_image, write

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "simulation_output/pusht_object_history_comparison_v2"
PLAN = {
    "schema": "pusht_direct_scorer_final_inspection_v1",
    "checkpoint_selection": "fixed_final10000_only",
    "validation": "existing_final_reports_no_full_validation_rerun",
    "visual_selection": "same_preselected_validation_episode0_frame76",
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(out, arrays, sample, comparison):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (24, 20))
    sheet = Image.new("RGB", (1040, 780), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "DS0 固定验证片段：直接预测目标改善量，不生成未来图像", font=title, fill="black")
    draw.text((20, 55), "episode 0 / frame 76；沿用原先片段，未按本次结果选例。", font=font, fill="black")
    for x, label, grid in (
        (60, "当前观测", arrays["current"][0]),
        (400, "实际 +8 步（仅作评价）", arrays["target"][0, 0, -1]),
        (740, "固定训练目标", arrays["goal"]),
    ):
        draw.text((x, 100), label, font=font, fill="black")
        sheet.paste(grid_image(grid), (x, 140))
    draw.text((20, 375), f"真实目标改善量 {sample['observed_effect']:+.6f}；DS0 预测 {sample['predicted_effect']:+.6f}", font=title, fill="black")
    draw.text((20, 418), "正值 = 更接近目标；以下为同一目标量的绝对误差，越低越好。", font=font, fill="black")
    draw.text((20, 463), "方法", font=font, fill="black")
    draw.text((355, 463), "此固定片段", font=font, fill="black")
    draw.text((665, 463), "全部 2002 验证窗口", font=font, fill="black")
    for row, (key, label) in enumerate((("direct_scorer", "DS0 直接评分"), ("residual", "单帧残差预测后评分"),
                                       ("history", "双帧历史预测后评分"), ("zero_effect_persistence", "零变化基线"))):
        y = 501 + row * 39
        draw.text((20, y), label, font=font, fill="black")
        draw.text((355, y), f"{sample['absolute_errors'][key]:.6f}", font=font, fill="black")
        draw.text((665, y), f"{comparison['window_mean'][key]:.6f}", font=font, fill="black")
    draw.text((20, 675), "白色 = 0，黑色 = 1；图中全是实际观测网格，不是 DS0 生成结果。", font=font, fill="black")
    draw.text((20, 713), "单种子、复用开发验证集；训练目标与容量不同；尚不能说明候选选择或策略收益。", font=font, fill="black")
    sheet.save(out / "fixed_validation_direct_scorer_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_direct_scorer_inspection_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "started.json", PLAN)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    report, run, status = (read(ds.TRAIN_ROOT / name) for name in ("report.json", "run.json", "status.json"))
    rows = [json.loads(line) for line in (ds.TRAIN_ROOT / "metrics.jsonl").read_text().splitlines()]
    checkpoint = ds.TRAIN_ROOT / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if status != {"status": "completed", "step": 10000} or report["optimizer_steps"] != 10000:
        raise ValueError("requires the completed fixed run")
    if any(x["plan"] != ds.PLAN or x["cache_identity"] != report["cache_identity"] for x in (run, payload, report)):
        raise ValueError("saved plan/cache identity mismatch")
    if not report["checkpoint_reload_exact"] or payload["target_stats"] != report["target_stats"]:
        raise ValueError("final reload/target statistics mismatch")
    if [r["step"] for r in rows] != list(range(100, 10001, 100)) or not np.isfinite(
            [[r["loss_interval_mean"], r["grad_norm"]] for r in rows]).all():
        raise ValueError("missing, duplicated or nonfinite training logs")
    optimizer_steps = sorted({int(v["step"]) for v in payload["optimizer"]["state"].values()})
    if optimizer_steps != [10000]:
        raise ValueError("optimizer state has not reached the fixed budget")
    prior = read(PRIOR / "report.json")
    references = {}
    for name in ("residual", "history"):
        path = Path(prior["runs"][name]["checkpoint"])
        r = read(path.parent / "report.json")
        p = torch.load(path, map_location="cpu", weights_only=False)
        if r["cache_identity"] != report["cache_identity"] or not torch.equal(p["sampler_rng"], payload["sampler_rng"]):
            raise ValueError("reference cache/split/normalization or final sampler differs")
        for key in ("seed", "steps", "batch_size", "lr", "weight_decay", "grad_clip", "sampling"):
            if r["plan"][key] != ds.PLAN[key]:
                raise ValueError(f"shared training budget differs: {key}")
        for key in ("eligible_windows", "eligible_episodes"):
            if r["validation"][key] != report["validation"][key]:
                raise ValueError(f"validation population differs: {key}")
        references[name] = r["validation"]["metrics"]
    metrics = report["validation"]["metrics"]
    comparison = {}
    for aggregation in ("window_mean", "episode_macro"):
        values = {name: m["all"][aggregation]["effect_mae"] for name, m in metrics.items()}
        values.update({name: m["observed_actions"][aggregation]["terminal_goal_cost_mae"] for name, m in references.items()})
        comparison[aggregation] = values
        comparison[aggregation + "_DS0_minus_reference"] = {
            name: changes(values["direct_scorer"], value) for name, value in values.items() if name != "direct_scorer"}
    comparison["per_episode_direction"] = {}
    current = metrics["direct_scorer"]["all"]["per_episode"]
    for name, m in references.items():
        other = m["observed_actions"]["per_episode"]
        if set(current) != set(other):
            raise ValueError("reference episode set differs")
        deltas = np.array([current[k]["effect_mae"] - other[k]["terminal_goal_cost_mae"] for k in current])
        comparison["per_episode_direction"][name] = {"better": int((deltas < 0).sum()), "equal": int((deltas == 0).sum()), "worse": int((deltas > 0).sum())}
    # Algebraically equal metrics can have tiny differences from float32 reduction order.
    comparison["persistence_metric_absolute_difference"] = abs(
        comparison["window_mean"]["zero_effect_persistence"] - references["residual"]["persistence"]["window_mean"]["terminal_goal_cost_mae"])
    if comparison["persistence_metric_absolute_difference"] > 1e-6:
        raise ValueError("persistence reference disagrees beyond floating-point tolerance")
    model = ds.load_final(checkpoint)
    with np.load(PRIOR / "fixed_window_predictions.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("current", "current_agent_xy", "actions", "target", "goal")}
    if not np.array_equal(arrays["goal"], model.goal.cpu().numpy()) or prior["sample"]["global_index"] != 76:
        raise ValueError("fixed validation input/goal changed")
    z, xy, actions, target = (torch.as_tensor(arrays[key], device="cuda") for key in ("current", "current_agent_xy", "actions", "target"))
    with torch.inference_mode():
        prediction = float(model.predict_effect(z, xy, actions)[0, 0])
        actual = float(ds.observed_effect(z, target, model.goal)[0, 0])
    sample = {"episode_index": 0, "frame_index": 76, "global_index": 76,
              "source": str(PRIOR / "fixed_window_predictions.npz"),
              "observed_effect": actual, "predicted_effect": prediction,
              "absolute_errors": {"direct_scorer": abs(prediction - actual), "zero_effect_persistence": abs(actual),
                  **{name: prior["sample"]["metrics"][name]["terminal_goal_cost_abs_error"] for name in references}}}
    render(args.out, arrays, sample, comparison)
    result = {"status": "completed_fixed_final_direct_scorer_inspection", "plan": PLAN,
        "checkpoint": str(checkpoint), "optimizer_state_steps": optimizer_steps,
        "logged_intervals": len(rows), "finite_logs": True,
        "first_last_interval_loss": [rows[0]["loss_interval_mean"], rows[-1]["loss_interval_mean"]],
        "same_cache_split_goal_normalization": True, "same_shared_training_budget": True, "sampler_final_states_equal": True,
        "parameter_counts": {"direct_scorer": report["parameter_count"], **{k: v["parameter_count"] for k, v in prior["runs"].items()}},
        "comparison": comparison, "sample": sample, "new_forward_calls": 1, "full_validation_reruns": 0,
        "training_elapsed_seconds": report["elapsed_seconds"], "inspection_elapsed_seconds": time.monotonic() - started,
        "visual_status": "not_viewed", "capacity_and_training_objective_matched": False,
        "decision": "retain_ACT_DS0_beats_constant_baselines_but_not_existing_predictors_selection_untested"}
    write(args.out / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
