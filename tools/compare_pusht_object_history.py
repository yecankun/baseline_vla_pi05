"""Read fixed-final history/residual results and the same validation example.

No training, full validation rerun, candidate recovery or environment execution.
The extra history projection capacity is reported, not treated as controlled.
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
    from . import pusht_object_dynamics as base
    from . import pusht_object_residual_dynamics as residual
    from . import pusht_object_history_dynamics as history
    from .compare_pusht_object_residual import changes, read_run
    from .inspect_pusht_object_dynamics import grid_image, write
else:
    import pusht_object_dynamics as base
    import pusht_object_residual_dynamics as residual
    import pusht_object_history_dynamics as history
    from compare_pusht_object_residual import changes, read_run
    from inspect_pusht_object_dynamics import grid_image, write

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "simulation_output/pusht_object_residual_comparison_v1"
PLAN = {
    "schema": "pusht_object_history_final_comparison_v1",
    "checkpoint_selection": "fixed_final10000_only",
    "validation": "read_existing_final_reports_no_validation_rerun",
    "visual_selection": "same_lowest_ID_validation_episode_middle_eligible_window",
    "display_offsets": [1, 4, 8], "display_range": [0, 1],
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False,
    "capacity_matched_history_off_training": False,
}


def compare_metrics(residual_report, history_report):
    r, h = (x["validation"]["metrics"] for x in (residual_report, history_report))
    if r["persistence"] != h["persistence"]:
        raise ValueError("persistence differs across fixed final evaluations")
    result = {}
    for aggregation in ("window_mean", "episode_macro"):
        result[aggregation] = {}
        for metric in base.PLAN["heldout_metrics"]:
            rv, hv = (x["observed_actions"][aggregation][metric] for x in (r, h))
            pv = r["persistence"][aggregation][metric]
            result[aggregation][metric] = {
                "residual": rv, "history": hv, "persistence": pv,
                "history_minus_residual": changes(hv, rv),
                "history_minus_persistence": changes(hv, pv),
                "residual_mean_action_ablation": r["mean_action_inference_ablation"][aggregation][metric],
                "history_mean_action_ablation": h["mean_action_inference_ablation"][aggregation][metric],
            }
    result["per_episode_direction"] = {}
    he = h["observed_actions"]["per_episode"]
    for name, reference in (("residual", r["observed_actions"]), ("persistence", r["persistence"])):
        re = reference["per_episode"]
        if set(he) != set(re):
            raise ValueError("validation episode set differs")
        result["per_episode_direction"][name] = {}
        for metric in base.PLAN["heldout_metrics"]:
            delta = np.array([he[ep][metric] - re[ep][metric] for ep in sorted(he)])
            result["per_episode_direction"][name][metric] = {
                "better": int((delta < 0).sum()), "equal": int((delta == 0).sum()),
                "worse": int((delta > 0).sum())}
    return result


def render(out, data, index, previous, forecasts, sample):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (22, 18))
    sheet = Image.new("RGB", (1280, 1220), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "单帧残差 vs 双帧历史：相同 10,000 步；固定验证片段（非优选案例）", font=title, fill="black")
    draw.text((12, 47), f"episode={data.episodes[index]}，frame={data.frames[index]}；RGB 派生/预测占据网格，不是生成 RGB。", font=font, fill="black")
    for column, offset in enumerate((0, 1, 4, 8)):
        left = 170 + 270 * column
        draw.text((left, 87), "当前 / 历史输入" if offset == 0 else f"未来 +{offset} 步", font=title, fill="black")
        sheet.paste(grid_image(data.grids[index + offset].cpu().numpy()), (left, 132))
        for row, name in enumerate(("residual", "history", "persistence"), start=1):
            top = 132 + 232 * row
            if offset:
                sheet.paste(grid_image(forecasts[name][0, 0, offset - 1].cpu().numpy()), (left, top))
            elif name == "history":
                sheet.paste(grid_image(data.grids[previous].cpu().numpy()), (left, top))
                draw.text((left, top + 210), f"上一帧 {data.frames[previous]}；valid={int(data.history_valid[index])}", font=font, fill="black")
            else:
                draw.text((left, top + 85), "只用当前网格 / XY" if name == "residual" else "重复当前网格", font=font, fill="black")
    for row, label in enumerate(("实际观测网格", "单帧残差预测", "双帧历史预测", "保持不动基线")):
        draw.text((10, 222 + row * 232), label, font=font, fill="black")
    for column, (name, label) in enumerate((("residual", "单帧残差"), ("history", "双帧历史"), ("persistence", "保持不动"))):
        draw.text((15 + 415 * column, 1070), f"{label}：终点 Dice {sample[name]['dice_per_step'][-1]:.6f}", font=font, fill="black")
    draw.text((12, 1110), "白色=0，黑色=1；所有网格范围固定。单片段不能代表整体，也不能证明 ACT 策略收益。", font=font, fill="black")
    draw.text((12, 1150), "历史版参数增加 6.75%；尚无等容量 history-off 训练对照，不能将改善全部归因于历史输入。", font=font, fill="black")
    sheet.save(out / "fixed_validation_history_comparison_zh.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_history_comparison_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "started.json", PLAN)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = history.ObjectData()
    rr, rm, rs, ri = read_run(residual, data.identity)
    hr, hm, hs, hi = read_run(history, data.identity)
    changed = {"schema", "architecture", "initialization", "temporal_mapping", "base_model_schema", "comparison"}
    unexpected = [k for k, v in rr["plan"].items() if k not in changed and hr["plan"][k] != v]
    if unexpected:
        raise ValueError(f"shared optimizer, sampler, loss or evaluation plan differs: {unexpected}")
    if not torch.equal(rs, hs):
        raise ValueError("final sampler state differs")
    if hi["parameter_count"] - ri["parameter_count"] != history.PLAN["history_projection_parameters"]:
        raise ValueError("parameter delta differs from prepared history projection")
    if any(hr["validation"][k] != v for k, v in rr["validation"].items() if k != "metrics"):
        raise ValueError("validation population/protocol differs")
    comparison = compare_metrics(rr, hr)
    episode = min(data.manifest["split"]["val_episodes"])
    eligible = data.val[data.episodes[data.val] == episode]
    index = int(eligible[len(eligible) // 2])
    previous = int(data.previous_indices[index])
    prior = json.loads((PRIOR / "report.json").read_text())
    if prior["sample"]["global_index"] != index or prior["runs"]["residual"]["checkpoint"] != str(residual.TRAIN_ROOT / "final.pt"):
        raise ValueError("fixed example/checkpoint differs from previous inspection")
    z, context, actions, target = data.batch([index])
    with torch.inference_mode():
        forecasts = {"residual": rm(z, context.current_xy, actions),
            "history": hm(z, context, actions), "persistence": z[:, None, None].expand_as(target)}
        sample = {name: {
            "dice_per_step": base.dice_cost(p, target)[0, 0].cpu().tolist(),
            "terminal_goal_cost_abs_error": float((base.dice_cost(p[0, 0, -1], data.goal) - base.dice_cost(target[0, 0, -1], data.goal)).abs()),
            "terminal_area_relative_error": float((p[0, 0, -1].sum() - target[0, 0, -1].sum()).abs() / target[0, 0, -1].sum()),
        } for name, p in forecasts.items()}
    with np.load(PRIOR / "fixed_window_predictions.npz", allow_pickle=False) as old:
        for key, value in (("current", z), ("current_agent_xy", context.current_xy), ("actions", actions),
                           ("target", target), ("goal", data.goal), ("residual", forecasts["residual"]), ("persistence", forecasts["persistence"])):
            if not np.array_equal(old[key], value.cpu().numpy()):
                raise ValueError(f"previous fixed input/target/reference changed: {key}")
    np.savez_compressed(args.out / "fixed_window_predictions.npz", current=z.cpu().numpy(),
        current_agent_xy=context.current_xy.cpu().numpy(), previous_grid=context.previous_grid.cpu().numpy(),
        previous_agent_xy=context.previous_xy.cpu().numpy(), history_valid=context.valid.cpu().numpy(),
        actions=actions.cpu().numpy(), target=target.cpu().numpy(), goal=data.goal.cpu().numpy(),
        **{name: p.cpu().numpy() for name, p in forecasts.items()})
    render(args.out, data, index, previous, forecasts, sample)
    result = {
        "status": "completed_fixed_final_history_comparison", "plan": PLAN,
        "runs": {"residual": ri, "history": hi}, "same_cache_split_normalization": True,
        "same_shared_training_plan": True, "sampler_final_state_equal": True,
        "validation_windows": hr["validation"]["eligible_windows"],
        "validation_episodes": hr["validation"]["eligible_episodes"], "comparison": comparison,
        "sample": {"episode_index": int(episode), "frame_index": int(data.frames[index]), "global_index": index,
            "previous_global_index": previous, "previous_frame_index": int(data.frames[previous]),
            "history_valid": bool(data.history_valid[index]), "metrics": sample,
            "matches_previous_input_target_residual_prediction_exactly": True},
        "new_forward_calls": 2, "full_validation_reruns": 0, "visual_status": "not_viewed",
        "elapsed_seconds": time.monotonic() - started,
        "decision": "retain_ACT_single_seed_prediction_only_capacity_confounded_no_policy_benefit_claim",
    }
    write(args.out / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
