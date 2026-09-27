"""Read fixed-final reports and inspect the same preselected validation window.

No fitting, full validation rerun, candidate ranking or environment execution.
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
    from .inspect_pusht_object_dynamics import grid_image, write
else:
    import pusht_object_dynamics as base
    import pusht_object_residual_dynamics as residual
    from inspect_pusht_object_dynamics import grid_image, write

ROOT = Path(__file__).resolve().parents[1]
PLAN = {"schema": "pusht_object_residual_final_comparison_v1",
    "checkpoint_selection": "fixed_final10000_only",
    "validation": "read_existing_final_reports_no_validation_rerun",
    "visual_selection": "same_lowest_ID_validation_episode_middle_eligible_window",
    "display_offsets": [1, 4, 8], "display_range": [0, 1],
    "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
    "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False}


def read_run(api, identity):
    report = json.loads((api.TRAIN_ROOT / "report.json").read_text())
    run = json.loads((api.TRAIN_ROOT / "run.json").read_text())
    status = json.loads((api.TRAIN_ROOT / "status.json").read_text())
    rows = [json.loads(line) for line in (api.TRAIN_ROOT / "metrics.jsonl").read_text().splitlines()]
    if report["schema"] != api.SCHEMA or report["status"] != "completed_fixed_object_dynamics_training":
        raise ValueError("requires completed matching fixed-final training")
    if status != {"status": "completed", "step": 10000} or report["optimizer_steps"] != 10000:
        raise ValueError("training budget not completed")
    if report["plan"] != api.PLAN or run["plan"] != api.PLAN:
        raise ValueError("training plan changed")
    if [r["step"] for r in rows] != list(range(100, 10001, 100)):
        raise ValueError("missing/duplicated training log intervals")
    if not np.isfinite([[r["loss_interval_mean"], r["grad_norm"], r["session_seconds"]] for r in rows]).all():
        raise ValueError("nonfinite training log")
    checkpoint = api.TRAIN_ROOT / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(x["cache_identity"] != identity for x in (payload, report, run)):
        raise ValueError("checkpoint/report/current-cache mismatch")
    states = payload["optimizer"]["state"]
    if {int(v["step"]) for v in states.values()} != {10000} or len(states) != len(payload["optimizer"]["param_groups"][0]["params"]):
        raise ValueError("optimizer parameter states have not all reached10000")
    model = api.load_final(checkpoint)
    return report, model, payload["sampler_rng"], {
        "checkpoint": str(checkpoint), "optimizer_state_steps": [10000],
        "logged_intervals": len(rows), "finite_logs": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": report["elapsed_seconds"],
        "first_last_interval_loss": [rows[0]["loss_interval_mean"], rows[-1]["loss_interval_mean"]]}


def changes(value, reference):
    return {"absolute": value - reference,
        "relative_percent": 100 * (value - reference) / reference if reference else None}


def compare_metrics(base_report, residual_report):
    b, r = (x["validation"]["metrics"] for x in (base_report, residual_report))
    if b["persistence"] != r["persistence"]:
        raise ValueError("persistence baseline differs across final evaluations")
    result = {}
    for aggregation in ("window_mean", "episode_macro"):
        result[aggregation] = {}
        for metric in base.PLAN["heldout_metrics"]:
            bv, rv = (x["observed_actions"][aggregation][metric] for x in (b, r))
            pv = b["persistence"][aggregation][metric]
            result[aggregation][metric] = {"base": bv, "residual": rv, "persistence": pv,
                "residual_minus_base": changes(rv, bv), "residual_minus_persistence": changes(rv, pv),
                "base_mean_action_ablation": b["mean_action_inference_ablation"][aggregation][metric],
                "residual_mean_action_ablation": r["mean_action_inference_ablation"][aggregation][metric]}
    result["per_episode_direction"] = {}
    re = r["observed_actions"]["per_episode"]
    for reference_name, reference in (("base", b["observed_actions"]), ("persistence", b["persistence"])):
        be = reference["per_episode"]
        if set(re) != set(be):
            raise ValueError("validation episode set changed")
        result["per_episode_direction"][reference_name] = {}
        for metric in base.PLAN["heldout_metrics"]:
            delta = np.array([re[ep][metric] - be[ep][metric] for ep in sorted(re)])
            result["per_episode_direction"][reference_name][metric] = {
                "better": int((delta < 0).sum()), "equal": int((delta == 0).sum()), "worse": int((delta > 0).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_residual_comparison_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "started.json", PLAN)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = base.ObjectData()
    br, bm, bs, bi = read_run(base, data.identity)
    rr, rm, rs, ri = read_run(residual, data.identity)
    expected_difference = {"schema", "architecture"}
    if any(rr["plan"][k] != v for k, v in br["plan"].items() if k not in expected_difference):
        raise ValueError("base/residual shared budget, loss or evaluation setting differs")
    if not torch.equal(bs, rs) or bi["parameter_count"] != ri["parameter_count"]:
        raise ValueError("sampler final state or parameter count differs")
    if any(br["validation"][k] != rr["validation"][k] for k in br["validation"] if k != "metrics"):
        raise ValueError("final evaluation populations/protocols differ")
    comparisons = compare_metrics(br, rr)
    episode = min(data.manifest["split"]["val_episodes"])
    eligible = data.val[data.episodes[data.val] == episode]
    index = int(eligible[len(eligible) // 2])
    prior = json.loads((ROOT / "simulation_output/pusht_object_dynamics_inspection_v1/report.json").read_text())
    if prior["sample"]["global_index"] != index or prior["checkpoint"] != str(base.TRAIN_ROOT / "final.pt"):
        raise ValueError("fixed visual window differs from previous inspection")
    z, state, actions, target = data.batch([index])
    with torch.inference_mode():
        forecasts = {"base": bm(z, state, actions), "residual": rm(z, state, actions),
            "persistence": z[:, None, None].expand_as(target)}
        sample = {name: {"dice_per_step": base.dice_cost(p, target)[0, 0].cpu().tolist(),
            "terminal_goal_cost_abs_error": float((base.dice_cost(p[0, 0, -1], data.goal) - base.dice_cost(target[0, 0, -1], data.goal)).abs()),
            "terminal_area_relative_error": float((p[0, 0, -1].sum() - target[0, 0, -1].sum()).abs() / target[0, 0, -1].sum())}
            for name, p in forecasts.items()}
    with np.load(ROOT / "simulation_output/pusht_object_dynamics_inspection_v1/fixed_window_predictions.npz", allow_pickle=False) as old:
        for key, value in (("current", z), ("actions", actions), ("current_agent_xy", state), ("target", target), ("observed_actions", forecasts["base"])):
            if not np.array_equal(old[key], value.cpu().numpy()):
                raise ValueError("previous fixed input/target/base forecast changed")
    np.savez_compressed(args.out / "fixed_window_predictions.npz", current=z.cpu().numpy(),
        actions=actions.cpu().numpy(), current_agent_xy=state.cpu().numpy(), target=target.cpu().numpy(),
        goal=data.goal.cpu().numpy(), **{k: v.cpu().numpy() for k, v in forecasts.items()})
    title = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    sheet = Image.new("RGB", (1280, 1170), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "原模型 vs 残差模型：相同 10,000 步；固定验证片段（非优选案例）", font=title, fill="black")
    draw.text((12, 47), f"episode={episode}，frame={data.frames[index]}；图中均为 RGB 提取/预测的占据网格，不是生成 RGB。", font=font, fill="black")
    for column, offset in enumerate((0, 1, 4, 8)):
        left = 170 + 270 * column
        draw.text((left, 87), "当前输入" if offset == 0 else f"未来 +{offset} 步", font=title, fill="black")
        sheet.paste(grid_image(data.grids[index + offset].cpu().numpy()), (left, 132))
        for row, name in enumerate(("base", "residual", "persistence"), start=1):
            top = 132 + 232 * row
            if offset:
                sheet.paste(grid_image(forecasts[name][0, 0, offset - 1].cpu().numpy()), (left, top))
            else:
                draw.text((left, top + 85), "同一当前网格 / XY / 动作" if name != "persistence" else "重复当前网格", font=font, fill="black")
    for row, label in enumerate(("实际观测网格", "原模型预测", "残差模型预测", "保持不动基线")):
        draw.text((10, 222 + row * 232), label, font=font, fill="black")
    for row, (key, label) in enumerate((("base", "原模型"), ("residual", "残差模型"), ("persistence", "保持不动"))):
        draw.text((15 + 405 * row, 1070), f"{label}：终点 Dice {sample[key]['dice_per_step'][-1]:.6f}", font=font, fill="black")
    draw.text((12, 1110), "白色=0，黑色=1，所有网格范围固定；单片段不能代表整体，更不能证明 ACT 策略收益。", font=font, fill="black")
    sheet.save(args.out / "fixed_validation_comparison_zh.png")
    result = {"status": "completed_fixed_final_residual_comparison", "plan": PLAN,
        "runs": {"base": bi, "residual": ri}, "same_cache_split_normalization": True,
        "same_shared_training_plan": True, "sampler_final_state_equal": True,
        "validation_windows": rr["validation"]["eligible_windows"],
        "validation_episodes": rr["validation"]["eligible_episodes"], "comparison": comparisons,
        "sample": {"episode_index": int(episode), "frame_index": int(data.frames[index]),
            "global_index": index, "matches_previous_input_target_base_prediction_exactly": True, "metrics": sample},
        "new_forward_calls": 2, "full_validation_reruns": 0, "visual_status": "not_viewed",
        "elapsed_seconds": time.monotonic() - started,
        "decision": "retain_ACT_no_policy_benefit_claim_single_seed_fixed_prediction_comparison"}
    write(args.out / "report.json", result)
    print(json.dumps({"status": result["status"], "comparison": comparisons,
        "sample": result["sample"], "runs": result["runs"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
