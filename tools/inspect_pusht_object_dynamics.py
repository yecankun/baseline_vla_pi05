"""Read back the fixed final training and view one predetermined held-out window.

No training/environment execution, candidate selection or validation sweep.
The lowest-ID validation episode's middle eligible window is fixed by order.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_object_dynamics as wm
    from . import pusht_object_goal as vision
    from . import pusht_bc_act_training_data as dm
else:
    import pusht_object_dynamics as wm
    import pusht_object_goal as vision
    import pusht_bc_act_training_data as dm

ROOT = Path(__file__).resolve().parents[1]
PLAN = {"schema": "pusht_object_dynamics_final_inspection_v1",
    "selection": "lowest_ID_validation_episode_middle_eligible_window_not_loss_or_outcome",
    "display_offsets": [1, 4, 8], "display_range": [0, 1],
    "checkpoint": "fixed_final10000_only", "optimizer_steps": 0,
    "environment_steps": 0, "hardware_actions": 0, "candidate_ranking": False}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def grid_image(grid):
    # White0 -> black1; fixed scale for observations, predictions and persistence.
    return Image.fromarray(np.rint(255 * (1 - np.clip(grid, 0, 1))).astype(np.uint8)).convert("RGB").resize((208, 208), Image.Resampling.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_object_dynamics_inspection_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "started.json", PLAN)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    report = json.loads((wm.TRAIN_ROOT / "report.json").read_text())
    run = json.loads((wm.TRAIN_ROOT / "run.json").read_text())
    status = json.loads((wm.TRAIN_ROOT / "status.json").read_text())
    rows = [json.loads(line) for line in (wm.TRAIN_ROOT / "metrics.jsonl").read_text().splitlines()]
    if report["status"] != "completed_fixed_object_dynamics_training" or status != {"status": "completed", "step": 10000}:
        raise ValueError("requires completed fixed training, not a partial/resumed trial")
    if report["plan"] != wm.PLAN or run["plan"] != wm.PLAN or report["optimizer_steps"] != 10000:
        raise ValueError("training plan/step changed")
    if [r["step"] for r in rows] != list(range(100, 10001, 100)):
        raise ValueError("training log missing/duplicated intervals; inspect resume history separately")
    if not np.isfinite([[r["loss_interval_mean"], r["grad_norm"], r["session_seconds"]] for r in rows]).all():
        raise ValueError("nonfinite saved training metric")
    data = wm.ObjectData()
    payload = torch.load(wm.TRAIN_ROOT / "final.pt", map_location="cpu", weights_only=False)
    if payload["cache_identity"] != data.identity or report["cache_identity"] != data.identity or run["cache_identity"] != data.identity:
        raise ValueError("trained checkpoint/report/cache identity mismatch")
    optimizer_steps = sorted({int(v["step"]) for v in payload["optimizer"]["state"].values()})
    if optimizer_steps != [10000] or len(payload["optimizer"]["state"]) != len(payload["optimizer"]["param_groups"][0]["params"]):
        raise ValueError("not all AdamW parameter states completed10000 updates")
    model = wm.load_final(wm.TRAIN_ROOT / "final.pt")
    episode = min(data.manifest["split"]["val_episodes"])
    eligible = data.val[data.episodes[data.val] == episode]
    if not len(eligible):
        raise ValueError("no eligible window in predetermined validation episode")
    index = int(eligible[len(eligible) // 2])
    z, state, actions, target = data.batch([index])
    with torch.inference_mode():
        predicted = model(z, state, actions)
        mean_action = model(z, state, model.action_mean.expand_as(actions))
        persistence = z[:, None, None].expand_as(target)
        sample = {}
        for name, p in (("observed_actions", predicted), ("mean_action_inference_ablation", mean_action), ("persistence", persistence)):
            sample[name] = {"dice_per_step": wm.dice_cost(p, target)[0, 0].cpu().tolist(),
                "terminal_goal_cost": float(wm.dice_cost(p[0, 0, -1], data.goal)),
                "terminal_goal_cost_abs_error": float((wm.dice_cost(p[0, 0, -1], data.goal) - wm.dice_cost(target[0, 0, -1], data.goal)).abs())}
    np.savez_compressed(args.out / "fixed_window_predictions.npz", current=z.cpu().numpy(),
        observed_actions=predicted.cpu().numpy(), mean_actions=mean_action.cpu().numpy(),
        persistence=persistence.cpu().numpy(), target=target.cpu().numpy(),
        actions=actions.cpu().numpy(), current_agent_xy=state.cpu().numpy(), goal=data.goal.cpu().numpy())
    # Actual future RGB is fetched only for post-prediction visual evidence.
    source = dm.load_training_data()
    if source.split != data.manifest["split"] or source.binding["revision"] != data.manifest["source_revision"]:
        raise ValueError("source RGB binding changed")
    for offset in range(9):
        observed = vision.observe_rgb(source._images[index + offset])
        if not observed.valid or not np.array_equal(observed.features.values[0], data.grids[index + offset].cpu().numpy()):
            raise ValueError("display RGB does not match the cached observed grid")
    title = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    sheet = Image.new("RGB", (1280, 1120), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "训练后物体网格预测：固定验证片段；同一占据值灰度范围 [0,1]", font=title, fill="black")
    draw.text((12, 46), f"episode={episode}，frame={data.frames[index]}；最低编号验证 episode 的中间窗口，非优选案例。", font=font, fill="black")
    for column, offset in enumerate((0, 1, 4, 8)):
        left = 170 + 270 * column
        draw.text((left, 80), "当前输入" if offset == 0 else f"未来 +{offset} 步", font=title, fill="black")
        sheet.paste(Image.fromarray(source._images[index + offset]).resize((208, 208)), (left, 112))
        sheet.paste(grid_image(data.grids[index + offset].cpu().numpy()), (left, 352))
        if offset:
            sheet.paste(grid_image(predicted[0, 0, offset - 1].cpu().numpy()), (left, 592))
            sheet.paste(grid_image(z[0].cpu().numpy()), (left, 832))
            draw.text((left, 1047), f"Dice：模型 {sample['observed_actions']['dice_per_step'][offset-1]:.3f}", font=font, fill="black")
            draw.text((left, 1072), f"保持不动 {sample['persistence']['dice_per_step'][offset-1]:.3f}（低优）", font=font, fill="black")
        else:
            draw.text((left, 656), "以上方当前网格、", font=font, fill="black")
            draw.text((left, 684), "当前 XY 和动作作为输入", font=font, fill="black")
            draw.text((left, 918), "重复当前物体网格", font=font, fill="black")
    for row, label in enumerate(("实际 RGB", "RGB 物体网格", "模型预测网格", "保持不动基线")):
        draw.text((10, 194 + row * 240), label, font=font, fill="black")
    sheet.save(args.out / "fixed_validation_prediction_zh.png")
    write(args.out / "report.json", {"status": "completed_fixed_object_grid_prediction_inspection", "plan": PLAN,
        "checkpoint": str(wm.TRAIN_ROOT / "final.pt"), "logged_intervals": len(rows),
        "first_last_logged_step": [rows[0]["step"], rows[-1]["step"]], "optimizer_state_steps": optimizer_steps,
        "source_cache_checkpoint_match": True, "validation_RGB_grid_match": True,
        "sample": {"episode_index": int(episode), "global_index": index, "frame_index": int(data.frames[index]),
                   "metrics": sample, "selection_is_outcome_independent": True},
        "raw_RGB_is_observed_not_generated": True, "predictions_are_grids_not_RGB": True,
        "visual_status": "not_viewed", "optimizer_steps": 0, "environment_steps": 0,
        "hardware_actions": 0, "policy_benefit_evaluated": False, "candidate_ranking_evaluated": False,
        "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "completed", "episode": int(episode), "frame": int(data.frames[index]),
                      "sample": sample, "out": str(args.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
