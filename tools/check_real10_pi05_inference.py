"""Check final real10 checkpoints and observation-only inference; never actuate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from export_openpi_compat_to_lerobot import read_json, read_jsonl, write_json
from real10_pi05_policy import Real10PI05Policy
from train_pi05_mixed_head_adapter import piper_metrics


def training_summary(root):
    result = {}
    for name, key, expected_steps in (("elite", "train_loss", 3000), ("piper", "piper_loss", 1000)):
        log = read_json(root / name / "train_log.json")
        rows = log["steps"]
        if [row["step"] for row in rows] != list(range(1, expected_steps + 1)):
            raise ValueError(f"incomplete {name} training steps")
        if any(not np.isfinite(row[key]) or not np.isfinite(row["grad_norm"]) for row in rows):
            raise ValueError(f"non-finite {name} loss/gradient")
        if log["val_samples"] or not log["train_only"] or log["final_val"] is not None:
            raise ValueError(f"unexpected validation in {name}")
        result[name] = {"steps": len(rows), "train_samples": log["train_samples"],
                        "elapsed_seconds": log["elapsed_sec"],
                        "first100_mean_loss": float(np.mean([x[key] for x in rows[:100]])),
                        "last100_mean_loss": float(np.mean([x[key] for x in rows[-100:]])),
                        "last_loss": rows[-1][key], "finite_loss_and_gradients": True}
    return result


def render_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    samples = report["sample_predictions"]
    target = np.array([s["target_elite_tcp_delta_6d"][:3] for s in samples])
    predicted = np.array([s["prediction"]["elite_tcp_delta_6d"][:3] for s in samples])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for dimension, ax in enumerate(axes):
        ax.plot(target[:, dimension], "o-", lw=1, ms=3, label="记录动作")
        ax.plot(predicted[:, dimension], "x--", lw=1, ms=4, label="模型输出")
        ax.set_ylabel(f"Δ{'xyz'[dimension]}（毫米）", fontproperties=font)
        ax.grid(alpha=.2)
        ax.legend(prop=font, loc="upper right")
    axes[-1].set_xlabel("诊断样本编号（每条轨迹的起点、首次递丝、最大平移；非连续时序）", fontproperties=font)
    fig.suptitle("真实十条数据：最终模型离线推理检查\n训练内诊断样本，不代表验证集效果或实机成功率", fontproperties=font, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    summary = training_summary(args.training_root)
    rows = read_jsonl(args.pack / "index.jsonl")
    manifest = read_json(args.pack / "manifest.json")
    with np.load(args.pack / "openpi_arrays.npz") as source:
        states = source["state_32"].copy()
        target_elite = source["elite_tcp_delta_6d"].copy()
        target_piper = source["piper_intent_id"].copy()
    model = Real10PI05Policy(args.training_root / "elite/final_policy.pt",
                            args.training_root / "piper/mixed_head_policy.pt", device=args.device)
    if model.metadata["elite_step"] != 3000 or model.metadata["piper_step"] != 1000:
        raise ValueError("unexpected checkpoint steps")
    print(f"final checkpoints loaded in {model.metadata['load_seconds']:.1f}s", flush=True)
    selected = []
    for episode in manifest["source_episodes"]:
        indices = [i for i, row in enumerate(rows) if row["source"]["episode"] == episode]
        first_feed = next(i for i in indices if target_piper[i] == 2)
        largest_motion = max(indices, key=lambda i: float(np.linalg.norm(target_elite[i, :3])))
        selected.extend(dict.fromkeys((indices[0], first_feed, largest_motion)))

    projected_states = []
    hook = model.policy.model.state_proj.register_forward_pre_hook(
        lambda module, inputs: projected_states.append(inputs[0].detach().cpu().clone()))
    examples = []
    try:
        for number, i in enumerate(selected, 1):
            row, state = rows[i], states[i]
            previous = {"piper_busy": bool(state[16]), "piper_step_after_command": float(state[6])} if state[27] else None
            images = {view: cv2.imread(str(args.pack / "images" / row[f"observation.images.{view}"])) for view in ("side", "top")}
            request = dict(side_bgr=images["side"], top_bgr=images["top"], task=row["task"],
                           elite_tcp_pose_6d=state[:6], previous_controller_state=previous)
            raw = model.observation(**request)
            np.testing.assert_array_equal(raw["observation.state"].numpy(), state)
            if any("action" in key or "target" in key for key in raw):
                raise ValueError("target leaked into observation-only request")
            expected_state = model.preprocessor(raw)["observation.state"].detach().cpu()
            before = len(projected_states)
            prediction = model.predict(**request)
            if len(projected_states) != before + 1:
                raise ValueError("the state token was not consumed exactly once in PI05 sampling")
            torch.testing.assert_close(projected_states[-1], expected_state, rtol=0, atol=0)
            if model.policy.model._project2026_state_context is not None:
                raise ValueError("state context was not cleared after prediction")
            examples.append({"pack_index": i, "sample_id": row["sample_id"], "task": row["task"],
                             "source": row["source"], "prediction": prediction,
                             "target_elite_tcp_delta_6d": target_elite[i].tolist(),
                             "target_piper_intent_id": int(target_piper[i])})
            print(f"observation-only inference {number}/{len(selected)}: state token consumed", flush=True)
    finally:
        hook.remove()

    probabilities = torch.cat([model.piper_probabilities(states[start:start+256],
                                 [row["task"] for row in rows[start:start+256]])
                               for start in range(0, len(rows), 256)])
    head_fit = piper_metrics(torch.from_numpy(target_piper), probabilities.argmax(-1))
    for example in examples:
        np.testing.assert_allclose(probabilities[example["pack_index"]].numpy(),
                                   example["prediction"]["diagnostics"]["piper_probabilities_retract_hold_feed"], atol=2e-5)
    errors = np.abs(np.array([example["prediction"]["elite_tcp_delta_6d"][:3] for example in examples]) - target_elite[selected, :3])
    latency = [e["prediction"]["diagnostics"]["inference_seconds"] for e in examples]
    report = {
        "schema": "project2026_real10_inference_check_v1", "status": "passed",
        "scope": "train-set fit and observation-only interface verification, not validation or hardware",
        "training": summary, "checkpoint_metadata": model.metadata,
        "state_token_calls": len(projected_states), "normalized_state_token_values_match": True,
        "observation_encoder_equals_pack": True, "labels_in_policy_request": False,
        "sample_selection": "per episode: first observation, first feed, largest translation norm; deduplicated",
        "sampled_elite_translation_mae_mm": float(errors.mean()),
        "sampled_elite_translation_mae_xyz_mm": errors.mean(axis=0).tolist(),
        "inference_latency_seconds": {"first": latency[0], "warm_median": float(np.median(latency[1:])),
                                      "warm_p95": float(np.quantile(latency[1:], .95)), "max": max(latency)},
        "piper_all_2719_training_rows": head_fit,
        "rotation_output_all_zero": all(e["prediction"]["elite_tcp_delta_6d"][3:] == [0, 0, 0] for e in examples),
        "sample_predictions": examples, "hardware_executed": False, "optimizer_steps_this_check": 0,
        "visual_status": "not_viewed", "elapsed_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "report.json", report)
    # A ready single-observation CLI request. Paths are absolute for onsite use;
    # it contains no source action, label, episode ID, or diagnostic target.
    i = selected[0]
    write_json(args.out / "example_request.json", {
        "side_image": str((args.pack / "images" / rows[i]["observation.images.side"]).resolve()),
        "top_image": str((args.pack / "images" / rows[i]["observation.images.top"]).resolve()),
        "task": rows[i]["task"], "elite_tcp_pose_6d": states[i, :6].tolist(), "previous_controller_state": None,
    })
    print(json.dumps({k: report[k] for k in ("status", "state_token_calls", "sampled_elite_translation_mae_mm",
                    "inference_latency_seconds", "piper_all_2719_training_rows", "elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("simulation_output/real10_pi05_train_v1"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_inference_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-report", type=Path, help="Windows only: render a downloaded report, no inference")
    args = parser.parse_args()
    if args.render_report:
        render_report(read_json(args.render_report), args.render_report.with_name("inference_preview_zh.png"))
    else:
        if args.out.exists():
            raise FileExistsError(f"preserve existing inference report: {args.out}")
        run(args)


if __name__ == "__main__":
    main()
