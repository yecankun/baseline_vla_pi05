"""Train the authorized, fixed terminal-score pair; never run the environment.

Reuses DirectActionScorer and its loss unchanged. --train is explicit; --resume
continues only this pair and skips completed arms. No candidate/model selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import pusht_direct_action_scorer as ds
else:
    import pusht_direct_action_scorer as ds

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "simulation_output/pusht_terminal_score_pair_pretrain_v1"
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_terminal_score_pair_v1")
SCHEMA = "pusht_terminal_score_pair_v1"
VARIANTS = ("visual_terminal", "reward_terminal")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PairedData:
    """Only original observable inputs reach the network; labels remain separate."""

    def __init__(self, prepared=PREPARED):
        self.contract = c = read(prepared / "comparison_contract.json")
        prepared_report = read(prepared / "report.json")
        self.base = b = ds.base.ObjectData()
        if (c["schema"] != SCHEMA or c["variants"] != list(VARIANTS)
                or c["cache_identity"] != b.identity
                or c["architecture"] != ds.PLAN["architecture"]
                or c["candidate_rule"] != ds.PLAN["candidate_rule"]
                or c["fixed_budget"] != {k: ds.PLAN[k] for k in c["fixed_budget"]}
                or not prepared_report["all_reward_targets_equal_static_native_geometry_float32"]):
            raise ValueError("requires the completed fixed paired-target preparation")
        with np.load(prepared / "paired_terminal_targets.npz", allow_pickle=False) as a:
            self.anchors = a["anchors"]
            raw = a["targets"]
            expected = np.concatenate((b.train, b.val))
            if not (np.array_equal(self.anchors, expected)
                    and np.array_equal(a["is_train"], np.arange(len(expected)) < len(b.train))
                    and np.array_equal(a["episodes"], b.episodes[expected])
                    and np.array_equal(a["frames"], b.frames[expected])
                    and raw.shape == (23960, 2) and np.isfinite(raw).all()):
                raise ValueError("target rows or original cohort changed")
        self.train_count = len(b.train)
        self.val_rows = np.arange(self.train_count, len(self.anchors))
        for j, variant in enumerate(VARIANTS):
            actual = raw[:self.train_count, j].astype(np.float64)
            stats = c["target_normalization"][variant]
            if stats["mean"] != float(actual.mean()) or stats["std"] != float(actual.std()):
                raise ValueError("target statistics must be original training-only values")
        self.raw = raw
        self.targets = torch.from_numpy(raw).cuda()

    def inputs(self, rows):
        ids = torch.as_tensor(self.anchors[rows], device="cuda", dtype=torch.long)
        actions = ids[:, None] + torch.arange(ds.HORIZON, device="cuda")[None]
        return self.base.grids[ids], self.base.states[ids], self.base.actions[actions][:, None]


def score_metrics(prediction, target, episodes):
    """Report raw scalar units, with episode-macro and exact-zero strata."""
    error = np.asarray(prediction, dtype=np.float64) - target
    if not np.isfinite(error).all():
        raise ValueError("nonfinite final validation prediction")

    def metric(mask):
        e = error[mask]
        return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
                "bias": float(e.mean())}

    result = {}
    for name, mask in (("all", np.ones(len(error), dtype=bool)), ("zero_target", target == 0),
                       ("nonzero_target", target != 0)):
        per_episode = {str(int(e)): metric(mask & (episodes == e)) for e in np.unique(episodes[mask])}
        result[name] = {"windows": int(mask.sum()), "episodes": len(per_episode),
                        "window_mean": metric(mask) if mask.any() else None,
                        "episode_macro": {k: float(np.mean([x[k] for x in per_episode.values()]))
                                          for k in ("mae", "rmse", "bias")} if per_episode else None,
                        "per_episode": per_episode}
    return result


def load_final(path, device="cuda"):
    """Load this pair only; old DS0's predict_effect method now returns a terminal score."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if (payload["schema"] != SCHEMA or payload["variant"] not in VARIANTS
            or payload["kind"] != "final" or payload["step"] != 10000):
        raise ValueError("requires a fixed-final terminal-score checkpoint, not old DS0")
    model = ds.DirectActionScorer(payload["contract"]["cache_identity"]["normalization"],
                                 payload["target_stats"], payload["model"]["goal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False), payload


@torch.inference_mode()
def evaluate(model, data, column, out):
    model.eval()
    predictions = []
    for lo in range(0, len(data.val_rows), 64):
        rows = data.val_rows[lo:lo + 64]
        predictions.append(model.predict_effect(*data.inputs(rows))[:, 0].cpu().numpy())
    prediction = np.concatenate(predictions)
    target = data.raw[data.val_rows, column]
    anchors = data.anchors[data.val_rows]
    episodes = data.base.episodes[anchors]
    baseline = np.full_like(target, float(model.target_mean))
    np.savez_compressed(out / "validation_predictions.npz", anchors=anchors, episodes=episodes,
                        frames=data.base.frames[anchors], prediction=prediction, target=target)
    return {"scorer": score_metrics(prediction, target, episodes),
            "training_mean": score_metrics(baseline, target, episodes),
            "prediction_units": data.contract["targets"][VARIANTS[column]],
            "validation_for_checkpoint_selection": False, "fresh_test": False,
            "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False}


def train_arm(data, variant, root, resume, stop):
    out = root / variant
    c, budget = data.contract, data.contract["fixed_budget"]
    column = VARIANTS.index(variant)
    if (out / "report.json").exists():
        report = read(out / "report.json")
        if not resume or report["contract"] != c or report["optimizer_steps"] != budget["steps"]:
            raise ValueError("preserve completed arm; only resume this same pair")
        return report
    stats = c["target_normalization"][variant]
    torch.manual_seed(budget["seed"])
    model = ds.DirectActionScorer(data.base.stats, stats, data.base.goal).cuda()
    expected = torch.load(root / "initial_trainable.pt", map_location="cpu", weights_only=True)
    if not all(torch.equal(p.detach().cpu(), expected[n]) for n, p in model.named_parameters()):
        raise ValueError("paired initial trainable parameters differ")
    # Reset NumPy per arm and compare the complete ordered schedule, not a hash.
    schedule = np.random.default_rng(budget["seed"]).integers(
        data.train_count, size=(budget["steps"], budget["batch_size"]), dtype=np.int32)
    if not np.array_equal(schedule, np.load(root / "sampled_rows.npy", allow_pickle=False)):
        raise ValueError("paired optimizer-update sampling differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"])
    first, previous_seconds, previous_peak = 0, 0., 0
    if (out / "last.pt").exists():
        if not resume:
            raise FileExistsError(f"preserve partial run: {out}")
        payload = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if payload["contract"] != c or payload["variant"] != variant or payload["schema"] != SCHEMA:
            raise ValueError("resume checkpoint does not match this arm/contract")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        first = payload["step"]
        previous_seconds, previous_peak = payload["training_seconds"], payload["peak_cuda_allocated_bytes"]
        if not 0 <= first <= budget["steps"]:
            raise ValueError("checkpoint step is outside the fixed budget")
        del payload
    else:
        if out.exists():
            raise FileExistsError(f"partial arm has no resumable last.pt: {out}")
        out.mkdir()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()

    def save(step, kind="resume"):
        path = out / ("final.pt" if kind == "final" else "last.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"schema": SCHEMA, "variant": variant, "kind": kind, "step": step,
                    "contract": c, "target_stats": stats, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "next_schedule_row": step,
                    "training_seconds": previous_seconds + time.monotonic() - started,
                    "peak_cuda_allocated_bytes": max(previous_peak, torch.cuda.max_memory_allocated()),
                    "target_is_terminal_not_delta": True}, temporary)
        os.replace(temporary, path)

    if first == 0:
        save(0)
    model.train()
    interval, count = 0., 0
    for step in range(first + 1, budget["steps"] + 1):
        rows = schedule[step - 1]
        inputs = data.inputs(rows)
        target = data.targets[torch.as_tensor(rows, device="cuda", dtype=torch.long), column, None]
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(*inputs), target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite {variant} loss at update {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), budget["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        interval, count = interval + float(loss.detach()), count + 1
        if step % budget["log_every"] == 0:
            row = {"variant": variant, "step": step, "mean_standardized_mse": interval / count,
                   "grad_norm": float(grad), "resume_from_step": first,
                   "training_seconds": previous_seconds + time.monotonic() - started}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if step % budget["checkpoint_every"] == 0:
                print(json.dumps(row), flush=True)
            interval, count = 0., 0
        if step % budget["checkpoint_every"] == 0 or stop["requested"]:
            save(step)
            write(out / "status.json", {"status": "interrupted" if stop["requested"] else "training", "step": step})
        if stop["requested"]:
            return None
    torch.cuda.synchronize()
    save(budget["steps"], "final")
    training_seconds = previous_seconds + time.monotonic() - started
    peak = max(previous_peak, torch.cuda.max_memory_allocated())
    validation_started = time.monotonic()
    validation = evaluate(model, data, column, out)
    loaded, payload = load_final(out / "final.pt")
    if not all(int(state["step"]) == budget["steps"] for state in payload["optimizer"]["state"].values()):
        raise ValueError("final optimizer state did not reach exactly the fixed budget")
    with torch.inference_mode():
        inputs = data.inputs(data.val_rows[:2])
        if not torch.equal(model(*inputs), loaded(*inputs)):
            raise ValueError("final checkpoint reload prediction mismatch")
    report = {"schema": SCHEMA, "status": "completed_fixed_terminal_score_training", "variant": variant,
              "contract": c, "parameter_count": sum(p.numel() for p in model.parameters()),
              "optimizer_steps": budget["steps"], "initial_parameters_exactly_paired": True,
              "all_ordered_sampled_rows_exactly_paired": True, "checkpoint_reload_exact": True,
              "final_optimizer_state_steps_exact": True, "validation": validation,
              "training_seconds": training_seconds, "validation_reload_seconds": time.monotonic() - validation_started,
              "peak_cuda_allocated_bytes": peak, "resume_from_step": first,
              "environment_steps": 0, "hardware_actions": 0, "candidate_ranking_evaluated": False,
              "final_checkpoint": str(out / "final.pt")}
    write(out / "report.json", report)
    write(out / "status.json", {"status": "completed", "step": budget["steps"]})
    print(json.dumps({"variant": variant, "training_seconds": training_seconds,
                      "validation": validation["scorer"]["all"]}, ensure_ascii=False), flush=True)
    del model, optimizer, loaded, payload
    torch.cuda.empty_cache()
    return report


def validation_sheet(data, root):
    """First six validation episodes, middle eligible window each; no cherry picking."""
    saved = [np.load(root / variant / "validation_predictions.npz", allow_pickle=False) for variant in VARIANTS]
    anchors, episodes = saved[0]["anchors"], saved[0]["episodes"]
    chosen = [np.flatnonzero(episodes == ep)[np.count_nonzero(episodes == ep) // 2]
              for ep in np.unique(episodes)[:6]]
    sheet = Image.new("RGB", (1170, 810), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font, small = ImageFont.truetype(font_path, 23), ImageFont.truetype(font_path, 18)
    draw.text((20, 12), "末端评分配对训练：固定验证片段的预测读回", font=font, fill="black")
    draw.text((20, 48), "两组监督单位不同；这里不是候选动作排序，也不是策略成功率。", font=small, fill="#444444")
    for k, row in enumerate(chosen):
        x, y = 20 + k % 3 * 385, 90 + k // 3 * 335
        anchor = int(anchors[row])
        draw.text((x, y), f"episode {int(episodes[row])} / frame {int(data.base.frames[anchor])}", font=small, fill="black")
        for j, (name, frame) in enumerate((("当前输入 t", anchor), ("真实后继 t+8", anchor + ds.HORIZON))):
            left = x + j * 175
            draw.text((left, y + 32), name, font=small, fill="#444444")
            grid = data.base.grids[frame].cpu().numpy()
            tile = Image.fromarray(np.rint(255 * (1 - grid)).astype(np.uint8)).resize((144, 144), Image.Resampling.NEAREST)
            sheet.paste(tile, (left, y + 61))
        for j, name in enumerate(("视觉分数", "任务 reward")):
            truth, prediction = float(saved[j]["target"][row]), float(saved[j]["prediction"][row])
            draw.text((x, y + 219 + j * 29), f"{name}：真值 {truth:+.4f} 预测 {prediction:+.4f}", font=small, fill="black")
        draw.text((x, y + 281), "后继图像和真值仅用于离线监督/评估", font=small, fill="#666666")
    draw.text((20, 767), "固定取前六个验证 episode 的中间窗口；单种子；保留普通 ACT / real10。", font=small, fill="black")
    sheet.save(root / "validation_score_readback_zh.png")
    for values in saved:
        values.close()
    return [int(anchors[row]) for row in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT)
    args = parser.parse_args()
    if (args.out / "report.json").exists() or (args.out.exists() and not args.resume):
        raise FileExistsError(f"preserve existing run: {args.out}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = PairedData()
    budget = data.contract["fixed_budget"]
    if args.resume:
        if read(args.out / "run.json")["contract"] != data.contract:
            raise ValueError("cannot resume a changed paired contract")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(budget["seed"])
        initial = ds.DirectActionScorer(data.base.stats, data.contract["target_normalization"][VARIANTS[0]], data.base.goal)
        torch.save({name: p.detach().cpu().clone() for name, p in initial.named_parameters()}, args.out / "initial_trainable.pt")
        del initial
        schedule = np.random.default_rng(budget["seed"]).integers(
            data.train_count, size=(budget["steps"], budget["batch_size"]), dtype=np.int32)
        np.save(args.out / "sampled_rows.npy", schedule)
        write(args.out / "run.json", {"schema": SCHEMA, "contract": data.contract,
              "authorization": "user_approved_implementation_and_fixed_paired_training_20260916",
              "historical_preparation_authorization_flag_unchanged": True,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "source_entrypoint": str(Path(__file__).resolve())})
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested; save current update and do not start another arm", flush=True)

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    reports = {}
    for variant in VARIANTS:
        report = train_arm(data, variant, args.out, args.resume, stop)
        if report is None or stop["requested"]:
            write(args.out / "status.json", {"status": "interrupted", "active_variant": variant})
            return 2
        reports[variant] = report
    examples = validation_sheet(data, args.out)
    compact = {v: {"scorer": r["validation"]["scorer"]["all"],
                   "training_mean": r["validation"]["training_mean"]["all"],
                   "training_seconds": r["training_seconds"],
                   "peak_cuda_allocated_bytes": r["peak_cuda_allocated_bytes"],
                   "final_checkpoint": r["final_checkpoint"]} for v, r in reports.items()}
    write(args.out / "report.json", {"schema": SCHEMA, "status": "completed_matched_terminal_training",
          "contract": data.contract, "results": compact, "optimizer_steps_per_arm": budget["steps"],
          "total_optimizer_steps": 2 * budget["steps"], "parameter_count_each": reports[VARIANTS[0]]["parameter_count"],
          "same_initial_trainable_parameters": True, "same_complete_ordered_sampled_rows": True,
          "final_optimizer_and_reload_verified_both": True, "train_windows": data.train_count,
          "validation_windows": len(data.val_rows), "visual_examples": examples, "visual_status": "not_viewed",
          "environment_steps": 0, "hardware_actions": 0, "candidate_ranking_evaluated": False,
          "decision": "training_complete_no_cross_unit_error_winner_keep_ACT_pending_candidate_ranking"})
    write(args.out / "status.json", {"status": "completed", "optimizer_steps_per_arm": budget["steps"]})
    print(json.dumps({"status": "completed_matched_terminal_training", "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
