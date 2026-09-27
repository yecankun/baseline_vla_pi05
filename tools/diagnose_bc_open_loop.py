import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.train_dual_arm_baseline import DualArmCameraStatePolicy, DualArmGuidewireDataset


def quantile(values, p):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, p))


def stats(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.quantile(arr, 0.5)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def linf(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def l2(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum(diff * diff)))


def elite_current(sample, names):
    joints = sample.get("state", {}).get("robot_state", {}).get("elite_joints", {})
    return np.asarray([float(joints[name]) for name in names], dtype=np.float32)


def action_elite(sample, names):
    joints = sample.get("action", {}).get("elite_joints", {})
    return np.asarray([float(joints[name]) for name in names], dtype=np.float32)


def elite_tcp_pose(sample):
    state = sample.get("state", {})
    pose = state.get("elite_tcp_pose_6d")
    if pose is None:
        pose = state.get("robot_state", {}).get("elite_tcp_pose_6d")
    if pose is None:
        xyz = np.asarray(state.get("elirobot_pose", [math.nan] * 3), dtype=np.float32).reshape(3) * 1000.0
        return np.asarray([*xyz.tolist(), 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"elite_tcp_pose_6d expected 6 values, got {arr.size}")
    return arr.astype(np.float32)


def action_elite_tcp_delta(sample):
    action = sample.get("action", {})
    if "elite_tcp_delta_6d" in action:
        arr = np.asarray(action["elite_tcp_delta_6d"], dtype=np.float32).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"elite_tcp_delta_6d expected 6 values, got {arr.size}")
        return arr.astype(np.float32)
    if "elite_tcp_pose_6d" in action:
        target = np.asarray(action["elite_tcp_pose_6d"], dtype=np.float32).reshape(-1)
        if target.size != 6:
            raise ValueError(f"elite_tcp_pose_6d expected 6 values, got {target.size}")
        return (target.astype(np.float32) - elite_tcp_pose(sample)).astype(np.float32)
    raise ValueError("tcp_delta diagnostic requires action.elite_tcp_delta_6d or action.elite_tcp_pose_6d")


def sample_scalar(sample, key, default=math.nan):
    value = sample.get("state", {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = DualArmCameraStatePolicy(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt.get("action_dim", 4)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt, model


@torch.no_grad()
def predict_all(model, loader, device):
    preds = []
    targets = []
    for batch in loader:
        side = batch["side"].to(device)
        top = batch["top"].to(device)
        state = batch["state"].to(device)
        pred = model(side, top, state).detach().cpu().numpy()
        target = batch["action"].detach().cpu().numpy()
        preds.append(pred)
        targets.append(target)
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary):
    lines = [
        "# BC Open-Loop Diagnostic",
        "",
        f"checkpoint: `{summary['checkpoint']}`",
        f"manifest: `{summary['manifest']}`",
        f"samples: `{summary['sample_count']}`",
        f"action_mode: `{summary['action_mode']}`",
        f"elite_action_representation: `{summary['elite_action_representation']}`",
        f"elite_metric_space: `{summary.get('elite_metric_space', 'elite_joint_radians')}`",
        "",
        "## Key Metrics",
        "",
        "| metric | median | p95 | max | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        "piper_abs_error",
        "elite_linf_error",
        "elite_l2_error",
        "expert_elite_target_step_linf",
        "pred_elite_target_step_linf",
        "expert_elite_delta_from_current_linf",
        "pred_elite_delta_from_current_linf",
        "tip_to_magnetic",
        "contact_strength",
        "distance_to_wall",
    ]:
        item = summary["metrics"].get(key, {})
        lines.append(
            f"| `{key}` | {item.get('median', float('nan')):.6g} | "
            f"{item.get('p95', float('nan')):.6g} | {item.get('max', float('nan')):.6g} | "
            f"{item.get('mean', float('nan')):.6g} |"
        )
    if "piper_confusion" in summary:
        lines.extend(["", "## Piper Classification", ""])
        lines.append(
            f"mismatch_fraction: `{summary['metrics'].get('piper_sign_mismatch_fraction', float('nan')):.6g}`"
        )
        lines.extend(["", "| target | pred | count |", "|---:|---:|---:|"])
        for item in summary.get("piper_confusion", []):
            lines.append(f"| {item['target']} | {item['pred']} | {item['count']} |")
    if "piper_period_error" in summary:
        period_info = summary["piper_period_error"]
        lines.extend(["", "## Piper Period Error", ""])
        lines.append(f"period: `{period_info.get('period')}`")
        lines.extend(["", "| step_mod | errors | count | error_fraction |", "|---:|---:|---:|---:|"])
        for item in period_info.get("bins", []):
            lines.append(
                f"| {item['step_mod']} | {item['errors']} | {item['count']} | {item['error_fraction']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Hints",
            "",
            "- If predicted Elite target step p95 is much larger than expert target step p95, the model is already temporally jumpy on expert states.",
            "- If Elite target error is large but Piper error is small, prioritize Elite representation/loss over Piper feed changes.",
            "- If contact-heavy samples have larger Elite error, the rollout contact issue is likely policy/model-side, not only execution limiting.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Open-loop diagnostic for a trained BC guidewire policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt, model = load_model(args.checkpoint, device)
    elite_action_representation = str(ckpt.get("elite_action_representation", "absolute"))
    dataset = DualArmGuidewireDataset(
        args.manifest,
        image_size=int(ckpt.get("image_size", args.image_size)),
        max_steps=int(ckpt.get("max_steps", args.max_steps)),
        strict_files=True,
        elite_action_representation=elite_action_representation,
        piper_head=str(ckpt.get("piper_head", "feed_regression")),
        observation_schema=str(ckpt.get("observation_schema", "full_sim_state")),
        piper_command_period_for_state=int(ckpt.get("piper_command_period_for_state", 40)),
    )
    if args.max_samples > 0:
        indices = list(range(min(args.max_samples, len(dataset))))
        eval_dataset = Subset(dataset, indices)
        samples = [dataset.samples[i] for i in indices]
    else:
        eval_dataset = dataset
        samples = dataset.samples
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred, target = predict_all(model, loader, device)

    action_mode = str(ckpt.get("action_mode", dataset.action_mode))
    piper_head = str(ckpt.get("piper_head", "feed_regression"))
    piper_step_class_values = list(ckpt.get("piper_step_class_values", [-1, 0, 1]))
    if action_mode == "joint":
        piper_dim = len(dataset.piper_joint_names)
    elif action_mode == "piper_feed_elite_joint" and piper_head == "step_classification":
        piper_dim = len(piper_step_class_values)
    else:
        piper_dim = 1
    elite_names = list(ckpt.get("elite_joint_names", dataset.elite_joint_names))
    elite_dim = 6 if elite_action_representation == "tcp_delta" else len(elite_names)
    pred_piper = pred[:, :piper_dim]
    target_piper = target[:, :piper_dim]
    pred_elite_model_space = pred[:, piper_dim : piper_dim + elite_dim]
    target_elite_model_space = target[:, piper_dim : piper_dim + elite_dim]

    if elite_action_representation == "tcp_delta":
        current_elite = np.stack([elite_tcp_pose(sample) for sample in samples], axis=0)
        target_elite_model_space = np.stack([action_elite_tcp_delta(sample) for sample in samples], axis=0)
        expert_elite_abs = current_elite + target_elite_model_space
        pred_elite_abs = current_elite + pred_elite_model_space
        target_elite_abs = expert_elite_abs
    else:
        current_elite = np.stack([elite_current(sample, elite_names) for sample in samples], axis=0)
        expert_elite_abs = np.stack([action_elite(sample, elite_names) for sample in samples], axis=0)
    if elite_action_representation == "delta":
        pred_elite_abs = current_elite + pred_elite_model_space
        target_elite_abs = current_elite + target_elite_model_space
    elif elite_action_representation == "absolute":
        pred_elite_abs = pred_elite_model_space
        target_elite_abs = target_elite_model_space

    if action_mode == "piper_feed_elite_joint" and piper_head == "step_classification":
        pred_piper_class = np.argmax(pred_piper, axis=1)
        target_piper_class = np.argmax(target_piper, axis=1)
        class_values = np.asarray(piper_step_class_values, dtype=np.float32)
        pred_piper_value = class_values[pred_piper_class]
        target_piper_value = class_values[target_piper_class]
        piper_abs_error = np.abs(pred_piper_value - target_piper_value)
        piper_sign_mismatch = pred_piper_class != target_piper_class
    else:
        pred_piper_value = pred_piper[:, 0]
        target_piper_value = target_piper[:, 0]
        piper_abs_error = np.max(np.abs(pred_piper - target_piper), axis=1)
        piper_sign_mismatch = np.sign(pred_piper[:, 0]) != np.sign(target_piper[:, 0])
    elite_linf_error = np.max(np.abs(pred_elite_abs - expert_elite_abs), axis=1)
    elite_l2_error = np.sqrt(np.sum((pred_elite_abs - expert_elite_abs) ** 2, axis=1))
    expert_delta_current = np.max(np.abs(expert_elite_abs - current_elite), axis=1)
    pred_delta_current = np.max(np.abs(pred_elite_abs - current_elite), axis=1)

    rows = []
    by_episode = {}
    for idx, sample in enumerate(samples):
        by_episode.setdefault(str(sample.get("episode", "")), []).append(idx)
        tip = np.asarray(sample.get("state", {}).get("tip_pos", [math.nan] * 3), dtype=np.float64)
        mag = np.asarray(sample.get("state", {}).get("magnetic_pose", [math.nan] * 3), dtype=np.float64)
        rows.append(
            {
                "idx": idx,
                "episode": sample.get("episode", ""),
                "task": sample.get("task", ""),
                "step": sample.get("step", ""),
                "path_progress": sample_scalar(sample, "path_progress"),
                "distance_to_target": sample_scalar(sample, "distance_to_target"),
                "distance_to_wall": sample_scalar(sample, "distance_to_wall"),
                "contact_strength": sample_scalar(sample, "contact_strength"),
                "tip_to_magnetic": l2(tip, mag),
                "piper_target": float(target_piper_value[idx]),
                "piper_pred": float(pred_piper_value[idx]),
                "piper_abs_error": float(piper_abs_error[idx]),
                "piper_sign_mismatch": bool(piper_sign_mismatch[idx]),
                "elite_linf_error": float(elite_linf_error[idx]),
                "elite_l2_error": float(elite_l2_error[idx]),
                "expert_delta_current_linf": float(expert_delta_current[idx]),
                "pred_delta_current_linf": float(pred_delta_current[idx]),
            }
        )

    expert_steps = []
    pred_steps = []
    model_space_expert_steps = []
    model_space_pred_steps = []
    for indices in by_episode.values():
        indices.sort(key=lambda i: int(samples[i].get("step", 0)))
        for a, b in zip(indices, indices[1:]):
            if samples[a].get("task") != samples[b].get("task"):
                continue
            expert_steps.append(linf(expert_elite_abs[a], expert_elite_abs[b]))
            pred_steps.append(linf(pred_elite_abs[a], pred_elite_abs[b]))
            model_space_expert_steps.append(linf(target_elite_model_space[a], target_elite_model_space[b]))
            model_space_pred_steps.append(linf(pred_elite_model_space[a], pred_elite_model_space[b]))

    contact_values = [row["contact_strength"] for row in rows]
    tip_mag_values = [row["tip_to_magnetic"] for row in rows]
    wall_values = [row["distance_to_wall"] for row in rows]
    high_contact = [idx for idx, row in enumerate(rows) if row["contact_strength"] > 0.1]
    low_contact = [idx for idx, row in enumerate(rows) if row["contact_strength"] <= 0.1]

    def subset_metric(indices, values):
        return stats([values[i] for i in indices])

    piper_confusion = None
    piper_period_error = None
    if action_mode == "piper_feed_elite_joint" and piper_head == "step_classification":
        confusion_counts = Counter((float(t), float(p)) for t, p in zip(target_piper_value, pred_piper_value))
        piper_confusion = [
            {"target": target, "pred": pred_value, "count": int(count)}
            for (target, pred_value), count in sorted(confusion_counts.items())
        ]
        period = int(ckpt.get("piper_command_period_for_state", 40))
        if period > 0:
            buckets = defaultdict(lambda: [0, 0])
            for idx, sample in enumerate(samples):
                step_mod = int(float(sample.get("step", 0))) % period
                buckets[step_mod][0] += int(bool(piper_sign_mismatch[idx]))
                buckets[step_mod][1] += 1
            piper_period_error = {
                "period": period,
                "bins": [
                    {
                        "step_mod": int(step_mod),
                        "errors": int(errors),
                        "count": int(count),
                        "error_fraction": float(errors / max(count, 1)),
                    }
                    for step_mod, (errors, count) in sorted(buckets.items())
                ],
            }

    summary = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "out": args.out,
        "device": str(device),
        "sample_count": len(samples),
        "action_mode": action_mode,
        "observation_schema": str(ckpt.get("observation_schema", "full_sim_state")),
        "piper_head": piper_head,
        "piper_command_period_for_state": int(ckpt.get("piper_command_period_for_state", 40)),
        "piper_step_class_values": piper_step_class_values,
        "elite_action_representation": elite_action_representation,
        "elite_metric_space": "tcp_pose6d_mm_rad" if elite_action_representation == "tcp_delta" else "elite_joint_radians",
        "piper_dim": piper_dim,
        "elite_joint_names": elite_names,
        "metrics": {
            "piper_abs_error": stats(piper_abs_error),
            "piper_sign_mismatch_fraction": float(np.mean(piper_sign_mismatch.astype(np.float32))),
            "elite_linf_error": stats(elite_linf_error),
            "elite_l2_error": stats(elite_l2_error),
            "expert_elite_target_step_linf": stats(expert_steps),
            "pred_elite_target_step_linf": stats(pred_steps),
            "expert_model_space_step_linf": stats(model_space_expert_steps),
            "pred_model_space_step_linf": stats(model_space_pred_steps),
            "expert_elite_delta_from_current_linf": stats(expert_delta_current),
            "pred_elite_delta_from_current_linf": stats(pred_delta_current),
            "tip_to_magnetic": stats(tip_mag_values),
            "contact_strength": stats(contact_values),
            "distance_to_wall": stats(wall_values),
            "elite_linf_error_high_contact": subset_metric(high_contact, elite_linf_error),
            "elite_linf_error_low_contact": subset_metric(low_contact, elite_linf_error),
        },
        "worst_elite_linf_error": sorted(rows, key=lambda r: r["elite_linf_error"], reverse=True)[:20],
        "worst_piper_abs_error": sorted(rows, key=lambda r: r["piper_abs_error"], reverse=True)[:20],
        "high_contact_fraction": float(len(high_contact) / max(len(rows), 1)),
    }
    if piper_confusion is not None:
        summary["piper_confusion"] = piper_confusion
    if piper_period_error is not None:
        summary["piper_period_error"] = piper_period_error

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "open_loop_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "open_loop_samples.csv", rows)
    write_report(out_dir / "open_loop_summary.md", summary)

    print(f"saved open-loop diagnostic to {out_dir}")
    print(
        "piper_abs median/p95="
        f"{summary['metrics']['piper_abs_error']['median']:.4f}/"
        f"{summary['metrics']['piper_abs_error']['p95']:.4f}; "
        "elite_linf median/p95="
        f"{summary['metrics']['elite_linf_error']['median']:.4f}/"
        f"{summary['metrics']['elite_linf_error']['p95']:.4f}; "
        "expert_step_p95="
        f"{summary['metrics']['expert_elite_target_step_linf']['p95']:.4f}; "
        "pred_step_p95="
        f"{summary['metrics']['pred_elite_target_step_linf']['p95']:.4f}"
    )


if __name__ == "__main__":
    main()
