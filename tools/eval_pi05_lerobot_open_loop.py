from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pi05_state_conditioning import enable_state_conditioning

from probe_pi05_lerobot_adapter import (
    ACTIVE_ACTION_DIMS,
    PI05ProbeDataset,
    build_config,
    collate_probe_batch,
    compute_dataset_stats,
    import_lerobot_dataset,
    import_pi05,
    set_seed,
    stats_to_json,
)
from pi05_temporal_metrics import temporal_stratified_metrics
from train_pi05_lerobot_adapter import sample_stats_indices, split_by_episode, split_details


def to_float_list(values: torch.Tensor, *, max_len: int | None = None) -> list[float]:
    array = values.detach().float().cpu().flatten()
    if max_len is not None:
        array = array[:max_len]
    return [float(x) for x in array.tolist()]


def tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().float().cpu().flatten()
    if flat.numel() == 0:
        return {"count": 0}
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def per_dim_tensor_stats(values: torch.Tensor) -> list[dict[str, Any]]:
    tensor = values.detach().float().cpu()
    if tensor.numel() == 0:
        return []
    tensor = tensor.reshape(-1, tensor.shape[-1])
    return [tensor_stats(tensor[:, dim]) for dim in range(tensor.shape[1])]


def confusion_matrix(target: torch.Tensor, pred: torch.Tensor, *, classes: int = 3) -> list[list[int]]:
    matrix = torch.zeros((classes, classes), dtype=torch.int64)
    for t, p in zip(target.detach().cpu().flatten(), pred.detach().cpu().flatten(), strict=False):
        ti = int(t.item())
        pi = int(p.item())
        if 0 <= ti < classes and 0 <= pi < classes:
            matrix[ti, pi] += 1
    return matrix.tolist()


def confusion_counts(matrix: list[list[int]]) -> dict[str, Any]:
    if not matrix:
        return {"target_counts": [], "pred_counts": [], "majority_baseline": 0.0, "per_class_recall": []}
    target_counts = [int(sum(row)) for row in matrix]
    pred_counts = [int(sum(matrix[row][col] for row in range(len(matrix)))) for col in range(len(matrix[0]))]
    total = sum(target_counts)
    return {
        "target_counts": target_counts,
        "pred_counts": pred_counts,
        "majority_baseline": float(max(target_counts) / total) if total else 0.0,
        "per_class_recall": [
            (float(matrix[idx][idx] / target_counts[idx]) if target_counts[idx] else None)
            for idx in range(len(matrix))
        ],
    }


def load_checkpoint(path: Path, policy: torch.nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if state is None:
        raise ValueError(f"checkpoint does not contain model_state: {path}")
    state_conditioning = checkpoint.get("state_conditioning") if isinstance(checkpoint, dict) else None
    if isinstance(state_conditioning, dict) and state_conditioning.get("enabled"):
        enable_state_conditioning(policy, int(state_conditioning["state_dim"]))
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return checkpoint


@torch.no_grad()
def evaluate_open_loop(
    policy: torch.nn.Module,
    preprocessor: Any,
    loader: DataLoader,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    max_batches: int,
    seed: int,
    save_samples: int,
    sample_mode: str,
) -> dict[str, Any]:
    policy.eval()
    action_mean = action_mean.to(next(policy.parameters()).device).view(1, 1, -1)
    action_std = action_std.to(next(policy.parameters()).device).view(1, 1, -1)

    rows: list[dict[str, Any]] = []
    norm_abs_errors: list[torch.Tensor] = []
    raw_abs_errors: list[torch.Tensor] = []
    elite_abs_errors: list[torch.Tensor] = []
    elite_targets: list[torch.Tensor] = []
    elite_preds: list[torch.Tensor] = []
    piper_targets: list[torch.Tensor] = []
    piper_preds: list[torch.Tensor] = []
    episode_indices: list[torch.Tensor] = []
    frame_indices: list[torch.Tensor] = []
    episode_stats: dict[int, dict[str, float]] = {}
    active_mse_sum = 0.0
    active_count = 0
    seen_samples = 0
    sample_rng = random.Random(seed + 7919)

    generator_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=generator_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            processed = preprocessor(raw_batch)
            if hasattr(policy, "reset"):
                policy.reset()
            pred_norm = policy.predict_action_chunk(processed)
            target_raw = raw_batch["action"].to(pred_norm.device).float()
            target_norm = processed["action"].to(pred_norm.device).float()
            pred_raw = pred_norm * action_std + action_mean

            norm_err = (pred_norm[:, :, :ACTIVE_ACTION_DIMS] - target_norm[:, :, :ACTIVE_ACTION_DIMS]).abs()
            raw_err = (pred_raw[:, :, :ACTIVE_ACTION_DIMS] - target_raw[:, :, :ACTIVE_ACTION_DIMS]).abs()
            elite_err = raw_err[:, :, :6]
            active_mse_sum += float(
                torch.square(pred_norm[:, :, :ACTIVE_ACTION_DIMS] - target_norm[:, :, :ACTIVE_ACTION_DIMS]).sum().item()
            )
            active_count += int(norm_err.numel())
            norm_abs_errors.append(norm_err.detach().cpu())
            raw_abs_errors.append(raw_err.detach().cpu())
            elite_abs_errors.append(elite_err.detach().cpu())
            elite_targets.append(target_raw[:, 0, :3].detach().cpu())
            elite_preds.append(pred_raw[:, 0, :3].detach().cpu())

            target_piper = target_raw[:, 0, 6:9].argmax(dim=1).detach().cpu()
            pred_piper = pred_raw[:, 0, 6:9].argmax(dim=1).detach().cpu()
            piper_targets.append(target_piper)
            piper_preds.append(pred_piper)
            episode_indices.append(raw_batch["episode_index"].detach().cpu())
            frame_indices.append(raw_batch["frame_index"].detach().cpu())

            for item in range(target_raw.shape[0]):
                episode = int(raw_batch["episode_index"][item].item())
                episode_row = episode_stats.setdefault(
                    episode,
                    {"records": 0.0, "elite_abs_sum": 0.0, "piper_correct": 0.0},
                )
                episode_row["records"] += 1.0
                episode_row["elite_abs_sum"] += float(elite_err[item].mean().detach().cpu().item())
                episode_row["piper_correct"] += float((target_piper[item] == pred_piper[item]).item())

                sample = {
                    "batch_index": int(batch_idx),
                    "episode_index": episode,
                    "frame_index": int(raw_batch["frame_index"][item].item()),
                    "target_action_0_9": to_float_list(target_raw[item, 0, :ACTIVE_ACTION_DIMS]),
                    "pred_action_0_9": to_float_list(pred_raw[item, 0, :ACTIVE_ACTION_DIMS]),
                    "abs_error_0_9": to_float_list(raw_err[item, 0, :ACTIVE_ACTION_DIMS]),
                    "target_piper_id": int(target_piper[item].item()),
                    "pred_piper_id": int(pred_piper[item].item()),
                }
                if save_samples > 0 and sample_mode == "first" and len(rows) < save_samples:
                    rows.append(sample)
                elif save_samples > 0 and sample_mode == "random":
                    if len(rows) < save_samples:
                        rows.append(sample)
                    else:
                        replace_idx = sample_rng.randint(0, seen_samples)
                        if replace_idx < save_samples:
                            rows[replace_idx] = sample
                seen_samples += 1

    norm_abs_matrix = (
        torch.cat([x.reshape(-1, ACTIVE_ACTION_DIMS) for x in norm_abs_errors])
        if norm_abs_errors
        else torch.empty(0, ACTIVE_ACTION_DIMS)
    )
    raw_abs_matrix = (
        torch.cat([x.reshape(-1, ACTIVE_ACTION_DIMS) for x in raw_abs_errors])
        if raw_abs_errors
        else torch.empty(0, ACTIVE_ACTION_DIMS)
    )
    elite_abs_matrix = (
        torch.cat([x.reshape(-1, 6) for x in elite_abs_errors])
        if elite_abs_errors
        else torch.empty(0, 6)
    )
    norm_abs = norm_abs_matrix.flatten()
    raw_abs = raw_abs_matrix.flatten()
    elite_abs = elite_abs_matrix.flatten()
    piper_target = torch.cat(piper_targets) if piper_targets else torch.empty(0, dtype=torch.long)
    piper_pred = torch.cat(piper_preds) if piper_preds else torch.empty(0, dtype=torch.long)
    elite_target = torch.cat(elite_targets) if elite_targets else torch.empty(0, 3)
    elite_pred = torch.cat(elite_preds) if elite_preds else torch.empty(0, 3)
    elite_translation_abs = (elite_pred - elite_target).abs()
    episode_all = torch.cat(episode_indices) if episode_indices else torch.empty(0, dtype=torch.long)
    frame_all = torch.cat(frame_indices) if frame_indices else torch.empty(0, dtype=torch.long)
    piper_acc = float((piper_target == piper_pred).float().mean().item()) if piper_target.numel() else 0.0
    piper_confusion = confusion_matrix(piper_target, piper_pred, classes=3)
    return {
        "records": int(piper_target.numel()),
        "active_action_dims": ACTIVE_ACTION_DIMS,
        "normalized_active_mse": active_mse_sum / max(active_count, 1),
        "normalized_active_mae": tensor_stats(norm_abs),
        "normalized_active_mae_by_dim": per_dim_tensor_stats(norm_abs_matrix),
        "raw_action_0_9_mae": tensor_stats(raw_abs),
        "raw_action_0_9_mae_by_dim": per_dim_tensor_stats(raw_abs_matrix),
        "elite_tcp_delta_0_6_mae": tensor_stats(elite_abs),
        "elite_tcp_delta_0_6_mae_by_dim": per_dim_tensor_stats(elite_abs_matrix),
        "elite_translation_0_3_mae": tensor_stats(elite_translation_abs),
        "elite_translation_0_3_mae_by_dim": per_dim_tensor_stats(elite_translation_abs),
        "piper_intent_argmax_acc": piper_acc,
        "piper_confusion_target_rows_pred_cols": piper_confusion,
        "piper_confusion_counts": confusion_counts(piper_confusion),
        "temporal_stratification": temporal_stratified_metrics(
            episode_indices=episode_all,
            frame_indices=frame_all,
            piper_target=piper_target,
            piper_pred=piper_pred,
            elite_translation_target=elite_target,
            elite_translation_pred=elite_pred,
        ),
        "episode_metrics": {
            str(episode): {
                "records": int(stats["records"]),
                "elite_mae_mean": float(stats["elite_abs_sum"] / max(stats["records"], 1.0)),
                "piper_acc": float(stats["piper_correct"] / max(stats["records"], 1.0)),
            }
            for episode, stats in sorted(episode_stats.items())
        },
        "sample_mode": sample_mode,
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpointed LeRobot PI05 adapter by open-loop action prediction on held-out records. "
            "This is a synthetic-data feasibility diagnostic; action_32 is still a compatibility tensor."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--split-manifest", type=Path, help="Use the authoritative LeRobot export episode split.")
    parser.add_argument("--max-stats-records", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-seed", type=int, default=20260717)
    parser.add_argument("--save-samples", type=int, default=16)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--num-inference-steps", type=int, help="Override PI05 diffusion inference steps for action sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--tokenizer-max-length", type=int, default=200)
    parser.add_argument("--paligemma-variant", default="gemma_2b", choices=["gemma_300m", "gemma_2b"])
    parser.add_argument("--action-expert-variant", default="gemma_300m", choices=["gemma_300m", "gemma_2b"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--freeze-vision-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision-encoder", dest="freeze_vision_encoder", action="store_false")
    parser.add_argument("--train-expert-only", action="store_true", default=True)
    parser.add_argument("--no-train-expert-only", dest="train_expert_only", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()
    if args.max_records is None and args.split_manifest is None:
        args.max_records = 4096

    set_seed(args.seed)
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    if len(dataset) == 0:
        raise ValueError(f"empty LeRobotDataset: root={args.root} repo_id={args.repo_id}")
    sample = dataset[0]
    state_dim = int(sample["observation.state"].numel())
    action_dim = int(sample["action"].numel())
    if state_dim != 32 or action_dim != 32:
        raise ValueError(f"expected 32D state/action compatibility tensors, got state={state_dim} action={action_dim}")

    train_indices, val_indices, split_mode = split_by_episode(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_records=args.max_records,
        split_manifest=args.split_manifest,
    )
    stats_indices = sample_stats_indices(train_indices, max_stats_records=args.max_stats_records, seed=args.seed)
    dataset_stats = compute_dataset_stats(dataset, stats_indices)

    val_loader = DataLoader(
        PI05ProbeDataset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_probe_batch,
        drop_last=False,
    )

    pi05 = import_pi05()
    config = build_config(args, pi05, state_dim, action_dim)
    if args.num_inference_steps is not None:
        if not hasattr(config, "num_inference_steps"):
            raise ValueError("PI05 config does not expose num_inference_steps")
        config.num_inference_steps = int(args.num_inference_steps)
    preprocessor, _postprocessor = pi05["make_pi05_pre_post_processors"](config, dataset_stats)
    policy = pi05["PI05Policy"](config).to(args.device)
    checkpoint = load_checkpoint(args.checkpoint, policy)
    result = evaluate_open_loop(
        policy,
        preprocessor,
        val_loader,
        action_mean=dataset_stats["action"]["mean"],
        action_std=dataset_stats["action"]["std"],
        max_batches=args.max_val_batches,
        seed=args.eval_seed,
        save_samples=args.save_samples,
        sample_mode=args.sample_mode,
    )

    summary = {
        "root": str(args.root),
        "repo_id": args.repo_id,
        "checkpoint": str(args.checkpoint),
        "dataset_len": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "stats_samples": len(stats_indices),
        "split": split_mode,
        "split_details": split_details(
            split_mode=split_mode,
            split_manifest=args.split_manifest,
            train_samples=len(train_indices),
            val_samples=len(val_indices),
        ),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pi05_config": {
            "paligemma_variant": args.paligemma_variant,
            "action_expert_variant": args.action_expert_variant,
            "dtype": args.dtype,
            "device": args.device,
            "freeze_vision_encoder": bool(args.freeze_vision_encoder),
            "train_expert_only": bool(args.train_expert_only),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "num_inference_steps": int(getattr(config, "num_inference_steps", -1)),
        },
        "checkpoint_final_val": checkpoint.get("final_val") if isinstance(checkpoint, dict) else None,
        "normalization_stats": stats_to_json(dataset_stats),
        "open_loop": result,
        "semantics": {
            "scope": "held-out open-loop prediction diagnostic only",
            "action": "action_32 compatibility tensor; dims 0:6 are Elite TCP delta, dims 6:9 are Piper intent one-hot compatibility, dims 9:32 are padding",
            "preferred_project_target": "Elite TCP delta regression plus Piper discrete intent classification",
            "not_validation": "do not interpret this sim-data diagnostic as real-system validation",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "records={records} norm_mse={mse:.6f} elite_translation_mae={elite:.6f} piper_acc={piper:.3f}".format(
            records=result["records"],
            mse=result["normalized_active_mse"],
            elite=result["elite_translation_0_3_mae"].get("mean", 0.0),
            piper=result["piper_intent_argmax_acc"],
        ),
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
