from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pi05_e2_overlay import load_e2_overlay
from pi05_pretrained_loader import add_pretrained_arguments, initialize_pi05_policy
from pi05_state_conditioning import enable_state_conditioning
from probe_pi05_lerobot_adapter import (
    PI05ProbeDataset,
    build_config,
    collate_probe_batch,
    compute_dataset_stats,
    import_lerobot_dataset,
    import_pi05,
    set_seed,
)
from pi05_temporal_metrics import temporal_stratified_metrics
from train_pi05_lerobot_adapter import sample_stats_indices, split_by_episode, split_details
from train_pi05_mixed_head_adapter import (
    ELITE_ACTION_DIMS,
    PiperIntentHead,
    load_policy_checkpoint,
    piper_metrics,
    piper_head_logits,
)


def make_loader(dataset: Any, indices: list[int], *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        PI05ProbeDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_probe_batch,
        drop_last=False,
    )


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().flatten()
    if not flat.numel():
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def per_dim_stats(values: torch.Tensor) -> list[dict[str, float]]:
    if not values.numel():
        return []
    return [tensor_stats(values[:, dim]) for dim in range(values.shape[1])]


def load_mixed_head_checkpoint(
    path: Path,
    *,
    prefix_dim: int,
    state_dim: int,
    device: str,
) -> tuple[PiperIntentHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "piper_head_state" not in checkpoint:
        raise ValueError(f"checkpoint does not contain piper_head_state: {path}")
    if int(checkpoint.get("prefix_dim", prefix_dim)) != prefix_dim:
        raise ValueError(f"prefix dimension mismatch in {path}")
    if int(checkpoint.get("state_dim", state_dim)) != state_dim:
        raise ValueError(f"state dimension mismatch in {path}")
    saved_args = checkpoint.get("args", {})
    head_mode = str(checkpoint.get("head_mode", saved_args.get("head_mode", "legacy_concat")))
    head_feature_dim = int(
        checkpoint.get("head_feature_dim", saved_args.get("head_feature_dim", 128))
    )
    head = PiperIntentHead(
        prefix_dim=prefix_dim,
        state_dim=state_dim,
        hidden_dim=int(saved_args.get("head_hidden_dim", 256)),
        dropout=float(saved_args.get("head_dropout", 0.1)),
        mode=head_mode,
        feature_dim=head_feature_dim,
    ).to(device)
    missing, unexpected = head.load_state_dict(checkpoint["piper_head_state"], strict=False)
    if missing or unexpected:
        raise ValueError(f"Piper head state mismatch: missing={missing} unexpected={unexpected}")
    return head, checkpoint


def verify_policy_provenance(
    policy_initialization: dict[str, Any],
    mixed_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    expected = mixed_checkpoint.get("policy_initialization")
    if not isinstance(expected, dict):
        raise ValueError("mixed-head checkpoint does not contain policy_initialization provenance")
    expected_status = expected.get("status")
    actual_status = policy_initialization.get("status")
    report: dict[str, Any] = {"status": actual_status, "matched": True}
    if expected_status == "loaded" and actual_status in {
        "loaded",
        "finetuned_policy_checkpoint_loaded",
    }:
        actual_pretrained = (
            policy_initialization
            if actual_status == "loaded"
            else policy_initialization.get("base_pretrained", {})
        )
        expected_resolved = expected.get("resolved", {})
        actual_resolved = actual_pretrained.get("resolved", {})
        for key in ("source", "resolved_revision"):
            if expected_resolved.get(key) != actual_resolved.get(key):
                raise ValueError(
                    f"pretrained {key} mismatch: checkpoint={expected_resolved.get(key)!r} "
                    f"evaluation={actual_resolved.get(key)!r}"
                )
        expected_coverage = float(expected.get("loaded_parameter_fraction", 0.0))
        actual_coverage = float(actual_pretrained.get("loaded_parameter_fraction", 0.0))
        if actual_coverage < expected_coverage or actual_coverage < 0.99:
            raise ValueError(
                f"pretrained coverage mismatch: checkpoint={expected_coverage:.6f} "
                f"evaluation={actual_coverage:.6f}"
            )
        report.update(
            {
                "source": actual_resolved.get("source"),
                "resolved_revision": actual_resolved.get("resolved_revision"),
                "loaded_parameter_fraction": actual_coverage,
            }
        )
        if actual_status == "finetuned_policy_checkpoint_loaded":
            report.update(
                {
                    "path": policy_initialization.get("path"),
                    "checkpoint_selection": policy_initialization.get("checkpoint_selection"),
                    "checkpoint_step": policy_initialization.get("checkpoint_step"),
                    "active_action_dims": policy_initialization.get("active_action_dims"),
                    "loss_scope": policy_initialization.get("loss_scope"),
                }
            )
    elif actual_status == "legacy_policy_checkpoint_loaded":
        if expected_status != actual_status:
            raise ValueError(
                f"policy initialization status mismatch: checkpoint={expected_status!r} "
                f"evaluation={actual_status!r}"
            )
        if expected.get("path") != policy_initialization.get("path"):
            raise ValueError(
                f"legacy base checkpoint mismatch: checkpoint={expected.get('path')!r} "
                f"evaluation={policy_initialization.get('path')!r}"
            )
        report["path"] = policy_initialization.get("path")
    else:
        raise ValueError(f"unsupported policy initialization provenance status: {actual_status!r}")
    return report


@torch.no_grad()
def evaluate_open_loop(
    policy: torch.nn.Module,
    piper_head: PiperIntentHead,
    preprocessor: Any,
    loader: DataLoader,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    max_batches: int,
    seed: int,
    save_samples: int,
    offline_roles: list[str] | None = None,
) -> dict[str, Any]:
    policy.eval()
    piper_head.eval()
    device = next(policy.parameters()).device
    action_mean = action_mean.to(device).view(1, 1, -1)
    action_std = action_std.to(device).view(1, 1, -1)
    elite_errors: list[torch.Tensor] = []
    elite_targets: list[torch.Tensor] = []
    elite_preds: list[torch.Tensor] = []
    piper_targets: list[torch.Tensor] = []
    piper_preds: list[torch.Tensor] = []
    episode_indices: list[torch.Tensor] = []
    frame_indices: list[torch.Tensor] = []
    examples: list[dict[str, Any]] = []
    sample_rng = random.Random(seed + 7919)
    seen = 0
    generator_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    with torch.random.fork_rng(devices=generator_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            processed = preprocessor(raw_batch)
            piper_logits = piper_head_logits(policy, piper_head, processed)
            piper_target = raw_batch["piper_intent_id"].long().to(piper_logits.device)
            piper_pred = piper_logits.argmax(dim=-1)

            if hasattr(policy, "reset"):
                policy.reset()
            pred_norm = policy.predict_action_chunk(processed)
            pred_raw = pred_norm * action_std + action_mean
            target_raw = raw_batch["action"].to(pred_raw.device).float()
            elite_error = (
                pred_raw[:, 0, :ELITE_ACTION_DIMS] - target_raw[:, 0, :ELITE_ACTION_DIMS]
            ).abs()
            elite_errors.append(elite_error.detach().cpu())
            elite_targets.append(target_raw[:, 0, :3].detach().cpu())
            elite_preds.append(pred_raw[:, 0, :3].detach().cpu())
            piper_targets.append(piper_target.detach().cpu())
            piper_preds.append(piper_pred.detach().cpu())
            episode_indices.append(raw_batch["episode_index"].detach().cpu())
            frame_indices.append(raw_batch["frame_index"].detach().cpu())

            probabilities = torch.softmax(piper_logits.float(), dim=-1)
            for item in range(target_raw.shape[0]):
                example = {
                    "batch_index": int(batch_idx),
                    "episode_index": int(raw_batch["episode_index"][item].item()),
                    "frame_index": int(raw_batch["frame_index"][item].item()),
                    "target_elite_6d": target_raw[item, 0, :ELITE_ACTION_DIMS].detach().cpu().tolist(),
                    "pred_elite_6d": pred_raw[item, 0, :ELITE_ACTION_DIMS].detach().cpu().tolist(),
                    "target_piper_id": int(piper_target[item].item()),
                    "pred_piper_id": int(piper_pred[item].item()),
                    "piper_probabilities": probabilities[item].detach().cpu().tolist(),
                }
                if len(examples) < save_samples:
                    examples.append(example)
                elif save_samples > 0:
                    replacement = sample_rng.randint(0, seen)
                    if replacement < save_samples:
                        examples[replacement] = example
                seen += 1

    elite_matrix = torch.cat(elite_errors) if elite_errors else torch.empty(0, ELITE_ACTION_DIMS)
    piper_target_all = torch.cat(piper_targets) if piper_targets else torch.empty(0, dtype=torch.long)
    piper_pred_all = torch.cat(piper_preds) if piper_preds else torch.empty(0, dtype=torch.long)
    elite_target_all = torch.cat(elite_targets) if elite_targets else torch.empty(0, 3)
    elite_pred_all = torch.cat(elite_preds) if elite_preds else torch.empty(0, 3)
    episode_all = torch.cat(episode_indices) if episode_indices else torch.empty(0, dtype=torch.long)
    frame_all = torch.cat(frame_indices) if frame_indices else torch.empty(0, dtype=torch.long)
    role_metrics = None
    if offline_roles is not None:
        if len(offline_roles) < int(piper_target_all.numel()):
            raise ValueError(
                f"offline role count is shorter than predictions: {len(offline_roles)} < {piper_target_all.numel()}"
            )
        selected_roles = offline_roles[: int(piper_target_all.numel())]
        role_metrics = {}
        for role in sorted(set(selected_roles)):
            mask = torch.tensor([value == role for value in selected_roles], dtype=torch.bool)
            role_metrics[role] = {
                "records": int(mask.sum().item()),
                "elite_translation_0_3_abs_error": tensor_stats(elite_matrix[mask, :3]),
                "piper": piper_metrics(piper_target_all[mask], piper_pred_all[mask]),
            }
    return {
        "records": int(piper_target_all.numel()),
        "elite_tcp_delta_abs_error": tensor_stats(elite_matrix),
        "elite_tcp_delta_abs_error_by_dim": per_dim_stats(elite_matrix),
        "elite_translation_0_3_abs_error": tensor_stats(elite_matrix[:, :3]),
        "elite_translation_0_3_abs_error_by_dim": per_dim_stats(elite_matrix[:, :3]),
        "piper": piper_metrics(piper_target_all, piper_pred_all),
        "temporal_stratification": temporal_stratified_metrics(
            episode_indices=episode_all,
            frame_indices=frame_all,
            piper_target=piper_target_all,
            piper_pred=piper_pred_all,
            elite_translation_target=elite_target_all,
            elite_translation_pred=elite_pred_all,
        ),
        "offline_role_stratification": role_metrics,
        "samples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the project PI05 mixed action adapter on held-out episodes. Elite uses PI05 continuous "
            "sampling and Piper uses the explicit classifier; this is sim-data feasibility only."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-policy-checkpoint", type=Path)
    parser.add_argument("--mixed-head-checkpoint", type=Path, required=True)
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
    parser.add_argument("--save-samples", type=int, default=32)
    parser.add_argument("--offline-overlay-pack", type=Path)
    parser.add_argument("--offline-overlay-project-root", type=Path, default=Path("."))
    parser.add_argument("--allow-diagnostic-overlay", action="store_true")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--require-best-checkpoint", action="store_true")
    finetune_group = parser.add_mutually_exclusive_group()
    finetune_group.add_argument("--require-elite-only-finetune", action="store_true")
    finetune_group.add_argument("--require-elite-translation-finetune", action="store_true")
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
    add_pretrained_arguments(parser)
    args = parser.parse_args()
    if args.max_records is None and args.split_manifest is None:
        args.max_records = 4096

    if args.base_policy_checkpoint is not None and args.pretrained_name_or_path:
        raise ValueError("choose either --base-policy-checkpoint or --pretrained-name-or-path, not both")
    if args.allow_diagnostic_overlay and args.offline_overlay_pack is None:
        raise ValueError("--allow-diagnostic-overlay requires --offline-overlay-pack")

    set_seed(args.seed)
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    sample = dataset[0]
    state_dim = int(sample["observation.state"].numel())
    action_dim = int(sample["action"].numel())
    train_indices, val_indices, split_mode = split_by_episode(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_records=args.max_records,
        split_manifest=args.split_manifest,
    )
    stats_indices = sample_stats_indices(train_indices, max_stats_records=args.max_stats_records, seed=args.seed)
    dataset_stats = compute_dataset_stats(dataset, stats_indices)
    offline_roles = None
    overlay_evaluation = None
    if args.offline_overlay_pack is not None:
        overlay_dataset, overlay_evaluation = load_e2_overlay(
            args.offline_overlay_pack,
            project_root=args.offline_overlay_project_root,
            base_state_mean=dataset_stats["observation.state"]["mean"],
            image_size=args.image_size,
            allow_diagnostic_overlay=args.allow_diagnostic_overlay,
        )
        loader = DataLoader(
            overlay_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_probe_batch,
            drop_last=False,
        )
        offline_roles = list(overlay_dataset.sample_roles)
    else:
        loader = make_loader(dataset, val_indices, batch_size=args.batch_size, num_workers=args.num_workers)

    pi05 = import_pi05()
    config = build_config(args, pi05, state_dim, action_dim)
    if args.num_inference_steps is not None:
        config.num_inference_steps = int(args.num_inference_steps)
    preprocessor, _postprocessor = pi05["make_pi05_pre_post_processors"](config, dataset_stats)
    if args.base_policy_checkpoint is not None:
        policy = pi05["PI05Policy"](config).to(args.device)
        policy_metadata = load_policy_checkpoint(args.base_policy_checkpoint, policy)
        pretrained_origin = policy_metadata.get("pretrained")
        if isinstance(pretrained_origin, dict) and pretrained_origin.get("status") == "loaded":
            policy_initialization = {
                "status": "finetuned_policy_checkpoint_loaded",
                "path": str(args.base_policy_checkpoint),
                "base_pretrained": pretrained_origin,
                "state_conditioning": policy_metadata.get("state_conditioning"),
                "checkpoint_selection": policy_metadata.get("checkpoint_selection"),
                "checkpoint_step": policy_metadata.get("checkpoint_step"),
                "active_action_dims": policy_metadata.get("active_action_dims"),
                "loss_scope": policy_metadata.get("loss_scope"),
                "training_sampler": policy_metadata.get("training_sampler"),
                "training_overlay": policy_metadata.get("training_overlay"),
            }
        else:
            policy_initialization = {
                "status": "legacy_policy_checkpoint_loaded",
                "path": str(args.base_policy_checkpoint),
            }
    else:
        policy, policy_initialization = initialize_pi05_policy(
            pi05["PI05Policy"],
            config,
            args,
        )
    prefix_dim = int(policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    piper_head, mixed_checkpoint = load_mixed_head_checkpoint(
        args.mixed_head_checkpoint,
        prefix_dim=prefix_dim,
        state_dim=state_dim,
        device=args.device,
    )
    checkpoint_selection = mixed_checkpoint.get("checkpoint_selection")
    checkpoint_step = mixed_checkpoint.get("checkpoint_step")
    if args.require_best_checkpoint and checkpoint_selection != "best_validation_balanced_accuracy":
        raise ValueError(
            "--require-best-checkpoint rejected mixed head with "
            f"checkpoint_selection={checkpoint_selection!r}"
        )
    if args.require_elite_only_finetune:
        if policy_initialization.get("status") != "finetuned_policy_checkpoint_loaded":
            raise ValueError("--require-elite-only-finetune requires a derived pretrained policy checkpoint")
        if policy_initialization.get("active_action_dims") != 6:
            raise ValueError(
                "--require-elite-only-finetune requires active_action_dims=6, got "
                f"{policy_initialization.get('active_action_dims')!r}"
            )
        if not str(policy_initialization.get("loss_scope", "")).startswith("Elite TCP delta dims 0:6"):
            raise ValueError(
                "--require-elite-only-finetune rejected loss_scope="
                f"{policy_initialization.get('loss_scope')!r}"
            )
    if args.require_elite_translation_finetune:
        if policy_initialization.get("status") != "finetuned_policy_checkpoint_loaded":
            raise ValueError("--require-elite-translation-finetune requires a derived pretrained policy checkpoint")
        if policy_initialization.get("active_action_dims") != 3:
            raise ValueError(
                "--require-elite-translation-finetune requires active_action_dims=3, got "
                f"{policy_initialization.get('active_action_dims')!r}"
            )
        if not str(policy_initialization.get("loss_scope", "")).startswith(
            "Elite TCP translation delta dims 0:3"
        ):
            raise ValueError(
                "--require-elite-translation-finetune rejected loss_scope="
                f"{policy_initialization.get('loss_scope')!r}"
            )
    provenance_verification = verify_policy_provenance(
        policy_initialization,
        mixed_checkpoint,
    )

    result = evaluate_open_loop(
        policy,
        piper_head,
        preprocessor,
        loader,
        action_mean=dataset_stats["action"]["mean"],
        action_std=dataset_stats["action"]["std"],
        max_batches=args.max_val_batches,
        seed=args.eval_seed,
        save_samples=args.save_samples,
        offline_roles=offline_roles,
    )
    report = {
        "scope": (
            "PI05 mixed-action E2 training-overlay descriptive diagnostic; not validation or model selection"
            if overlay_evaluation is not None
            else "PI05 mixed-action held-out sim-data feasibility; not real-system validation"
        ),
        "root": str(args.root),
        "repo_id": args.repo_id,
        "base_policy_checkpoint": str(args.base_policy_checkpoint) if args.base_policy_checkpoint else None,
        "policy_initialization": policy_initialization,
        "mixed_head_checkpoint": str(args.mixed_head_checkpoint),
        "mixed_head_checkpoint_selection": checkpoint_selection,
        "mixed_head_checkpoint_step": checkpoint_step,
        "mixed_head_training_sampler": mixed_checkpoint.get("training_sampler"),
        "mixed_head_training_overlay": mixed_checkpoint.get("training_overlay"),
        "mixed_head_train_piper_class_counts": mixed_checkpoint.get("train_piper_class_counts"),
        "mixed_head_class_weights": mixed_checkpoint.get("class_weights"),
        "mixed_head_class_weight_source": mixed_checkpoint.get("class_weight_source"),
        "policy_provenance_verification": provenance_verification,
        "split": split_mode,
        "split_details": split_details(
            split_mode=split_mode,
            split_manifest=args.split_manifest,
            train_samples=len(train_indices),
            val_samples=len(val_indices),
        ),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "num_inference_steps": int(config.num_inference_steps),
        "evaluation": {
            "seed": int(args.seed),
            "eval_seed": int(args.eval_seed),
            "max_val_batches": int(args.max_val_batches),
            "batch_size": int(args.batch_size),
            "set": "e2_training_overlay" if overlay_evaluation is not None else "fixed_base_validation",
        },
        "offline_overlay_evaluation": overlay_evaluation,
        "semantics": {
            **(mixed_checkpoint.get("semantics") or {}),
            "pi05_state_conditioning": policy_initialization.get("state_conditioning"),
            "pi05_state_limit": (
                "state prefix adapter consumes observation.state; exact contact/wall/route truth remains excluded"
                if isinstance(policy_initialization.get("state_conditioning"), dict)
                and policy_initialization["state_conditioning"].get("enabled")
                else (mixed_checkpoint.get("semantics") or {}).get("pi05_state_limit")
            ),
            "offline_role_limit": (
                "sample_role is joined after prediction for reporting only and is absent from model batches"
                if overlay_evaluation is not None
                else None
            ),
        },
        "piper_head_feature_contract": piper_head.feature_contract(),
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"records={result['records']} elite_translation_mae={result['elite_translation_0_3_abs_error']['mean']:.6f} "
        f"piper_acc={result['piper']['accuracy']:.4f} "
        f"piper_balanced_acc={result['piper']['balanced_accuracy']:.4f} "
        f"majority={result['piper']['majority_baseline']:.4f} wrote={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
