from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from pi05_action_effect_dataset import (
    ActionEffectFeaturePack,
    ActionEffectWindowDataset,
    build_temporal_windows,
    compute_train_normalization,
    load_episode_split,
    source_episode_balanced_weights,
)
from pi05_action_effect_world_model import (
    ActionEffectLossConfig,
    ActionEffectWorldModel,
    ActionEffectWorldModelConfig,
    action_effect_world_model_loss,
    paired_degradation_consistency_loss,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _evaluate(
    model: ActionEffectWorldModel,
    loader: DataLoader,
    device: torch.device,
    loss_config: ActionEffectLossConfig,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                history_visual_latent=batch["history_visual_latent"],
                history_state=batch["history_state"],
                history_visual_valid=batch["history_visual_valid"],
                history_state_valid=batch["history_state_valid"],
                task_id=batch["task_id"],
                candidate_actions=batch["candidate_actions"],
            )
            loss, components = _loss_with_optional_degradation(
                model, outputs, batch, loss_config
            )
            values = {"total": loss, **components}
            batch_size = int(batch["task_id"].shape[0])
            count += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    if count == 0:
        raise ValueError("validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def _loss_with_optional_degradation(
    model: ActionEffectWorldModel,
    clean_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_config: ActionEffectLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, components = action_effect_world_model_loss(clean_outputs, batch, loss_config)
    pair_mask = batch["degradation_pair_valid"].bool()
    if pair_mask.any():
        degraded_outputs = model(
            history_visual_latent=batch["history_degraded_visual_latent"],
            history_state=batch["history_state"],
            history_visual_valid=batch["history_visual_valid"],
            history_state_valid=batch["history_state_valid"],
            task_id=batch["task_id"],
            candidate_actions=batch["candidate_actions"],
        )
        consistency = paired_degradation_consistency_loss(
            clean_outputs, degraded_outputs, pair_mask
        )
    else:
        consistency = clean_outputs["pred_next_visual_latent"].sum() * 0.0
    components = {**components, "degradation_consistency": consistency}
    total = total + loss_config.degradation_consistency_weight * consistency
    return total, components


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the short-horizon PI05 action-effect world model from a validated feature pack."
    )
    parser.add_argument("--feature-pack", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--real-draw-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("max-steps and batch-size must be positive")
    _set_seed(args.seed)
    pack = ActionEffectFeaturePack(args.feature_pack)
    split = load_episode_split(args.split_manifest)
    train_episodes = {int(value) for value in split["train_episode_indices"]}
    validation_episodes = {int(value) for value in split["validation_episode_indices"]}
    pack.validate_episode_split(split)

    window_args = {
        "episode_index": pack.arrays["episode_index"],
        "frame_index": pack.arrays["frame_index"],
        "task_id": pack.arrays["task_id"],
        "domain_id": pack.arrays["domain_id"],
        "context_len": args.context_len,
        "horizon": args.horizon,
    }
    train_windows = build_temporal_windows(
        **window_args, allowed_episodes=train_episodes
    )
    validation_windows = build_temporal_windows(
        **window_args, allowed_episodes=validation_episodes
    )
    if not train_windows or not validation_windows:
        raise ValueError("train and validation must each contain at least one temporal window")
    normalization = compute_train_normalization(pack, train_episodes)
    train_dataset = ActionEffectWindowDataset(pack, train_windows, normalization)
    validation_dataset = ActionEffectWindowDataset(pack, validation_windows, normalization)
    sampler_weights = source_episode_balanced_weights(
        train_windows,
        real_draw_fraction=args.real_draw_fraction,
        episode_source=pack.episode_source_map(),
    )
    draws = int(args.max_steps) * int(args.batch_size)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sampler_weights,
        num_samples=draws,
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    config = ActionEffectWorldModelConfig(
        visual_dim=pack.visual_dim,
        hidden_dim=args.hidden_dim,
        context_len=args.context_len,
        horizon=args.horizon,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = ActionEffectWorldModel(config).to(device)
    loss_config = ActionEffectLossConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    first_batch = next(iter(train_loader))
    first_batch = {key: value.to(device) for key, value in first_batch.items()}
    first_outputs = model(
        history_visual_latent=first_batch["history_visual_latent"],
        history_state=first_batch["history_state"],
        history_visual_valid=first_batch["history_visual_valid"],
        history_state_valid=first_batch["history_state_valid"],
        task_id=first_batch["task_id"],
        candidate_actions=first_batch["candidate_actions"],
    )
    initial_loss, initial_components = _loss_with_optional_degradation(
        model, first_outputs, first_batch, loss_config
    )

    summary: dict[str, Any] = {
        "schema": "project2026_action_effect_world_model_train_summary_v1",
        "mode": "dry_run" if args.dry_run else "train",
        "feature_pack": str(args.feature_pack),
        "split_manifest": str(args.split_manifest),
        "model": model.metadata(),
        "feature_pack_fingerprints": {
            "manifest_sha256": pack.manifest_sha256,
            "arrays_sha256": pack.arrays_sha256,
        },
        "normalization": {
            "scope": "training_episodes_only",
            "record_count": int(normalization["record_count"][0]),
            "state_mean": normalization["state_mean"].tolist(),
            "state_std": normalization["state_std"].tolist(),
            "state_count": normalization["state_count"].tolist(),
            "elite_mean": normalization["elite_mean"].tolist(),
            "elite_std": normalization["elite_std"].tolist(),
            "missing_state_imputation_after_normalization": 0.0,
        },
        "loss_config": asdict(loss_config),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "sampler": {
            "mode": "source_and_episode_balanced_weighted_replacement",
            "real_draw_fraction": args.real_draw_fraction,
            "optimizer_steps": args.max_steps,
            "draws": draws,
            "weights_sha256": hashlib.sha256(sampler_weights.numpy().tobytes()).hexdigest(),
        },
        "initial_batch_loss": float(initial_loss.item()),
        "initial_batch_components": {
            key: float(value.item()) for key, value in initial_components.items()
        },
        "training_started": not args.dry_run,
        "shared_output_interface": "elite_tcp_delta_6d + piper_intent_id",
        "senior_historical_real_data_used": False,
    }

    if not args.dry_run:
        model.train()
        last_loss = None
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                history_visual_latent=batch["history_visual_latent"],
                history_state=batch["history_state"],
                history_visual_valid=batch["history_visual_valid"],
                history_state_valid=batch["history_state_valid"],
                task_id=batch["task_id"],
                candidate_actions=batch["candidate_actions"],
            )
            loss, _ = _loss_with_optional_degradation(
                model, outputs, batch, loss_config
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = float(loss.item())
            if step >= args.max_steps:
                break
        summary["final_train_loss"] = last_loss
        summary["validation"] = _evaluate(model, validation_loader, device, loss_config)
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_metadata": model.metadata(),
                "normalization": summary["normalization"],
                "summary": summary,
            },
            args.out / "action_effect_world_model.pt",
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"mode={summary['mode']} train_windows={len(train_windows)} "
        f"validation_windows={len(validation_windows)} initial_loss={initial_loss.item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
