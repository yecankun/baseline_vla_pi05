from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from itertools import cycle, repeat
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from pi05_e2_overlay import load_e2_overlay, make_weighted_training_loader
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
    stats_to_json,
)
from pi05_temporal_metrics import temporal_stratified_metrics
from train_pi05_lerobot_adapter import (
    REAL10_TRAIN_SCOPE,
    repeat_loader,
    sample_stats_indices,
    split_by_episode,
    split_details,
    trainable_parameter_summary,
    validate_train_only_options,
)


ELITE_ACTION_DIMS = 6
PIPER_CLASSES = 3
PIPER_HEAD_MODES = ("state_only", "prefix_state", "legacy_concat")


class PiperIntentHead(nn.Module):
    def __init__(
        self,
        prefix_dim: int,
        state_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        mode: str = "legacy_concat",
        feature_dim: int = 128,
    ):
        super().__init__()
        if mode not in PIPER_HEAD_MODES:
            raise ValueError(f"unknown Piper head mode {mode!r}; expected one of {PIPER_HEAD_MODES}")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.mode = mode
        self.prefix_dim = int(prefix_dim)
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        if mode == "legacy_concat":
            self.prefix_norm = nn.LayerNorm(prefix_dim)
            self.state_norm = nn.LayerNorm(state_dim)
            self.prefix_encoder = None
            self.state_encoder = None
            classifier_input_dim = prefix_dim + state_dim
        else:
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, feature_dim),
                nn.GELU(),
            )
            if mode == "prefix_state":
                self.prefix_encoder = nn.Sequential(
                    nn.LayerNorm(prefix_dim),
                    nn.Linear(prefix_dim, feature_dim),
                    nn.GELU(),
                )
                classifier_input_dim = 2 * feature_dim
            else:
                self.prefix_encoder = None
                classifier_input_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, PIPER_CLASSES),
        )

    @property
    def requires_prefix(self) -> bool:
        return self.mode != "state_only"

    def feature_contract(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "prefix_dim": self.prefix_dim,
            "state_dim": self.state_dim,
            "branch_feature_dim": self.feature_dim if self.mode != "legacy_concat" else None,
            "classifier_input_dim": int(self.classifier[0].in_features),
            "dimension_balanced": self.mode == "prefix_state",
        }

    @staticmethod
    def _pool_prefix(
        prefix_embs: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        if prefix_embs is None or prefix_pad_masks is None:
            raise ValueError("prefix embeddings and masks are required for this Piper head mode")
        weights = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_embs.dtype)
        return (prefix_embs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        prefix_embs: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "legacy_concat":
            pooled = self._pool_prefix(prefix_embs, prefix_pad_masks)
            features = torch.cat(
                [self.prefix_norm(pooled.float()), self.state_norm(state.float())],
                dim=-1,
            )
        elif self.mode == "state_only":
            state_features = self.state_encoder(state.float())
            features = state_features
        else:
            state_features = self.state_encoder(state.float())
            pooled = self._pool_prefix(prefix_embs, prefix_pad_masks)
            prefix_features = self.prefix_encoder(pooled.float())
            features = torch.cat([prefix_features, state_features], dim=-1)
        return self.classifier(features)


def rng_context(seed: int | None):
    if seed is None:
        return nullcontext()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devices)


def set_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        PI05ProbeDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_probe_batch,
        drop_last=False,
        generator=generator,
    )


def load_policy_checkpoint(path: Path, policy: nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if state is None and isinstance(checkpoint, dict):
        state = checkpoint.get("policy_state")
    if state is None:
        raise ValueError(f"checkpoint does not contain model_state or policy_state: {path}")
    state_conditioning = checkpoint.get("state_conditioning") if isinstance(checkpoint, dict) else None
    if isinstance(state_conditioning, dict) and state_conditioning.get("enabled"):
        enable_state_conditioning(policy, int(state_conditioning["state_dim"]))
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state", "policy_state"}
    }
    del state
    del checkpoint
    return metadata


def class_counts(dataset: Any, indices: list[int]) -> list[int]:
    counts = [0] * PIPER_CLASSES
    for idx in indices:
        label = int(dataset[idx]["piper_intent_id"])
        if label < 0 or label >= PIPER_CLASSES:
            raise ValueError(f"invalid Piper intent id {label} at dataset index {idx}")
        counts[label] += 1
    return counts


def balanced_class_weights(counts: list[int], device: str) -> torch.Tensor:
    present = sum(count > 0 for count in counts)
    total = sum(counts)
    if present == 0:
        raise ValueError("no Piper labels in training split")
    values = [total / (present * count) if count > 0 else 0.0 for count in counts]
    return torch.tensor(values, dtype=torch.float32, device=device)


def piper_metrics(target: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    matrix = [[0 for _ in range(PIPER_CLASSES)] for _ in range(PIPER_CLASSES)]
    for true_id, pred_id in zip(target.tolist(), pred.tolist(), strict=True):
        matrix[int(true_id)][int(pred_id)] += 1
    supports = [sum(row) for row in matrix]
    recalls = [matrix[idx][idx] / supports[idx] for idx in range(PIPER_CLASSES) if supports[idx] > 0]
    majority = max(supports) / max(sum(supports), 1)
    return {
        "accuracy": float((target == pred).float().mean().item()) if target.numel() else 0.0,
        "balanced_accuracy": float(sum(recalls) / max(len(recalls), 1)),
        "majority_baseline": float(majority),
        "support": supports,
        "confusion_target_rows_pred_cols": matrix,
    }


def validation_score(evaluation: dict[str, Any], step: int) -> tuple[float, float, float, int]:
    metrics = evaluation["piper"]
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -float(evaluation["piper_ce"]),
        -int(step),
    )


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def prepare_prefix(policy: nn.Module, processed: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(processed)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad_masks, _prefix_att_masks = policy.model.embed_prefix(
        images,
        img_masks,
        tokens,
        masks,
    )
    return prefix_embs, prefix_pad_masks


def piper_head_logits(
    policy: nn.Module,
    piper_head: PiperIntentHead,
    processed: dict[str, Any],
) -> torch.Tensor:
    if piper_head.requires_prefix:
        prefix_embs, prefix_pad_masks = prepare_prefix(policy, processed)
    else:
        prefix_embs, prefix_pad_masks = None, None
    return piper_head(prefix_embs, prefix_pad_masks, processed["observation.state"])


def elite_diffusion_loss(policy: nn.Module, processed: dict[str, Any]) -> torch.Tensor:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(processed)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(processed)
    losses = policy.model.forward(images, img_masks, tokens, masks, actions)
    return losses[:, :, :ELITE_ACTION_DIMS].mean()


@torch.no_grad()
def evaluate(
    policy: nn.Module,
    piper_head: PiperIntentHead,
    preprocessor: Any,
    loader: DataLoader,
    *,
    max_batches: int,
    eval_seed: int,
    report_elite_loss: bool,
    class_weight: torch.Tensor | None,
) -> dict[str, Any]:
    policy_was_training = policy.training
    head_was_training = piper_head.training
    policy.eval()
    piper_head.eval()
    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    episode_indices: list[torch.Tensor] = []
    frame_indices: list[torch.Tensor] = []
    total_ce = 0.0
    total_elite = 0.0
    count = 0
    with rng_context(eval_seed):
        set_torch_seed(eval_seed)
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            processed = preprocessor(raw_batch)
            logits = piper_head_logits(policy, piper_head, processed)
            target = raw_batch["piper_intent_id"].to(logits.device)
            total_ce += float(F.cross_entropy(logits, target, weight=class_weight).item())
            if report_elite_loss:
                total_elite += float(elite_diffusion_loss(policy, processed).item())
            targets.append(target.detach().cpu())
            preds.append(logits.argmax(dim=-1).detach().cpu())
            episode_indices.append(raw_batch["episode_index"].detach().cpu())
            frame_indices.append(raw_batch["frame_index"].detach().cpu())
            count += 1
    policy.train(policy_was_training)
    piper_head.train(head_was_training)
    target_all = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    pred_all = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    episode_all = torch.cat(episode_indices) if episode_indices else torch.empty(0, dtype=torch.long)
    frame_all = torch.cat(frame_indices) if frame_indices else torch.empty(0, dtype=torch.long)
    return {
        "batches": count,
        "piper_ce": total_ce / max(count, 1),
        "elite_diffusion_loss_0_6": total_elite / max(count, 1) if report_elite_loss else None,
        "piper": piper_metrics(target_all, pred_all),
        "piper_temporal": temporal_stratified_metrics(
            episode_indices=episode_all,
            frame_indices=frame_all,
            piper_target=target_all,
            piper_pred=pred_all,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a project mixed action adapter around LeRobot PI05: Elite dims 0:6 remain continuous "
            "PI05 outputs while Piper intent uses an explicit three-class head. This is sim-data feasibility only."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init-policy-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--split-manifest", type=Path, help="Use the authoritative LeRobot export episode split.")
    parser.add_argument("--train-only", action="store_true", help="Real10 prototype only: no validation or best-checkpoint selection.")
    parser.add_argument("--max-stats-records", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=20260717)
    parser.add_argument("--overfit-first-batch", action="store_true")
    parser.add_argument("--freeze-pi05", action="store_true")
    parser.add_argument("--report-elite-val-loss", action="store_true")
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--elite-loss-weight", type=float, default=1.0)
    parser.add_argument("--piper-loss-weight", type=float, default=1.0)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-mode", choices=["state_only", "prefix_state"], default="prefix_state")
    parser.add_argument("--head-feature-dim", type=int, default=128)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--weighted-training-sampler",
        action="store_true",
        help="Use the deterministic weighted-with-replacement sampler for paired comparisons.",
    )
    parser.add_argument("--training-overlay-pack", type=Path)
    parser.add_argument("--training-overlay-project-root", type=Path, default=Path("."))
    parser.add_argument("--allow-diagnostic-overlay", action="store_true")
    add_pretrained_arguments(parser)
    args = parser.parse_args()
    validate_train_only_options(args)
    if args.train_only and not (args.freeze_pi05 and args.head_mode == "state_only" and args.init_policy_checkpoint):
        raise ValueError("real10 Piper requires a frozen Elite checkpoint and the existing state_only head")
    if args.max_records is None and args.split_manifest is None:
        args.max_records = 512

    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.allow_diagnostic_overlay and args.training_overlay_pack is None:
        raise ValueError("--allow-diagnostic-overlay requires --training-overlay-pack")
    if args.init_policy_checkpoint is not None and args.pretrained_name_or_path:
        raise ValueError("choose either --init-policy-checkpoint or --pretrained-name-or-path, not both")
    if args.freeze_pi05 and not (
        args.init_policy_checkpoint is not None
        or args.pretrained_name_or_path
        or args.allow_random_init
    ):
        raise ValueError(
            "--freeze-pi05 requires a policy source: --pretrained-name-or-path, "
            "--init-policy-checkpoint, or explicit --allow-random-init"
        )
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=not args.train_only)

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
        train_only=args.train_only,
    )
    stats_indices = train_indices if args.train_only else sample_stats_indices(train_indices, max_stats_records=args.max_stats_records, seed=args.seed)
    dataset_stats = compute_dataset_stats(dataset, stats_indices)
    overlay_dataset = None
    overlay_report = None
    if args.training_overlay_pack is not None:
        if not args.weighted_training_sampler:
            raise ValueError("--training-overlay-pack requires --weighted-training-sampler")
        overlay_dataset, overlay_report = load_e2_overlay(
            args.training_overlay_pack,
            project_root=args.training_overlay_project_root,
            base_state_mean=dataset_stats["observation.state"]["mean"],
            image_size=args.image_size,
            allow_diagnostic_overlay=args.allow_diagnostic_overlay,
        )
    if args.weighted_training_sampler:
        train_loader, training_sampler = make_weighted_training_loader(
            dataset,
            train_indices,
            overlay_dataset=overlay_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    else:
        train_loader = make_loader(
            dataset,
            train_indices,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        training_sampler = {"mode": "epoch_shuffle_without_replacement", "seed": int(args.seed)}
    val_loader = None if args.train_only else make_loader(
        dataset,
        val_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    pi05 = import_pi05()
    config = build_config(args, pi05, state_dim, action_dim)
    preprocessor, _postprocessor = pi05["make_pi05_pre_post_processors"](config, dataset_stats)
    loaded_checkpoint: dict[str, Any] | None = None
    if args.init_policy_checkpoint is not None:
        policy = pi05["PI05Policy"](config).to(args.device)
        loaded_checkpoint = load_policy_checkpoint(args.init_policy_checkpoint, policy)
        policy_initialization = {
            "status": "legacy_policy_checkpoint_loaded",
            "path": str(args.init_policy_checkpoint),
        }
    else:
        policy, policy_initialization = initialize_pi05_policy(
            pi05["PI05Policy"],
            config,
            args,
        )

    prefix_dim = int(policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    piper_head = PiperIntentHead(
        prefix_dim=prefix_dim,
        state_dim=state_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
        mode=args.head_mode,
        feature_dim=args.head_feature_dim,
    ).to(args.device)

    if args.freeze_pi05:
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval()
    else:
        policy.train()
    piper_head.train()

    train_counts = class_counts(dataset, train_indices)
    class_weight = (
        balanced_class_weights(train_counts, args.device)
        if args.class_weighting == "balanced"
        else None
    )
    parameter_groups: list[dict[str, Any]] = [
        {"params": list(piper_head.parameters()), "lr": args.head_lr},
    ]
    policy_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if policy_params:
        parameter_groups.append({"params": policy_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    optimized_params = [parameter for group in parameter_groups for parameter in group["params"]]

    args_json = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    log: dict[str, Any] = {
        "scope": REAL10_TRAIN_SCOPE if args.train_only else "PI05 mixed-action algorithm feasibility; not real-system validation",
        "root": str(args.root),
        "repo_id": args.repo_id,
        "dataset_len": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_only": args.train_only,
        "validation_performed": not args.train_only,
        "split": split_mode,
        "split_details": split_details(
            split_mode=split_mode,
            split_manifest=args.split_manifest,
            train_samples=len(train_indices),
            val_samples=len(val_indices),
        ),
        "args": args_json,
        "train_piper_class_counts": train_counts,
        "class_weights": class_weight.detach().cpu().tolist() if class_weight is not None else None,
        "class_weight_source": "base_training_split_only",
        "policy_parameters": trainable_parameter_summary(policy),
        "piper_head_parameters": trainable_parameter_summary(piper_head),
        "policy_initialization": policy_initialization,
        "piper_head_feature_contract": piper_head.feature_contract(),
        "training_sampler": training_sampler,
        "training_overlay": overlay_report,
        "normalization_stats": stats_to_json(dataset_stats),
        "semantics": {
            "elite": "PI05 continuous diffusion dims 0:6",
            "piper": "explicit three-class head trained from piper_intent_id",
            "piper_class_weight_source": "original base training split only; overlay affects sampling only",
            "piper_features": piper_head.feature_contract(),
            "pi05_state_limit": "LeRobot 0.4.4 PI05 diffusion path does not consume observation.state; this first mixed-head closes Piper semantics only",
            "compatibility_dims_6_32": "excluded from the project Elite diffusion loss and not interpreted as controls",
        },
        "steps": [],
    }
    if loaded_checkpoint is not None:
        log["init_checkpoint_final_val"] = loaded_checkpoint.get("final_val")
    if args.train_only:
        log["semantics"]["elite"] = "frozen real10 state-conditioned Elite translation policy; rotation not trained"
        log["semantics"]["pi05_state_limit"] = "state conditioning restored from the frozen Elite checkpoint"

    initial_val = None if args.train_only else evaluate(
        policy,
        piper_head,
        preprocessor,
        val_loader,
        max_batches=args.val_batches,
        eval_seed=args.eval_seed,
        report_elite_loss=args.report_elite_val_loss,
        class_weight=class_weight,
    )
    log["initial_val"] = initial_val
    if initial_val is not None:
        print(
            f"initial_val_piper_acc={initial_val['piper']['accuracy']:.4f} "
            f"balanced_acc={initial_val['piper']['balanced_accuracy']:.4f} "
            f"majority={initial_val['piper']['majority_baseline']:.4f}", flush=True,
        )
    else:
        print(f"real10 train-only: {len(train_indices)} samples, validation disabled, final checkpoint only", flush=True)

    if args.overfit_first_batch:
        train_iter = repeat(next(iter(train_loader)))
    elif args.weighted_training_sampler:
        train_iter = iter(train_loader)
    elif args.train_only:
        train_iter = repeat_loader(train_loader)
    else:
        train_iter = cycle(train_loader)
    start_time = time.time()
    best_val: dict[str, Any] | None = None
    best_step: int | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    for step in range(1, args.max_steps + 1):
        raw_batch = next(train_iter)
        processed = preprocessor(raw_batch)
        logits = piper_head_logits(policy, piper_head, processed)
        target = raw_batch["piper_intent_id"].to(logits.device)
        piper_loss = F.cross_entropy(logits, target, weight=class_weight)
        if args.freeze_pi05:
            elite_loss = None
            loss = args.piper_loss_weight * piper_loss
        else:
            elite_loss = elite_diffusion_loss(policy, processed)
            loss = args.elite_loss_weight * elite_loss + args.piper_loss_weight * piper_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optimized_params, args.grad_clip_norm)
        optimizer.step()

        row: dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "piper_loss": float(piper_loss.detach().cpu().item()),
            "elite_loss": float(elite_loss.detach().cpu().item()) if elite_loss is not None else None,
            "train_piper_acc": float((logits.argmax(dim=-1) == target).float().mean().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "elapsed_sec": time.time() - start_time,
        }
        if not args.train_only and (step % args.eval_every == 0 or step == args.max_steps):
            row["val"] = evaluate(
                policy,
                piper_head,
                preprocessor,
                val_loader,
                max_batches=args.val_batches,
                eval_seed=args.eval_seed,
                report_elite_loss=args.report_elite_val_loss,
                class_weight=class_weight,
            )
            if best_val is None or validation_score(row["val"], step) > validation_score(
                best_val, best_step or step
            ):
                best_val = row["val"]
                best_step = step
                if args.freeze_pi05:
                    best_head_state = cpu_state_dict(piper_head)
        log["steps"].append(row)
        if step % args.log_every == 0 or "val" in row:
            message = (
                f"step {step:04d}: loss={row['loss']:.6f} piper_loss={row['piper_loss']:.6f} "
                f"train_piper_acc={row['train_piper_acc']:.3f}"
            )
            if "val" in row:
                message += (
                    f" val_piper_acc={row['val']['piper']['accuracy']:.3f} "
                    f"val_balanced_acc={row['val']['piper']['balanced_accuracy']:.3f}"
                )
            print(message, flush=True)

    final_val = None if args.train_only else evaluate(
        policy,
        piper_head,
        preprocessor,
        val_loader,
        max_batches=args.val_batches,
        eval_seed=args.eval_seed,
        report_elite_loss=args.report_elite_val_loss,
        class_weight=class_weight,
    )
    log["final_val"] = final_val
    log["best_val"] = best_val
    log["best_step"] = best_step
    log["elapsed_sec"] = time.time() - start_time

    if args.save_checkpoint:
        checkpoint_common: dict[str, Any] = {
            "args": args_json,
            "semantics": log["semantics"],
            "normalization_stats": log["normalization_stats"],
            "split_details": log["split_details"],
            "train_only": args.train_only,
            "prefix_dim": prefix_dim,
            "state_dim": state_dim,
            "head_mode": args.head_mode,
            "head_feature_dim": args.head_feature_dim,
            "policy_initialization": policy_initialization,
            "final_val": final_val,
            "base_policy_checkpoint": str(args.init_policy_checkpoint) if args.init_policy_checkpoint else None,
            "train_piper_class_counts": train_counts,
            "class_weights": class_weight.detach().cpu().tolist() if class_weight is not None else None,
            "class_weight_source": "base_training_split_only",
            "training_sampler": training_sampler,
            "training_overlay": overlay_report,
        }
        checkpoint: dict[str, Any] = {
            **checkpoint_common,
            "piper_head_state": piper_head.state_dict(),
            "checkpoint_selection": "final_step",
            "checkpoint_step": args.max_steps,
        }
        if not args.freeze_pi05:
            checkpoint["policy_state"] = policy.state_dict()
        torch.save(checkpoint, args.out / "mixed_head_policy.pt")
        if args.freeze_pi05 and best_head_state is not None and best_step is not None:
            torch.save(
                {
                    **checkpoint_common,
                    "piper_head_state": best_head_state,
                    "final_val": best_val,
                    "checkpoint_selection": "best_validation_balanced_accuracy",
                    "checkpoint_step": best_step,
                },
                args.out / "best_mixed_head_policy.pt",
            )

    summary = {
        "train_only": args.train_only,
        "validation_performed": not args.train_only,
        "last_train_piper_loss": log["steps"][-1]["piper_loss"],
        "initial_val": initial_val,
        "best_val": best_val,
        "best_step": best_step,
        "final_val": final_val,
        "train_piper_class_counts": train_counts,
        "class_weights": class_weight.detach().cpu().tolist() if class_weight is not None else None,
        "class_weight_source": "base_training_split_only",
        "training_sampler": training_sampler,
        "training_overlay": overlay_report,
        "checkpoint_saved": bool(args.save_checkpoint),
        "best_checkpoint_saved": bool(
            args.save_checkpoint and args.freeze_pi05 and best_head_state is not None
        ),
        "base_policy_reused": bool(
            args.freeze_pi05
            and (args.init_policy_checkpoint is not None or args.pretrained_name_or_path)
        ),
    }
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.train_only:
        print(f"real10 final_step={args.max_steps} last_train_piper_loss={log['steps'][-1]['piper_loss']:.6f} wrote={args.out}", flush=True)
    else:
        print(
            f"final_val_piper_acc={final_val['piper']['accuracy']:.4f} "
            f"balanced_acc={final_val['piper']['balanced_accuracy']:.4f} "
            f"majority={final_val['piper']['majority_baseline']:.4f} wrote={args.out}", flush=True,
        )


if __name__ == "__main__":
    main()
