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
from torch.utils.data import DataLoader

from pi05_pretrained_loader import add_pretrained_arguments, initialize_pi05_policy
from pi05_e2_overlay import load_e2_overlay, make_weighted_training_loader
from pi05_state_conditioning import enable_state_conditioning, state_conditioning_enabled

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


ELITE_ACTION_DIMS = 6
ELITE_TRANSLATION_DIMS = 3
EXPORT_MANIFEST_SCHEMA = "project_2026_lerobot_export_v0"
REAL10_TRAIN_SCOPE = "user_authorized_real10_train_only_prototype"


def loss_scope(active_action_dims: int) -> str:
    if active_action_dims == ELITE_TRANSLATION_DIMS:
        return "Elite TCP translation delta dims 0:3 only; rotation and Piper compatibility dims excluded"
    if active_action_dims == ELITE_ACTION_DIMS:
        return "Elite TCP delta dims 0:6 only; Piper compatibility dims excluded"
    if active_action_dims == ACTIVE_ACTION_DIMS:
        return "legacy active compatibility dims 0:9"
    raise ValueError(f"unsupported active action dims: {active_action_dims}")


def record_level_split(selected: list[int], *, val_fraction: float) -> tuple[list[int], list[int]]:
    if len(selected) < 2:
        raise ValueError(f"need at least two selected records for fallback split, got {len(selected)}")
    val_count = max(1, int(round(len(selected) * float(val_fraction))))
    val_count = min(val_count, len(selected) - 1)
    return selected[:-val_count], selected[-val_count:]


def split_by_episode(
    dataset: Any,
    *,
    val_fraction: float,
    seed: int,
    max_records: int | None,
    split_manifest: Path | None = None,
    train_only: bool = False,
) -> tuple[list[int], list[int], str]:
    if train_only and split_manifest is None:
        raise ValueError("--train-only requires the real10 export --split-manifest")
    if split_manifest is not None:
        if max_records is not None:
            raise ValueError("--split-manifest requires all records; do not pass --max-records")
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != EXPORT_MANIFEST_SCHEMA:
            raise ValueError(f"unexpected split manifest schema: {manifest.get('schema')!r}")
        if int(manifest.get("frames", -1)) != len(dataset):
            raise ValueError(
                f"split manifest frames {manifest.get('frames')!r} != dataset length {len(dataset)}"
            )
        train_episodes = {int(value) for value in manifest.get("train_episode_indices", [])}
        val_episodes = {int(value) for value in manifest.get("val_episode_indices", [])}
        if train_only:
            if manifest.get("training_use", {}).get("scope") != REAL10_TRAIN_SCOPE:
                raise ValueError("--train-only is restricted to the explicit real10 prototype export")
            if not train_episodes or val_episodes:
                raise ValueError("real10 train-only export must contain train episodes and no validation")
        elif not train_episodes or not val_episodes:
            raise ValueError("split manifest must contain non-empty train and validation episode lists")
        overlap = train_episodes & val_episodes
        if overlap:
            raise ValueError(f"split manifest train/validation overlap: {sorted(overlap)}")
        train_indices: list[int] = []
        val_indices: list[int] = []
        observed_episodes: set[int] = set()
        for idx in range(len(dataset)):
            episode = int(dataset[idx]["episode_index"])
            observed_episodes.add(episode)
            if episode in train_episodes:
                train_indices.append(idx)
            elif episode in val_episodes:
                val_indices.append(idx)
            else:
                raise ValueError(f"dataset episode {episode} is absent from split manifest")
        expected_episodes = train_episodes | val_episodes
        if observed_episodes != expected_episodes:
            raise ValueError(
                "split manifest/dataset episode mismatch: "
                f"missing={sorted(expected_episodes - observed_episodes)} "
                f"unexpected={sorted(observed_episodes - expected_episodes)}"
            )
        if int(manifest.get("episodes", -1)) != len(observed_episodes):
            raise ValueError(
                f"split manifest episodes {manifest.get('episodes')!r} != observed {len(observed_episodes)}"
            )
        mode = "real10_export_train_only" if train_only else "export_manifest_episode_split"
        return train_indices, val_indices, mode

    selected = list(range(len(dataset)))
    if max_records is not None:
        selected = selected[: max(int(max_records), 0)]
    if len(selected) < 2:
        raise ValueError(f"need at least two selected records, got {len(selected)}")
    episode_to_indices: dict[int, list[int]] = {}
    for idx in selected:
        item = dataset[idx]
        episode = int(item["episode_index"])
        episode_to_indices.setdefault(episode, []).append(idx)
    episodes = sorted(episode_to_indices)
    if len(episodes) < 2:
        train_indices, val_indices = record_level_split(selected, val_fraction=val_fraction)
        return train_indices, val_indices, "record_level_fallback_single_episode"
    rng = random.Random(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * float(val_fraction))))
    val_episodes = set(episodes[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for episode in sorted(episode_to_indices):
        if episode in val_episodes:
            val_indices.extend(episode_to_indices[episode])
        else:
            train_indices.extend(episode_to_indices[episode])
    if not train_indices or not val_indices:
        train_indices, val_indices = record_level_split(selected, val_fraction=val_fraction)
        return train_indices, val_indices, "record_level_fallback_empty_episode_split"
    return train_indices, val_indices, "episode_level"


def split_details(
    *,
    split_mode: str,
    split_manifest: Path | None,
    train_samples: int,
    val_samples: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": split_mode,
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "manifest": str(split_manifest) if split_manifest is not None else None,
    }
    if split_manifest is not None:
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        report.update(
            {
                "manifest_schema": manifest.get("schema"),
                "manifest_frames": manifest.get("frames"),
                "manifest_episodes": manifest.get("episodes"),
                "train_episode_indices": manifest.get("train_episode_indices"),
                "val_episode_indices": manifest.get("val_episode_indices"),
                "adapter_version": manifest.get("adapter_version"),
                "training_use": manifest.get("training_use"),
                "timing": manifest.get("timing"),
                "input_preprocessing": manifest.get("input_preprocessing"),
                "openpi_compat": manifest.get("openpi_compat"),
            }
        )
    return report


def sample_stats_indices(indices: list[int], *, max_stats_records: int, seed: int) -> list[int]:
    if len(indices) <= max_stats_records:
        return list(indices)
    rng = random.Random(seed)
    return sorted(rng.sample(indices, max_stats_records))


def repeat_loader(loader):
    """Reshuffle each real-data epoch without caching image batches in cycle()."""
    while True:
        yield from loader


def validate_train_only_options(args) -> None:
    if not args.train_only:
        return
    if args.max_steps < 1 or not args.save_checkpoint:
        raise ValueError("real10 train-only requires positive steps and --save-checkpoint")
    if args.overfit_first_batch or args.training_overlay_pack or args.weighted_training_sampler:
        raise ValueError("real10 uses all original training episodes, without overfit/overlay sampling")


def summarize_loss_dict(loss_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in loss_dict.items():
        if isinstance(value, (int, float, str)):
            out[key] = value
        elif key == "loss_per_dim":
            out[key] = list(value[:ACTIVE_ACTION_DIMS])
        else:
            out[key] = str(value)[:200]
    return out


def compute_policy_loss(
    policy: torch.nn.Module,
    processed: dict[str, Any],
    *,
    active_action_dims: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if active_action_dims == ACTIVE_ACTION_DIMS:
        return policy(processed)
    if active_action_dims not in {ELITE_TRANSLATION_DIMS, ELITE_ACTION_DIMS}:
        raise ValueError(f"unsupported active action dims: {active_action_dims}")
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(processed)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(processed)
    if state_conditioning_enabled(policy):
        losses = policy.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            state=processed["observation.state"],
        )
    else:
        losses = policy.model.forward(images, img_masks, tokens, masks, actions)
    elite_losses = losses[:, :, :active_action_dims]
    return elite_losses.mean(), {
        "loss_per_dim": elite_losses.detach().float().mean(dim=(0, 1)).cpu().tolist(),
        "loss_scope": (
            "elite_tcp_translation_delta_0_3"
            if active_action_dims == ELITE_TRANSLATION_DIMS
            else "elite_tcp_delta_0_6"
        ),
    }


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


@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    preprocessor: Any,
    loader: DataLoader,
    *,
    max_batches: int,
    eval_seed: int | None,
    active_action_dims: int,
) -> dict[str, Any]:
    policy.eval()
    with rng_context(eval_seed):
        set_torch_seed(eval_seed)
        total = 0.0
        count = 0
        last_loss_dict: dict[str, Any] = {}
        per_dim_total: torch.Tensor | None = None
        per_dim_count = 0
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            processed = preprocessor(raw_batch)
            loss, loss_dict = compute_policy_loss(
                policy,
                processed,
                active_action_dims=active_action_dims,
            )
            total += float(loss.detach().cpu().item())
            count += 1
            last_loss_dict = summarize_loss_dict(loss_dict)
            loss_per_dim = loss_dict.get("loss_per_dim")
            if loss_per_dim is not None:
                values = torch.as_tensor(loss_per_dim, dtype=torch.float64).detach().cpu()
                per_dim_total = values if per_dim_total is None else per_dim_total + values
                per_dim_count += 1
    policy.train()
    return {
        "loss": total / max(count, 1),
        "batches": count,
        "last_loss_dict": last_loss_dict,
        "mean_loss_per_dim": (
            (per_dim_total / per_dim_count).tolist()
            if per_dim_total is not None and per_dim_count > 0
            else None
        ),
    }


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


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tensors = sum(1 for _ in model.parameters())
    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "parameter_tensors": int(tensors),
        "trainable_parameter_tensors": int(trainable_tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short official LeRobot PI05 adapter training smoke on the project_2026 LeRobotDataset export. "
            "This only tests whether PI05 loss can optimize on project data; it is not real-system validation."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--split-manifest", type=Path, help="Use the authoritative LeRobot export episode split.")
    parser.add_argument("--train-only", action="store_true", help="Real10 prototype only: no validation or model selection.")
    parser.add_argument("--max-stats-records", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=10007, help="Fix PI05 validation noise/time sampling for comparable validation losses.")
    parser.add_argument("--fixed-train-loss-seed", type=int, help="Fix PI05 training noise/time sampling. Use only for overfit diagnostics.")
    parser.add_argument("--overfit-first-batch", action="store_true", help="Repeat one training batch to test whether the optimizer can reduce PI05 loss.")
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument(
        "--elite-only-loss",
        action="store_true",
        help="Optimize only Elite TCP-delta dims 0:6; exclude Piper compatibility one-hot dims from diffusion loss.",
    )
    loss_group.add_argument(
        "--elite-translation-only-loss",
        action="store_true",
        help="Optimize only Elite TCP translation-delta dims 0:3; exclude zero-rotation and Piper compatibility dims.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-5)
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
    parser.add_argument("--save-checkpoint", action="store_true", help="Save the final PI05 policy state dict. This can be large.")
    parser.add_argument(
        "--weighted-training-sampler",
        action="store_true",
        help="Use a deterministic weighted-with-replacement train sampler for paired comparisons.",
    )
    parser.add_argument("--training-overlay-pack", type=Path)
    parser.add_argument("--training-overlay-project-root", type=Path, default=Path("."))
    parser.add_argument("--allow-diagnostic-overlay", action="store_true")
    parser.add_argument(
        "--state-conditioning",
        action="store_true",
        help="Add a trainable 32D observation.state prefix token after pretrained loading.",
    )
    add_pretrained_arguments(parser)
    args = parser.parse_args()
    validate_train_only_options(args)
    if args.train_only and not (args.elite_translation_only_loss and args.state_conditioning):
        raise ValueError("real10 Elite training requires translation-only loss and state conditioning")
    if args.max_records is None and args.split_manifest is None:
        args.max_records = 512
    if args.allow_diagnostic_overlay and args.training_overlay_pack is None:
        raise ValueError("--allow-diagnostic-overlay requires --training-overlay-pack")
    active_action_dims = (
        ELITE_TRANSLATION_DIMS
        if args.elite_translation_only_loss
        else ELITE_ACTION_DIMS if args.elite_only_loss else ACTIVE_ACTION_DIMS
    )
    active_loss_scope = loss_scope(active_action_dims)

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
    policy, pretrained_report = initialize_pi05_policy(pi05["PI05Policy"], config, args)
    state_conditioning = None
    if args.state_conditioning:
        state_conditioning = enable_state_conditioning(policy, state_dim)
    policy.train()

    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        raise ValueError("PI05 policy has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    log: dict[str, Any] = {
        "root": str(args.root),
        "repo_id": args.repo_id,
        "dataset_len": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_only": args.train_only,
        "validation_performed": not args.train_only,
        "stats_samples": len(stats_indices),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "active_action_dims": active_action_dims,
        "split": split_mode,
        "split_details": split_details(
            split_mode=split_mode,
            split_manifest=args.split_manifest,
            train_samples=len(train_indices),
            val_samples=len(val_indices),
        ),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "parameter_summary": trainable_parameter_summary(policy),
        "pretrained": pretrained_report,
        "state_conditioning": state_conditioning,
        "training_sampler": training_sampler,
        "training_overlay": overlay_report,
        "normalization_stats": stats_to_json(dataset_stats),
        "semantics": {
            "scope": REAL10_TRAIN_SCOPE if args.train_only else "short PI05 adapter training smoke only",
            "action": "action_32 compatibility tensor; dims 0:6 are Elite TCP delta, dims 6:9 are Piper intent one-hot compatibility, dims 9:32 are padding",
            "loss_scope": active_loss_scope,
            "preferred_project_target": "Elite TCP delta regression plus Piper discrete intent classification",
            "state_conditioning": (
                "32D observation.state projected to one prefix token after image/language prefix"
                if state_conditioning is not None
                else "disabled"
            ),
            "not_validation": "training fit is not held-out evaluation or real-system validation",
        },
        "steps": [],
    }

    start_time = time.time()
    if args.overfit_first_batch:
        train_iter = repeat(next(iter(train_loader)))
    elif args.weighted_training_sampler:
        train_iter = iter(train_loader)
    elif args.train_only:
        train_iter = repeat_loader(train_loader)
    else:
        train_iter = cycle(train_loader)
    initial_val = None if args.train_only else evaluate(
        policy,
        preprocessor,
        val_loader,
        max_batches=args.val_batches,
        eval_seed=args.eval_seed,
        active_action_dims=active_action_dims,
    )
    log["initial_val"] = initial_val
    if initial_val is not None:
        print(f"initial_val_loss={initial_val['loss']:.6f} batches={initial_val['batches']}", flush=True)
    else:
        print(f"real10 train-only: {len(train_indices)} samples, validation disabled, final checkpoint only", flush=True)

    for step in range(1, args.max_steps + 1):
        raw_batch = next(train_iter)
        processed = preprocessor(raw_batch)
        with rng_context(args.fixed_train_loss_seed):
            set_torch_seed(args.fixed_train_loss_seed)
            loss, loss_dict = compute_policy_loss(
                policy,
                processed,
                active_action_dims=active_action_dims,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        optimizer.step()

        row: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "loss_dict": summarize_loss_dict(loss_dict),
            "elapsed_sec": time.time() - start_time,
        }
        if not args.train_only and (step % args.eval_every == 0 or step == args.max_steps):
            row["val"] = evaluate(
                policy,
                preprocessor,
                val_loader,
                max_batches=args.val_batches,
                eval_seed=args.eval_seed,
                active_action_dims=active_action_dims,
            )
        log["steps"].append(row)

        if step % args.log_every == 0 or "val" in row:
            msg = f"step {step:04d}: train_loss={row['train_loss']:.6f} grad_norm={row['grad_norm']:.4f}"
            if "val" in row:
                msg += f" val_loss={row['val']['loss']:.6f}"
            print(msg, flush=True)

    final_val = None if args.train_only else evaluate(
        policy,
        preprocessor,
        val_loader,
        max_batches=args.val_batches,
        eval_seed=args.eval_seed,
        active_action_dims=active_action_dims,
    )
    log["final_val"] = final_val
    log["elapsed_sec"] = time.time() - start_time
    log["loss_delta"] = None if args.train_only else {
        "initial_val_loss": float(initial_val["loss"]),
        "final_val_loss": float(final_val["loss"]),
        "final_minus_initial": float(final_val["loss"] - initial_val["loss"]),
    }

    if args.save_checkpoint:
        torch.save(
            {
                "model_state": policy.state_dict(),
                "args": log["args"],
                "parameter_summary": log["parameter_summary"],
                "pretrained": pretrained_report,
                "normalization_stats": log["normalization_stats"],
                "split_details": log["split_details"],
                "train_only": args.train_only,
                "final_val": final_val,
                "semantics": log["semantics"],
                "checkpoint_selection": "final_step",
                "checkpoint_step": args.max_steps,
                "active_action_dims": log["active_action_dims"],
                "loss_scope": log["semantics"]["loss_scope"],
                "state_conditioning": state_conditioning,
                "training_sampler": training_sampler,
                "training_overlay": overlay_report,
            },
            args.out / "final_policy.pt",
        )

    (args.out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "initial_val": initial_val,
                "train_only": args.train_only,
                "validation_performed": not args.train_only,
                "last_train_loss": log["steps"][-1]["train_loss"] if log["steps"] else None,
                "final_val": final_val,
                "loss_delta": log["loss_delta"],
                "parameter_summary": log["parameter_summary"],
                "pretrained": pretrained_report,
                "state_conditioning": state_conditioning,
                "training_sampler": training_sampler,
                "training_overlay": overlay_report,
                "checkpoint_saved": bool(args.save_checkpoint),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.train_only:
        print(f"real10 final_step={args.max_steps} last_train_loss={log['steps'][-1]['train_loss']:.6f} wrote={args.out}", flush=True)
    else:
        print(
            f"final_val_loss={final_val['loss']:.6f} "
            f"delta={log['loss_delta']['final_minus_initial']:.6f} wrote={args.out}", flush=True,
        )


if __name__ == "__main__":
    main()
