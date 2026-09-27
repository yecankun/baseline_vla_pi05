from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pi05_action_effect_world_model import (
    action32_candidates_to_active,
    extract_pi05_multiview_visual_latent,
)
from pi05_pretrained_loader import add_pretrained_arguments, initialize_pi05_policy
from probe_pi05_lerobot_adapter import (
    PI05ProbeDataset,
    build_config,
    collate_probe_batch,
    compute_dataset_stats,
    import_lerobot_dataset,
    import_pi05,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read one existing LeRobot record through the pinned PI05 image path and verify the "
            "action-effect frozen multiview latent/candidate interfaces without training."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-stats-records", type=int, default=851)
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
    parser.add_argument("--train-expert-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    add_pretrained_arguments(parser, allow_random_init=False)
    args = parser.parse_args()

    set_seed(args.seed)
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(f"sample-index {args.sample_index} outside dataset length {len(dataset)}")
    sample = dataset[args.sample_index]
    state_dim = int(sample["observation.state"].numel())
    action_dim = int(sample["action"].numel())
    if state_dim != 32 or action_dim != 32:
        raise ValueError(f"expected state/action 32D, got state={state_dim} action={action_dim}")
    stats_indices = list(range(min(len(dataset), max(int(args.max_stats_records), 1))))
    dataset_stats = compute_dataset_stats(dataset, stats_indices)
    probe_dataset = PI05ProbeDataset(dataset, [args.sample_index])
    raw_batch = next(
        iter(
            DataLoader(
                probe_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_probe_batch,
            )
        )
    )

    pi05 = import_pi05()
    config = build_config(args, pi05, state_dim, action_dim)
    preprocessor, _postprocessor = pi05["make_pi05_pre_post_processors"](
        config, dataset_stats
    )
    processed = preprocessor(raw_batch)
    policy, pretrained = initialize_pi05_policy(pi05["PI05Policy"], config, args)
    policy.eval()
    image_keys = ("observation.images.side", "observation.images.top")
    missing = [key for key in image_keys if key not in processed]
    if missing:
        raise ValueError(f"PI05 preprocessor omitted required image keys: {missing}")
    latent = extract_pi05_multiview_visual_latent(
        policy,
        [processed[key] for key in image_keys],
    )
    if latent.ndim != 3 or latent.shape[:2] != (1, 2):
        raise ValueError(f"unexpected PI05 multiview latent shape: {tuple(latent.shape)}")
    active = action32_candidates_to_active(raw_batch["action"].unsqueeze(1))
    if tuple(active.shape) != (1, 1, 1, 9):
        raise ValueError(f"unexpected active action shape: {tuple(active.shape)}")

    latent_cpu = latent.detach().float().cpu().contiguous()
    report = {
        "schema": "project2026_pi05_action_effect_feature_interface_probe_v1",
        "status": "passed",
        "scope": "existing accepted simulation record; feature interface only; no training",
        "dataset": {
            "root": str(args.root),
            "repo_id": args.repo_id,
            "records": len(dataset),
            "sample_index": args.sample_index,
            "stats_records": len(stats_indices),
        },
        "pretrained": pretrained,
        "visual_latent": {
            "extractor": "pi05_multiview_mean_patch_v1",
            "shape": list(latent_cpu.shape),
            "dtype": str(latent_cpu.dtype),
            "finite": bool(torch.isfinite(latent_cpu).all()),
            "sha256": hashlib.sha256(latent_cpu.numpy().tobytes()).hexdigest(),
            "backbone_frozen": True,
        },
        "active_action": {
            "shape": list(active.shape),
            "interface": "elite_tcp_delta_6d + piper_intent_id",
        },
        "optimizer_steps": 0,
        "training_started": False,
        "exact_truth_policy_input_allowed": False,
        "senior_historical_real_data_used": False,
    }
    if not report["visual_latent"]["finite"]:
        raise ValueError("PI05 multiview latent contains non-finite values")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"status=passed latent_shape={list(latent_cpu.shape)} "
        f"active_action_shape={list(active.shape)} optimizer_steps=0",
        flush=True,
    )


if __name__ == "__main__":
    main()

