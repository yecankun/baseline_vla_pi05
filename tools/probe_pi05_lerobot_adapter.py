from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pi05_pretrained_loader import add_pretrained_arguments, initialize_pi05_policy


ACTIVE_ACTION_DIMS = 9
DEFAULT_TASK_TEXT = {
    "left": "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the left branch.",
    "right": "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the right branch.",
    "unknown": "Use Piper insertion and Elite magnetic guidance to steer the guidewire.",
}


def import_lerobot_dataset():
    errors: list[str] = []
    try:
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.datasets: {exc!r}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.datasets.lerobot_dataset: {exc!r}")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.common.datasets.lerobot_dataset: {exc!r}")
    raise ImportError(f"Could not import LeRobotDataset. Errors: {errors}")


def import_pi05():
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    except ImportError as exc:
        raise ImportError(
            "Could not import LeRobot PI05 policy components. Install the policy dependencies in the "
            "4090 conda environment, for example: pip install 'lerobot[transformers-dep]'"
        ) from exc

    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "PI05Config": PI05Config,
        "PI05Policy": PI05Policy,
        "make_pi05_pre_post_processors": make_pi05_pre_post_processors,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(dataset: Any, *, max_records: int | None, seed: int) -> list[int]:
    indices = list(range(len(dataset)))
    if max_records is not None:
        indices = indices[: max(int(max_records), 0)]
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def normalize_stats(values: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    mean = values.mean(dim=0).float()
    std = values.std(dim=0, unbiased=False).float()
    std = torch.where(std < eps, torch.ones_like(std), std)
    return {"mean": mean, "std": std}


def compute_dataset_stats(dataset: Any, indices: list[int]) -> dict[str, dict[str, torch.Tensor]]:
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for idx in indices:
        item = dataset[idx]
        states.append(item["observation.state"].float())
        actions.append(item["action"].float())
    return {
        "observation.state": normalize_stats(torch.stack(states, dim=0)),
        "action": normalize_stats(torch.stack(actions, dim=0)),
    }


def task_from_state_one_hot(state: torch.Tensor) -> str:
    if state.numel() >= 16:
        left_score = float(state[14].item())
        right_score = float(state[15].item())
        if left_score > right_score:
            return DEFAULT_TASK_TEXT["left"]
        if right_score > left_score:
            return DEFAULT_TASK_TEXT["right"]
    return DEFAULT_TASK_TEXT["unknown"]


def ensure_image_float(image: torch.Tensor) -> torch.Tensor:
    value = image.float()
    if value.numel() and float(value.max().item()) > 2.0:
        value = value / 255.0
    return value.clamp(0.0, 1.0)


class PI05ProbeDataset(Dataset):
    def __init__(self, dataset: Any, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataset[self.indices[item]]
        state = row["observation.state"].float()
        task = row.get("task")
        if not isinstance(task, str) or not task.strip():
            task = task_from_state_one_hot(state)
        return {
            "observation.images.side": ensure_image_float(row["observation.images.side"]),
            "observation.images.top": ensure_image_float(row["observation.images.top"]),
            "observation.state": state,
            "action": row["action"].float().unsqueeze(0),
            "task": task,
            "elite_tcp_delta_6d": row["elite_tcp_delta_6d"].float(),
            "piper_intent_id": torch.as_tensor(int(row["piper_intent_id"]), dtype=torch.long),
            "episode_index": torch.as_tensor(int(row["episode_index"]), dtype=torch.long),
            "frame_index": torch.as_tensor(int(row["frame_index"]), dtype=torch.long),
        }


def collate_probe_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observation.images.side": torch.stack([row["observation.images.side"] for row in rows], dim=0),
        "observation.images.top": torch.stack([row["observation.images.top"] for row in rows], dim=0),
        "observation.state": torch.stack([row["observation.state"] for row in rows], dim=0),
        "action": torch.stack([row["action"] for row in rows], dim=0),
        "task": [str(row["task"]) for row in rows],
        "elite_tcp_delta_6d": torch.stack([row["elite_tcp_delta_6d"] for row in rows], dim=0),
        "piper_intent_id": torch.stack([row["piper_intent_id"] for row in rows], dim=0),
        "episode_index": torch.stack([row["episode_index"] for row in rows], dim=0),
        "frame_index": torch.stack([row["frame_index"] for row in rows], dim=0),
    }


def tensor_shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, list):
        return {"type": "list", "len": len(value), "sample": str(value[0])[:120] if value else ""}
    return {"type": type(value).__name__, "value": str(value)[:120]}


def build_config(args: argparse.Namespace, pi05: dict[str, Any], state_dim: int, action_dim: int):
    FeatureType = pi05["FeatureType"]
    NormalizationMode = pi05["NormalizationMode"]
    PolicyFeature = pi05["PolicyFeature"]
    PI05Config = pi05["PI05Config"]

    input_features = {
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    return PI05Config(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        use_amp=args.use_amp,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        dtype=args.dtype,
        chunk_size=1,
        n_action_steps=1,
        max_state_dim=max(args.max_state_dim, state_dim),
        max_action_dim=max(args.max_action_dim, action_dim),
        image_resolution=(args.image_size, args.image_size),
        tokenizer_max_length=args.tokenizer_max_length,
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
        },
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def stats_to_json(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            stat_name: tensor.detach().cpu().numpy().astype(float).tolist()
            for stat_name, tensor in value.items()
        }
        for key, value in stats.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the official LeRobot PI05 policy path on the project_2026 LeRobotDataset export. "
            "This is an interface smoke test: action_32 remains a compatibility tensor, while the "
            "preferred project target is still Elite TCP delta plus Piper intent."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-stats-records", type=int, default=2048)
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
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--train-expert-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--backward", action="store_true")
    add_pretrained_arguments(parser)
    args = parser.parse_args()

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

    all_indices = split_indices(dataset, max_records=args.max_records, seed=args.seed)
    if not all_indices:
        raise ValueError("no records selected")
    stats_indices = list(range(min(len(dataset), max(int(args.max_stats_records), 1))))
    dataset_stats = compute_dataset_stats(dataset, stats_indices)

    probe_set = PI05ProbeDataset(dataset, all_indices)
    loader = DataLoader(
        probe_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_probe_batch,
    )
    raw_batch = next(iter(loader))

    pi05 = import_pi05()
    config = build_config(args, pi05, state_dim, action_dim)
    make_pre_post = pi05["make_pi05_pre_post_processors"]
    try:
        preprocessor, _postprocessor = make_pre_post(config, dataset_stats)
    except ImportError as exc:
        raise ImportError(
            "PI05 preprocessor could not be built, likely because tokenizer dependencies are missing. "
            "In the 4090 env, run: pip install 'lerobot[transformers-dep]'"
        ) from exc

    processed = preprocessor(raw_batch)
    summary: dict[str, Any] = {
        "root": str(args.root),
        "repo_id": args.repo_id,
        "dataset_len": len(dataset),
        "selected_records": len(all_indices),
        "stats_records": len(stats_indices),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "active_action_dims": ACTIVE_ACTION_DIMS,
        "chunk_size": 1,
        "raw_batch": {key: tensor_shape(value) for key, value in raw_batch.items()},
        "processed_batch": {key: tensor_shape(value) for key, value in processed.items()},
        "pi05_config": {
            "paligemma_variant": args.paligemma_variant,
            "action_expert_variant": args.action_expert_variant,
            "dtype": args.dtype,
            "device": args.device,
            "freeze_vision_encoder": bool(args.freeze_vision_encoder),
            "train_expert_only": bool(args.train_expert_only),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
        "normalization_stats": stats_to_json(dataset_stats),
        "semantics": {
            "action": "action_32 compatibility tensor; dims 0:6 are Elite TCP delta, dims 6:9 are Piper intent one-hot compatibility, dims 9:32 are padding",
            "preferred_project_target": "Elite TCP delta regression plus Piper discrete intent classification",
            "probe_scope": "PI05 interface feasibility only, not real-system validation",
        },
    }

    if not args.preprocess_only:
        PI05Policy = pi05["PI05Policy"]
        policy, pretrained_report = initialize_pi05_policy(PI05Policy, config, args)
        summary["pretrained"] = pretrained_report
        policy.train()
        loss, loss_dict = policy(processed)
        summary["loss"] = float(loss.detach().cpu().item())
        summary["loss_dict"] = {
            key: (value if isinstance(value, (int, float, str)) else value[:ACTIVE_ACTION_DIMS])
            for key, value in loss_dict.items()
        }
        if args.backward:
            loss.backward()
            grad_norm_sq = 0.0
            grad_param_count = 0
            for param in policy.parameters():
                if param.grad is not None:
                    grad_norm_sq += float(param.grad.detach().float().pow(2).sum().cpu().item())
                    grad_param_count += 1
            summary["backward"] = {
                "ok": True,
                "grad_param_count": grad_param_count,
                "grad_l2": float(grad_norm_sq**0.5),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["dataset_len", "selected_records", "state_dim", "action_dim"]}, indent=2))
    if "loss" in summary:
        print(f"pi05 probe loss={summary['loss']:.6f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
