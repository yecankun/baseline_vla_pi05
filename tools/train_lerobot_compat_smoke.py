from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable


ACTIVE_ACTION_DIMS = 9


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_stats(values: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0).float()
    std = values.std(dim=0, unbiased=False).float()
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean, std


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    weighted = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return weighted.sum() / denom


def split_by_episode(dataset: Any, *, val_fraction: float, seed: int, max_records: int | None) -> tuple[list[int], list[int]]:
    selected = list(range(len(dataset)))
    if max_records is not None:
        selected = selected[: max(int(max_records), 0)]
    episode_to_indices: dict[int, list[int]] = {}
    for idx in selected:
        item = dataset[idx]
        episode = int(item["episode_index"])
        episode_to_indices.setdefault(episode, []).append(idx)
    episodes = sorted(episode_to_indices)
    rng = random.Random(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * float(val_fraction)))) if len(episodes) > 1 else 0
    val_episodes = set(episodes[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for episode in sorted(episode_to_indices):
        if episode in val_episodes:
            val_indices.extend(episode_to_indices[episode])
        else:
            train_indices.extend(episode_to_indices[episode])
    if not train_indices or not val_indices:
        raise ValueError(f"need non-empty train/val split, got train={len(train_indices)} val={len(val_indices)}")
    return train_indices, val_indices


def compute_state_action_stats(dataset: Any, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for idx in indices:
        item = dataset[idx]
        states.append(item["observation.state"].float())
        actions.append(item["action"].float())
    state_values = torch.stack(states, dim=0)
    action_values = torch.stack(actions, dim=0)
    return (*normalize_stats(state_values), *normalize_stats(action_values))


class LeRobotCompatDataset(Dataset):
    def __init__(
        self,
        *,
        dataset: Any,
        indices: list[int],
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        image_normalization: str,
    ):
        self.dataset = dataset
        self.indices = indices
        self.state_mean = state_mean.float()
        self.state_std = state_std.float()
        self.action_mean = action_mean.float()
        self.action_std = action_std.float()
        self.image_normalization = image_normalization
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def _image(self, value: torch.Tensor) -> torch.Tensor:
        image = value.float()
        if image.max() > 2.0:
            image = image / 255.0
        if self.image_normalization == "imagenet":
            image = (image - self.image_mean) / self.image_std
        return image

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        row = self.dataset[idx]
        state = (row["observation.state"].float() - self.state_mean) / self.state_std
        action = (row["action"].float() - self.action_mean) / self.action_std
        return {
            "side": self._image(row["observation.images.side"]),
            "top": self._image(row["observation.images.top"]),
            "state": state.float(),
            "action": action.float(),
            "elite": row["elite_tcp_delta_6d"].float(),
            "piper": torch.as_tensor(int(row["piper_intent_id"]), dtype=torch.long),
        }


class SmallImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class LeRobotCompatSmokePolicy(nn.Module):
    def __init__(self, *, state_dim: int = 32, action_dim: int = 32):
        super().__init__()
        self.side_encoder = SmallImageEncoder()
        self.top_encoder = SmallImageEncoder()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(192, action_dim)
        self.piper_head = nn.Linear(192, 3)

    def forward(self, side: torch.Tensor, top: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([self.side_encoder(side), self.top_encoder(top), self.state_encoder(state)], dim=1)
        hidden = self.trunk(feat)
        return self.action_head(hidden), self.piper_head(hidden)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_mask: torch.Tensor,
    action_loss_weight: float,
    piper_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total = total_action = total_piper = total_acc = total_action_mae = total_elite_mae = 0.0
    count = 0
    mean = action_mean.to(device)
    std = action_std.to(device)
    mask = action_mask.to(device)
    for batch in loader:
        side = batch["side"].to(device)
        top = batch["top"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        piper = batch["piper"].to(device)
        elite = batch["elite"].to(device)
        pred_action, piper_logits = model(side, top, state)
        action_loss = masked_smooth_l1(pred_action, action, mask.expand_as(action))
        piper_loss = F.cross_entropy(piper_logits, piper)
        loss = action_loss_weight * action_loss + piper_loss_weight * piper_loss
        pred_raw = pred_action * std + mean
        true_raw = action * std + mean
        active_mae = (torch.abs(pred_raw - true_raw) * mask).sum() / (mask.sum() * len(piper)).clamp_min(1.0)
        elite_mae = torch.mean(torch.abs(pred_raw[:, :6] - elite))
        bs = len(piper)
        total += float(loss.item()) * bs
        total_action += float(action_loss.item()) * bs
        total_piper += float(piper_loss.item()) * bs
        total_acc += float((piper_logits.argmax(dim=1) == piper).float().mean().item()) * bs
        total_action_mae += float(active_mae.item()) * bs
        total_elite_mae += float(elite_mae.item()) * bs
        count += bs
    denom = max(count, 1)
    return {
        "loss": total / denom,
        "action_loss": total_action / denom,
        "piper_loss": total_piper / denom,
        "piper_acc": total_acc / denom,
        "action_active_mae": total_action_mae / denom,
        "elite_raw_mae": total_elite_mae / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small smoke model directly from a LeRobotDataset export.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--piper-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-normalization", choices=["none", "imagenet"], default="imagenet")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    train_indices, val_indices = split_by_episode(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_records=args.max_records,
    )
    state_mean, state_std, action_mean, action_std = compute_state_action_stats(dataset, train_indices)
    train_set = LeRobotCompatDataset(
        dataset=dataset,
        indices=train_indices,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        image_normalization=args.image_normalization,
    )
    val_set = LeRobotCompatDataset(
        dataset=dataset,
        indices=val_indices,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        image_normalization=args.image_normalization,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    state_dim = int(dataset[0]["observation.state"].numel())
    action_dim = int(dataset[0]["action"].numel())
    action_mask = torch.zeros(action_dim, dtype=torch.float32)
    action_mask[: min(ACTIVE_ACTION_DIMS, action_dim)] = 1.0
    device = torch.device(args.device)
    model = LeRobotCompatSmokePolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "root": str(args.root),
        "repo_id": args.repo_id,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "active_action_dims": ACTIVE_ACTION_DIMS,
        "split": "episode_level",
        "epochs": [],
        "normalization": {
            "state_mean": state_mean.numpy().astype(float).tolist(),
            "state_std": state_std.numpy().astype(float).tolist(),
            "action_mean": action_mean.numpy().astype(float).tolist(),
            "action_std": action_std.numpy().astype(float).tolist(),
        },
        "semantics": {
            "action": "action_32 compatibility tensor; only dims 0:9 are active in the smoke loss",
            "elite_target": "elite_tcp_delta_6d explicit field",
            "piper_target": "piper_intent_id explicit field",
        },
    }

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_action = total_piper = total_acc = 0.0
        count = 0
        mask = action_mask.to(device)
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            piper = batch["piper"].to(device)
            pred_action, piper_logits = model(side, top, state)
            action_loss = masked_smooth_l1(pred_action, action, mask.expand_as(action))
            piper_loss = F.cross_entropy(piper_logits, piper)
            loss = args.action_loss_weight * action_loss + args.piper_loss_weight * piper_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = len(piper)
            total += float(loss.item()) * bs
            total_action += float(action_loss.item()) * bs
            total_piper += float(piper_loss.item()) * bs
            total_acc += float((piper_logits.argmax(dim=1) == piper).float().mean().item()) * bs
            count += bs

        train = {
            "loss": total / max(count, 1),
            "action_loss": total_action / max(count, 1),
            "piper_loss": total_piper / max(count, 1),
            "piper_acc": total_acc / max(count, 1),
        }
        val = evaluate(
            model,
            val_loader,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
            action_mask=action_mask,
            action_loss_weight=args.action_loss_weight,
            piper_loss_weight=args.piper_loss_weight,
        )
        row = {"epoch": epoch, "train": train, "val": val}
        log["epochs"].append(row)
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
            f"train_piper_acc={train['piper_acc']:.3f} val_piper_acc={val['piper_acc']:.3f} "
            f"val_action_mae={val['action_active_mae']:.4f} val_elite_mae={val['elite_raw_mae']:.4f}"
        )
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "normalization": log["normalization"],
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val,
                },
                args.out / "best_model.pt",
            )

    (args.out / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps({"best_val_loss": best_val, "last_epoch": log["epochs"][-1]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
