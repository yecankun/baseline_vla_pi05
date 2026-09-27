from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"pack manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "project_2026_pi_style_training_pack_v0":
        raise ValueError(f"unexpected pack schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def resolve_image_root(pack_dir: Path, manifest: dict[str, Any]) -> Path:
    image_root = Path(str(manifest.get("output", {}).get("image_root", "")))
    if image_root.exists():
        return image_root
    candidate = pack_dir / image_root
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"image root not found: {image_root}")


def normalize_stats(values: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def state_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            arrays["elite_tcp_pose_6d"].astype(np.float32),
            arrays["piper_state"].astype(np.float32),
            arrays["tactile_context"].astype(np.float32),
            arrays["tactile_context_valid"].astype(np.float32),
            arrays["task_one_hot"].astype(np.float32),
        ],
        axis=1,
    )


class PiStylePackDataset(Dataset):
    def __init__(
        self,
        *,
        index_rows: list[dict[str, Any]],
        arrays: dict[str, np.ndarray],
        image_root: Path,
        indices: list[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        elite_mean: np.ndarray,
        elite_std: np.ndarray,
        image_size: int,
    ):
        self.index_rows = index_rows
        self.arrays = arrays
        self.image_root = image_root
        self.indices = indices
        self.state_values = state_array(arrays)
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.elite_mean = elite_mean.astype(np.float32)
        self.elite_std = elite_std.astype(np.float32)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.image_root / rel_path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        row = self.index_rows[idx]
        side = self._load_image(row["images"]["side"]["path"])
        top = self._load_image(row["images"]["top"]["path"])
        state = (self.state_values[idx] - self.state_mean) / self.state_std
        elite = (self.arrays["elite_tcp_delta_6d"][idx].astype(np.float32) - self.elite_mean) / self.elite_std
        piper = int(self.arrays["piper_intent_id"][idx])
        return {
            "side": side,
            "top": top,
            "state": torch.from_numpy(state.astype(np.float32)),
            "elite": torch.from_numpy(elite.astype(np.float32)),
            "piper": torch.tensor(piper, dtype=torch.long),
        }


class SmallImageEncoder(nn.Module):
    def __init__(self):
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
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class PiStyleSmokePolicy(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.side_encoder = SmallImageEncoder()
        self.top_encoder = SmallImageEncoder()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.elite_head = nn.Linear(128, 6)
        self.piper_head = nn.Linear(128, 3)

    def forward(self, side: torch.Tensor, top: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([self.side_encoder(side), self.top_encoder(top), self.state_encoder(state)], dim=1)
        hidden = self.trunk(feat)
        return self.elite_head(hidden), self.piper_head(hidden)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    elite_mean: np.ndarray,
    elite_std: np.ndarray,
    elite_loss_weight: float,
    piper_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total = total_elite = total_piper = total_acc = total_mae = 0.0
    count = 0
    mean = torch.as_tensor(elite_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(elite_std, dtype=torch.float32, device=device)
    for batch in loader:
        side = batch["side"].to(device)
        top = batch["top"].to(device)
        state = batch["state"].to(device)
        elite = batch["elite"].to(device)
        piper = batch["piper"].to(device)
        elite_pred, piper_logits = model(side, top, state)
        elite_loss = F.smooth_l1_loss(elite_pred, elite)
        piper_loss = F.cross_entropy(piper_logits, piper)
        loss = elite_loss_weight * elite_loss + piper_loss_weight * piper_loss
        bs = len(piper)
        total += float(loss.item()) * bs
        total_elite += float(elite_loss.item()) * bs
        total_piper += float(piper_loss.item()) * bs
        total_acc += float((piper_logits.argmax(dim=1) == piper).float().mean().item()) * bs
        total_mae += float(torch.mean(torch.abs((elite_pred * std + mean) - (elite * std + mean))).item()) * bs
        count += bs
    denom = max(count, 1)
    return {
        "loss": total / denom,
        "elite_loss": total_elite / denom,
        "piper_loss": total_piper / denom,
        "piper_acc": total_acc / denom,
        "elite_raw_mae": total_mae / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small mixed-head smoke model from a pi-style training pack.")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--elite-loss-weight", type=float, default=1.0)
    parser.add_argument("--piper-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    pack_dir, pack_manifest = resolve_pack(args.pack)
    image_root = resolve_image_root(pack_dir, pack_manifest)
    index_rows = read_jsonl(pack_dir / str(pack_manifest["output"]["index_jsonl"]))
    arrays_npz = np.load(pack_dir / str(pack_manifest["output"]["arrays_npz"]))
    arrays = {name: arrays_npz[name] for name in arrays_npz.files}

    selected = list(range(len(index_rows)))
    if args.max_records is not None:
        selected = selected[: max(int(args.max_records), 0)]
    train_indices = [idx for idx in selected if index_rows[idx].get("split") == "train"]
    val_indices = [idx for idx in selected if index_rows[idx].get("split") == "val"]
    if not train_indices or not val_indices:
        raise ValueError(f"need non-empty train and val splits, got train={len(train_indices)} val={len(val_indices)}")

    states = state_array(arrays)
    state_mean, state_std = normalize_stats(states[train_indices])
    elite_mean, elite_std = normalize_stats(arrays["elite_tcp_delta_6d"][train_indices].astype(np.float32))

    train_set = PiStylePackDataset(
        index_rows=index_rows,
        arrays=arrays,
        image_root=image_root,
        indices=train_indices,
        state_mean=state_mean,
        state_std=state_std,
        elite_mean=elite_mean,
        elite_std=elite_std,
        image_size=args.image_size,
    )
    val_set = PiStylePackDataset(
        index_rows=index_rows,
        arrays=arrays,
        image_root=image_root,
        indices=val_indices,
        state_mean=state_mean,
        state_std=state_std,
        elite_mean=elite_mean,
        elite_std=elite_std,
        image_size=args.image_size,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = PiStyleSmokePolicy(state_dim=states.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "pack": str(args.pack),
        "image_root": str(image_root),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "state_dim": int(states.shape[1]),
        "epochs": [],
        "normalization": {
            "state_mean": state_mean.astype(float).tolist(),
            "state_std": state_std.astype(float).tolist(),
            "elite_mean": elite_mean.astype(float).tolist(),
            "elite_std": elite_std.astype(float).tolist(),
        },
    }

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_elite = total_piper = total_acc = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            elite = batch["elite"].to(device)
            piper = batch["piper"].to(device)
            elite_pred, piper_logits = model(side, top, state)
            elite_loss = F.smooth_l1_loss(elite_pred, elite)
            piper_loss = F.cross_entropy(piper_logits, piper)
            loss = args.elite_loss_weight * elite_loss + args.piper_loss_weight * piper_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = len(piper)
            total += float(loss.item()) * bs
            total_elite += float(elite_loss.item()) * bs
            total_piper += float(piper_loss.item()) * bs
            total_acc += float((piper_logits.argmax(dim=1) == piper).float().mean().item()) * bs
            count += bs

        train = {
            "loss": total / max(count, 1),
            "elite_loss": total_elite / max(count, 1),
            "piper_loss": total_piper / max(count, 1),
            "piper_acc": total_acc / max(count, 1),
        }
        val = evaluate(
            model,
            val_loader,
            device,
            elite_mean,
            elite_std,
            elite_loss_weight=args.elite_loss_weight,
            piper_loss_weight=args.piper_loss_weight,
        )
        row = {"epoch": epoch, "train": train, "val": val}
        log["epochs"].append(row)
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
            f"train_piper_acc={train['piper_acc']:.3f} val_piper_acc={val['piper_acc']:.3f} "
            f"val_elite_mae={val['elite_raw_mae']:.4f}"
        )
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": int(states.shape[1]),
                    "normalization": log["normalization"],
                    "pack_manifest": pack_manifest,
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
