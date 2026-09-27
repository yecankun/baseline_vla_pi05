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


EXPECTED_SCHEMA = "project_2026_openpi_compat_pack_v0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"OpenPI-compatible pack manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected pack schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def resolve_ref(base: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    direct = base / path
    if direct.exists():
        return direct
    return path


def normalize_stats(values: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def apply_state_ablation(state: np.ndarray, mode: str) -> np.ndarray:
    ablated = state.astype(np.float32, copy=True)
    if mode == "full":
        return ablated
    if mode == "drop_tactile_values":
        ablated[:, 8:11] = 0.0
        return ablated
    if mode == "drop_tactile_all":
        ablated[:, 8:14] = 0.0
        return ablated
    raise ValueError(f"unknown tactile ablation mode: {mode!r}")


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    weighted = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return weighted.sum() / denom


class OpenPICompatDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        arrays: dict[str, np.ndarray],
        state_values: np.ndarray,
        image_root: Path,
        indices: list[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        image_size: int,
    ):
        self.rows = rows
        self.arrays = arrays
        self.state_values = state_values.astype(np.float32)
        self.image_root = image_root
        self.indices = indices
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)
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
        row = self.rows[idx]
        side = self._load_image(str(row["observation.images.side"]))
        top = self._load_image(str(row["observation.images.top"]))
        state = (self.state_values[idx].astype(np.float32) - self.state_mean) / self.state_std
        action = (self.arrays["action_32"][idx].astype(np.float32) - self.action_mean) / self.action_std
        return {
            "side": side,
            "top": top,
            "state": torch.from_numpy(state.astype(np.float32)),
            "state_mask": torch.from_numpy(self.arrays["state_mask_32"][idx].astype(np.bool_)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "action_mask": torch.from_numpy(self.arrays["action_mask_32"][idx].astype(np.bool_)),
            "elite": torch.from_numpy(self.arrays["elite_tcp_delta_6d"][idx].astype(np.float32)),
            "piper": torch.tensor(int(self.arrays["piper_intent_id"][idx]), dtype=torch.long),
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


class OpenPICompatSmokePolicy(nn.Module):
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
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_loss_weight: float,
    piper_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total = total_action = total_piper = total_acc = total_elite_mae = total_action_mae = 0.0
    count = 0
    mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    for batch in loader:
        side = batch["side"].to(device)
        top = batch["top"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        action_mask = batch["action_mask"].to(device)
        piper = batch["piper"].to(device)
        raw_elite = batch["elite"].to(device)
        pred_action, piper_logits = model(side, top, state)
        action_loss = masked_smooth_l1(pred_action, action, action_mask)
        piper_loss = F.cross_entropy(piper_logits, piper)
        loss = action_loss_weight * action_loss + piper_loss_weight * piper_loss
        pred_raw = pred_action * std + mean
        true_raw = action * std + mean
        mask_float = action_mask.float()
        action_mae = (torch.abs(pred_raw - true_raw) * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        elite_mae = torch.mean(torch.abs(pred_raw[:, :6] - raw_elite))
        bs = len(piper)
        total += float(loss.item()) * bs
        total_action += float(action_loss.item()) * bs
        total_piper += float(piper_loss.item()) * bs
        total_acc += float((piper_logits.argmax(dim=1) == piper).float().mean().item()) * bs
        total_action_mae += float(action_mae.item()) * bs
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
    parser = argparse.ArgumentParser(
        description="Train a small OpenPI-compatible mixed-head smoke model from a project_2026 compat pack."
    )
    parser.add_argument("pack", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--piper-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--tactile-ablation",
        choices=["full", "drop_tactile_values", "drop_tactile_all"],
        default="full",
        help=(
            "Ablate tactile/contact context in state_32 before normalization. "
            "drop_tactile_values zeros dims 8:11; drop_tactile_all zeros dims 8:14."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    pack_dir, manifest = resolve_pack(args.pack)
    arrays_path = pack_dir / str(manifest["output"]["arrays_npz"])
    index_path = pack_dir / str(manifest["output"]["index_jsonl"])
    image_root = resolve_ref(pack_dir, str(manifest["output"]["image_root"]))
    arrays_npz = np.load(arrays_path)
    arrays = {name: arrays_npz[name] for name in arrays_npz.files}
    rows = read_jsonl(index_path)

    selected = list(range(len(rows)))
    if args.max_records is not None:
        selected = selected[: max(int(args.max_records), 0)]
    train_indices = [idx for idx in selected if rows[idx].get("split") == "train"]
    val_indices = [idx for idx in selected if rows[idx].get("split") == "val"]
    if not train_indices or not val_indices:
        raise ValueError(f"need non-empty train and val splits, got train={len(train_indices)} val={len(val_indices)}")

    state_values = apply_state_ablation(arrays["state_32"], args.tactile_ablation)
    state_mean, state_std = normalize_stats(state_values[train_indices].astype(np.float32))
    action_mean, action_std = normalize_stats(arrays["action_32"][train_indices].astype(np.float32))

    train_set = OpenPICompatDataset(
        rows=rows,
        arrays=arrays,
        state_values=state_values,
        image_root=image_root,
        indices=train_indices,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        image_size=args.image_size,
    )
    val_set = OpenPICompatDataset(
        rows=rows,
        arrays=arrays,
        state_values=state_values,
        image_root=image_root,
        indices=val_indices,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        image_size=args.image_size,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = OpenPICompatSmokePolicy(
        state_dim=int(arrays["state_32"].shape[1]),
        action_dim=int(arrays["action_32"].shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "pack": str(args.pack),
        "image_root": str(image_root),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "state_dim": int(arrays["state_32"].shape[1]),
        "action_dim": int(arrays["action_32"].shape[1]),
        "tactile_ablation": args.tactile_ablation,
        "tactile_dims": {
            "values": [8, 11],
            "validity": [11, 14],
        },
        "semantics": manifest.get("openpi_compat", {}).get("important_semantics", []),
        "epochs": [],
        "normalization": {
            "state_mean": state_mean.astype(float).tolist(),
            "state_std": state_std.astype(float).tolist(),
            "action_mean": action_mean.astype(float).tolist(),
            "action_std": action_std.astype(float).tolist(),
        },
    }

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_action = total_piper = total_acc = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            action_mask = batch["action_mask"].to(device)
            piper = batch["piper"].to(device)
            pred_action, piper_logits = model(side, top, state)
            action_loss = masked_smooth_l1(pred_action, action, action_mask)
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
                    "state_dim": int(arrays["state_32"].shape[1]),
                    "action_dim": int(arrays["action_32"].shape[1]),
                    "normalization": log["normalization"],
                    "pack_manifest": manifest,
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
