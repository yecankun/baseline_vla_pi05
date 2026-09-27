import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


STATE_KEYS = [
    "tip_pos",
    "heading",
    "target_pos",
    "contact_normal",
]


def build_state_vector(sample):
    state = sample["state"]
    values = []
    for key in STATE_KEYS:
        values.extend(state[key])
    values.append(float(state["contact_flag"]))
    values.append(float(state["contact_strength"]))
    values.append(float(sample["task"] == "left"))
    values.append(float(sample["task"] == "right"))
    return np.asarray(values, dtype=np.float32)


class GuidewireDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 224, strict_files: bool = False):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        samples = [s for s in manifest["samples"] if "side" in s["images"] and "top" in s["images"]]
        self.samples = []
        missing = 0
        for sample in samples:
            side_path = self._resolve_path(sample["images"]["side"])
            top_path = self._resolve_path(sample["images"]["top"])
            if side_path.exists() and top_path.exists():
                self.samples.append(sample)
            else:
                missing += 1
                if strict_files:
                    missing_path = side_path if not side_path.exists() else top_path
                    raise FileNotFoundError(missing_path)
        if missing:
            print(f"Warning: skipped {missing} samples with missing side/top images")
        if not self.samples:
            raise RuntimeError("No camera_pair samples found in manifest")
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _resolve_path(path_str):
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path_str):
        path = self._resolve_path(path_str)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        side = self._read_image(sample["images"]["side"])
        top = self._read_image(sample["images"]["top"])
        state = torch.tensor(build_state_vector(sample), dtype=torch.float32)
        action = torch.tensor(sample["action"][:3], dtype=torch.float32)
        return {"side": side, "top": top, "state": state, "action": action}


class DualCameraStatePolicy(nn.Module):
    def __init__(self, state_dim: int, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        side_resnet = models.resnet18(weights=weights)
        top_resnet = models.resnet18(weights=weights)
        self.side_encoder = nn.Sequential(*list(side_resnet.children())[:-1])
        self.top_encoder = nn.Sequential(*list(top_resnet.children())[:-1])
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, side, top, state):
        side_feat = self.side_encoder(side).flatten(1)
        top_feat = self.top_encoder(top).flatten(1)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([side_feat, top_feat, state_feat], dim=1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            target = batch["action"].to(device)
            pred = model(side, top, state)
            loss = criterion(pred, target)
            total_loss += float(loss.item()) * len(target)
            total_count += len(target)
    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a simple dual-camera + state baseline policy.")
    parser.add_argument("--manifest", default="simulation_output/vla_dataset_v1/manifest.json")
    parser.add_argument("--out", default="simulation_output/baseline_dual_camera_state")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--strict-files", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GuidewireDataset(args.manifest, image_size=args.image_size, strict_files=args.strict_files)
    state_dim = int(build_state_vector(dataset.samples[0]).shape[0])
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DualCameraStatePolicy(state_dim=state_dim, pretrained=args.pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "manifest": args.manifest,
        "device": str(device),
        "samples": len(dataset),
        "train_samples": train_len,
        "val_samples": val_len,
        "epochs": [],
    }
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            target = batch["action"].to(device)
            pred = model(side, top, state)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(target)
            train_count += len(target)

        train_loss /= max(train_count, 1)
        val_loss = evaluate(model, val_loader, device, criterion)
        log["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "image_size": args.image_size,
                    "val_loss": best_val,
                },
                out_dir / "best_model.pt",
            )

    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved best model and log to {out_dir}")


if __name__ == "__main__":
    main()
