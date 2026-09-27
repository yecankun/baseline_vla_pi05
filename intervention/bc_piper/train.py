# train_temporal_multimodal.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

from intervention.bc_piper.dataset import PiperDataset
from intervention.bc_piper.model import SingleFramePiperPolicyModel
# (dataset import handled later when using SimulatedSurgicalDataset)
import argparse
from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.enabled = False
# python

def compute_loss(small_logits,  small_gt):
    """
    大臂使用 MSE（回归），小臂使用 BCEWithLogits（分类）
    返回： total_loss, loss_large, loss_small
    """

    # 小臂：保证 logits 和 target 形状一致并为浮点
    small_logits = small_logits.view(-1)
    small_gt = small_gt.view(-1).float()
    loss_small = nn.BCEWithLogitsLoss()(small_logits, small_gt)
    return loss_small

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch



@torch.no_grad()
def evaluate_fixed_threshold(model, dataloader, device, thresholds=[0.3, 0.5, 0.8, 0.9]):
    model.eval()

    all_probs = []
    all_gts = []

    for batch in dataloader:
        image_seq = batch['image'].to(device)
        pose_seq = batch['pose'].to(device)
        count_seq = batch['count'].to(device)
        task_id = batch['task_id'].to(device)
        small_gt = batch['small_action'].to(device)

        logits = model(image_seq, pose_seq, count_seq, task_id)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_gts.append(small_gt.cpu())

    probs = torch.cat(all_probs)
    gts = torch.cat(all_gts)

    results = {}

    for th in thresholds:
        preds = (probs >= th).long()

        TP = ((preds == 1) & (gts == 1)).sum().item()
        FP = ((preds == 1) & (gts == 0)).sum().item()
        FN = ((preds == 0) & (gts == 1)).sum().item()

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)

        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        results[th] = (precision, recall, f1)

    return results

#你现在 deterministic 是 small_logits > 0（阈值=0.5）。
#我们改成：在验证集上扫描概率阈值 p_th，找到最优。
@torch.no_grad()
def find_best_threshold(model, dataloader, device, task_fixed=None):
    model.eval()
    all_probs = []
    all_gts = []

    for batch in dataloader:
        image_seq = batch['image'].to(device)
        pose_seq  = batch['pose'].to(device)
        count_seq = batch['count'].to(device)
        task_id   = batch['task_id'].to(device)
        gt        = batch['small_action'].to(device).view(-1).long()

        logits = model(image_seq, pose_seq, count_seq, task_id)
        probs  = torch.sigmoid(logits).view(-1)

        all_probs.append(probs.cpu())
        all_gts.append(gt.cpu())

    probs = torch.cat(all_probs)
    gts   = torch.cat(all_gts)

    # 扫描阈值
    best = {"th": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    thresholds = torch.linspace(0.01, 0.99, steps=99)

    for th in thresholds:
        pred = (probs >= th).long()
        tp = ((pred == 1) & (gts == 1)).sum().item()
        fp = ((pred == 1) & (gts == 0)).sum().item()
        fn = ((pred == 0) & (gts == 1)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best["f1"]:
            best = {"th": float(th.item()), "f1": float(f1),
                    "precision": float(precision), "recall": float(recall)}

    return best

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        image_seq = batch['image'].to(device)
        pose_seq  = batch['pose'].to(device)
        count_seq = batch['count'].to(device)
        task_id   = batch['task_id'].to(device)
        small_gt  = batch['small_action'].to(device)

        optimizer.zero_grad()
        small_pred = model(image_seq, pose_seq, count_seq, task_id)
        loss = compute_loss(small_pred, small_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        image_seq = batch['image'].to(device)
        pose_seq  = batch['pose'].to(device)
        count_seq = batch['count'].to(device)
        task_id   = batch['task_id'].to(device)
        small_gt  = batch['small_action'].to(device)

        small_pred = model(image_seq, pose_seq, count_seq, task_id)
        loss = compute_loss(small_pred, small_gt)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # ===== 加载数据 =====
    # TODO: 替换为您的真实专家演示数据（必须包含5帧历史）
    # 示例格式：每个样本是一个字典，含 'image_seq', 'pose_seq', 'count_seq', 'task_id', 'large_action', 'small_action'

    train_dataset = PiperDataset(data_type='train')
    val_dataset = PiperDataset(data_type='test')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ===== 初始化模型 =====
    model = SingleFramePiperPolicyModel(
        d_model=128,
        nhead=8,
        num_transformer_layers=1,
        num_tasks=args.num_tasks,
        dropout=0.1
    ).to(args.device)
    # 3_3: 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)


    best_val_loss = float('inf')
    writer = SummaryWriter("././curves/2026_3_4")
    model_root_dir = 'model_state'

    # ===== 训练循环 =====
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate_epoch(model, val_loader, args.device)

        metrics = evaluate_fixed_threshold(model, val_loader, args.device)

        for th, (p, r, f1) in metrics.items():
            print(f"[Fixed th={th}] P={p:.3f} R={r:.3f} F1={f1:.3f}")
            
        best_th = find_best_threshold(model, val_loader, args.device)
        print(f"Best threshold on val: th={best_th['th']:.2f} | "
              f"F1={best_th['f1']:.3f} P={best_th['precision']:.3f} R={best_th['recall']:.3f}")
        writer.add_scalar("val_best_th", best_th["th"], epoch)
        writer.add_scalar("val_f1_best_th", best_th["f1"], epoch)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f}  | "
              f"Val Loss: {val_loss:.6f}  | "
              f"Time: {epoch_time:.2f}s")

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("epoch_time_seconds", epoch_time, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_piper_policy.pth')
            print(f"✅ New best model saved (val loss: {val_loss:.6f})")
        # 每 10 个 epoch 保存一次模型（无论是否是最佳模型），以便后续分析训练过程中的模型演变
        if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'{model_root_dir}/piper_policy_epoch_{epoch + 1}.pth')
                print(f"Checkpoint saved at epoch {epoch + 1}")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_piper_policy.pth')
    print("Training completed. Final model saved.")


if __name__ == '__main__':
    main()
    ##试试增加数据集 去掉lstm