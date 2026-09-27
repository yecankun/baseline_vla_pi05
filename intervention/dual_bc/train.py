# train_temporal_multimodal.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from torch.utils.tensorboard import SummaryWriter

from intervention.dual_bc.dataset import DualDataset
from intervention.dual_bc.model import SingleFrameDualPolicyModel


# torch.backends.cudnn.enabled = False
# python
def compute_loss(large_pred, small_logits, large_gt, small_gt, w_large=0.7762, w_small=1.2238):
    """
    大臂使用 MSE（回归），小臂使用 BCEWithLogits（分类）
    返回： total_loss, loss_large, loss_small
    """
    # 大臂：MSE
    # large_pred: [B, 6], large_gt: [B, 6]
    loss_large = nn.MSELoss()(large_pred, large_gt)

    # 小臂：保证 logits 和 target 形状一致并为浮点
    small_logits = small_logits.view(-1)
    small_gt = small_gt.view(-1).float()
    loss_small = nn.BCEWithLogitsLoss()(small_logits, small_gt)

    total_loss = w_large * loss_large + w_small * loss_small
    return total_loss, loss_large, loss_small

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

        large_action, logits = model(image_seq, pose_seq, count_seq, task_id)
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

        large_action, logits = model(image_seq, pose_seq, count_seq, task_id)
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
    total_loss = total_large = total_small = 0.0
    n_batches = 0

    for batch in dataloader:
        # [B, T, ...] 序列输入
        image_seq = batch['image'].to(device)       # [B, T, 3, 224, 224]
        pose_seq = batch['pose'].to(device)         # [B, T, 6]
        count_seq = batch['count'].to(device)       # [B, T]
        task_id = batch['task_id'].to(device)           # [B]

        large_gt = batch['large_action'].to(device)     # [B, 6]
        small_gt = batch['small_action'].to(device)     # [B]

        optimizer.zero_grad()

        #debug_tensors([image_seq, pose_seq, count_seq, task_id],names=["image_seq", "pose_seq", "count_seq", "seq_feat"])
        large_pred, small_logits = model(image_seq, pose_seq, count_seq, task_id)

        loss, loss_l, loss_s = compute_loss(
            large_pred, small_logits,
            large_gt, small_gt
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_large += loss_l.item()
        total_small += loss_s.item()
        n_batches += 1

    return total_loss / n_batches, total_large / n_batches, total_small / n_batches


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = total_large = total_small = 0.0
    n_batches = 0

    for batch in dataloader:
        image_seq = batch['image'].to(device)
        pose_seq = batch['pose'].to(device)
        count_seq = batch['count'].to(device)
        task_id = batch['task_id'].to(device)
        large_gt = batch['large_action'].to(device)
        small_gt = batch['small_action'].to(device)

        large_pred, small_logits = model(image_seq, pose_seq, count_seq, task_id)
        loss, loss_l, loss_s = compute_loss(
            large_pred, small_logits,
            large_gt, small_gt
        )

        total_loss += loss.item()
        total_large += loss_l.item()
        total_small += loss_s.item()
        n_batches += 1

    return total_loss / n_batches, total_large / n_batches, total_small / n_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # ===== 加载数据 =====
    # TODO: 替换为您的真实专家演示数据（必须包含5帧历史）
    # 示例格式：每个样本是一个字典，含 'image_seq', 'pose_seq', 'count_seq', 'task_id', 'large_action', 'small_action'

    train_dataset = DualDataset(data_type='train')
    val_dataset = DualDataset(data_type='test')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ===== 初始化模型 =====
    model = SingleFrameDualPolicyModel(
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

    # w_l, w_s = estimate_static_weights(model, train_loader, args.device, n_batches=20)
    # print(f"Using static weights: w_large={w_l:.4f}, w_small={w_s:.4f}")

    # ===== 训练循环 =====
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_l, train_s = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_l, val_s = validate_epoch(model, val_loader, args.device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} (L: {train_l:.6f}, S: {train_s:.6f}) | "
              f"Val Loss: {val_loss:.6f} (L: {val_l:.6f}, S: {val_s:.6f}) | "
              f"Time: {epoch_time:.2f}s")

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("train_large_loss", train_l, epoch)
        writer.add_scalar("train_small_loss", train_s, epoch)
        writer.add_scalar("val_large_loss", val_l, epoch)
        writer.add_scalar("val_small_loss", val_s, epoch)

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

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_dual_policy.pth')
            print(f"✅ New best model saved (val loss: {val_loss:.6f})")
        # 每 10 个 epoch 保存一次模型（无论是否是最佳模型），以便后续分析训练过程中的模型演变
        if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'{model_root_dir}/dual_policy_epoch_{epoch + 1}.pth')
                print(f"Checkpoint saved at epoch {epoch + 1}")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_dual_policy.pth')
    print("Training completed. Final model saved.")

def estimate_static_weights(model, dataloader, device, n_batches=20, eps=1e-8):
    """
    在训练前用前 n_batches 批次估计 loss_large 和 loss_small 的平均量级，
    然后返回静态权重 (w_large, w_small) 使两项初始贡献相近：
        w_large * L_mean  ~=  w_small * S_mean
    我们常把 w_large = 1.0，w_small = L_mean / (S_mean + eps)
    """
    model.eval()
    total_L = 0.0
    total_S = 0.0
    seen = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            image_seq = batch['image'].to(device)
            pose_seq = batch['pose'].to(device)
            count_seq = batch['count'].to(device)
            task_id = batch['task_id'].to(device)
            large_gt = batch['large_action'].to(device)
            small_gt = batch['small_action'].to(device)

            # model now returns (large_pred, small_logits)
            large_pred, small_logits = model(image_seq, pose_seq, count_seq, task_id)
            # 复用 compute_loss（MSE + BCE）并取分量
            _, loss_l, loss_s = compute_loss(large_pred, small_logits, large_gt, small_gt, w_large=1.0, w_small=1.0)
            total_L += float(loss_l.item())
            total_S += float(loss_s.item())
            seen += 1

    if seen == 0:
        return 1.0, 1.0

    L_mean = total_L / seen
    S_mean = total_S / seen

    # 如果 S_mean 非常小，避免除以零并限制权重范围
    w_large = 1.0
    w_small = float(L_mean / (S_mean + eps))

    # 可选：限制权重在合理范围，避免极端值
    w_small = max(1e-3, min(1e3, w_small))

    # 可选：归一化使 w_large + w_small = 2.0（保持整体规模可控）
    s = w_large + w_small
    w_large /= s / 2.0
    w_small /= s / 2.0

    return w_large, w_small


if __name__ == '__main__':
    main()
    ##试试增加数据集 去掉lstm