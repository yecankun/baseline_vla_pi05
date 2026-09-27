# train_temporal_multimodal.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from torch.utils.tensorboard import SummaryWriter

from intervention.bc_elirobot.dataset import DualDataset
from intervention.bc_elirobot.model import SingleFrameElirobotPolicyModel


# torch.backends.cudnn.enabled = False
# python
def compute_loss(large_pred, large_gt):
    """
    大臂使用 MSE（回归），小臂使用 BCEWithLogits（分类）
    返回： total_loss, loss_large, loss_small
    """
    # 大臂：MSE
    # large_pred: [B, 6], large_gt: [B, 6]
    loss_large = nn.MSELoss()(large_pred, large_gt)

    return  loss_large

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch


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
        large_pred = model(image_seq, pose_seq, count_seq, task_id)

        loss = compute_loss(
            large_pred,
            large_gt
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


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

        large_pred = model(image_seq, pose_seq, count_seq, task_id)
        loss = compute_loss(
            large_pred,
            large_gt
        )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
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
    model = SingleFrameElirobotPolicyModel(
        d_model=128,
        nhead=8,
        num_transformer_layers=1,
        num_tasks=args.num_tasks,
        dropout=0.1
    ).to(args.device)
    # 3_3: 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    writer = SummaryWriter("././left_curves/2026_3_6")
    model_root_dir = 'left_model_state'


    # ===== 训练循环 =====
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate_epoch(model, val_loader, args.device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_root_dir}/best_elirobot_policy.pth')
            print(f"✅ New best model saved (val loss: {val_loss:.6f})")
        # 每 10 个 epoch 保存一次模型（无论是否是最佳模型），以便后续分析训练过程中的模型演变
        if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'{model_root_dir}/elirobot_policy_epoch_{epoch + 1}.pth')
                print(f"Checkpoint saved at epoch {epoch + 1}")

    # 保存最终模型
    torch.save(model.state_dict(), f'{model_root_dir}/final_elirobot_policy.pth')
    print("Training completed. Final model saved.")


if __name__ == '__main__':
    main()
    # left best在197
