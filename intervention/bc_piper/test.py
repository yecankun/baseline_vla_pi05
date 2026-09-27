import time
import torch
import torch.nn as nn
import torch.optim as optim

from intervention.bc_piper.dataset import PiperDataset
from intervention.bc_piper.model import SingleFramePiperPolicyModel
# (dataset import handled later when using SimulatedSurgicalDataset)
import argparse
from torch.utils.tensorboard import SummaryWriter

from intervention.bc_piper.train import evaluate_fixed_threshold, find_best_threshold


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

    model_root_dir = 'model_state'
    model.load_state_dict(torch.load(f'{model_root_dir}/piper_policy_epoch_20.pth', map_location=args.device))

    metrics = evaluate_fixed_threshold(model, val_loader, args.device)

    for th, (p, r, f1) in metrics.items():
        print(f"[Fixed th={th}] P={p:.3f} R={r:.3f} F1={f1:.3f}")

    best_th = find_best_threshold(model, val_loader, args.device)
    print(f"Best threshold on val: th={best_th['th']:.2f} | "
          f"F1={best_th['f1']:.3f} P={best_th['precision']:.3f} R={best_th['recall']:.3f}")


if __name__ == '__main__':
    main()