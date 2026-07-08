"""
相位轨迹可视化实验 — 训练脚本。

训练 SPRiF 和 ASRNN 在合成相位轨迹任务上，
损失函数 = MSE(delay 阶段) + beta * firing_rate。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config import (
    FIGURE_DIR, CHECKPOINT_DIR,
    T_CUE, BETA, LR, EPOCHS, BATCH_SIZE,
    TRAIN_N, VAL_N, GRAD_CLIP,
    get_args,
)
from generate_data import generate_dataset, PhaseTrajectoryDataset
from models import SPRiFTrajectoryNet, ASRNNTrajectoryNet


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """训练一个 epoch，返回 (avg_loss, avg_mse, avg_spike_rate)。"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_spike_rate = 0.0
    n_batches = 0

    for x, probe_mask, target in loader:
        x = x.to(device)
        probe_mask = probe_mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        readout_seq, spike_rate = model(x, probe_mask)

        # MSE loss 仅在 delay 阶段 (t >= T_CUE)
        readout_delay = readout_seq[:, T_CUE:, :]
        target_delay = target[:, T_CUE:, :]
        mse = criterion(readout_delay, target_delay)

        # 总损失 = MSE + beta * spike_rate
        loss = mse + BETA * spike_rate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_spike_rate += spike_rate.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_mse / n_batches,
        total_spike_rate / n_batches,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """验证，返回 (avg_loss, avg_mse, avg_spike_rate)。"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_spike_rate = 0.0
    n_batches = 0

    for x, probe_mask, target in loader:
        x = x.to(device)
        probe_mask = probe_mask.to(device)
        target = target.to(device)

        readout_seq, spike_rate = model(x, probe_mask)

        readout_delay = readout_seq[:, T_CUE:, :]
        target_delay = target[:, T_CUE:, :]
        mse = criterion(readout_delay, target_delay)
        loss = mse + BETA * spike_rate

        total_loss += loss.item()
        total_mse += mse.item()
        total_spike_rate += spike_rate.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_mse / n_batches,
        total_spike_rate / n_batches,
    )


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> str:
    """训练单个模型，返回最佳 checkpoint 路径。"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_checkpoint = None
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mse, train_sr = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_mse, val_sr = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} MSE: {train_mse:.4f} SR: {train_sr:.4f} | "
            f"Val Loss: {val_loss:.4f} MSE: {val_mse:.4f} SR: {val_sr:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"TrajectoryViz_{model_name}_mse{best_val_mse:.4f}.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint = checkpoint_path
            print(f"  -> Saved best model: {os.path.basename(checkpoint_path)}")

    print(f"\n{model_name} training complete. Best Val MSE: {best_val_mse:.4f}")
    return best_checkpoint


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Figure output: {FIGURE_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    # 生成数据
    print("\nGenerating training data...")
    train_samples = generate_dataset(TRAIN_N, jitter=True, seed=args.seed)
    val_samples = generate_dataset(VAL_N, jitter=False, seed=args.seed + 1)
    train_loader = DataLoader(
        PhaseTrajectoryDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PhaseTrajectoryDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"  Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    # 训练 SPRiF
    if args.model in ["sprif", "both"]:
        sprif_model = SPRiFTrajectoryNet(hidden_size=args.hidden_size)
        train_model("SPRiF", sprif_model, train_loader, val_loader, args, device)

    # 训练 ASRNN
    if args.model in ["asrnn", "both"]:
        asrnn_model = ASRNNTrajectoryNet(hidden_size=args.hidden_size)
        train_model("ASRNN", asrnn_model, train_loader, val_loader, args, device)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
