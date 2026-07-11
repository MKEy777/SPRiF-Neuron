"""
SPRiF-SHD: Spiking Heidelberg Digits classification with SPRiF neuron networks.

Usage:
    python train.py --train-dir /path/to/train_1ms --test-dir /path/to/test_1ms
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from core_algorithm.utils import set_seed
from core_algorithm.sprif_layer import SPRiFNeuronLayer
from model import SPRiFSHDNet
from data import SHDDataset

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="SPRiF-SHD: Spiking Heidelberg Digits")
    # Training
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    # Model
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--mode", type=str, default="srnn")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--input-size", type=int, default=700)
    parser.add_argument("--warmup-steps", type=int, default=10)
    # Neuron
    parser.add_argument("--neuron-threshold", type=float, default=1.0)
    parser.add_argument("--neuron-init-std", type=float, default=0.2)
    # Data
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    # Scheduler
    parser.add_argument("--scheduler-step", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_dataset = SHDDataset(args.train_dir, train=True)
    test_dataset = SHDDataset(args.test_dir, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Model
    recurrent_flags = tuple(args.mode.lower() == "srnn" for _ in args.hidden_sizes)
    model = SPRiFSHDNet(
        input_size=args.input_size,
        hidden_sizes=list(args.hidden_sizes),
        num_classes=args.num_classes,
        dropout=args.dropout,
        recurrent_flags=recurrent_flags,
        warmup_steps=args.warmup_steps,
        neuron_kwargs={
            "threshold": args.neuron_threshold,
            "init_std": args.neuron_init_std,
        },
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0.0
    best_ckpt_path = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=-1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item() * x.size(0)
                test_correct += (logits.argmax(dim=-1) == y).sum().item()
                test_total += x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_loss /= max(test_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            hs_str = "hs" + "".join(str(h) for h in args.hidden_sizes)
            save_name = (
                f"SPRiFSHDNet_{hs_str}_bs{args.batch_size}"
                f"_lr{args.lr}_seed{args.seed}_acc{best_test_acc:.2f}.pth"
            )
            if best_ckpt_path is not None and best_ckpt_path != save_name and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                except OSError:
                    pass
            torch.save(model.state_dict(), save_name)
            best_ckpt_path = save_name
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  -> Early stopping triggered. Best Test Acc: {best_test_acc:.2f}%")
                break

    print(f"\nTraining complete. Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
