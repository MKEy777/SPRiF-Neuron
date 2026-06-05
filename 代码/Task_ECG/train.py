"""
SPRiF-ECG: QTDB ECG classification with SPRiF neuron networks.

Usage:
    python train.py --train-mat ./data/QTDB_train.mat --test-mat ./data/QTDB_test.mat
"""

import argparse

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from core_algorithm.utils import set_seed, convert_dataset_wtime
from core_algorithm.sprif_layer import SPRiFNeuronLayer
from model import SPRiFECGModel


def get_args():
    parser = argparse.ArgumentParser(description="SPRiF-ECG: QTDB ECG Classification")
    # Training
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1111)
    # Model
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[36])
    parser.add_argument("--out-size", type=int, default=6)
    parser.add_argument("--mode", type=str, default="srnn")
    # Neuron
    parser.add_argument("--neuron-threshold", type=float, default=1.0)
    parser.add_argument("--neuron-init-std", type=float, default=0.05)
    parser.add_argument("--neuron-bias", action="store_true", default=True)
    # Data
    parser.add_argument("--train-mat", type=str, required=True)
    parser.add_argument("--test-mat", type=str, required=True)
    # Scheduler
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.75)
    return parser.parse_args()


def compute_loss_and_correct(logits, labels, criterion):
    l_task = criterion(logits.permute(0, 2, 1).reshape(-1, logits.size(1)), labels.reshape(-1))
    loss = l_task * logits.size(2)
    pred = logits.argmax(dim=1)
    correct = pred.eq(labels).sum().item()
    total = labels.numel()
    return loss, correct, total


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_mat = scipy.io.loadmat(args.train_mat)
    test_mat = scipy.io.loadmat(args.test_mat)
    _, train_x, train_y = convert_dataset_wtime(train_mat)
    _, test_x, test_y = convert_dataset_wtime(test_mat)

    train_x = torch.from_numpy(train_x).float()
    test_x = torch.from_numpy(test_x).float()
    train_y = torch.from_numpy(train_y).long()
    test_y = torch.from_numpy(test_y).long()

    label_min = min(train_y.min().item(), test_y.min().item())
    if label_min != 0:
        train_y -= label_min
        test_y -= label_min

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True,
    )

    print(f"Train samples: {train_x.shape[0]}, Test samples: {test_x.shape[0]}")
    print(f"Input size: {train_x.shape[2]}, Output size: {args.out_size}")

    # Model
    model = SPRiFECGModel(
        input_size=train_x.shape[2],
        hidden_sizes=list(args.hidden_sizes),
        output_size=args.out_size,
        mode=args.mode,
        neuron_kwargs={
            "threshold": args.neuron_threshold,
            "init_std": args.neuron_init_std,
            "bias": args.neuron_bias,
        },
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            logits = model(inputs)
            loss, correct, total = compute_loss_and_correct(logits, labels, criterion)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_correct += correct
            train_total += total

        scheduler.step()

        # Evaluation
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                logits = model(inputs)
                loss, correct, total = compute_loss_and_correct(logits, labels, criterion)
                test_loss_sum += loss.item()
                test_correct += correct
                test_total += total

        train_loss = train_loss_sum / max(len(train_loader), 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_loss = test_loss_sum / max(len(test_loader), 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            hs_str = "hs" + "".join(str(h) for h in args.hidden_sizes)
            save_name = (
                f"SPRiFECGModel_{hs_str}_bs{args.batch_size}"
                f"_lr{args.lr}_seed{args.seed}_acc{best_test_acc:.2f}.pth"
            )
            torch.save(model.state_dict(), save_name)

    print(f"\nTraining complete. Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
