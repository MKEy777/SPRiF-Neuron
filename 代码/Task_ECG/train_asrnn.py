"""
Training script for ASRNN on ECG (QTDB Heartbeat Classification).

This script trains the ASRNN model for the robustness experiment comparison.

Usage:
    cd 代码/Task_ECG
    python train_asrnn.py --train-mat data/QTDB_train.mat --test-mat data/QTDB_test.mat --epochs 250
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import scipy.io

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model_wrapper"))

from core_algorithm.utils import convert_dataset_wtime
from model_wrapper.asrnn_ecg import ASRNNECGNet


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device, dims):
    """Training loop for ECG."""
    best_acc = 0.0
    best_ckpt_path = None

    for epoch in range(epochs):
        model.train()
        train_acc = 0
        train_loss_sum = 0
        sum_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, dims["samples"], dims["input"]).requires_grad_().to(device)
            labels = labels.view(-1, dims["samples"]).long().to(device)

            optimizer.zero_grad()
            logits = model(images)  # [B, C, T]

            # Dense per-timestep cross-entropy (matches SPRiF train.py convention).
            # Multiply by T so the gradient scale is equivalent to summing over
            # timesteps (not averaging), otherwise Adam sees a T-fold shrunk signal.
            l_task = criterion(
                logits.permute(0, 2, 1).reshape(-1, logits.size(1)),
                labels.reshape(-1),
            )
            train_loss = l_task * logits.size(2)

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            predicted = logits.argmax(dim=1)  # [B, T]
            train_acc += (predicted == labels).sum()
            sum_samples += labels.numel()

        if scheduler:
            scheduler.step()

        train_acc = train_acc.data.cpu().numpy() / sum_samples

        # Validation
        model.eval()
        valid_acc = 0
        sum_samples_valid = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, dims["samples"], dims["input"]).to(device)
                labels = labels.view(-1, dims["samples"]).long().to(device)
                logits = model(images)  # [B, C, T]
                predicted = logits.argmax(dim=1)  # [B, T]
                valid_acc += (predicted == labels).sum()
                sum_samples_valid += labels.numel()

        valid_acc = valid_acc.data.cpu().numpy() / sum_samples_valid

        print(f'Epoch {epoch}: Train Loss: {train_loss_sum/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

        # Save best model (放宽阈值：只要验证准确率提升就保存)
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = f'ASRNNECGModel_acc{best_acc:.4f}.pth'
            if best_ckpt_path is not None and best_ckpt_path != save_path and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                except OSError:
                    pass
            torch.save(model.state_dict(), save_path)
            best_ckpt_path = save_path
            print(f'Saved best model: {save_path}')

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train ASRNN on ECG')
    parser.add_argument('--train-mat', type=str, default='data/QTDB_train.mat',
                        help='Path to training MAT file')
    parser.add_argument('--test-mat', type=str, default='data/QTDB_test.mat',
                        help='Path to test MAT file')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=36,
                        help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    if not os.path.exists(args.train_mat):
        print(f"Training data not found at {args.train_mat}")
        print("Please place QTDB_train.mat in data/")
        return

    train_mat = scipy.io.loadmat(args.train_mat)
    test_mat = scipy.io.loadmat(args.test_mat)

    train_dt, train_x, train_y = convert_dataset_wtime(train_mat)
    test_dt, test_x, test_y = convert_dataset_wtime(test_mat)

    nb_of_sample, seq_dim, input_dim = np.shape(train_x)
    print(f'Sequence length: {seq_dim}, Input dimension: {input_dim}')
    print(f'Training samples: {nb_of_sample}, Test samples: {len(test_x)}')

    # Create data loaders
    sub_seq_length = 10
    hidden_dim = args.hidden_size
    output_dim = 6

    train_data = TensorDataset(torch.from_numpy(train_x * 1.), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)

    test_data = TensorDataset(torch.from_numpy(test_x * 1.), torch.from_numpy(test_y))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size, drop_last=False)

    # Create model
    model = ASRNNECGNet(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_classes=output_dim,
        sub_seq_length=sub_seq_length,
        device=str(device)
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()

    base_params = [
        model.i2h.weight, model.i2h.bias,
        model.h2h.weight, model.h2h.bias,
        model.h2o.weight, model.h2o.bias
    ]

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.tau_m_h, 'lr': args.lr * 3},
        {'params': model.tau_m_o, 'lr': args.lr * 2},
        {'params': model.tau_adp_h, 'lr': args.lr * 3},
        {'params': model.tau_adp_o, 'lr': args.lr * 2},
    ], lr=args.lr)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    dims = {
        "samples": seq_dim,
        "input": input_dim,
        "hidden": hidden_dim,
        "output": output_dim,
        "sub_seq": sub_seq_length
    }

    # Train
    best_acc = train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, device, dims)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
