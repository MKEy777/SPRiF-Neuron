"""
Training script for ASRNN on Google Speech Commands (GSC).

This script trains the ASRNN model for the robustness experiment comparison.

Usage:
    cd 代码/Task_GSC
    python train_asrnn.py --data-root data/SpeechCommands --epochs 150
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torch.optim.lr_scheduler import StepLR

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model_wrapper"))

# Use the same data pipeline as train.py for consistent preprocessing.
from data import SpeechCommandsDataset, Pad, MelSpectrogram, Rescale
from model_wrapper.asrnn_gsc import ASRNNGSCNet

# Hyperparameters (match train.py defaults)
sr = 16000
size = 16000
n_fft = int(30e-3 * sr)
hop_length = int(10e-3 * sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True


def collate_fn(data):
    x_batch = np.array([d[0] for d in data])
    std = x_batch.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return torch.tensor(x_batch / std).float(), torch.tensor([d[1] for d in data]).long()


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    """Training loop."""
    best_acc = 0.0
    best_ckpt_path = None

    for epoch in range(epochs):
        model.train()
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 3, 101, 40).to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions, labels)

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            train_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()

        if scheduler:
            scheduler.step()

        train_acc = train_acc.data.cpu().numpy() / sum_sample

        # Validation
        model.eval()
        valid_acc = 0
        sum_sample_valid = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.view(-1, 3, 101, 40).to(device, non_blocking=True)
                labels = labels.view(-1).long().to(device, non_blocking=True)
                predictions = model(images)
                _, predicted = torch.max(predictions.data, 1)
                valid_acc += (predicted == labels).sum()
                sum_sample_valid += predicted.numel()

        valid_acc = valid_acc.data.cpu().numpy() / sum_sample_valid

        print(f'Epoch {epoch}: Train Loss: {train_loss_sum/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

        # Save best model
        if valid_acc > best_acc and train_acc > 0.85:
            best_acc = valid_acc
            save_path = f'ASRNNGSCNet_acc{best_acc:.4f}.pth'
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
    parser = argparse.ArgumentParser(description='Train ASRNN on GSC')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'dataset/SpeechCommands/speech_commands_v0.02'),
                        help='Path to SpeechCommands dataset root '
                             '(contains testing_list.txt / validation_list.txt)')
    parser.add_argument('--cache-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'dataset/SpeechCommands/cache_power_to_db'),
                        help='Optional cache root for pre-computed mel-spectrograms '
                             '(pass "none" to disable disk cache)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data preprocessing (match train.py exactly)
    transform = torchvision.transforms.Compose(
        [
            Pad(size),
            MelSpectrogram(
                sr, n_fft, hop_length, n_mels,
                fmin, fmax, delta_order, stack=stack,
            ),
            Rescale(),
        ]
    )

    # Match train.py: dataset lives under a single root containing
    # testing_list.txt / validation_list.txt. The dataset class itself
    # splits samples into train / valid based on those lists.
    data_root = args.data_root
    if not os.path.exists(data_root):
        print(f"Dataset not found at {data_root}")
        print("Please download GSC dataset first (e.g. python download_GSC.py)")
        return

    # Build label dictionary by scanning sub-folders of the data root.
    label_dct = {}
    for w in os.listdir(data_root):
        full = os.path.join(data_root, w)
        if os.path.isdir(full) and w[0] != "_":
            label_dct[w] = len(label_dct)
    # Collapse non-keyword classes into _unknown_ to match train.py's 12-class scheme.
    testing_words = ["yes", "no", "up", "down", "left", "right",
                     "on", "off", "stop", "go"]
    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}
    for w in list(os.listdir(data_root)):
        full = os.path.join(data_root, w)
        if os.path.isdir(full) and w[0] != "_" and w not in label_dct:
            label_dct[w] = label_dct["_unknown_"]

    print(f"Number of classes: {len(set(label_dct.values()))}")

    cache_root = args.cache_root if args.cache_root else data_root

    # Create datasets
    train_dataset = SpeechCommandsDataset(
        data_root, label_dct, mode="train",
        transform=transform, cache_root=cache_root,
    )
    train_sampler = WeightedRandomSampler(
        train_dataset.weights, len(train_dataset.weights)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    valid_dataset = SpeechCommandsDataset(
        data_root, label_dct, mode="valid",
        transform=transform, cache_root=cache_root,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = ASRNNGSCNet(
        input_size=120,  # 3 * 40
        hidden_size=args.hidden_size,
        num_classes=12,
        device=str(device)
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()

    base_params = [
        model.dense_1.dense.weight, model.dense_1.dense.bias,
        model.rnn_1.dense.weight, model.rnn_1.dense.bias,
        model.rnn_1.recurrent.weight, model.rnn_1.recurrent.bias,
        model.dense_2.dense.weight, model.dense_2.dense.bias,
    ]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr},
        {'params': model.thr, 'lr': args.lr * 0.01},
        {'params': model.dense_1.tau_m, 'lr': args.lr * 2},
        {'params': model.dense_2.tau_m, 'lr': args.lr * 2},
        {'params': model.rnn_1.tau_m, 'lr': args.lr * 2},
        {'params': model.dense_1.tau_adp, 'lr': args.lr * 2},
        {'params': model.rnn_1.tau_adp, 'lr': args.lr * 2},
    ], lr=args.lr)

    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Train
    best_acc = train(model, train_loader, valid_loader, criterion, optimizer, scheduler, args.epochs, device)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
