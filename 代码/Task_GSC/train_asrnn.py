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
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import StepLR

# Add paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model_wrapper"))

from data_asrnn import SpeechCommandsDataset, Pad, MelSpectrogram, Rescale
from model_wrapper.asrnn_gsc import ASRNNGSCNet

# Hyperparameters
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
    X_batch = np.array([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1
    X_batch = torch.tensor(X_batch / std)
    y_batch = torch.tensor([d[1] for d in data])
    return X_batch, y_batch


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device):
    """Training loop."""
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 3, 101, 40).to(device)
            labels = labels.view(-1).long().to(device)

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
            for images, labels in test_loader:
                images = images.view(-1, 3, 101, 40).to(device)
                labels = labels.view(-1).long().to(device)
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
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model: {save_path}')

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train ASRNN on GSC')
    parser.add_argument('--data-root', type=str, default='/root/autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02',
                        help='Path to SpeechCommands dataset')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
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

    # Data preprocessing
    melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
    pad = Pad(size)
    rescale = Rescale()
    transform = torchvision.transforms.Compose([pad, melspec, rescale])

    # Load label dictionary
    train_data_root = os.path.join(args.data_root, 'train')
    test_data_root = os.path.join(args.data_root, 'test')

    if not os.path.exists(train_data_root):
        print(f"Training data not found at {train_data_root}")
        print("Please download GSC dataset first")
        return

    training_words = [x for x in os.listdir(train_data_root)
                      if os.path.isdir(os.path.join(train_data_root, x)) and x[0] != "_"]
    testing_words = [x for x in os.listdir(test_data_root)
                     if os.path.isdir(os.path.join(test_data_root, x)) and x[0] != "_"]

    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}
    for w in training_words:
        label = label_dct.get(w)
        if label is None:
            label_dct[w] = label_dct["_unknown_"]

    print(f"Number of classes: {len(set(label_dct.values()))}")

    # Create datasets
    train_dataset = SpeechCommandsDataset(
        train_data_root, label_dct, transform=transform, mode="train", cache_root=train_data_root
    )
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_dataset.weights, len(train_dataset.weights)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4,
        sampler=train_sampler, collate_fn=collate_fn
    )

    test_dataset = SpeechCommandsDataset(
        test_data_root, label_dct, transform=transform, mode="test", cache_root=test_data_root
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn
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
    best_acc = train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, device)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
