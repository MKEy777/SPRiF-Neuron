import argparse
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler

from core_algorithm.utils import set_seed
from core_algorithm.sprif_layer import SPRiFNeuronLayer
from model import SPRiFGSCNet
from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="SPRiF-GSC: Google Speech Commands")

    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-classes", type=int, default=12)

    parser.add_argument("--neuron-threshold", type=float, default=0.8)
    parser.add_argument("--neuron-init-std", type=float, default=0.1)

    parser.add_argument("--tau-alpha-min", type=float, default=10.0)
    parser.add_argument("--tau-alpha-max", type=float, default=80.0)
    parser.add_argument("--tau-rho-min", type=float, default=4.0)
    parser.add_argument("--tau-rho-max", type=float, default=30.0)
    parser.add_argument("--tau-eta-min", type=float, default=0.8)
    parser.add_argument("--tau-eta-max", type=float, default=8.0)
    parser.add_argument("--omega-min", type=float, default=0.04)
    parser.add_argument("--omega-max", type=float, default=0.40)

    parser.add_argument("--data-root", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "dataset/SpeechCommands/speech_commands_v0.02"))
    parser.add_argument("--cache-root", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "dataset/SpeechCommands/cache_power_to_db"))

    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=101)
    parser.add_argument("--input-size", type=int, default=120)
    parser.add_argument("--wav-size", type=int, default=16000)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--fmin", type=int, default=20)
    parser.add_argument("--fmax", type=int, default=4000)
    parser.add_argument("--delta-order", type=int, default=2)

    parser.add_argument("--scheduler-step", type=int, default=30)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    return parser.parse_args()

def collate_fn(data):
    return torch.tensor(np.array([d[0] for d in data])).float(), torch.tensor([d[1] for d in data]).long()

def build_loaders(args):
    testing_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}

    for w in os.listdir(args.data_root):
        full = os.path.join(args.data_root, w)
        if os.path.isdir(full) and w[0] != "_" and w not in label_dct:
            label_dct[w] = label_dct["_unknown_"]

    n_fft = int(30e-3 * args.sr)
    hop_length = int(10e-3 * args.sr)

    transform = torchvision.transforms.Compose(
        [
            Pad(args.wav_size),
            MelSpectrogram(
                args.sr, n_fft, hop_length, args.n_mels,
                args.fmin, args.fmax, args.delta_order, stack=True,
            ),
            Rescale(),
        ]
    )

    cache_root = args.cache_root if args.cache_root else args.data_root

    print("Initializing Datasets (Checking/Creating disk cache)...")
    train_dataset = SpeechCommandsDataset(
        args.data_root, label_dct, mode="train",
        transform=transform, cache_root=cache_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights)),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    valid_dataset = SpeechCommandsDataset(
        args.data_root, label_dct, mode="valid",
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

    return train_loader, valid_loader

@torch.no_grad()
def evaluate(model, loader, criterion, device, seq_len=101, n_mels=40, input_size=120):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.view(-1, 3, seq_len, n_mels).to(device, non_blocking=True)
        x = x.permute(0, 2, 1, 3).reshape(-1, seq_len, input_size)
        y = y.to(device, non_blocking=True)

        logits, _ = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, valid_loader = build_loaders(args)

    neuron_kwargs = {
        "threshold": args.neuron_threshold,
        "init_std": args.neuron_init_std,
        "tau_alpha_range": (args.tau_alpha_min, args.tau_alpha_max),
        "tau_rho_range": (args.tau_rho_min, args.tau_rho_max),
        "tau_eta_range": (args.tau_eta_min, args.tau_eta_max),
        "omega_range": (args.omega_min * math.pi, args.omega_max * math.pi),
    }
    model = SPRiFGSCNet(
        input_size=args.input_size,
        hidden_sizes=tuple(args.hidden_sizes),
        num_classes=args.num_classes,
        dropout=args.dropout,
        recurrent_flags=tuple(True for _ in args.hidden_sizes),
        neuron_kwargs=neuron_kwargs,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma,
    )
    criterion = nn.NLLLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.view(-1, 3, args.seq_len, args.n_mels).to(device, non_blocking=True)
            x = x.permute(0, 2, 1, 3).reshape(-1, args.seq_len, args.input_size)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)

        scheduler.step()

        train_loss = total_loss / total
        train_acc = total_correct / total
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device,
                                      seq_len=args.seq_len, n_mels=args.n_mels, input_size=args.input_size)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            hs_str = "hs" + "".join(str(h) for h in args.hidden_sizes)
            save_name = (
                f"SPRiFGSCNet_{hs_str}_bs{args.batch_size}"
                f"_lr{args.lr}_seed{args.seed}_acc{best_val_acc:.4f}.pth"
            )
            torch.save(model.state_dict(), save_name)

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()

