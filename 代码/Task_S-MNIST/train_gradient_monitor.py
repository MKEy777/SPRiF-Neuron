"""
Gradient-monitored training for SPRiF / LIF on S-MNIST.
Records per-chunk gradient norms for gradient stability analysis.

Usage:
    python train_gradient_monitor.py          # SPRiF
    python train_gradient_monitor.py --model lif   # LIF
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from core_algorithm.utils import set_seed
from model import SequentialMNIST, SPRiFSMNISTNet
from model_lif import LIFSMNISTNet


def get_args():
    parser = argparse.ArgumentParser(description="Gradient-monitored S-MNIST training")
    parser.add_argument("--model", type=str, default="sprif", choices=["sprif", "lif"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--mode", type=str, default="sfnn")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--tbptt-len", type=int, default=262)
    parser.add_argument("--scheduler-step", type=int, default=50)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient max norm clipping (0 = no clip)")
    parser.add_argument("--grad-log-dir", type=str, default="./grad_logs")
    parser.add_argument("--save-cpkt", action="store_true", default=True)
    return parser.parse_args()


def compute_gradient_norms(model, model_type):
    records = {}
    total_sq = 0.0
    layer_sq = [0.0 for _ in range(2)]
    slow_sq = 0.0
    fast_sq = 0.0
    readout_sq = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_norm_sq = p.grad.norm().item() ** 2
        total_sq += g_norm_sq

        if name.startswith("readout"):
            readout_sq += g_norm_sq
        elif name.startswith("layers.0"):
            layer_sq[0] += g_norm_sq
            if model_type == "sprif":
                pname = name.split(".")[-1]
                if pname in ("alpha_raw", "rho_raw", "omega_raw", "G"):
                    slow_sq += g_norm_sq
                else:
                    fast_sq += g_norm_sq
        elif name.startswith("layers.1"):
            layer_sq[1] += g_norm_sq
            if model_type == "sprif":
                pname = name.split(".")[-1]
                if pname in ("alpha_raw", "rho_raw", "omega_raw", "G"):
                    slow_sq += g_norm_sq
                else:
                    fast_sq += g_norm_sq

    records["total"] = math.sqrt(total_sq)
    records["layer_0"] = math.sqrt(layer_sq[0])
    records["layer_1"] = math.sqrt(layer_sq[1])
    records["readout"] = math.sqrt(readout_sq)
    if model_type == "sprif":
        records["slow"] = math.sqrt(slow_sq)
        records["fast"] = math.sqrt(fast_sq)
    return records


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}, Device: {device}, Seed: {args.seed}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform,
    )
    test_mnist = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform,
    )

    train_dataset = SequentialMNIST(train_mnist)
    test_dataset = SequentialMNIST(test_mnist)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2,
    )

    if args.model == "sprif":
        model = SPRiFSMNISTNet(
            input_size=1, hidden_sizes=list(args.hidden_sizes),
            num_classes=args.num_classes, mode=args.mode,
            warmup_steps=args.warmup_steps,
        ).to(device)
    else:
        model = LIFSMNISTNet(
            input_size=1, hidden_sizes=list(args.hidden_sizes),
            num_classes=args.num_classes, mode=args.mode,
            warmup_steps=args.warmup_steps,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.grad_log_dir, exist_ok=True)
    grad_log = []

    best_test_acc = 0.0
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_grad_records = {
            "total_grad_norm": [],
            "layer_0_grad_norm": [],
            "layer_1_grad_norm": [],
            "readout_grad_norm": [],
        }
        if args.model == "sprif":
            epoch_grad_records["slow_grad_norm"] = []
            epoch_grad_records["fast_grad_norm"] = []

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)
            B, T, F = x.shape

            states = model.init_states(B, device=device, dtype=x.dtype)
            batch_loss = 0.0
            chunk_count = 0
            train_logits_list = []

            for start in range(0, T, args.tbptt_len):
                end = min(start + args.tbptt_len, T)
                chunk = x[:, start:end]

                out = chunk
                new_states = []
                for i, layer in enumerate(model.layers):
                    out, ns = layer.forward_with_state(out, states[i], batch_first=True)
                    new_states.append(ns)

                logits_chunk = model.readout(out)

                local_warmup = max(model.warmup_steps - start, 0)
                if local_warmup >= logits_chunk.size(1):
                    states = model.detach_states(new_states)
                    continue

                valid_logits = logits_chunk[:, local_warmup:, :]
                chunk_logits = valid_logits.mean(dim=1)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(chunk_logits, y)
                loss.backward()

                grad_info = compute_gradient_norms(model, args.model)
                epoch_grad_records["total_grad_norm"].append(grad_info["total"])
                epoch_grad_records["layer_0_grad_norm"].append(grad_info["layer_0"])
                epoch_grad_records["layer_1_grad_norm"].append(grad_info["layer_1"])
                epoch_grad_records["readout_grad_norm"].append(grad_info["readout"])
                if args.model == "sprif":
                    epoch_grad_records["slow_grad_norm"].append(grad_info["slow"])
                    epoch_grad_records["fast_grad_norm"].append(grad_info["fast"])

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                batch_loss += loss.item()
                chunk_count += 1
                train_logits_list.append(chunk_logits.detach())

                states = model.detach_states(new_states)

            train_loss += batch_loss / max(chunk_count, 1) * x.size(0)
            with torch.no_grad():
                avg_logits = torch.stack(train_logits_list).mean(dim=0)
                train_correct += (avg_logits.argmax(dim=-1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)
                logits = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item() * x.size(0)
                test_correct += (logits.argmax(dim=-1) == y).sum().item()
                test_total += x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_loss /= max(test_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)

        for k in epoch_grad_records:
            if len(epoch_grad_records[k]) == 0:
                epoch_grad_records[k] = [0.0]

        epoch_entry = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "total_grad_mean": float(np.mean(epoch_grad_records["total_grad_norm"])),
            "total_grad_std": float(np.std(epoch_grad_records["total_grad_norm"])),
            "total_grad_max": float(max(epoch_grad_records["total_grad_norm"])),
            "total_grad_min": float(min(epoch_grad_records["total_grad_norm"])),
            "layer_0_grad_mean": float(np.mean(epoch_grad_records["layer_0_grad_norm"])),
            "layer_1_grad_mean": float(np.mean(epoch_grad_records["layer_1_grad_norm"])),
            "readout_grad_mean": float(np.mean(epoch_grad_records["readout_grad_norm"])),
            "total_grad_raw": [float(v) for v in epoch_grad_records["total_grad_norm"]],
            "layer_0_grad_raw": [float(v) for v in epoch_grad_records["layer_0_grad_norm"]],
            "layer_1_grad_raw": [float(v) for v in epoch_grad_records["layer_1_grad_norm"]],
        }
        if args.model == "sprif":
            epoch_entry["slow_grad_mean"] = float(np.mean(epoch_grad_records["slow_grad_norm"]))
            epoch_entry["fast_grad_mean"] = float(np.mean(epoch_grad_records["fast_grad_norm"]))
            epoch_entry["slow_grad_raw"] = [float(v) for v in epoch_grad_records["slow_grad_norm"]]
            epoch_entry["fast_grad_raw"] = [float(v) for v in epoch_grad_records["fast_grad_norm"]]

        grad_log.append(epoch_entry)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"||∇||: {epoch_entry['total_grad_mean']:.4f}±{epoch_entry['total_grad_std']:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_cpkt:
                hs_str = "hs" + "".join(str(h) for h in args.hidden_sizes)
                prefix = "SPRiFSMNISTNet" if args.model == "sprif" else "LIFSMNISTNet"
                save_name = (
                    f"{prefix}_{hs_str}_bs{args.batch_size}"
                    f"_lr{args.lr}_seed{args.seed}_acc{best_test_acc:.2f}.pth"
                )
                if best_ckpt_path is not None and best_ckpt_path != save_name and os.path.exists(best_ckpt_path):
                    try:
                        os.remove(best_ckpt_path)
                    except OSError:
                        pass
                torch.save(model.state_dict(), save_name)
                best_ckpt_path = save_name

    hs_str = "hs" + "".join(str(h) for h in args.hidden_sizes)
    log_name = (
        f"grad_log_{args.model}_{hs_str}_bs{args.batch_size}"
        f"_lr{args.lr}_seed{args.seed}.json"
    )
    log_path = os.path.join(args.grad_log_dir, log_name)
    with open(log_path, "w") as f:
        json.dump(grad_log, f, indent=2)
    print(f"Gradient log saved to {log_path}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
