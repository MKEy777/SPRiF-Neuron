
import argparse
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib import set_seed
from model_ablation_a import PermutedMNIST, SPRiFpSMNISTNetAblationA

def diagnose_model(model, label=""):
    for li, layer in enumerate(model.layers):
        runtime = layer._precompute_runtime_params()
        alpha = runtime["alpha"].detach()
        print(f"  [{label}] L{li} alpha:          min={alpha.min():.6f}  max={alpha.max():.6f}  mean={alpha.mean():.6f}")
        rho = runtime["rho"].detach()
        print(f"  [{label}] L{li} rho:            min={rho.min():.6f}  max={rho.max():.6f}  mean={rho.mean():.6f}")
        eta = runtime["eta"].squeeze(0).detach()
        print(f"  [{label}] L{li} eta[0]:         min={eta[:,0].min():.6f}  max={eta[:,0].max():.6f}  mean={eta[:,0].mean():.6f}")
        print(f"  [{label}] L{li} eta[1]:         min={eta[:,1].min():.6f}  max={eta[:,1].max():.6f}  mean={eta[:,1].mean():.6f}")
        chi = runtime["fast_coupling"].detach()
        print(f"  [{label}] L{li} fast_coupling:  min={chi.min():.6f}  max={chi.max():.6f}  mean={chi.mean():.6f}  abs_mean={chi.abs().mean():.6f}")
        lam = runtime["lambda_reset"].detach()
        print(f"  [{label}] L{li} lambda_reset:   min={lam.min():.6f}  max={lam.max():.6f}  mean={lam.mean():.6f}")
        G = layer.G.detach()
        G0_norm = G[:, 0, :].norm(dim=1)
        G1_norm = G[:, 1, :].norm(dim=1)
        print(f"  [{label}] L{li} G0_norm:        min={G0_norm.min():.4f}  max={G0_norm.max():.4f}  mean={G0_norm.mean():.4f}")
        print(f"  [{label}] L{li} G1_norm:        min={G1_norm.min():.4f}  max={G1_norm.max():.4f}  mean={G1_norm.mean():.4f}")
        iw = layer.input_linear.weight.detach()
        print(f"  [{label}] L{li} input_linear W:  norm={iw.norm():.4f}  abs_mean={iw.abs().mean():.6f}")
        if layer.recurrent_linear is not None:
            rw = layer.recurrent_linear.weight.detach()
            print(f"  [{label}] L{li} recurrent  W:   norm={rw.norm():.4f}  abs_mean={rw.abs().mean():.6f}")
        raw_alpha = layer.alpha_raw.detach()
        print(f"  [{label}] L{li} alpha_raw:      min={raw_alpha.min():.4f}  max={raw_alpha.max():.4f}  mean={raw_alpha.mean():.4f}")
        raw_rho = layer.rho_raw.detach()
        print(f"  [{label}] L{li} rho_raw:        min={raw_rho.min():.4f}  max={raw_rho.max():.4f}  mean={raw_rho.mean():.4f}")
    rw = model.readout.weight.detach()
    print(f"  [{label}] readout W:       norm={rw.norm():.4f}  abs_mean={rw.abs().mean():.6f}")

def check_weight_nan(model):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print(f"  *** NaN/Inf in parameter: {name} ***")
            print(f"      min={p.min()}  max={p.max()}  has_nan={torch.isnan(p).any()}  has_inf={torch.isinf(p).any()}")
            return True
    return False

def get_args():
    parser = argparse.ArgumentParser(
        description="SPRiF Ablation A: PS-MNIST with ω=0 (no rotation coupling)"
    )

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 210])
    parser.add_argument("--mode", type=str, default="srnn")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)

    parser.add_argument("--tbptt-len", type=int, default=262,
                        help="TBPTT chunk length (0 = full BPTT)")

    parser.add_argument("--scheduler-step", type=int, default=50)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Ablation A: ω=0, no rotation coupling (3D slow, 2D fast, G=2x3)")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform,
    )
    test_mnist = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform,
    )

    torch.manual_seed(args.seed)
    perm = torch.randperm(784)
    train_dataset = PermutedMNIST(train_mnist, perm)
    test_dataset = PermutedMNIST(test_mnist, perm)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2,
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = SPRiFpSMNISTNetAblationA(
        input_size=1,
        hidden_sizes=list(args.hidden_sizes),
        num_classes=args.num_classes,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("\n=== Initial parameters ===")
    diagnose_model(model, label="init")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        nan_detected = False

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)
            B, T, F = x.shape

            tbptt_len = args.tbptt_len
            states = model.init_states(B, device=device, dtype=x.dtype)

            batch_loss = 0.0
            chunk_count = 0
            train_logits_list = []

            for start in range(0, T, tbptt_len):
                end = min(start + tbptt_len, T)
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

                optimizer.zero_grad()
                loss = criterion(chunk_logits, y)

                if not torch.isfinite(loss):
                    print(f"\n*** NaN/Inf loss at epoch {epoch}, batch {batch_idx}, chunk {start} ***")
                    print(f"    chunk_logits: min={chunk_logits.min():.4f} max={chunk_logits.max():.4f}")
                    print(f"    chunk_logits has_nan={torch.isnan(chunk_logits).any()} has_inf={torch.isinf(chunk_logits).any()}")

                    for i, ns in enumerate(new_states):
                        for k, v in ns.items():
                            print(f"    L{i} state['{k}']: shape={v.shape} min={v.min():.4f} max={v.max():.4f} "
                                  f"has_nan={torch.isnan(v).any()} has_inf={torch.isinf(v).any()}")
                    diagnose_model(model, label=f"NaN-epoch{epoch}-batch{batch_idx}")
                    nan_detected = True
                    break

                loss.backward()

                if epoch >= 40 and batch_idx == 0 and start == 0:
                    total_grad_norm = 0.0
                    for name, p in model.named_parameters():
                        if p.grad is not None and torch.isfinite(p.grad).all():
                            g_norm = p.grad.norm().item()
                            total_grad_norm += g_norm ** 2
                            if g_norm > 10:
                                print(f"    [grad] {name}: norm={g_norm:.2f}")
                    total_grad_norm = math.sqrt(total_grad_norm)
                    if total_grad_norm > 10:
                        print(f"    [grad] total_norm (pre-clip) = {total_grad_norm:.2f}")

                optimizer.step()

                if epoch >= 40:
                    if check_weight_nan(model):
                        nan_detected = True
                        break

                batch_loss += loss.item()
                chunk_count += 1
                train_logits_list.append(chunk_logits.detach())

                states = model.detach_states(new_states)

            if nan_detected:
                break

            train_loss += batch_loss / max(chunk_count, 1) * x.size(0)
            with torch.no_grad():
                avg_logits = torch.stack(train_logits_list).mean(dim=0)
                train_correct += (avg_logits.argmax(dim=-1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        diagnose_model(model, label=f"epoch{epoch:03d}")

        if nan_detected:
            print(f"\n*** Training aborted at epoch {epoch} due to NaN ***")
            break

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

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print(f"\nAblation A complete. Best test accuracy: {best_test_acc:.2f}%")

if __name__ == "__main__":
    main()

