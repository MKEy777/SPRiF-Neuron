
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from core_algorithm.utils import set_seed
from model_ablation_a import SequentialMNIST, SPRiFSMNISTNetAblationA

def get_args():
    parser = argparse.ArgumentParser(
        description="SPRiF Ablation A: S-MNIST with ω=0 (no rotation coupling)"
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

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = SPRiFSMNISTNetAblationA(
        input_size=1,
        hidden_sizes=list(args.hidden_sizes),
        num_classes=args.num_classes,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
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
                loss.backward()
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

