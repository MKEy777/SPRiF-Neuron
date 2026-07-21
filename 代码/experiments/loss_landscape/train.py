import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from config import get_args
from data import get_loaders
from models import build_model

sys.path.insert(0, cfg.SMNIST_DIR)
from core_algorithm.utils import set_seed

def train_one_neuron(neuron_type, args, device):
    print(f"\n{'='*60}\n训练神经元: {neuron_type}\n{'='*60}")

    train_loader, test_loader = get_loaders(args.batch_size, device, subset=args.subset)
    model = build_model(neuron_type, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER_STEP, gamma=cfg.SCHEDULER_GAMMA)
    criterion = nn.CrossEntropyLoss()
    pin = device.type == "cuda"

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{neuron_type}_best.pth")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            B, T, _ = x.shape

            tbptt_len = cfg.TBPTT_LEN
            states = model.init_states(B, device=device, dtype=x.dtype)
            batch_loss, chunk_count = 0.0, 0
            logits_list = []

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

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(chunk_logits, y)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                chunk_count += 1
                logits_list.append(chunk_logits.detach())
                states = model.detach_states(new_states)

            train_loss += batch_loss / max(chunk_count, 1) * B
            with torch.no_grad():
                avg_logits = torch.stack(logits_list).mean(dim=0)
                train_correct += (avg_logits.argmax(dim=-1) == y).sum().item()
            train_total += B

        scheduler.step()

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                logits = model(x)
                test_correct += (logits.argmax(dim=-1) == y).sum().item()
                test_total += x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)
        print(f"[{neuron_type}] Epoch {epoch:03d} | "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
              f"Test Acc {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "neuron_type": neuron_type,
                "state_dict": model.state_dict(),
                "hidden_sizes": list(cfg.HIDDEN_SIZES),
                "test_acc": best_acc,
            }, ckpt_path)

    print(f"[{neuron_type}] 训练完成，最佳测试精度 {best_acc:.2f}%，权重 -> {ckpt_path}")
    return best_acc

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neurons = ["sprif", "lif"] if args.neuron == "both" else [args.neuron]
    for nt in neurons:
        train_one_neuron(nt, args, device)

if __name__ == "__main__":
    main()

