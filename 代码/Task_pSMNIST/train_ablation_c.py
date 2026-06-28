"""SPRiF Ablation C - PS-MNIST with scalar reset (lambda=0)."""
import argparse, torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from lib import set_seed
from model_ablation_c import PermutedMNIST, SPRiFpSMNISTNetAblationC

def get_args():
    p = argparse.ArgumentParser(description="SPRiF Ablation C: PS-MNIST scalar reset")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 256])
    p.add_argument("--mode", type=str, default="srnn")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--tbptt-len", type=int, default=262, help="TBPTT chunk (0=full BPTT)")
    p.add_argument("--scheduler-step", type=int, default=50)
    p.add_argument("--scheduler-gamma", type=float, default=0.1)
    return p.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Ablation C: scalar reset (lambda=0, fixed reset direction [1,0])")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_mnist  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    torch.manual_seed(args.seed)
    perm = torch.randperm(784)
    train_ds = PermutedMNIST(train_mnist, perm)
    test_ds  = PermutedMNIST(test_mnist, perm)
    pin_memory = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4,
                          pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    model = SPRiFpSMNISTNetAblationC(input_size=1, hidden_sizes=list(args.hidden_sizes),
                                     num_classes=args.num_classes, mode=args.mode,
                                     warmup_steps=args.warmup_steps).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = StepLR(opt, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_ld:
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

                opt.zero_grad(set_to_none=True)
                loss = crit(chunk_logits, y)
                loss.backward(); opt.step()

                batch_loss += loss.item()
                chunk_count += 1
                train_logits_list.append(chunk_logits.detach())

                states = model.detach_states(new_states)

            tl += batch_loss / max(chunk_count, 1) * x.size(0)
            with torch.no_grad():
                avg_logits = torch.stack(train_logits_list).mean(dim=0)
                tc += (avg_logits.argmax(dim=-1) == y).sum().item()
            tt += x.size(0)
        sch.step()
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)
                logits = model(x)  # Full BPTT for eval
                loss = crit(logits, y)
                vl += loss.item() * x.size(0)
                vc += (logits.argmax(dim=-1) == y).sum().item()
                vt += x.size(0)
        ta = 100.0 * tc / max(tt, 1)
        va = 100.0 * vc / max(vt, 1)
        print(f"Epoch {ep:03d} | Train: {tl/max(tt,1):.4f} {ta:.2f}% | Test: {vl/max(vt,1):.4f} {va:.2f}%")
        if va > best:
            best = va
    print(f"\nAblation C complete. Best test acc: {best:.2f}%")
if __name__ == "__main__": main()
