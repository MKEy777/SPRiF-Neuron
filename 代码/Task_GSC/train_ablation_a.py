"""SPRiF Ablation A — GSC training with ω=0 (no rotation coupling, 3D slow state)."""

import os
import warnings
import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, WeightedRandomSampler
from model_ablation_a import SPRiFGSCNetAblationA
from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CONFIG = {
    "lr": 3e-3, "epochs": 150, "patience": 40, "batch_size": 200,
    "num_workers": 8, "weight_decay": 1e-4, "grad_clip": 10.0, "seed": 42,
    "data_root": "/root/autodl-tmp/dataset/SpeechCommands/speech_commands_v0.02",
    "cache_root": "/root/autodl-tmp/dataset/SpeechCommands/cache_power_to_db",
    "hidden_sizes": [300], "recurrent_flags": [True], "mode": "srnn", "dropout": 0.15,
    "input_size": 120, "num_classes": 12, "n_mels": 40, "seq_len": 101,
    "wav_size": 16000, "sr": 16000, "n_fft": int(30e-3 * 16000),
    "hop_length": int(10e-3 * 16000), "fmin": 20, "fmax": 4000, "delta_order": 2,
    "neuron_threshold": 1.0, "neuron_init_std": 0.1,
}

# Ablation A: no omega_range (ω=0); alpha, rho, eta retained
HYPERPARAM_SEARCH = [
    {"lr": 5e-3, "neuron_threshold": 0.8, "dropout": 0.15,
     "tau_alpha_range": (10.0, 80.0), "tau_rho_range": (4.0, 30.0),
     "tau_eta_range": (0.8, 8.0)},
]


def build_neuron_kwargs(config: dict) -> dict:
    kwargs = {"threshold": config["neuron_threshold"], "init_std": config["neuron_init_std"]}
    for k in ["tau_alpha_range", "tau_rho_range", "tau_eta_range", "omega_range"]:
        if k in config:
            kwargs[k] = config[k]
    return kwargs


def _recurrent_flags_from_mode(config: dict):
    if "recurrent_flags" in config and config["recurrent_flags"] is not None:
        return tuple(config["recurrent_flags"])
    return tuple(config["mode"].lower() == "srnn" for _ in config["hidden_sizes"])


def collate_fn(data):
    x_batch = np.array([d[0] for d in data])
    std = x_batch.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return torch.tensor(x_batch / std).float(), torch.tensor([d[1] for d in data]).long()


def collect_param_stats(model) -> dict:
    stats = {}
    for li, layer in enumerate(model.layers):
        runtime = layer._precompute_runtime_params()
        stats[f"layer{li}/alpha_mean"] = runtime["alpha"].detach().mean().item()
        stats[f"layer{li}/alpha_min"]  = runtime["alpha"].detach().min().item()
        stats[f"layer{li}/alpha_max"]  = runtime["alpha"].detach().max().item()
        rho = runtime["rho"].detach()
        stats[f"layer{li}/rho_mean"] = rho.mean().item()
        stats[f"layer{li}/rho_min"]  = rho.min().item()
        stats[f"layer{li}/rho_max"]  = rho.max().item()
        eta = runtime["eta"].squeeze(0).detach()
        stats[f"layer{li}/eta0_mean"] = eta[:, 0].mean().item()
        stats[f"layer{li}/eta1_mean"] = eta[:, 1].mean().item()
        chi = runtime["fast_coupling"].detach()
        stats[f"layer{li}/chi_abs_mean"] = chi.abs().mean().item()
        G = layer.G.detach()
        stats[f"layer{li}/G0_norm_mean"] = G[:, 0, :].norm(dim=1).mean().item()
        stats[f"layer{li}/G1_norm_mean"] = G[:, 1, :].norm(dim=1).mean().item()
        lam = runtime["lambda_reset"].detach()
        stats[f"layer{li}/lambda_abs_mean"] = lam.abs().mean().item()
    return stats


def preprocess_batch(x, y, device, config):
    x = x.view(-1, 3, config["seq_len"], config["n_mels"]).to(device, non_blocking=True)
    x = x.permute(0, 2, 1, 3).reshape(-1, config["seq_len"], config["input_size"])
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, loader, criterion, device, config):
    model.eval()
    total_loss, total_correct, total, avg_spike = 0.0, 0, 0, 0.0
    for x, y in loader:
        x, y = preprocess_batch(x, y, device, config)
        logits, aux = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
        avg_spike += aux["spike_rate"].item() * x.size(0)
    metrics = {"val_loss": total_loss / total, "val_acc": total_correct / total,
               "val_spike": avg_spike / total}
    metrics.update(collect_param_stats(model))
    return metrics


def run_experiment(params, train_loader, valid_loader, device):
    config = dict(DEFAULT_CONFIG); config.update(params)
    model = SPRiFGSCNetAblationA(
        input_size=config["input_size"], hidden_sizes=tuple(config["hidden_sizes"]),
        num_classes=config["num_classes"], dropout=config["dropout"],
        recurrent_flags=_recurrent_flags_from_mode(config),
        neuron_kwargs=build_neuron_kwargs(config)).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_val_acc, patience_counter = 0.0, 0
    best_ckpt_path = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = preprocess_batch(x, y, device, config)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        train_loss = total_loss / total; train_acc = total_correct / total
        metrics = evaluate(model, valid_loader, criterion, device, config)

        print(f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_loss={metrics['val_loss']:.4f} | "
              f"val_acc={metrics['val_acc']:.4f} | val_spike={metrics['val_spike']:.4f} | "
              f"alpha={metrics['layer0/alpha_mean']:.3f} | "
              f"rho={metrics['layer0/rho_mean']:.3f} | "
              f"eta0={metrics['layer0/eta0_mean']:.3f} | "
              f"eta1={metrics['layer0/eta1_mean']:.3f} | "
              f"chi={metrics.get('layer0/chi_abs_mean',0):.4f} | "
              f"G0={metrics['layer0/G0_norm_mean']:.3f} | "
              f"G1={metrics['layer0/G1_norm_mean']:.3f} | "
              f"lambda={metrics['layer0/lambda_abs_mean']:.3f}")

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]; patience_counter = 0
            hs_str = "hs" + "".join(str(h) for h in config["hidden_sizes"])
            save_name = (f"SPRiFGSCNetAblationA_{hs_str}_bs{config['batch_size']}"
                         f"_lr{config['lr']}_seed{config['seed']}_acc{best_val_acc:.4f}.pth")
            if best_ckpt_path is not None and best_ckpt_path != save_name and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                except OSError:
                    pass
            torch.save(model.state_dict(), save_name)
            best_ckpt_path = save_name
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("  -> Early stopping triggered."); break
    return best_val_acc


def build_loaders(config):
    testing_words = ["yes","no","up","down","left","right","on","off","stop","go"]
    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}
    for w in os.listdir(config["data_root"]):
        full = os.path.join(config["data_root"], w)
        if os.path.isdir(full) and w[0] != "_" and w not in label_dct:
            label_dct[w] = label_dct["_unknown_"]

    transform = torchvision.transforms.Compose([
        Pad(config["wav_size"]),
        MelSpectrogram(config["sr"], config["n_fft"], config["hop_length"],
                       config["n_mels"], config["fmin"], config["fmax"],
                       config["delta_order"], stack=True),
        Rescale()])
    print("Initializing Datasets (Checking/Creating disk cache)...")
    train_dataset = SpeechCommandsDataset(config["data_root"], label_dct, mode="train",
                                          transform=transform, cache_root=config["cache_root"])
    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              num_workers=config["num_workers"], sampler=train_sampler,
                              collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    valid_dataset = SpeechCommandsDataset(config["data_root"], label_dct, mode="valid",
                                          transform=transform, cache_root=config["cache_root"])
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"],
                              shuffle=False, num_workers=config["num_workers"],
                              collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader


def main():
    set_seed(DEFAULT_CONFIG["seed"]); device = Config.device
    train_loader, valid_loader = build_loaders(DEFAULT_CONFIG)
    combinations = list(HYPERPARAM_SEARCH)
    print(f"Total configurations: {len(combinations)}")
    best_overall_acc, best_config = 0.0, None
    for idx, params in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Config: {params}")
        val_acc = run_experiment(params, train_loader, valid_loader, device)
        print(f"  -> best_val_acc={val_acc:.4f}")
        if val_acc > best_overall_acc:
            best_overall_acc, best_config = val_acc, params
    print(f"\nAblation A finished. Best Accuracy: {best_overall_acc:.4f} with {best_config}")


if __name__ == "__main__":
    main()
