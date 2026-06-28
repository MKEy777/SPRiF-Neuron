"""
SPRiF Reset Direction Analysis — learned λ_j distribution and functional correlates.

Extracts learnable lambda_reset (projective reset direction) from trained SPRiF
neurons, computes per-neuron firing rates on test data, and analyzes correlations
between reset strategy and spectral parameters / firing behavior.

Usage:
    cd 代码/experiments
    python reset_analysis/reset_analyze.py
"""

import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _add_path(task_dir: str) -> str:
    p = os.path.join(ROOT, task_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def _find_checkpoint(task_dir: str, class_prefix: str) -> Optional[str]:
    pattern = os.path.join(task_dir, f"{class_prefix}_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    best = None
    best_acc = -1.0
    for f in files:
        base = os.path.basename(f)
        try:
            acc_str = base.rsplit("_acc", 1)[1].replace(".pth", "")
            acc = float(acc_str)
        except (ValueError, IndexError):
            acc = 0.0
        if acc > best_acc:
            best_acc = acc
            best = f
    return best


def _train_task(task_dir: str, train_script: str, extra_args: List[str]):
    cwd = os.path.join(ROOT, task_dir)
    script = os.path.join(cwd, train_script)
    if not os.path.exists(script):
        raise FileNotFoundError(f"Training script not found: {script}")
    cmd = [sys.executable, script] + extra_args
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Working dir: {cwd}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Model loading + test DataLoader per task
# ---------------------------------------------------------------------------

def _load_psmnist(task_dir: str) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
    """Load pSMNIST model and return (model, test_loader)."""
    import torchvision
    import torchvision.transforms as transforms
    from model import SPRiFpSMNISTNet

    task_abs = os.path.join(ROOT, task_dir)

    ckpt = _find_checkpoint(task_abs, "SPRiFpSMNISTNet")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
    else:
        print("  No checkpoint found. Training pSMNIST model...")
        _train_task(
            task_dir, "train.py",
            [
                "--lr", "1e-2", "--epochs", "150", "--batch-size", "512",
                "--seed", "0", "--hidden-sizes", "64", "256", "--mode", "srnn",
            ],
        )
        ckpt = _find_checkpoint(task_abs, "SPRiFpSMNISTNet")
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found.")

    model = SPRiFpSMNISTNet(
        input_size=1, hidden_sizes=[64, 256], num_classes=10,
        mode="srnn", warmup_steps=0,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    # Build test DataLoader — permutation must match training seed (0)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_mnist = torchvision.datasets.MNIST(
        root=os.path.join(task_abs, "data"), train=False,
        download=True, transform=transform,
    )
    torch.manual_seed(0)
    perm = torch.randperm(784)
    from model import PermutedMNIST
    test_dataset = PermutedMNIST(test_mnist, perm)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2,
    )
    return model, test_loader


def _load_gsc(task_dir: str) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
    """Load GSC model and return (model, test_loader)."""
    import torchvision
    from torch.utils.data import DataLoader
    from model import SPRiFGSCNet

    task_abs = os.path.join(ROOT, task_dir)
    data_root = os.path.join(task_abs, "data", "SpeechCommands")

    ckpt = _find_checkpoint(task_abs, "SPRiFGSCNet")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
    else:
        print("  No checkpoint found. Training GSC model...")
        if not os.path.exists(data_root):
            raise FileNotFoundError(
                f"GSC data not found at {data_root}. Run Task_GSC/download_GSC.py first."
            )
        _train_task(
            task_dir, "train.py",
            [
                "--data-root", os.path.join("data", "SpeechCommands"),
                "--lr", "3e-3", "--epochs", "150", "--batch-size", "200",
                "--seed", "42", "--hidden-sizes", "300",
                "--neuron-threshold", "1.0", "--neuron-init-std", "0.1",
            ],
        )
        ckpt = _find_checkpoint(task_abs, "SPRiFGSCNet")
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found.")

    model = SPRiFGSCNet(
        input_size=120, hidden_sizes=[300], num_classes=12,
        dropout=0.0, recurrent_flags=(True,),
        neuron_kwargs={"threshold": 1.0, "init_std": 0.1},
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    # Build test DataLoader — mirror train.py build_loaders
    from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

    testing_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}
    for w in os.listdir(data_root):
        full = os.path.join(data_root, w)
        if os.path.isdir(full) and w[0] != "_" and w not in label_dct:
            label_dct[w] = label_dct["_unknown_"]

    sr = 16000
    n_fft = int(30e-3 * sr)
    hop_length = int(10e-3 * sr)

    transform = torchvision.transforms.Compose([
        Pad(16000),
        MelSpectrogram(sr, n_fft, hop_length, 40, 20, 4000, 2, stack=True),
        Rescale(),
    ])

    def collate_fn(data):
        x_batch = np.array([d[0] for d in data])
        std = x_batch.std(axis=(0, 2), keepdims=True)
        std[std == 0] = 1.0
        return torch.tensor(x_batch / std).float(), torch.tensor([d[1] for d in data]).long()

    test_dataset = SpeechCommandsDataset(
        data_root, label_dct, mode="test",
        transform=transform, cache_root=data_root,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    return model, test_loader


def _load_ecg(task_dir: str) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
    """Load ECG model and return (model, test_loader)."""
    import scipy.io
    from torch.utils.data import DataLoader, TensorDataset
    from core_algorithm.utils import convert_dataset_wtime
    from model import SPRiFECGModel

    task_abs = os.path.join(ROOT, task_dir)

    ckpt = _find_checkpoint(task_abs, "SPRiFECGModel")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
        # Need data to determine input_size even when loading checkpoint
        train_mat_path = os.path.join(task_abs, "data", "QTDB_train.mat")
        if not os.path.exists(train_mat_path):
            raise FileNotFoundError(
                f"ECG data not found at {train_mat_path}. "
                "Place QTDB_train.mat / QTDB_test.mat in Task_ECG/data/"
            )
        mat = scipy.io.loadmat(train_mat_path)
        _, train_x, _ = convert_dataset_wtime(mat)
        input_size = train_x.shape[2]

        model = SPRiFECGModel(
            input_size=input_size, hidden_sizes=[36], output_size=6,
            mode="srnn",
            neuron_kwargs={"threshold": 0.6, "init_std": 0.05, "bias": True},
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    else:
        print("  No checkpoint found. Training ECG model...")
        _train_task(
            task_dir, "train.py",
            [
                "--train-mat", os.path.join("data", "QTDB_train.mat"),
                "--test-mat", os.path.join("data", "QTDB_test.mat"),
                "--lr", "1e-2", "--epochs", "250", "--batch-size", "64",
                "--seed", "1111", "--hidden-sizes", "36",
                "--neuron-threshold", "0.6",
            ],
        )
        ckpt = _find_checkpoint(task_abs, "SPRiFECGModel")
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found.")
        return _load_ecg(task_dir)  # recurse once to get model + loader

    # Build test DataLoader
    test_mat_path = os.path.join(task_abs, "data", "QTDB_test.mat")
    if not os.path.exists(test_mat_path):
        raise FileNotFoundError(f"ECG test data not found at {test_mat_path}")
    test_mat = scipy.io.loadmat(test_mat_path)
    _, test_x, test_y = convert_dataset_wtime(test_mat)

    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).long()
    train_mat = scipy.io.loadmat(os.path.join(task_abs, "data", "QTDB_train.mat"))
    _, _, train_y = convert_dataset_wtime(train_mat)
    train_y = torch.from_numpy(train_y).long()
    label_min = min(train_y.min().item(), test_y.min().item())
    if label_min != 0:
        test_y -= label_min

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=64, shuffle=False, num_workers=2,
    )
    return model, test_loader


# ---------------------------------------------------------------------------
# Firing rate computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_firing_rates(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 30,
) -> Dict[int, torch.Tensor]:
    """Compute per-neuron mean firing rate on test data.

    Returns:
        dict mapping layer_idx -> tensor of shape (hidden_size,) with
        mean spike count per neuron per timestep.
    """
    model.eval()
    spike_counts = [torch.zeros(layer.hidden_size) for layer in model.layers]
    total_steps = 0

    for batch_idx, (x, _) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        x = x.to(device)
        B, T = x.shape[0], x.shape[1]

        out = x
        for li, layer in enumerate(model.layers):
            state = layer.init_state(B, device=device)
            runtime = layer._precompute_runtime_params()
            layer_spikes = []
            for t in range(T):
                spike, _, state = layer.forward_step(out[:, t, :], state, runtime)
                layer_spikes.append(spike.detach().cpu())
            spike_seq = torch.stack(layer_spikes, dim=0)  # (T, B, H)
            spike_counts[li] += spike_seq.sum(dim=(0, 1))
            out = spike_seq.permute(1, 0, 2)  # (B, T, H)

        total_steps += B * T

    if total_steps == 0:
        return {li: torch.zeros(layer.hidden_size) for li, layer in enumerate(model.layers)}

    return {li: counts / total_steps for li, counts in enumerate(spike_counts)}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_and_merge(
    model: nn.Module,
    rates: Dict[int, torch.Tensor],
) -> pd.DataFrame:
    """Combine spectral params + lambda_reset + firing rates into one DataFrame."""
    records = []
    for li, layer in enumerate(model.layers):
        params = layer.get_spectral_parameters()
        alpha = params["alpha"].detach().cpu().numpy()
        rho = params["rho"].detach().cpu().numpy()
        omega = params["omega"].detach().cpu().numpy()
        eta = params["eta"].detach().cpu().numpy()             # (H, 2)
        lam = params["lambda_reset"].detach().cpu().numpy()     # (H,)
        fr = rates[li].numpy()

        for ni in range(len(alpha)):
            records.append({
                "layer": li,
                "neuron": ni,
                "alpha": alpha[ni],
                "rho": rho[ni],
                "omega": omega[ni],
                "eta_0": eta[ni, 0],
                "eta_1": eta[ni, 1],
                "lambda_reset": lam[ni],
                "firing_rate": fr[ni],
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

LAYER_MARKERS = {0: "o", 1: "s"}


def plot_lambda_distribution(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Histogram of lambda_reset per task, overlaid per layer."""
    tasks = [t for t in ["ECG", "GSC", "pSMNIST"] if t in dfs]
    if not tasks:
        return

    fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 3.5 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task_name in zip(axes, tasks):
        df = dfs[task_name]
        for li in sorted(df["layer"].unique()):
            sub = df[df["layer"] == li]
            values = sub["lambda_reset"].dropna()
            if len(values) == 0:
                continue
            sns.histplot(
                values, ax=ax, kde=True, bins=min(50, max(10, len(values) // 5)),
                label=f"Layer {li} (n={len(values)})",
                alpha=0.45, line_kws={"linewidth": 1.8},
            )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="λ=0")
        ax.set_xlabel(r"$\lambda$ (reset direction)")
        ax.set_ylabel("Count")
        ax.set_title(f"{task_name} — Learned Reset Direction Distribution", fontweight="bold")
        ax.legend(frameon=True, fontsize=9)

        # Print stats
        all_vals = df["lambda_reset"].dropna()
        print(f"  {task_name} λ: [{all_vals.min():.4f}, {all_vals.mean():.4f}, "
              f"{all_vals.max():.4f}], std={all_vals.std():.4f}, "
              f"frac<0={ (all_vals < 0).mean():.2%}")

    fig.suptitle("SPRiF Projective Reset Direction (λ) Distributions", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "lambda_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_lambda_vs_firing_rate(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Scatter: lambda_reset vs firing rate, per task, colored by layer."""
    tasks = [t for t in ["ECG", "GSC", "pSMNIST"] if t in dfs]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task_name in zip(axes, tasks):
        df = dfs[task_name]
        for li in sorted(df["layer"].unique()):
            sub = df[df["layer"] == li]
            ax.scatter(
                sub["lambda_reset"], sub["firing_rate"],
                alpha=0.5, s=25, marker=LAYER_MARKERS.get(li, "o"),
                label=f"Layer {li}", edgecolors="none",
            )
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\lambda$ (reset direction)")
        ax.set_ylabel("Firing rate (spikes / neuron / step)")
        ax.set_title(task_name, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

        # Correlation
        valid = df.dropna(subset=["lambda_reset", "firing_rate"])
        if len(valid) > 2:
            corr = np.corrcoef(valid["lambda_reset"], valid["firing_rate"])[0, 1]
            ax.text(0.95, 0.95, f"r={corr:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    fig.suptitle("Reset Direction vs Firing Rate", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "lambda_vs_firing_rate.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_lambda_vs_alpha(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Scatter: lambda_reset vs alpha, per task, colored by layer."""
    tasks = [t for t in ["ECG", "GSC", "pSMNIST"] if t in dfs]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task_name in zip(axes, tasks):
        df = dfs[task_name]
        for li in sorted(df["layer"].unique()):
            sub = df[df["layer"] == li]
            ax.scatter(
                sub["alpha"], sub["lambda_reset"],
                alpha=0.5, s=25, marker=LAYER_MARKERS.get(li, "o"),
                label=f"Layer {li}", edgecolors="none",
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\alpha$ (real decay)")
        ax.set_ylabel(r"$\lambda$ (reset direction)")
        ax.set_title(task_name, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

        valid = df.dropna(subset=["alpha", "lambda_reset"])
        if len(valid) > 2:
            corr = np.corrcoef(valid["alpha"], valid["lambda_reset"])[0, 1]
            ax.text(0.95, 0.95, f"r={corr:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    fig.suptitle("Reset Direction vs Real Decay (α)", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "lambda_vs_alpha.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_lambda_vs_omega(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Scatter: lambda_reset vs omega, per task, colored by layer."""
    tasks = [t for t in ["ECG", "GSC", "pSMNIST"] if t in dfs]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task_name in zip(axes, tasks):
        df = dfs[task_name]
        for li in sorted(df["layer"].unique()):
            sub = df[df["layer"] == li]
            ax.scatter(
                sub["omega"], sub["lambda_reset"],
                alpha=0.5, s=25, marker=LAYER_MARKERS.get(li, "o"),
                label=f"Layer {li}", edgecolors="none",
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\omega$ (radians)")
        ax.set_ylabel(r"$\lambda$ (reset direction)")
        ax.set_title(task_name, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

        valid = df.dropna(subset=["omega", "lambda_reset"])
        if len(valid) > 2:
            corr = np.corrcoef(valid["omega"], valid["lambda_reset"])[0, 1]
            ax.text(0.95, 0.95, f"r={corr:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    fig.suptitle("Reset Direction vs Rotation Frequency (ω)", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "lambda_vs_omega.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_lambda_stats(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Concatenate per-task DataFrames and save to CSV."""
    combined = []
    for task_name, df in dfs.items():
        df = df.copy()
        df.insert(0, "task", task_name)
        combined.append(df)
    full = pd.concat(combined, ignore_index=True)
    csv_path = os.path.join(out_dir, "lambda_stats.csv")
    full.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

LOADERS = {
    "ECG": ("Task_ECG", _load_ecg),
    "GSC": ("Task_GSC", _load_gsc),
    "pSMNIST": ("Task_pSMNIST", _load_psmnist),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dfs: Dict[str, pd.DataFrame] = {}

    for task_name, (task_dir, loader_fn) in LOADERS.items():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        _add_path(task_dir)

        try:
            model, test_loader = loader_fn(task_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        model.to(device)
        model.eval()

        # Compute per-neuron firing rates
        print(f"  Computing firing rates ({len(model.layers)} layer(s))...")
        rates = compute_firing_rates(model, test_loader, device, num_batches=30)

        for li, layer in enumerate(model.layers):
            fr = rates[li]
            print(f"    Layer {li}: firing rate [{fr.min():.4f}, {fr.mean():.4f}, {fr.max():.4f}]")

        # Extract params + lambda + firing rates
        df = extract_and_merge(model, rates)
        print(f"  Extracted {len(df)} neurons across {df['layer'].nunique()} layer(s)")
        dfs[task_name] = df

    if not dfs:
        print("\nNo models loaded. Aborting.")
        return

    print(f"\n{'='*50}")
    print("Plotting...")

    plot_lambda_distribution(dfs, out_dir)
    plot_lambda_vs_firing_rate(dfs, out_dir)
    plot_lambda_vs_alpha(dfs, out_dir)
    plot_lambda_vs_omega(dfs, out_dir)
    save_lambda_stats(dfs, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
