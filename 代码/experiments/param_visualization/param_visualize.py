"""
SPRiF 学习参数可视化 — 跨任务对比 α, ρ, ω 分布。

遍历已训练 SPRiF 模型（ECG/GSC/pSMNIST），提取谱参数，
画 1×3 KDE 直方图。有已保存模型则加载，无则训练。

Usage:
    cd 代码/experiments
    python param_visualization/param_visualize.py
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
sns.set_context("paper", font_scale=1.3)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "param_visualization",
)


def _add_path(task_dir: str) -> str:
    """Add task directory to sys.path, return its absolute path."""
    p = os.path.join(ROOT, task_dir)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for name in list(sys.modules):
        if name in {"model", "data", "core_algorithm"} or name.startswith("core_algorithm."):
            sys.modules.pop(name, None)
    return p


def _find_checkpoint(task_dir: str, class_prefix: str) -> Optional[str]:
    """Find the .pth file with highest accuracy in *task_dir*."""
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


# ---------------------------------------------------------------------------
# Training fallback
# ---------------------------------------------------------------------------

def _train_task(
    task_dir: str,
    train_script: str,
    extra_args: List[str],
):
    """Run a task's train.py via subprocess from within its directory."""
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
# Model loading per task
# ---------------------------------------------------------------------------

def _load_ecg(task_dir: str) -> Tuple[nn.Module, str]:
    """Load or train SPRiFECGModel."""
    import scipy.io

    _add_path(task_dir)
    from core_algorithm.utils import convert_dataset_wtime
    from model import SPRiFECGModel

    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFECGModel")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
        train_mat_path = os.path.join(ROOT, task_dir, "data", "QTDB_train.mat")
        if not os.path.exists(train_mat_path):
            raise FileNotFoundError(
                f"ECG data not found at {train_mat_path}. "
                "Place QTDB_train.mat / QTDB_test.mat in Task_ECG/data/"
            )
        mat = scipy.io.loadmat(train_mat_path)
        _, train_x, _ = convert_dataset_wtime(mat)
        input_size = train_x.shape[2]

        model = SPRiFECGModel(
            input_size=input_size,
            hidden_sizes=[36],
            output_size=6,
            mode="srnn",
            neuron_kwargs={"threshold": 0.6, "init_std": 0.05, "bias": True},
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return model, ckpt

    print("  No checkpoint found. Training ECG model...")
    train_mat = os.path.join(ROOT, task_dir, "data", "QTDB_train.mat")
    test_mat = os.path.join(ROOT, task_dir, "data", "QTDB_test.mat")
    if not os.path.exists(train_mat):
        raise FileNotFoundError(f"ECG training data not found at {train_mat}")
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
    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFECGModel")
    if ckpt is None:
        raise RuntimeError("Training completed but no checkpoint found.")
    return _load_ecg(task_dir)


def _load_gsc(task_dir: str) -> Tuple[nn.Module, str]:
    """Load or train SPRiFGSCNet."""
    _add_path(task_dir)
    from model import SPRiFGSCNet

    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFGSCNet")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
        model = SPRiFGSCNet(
            input_size=120,
            hidden_sizes=[300],
            num_classes=12,
            dropout=0.0,
            recurrent_flags=(True,),
            neuron_kwargs={"threshold": 1.0, "init_std": 0.1},
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return model, ckpt

    print("  No checkpoint found. Training GSC model...")
    data_root = os.path.join(ROOT, task_dir, "data", "SpeechCommands")
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"GSC data not found at {data_root}. "
            "Run Task_GSC/download_GSC.py first."
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
    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFGSCNet")
    if ckpt is None:
        raise RuntimeError("Training completed but no checkpoint found.")
    return _load_gsc(task_dir)


def _load_psmnist(task_dir: str) -> Tuple[nn.Module, str]:
    """Load or train SPRiFpSMNISTNet."""
    _add_path(task_dir)
    from model import SPRiFpSMNISTNet

    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFpSMNISTNet")
    if ckpt is not None:
        print(f"  Found checkpoint: {os.path.basename(ckpt)}")
        model = SPRiFpSMNISTNet(
            input_size=1,
            hidden_sizes=[64, 256],
            num_classes=10,
            mode="srnn",
            warmup_steps=0,
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return model, ckpt

    print("  No checkpoint found. Training pSMNIST model...")
    _train_task(
        task_dir, "train.py",
        [
            "--lr", "1e-2", "--epochs", "150", "--batch-size", "512",
            "--seed", "0", "--hidden-sizes", "64", "256", "--mode", "srnn",
        ],
    )
    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFpSMNISTNet")
    if ckpt is None:
        raise RuntimeError("Training completed but no checkpoint found.")
    return _load_psmnist(task_dir)


# ---------------------------------------------------------------------------
# Spectral parameter extraction
# ---------------------------------------------------------------------------

def extract_params(model: nn.Module) -> pd.DataFrame:
    """Extract alpha, rho, omega from all SPRiFNeuronLayer layers."""
    records = []
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model has no '.layers' attribute.")

    for li, layer in enumerate(layers):
        if not hasattr(layer, "get_spectral_parameters"):
            continue
        params = layer.get_spectral_parameters()
        alpha = params["alpha"].detach().cpu().numpy()
        rho = params["rho"].detach().cpu().numpy()
        omega = params["omega"].detach().cpu().numpy()
        for ni in range(len(alpha)):
            records.append({
                "layer": li,
                "neuron": ni,
                "alpha": alpha[ni],
                "rho": rho[ni],
                "omega": omega[ni],
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """1x3 KDE histogram comparing alpha, rho, omega across tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    param_info = [
        ("alpha", r"$\alpha$ (real decay)", axes[0]),
        ("rho", r"$\rho$ (rotation decay)", axes[1]),
        ("omega", r"$\omega$ (radians)", axes[2]),
    ]

    palette = {"ECG": "#2b83ba", "GSC": "#fdae61", "pSMNIST": "#d7191c"}

    for param, xlabel, ax in param_info:
        for task_name, df in dfs.items():
            values = df[param].dropna()
            if len(values) == 0:
                continue
            sns.histplot(
                values, ax=ax, kde=True, label=task_name,
                color=palette.get(task_name), alpha=0.35, bins=40,
                line_kws={"linewidth": 2},
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        if param == "alpha":
            ax.legend(frameon=True, fontsize=9)

    fig.suptitle("SPRiF Learned Spectral Parameters Across Tasks", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "param_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_csv(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Concatenate per-task DataFrames and save to CSV."""
    combined = []
    for task_name, df in dfs.items():
        df = df.copy()
        df.insert(0, "task", task_name)
        combined.append(df)
    full = pd.concat(combined, ignore_index=True)
    csv_path = os.path.join(out_dir, "param_per_task.csv")
    full.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Per-layer distribution plot
# ---------------------------------------------------------------------------

def plot_per_layer_distributions(dfs: Dict[str, pd.DataFrame], out_dir: str):
    """Per-layer KDE histograms of alpha, rho, omega.

    One row per (task, layer) combination, one column per spectral parameter.
    Only pSMNIST has multiple layers (L0, L1); GSC and ECG are single-layer.
    """
    # Build ordered list of (row_label, df_subset) for all (task, layer) pairs
    rows_data = []
    for task_name in ["pSMNIST", "GSC", "ECG"]:
        if task_name not in dfs:
            continue
        df = dfs[task_name]
        for layer_idx in sorted(df["layer"].unique()):
            label = f"{task_name}  L{layer_idx}"
            rows_data.append((label, df[df["layer"] == layer_idx]))

    n_rows = len(rows_data)
    if n_rows == 0:
        print("  No data for per-layer plot.")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    param_info = [
        ("alpha", r"$\alpha$ (real decay)"),
        ("rho", r"$\rho$ (rotation decay)"),
        ("omega", r"$\omega$ (radians)"),
    ]

    layer_colors = plt.cm.tab10(np.linspace(0, 1, n_rows))

    for row_idx, (label, sub_df) in enumerate(rows_data):
        for col_idx, (param, xlabel) in enumerate(param_info):
            ax = axes[row_idx, col_idx]
            values = sub_df[param].dropna()
            if len(values) == 0:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
                continue
            sns.histplot(
                values, ax=ax, kde=True, color=layer_colors[row_idx],
                alpha=0.5, bins=min(40, max(10, len(values) // 4)),
                line_kws={"linewidth": 1.8},
            )
            # Stats annotation
            mean_val = values.mean()
            std_val = values.std()
            ax.text(
                0.95, 0.92, f"μ={mean_val:.3f}\nσ={std_val:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        # Row label on leftmost column
        axes[row_idx, 0].set_ylabel(f"{label}\nCount", fontweight="bold")

    # Column titles
    for col_idx, (_, xlabel) in enumerate(param_info):
        axes[0, col_idx].set_title(xlabel, fontweight="bold")
        axes[-1, col_idx].set_xlabel(xlabel)

    fig.suptitle("SPRiF Learned Spectral Parameters — Per Layer", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "param_per_layer.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


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
    out_dir = FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}

    for task_name, (task_dir, loader_fn) in LOADERS.items():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        task_abs = os.path.join(ROOT, task_dir)
        _add_path(task_dir)
        try:
            model, ckpt_path = loader_fn(task_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        df = extract_params(model)
        print(f"  Extracted {len(df)} neurons ({df['layer'].nunique()} layers)")
        print(f"  alpha: [{df['alpha'].min():.4f}, {df['alpha'].mean():.4f}, {df['alpha'].max():.4f}]")
        print(f"  rho:   [{df['rho'].min():.4f}, {df['rho'].mean():.4f}, {df['rho'].max():.4f}]")
        print(f"  omega: [{df['omega'].min():.4f}, {df['omega'].mean():.4f}, {df['omega'].max():.4f}]")
        dfs[task_name] = df

    if not dfs:
        print("\nNo models loaded. Aborting.")
        return

    print(f"\n{'='*50}")
    print("Plotting...")
    plot_distributions(dfs, out_dir)
    plot_per_layer_distributions(dfs, out_dir)
    save_csv(dfs, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
