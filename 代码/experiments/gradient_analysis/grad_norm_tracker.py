"""
Exp 1-3: Gradient norm analysis from train_gradient_monitor.py logs.

Produces:
  Exp 1 — grad_norm_trajectory.png   (total grad norm vs epoch, SPRiF vs LIF)
  Exp 2 — grad_norm_distribution.png (histogram/KDE of all per-chunk grad norms)
  Exp 3 — sprif_slow_vs_fast.png     (slow-state vs fast-state grad norms)

Usage:
    cd 代码/experiments
    python gradient_analysis/grad_norm_tracker.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "gradient_analysis",
)
os.makedirs(FIGURE_DIR, exist_ok=True)


def find_logs(log_dir):
    sprif_logs = sorted(glob.glob(os.path.join(log_dir, "grad_log_sprif_*.json")))
    lif_logs = sorted(glob.glob(os.path.join(log_dir, "grad_log_lif_*.json")))
    return sprif_logs, lif_logs


def load_logs(paths):
    data = []
    for p in paths:
        with open(p) as f:
            data.append(json.load(f))
    return data


def extract_metric(epoch_logs, metric):
    """Extract per-epoch metric (mean across chunks)."""
    vals = []
    for epoch_data in epoch_logs:
        if metric in epoch_data:
            vals.append(epoch_data[metric])
        elif metric in epoch_data[0]:
            vals.append(epoch_data[0][metric])
    return np.array(vals)


def extract_raw_norms(epoch_logs, key="total_grad_raw"):
    """Extract all per-chunk grad norms across all epochs."""
    all_norms = []
    for epoch_data in epoch_logs:
        if key in epoch_data:
            all_norms.extend(epoch_data[key])
        elif "total_grad_raw" in epoch_data[0]:
            all_norms.extend(epoch_data[0][key])
    return np.array(all_norms)


def plot_exp1(sprif_data, lif_data, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    colors = {"sprif": "#E64B35", "lif": "#4DBBD5"}

    for ax, metric, ylabel in [
        (axes[0], "total_grad_mean", "Mean ||∇L|| per epoch"),
        (axes[1], "total_grad_std", "Std ||∇L|| per epoch"),
    ]:
        for model_name, all_logs in [("sprif", sprif_data), ("lif", lif_data)]:
            for run_logs in all_logs:
                vals = extract_metric(run_logs, metric)
                epochs = np.arange(1, len(vals) + 1)
                label = "SPRiF" if model_name == "sprif" else "LIF"
                ax.plot(epochs, vals, color=colors[model_name], label=label if id(run_logs) == id(all_logs[0]) else "", linewidth=2)

        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Exp 1: Gradient Norm Trajectory During Training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, "grad_norm_trajectory.png"), dpi=150)
    plt.close(fig)
    print(f"Saved grad_norm_trajectory.png")


def plot_exp2(sprif_data, lif_data, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, (model_name, all_logs, color) in enumerate([
        ("sprif", sprif_data, "#E64B35"),
        ("lif", lif_data, "#4DBBD5"),
    ]):
        all_norms = np.array(extract_raw_norms(all_logs[0], "total_grad_raw"))

        ax = axes[idx]
        ax.hist(all_norms, bins=80, color=color, alpha=0.7, density=True)
        ax.set_xlabel("||∇L||")
        ax.set_ylabel("Density")
        ax.set_title(f"{'SPRiF' if model_name == 'sprif' else 'LIF'}\n"
                     f"Mean={all_norms.mean():.4f}  Std={all_norms.std():.4f}  "
                     f"P99={np.percentile(all_norms, 99):.4f}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: Gradient Norm Distribution (all per-chunk norms)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(save_dir, "grad_norm_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved grad_norm_distribution.png")


def plot_exp3(sprif_data, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    run_logs = sprif_data[0]
    slow_vals = extract_metric(run_logs, "slow_grad_mean")
    fast_vals = extract_metric(run_logs, "fast_grad_mean")
    epochs = np.arange(1, len(slow_vals) + 1)
    axes[0].plot(epochs, slow_vals, label="Slow-state params", color="#E64B35", linewidth=2)
    axes[0].plot(epochs, fast_vals, label="Fast-state params", color="#4DBBD5", linewidth=2)

    axes[0].set_ylabel("Mean ||∇|| per epoch")
    axes[0].legend(fontsize=10)
    axes[0].set_title("SPRiF: Slow-state vs Fast-state Gradient Norms")
    axes[0].grid(True, alpha=0.3)

    ratio = np.array(slow_vals) / (np.array(fast_vals) + 1e-8)
    axes[1].plot(epochs, ratio, color="#7E6148", linewidth=2)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_ylabel("Slow / Fast Ratio")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exp 3: SPRiF Internal Gradient Decomposition", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "sprif_slow_vs_fast.png"), dpi=150)
    plt.close(fig)
    print(f"Saved sprif_slow_vs_fast.png")


def main():
    log_dir = os.path.join(ROOT, "Task_S-MNIST", "grad_logs")
    if not os.path.isdir(log_dir):
        print(f"Gradient log directory not found: {log_dir}")
        print("Run train_gradient_monitor.py first to generate logs.")
        sys.exit(1)

    sprif_paths, lif_paths = find_logs(log_dir)
    print(f"Found {len(sprif_paths)} SPRiF logs, {len(lif_paths)} LIF logs")

    if len(sprif_paths) == 0 and len(lif_paths) == 0:
        print("No gradient logs found. Run train_gradient_monitor.py first.")
        sys.exit(1)

    sprif_data = load_logs(sprif_paths)
    lif_data = load_logs(lif_paths)

    if sprif_data and lif_data:
        plot_exp1(sprif_data, lif_data, FIGURE_DIR)

    if sprif_data and lif_data:
        plot_exp2(sprif_data, lif_data, FIGURE_DIR)

    if sprif_data:
        plot_exp3(sprif_data, FIGURE_DIR)

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
