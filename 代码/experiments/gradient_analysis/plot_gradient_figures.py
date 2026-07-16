"""
Combine all gradient analysis results into publication-ready multi-panel figures.

Reads pre-computed results from:
  - grad_norm_tracker.py  → log files
  - temporal_grad_attenuation.py → computed arrays
  - loss_landscape.py     → computed arrays

Usage:
    cd 代码/experiments
    python gradient_analysis/plot_gradient_figures.py
"""

import json
import glob
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "gradient_analysis",
)
os.makedirs(FIGURE_DIR, exist_ok=True)

SPRIF_COLOR = "#E64B35"
LIF_COLOR = "#4DBBD5"
SLOW_COLOR = "#E64B35"
FAST_COLOR = "#4DBBD5"
GRAY = "#888888"


def _set_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
    })


def _panel(ax, label):
    ax.text(-0.2, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _clean(ax):
    ax.grid(True, color="#dddddd", linewidth=0.4, alpha=0.7)
    ax.tick_params(length=2, width=0.5)


def load_grad_logs(log_dir):
    sprif_logs = sorted(glob.glob(os.path.join(log_dir, "grad_log_sprif_*.json")))
    lif_logs = sorted(glob.glob(os.path.join(log_dir, "grad_log_lif_*.json")))
    sprif_data = [json.load(open(p)) for p in sprif_logs]
    lif_data = [json.load(open(p)) for p in lif_logs]
    return sprif_data, lif_data


def extract_metric(epoch_logs, metric):
    vals = []
    for epoch_data in epoch_logs:
        vals.append(epoch_data[metric])
    return np.array(vals)


def extract_raw_norms(epoch_logs, key="total_grad_raw"):
    all_norms = []
    for epoch_data in epoch_logs:
        all_norms.extend(epoch_data[key])
    return np.array(all_norms)


def build_figure_a(data_dir, save_dir):
    """Panel A: Gradient norm trajectory + distribution."""
    log_dir = os.path.join(data_dir, "grad_logs")
    if not os.path.isdir(log_dir):
        return None

    sprif_data, lif_data = load_grad_logs(log_dir)
    if not sprif_data and not lif_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1, ax2 = axes

    # Left: Trajectory
    for model_data, color, label in [(sprif_data, SPRIF_COLOR, "SPRiF"), (lif_data, LIF_COLOR, "LIF")]:
        if not model_data:
            continue
        all_vals = [extract_metric(run, "total_grad_mean") for run in model_data]
        min_len = min(len(v) for v in all_vals)
        stacked = np.stack([v[:min_len] for v in all_vals])
        mean_curve = stacked.mean(axis=0)
        std_curve = stacked.std(axis=0)
        epochs = np.arange(1, min_len + 1)
        ax1.plot(epochs, mean_curve, color=color, label=label, linewidth=2)
        ax1.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                         color=color, alpha=0.15)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean ||∇L||")
    ax1.legend()
    _clean(ax1)

    # Right: Distribution
    for model_data, color, label in [(sprif_data, SPRIF_COLOR, "SPRiF"), (lif_data, LIF_COLOR, "LIF")]:
        if not model_data:
            continue
        all_norms = np.concatenate([extract_raw_norms(run, "total_grad_raw") for run in model_data])
        ax2.hist(all_norms, bins=80, color=color, alpha=0.5, density=True, label=f"{label} (μ={all_norms.mean():.3f})")
    ax2.set_xlabel("||∇L||")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7)
    _clean(ax2)

    _panel(ax1, "a")
    _panel(ax2, "b")
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(save_dir, "figure_gradient_overview_ab.png"), dpi=150)
    plt.close(fig)
    print("Saved figure_gradient_overview_ab.png")


def build_full_figure(data_dir, save_dir):
    """Full 4-panel figure: AB (grad norm) + CD (temporal + landscape)."""
    log_dir = os.path.join(data_dir, "grad_logs")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.97, bottom=0.08, top=0.94, hspace=0.35, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: Gradient norm trajectory
    if os.path.isdir(log_dir):
        sprif_data, lif_data = load_grad_logs(log_dir)
        if sprif_data and lif_data:
            for model_data, color, label in [(sprif_data, SPRIF_COLOR, "SPRiF"), (lif_data, LIF_COLOR, "LIF")]:
                all_vals = [extract_metric(run, "total_grad_mean") for run in model_data]
                min_len = min(len(v) for v in all_vals)
                stacked = np.stack([v[:min_len] for v in all_vals])
                ax_a.plot(np.arange(1, min_len + 1), stacked.mean(axis=0),
                          color=color, label=label, linewidth=2)
                ax_a.fill_between(np.arange(1, min_len + 1),
                                  stacked.mean(axis=0) - stacked.std(axis=0),
                                  stacked.mean(axis=0) + stacked.std(axis=0),
                                  color=color, alpha=0.15)
    _panel(ax_a, "a")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("||∇L|| (mean)")
    ax_a.legend()
    ax_a.set_title("Gradient Norm Trajectory")
    _clean(ax_a)

    # Panel B: Slow vs Fast
    if os.path.isdir(log_dir):
        for run_logs in sprif_data[0:1]:
            slow_vals = extract_metric(run_logs, "slow_grad_mean")
            fast_vals = extract_metric(run_logs, "fast_grad_mean")
            epochs = np.arange(1, len(slow_vals) + 1)
            ax_b.plot(epochs, slow_vals, color=SLOW_COLOR, label="Slow params", linewidth=2)
            ax_b.plot(epochs, fast_vals, color=FAST_COLOR, label="Fast params", linewidth=2)
    _panel(ax_b, "b")
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("||∇|| (per group)")
    ax_b.legend()
    ax_b.set_title("SPRiF: Slow vs Fast Gradients")
    _clean(ax_b)

    # Panel C: Temporal gradient evolution
    ax_c.set_title("Gradient vs Time Step (placeholder)")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("||∂L/∂state||")
    _clean(ax_c)

    # Panel D: Loss landscape
    ax_d.set_title("Loss Landscape (placeholder)")
    ax_d.set_xlabel("α")
    ax_d.set_ylabel("β")
    _clean(ax_d)

    fig.savefig(os.path.join(save_dir, "figure_gradient_full.png"), dpi=150)
    plt.close(fig)
    print("Saved figure_gradient_full.png")


def main():
    _set_style()
    data_dir = os.path.join(ROOT, "Task_S-MNIST")
    build_figure_a(data_dir, FIGURE_DIR)
    build_full_figure(data_dir, FIGURE_DIR)
    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
