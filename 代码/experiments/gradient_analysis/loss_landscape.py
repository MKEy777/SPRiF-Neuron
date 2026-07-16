"""
Loss Landscape visualization: 3D surface + 2D contour.
Compares SPRiF vs LIF on S-MNIST.

Method: Li et al. "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018).

Produces:
  loss_landscape_sprif_3d.png
  loss_landscape_lif_3d.png
  loss_landscape_contour_comparison.png

Usage:
    cd 代码/experiments
    python gradient_analysis/loss_landscape.py
"""

import glob
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_DIR = os.path.join(ROOT, "Task_S-MNIST")
sys.path.insert(0, TASK_DIR)

FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "gradient_analysis",
)
os.makedirs(FIGURE_DIR, exist_ok=True)

from model import SequentialMNIST, SPRiFSMNISTNet
from model_lif import LIFSMNISTNet


def find_checkpoint(task_path, prefix):
    pattern = os.path.join(task_path, f"{prefix}_*.pth")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_model_for_landscape(model_type, checkpoint_path, device):
    if model_type == "sprif":
        model = SPRiFSMNISTNet(input_size=1, hidden_sizes=[64, 256]).to(device)
    else:
        model = LIFSMNISTNet(input_size=1, hidden_sizes=[64, 256]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_data_subset(device, n=256):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = torchvision.datasets.MNIST(root=os.path.join(TASK_DIR, "data"), train=True, download=True, transform=transform)
    dataset = SequentialMNIST(train_mnist)
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    x, y = next(iter(loader))
    return x[:n].to(device), y[:n].to(device)


def filter_normalize_directions(model):
    directions = []
    for _ in range(2):
        dir_vec = []
        for p in model.parameters():
            d = torch.randn_like(p)
            scale = p.data.norm() / (d.norm() + 1e-8)
            dir_vec.append(d * scale)
        directions.append(dir_vec)
    return directions


def get_params_vector(model):
    return [p.data.clone() for p in model.parameters()]


def set_params(model, params):
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def compute_loss(model, x, y, criterion):
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
    return loss.item()


def eval_grid(model, directions, x, y, criterion, grid_size=15, radius=1.5):
    alphas = np.linspace(-radius, radius, grid_size)
    betas = np.linspace(-radius, radius, grid_size)
    base_params = get_params_vector(model)
    loss_grid = np.full((grid_size, grid_size), np.nan)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_params = []
            for bp, d0, d1 in zip(base_params, directions[0], directions[1]):
                new_params.append(bp + a * d0 + b * d1)
            set_params(model, new_params)
            loss_grid[i, j] = compute_loss(model, x, y, criterion)

    set_params(model, base_params)
    return alphas, betas, loss_grid


def plot_3d_surface(alphas, betas, loss_grid, title, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    A, B = np.meshgrid(alphas, betas)
    ax.plot_surface(A, B, loss_grid, cmap="viridis", alpha=0.9, edgecolor="none")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_contour_comparison(
    sprif_alpha, sprif_beta, sprif_loss,
    lif_alpha, lif_beta, lif_loss,
    save_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, alphas, betas, loss_grid, title, cmap in [
        (axes[0], sprif_alpha, sprif_beta, sprif_loss, "SPRiF", "viridis"),
        (axes[1], lif_alpha, lif_beta, lif_loss, "LIF", "plasma"),
    ]:
        A, B = np.meshgrid(alphas, betas)
        contour = ax.contourf(A, B, loss_grid, levels=20, cmap=cmap)
        ax.contour(A, B, loss_grid, levels=10, colors="white", linewidths=0.5, alpha=0.5)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_title(title)
        plt.colorbar(contour, ax=ax)

    fig.suptitle("Loss Landscape Contour (S-MNIST)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sprif_ckpt = find_checkpoint(TASK_DIR, "SPRiFSMNISTNet")
    lif_ckpt = find_checkpoint(TASK_DIR, "LIFSMNISTNet")

    x, y = load_data_subset(device, n=256)
    criterion = nn.CrossEntropyLoss()

    sa = sb = sl = None
    la = lb = ll = None

    if sprif_ckpt:
        print(f"Loading SPRiF from {sprif_ckpt}")
        model_s = load_model_for_landscape("sprif", sprif_ckpt, device)
        dirs_s = filter_normalize_directions(model_s)
        print("Evaluating SPRiF loss landscape grid...")
        sa, sb, sl = eval_grid(model_s, dirs_s, x, y, criterion, grid_size=15, radius=1.5)
        plot_3d_surface(sa, sb, sl, "SPRiF Loss Landscape", os.path.join(FIGURE_DIR, "loss_landscape_sprif_3d.png"))

    if lif_ckpt:
        print(f"Loading LIF from {lif_ckpt}")
        model_l = load_model_for_landscape("lif", lif_ckpt, device)
        dirs_l = filter_normalize_directions(model_l)
        print("Evaluating LIF loss landscape grid...")
        la, lb, ll = eval_grid(model_l, dirs_l, x, y, criterion, grid_size=15, radius=1.5)
        plot_3d_surface(la, lb, ll, "LIF Loss Landscape", os.path.join(FIGURE_DIR, "loss_landscape_lif_3d.png"))

    if sprif_ckpt and lif_ckpt:
        plot_contour_comparison(sa, sb, sl, la, lb, ll, os.path.join(FIGURE_DIR, "loss_landscape_contour_comparison.png"))

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
