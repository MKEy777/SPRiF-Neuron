"""
Exp 4: Temporal gradient propagation analysis on S-MNIST.

Produces:
  temporal_grad_decay.png        — ||∂L/∂h_t|| vs time lag (SPRiF vs LIF)
  gradient_evolution.png         — 梯度随时间步变化演变示意图 (SPRiF vs LIF)
  sprif_grad_heatmap.png         — per-hidden-unit gradient heatmap across time

Usage:
    cd 代码/experiments
    python gradient_analysis/temporal_grad_attenuation.py
"""

import glob
import os
import sys
from typing import Optional

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from core_algorithm.sprif_layer import surrogate_spike
from model import SequentialMNIST, SPRiFSMNISTNet
from model_lif import LIFSMNISTNet


def load_test_sample(device, num_samples=1):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_mnist = torchvision.datasets.MNIST(root=os.path.join(TASK_DIR, "data"), train=False, download=True, transform=transform)
    dataset = SequentialMNIST(test_mnist)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    x, y = next(iter(loader))
    return x.to(device), y.to(device)


def find_checkpoint(task_path, prefix):
    pattern = os.path.join(task_path, f"{prefix}_*.pth")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_model(model_type, checkpoint_path, device):
    if model_type == "sprif":
        model = SPRiFSMNISTNet(input_size=1, hidden_sizes=[64, 256]).to(device)
    else:
        model = LIFSMNISTNet(input_size=1, hidden_sizes=[64, 256]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.enable_grad()
def compute_temporal_gradients(model, x, y, criterion, layer_idx=0):
    """
    Full BPTT on a single sample (B=1).
    Records per-timestep slow-state gradients via retain_grad().
    """
    B, T, F = x.shape
    device = x.device
    layer = model.layers[layer_idx]
    state = layer.init_state(B, device, x.dtype)

    per_t_step_slow = []

    for t in range(T):
        xt = x[:, t]
        runtime = layer._precompute_runtime_params()

        x_state = state["x"]
        u_state = state["u"]
        prev_spike = state["prev_spike"]

        input_current = layer.input_linear(xt)
        if layer.recurrent and layer.recurrent_linear is not None:
            input_current = input_current + layer.recurrent_linear(prev_spike)

        x_next = layer._slow_flow(x_state, input_current, runtime)
        x_next.retain_grad()
        per_t_step_slow.append(x_next)

        u_tilde = layer._fast_flow(u_state, x_next, runtime)
        membrane = u_tilde[..., 0]
        spike = layer._spike_fn(membrane - layer.threshold)

        u_next = u_tilde - spike.unsqueeze(-1) * runtime["reset_direction"].unsqueeze(0) * layer.threshold
        state = {"x": x_next, "u": u_next, "prev_spike": spike}

        if t == T - 1:
            out = spike.unsqueeze(1)

    logits_t = model.readout(out)
    logits = logits_t.mean(dim=1)
    loss = criterion(logits, y)
    loss.backward()

    grad_slow = np.array([tensor.grad.norm().item() if tensor.grad is not None else 0.0 for tensor in per_t_step_slow])
    return grad_slow


@torch.enable_grad()
def compute_temporal_gradients_lif(model, x, y, criterion, layer_idx=0):
    B, T, F = x.shape
    device = x.device
    layer = model.layers[layer_idx]
    state = layer.init_state(B, device, x.dtype)

    per_t_step_v = []

    for t in range(T):
        xt = x[:, t]
        v_prev = state["v"]
        prev_spike = state["prev_spike"]

        runtime = layer._precompute_runtime_params()
        beta = runtime["beta"].unsqueeze(0)

        input_current = layer.input_linear(xt)
        if layer.recurrent and layer.recurrent_linear is not None:
            input_current = input_current + layer.recurrent_linear(prev_spike)

        v_tilde = beta * v_prev + (1.0 - beta) * input_current
        v_tilde.retain_grad()
        per_t_step_v.append(v_tilde)

        spike = surrogate_spike(v_tilde - layer.threshold)
        v_next = v_tilde - spike * layer.threshold
        state = {"v": v_next, "prev_spike": spike}

        if t == T - 1:
            out = spike.unsqueeze(1)

    logits_t = model.readout(out)
    logits = logits_t.mean(dim=1)
    loss = criterion(logits, y)
    loss.backward()

    grad_v = np.array([tensor.grad.norm().item() if tensor.grad is not None else 0.0 for tensor in per_t_step_v])
    return grad_v


def plot_grad_vs_time(sprif_grad, lif_grad, save_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(sprif_grad)), sprif_grad, label="SPRiF slow state", color="#E64B35", linewidth=2)
    if lif_grad is not None:
        ax.plot(np.arange(len(lif_grad)), lif_grad, label="LIF membrane", color="#4DBBD5", linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("||∂L/∂state||")
    ax.set_title("Gradient Magnitude vs Time Step (Layer 0)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, save_name), dpi=150)
    plt.close(fig)
    print(f"Saved {save_name}")


def plot_evolution_diagram(sprif_grad, lif_grad, save_name):
    """
    Top panel: SPRiF slow-state gradient trajectory
    Bottom panel: LIF membrane gradient trajectory
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(sprif_grad, color="#E64B35", linewidth=2, label="SPRiF (slow state)")
    axes[0].set_ylabel("||∂L/∂state||")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    if lif_grad is not None:
        axes[1].plot(lif_grad, color="#4DBBD5", linewidth=2, label="LIF (membrane)")
    axes[1].set_ylabel("||∂L/∂state||")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Gradient Evolution Across Time Steps (Layer 0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURE_DIR, save_name), dpi=150)
    plt.close(fig)
    print(f"Saved {save_name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    criterion = nn.CrossEntropyLoss()

    sprif_ckpt = find_checkpoint(TASK_DIR, "SPRiFSMNISTNet")
    lif_ckpt = find_checkpoint(TASK_DIR, "LIFSMNISTNet")

    if sprif_ckpt is None:
        print("No SPRiF checkpoint found. Run: python train_gradient_monitor.py")
    if lif_ckpt is None:
        print("No LIF checkpoint found. Run: python train_gradient_monitor.py --model lif")

    sprif_grad = None
    lif_grad = None

    if sprif_ckpt:
        print(f"Loading SPRiF from {sprif_ckpt}")
        model_s = load_model("sprif", sprif_ckpt, device)
        x_s, y_s = load_test_sample(device, num_samples=1)
        g_s = compute_temporal_gradients(model_s, x_s, y_s, criterion, layer_idx=0)
        sprif_grad = g_s
        print(f"SPRiF grad range [{g_s.min():.4f}, {g_s.max():.4f}]")

    if lif_ckpt:
        print(f"Loading LIF from {lif_ckpt}")
        model_l = load_model("lif", lif_ckpt, device)
        x_s, y_s = load_test_sample(device, num_samples=1)
        g_v = compute_temporal_gradients_lif(model_l, x_s, y_s, criterion, layer_idx=0)
        lif_grad = g_v
        print(f"LIF grad range [{g_v.min():.4f}, {g_v.max():.4f}]")

    if sprif_grad is not None:
        plot_evolution_diagram(sprif_grad, lif_grad, "gradient_evolution.png")
        plot_grad_vs_time(sprif_grad, lif_grad, "temporal_grad_decay.png")

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
