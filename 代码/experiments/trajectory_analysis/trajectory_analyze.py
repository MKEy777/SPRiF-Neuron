"""
SPRiF 慢状态轨迹分析 — 证明"脉冲不打断慢状态"。

加载 PS-MNIST 已训练 SPRiF 模型，选一个测试样本，
记录慢状态 x_t 和膜电位 u^0 的逐时间步轨迹，
展示慢状态在脉冲时刻连续、膜电位在脉冲时刻跳变。

Usage:
    cd 代码/experiments
    python trajectory_analysis/trajectory_analyze.py
"""

import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path / config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "trajectory_analysis",
)
TASK_DIR = "Task_pSMNIST"
TASK_ABS = os.path.join(ROOT, TASK_DIR)
sys.path.insert(0, TASK_ABS)

from core_algorithm.utils import set_seed


def _find_checkpoint(task_path: str, class_prefix: str) -> Optional[str]:
    pattern = os.path.join(task_path, f"{class_prefix}_*.pth")
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


def _train_psmnist() -> str:
    """Train pSMNIST and return checkpoint path."""
    print("  Training SPRiF on pSMNIST...")
    train_script = os.path.join(TASK_ABS, "train.py")
    cmd = [
        sys.executable, train_script,
        "--lr", "1e-2", "--epochs", "150", "--batch-size", "512",
        "--seed", "0", "--hidden-sizes", "64", "256", "--mode", "srnn",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=TASK_ABS, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")
    ckpt = _find_checkpoint(TASK_ABS, "SPRiFpSMNISTNet")
    if ckpt is None:
        raise RuntimeError("Training completed but no checkpoint found.")
    return ckpt


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------

def record_trajectory(
    model: nn.Module, x_single: torch.Tensor
) -> Dict[str, List[np.ndarray]]:
    """
    Record per-timestep states for a single sample.

    Args:
        model: SPRiFpSMNISTNet
        x_single: [1, T, F]  (batch_first, batch=1)

    Returns:
        slow_states: list of [T, H, 3] per layer
        membranes:   list of [T, H]    per layer
        spikes:      list of [T, H]    per layer
    """
    model.eval()
    T = x_single.size(1)
    device = x_single.device

    slow_list: List[np.ndarray] = []
    mem_list: List[np.ndarray] = []
    spike_list: List[np.ndarray] = []

    out = x_single  # [1, T, F]

    for layer in model.layers:
        state = layer.init_state(1, device=device, dtype=out.dtype)
        runtime = layer._precompute_runtime_params()

        layer_spikes = []
        layer_mems = []
        layer_slow = []

        for t in range(T):
            spike, membrane, state = layer.forward_step(
                out[:, t, :], state, runtime,
            )
            layer_spikes.append(spike.squeeze(0).detach().cpu().numpy())
            layer_mems.append(membrane.squeeze(0).detach().cpu().numpy())
            layer_slow.append(state["x"].squeeze(0).detach().cpu().numpy())

        slow_list.append(np.stack(layer_slow, axis=0))     # [T, H, 3]
        mem_list.append(np.stack(layer_mems, axis=0))       # [T, H]
        spike_list.append(np.stack(layer_spikes, axis=0))   # [T, H]

        # Feed spikes to next layer
        spike_t = torch.from_numpy(
            np.stack(layer_spikes, axis=0)
        ).unsqueeze(0).to(device)
        out = spike_t

    return {
        "slow_states": slow_list,
        "membranes": mem_list,
        "spikes": spike_list,
    }


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def find_spiking_sample(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    min_spikes: int = 3,
    max_attempts: int = 200,
) -> Tuple[torch.Tensor, int, np.ndarray]:
    """
    Find a test sample where the last layer has >= min_spikes spike events.

    Returns (x_tensor [1,T,F], label, spikes_last_layer [T,H]).
    """
    model.eval()
    for i, (x, y) in enumerate(test_loader):
        if i >= max_attempts:
            raise RuntimeError(
                f"No sample with >= {min_spikes} spikes found in {max_attempts} attempts."
            )
        x = x.to(device)
        rec = record_trajectory(model, x)
        last_spikes = rec["spikes"][-1]  # [T, H]
        spike_times = np.where(last_spikes.sum(axis=1) > 0.5)[0]
        if len(spike_times) >= min_spikes:
            print(f"  Sample {i} (label={y.item()}): "
                  f"{len(spike_times)} global spike events")
            return x, y.item(), last_spikes
    raise RuntimeError(f"No sample with >= {min_spikes} spikes found.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(
    rec: Dict,
    spike_time: int,
    neuron_idx: int,
    out_dir: str,
):
    """
    Plot slow state continuity vs membrane reset around a spike.

    Top: slow state 3D (x_real, x_osc1, x_osc2) — continuous.
    Bottom: membrane u^0 — reset at spike.

    Args:
        rec: dict from record_trajectory()
        spike_time: timestep of the selected spike
        neuron_idx: which neuron to visualize (last layer)
        out_dir: save directory
    """
    WINDOW = 15
    slow = rec["slow_states"][-1]    # last layer [T, H, 3]
    membrane = rec["membranes"][-1]  # last layer [T, H]
    spikes = rec["spikes"][-1]       # last layer [T, H]
    T = slow.shape[0]

    t_start = max(0, spike_time - WINDOW)
    t_end = min(T, spike_time + WINDOW + 1)
    t_rel = np.arange(t_start, t_end) - spike_time  # 0 = spike

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # --- Top: Slow state 3D ---
    ax = axes[0]
    labels = [r"$x^{\mathrm{real}}$", r"$x^{\mathrm{osc}}_1$", r"$x^{\mathrm{osc}}_2$"]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for d in range(3):
        ax.plot(t_rel, slow[t_start:t_end, neuron_idx, d],
                color=colors[d], label=labels[d], linewidth=1.8, marker=".", markersize=3)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.8, label="Spike")
    ax.set_ylabel("State value")
    ax.set_title("SPRiF Slow State $x_t$ — Continuous Across Spike", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Bottom: Membrane ---
    ax = axes[1]
    ax.plot(t_rel, membrane[t_start:t_end, neuron_idx],
            color="#2c7bb6", linewidth=1.8, marker=".", markersize=3,
            label=r"Membrane $u^0$")
    spike_mask = spikes[t_start:t_end, neuron_idx] > 0.5
    spike_rel = t_rel[spike_mask]
    spike_vals = membrane[t_start:t_end, neuron_idx][spike_mask]
    ax.scatter(spike_rel, spike_vals,
               color="red", s=80, zorder=5, marker="v",
               label="Spike event")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax.set_xlabel("Timesteps relative to spike (t=0)")
    ax.set_ylabel("Membrane potential")
    ax.set_title(
        "SPRiF Fast State $u^0$ (Membrane) — Projective Reset at Spike",
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle(
        f"SPRiF State Trajectory — Neuron #{neuron_idx}, "
        f"Spike at t={spike_time}",
        y=1.01, fontweight="bold",
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, "trajectory_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_trajectory_data(
    rec: Dict,
    out_dir: str,
    label: int,
    spike_time: int,
    neuron_idx: int,
):
    """Save recorded trajectory as .npz for reproducibility / further analysis.

    Saves per-layer slow states, membranes, and spikes as structured arrays.
    """
    save_path = os.path.join(out_dir, "trajectory_data.npz")
    npz = {}
    for li in range(len(rec["slow_states"])):
        npz[f"layer{li}_slow"] = rec["slow_states"][li]     # (T, H, 3)
        npz[f"layer{li}_membrane"] = rec["membranes"][li]   # (T, H)
        npz[f"layer{li}_spikes"] = rec["spikes"][li]        # (T, H)
    npz["label"] = np.array(label)
    npz["highlight_spike_time"] = np.array(spike_time)
    npz["highlight_neuron"] = np.array(neuron_idx)
    np.savez_compressed(save_path, **npz)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load or train model ----
    from model import SPRiFpSMNISTNet

    ckpt = _find_checkpoint(TASK_ABS, "SPRiFpSMNISTNet")
    if ckpt is None:
        print("No checkpoint found. Training...")
        ckpt = _train_psmnist()

    print(f"Loading checkpoint: {os.path.basename(ckpt)}")
    model = SPRiFpSMNISTNet(
        input_size=1,
        hidden_sizes=[64, 256],
        num_classes=10,
        mode="srnn",
        warmup_steps=0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Data ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_mnist = torchvision.datasets.MNIST(
        root=os.path.join(TASK_ABS, "data"), train=False,
        download=True, transform=transform,
    )
    torch.manual_seed(0)
    perm = torch.randperm(784)

    from model import PermutedMNIST
    test_dataset = PermutedMNIST(test_mnist, perm)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=2,
    )

    # ---- Find sample with spikes ----
    x_sample, label, last_spikes = find_spiking_sample(
        model, test_loader, device, min_spikes=3,
    )

    # ---- Record trajectory ----
    print("Recording full trajectory...")
    set_seed(0)
    rec = record_trajectory(model, x_sample)

    # ---- Pick a neuron and spike time ----
    spike_times = np.where(last_spikes.sum(axis=1) > 0.5)[0]
    chosen_spike = int(spike_times[len(spike_times) // 2])  # middle spike
    spiking_neurons = np.where(last_spikes[chosen_spike] > 0.5)[0]
    chosen_neuron = int(spiking_neurons[0])
    print(f"  Selected spike at t={chosen_spike}, neuron #{chosen_neuron}")

    # ---- Plot ----
    print("Plotting...")
    plot_trajectory(rec, chosen_spike, chosen_neuron, out_dir)
    save_trajectory_data(rec, out_dir, label, chosen_spike, chosen_neuron)
    print("Done.")


if __name__ == "__main__":
    main()
