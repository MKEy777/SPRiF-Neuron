"""
SPRiF Effective Temporal Kernel Analysis — Impulse Response Gallery.

Injects a unit impulse (bypassing input linear layer) into learned SPRiF neurons
and records the slow-state trajectory. Compares with the *trained* ASRNN
(adaptive LIF) baseline — using its actual learned per-neuron tau_m to compute
the impulse response of its membrane kernel — and visualizes the diversity of
learned temporal filters in both time and frequency domain.

Note: pSMNIST has no ASRNN checkpoint; its ASRNN comparison is skipped.
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
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "impulse_analysis",
)


def _add_path(task_dir: str) -> str:
    p = os.path.join(ROOT, task_dir)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for name in list(sys.modules):
        if name in {"model", "data", "core_algorithm"} or name.startswith("core_algorithm."):
            sys.modules.pop(name, None)
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
# Model loading per task
# ---------------------------------------------------------------------------

def _load_psmnist(task_dir: str) -> nn.Module:
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
        return model

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


def _load_gsc(task_dir: str) -> nn.Module:
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
        return model

    print("  No checkpoint found. Training GSC model...")
    data_root = os.path.join(ROOT, task_dir, "data", "SpeechCommands")
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
    ckpt = _find_checkpoint(os.path.join(ROOT, task_dir), "SPRiFGSCNet")
    if ckpt is None:
        raise RuntimeError("Training completed but no checkpoint found.")
    return _load_gsc(task_dir)


def _load_ecg(task_dir: str) -> nn.Module:
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
        return model

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


# ---------------------------------------------------------------------------
# Impulse response computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_impulse_responses(
    layer: nn.Module,
    T: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject unit impulse via input_current bypass, record slow state and membrane.

    Uses ``input_current`` on ``forward_step()`` to bypass ``input_linear``,
    so every neuron receives exactly the same unit pulse. Differences in the
    resulting trajectory are purely due to intrinsic spectral parameters.

    Args:
        layer: SPRiFNeuronLayer instance.
        T: number of timesteps to simulate.
        device: compute device.

    Returns:
        slow_resp:  (H, T, 3) — slow state x_t trajectory per neuron.
        membrane_resp: (H, T) — membrane potential u⁰ per neuron.
    """
    H = layer.hidden_size
    state = layer.init_state(1, device=device)
    runtime = layer._precompute_runtime_params()

    slow_list: List[torch.Tensor] = []
    membrane_list: List[torch.Tensor] = []

    for t in range(T):
        # Unit impulse at t=0, zero thereafter — injected directly into
        # the slow-state dynamics, bypassing input_linear.
        if t == 0:
            ic = torch.ones(1, H, device=device, dtype=state["x"].dtype)
        else:
            ic = torch.zeros(1, H, device=device, dtype=state["x"].dtype)

        spike, membrane, state = layer.forward_step(
            torch.zeros(1, layer.input_size, device=device),
            state,
            runtime,
            input_current=ic,
        )
        slow_list.append(state["x"].detach().cpu())
        membrane_list.append(membrane.detach().cpu())

    # state["x"] shape: (1, H, 3) → stacked over T along dim=1 → (1, T, H, 3)
    # squeeze batch → (T, H, 3), then permute to (H, T, 3)
    slow_resp = torch.stack(slow_list, dim=1).squeeze(0).permute(1, 0, 2).contiguous()  # (H, T, 3)
    membrane_resp = torch.stack(membrane_list, dim=1).squeeze(0).permute(1, 0).contiguous()  # (H, T)
    return slow_resp, membrane_resp


def compute_asrnn_impulse_response(
    tau_m: np.ndarray,
    T: int,
) -> np.ndarray:
    """ASRNN (adaptive LIF) impulse response of the membrane kernel.

    ASRNN membrane dynamics (no spike, no adaptation active):
        mem[t] = alpha * mem[t-1] + (1 - alpha) * input[t]
    where alpha = exp(-1 / tau_m). A unit impulse at t=0 yields the response:
        h[t] = (1 - alpha) * alpha^t

    Args:
        tau_m: array of learned per-neuron time constants, shape (H,).
        T: number of timesteps to simulate.

    Returns:
        resp: array of shape (H, T) — impulse response for each neuron.
    """
    alpha = np.exp(-1.0 / np.clip(tau_m, 1e-3, None))  # (H,)
    t = np.arange(T)[None, :]  # (1, T)
    resp = (1.0 - alpha[:, None]) * np.power(alpha[:, None], t)  # (H, T)
    return resp


# ---------------------------------------------------------------------------
# ASRNN model loading (adaptive LIF baseline)
# ---------------------------------------------------------------------------

ASRNN_ROOT = os.path.join(os.path.dirname(ROOT), "ASRNN")


def _find_asrnn_ckpt(task_folder: str, prefix: str, legacy_paths: List[str]) -> Optional[str]:
    """Search common locations for an ASRNN checkpoint.

    Order:
      1. legacy explicit paths (e.g. ASRNN/GSC/*.pth in old layout)
      2. Task_<X>/<prefix>_*.pth or Task_<X>/*.pth alongside SPRiF checkpoints
    """
    for p in legacy_paths:
        if os.path.exists(p):
            return p
    task_dir = os.path.join(ROOT, task_folder)
    for pattern in (f"{prefix}_*.pth", f"{prefix}*.pth", "ASRNN*.pth"):
        files = sorted(glob.glob(os.path.join(task_dir, pattern)))
        if files:
            # pick highest-acc one when the filename encodes _acc<num>
            best, best_acc = None, -1.0
            for f in files:
                base = os.path.basename(f)
                acc = 0.0
                if "_acc" in base:
                    try:
                        acc = float(base.rsplit("_acc", 1)[1].replace(".pth", ""))
                    except (ValueError, IndexError):
                        acc = 0.0
                if acc > best_acc:
                    best_acc, best = acc, f
            return best
    return None


def _extract_tau_m_from_state(state, candidate_keys: List[str]) -> Optional[np.ndarray]:
    """Extract a tau_m-like tensor from a checkpoint state_dict without instantiating the model.

    This avoids wrapper-class dependency issues (missing modules, input_size mismatch).
    """
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None
    for key in candidate_keys:
        if key in state:
            t = state[key]
            if torch.is_tensor(t):
                return t.detach().cpu().numpy().ravel()
    # Fallback: fuzzy match — any key ending with .tau_m / tau_m_h
    for key, t in state.items():
        if not torch.is_tensor(t):
            continue
        low = key.lower()
        if low.endswith("tau_m") or low.endswith("tau_m_h") or low.endswith(".tau_m"):
            arr = t.detach().cpu().numpy().ravel()
            print(f"  [fuzzy] using key '{key}' as tau_m (shape={arr.shape})")
            return arr
    return None


def _load_asrnn_gsc_tau_m() -> Optional[np.ndarray]:
    """Load ASRNN-GSC checkpoint and extract layer-1 tau_m directly from state_dict."""
    legacy = [os.path.join(ASRNN_ROOT, "GSC", "0.91398-gscv1-_class.pth")]
    ckpt = _find_asrnn_ckpt("Task_GSC", "ASRNNGSCNet", legacy)
    if ckpt is None:
        print(f"  [ASRNN-GSC] no checkpoint found (searched ASRNN/GSC/ and Task_GSC/)")
        return None
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        tau_m = _extract_tau_m_from_state(
            state,
            candidate_keys=["dense_1.tau_m", "layer1.tau_m", "tau_m_h"],
        )
        if tau_m is None:
            print(f"  [ASRNN-GSC] tau_m key not found in {os.path.basename(ckpt)}")
            return None
        print(f"  [ASRNN-GSC] loaded {os.path.basename(ckpt)}: "
              f"tau_m range [{tau_m.min():.2f}, {tau_m.max():.2f}] (H={len(tau_m)})")
        return tau_m
    except Exception as e:
        print(f"  [ASRNN-GSC] failed to load: {e}")
        return None


def _load_asrnn_ecg_tau_m() -> Optional[np.ndarray]:
    """Load ASRNN-ECG checkpoint and extract hidden-layer tau_m directly from state_dict."""
    legacy_dir = os.path.join(ASRNN_ROOT, "ECG")
    legacy = sorted(glob.glob(os.path.join(legacy_dir, "*.pth")))
    ckpt = _find_asrnn_ckpt("Task_ECG", "ASRNNECGModel", legacy)
    if ckpt is None:
        print(f"  [ASRNN-ECG] no checkpoint found (searched ASRNN/ECG/ and Task_ECG/)")
        return None
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        tau_m = _extract_tau_m_from_state(
            state,
            candidate_keys=["tau_m_h", "tau_m", "dense_1.tau_m"],
        )
        if tau_m is None:
            print(f"  [ASRNN-ECG] tau_m key not found in {os.path.basename(ckpt)}")
            return None
        print(f"  [ASRNN-ECG] loaded {os.path.basename(ckpt)}: "
              f"tau_m range [{tau_m.min():.2f}, {tau_m.max():.2f}] (H={len(tau_m)})")
        return tau_m
    except Exception as e:
        print(f"  [ASRNN-ECG] failed to load: {e}")
        return None


ASRNN_LOADERS = {
    "GSC": _load_asrnn_gsc_tau_m,
    "ECG": _load_asrnn_ecg_tau_m,
    # pSMNIST: no pretrained ASRNN checkpoint available
}


def _sampled_neurons(alpha: np.ndarray, n_samples: int = 8) -> np.ndarray:
    """Select evenly-spaced neuron indices covering the alpha range.

    Returns sorted indices into the alpha array.
    """
    H = len(alpha)
    order = np.argsort(alpha)
    n = min(n_samples, H)
    step = max(1, H // n)
    picked = [order[i] for i in range(0, H, step)[:n]]
    # Ensure endpoints are included
    if order[0] not in picked:
        picked.insert(0, order[0])
    if order[-1] not in picked:
        picked.append(order[-1])
    return np.array(sorted(picked, key=lambda i: alpha[i]))


# ---------------------------------------------------------------------------
# Plot: impulse response gallery
# ---------------------------------------------------------------------------

def plot_impulse_gallery(
    all_responses: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    all_alphas: Dict[str, Dict[int, np.ndarray]],
    all_omegas: Dict[str, Dict[int, np.ndarray]],
    out_dir: str,
    T: int = 100,
    n_cols: int = 8,
):
    """Grid of impulse responses for sampled neurons per task.

    Rows = tasks, columns = sampled neurons sorted by alpha.
    Shows x_real (main line), x_osc1/x_osc2 (thin dashed).
    """
    tasks = [t for t in ["pSMNIST", "GSC", "ECG"] if t in all_responses]
    if not tasks:
        return

    n_rows = len(tasks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, task_name in enumerate(tasks):
        # Use layer 0 only (first layer receives raw input)
        if 0 not in all_responses[task_name]:
            continue
        slow_resp, _ = all_responses[task_name][0]
        alphas = all_alphas[task_name][0]
        H = slow_resp.shape[0]

        indices = _sampled_neurons(alphas, n_cols)
        n_actual = min(len(indices), n_cols)

        for ci in range(n_cols):
            ax = axes[row_idx, ci]
            if ci >= n_actual:
                ax.set_visible(False)
                continue

            ni = indices[ci]
            ax.plot(slow_resp[ni, :, 0].numpy(), color="#1b9e77",
                    linewidth=1.5, label=r"$x^{\mathrm{real}}$")
            ax.plot(slow_resp[ni, :, 1].numpy(), color="#d95f02",
                    linewidth=0.6, linestyle="--", alpha=0.7, label=r"$x^{\mathrm{osc}}_1$")
            ax.plot(slow_resp[ni, :, 2].numpy(), color="#7570b3",
                    linewidth=0.6, linestyle="--", alpha=0.7, label=r"$x^{\mathrm{osc}}_2$")

            alpha_val = alphas[ni]
            omega_val = _extract_omega_for_neuron(all_omegas, task_name, 0, ni)
            ax.set_title(
                f"α={alpha_val:.3f}\nω={omega_val:.2f}",
                fontsize=7, fontweight="bold",
            )
            ax.set_xlim(0, T - 1)
            ax.tick_params(labelsize=6)

        # Row label and legend on first column
        axes[row_idx, 0].set_ylabel(f"{task_name}\n(H={H})", fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].legend(fontsize=6, loc="upper right", frameon=True)

    fig.suptitle(
        "SPRiF Impulse Response Gallery — Learned Temporal Kernels",
        y=1.01, fontweight="bold",
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, "impulse_response_gallery.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _extract_omega_for_neuron(
    all_omegas: Dict[str, Dict[int, np.ndarray]],
    task_name: str, layer_idx: int, neuron_idx: int,
) -> float:
    """Extract omega for a specific neuron from pre-collected data."""
    if task_name in all_omegas and layer_idx in all_omegas[task_name]:
        return all_omegas[task_name][layer_idx][neuron_idx]
    return 0.0


# ---------------------------------------------------------------------------
# Plot: frequency response
# ---------------------------------------------------------------------------

def plot_frequency_response(
    all_responses: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    all_alphas: Dict[str, Dict[int, np.ndarray]],
    out_dir: str,
    T: int = 100,
):
    """|FFT| of impulse responses, one subplot per task.

    Each line = one sampled neuron's |FFT(x_real)|, color-coded by alpha.
    Shows the diversity of frequency-domain filtering behavior.
    """
    tasks = [t for t in ["pSMNIST", "GSC", "ECG"] if t in all_responses]
    if not tasks:
        return

    n_rows = len(tasks)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Shared frequency axis
    freqs = np.fft.rfftfreq(T)  # normalized: 0 to 0.5 (Nyquist = pi rad/sample)

    for row_idx, task_name in enumerate(tasks):
        ax = axes[row_idx]
        if 0 not in all_responses[task_name]:
            continue
        slow_resp, _ = all_responses[task_name][0]
        alphas = all_alphas[task_name][0]
        H = slow_resp.shape[0]

        indices = _sampled_neurons(alphas, n_samples=12)

        # Color by alpha using viridis
        norm_alpha = (alphas[indices] - alphas.min()) / (alphas.max() - alphas.min() + 1e-8)
        cmap = plt.cm.viridis

        for i, ni in enumerate(indices):
            signal = slow_resp[ni, :, 0].numpy()  # x_real
            mag = np.abs(np.fft.rfft(signal))
            ax.plot(freqs, mag, color=cmap(norm_alpha[i]), linewidth=1.0,
                    alpha=0.85, label=f"α={alphas[ni]:.3f}" if i < 4 else "")

        ax.set_title(f"{task_name} — Frequency Response of $x^{{\\mathrm{{real}}}}$", fontweight="bold")
        ax.set_xlabel("Normalized frequency (cycles / timestep)")
        ax.set_ylabel("Magnitude  |FFT|")
        if row_idx == 0:
            ax.legend(fontsize=7, ncol=2, frameon=True)

        # Mark frequency range
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(0, 0.5)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(alphas.min(), alphas.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.92)
        cbar.set_label(r"$\alpha$ (decay)", fontsize=9)

    fig.suptitle("SPRiF Frequency-Domain Filtering Diversity", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "frequency_response.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot: ASRNN (adaptive LIF) comparison
# ---------------------------------------------------------------------------

def plot_asrnn_comparison(
    all_responses: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    all_alphas: Dict[str, Dict[int, np.ndarray]],
    asrnn_taus: Dict[str, np.ndarray],
    out_dir: str,
    T: int = 100,
):
    """Compare SPRiF impulse response with the trained ASRNN (adaptive LIF) kernel.

    For each task with an available ASRNN checkpoint, pick 3 SPRiF neurons
    (slow / medium / fast alpha). For each SPRiF neuron, we also pick the
    ASRNN neuron whose tau_m best matches the SPRiF-derived tau_effective, so
    both curves start with visually comparable time-scale. This isolates the
    *shape* difference (exponential vs. decay+oscillation) rather than a scale
    mismatch.
    """
    tasks = [t for t in ["pSMNIST", "GSC", "ECG"]
             if t in all_responses and t in asrnn_taus and asrnn_taus[t] is not None]
    if not tasks:
        print("  [ASRNN comparison] no task has both SPRiF response and ASRNN tau_m — skip.")
        return

    n_rows = len(tasks)
    n_cols = 3
    # Column order matches percentile_indices below: 10th → median → 90th percentile of α.
    # Small α ⇒ short τ (fast decay); large α ⇒ long τ (slow decay / long memory).
    labels = ["Fast α (short memory)", "Medium α", "Slow α (long memory)"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Collect raw curve data so the figure can be re-plotted from disk.
    dump: Dict[str, np.ndarray] = {}

    for row_idx, task_name in enumerate(tasks):
        slow_resp, _ = all_responses[task_name][0]
        alphas = all_alphas[task_name][0]
        asrnn_tau = asrnn_taus[task_name]
        asrnn_resp = compute_asrnn_impulse_response(asrnn_tau, T)  # (H_asrnn, T)

        alpha_sorted = np.sort(alphas)
        n_alpha = len(alpha_sorted)
        percentile_indices = [
            np.argmin(np.abs(alphas - alpha_sorted[max(0, n_alpha // 10)])),
            np.argmin(np.abs(alphas - np.median(alphas))),
            np.argmin(np.abs(alphas - alpha_sorted[min(n_alpha - 1, 9 * n_alpha // 10)])),
        ]

        for ci, (ni, label) in enumerate(zip(percentile_indices, labels)):
            ax = axes[row_idx, ci]
            alpha_val = alphas[ni]
            tau_sprif = float(np.clip(-1.0 / np.log(alpha_val + 1e-8), 0.1, 500.0))

            # Match ASRNN neuron by closest tau_m
            asrnn_idx = int(np.argmin(np.abs(asrnn_tau - tau_sprif)))
            tau_asrnn = float(asrnn_tau[asrnn_idx])

            sprif_x_real = slow_resp[ni, :, 0].numpy()

            osc_mag = np.sqrt(
                slow_resp[ni, :, 1].numpy() ** 2 + slow_resp[ni, :, 2].numpy() ** 2
            )
            asrnn_curve = asrnn_resp[asrnn_idx]

            # Save raw (unnormalized) curves + metadata for reproduction.
            prefix = f"{task_name}_col{ci}"
            dump[f"{prefix}_sprif_x_real"] = sprif_x_real
            dump[f"{prefix}_sprif_osc_mag"] = osc_mag
            dump[f"{prefix}_asrnn_kernel"] = asrnn_curve
            dump[f"{prefix}_meta"] = np.array(
                [alpha_val, tau_sprif, tau_asrnn, ni, asrnn_idx],
                dtype=np.float64,
            )  # order: alpha_sprif, tau_sprif, tau_asrnn, sprif_neuron_idx, asrnn_neuron_idx

            # Normalize both curves by their own peak for shape comparison
            def _norm(x):
                m = np.max(np.abs(x)) + 1e-12
                return x / m

            ax.plot(_norm(sprif_x_real), color="#1b9e77", linewidth=2.0,
                    label=fr"SPRiF $x^{{\mathrm{{real}}}}$  (τ≈{tau_sprif:.1f})")
            ax.plot(_norm(asrnn_curve), color="#e41a1c",
                    linewidth=1.8, linestyle="--",
                    label=fr"ASRNN kernel  (τ$_m$={tau_asrnn:.1f})")

            ax.plot(_norm(osc_mag), color="#377eb8", linewidth=0.8, linestyle=":",
                    alpha=0.8, label=r"$\sqrt{x_1^2+x_2^2}$ (normalized)")

            ax.set_title(f"{label}\nα={alpha_val:.3f}", fontsize=9, fontweight="bold")
            ax.set_xlim(0, T - 1)
            ax.axhline(0.0, color="gray", linewidth=0.4, alpha=0.5)
            if row_idx == 0:
                ax.legend(fontsize=7, frameon=True)

        axes[row_idx, 0].set_ylabel(task_name, fontweight="bold")

    fig.suptitle(
        "SPRiF vs Trained-ASRNN (Adaptive LIF) Temporal Kernel Comparison",
        y=1.01, fontweight="bold",
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, "asrnn_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Persist raw curves so the figure can be reproduced without re-running the
    # full pipeline (loading checkpoints, computing impulse responses).
    npz_path = os.path.join(out_dir, "asrnn_comparison_data.npz")
    dump["_tasks"] = np.array(tasks)
    dump["_col_labels"] = np.array(labels)
    dump["_T"] = np.array(T)
    np.savez(npz_path, **dump)
    print(f"  Saved: {npz_path}")


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

def save_raw_impulse_data(
    all_responses: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    all_alphas: Dict[str, Dict[int, np.ndarray]],
    all_omegas: Dict[str, Dict[int, np.ndarray]],
    asrnn_taus: Dict[str, Optional[np.ndarray]],
    out_dir: str,
    T: int,
):
    """Dump full raw curves so every plot can be regenerated offline.

    File: raw_impulse_responses.npz
      For each (task, layer):
        <task>_L<li>_slow_resp     : (H, T, 3) float32 — SPRiF slow-state (x_real, x1, x2)
        <task>_L<li>_membrane      : (H, T)    float32 — SPRiF membrane potential
        <task>_L<li>_alpha         : (H,)      float32
        <task>_L<li>_omega         : (H,)      float32
      For each task with ASRNN pretrained:
        <task>_asrnn_tau_m         : (H_asrnn,) float32
        <task>_asrnn_impulse       : (H_asrnn, T) float32
      Meta:
        _T, _tasks_with_asrnn
    """
    dump: Dict[str, np.ndarray] = {"_T": np.array(T)}
    for task_name, layer_map in all_responses.items():
        for li, (slow_resp, mem_resp) in layer_map.items():
            key = f"{task_name}_L{li}"
            dump[f"{key}_slow_resp"] = slow_resp.numpy().astype(np.float32)
            dump[f"{key}_membrane"] = mem_resp.numpy().astype(np.float32)
            dump[f"{key}_alpha"] = all_alphas[task_name][li].astype(np.float32)
            dump[f"{key}_omega"] = all_omegas[task_name][li].astype(np.float32)

    tasks_with_asrnn = []
    for tname, tau_m in asrnn_taus.items():
        if tau_m is None:
            continue
        tasks_with_asrnn.append(tname)
        dump[f"{tname}_asrnn_tau_m"] = tau_m.astype(np.float32)
        dump[f"{tname}_asrnn_impulse"] = compute_asrnn_impulse_response(
            tau_m, T
        ).astype(np.float32)
    dump["_tasks_with_asrnn"] = np.array(tasks_with_asrnn)

    npz_path = os.path.join(out_dir, "raw_impulse_responses.npz")
    np.savez_compressed(npz_path, **dump)
    print(f"  Saved: {npz_path}")


def save_impulse_stats(
    all_alphas: Dict[str, Dict[int, np.ndarray]],
    all_omegas: Dict[str, Dict[int, np.ndarray]],
    out_dir: str,
):
    """Save per-neuron spectral parameters + derived timescale to CSV."""
    records = []
    for task_name in ["pSMNIST", "GSC", "ECG"]:
        if task_name not in all_alphas:
            continue
        for li in all_alphas[task_name]:
            alphas = all_alphas[task_name][li]
            omegas = all_omegas.get(task_name, {}).get(li, np.zeros_like(alphas))
            for ni in range(len(alphas)):
                tau = float(np.clip(-1.0 / np.log(alphas[ni] + 1e-8), 0.1, 200.0))
                records.append({
                    "task": task_name,
                    "layer": li,
                    "neuron": ni,
                    "alpha": alphas[ni],
                    "omega": omegas[ni],
                    "tau_effective": tau,
                })
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "impulse_kernel_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

LOADERS = {
    "pSMNIST": ("Task_pSMNIST", _load_psmnist),
    "GSC": ("Task_GSC", _load_gsc),
    "ECG": ("Task_ECG", _load_ecg),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_responses: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    all_alphas: Dict[str, Dict[int, np.ndarray]] = {}
    all_omegas: Dict[str, Dict[int, np.ndarray]] = {}
    T = 100  # impulse response length

    for task_name, (task_dir, loader_fn) in LOADERS.items():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")
        task_abs = os.path.join(ROOT, task_dir)
        _add_path(task_dir)

        try:
            model = loader_fn(task_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        model.eval()
        model.to(device)

        all_responses[task_name] = {}
        all_alphas[task_name] = {}
        all_omegas[task_name] = {}

        for li, layer in enumerate(model.layers):
            # Only layer 0 gets clean impulse input; deeper layers receive spikes
            print(f"  Layer {li}: computing impulse responses for {layer.hidden_size} neurons...")
            slow_resp, mem_resp = compute_impulse_responses(layer, T=T, device=device)

            # Extract alpha/omega for sorting / color-coding
            params = layer.get_spectral_parameters()
            alphas_np = params["alpha"].detach().cpu().numpy()
            omegas_np = params["omega"].detach().cpu().numpy()

            all_responses[task_name][li] = (slow_resp, mem_resp)
            all_alphas[task_name][li] = alphas_np
            all_omegas[task_name][li] = omegas_np

            tau_from_alpha = -1.0 / np.log(alphas_np + 1e-8)
            print(f"    alpha: [{alphas_np.min():.4f}, {np.median(alphas_np):.4f}, {alphas_np.max():.4f}]")
            print(f"    tau:   [{tau_from_alpha.min():.1f}, {np.median(tau_from_alpha):.1f}, {tau_from_alpha.max():.1f}] steps")
            print(f"    omega: [{omegas_np.min():.4f}, {np.median(omegas_np):.4f}, {omegas_np.max():.4f}] rad")
            if li >= 1:
                print(f"    (Note: layer {li} receives spikes from layer {li-1}, "
                      f"not raw input — impulse analysis is for the intrinsic kernel only)")

    if not all_responses:
        print("\nNo models loaded. Aborting.")
        return

    print(f"\n{'='*50}")
    print("Loading ASRNN (adaptive LIF) baselines...")
    asrnn_taus: Dict[str, Optional[np.ndarray]] = {}
    for tname, loader in ASRNN_LOADERS.items():
        asrnn_taus[tname] = loader()

    print(f"\n{'='*50}")
    print("Plotting...")

    plot_impulse_gallery(all_responses, all_alphas, all_omegas, out_dir, T=T)
    plot_frequency_response(all_responses, all_alphas, out_dir, T=T)
    plot_asrnn_comparison(all_responses, all_alphas, asrnn_taus, out_dir, T=T)
    save_impulse_stats(all_alphas, all_omegas, out_dir)
    save_raw_impulse_data(all_responses, all_alphas, all_omegas, asrnn_taus, out_dir, T=T)

    print("\nDone.")


if __name__ == "__main__":
    main()
