"""
SPRiF Robustness Experiment R2: Sequence Length x Noise Coupling.

Varies input sequence length and applies fixed additive noise (sigma=0.05).
Tests whether SPRiF's spectral filtering provides sustained noise immunity
across temporal scales, vs LIF's single-timescale accumulation.

Datasets: GSC (truncate mel frames), QTDB (resample time axis).

Usage:
    cd 代码/experiments
    python robustness/sequence_noise.py
"""

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(
    os.path.dirname(ROOT), "experiment-design-20260606", "results",
)


def _add_path(task_dir: str) -> str:
    p = os.path.join(ROOT, task_dir)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for name in list(sys.modules):
        if (
            name in {"model", "model_lif", "model_wrapper", "data", "data_asrnn", "core_algorithm"}
            or name.startswith("core_algorithm.")
            or name.startswith("model_wrapper.")
        ):
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


# ---------------------------------------------------------------------------
# Sequence manipulation
# ---------------------------------------------------------------------------

def truncate_gsc(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Truncate GSC mel-spectrogram along time axis.

    x: (B, 3, seq_len, n_mels) after collate but before reshape.
    Returns x truncated to target_len time frames, then reshaped to (B, target_len, 120).
    """
    # x shape: (B, 3, 101, 40)
    x_trunc = x[:, :, :target_len, :]                     # (B, 3, target_len, 40)
    B = x_trunc.shape[0]
    return x_trunc.permute(0, 2, 1, 3).reshape(B, target_len, 120)


def resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample ECG signal along time axis using linear interpolation.

    x: (B, T_orig, F)
    Returns x of shape (B, target_len, F).
    """
    B, T_orig, F = x.shape
    if T_orig == target_len:
        return x
    # indices into original time axis
    src_idx = torch.linspace(0, T_orig - 1, target_len).to(x.device)
    lo = src_idx.long().clamp(0, T_orig - 1)
    hi = (lo + 1).clamp(0, T_orig - 1)
    frac = (src_idx - lo.float()).unsqueeze(-1).unsqueeze(0)  # (1, target_len, 1)
    x_resampled = x[:, lo, :] * (1 - frac) + x[:, hi, :] * frac
    return x_resampled


# ---------------------------------------------------------------------------
# Loaders (same pattern as noise_benchmark.py)
# ---------------------------------------------------------------------------

def _additive_np(x: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    noise = rng.randn(*x.shape).astype(np.float32) * sigma
    return x + noise


def _load_gsc_models(task_dir: str):
    """Load SPRiF and ASRNN GSC models. Train if needed."""
    _add_path(task_dir)
    from model import SPRiFGSCNet

    # Import ASRNN model
    sys.path.insert(0, os.path.join(ROOT, task_dir, "model_wrapper"))
    from asrnn_gsc import ASRNNGSCNet

    task_abs = os.path.join(ROOT, task_dir)
    data_root = os.path.join(task_abs, "autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02")

    models = {}

    # Load SPRiF
    prefix = "SPRiFGSCNet"
    ckpt = _find_checkpoint(task_abs, prefix)
    if ckpt is None:
        if not os.path.exists(data_root):
            raise FileNotFoundError("GSC data not found.")
        _train_subprocess(task_dir, "train.py", [
            "--data-root", "autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02",
            "--epochs", "150", "--batch-size", "200", "--seed", "42", "--hidden-sizes", "300",
        ])
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for SPRiFGSCNet.")

    print(f"  {prefix}: {os.path.basename(ckpt)}")
    sprif_model = SPRiFGSCNet(
        input_size=120, hidden_sizes=[300], num_classes=12,
        dropout=0.0, recurrent_flags=(True,),
        neuron_kwargs={
            "threshold": 0.8, "init_std": 0.15,
            "tau_alpha_range": (10.0, 80.0),
            "tau_rho_range": (4.0, 30.0),
            "tau_eta_range": (0.8, 8.0),
            "omega_range": (0.04 * math.pi, 0.40 * math.pi),
        },
    )
    sprif_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    sprif_model.eval()
    models["SPRiFGSCNet"] = sprif_model

    # Load ASRNN
    ckpt = _find_checkpoint(task_abs, "ASRNNGSCNet")
    if ckpt is None:
        if not os.path.exists(data_root):
            raise FileNotFoundError("GSC data not found.")
        _train_subprocess(task_dir, "train_asrnn.py", [
            "--data-root", "autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02",
            "--epochs", "30", "--batch-size", "32", "--seed", "0", "--hidden-size", "256",
        ])
        ckpt = _find_checkpoint(task_abs, "ASRNNGSCNet")
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for ASRNNGSCNet.")

    print(f"  ASRNNGSCNet: {os.path.basename(ckpt)}")
    asrnn_model = ASRNNGSCNet(
        input_size=120, hidden_size=256, num_classes=12,
        device="cpu"
    )
    asrnn_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    asrnn_model.eval()
    models["ASRNNGSCNet"] = asrnn_model

    return models["SPRiFGSCNet"], models["ASRNNGSCNet"]


def _load_ecg_models(task_dir: str):
    """Load SPRiF and ASRNN ECG models. Train if needed."""
    import scipy.io

    _add_path(task_dir)
    from core_algorithm.utils import convert_dataset_wtime

    from model import SPRiFECGModel

    # Import ASRNN model
    sys.path.insert(0, os.path.join(ROOT, task_dir, "model_wrapper"))
    from asrnn_ecg import ASRNNECGNet

    task_abs = os.path.join(ROOT, task_dir)
    train_mat_path = os.path.join(task_abs, "data", "QTDB_train.mat")
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError("ECG data not found.")

    mat = scipy.io.loadmat(train_mat_path)
    _, train_x, _ = convert_dataset_wtime(mat)
    input_size = train_x.shape[2]

    models = {}

    # Load SPRiF
    prefix = "SPRiFECGModel"
    ckpt = _find_checkpoint(task_abs, prefix)
    if ckpt is None:
        _train_subprocess(task_dir, "train.py", [
            "--train-mat", os.path.join("data", "QTDB_train.mat"),
            "--test-mat", os.path.join("data", "QTDB_test.mat"),
            "--epochs", "250", "--batch-size", "64",
            "--seed", "1111", "--hidden-sizes", "36",
        ])
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for SPRiFECGModel.")

    print(f"  {prefix}: {os.path.basename(ckpt)}")
    sprif_model = SPRiFECGModel(
        input_size=input_size, hidden_sizes=[36], output_size=6,
        mode="srnn",
        neuron_kwargs={
            "threshold": 0.6, "init_std": 0.05, "bias": True,
            "tau_alpha_range": (20.0, 120.0),
            "tau_rho_range": (4.0, 30.0),
            "tau_eta_range": (0.8, 8.0),
            "omega_range": (0.02 * math.pi, 0.20 * math.pi),
        },
    )
    sprif_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    sprif_model.eval()
    models["SPRiFECGModel"] = sprif_model

    # Load ASRNN
    ckpt = _find_checkpoint(task_abs, "ASRNNECGModel")
    if ckpt is None:
        _train_subprocess(task_dir, "train_asrnn.py", [
            "--train-mat", os.path.join("data", "QTDB_train.mat"),
            "--test-mat", os.path.join("data", "QTDB_test.mat"),
            "--epochs", "250", "--batch-size", "64",
            "--seed", "1111", "--hidden-size", "36",
        ])
        ckpt = _find_checkpoint(task_abs, "ASRNNECGModel")
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for ASRNNECGModel.")

    print(f"  ASRNNECGModel: {os.path.basename(ckpt)}")
    asrnn_model = ASRNNECGNet(
        input_size=input_size, hidden_size=36, num_classes=6,
        sub_seq_length=10, device="cpu"
    )
    asrnn_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    asrnn_model.eval()
    models["ASRNNECGModel"] = asrnn_model

    return models["SPRiFECGModel"], models["ASRNNECGModel"]


def _train_subprocess(task_dir: str, train_script: str, extra_args: List[str]):
    cwd = os.path.join(ROOT, task_dir)
    script = os.path.join(cwd, train_script)
    cmd = [sys.executable, script] + extra_args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.returncode}")


def _build_gsc_raw_loader(task_dir: str, batch_size: int = 200):
    """Build GSC test loader returning raw (B, 3, 101, 40) tensors."""
    import torchvision
    from torch.utils.data import DataLoader

    _add_path(task_dir)
    from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

    task_abs = os.path.join(ROOT, task_dir)
    data_root = os.path.join(task_abs, "autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02")

    testing_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}
    for w in os.listdir(data_root):
        full = os.path.join(data_root, w)
        if os.path.isdir(full) and w[0] != "_" and w not in label_dct:
            label_dct[w] = label_dct["_unknown_"]

    sr, n_fft, hop_length = 16000, int(30e-3 * 16000), int(10e-3 * 16000)
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
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=collate_fn)


def _build_ecg_raw_loader(task_dir: str, batch_size: int = 64):
    import scipy.io
    from torch.utils.data import DataLoader, TensorDataset

    _add_path(task_dir)
    from core_algorithm.utils import convert_dataset_wtime

    task_abs = os.path.join(ROOT, task_dir)
    test_mat = scipy.io.loadmat(os.path.join(task_abs, "data", "QTDB_test.mat"))
    train_mat = scipy.io.loadmat(os.path.join(task_abs, "data", "QTDB_train.mat"))
    _, test_x, test_y = convert_dataset_wtime(test_mat)
    _, _, train_y = convert_dataset_wtime(train_mat)

    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).long()
    train_y = torch.from_numpy(train_y).long()
    label_min = min(train_y.min().item(), test_y.min().item())
    if label_min != 0:
        test_y -= label_min

    return DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_gsc_seqlen(model, x_tensor, y_tensor, device):
    """Evaluate GSC model with pre-processed (B, T, 120) input."""
    total_correct = 0
    total = x_tensor.shape[0]
    x = x_tensor.to(device)
    y = y_tensor.to(device)
    logits, _ = model(x)
    total_correct += (logits.argmax(dim=-1) == y).sum().item()
    return total_correct / total


@torch.no_grad()
def _eval_ecg_seqlen(model, x_tensor, y_tensor, device):
    """Evaluate ECG model with pre-processed (B, T, F) input."""
    total_correct = 0
    total = y_tensor.numel()
    x = x_tensor.to(device)
    y = y_tensor.to(device)
    logits = model(x)
    pred = logits.argmax(dim=1)
    total_correct += pred.eq(y).sum().item()
    return total_correct / total


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

GSC_SEQ_LENS = [50, 75, 101]
ECG_SEQ_LENS = [150, 300, 600, "original"]
NOISE_SIGMA = 0.05


def run_gsc(spare_model, lif_model, raw_loader, device, rng) -> List[dict]:
    """Test GSC across different truncated sequence lengths."""
    results = []
    all_batches = list(raw_loader)

    for seq_len in GSC_SEQ_LENS:
        print(f"  GSC seq_len={seq_len}")
        # Prepare all batches at this seq_len
        x_all, y_all = [], []
        for x_raw, y in all_batches:
            x_seq = truncate_gsc(x_raw, seq_len)
            x_all.append(x_seq)
            y_all.append(y)
        x_cat = torch.cat(x_all, dim=0)
        y_cat = torch.cat(y_all, dim=0)

        # Clean
        acc_spare_clean = _eval_gsc_seqlen(spare_model, x_cat, y_cat, device)
        acc_lif_clean = _eval_gsc_seqlen(lif_model, x_cat, y_cat, device)

        # Noisy
        x_np = x_cat.numpy()
        x_noisy_np = _additive_np(x_np, NOISE_SIGMA, rng)
        x_noisy = torch.from_numpy(x_noisy_np).float()
        acc_spare_noisy = _eval_gsc_seqlen(spare_model, x_noisy, y_cat, device)
        acc_lif_noisy = _eval_gsc_seqlen(lif_model, x_noisy, y_cat, device)

        results.append({
            "dataset": "GSC",
            "seq_len": seq_len,
            "SPRiF_clean": round(acc_spare_clean, 4),
            "SPRiF_noisy": round(acc_spare_noisy, 4),
            "LIF_clean": round(acc_lif_clean, 4),
            "LIF_noisy": round(acc_lif_noisy, 4),
        })

        print(f"    SPRiF: clean={acc_spare_clean:.4f} noisy={acc_spare_noisy:.4f}")
        print(f"    LIF:   clean={acc_lif_clean:.4f} noisy={acc_lif_noisy:.4f}")

    return results


def run_ecg(spare_model, lif_model, raw_loader, device, rng) -> List[dict]:
    """Test ECG across different resampled sequence lengths."""
    results = []
    all_batches = list(raw_loader)

    for seq_len in ECG_SEQ_LENS:
        print(f"  QTDB seq_len={seq_len}")
        x_all, y_all = [], []
        for x_raw, y in all_batches:
            if seq_len == "original":
                x_seq = x_raw
            else:
                x_seq = resample_time(x_raw, seq_len)
            x_all.append(x_seq)
            y_all.append(y)
        x_cat = torch.cat(x_all, dim=0)
        y_cat = torch.cat(y_all, dim=0)

        # Clean
        acc_spare_clean = _eval_ecg_seqlen(spare_model, x_cat, y_cat, device)
        acc_lif_clean = _eval_ecg_seqlen(lif_model, x_cat, y_cat, device)

        # Noisy
        x_np = x_cat.numpy()
        x_noisy_np = _additive_np(x_np, NOISE_SIGMA, rng)
        x_noisy = torch.from_numpy(x_noisy_np).float()
        acc_spare_noisy = _eval_ecg_seqlen(spare_model, x_noisy, y_cat, device)
        acc_lif_noisy = _eval_ecg_seqlen(lif_model, x_noisy, y_cat, device)

        seq_label = seq_len if isinstance(seq_len, int) else "original"
        results.append({
            "dataset": "QTDB",
            "seq_len": seq_label,
            "SPRiF_clean": round(acc_spare_clean, 4),
            "SPRiF_noisy": round(acc_spare_noisy, 4),
            "LIF_clean": round(acc_lif_clean, 4),
            "LIF_noisy": round(acc_lif_noisy, 4),
        })

        print(f"    SPRiF: clean={acc_spare_clean:.4f} noisy={acc_spare_noisy:.4f}")
        print(f"    LIF:   clean={acc_lif_clean:.4f} noisy={acc_lif_noisy:.4f}")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sequence_noise(all_results: List[dict], out_dir: str):
    """Accuracy vs sequence length, 2 panels (GSC/QTDB), clean/noisy lines."""
    gsc_results = [r for r in all_results if r["dataset"] == "GSC"]
    qtdb_results = [r for r in all_results if r["dataset"] == "QTDB"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GSC panel
    ax = axes[0]
    if gsc_results:
        x_gsc = [r["seq_len"] for r in gsc_results]
        for model, color, marker in [("SPRiF", "#2b83ba", "o"), ("LIF", "#e41a1c", "s")]:
            clean_vals = [r[f"{model}_clean"] for r in gsc_results]
            noisy_vals = [r[f"{model}_noisy"] for r in gsc_results]
            ax.plot(x_gsc, clean_vals, color=color, marker=marker, linewidth=2,
                    label=f"{model} clean")
            ax.plot(x_gsc, noisy_vals, color=color, marker=marker, linewidth=2,
                    linestyle="--", label=f"{model} noisy (σ=0.05)")
        ax.set_xlabel("Sequence length (mel frames)")
        ax.set_ylabel("Accuracy")
        ax.set_title("GSC", fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

    # QTDB panel
    ax = axes[1]
    if qtdb_results:
        x_qtdb = list(range(len(qtdb_results)))
        labels_qtdb = [str(r["seq_len"]) for r in qtdb_results]
        for model, color, marker in [("SPRiF", "#2b83ba", "o"), ("LIF", "#e41a1c", "s")]:
            clean_vals = [r[f"{model}_clean"] for r in qtdb_results]
            noisy_vals = [r[f"{model}_noisy"] for r in qtdb_results]
            ax.plot(x_qtdb, clean_vals, color=color, marker=marker, linewidth=2,
                    label=f"{model} clean")
            ax.plot(x_qtdb, noisy_vals, color=color, marker=marker, linewidth=2,
                    linestyle="--", label=f"{model} noisy (σ=0.05)")
        ax.set_xticks(x_qtdb)
        ax.set_xticklabels(labels_qtdb)
        ax.set_xlabel("Sequence length (time steps)")
        ax.set_ylabel("Accuracy")
        ax.set_title("QTDB", fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

    fig.suptitle("Sequence Length x Noise Coupling", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "sequence_noise.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(42)

    all_results = []

    # GSC
    try:
        print("\nLoading GSC models...")
        sprim, lifm = _load_gsc_models("Task_GSC")
        sprim.to(device)
        lifm.to(device)
        loader = _build_gsc_raw_loader("Task_GSC")
        results = run_gsc(sprim, lifm, loader, device, rng)
        all_results.extend(results)
    except FileNotFoundError as e:
        print(f"  SKIP GSC: {e}")

    # QTDB
    try:
        print("\nLoading ECG models...")
        sprim, lifm = _load_ecg_models("Task_ECG")
        sprim.to(device)
        lifm.to(device)
        loader = _build_ecg_raw_loader("Task_ECG")
        results = run_ecg(sprim, lifm, loader, device, rng)
        all_results.extend(results)
    except FileNotFoundError as e:
        print(f"  SKIP QTDB: {e}")

    if not all_results:
        print("\nNo datasets evaluated. Aborting.")
        return

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "sequence_noise.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    print("Plotting...")
    plot_sequence_noise(all_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
