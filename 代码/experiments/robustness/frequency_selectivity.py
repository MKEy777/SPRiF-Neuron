"""
SPRiF Robustness Experiment R3: Frequency Selectivity (SPRiF-unique).

Injects sinusoidal perturbations at 5 frequencies x 3 amplitudes into test inputs.
Compares SPRiF vs LIF accuracy degradation as a function of perturbation frequency.
This experiment directly tests whether SPRiF's learned spectral parameters (omega)
provide frequency-selective noise filtering.

Perturbation: x_perturbed[t] = x[t] + A * sin(2 * pi * f_norm * t)
    where f_norm is normalized frequency (cycles per timestep).

Frequencies: 0.01pi, 0.05pi, 0.10pi, 0.25pi, 0.50pi (normalized)
Amplitudes: low, medium, high (relative to input std)

Datasets: GSC, QTDB.

Usage:
    cd 代码/experiments
    python robustness/frequency_selectivity.py
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


def _train_subprocess(task_dir: str, train_script: str, extra_args: List[str]):
    cwd = os.path.join(ROOT, task_dir)
    script = os.path.join(cwd, train_script)
    cmd = [sys.executable, script] + extra_args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.returncode}")


# ---------------------------------------------------------------------------
# Sinusoidal perturbation
# ---------------------------------------------------------------------------

def sinusoidal_perturbation(
    x: np.ndarray,
    freq_norm: float,
    amplitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Add sinusoidal perturbation to input along time axis.

    Args:
        x: input array, shape (B, T, ...) — time axis is dim 1.
        freq_norm: normalized frequency in [0, 0.5] (cycles/timestep).
        amplitude: peak amplitude of the sinusoid.
        rng: not used (deterministic perturbation).

    Returns:
        perturbed array, same shape as x.
    """
    B, T = x.shape[0], x.shape[1]
    t = np.arange(T, dtype=np.float32)
    # Random phase per sample for realism
    phase = rng.uniform(0, 2 * np.pi, size=(B, 1))
    sinusoid = amplitude * np.sin(2 * np.pi * freq_norm * t + phase)
    # Broadcast to match feature dimensions
    shape = [1] * x.ndim
    shape[0] = B
    shape[1] = T
    sinusoid = sinusoid.reshape(shape)
    return x + sinusoid


# ---------------------------------------------------------------------------
# Model loading (same pattern as noise_benchmark.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loaders (returning raw numpy for perturbation)
# ---------------------------------------------------------------------------

def _build_gsc_numpy_loader(task_dir: str):
    """Return (x_numpy, y_numpy) for the entire GSC test set."""
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
    loader = DataLoader(test_dataset, batch_size=200, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    x_all, y_all = [], []
    for x, y in loader:
        # Reshape to (B, T, 120) form
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], 101, 120)
        x_all.append(x.numpy())
        y_all.append(y.numpy())
    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)


def _build_ecg_numpy_loader(task_dir: str):
    """Return (x_numpy, y_numpy) for the entire ECG test set."""
    import scipy.io
    from torch.utils.data import DataLoader, TensorDataset

    _add_path(task_dir)
    from core_algorithm.utils import convert_dataset_wtime

    task_abs = os.path.join(ROOT, task_dir)
    test_mat = scipy.io.loadmat(os.path.join(task_abs, "data", "QTDB_test.mat"))
    train_mat = scipy.io.loadmat(os.path.join(task_abs, "data", "QTDB_train.mat"))
    _, test_x, test_y = convert_dataset_wtime(test_mat)
    _, _, train_y = convert_dataset_wtime(train_mat)

    label_min = min(train_y.min(), test_y.min())
    if label_min != 0:
        test_y -= label_min

    return test_x.astype(np.float32), test_y.astype(np.int64)


# ---------------------------------------------------------------------------
# Evaluation on numpy data
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_gsc_np(model, x_np, y_np, device, batch_size=200):
    """Evaluate GSC model on numpy (N, 101, 120) data."""
    total_correct, total = 0, len(y_np)
    for i in range(0, len(y_np), batch_size):
        x_b = torch.from_numpy(x_np[i:i + batch_size]).float().to(device)
        y_b = torch.from_numpy(y_np[i:i + batch_size]).long().to(device)
        logits, _ = model(x_b)
        total_correct += (logits.argmax(dim=-1) == y_b).sum().item()
    return total_correct / total


@torch.no_grad()
def _eval_ecg_np(model, x_np, y_np, device, batch_size=64):
    """Evaluate ECG model on numpy (N, T, F) data."""
    total_correct, total = 0, 0
    for i in range(0, len(y_np), batch_size):
        x_b = torch.from_numpy(x_np[i:i + batch_size]).float().to(device)
        y_b = torch.from_numpy(y_np[i:i + batch_size]).long().to(device)
        logits = model(x_b)
        pred = logits.argmax(dim=1)  # (B, T) — per-timestep class
        total_correct += pred.eq(y_b).sum().item()
        total += y_b.numel()
    return total_correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

FREQUENCIES = [
    ("f1_0.01pi", 0.005),   # 0.01 * pi / (2*pi) ... wait, freq_norm is in cycles/timestep
    ("f2_0.05pi", 0.025),   # 0.05 * pi / (2*pi)
    ("f3_0.10pi", 0.050),   # 0.10 * pi / (2*pi)
    ("f4_0.25pi", 0.125),   # 0.25 * pi / (2*pi)
    ("f5_0.50pi", 0.250),   # 0.50 * pi / (2*pi)
]
AMPLITUDES = [
    ("low", 0.02),
    ("med", 0.05),
    ("high", 0.10),
]


def run_frequency_experiment(
    dataset_name: str,
    spare_model, lif_model,
    x_clean: np.ndarray,
    y: np.ndarray,
    device,
    eval_fn,
    rng,
) -> List[dict]:
    """Run all frequency x amplitude conditions."""
    results = []

    # Baseline (clean)
    acc_spare_clean = eval_fn(spare_model, x_clean, y, device)
    acc_lif_clean = eval_fn(lif_model, x_clean, y, device)
    print(f"  Clean — SPRiF: {acc_spare_clean:.4f}  LIF: {acc_lif_clean:.4f}")

    for freq_label, freq_norm in FREQUENCIES:
        for amp_label, amplitude in AMPLITUDES:
            x_pert = sinusoidal_perturbation(x_clean.copy(), freq_norm, amplitude, rng)
            acc_s = eval_fn(spare_model, x_pert, y, device)
            acc_l = eval_fn(lif_model, x_pert, y, device)
            d_s = acc_spare_clean - acc_s
            d_l = acc_lif_clean - acc_l

            results.append({
                "dataset": dataset_name,
                "frequency": freq_label,
                "amplitude": amp_label,
                "SPRiF_delta_acc": round(d_s, 4),
                "LIF_delta_acc": round(d_l, 4),
            })
            print(f"    {freq_label} {amp_label}: SPRiF Δ={d_s:.4f}  LIF Δ={d_l:.4f}")

    return results, acc_spare_clean, acc_lif_clean


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_frequency_selectivity(all_results: List[dict], out_dir: str):
    """Heatmap-style line plot showing accuracy degradation vs frequency."""
    datasets = sorted(set(r["dataset"] for r in all_results))
    freq_labels = [f[0] for f in FREQUENCIES]
    amp_labels = [a[0] for a in AMPLITUDES]

    n_cols = len(datasets)
    n_rows = len(amp_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, ds in enumerate(datasets):
        ds_results = [r for r in all_results if r["dataset"] == ds]
        for row_idx, amp in enumerate(amp_labels):
            ax = axes[row_idx, col_idx]
            amp_results = [r for r in ds_results if r["amplitude"] == amp]
            by_freq = {r["frequency"]: r for r in amp_results}

            x = np.arange(len(freq_labels))
            sprim_d = [by_freq[f]["SPRiF_delta_acc"] for f in freq_labels]
            lif_d = [by_freq[f]["LIF_delta_acc"] for f in freq_labels]

            ax.plot(x, sprim_d, color="#2b83ba", marker="o", linewidth=2,
                    markersize=8, label="SPRiF")
            ax.plot(x, lif_d, color="#e41a1c", marker="s", linewidth=2,
                    markersize=8, label="LIF")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(freq_labels, fontsize=8, rotation=30)
            ax.set_ylabel("Δ Accuracy (clean − noisy)")
            if row_idx == 0:
                ax.set_title(f"{ds}  (amplitude={amp})", fontweight="bold")
            else:
                ax.set_title(f"amplitude={amp}")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True, fontsize=9)

    fig.suptitle("SPRiF vs LIF Frequency Selectivity", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "frequency_selectivity.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "GSC": {
        "task_dir": "Task_GSC",
        "load_models": _load_gsc_models,
        "build_numpy_loader": _build_gsc_numpy_loader,
        "eval_fn": _eval_gsc_np,
    },
    "QTDB": {
        "task_dir": "Task_ECG",
        "load_models": _load_ecg_models,
        "build_numpy_loader": _build_ecg_numpy_loader,
        "eval_fn": _eval_ecg_np,
    },
}


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(42)

    all_results = []

    for ds_name, config in DATASET_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            print("Loading models...")
            sprim, lifm = config["load_models"](config["task_dir"])
            sprim.to(device)
            lifm.to(device)

            print("Loading test data...")
            x_clean, y = config["build_numpy_loader"](config["task_dir"])

            results, clean_s, clean_l = run_frequency_experiment(
                ds_name, sprim, lifm, x_clean, y,
                device, config["eval_fn"], rng,
            )
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"  SKIP {ds_name}: {e}")
            continue

    if not all_results:
        print("\nNo datasets evaluated. Aborting.")
        return

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "frequency_selectivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    print("Plotting...")
    plot_frequency_selectivity(all_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
