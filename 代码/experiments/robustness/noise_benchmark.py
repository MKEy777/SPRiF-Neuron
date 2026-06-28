"""
SPRiF Robustness Experiment R1: Noise Robustness Benchmark.

Methodology (following DGN ICLR 2026): Train on CLEAN data only,
test on noisy data. Compares SPRiF vs LIF accuracy degradation.

Noise types:
    - Additive Gaussian: x_noisy = x + N(0, sigma^2)
    - Subtractive dropout: randomly zero out fraction p of elements
    - Mixed: additive + subtractive simultaneously

Datasets: GSC, QTDB.

Usage:
    cd 代码/experiments
    python robustness/noise_benchmark.py
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
    os.path.dirname(ROOT), "experiment-design-20260606", "results"
)


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
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def seed_generator(seed: int = 42):
    """Deterministic RNG for reproducible noise across models."""
    return np.random.RandomState(seed)


def additive_noise(x: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """Add Gaussian noise N(0, sigma^2). Does NOT modify input."""
    noise = rng.randn(*x.shape).astype(np.float32) * sigma
    return x + noise


def subtractive_noise(x: np.ndarray, p: float, rng: np.random.RandomState) -> np.ndarray:
    """Randomly zero out fraction p of elements."""
    mask = rng.rand(*x.shape) > p
    return x * mask.astype(np.float32)


def mixed_noise(x: np.ndarray, sigma: float, p: float, rng: np.random.RandomState) -> np.ndarray:
    """Additive + subtractive combined."""
    x = additive_noise(x, sigma, rng)
    x = subtractive_noise(x, p, rng)
    return x


# ---------------------------------------------------------------------------
# GSC model loading
# ---------------------------------------------------------------------------

def _load_gsc_models(task_dir: str):
    """Load both SPRiF and LIF GSC models. Train if needed."""
    _add_path(task_dir)
    from model import SPRiFGSCNet
    from model_lif import LIFGSCNet

    task_abs = os.path.join(ROOT, task_dir)
    data_root = os.path.join(task_abs, "data", "SpeechCommands")

    models = {}

    for cls, prefix in [(SPRiFGSCNet, "SPRiFGSCNet"), (LIFGSCNet, "LIFGSCNet")]:
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            train_script = "train.py" if prefix.startswith("SPRiF") else "train_lif.py"
            if not os.path.exists(data_root):
                raise FileNotFoundError(
                    f"GSC data not found at {data_root}. Run Task_GSC/download_GSC.py first."
                )
            _train_task(
                task_dir, train_script,
                [
                    "--data-root", os.path.join("data", "SpeechCommands"),
                    "--lr", "3e-3", "--epochs", "150", "--batch-size", "200",
                    "--seed", "42", "--hidden-sizes", "300",
                    "--neuron-threshold", "1.0", "--neuron-init-std", "0.1",
                ],
            )
            ckpt = _find_checkpoint(task_abs, prefix)
            if ckpt is None:
                raise RuntimeError(f"Training completed but no checkpoint found for {prefix}.")

        print(f"  {prefix}: {os.path.basename(ckpt)}")
        model = cls(
            input_size=120, hidden_sizes=[300], num_classes=12,
            dropout=0.0, recurrent_flags=(True,),
            neuron_kwargs={"threshold": 1.0, "init_std": 0.1},
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        models[prefix] = model

    return models["SPRiFGSCNet"], models["LIFGSCNet"]


def _build_gsc_test_loader(task_dir: str, batch_size: int = 200):
    """Build GSC test DataLoader."""
    import torchvision
    from torch.utils.data import DataLoader
    from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

    task_abs = os.path.join(ROOT, task_dir)
    data_root = os.path.join(task_abs, "data", "SpeechCommands")

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
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# ECG model loading
# ---------------------------------------------------------------------------

def _load_ecg_models(task_dir: str):
    """Load both SPRiF and LIF ECG models. Train if needed."""
    import scipy.io
    from core_algorithm.utils import convert_dataset_wtime

    _add_path(task_dir)
    from model import SPRiFECGModel
    from model_lif import LIFECGModel

    task_abs = os.path.join(ROOT, task_dir)
    train_mat_path = os.path.join(task_abs, "data", "QTDB_train.mat")
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError("ECG data not found.")

    mat = scipy.io.loadmat(train_mat_path)
    _, train_x, _ = convert_dataset_wtime(mat)
    input_size = train_x.shape[2]

    models = {}

    for cls, prefix in [(SPRiFECGModel, "SPRiFECGModel"), (LIFECGModel, "LIFECGModel")]:
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            train_script = "train.py" if prefix.startswith("SPRiF") else "train_lif.py"
            _train_task(
                task_dir, train_script,
                [
                    "--train-mat", os.path.join("data", "QTDB_train.mat"),
                    "--test-mat", os.path.join("data", "QTDB_test.mat"),
                    "--lr", "1e-2", "--epochs", "250", "--batch-size", "64",
                    "--seed", "1111", "--hidden-sizes", "36",
                    "--neuron-threshold", "0.6",
                ],
            )
            ckpt = _find_checkpoint(task_abs, prefix)
            if ckpt is None:
                raise RuntimeError(f"Training completed but no checkpoint found for {prefix}.")

        print(f"  {prefix}: {os.path.basename(ckpt)}")
        model = cls(
            input_size=input_size, hidden_sizes=[36], output_size=6,
            mode="srnn",
            neuron_kwargs={"threshold": 0.6, "init_std": 0.05, "bias": True},
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        models[prefix] = model

    return models["SPRiFECGModel"], models["LIFECGModel"]


def _build_ecg_test_loader(task_dir: str, batch_size: int = 64):
    """Build ECG test DataLoader."""
    import scipy.io
    from torch.utils.data import DataLoader, TensorDataset
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
def evaluate_gsc(model: nn.Module, loader, device, seq_len=101, n_mels=40, input_size=120):
    """Evaluate GSC model. Returns accuracy (float)."""
    model.eval()
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.view(-1, 3, seq_len, n_mels).to(device)
        x = x.permute(0, 2, 1, 3).reshape(-1, seq_len, input_size)
        y = y.to(device)
        logits, _ = model(x)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_ecg(model: nn.Module, loader, device):
    """Evaluate ECG model. Returns accuracy (float)."""
    model.eval()
    total_correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        total_correct += pred.eq(labels).sum().item()
        total += labels.numel()
    return total_correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# GSC noisy evaluation (noise applied to raw numpy before collate + reshape)
# ---------------------------------------------------------------------------

class NoisyGSCLoader:
    """Wraps a GSC test DataLoader and applies noise to the raw audio features.

    Noise is applied to the collated tensor (B, 3, seq_len, n_mels) after
    collate_fn normalisation. Outputs raw (B, 3, seq_len, n_mels) — the
    evaluate_gsc function handles the view/permute/reshape internally.
    """

    def __init__(self, base_loader, noise_fn, noise_args, rng):
        self.base_loader = base_loader
        self.noise_fn = noise_fn
        self.noise_args = noise_args
        self.rng = rng
        self._batches = list(base_loader)

    def __iter__(self):
        for x, y in self._batches:
            # x: (B, 3, seq_len, n_mels) after collate_fn
            x_np = x.numpy()
            x_np = self.noise_fn(x_np, *self.noise_args, self.rng)
            x = torch.from_numpy(x_np).float()
            yield x, y

    def __len__(self):
        return len(self._batches)


class NoisyECGLoader:
    """Wraps an ECG test DataLoader and applies noise to the raw signal."""

    def __init__(self, base_loader, noise_fn, noise_args, rng):
        self.base_loader = base_loader
        self.noise_fn = noise_fn
        self.noise_args = noise_args
        self.rng = rng
        self._batches = list(base_loader)

    def __iter__(self):
        for x, y in self._batches:
            x_np = x.numpy()
            x_np = self.noise_fn(x_np, *self.noise_args, self.rng)
            x = torch.from_numpy(x_np).float()
            yield x, y

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

NOISE_CONDITIONS = [
    # (label, noise_fn, args)
    ("clean", None, ()),
    ("additive_0.01", additive_noise, (0.01,)),
    ("additive_0.05", additive_noise, (0.05,)),
    ("additive_0.10", additive_noise, (0.10,)),
    ("subtractive_0.05", subtractive_noise, (0.05,)),
    ("subtractive_0.10", subtractive_noise, (0.10,)),
    ("subtractive_0.20", subtractive_noise, (0.20,)),
    ("mixed", mixed_noise, (0.05, 0.10)),
]

DATASET_CONFIG = {
    "GSC": {
        "task_dir": "Task_GSC",
        "load_models": _load_gsc_models,
        "build_loader": _build_gsc_test_loader,
        "evaluate": evaluate_gsc,
        "noisy_loader_cls": NoisyGSCLoader,
        "loader_kwargs": {"seq_len": 101, "n_mels": 40, "input_size": 120},
    },
    "QTDB": {
        "task_dir": "Task_ECG",
        "load_models": _load_ecg_models,
        "build_loader": _build_ecg_test_loader,
        "evaluate": evaluate_ecg,
        "noisy_loader_cls": NoisyECGLoader,
        "loader_kwargs": {},
    },
}


def run_benchmark(dataset_name: str, config: dict, out_dir: str) -> List[dict]:
    """Run all noise conditions for one dataset. Returns list of result dicts."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    sprif_model, lif_model = config["load_models"](config["task_dir"])
    sprif_model.to(device)
    lif_model.to(device)

    # Build base loader
    base_loader = config["build_loader"](config["task_dir"])

    results = []
    rng = seed_generator(42)  # same noise seed for both models

    for label, noise_fn, args in NOISE_CONDITIONS:
        print(f"  Condition: {label}")
        if noise_fn is None:
            # Clean — evaluate directly
            acc_sprif = config["evaluate"](sprif_model, base_loader, device, **config["loader_kwargs"])
            acc_lif = config["evaluate"](lif_model, base_loader, device, **config["loader_kwargs"])
        else:
            noisy_cls = config["noisy_loader_cls"]
            noisy_loader = noisy_cls(base_loader, noise_fn, args, rng)
            acc_sprif = config["evaluate"](sprif_model, noisy_loader, device, **config["loader_kwargs"])
            acc_lif = config["evaluate"](lif_model, noisy_loader, device, **config["loader_kwargs"])

        results.append({
            "dataset": dataset_name,
            "condition": label,
            "SPRiF_accuracy": round(acc_sprif, 4),
            "LIF_accuracy": round(acc_lif, 4),
        })
        print(f"    SPRiF: {acc_sprif:.4f}  |  LIF: {acc_lif:.4f}")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_benchmark(all_results: List[dict], out_dir: str):
    """Grouped bar chart comparing SPRiF vs LIF degradation per condition per dataset."""
    df_data = []
    for r in all_results:
        df_data.append(r)
    # Reorganize for grouped bar chart
    datasets = sorted(set(r["dataset"] for r in all_results))

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    conditions_order = [c[0] for c in NOISE_CONDITIONS]
    condition_labels = [c.replace("_", "\n") for c in conditions_order]

    colors = {"SPRiF": "#2b83ba", "LIF": "#e41a1c"}
    bar_width = 0.35

    for ax, ds in zip(axes, datasets):
        ds_results = [r for r in all_results if r["dataset"] == ds]
        by_cond = {r["condition"]: r for r in ds_results}

        x = np.arange(len(conditions_order))
        sprif_vals = [by_cond[c]["SPRiF_accuracy"] for c in conditions_order]
        lif_vals = [by_cond[c]["LIF_accuracy"] for c in conditions_order]

        bars1 = ax.bar(x - bar_width / 2, sprif_vals, bar_width, label="SPRiF",
                       color=colors["SPRiF"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + bar_width / 2, lif_vals, bar_width, label="LIF",
                       color=colors["LIF"], alpha=0.85, edgecolor="white")

        # Annotate degradation from clean
        clean_sprif = by_cond["clean"]["SPRiF_accuracy"]
        clean_lif = by_cond["clean"]["LIF_accuracy"]
        for i, c in enumerate(conditions_order):
            if c == "clean":
                continue
            d_sprif = clean_sprif - sprif_vals[i]
            d_lif = clean_lif - lif_vals[i]
            ax.text(x[i], sprif_vals[i] + 0.01, f"-{d_sprif:.3f}",
                    ha="center", fontsize=6, color=colors["SPRiF"], fontweight="bold")
            ax.text(x[i], lif_vals[i] + 0.01, f"-{d_lif:.3f}",
                    ha="center", fontsize=6, color=colors["LIF"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(ds, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.suptitle("SPRiF vs LIF Noise Robustness Benchmark", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "robustness_benchmark.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for ds_name, config in DATASET_CONFIG.items():
        try:
            results = run_benchmark(ds_name, config, out_dir)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"  SKIP {ds_name}: {e}")
            continue

    if not all_results:
        print("\nNo datasets evaluated. Aborting.")
        return

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "robustness_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    print("Plotting...")
    plot_benchmark(all_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
