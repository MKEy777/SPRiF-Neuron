
import glob
import json
import math
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(
    os.path.dirname(ROOT), "experiment-design-20260606", "results"
)

def _add_path(task_dir: str) -> str:
    p = os.path.join(ROOT, task_dir)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for name in list(sys.modules):
        if (
            name in {"model", "model_wrapper", "data", "data_asrnn", "core_algorithm"}
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

def seed_generator(seed: int = 42):
    return np.random.RandomState(seed)

def additive_noise(x: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    noise = rng.randn(*x.shape).astype(np.float32) * sigma
    return x + noise

def subtractive_noise(x: np.ndarray, p: float, rng: np.random.RandomState) -> np.ndarray:
    mask = rng.rand(*x.shape) > p
    return x * mask.astype(np.float32)

def mixed_noise(x: np.ndarray, sigma: float, p: float, rng: np.random.RandomState) -> np.ndarray:
    x = additive_noise(x, sigma, rng)
    x = subtractive_noise(x, p, rng)
    return x

def _load_gsc_models(task_dir: str):
    _add_path(task_dir)
    from model import SPRiFGSCNet

    sys.path.insert(0, os.path.join(ROOT, task_dir, "model_wrapper"))
    from asrnn_gsc import ASRNNGSCNet

    task_abs = os.path.join(ROOT, task_dir)
    _gsc_candidates = [
        os.path.join(task_abs, "dataset", "SpeechCommands", "speech_commands_v0.02"),
        os.path.join(task_abs, "dataset", "SpeechCommands", "speech_commands_v0.01"),
        os.path.join(task_abs, "data", "SpeechCommands"),
    ]
    data_root = next((p for p in _gsc_candidates if os.path.exists(p)), _gsc_candidates[0])

    models = {}

    prefix = "SPRiFGSCNet"
    ckpt = _find_checkpoint(task_abs, prefix)
    if ckpt is None:
        if not os.path.exists(data_root):
            raise FileNotFoundError(
                f"GSC data not found at {data_root}. Run Task_GSC/download_GSC.py first."
            )
        _train_task(
            task_dir, "train.py",
            [
                "--data-root", os.path.relpath(data_root, task_abs),
                "--lr", "5e-3", "--epochs", "150", "--batch-size", "200",
                "--seed", "42", "--hidden-sizes", "300",
                "--neuron-threshold", "0.8", "--neuron-init-std", "0.15",
                "--tau-alpha-min", "10.0", "--tau-alpha-max", "80.0",
                "--tau-rho-min", "4.0", "--tau-rho-max", "30.0",
                "--tau-eta-min", "0.8", "--tau-eta-max", "8.0",
                "--omega-min", "0.04", "--omega-max", "0.40",
            ],
        )
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for SPRiFGSCNet.")

    print(f"  {prefix}: {os.path.basename(ckpt)}")
    sprif_model = SPRiFGSCNet(
        input_size=120, hidden_sizes=[300], num_classes=12,
        dropout=0.15, recurrent_flags=(True,),
        neuron_kwargs={
            "threshold": 0.8, "init_std": 0.1,
            "tau_alpha_range": (10.0, 80.0),
            "tau_rho_range": (4.0, 30.0),
            "tau_eta_range": (0.8, 8.0),
            "omega_range": (0.04 * math.pi, 0.40 * math.pi),
        },
    )
    sprif_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    sprif_model.eval()
    models["SPRiFGSCNet"] = sprif_model

    ckpt = _find_checkpoint(task_abs, "ASRNNGSCNet")
    if ckpt is None:
        if not os.path.exists(data_root):
            raise FileNotFoundError(
                f"GSC data not found at {data_root}. Run Task_GSC/download_GSC.py first."
            )
        _train_task(
            task_dir, "train_asrnn.py",
            [
                "--data-root", os.path.relpath(data_root, task_abs),
                "--lr", "3e-3", "--epochs", "30", "--batch-size", "32",
                "--seed", "0", "--hidden-size", "256",
            ],
        )
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

def _build_gsc_test_loader(task_dir: str, batch_size: int = 200):
    import torchvision
    from torch.utils.data import DataLoader

    _add_path(task_dir)
    from data import MelSpectrogram, Pad, Rescale, SpeechCommandsDataset

    task_abs = os.path.join(ROOT, task_dir)
    _gsc_candidates = [
        os.path.join(task_abs, "dataset", "SpeechCommands", "speech_commands_v0.02"),
        os.path.join(task_abs, "dataset", "SpeechCommands", "speech_commands_v0.01"),
        os.path.join(task_abs, "data", "SpeechCommands"),
    ]
    data_root = next((p for p in _gsc_candidates if os.path.exists(p)), _gsc_candidates[0])

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

def _load_ecg_models(task_dir: str):
    import scipy.io

    _add_path(task_dir)
    from core_algorithm.utils import convert_dataset_wtime

    from model import SPRiFECGModel

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

    prefix = "SPRiFECGModel"
    ckpt = _find_checkpoint(task_abs, prefix)
    if ckpt is None:
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
        ckpt = _find_checkpoint(task_abs, prefix)
        if ckpt is None:
            raise RuntimeError("Training completed but no checkpoint found for SPRiFECGModel.")

    print(f"  {prefix}: {os.path.basename(ckpt)}")
    sprif_model = SPRiFECGModel(
        input_size=input_size, hidden_sizes=[36], output_size=6,
        mode="srnn",
        neuron_kwargs={"threshold": 0.6, "init_std": 0.05, "bias": True},
    )
    sprif_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    sprif_model.eval()
    models["SPRiFECGModel"] = sprif_model

    ckpt = _find_checkpoint(task_abs, "ASRNNECGModel")
    if ckpt is None:
        _train_task(
            task_dir, "train_asrnn.py",
            [
                "--train-mat", os.path.join("data", "QTDB_train.mat"),
                "--test-mat", os.path.join("data", "QTDB_test.mat"),
                "--lr", "1e-2", "--epochs", "250", "--batch-size", "64",
                "--seed", "1111", "--hidden-size", "36",
            ],
        )
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

def _build_ecg_test_loader(task_dir: str, batch_size: int = 64):
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

@torch.no_grad()
def evaluate_gsc(model: nn.Module, loader, device, seq_len=101, n_mels=40, input_size=120, model_name="SPRiF"):
    model.eval()
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.view(-1, 3, seq_len, n_mels).to(device)
        y = y.to(device)

        if model_name == "ASRNN":

            logits = model(x)
        else:

            x = x.permute(0, 2, 1, 3).reshape(-1, seq_len, input_size)
            logits, _ = model(x)

        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_correct / total if total > 0 else 0.0

@torch.no_grad()
def evaluate_ecg(model: nn.Module, loader, device, model_name="SPRiF"):
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

class NoisyGSCLoader:

    def __init__(self, base_loader, noise_fn, noise_args, rng):
        self.base_loader = base_loader
        self.noise_fn = noise_fn
        self.noise_args = noise_args
        self.rng = rng
        self._batches = []
        for x, y in base_loader:
            x_np = x.numpy()
            x_np = self.noise_fn(x_np, *self.noise_args, self.rng)
            self._batches.append((torch.from_numpy(x_np).float(), y))

    def __iter__(self):
        for x, y in self._batches:
            yield x, y

    def __len__(self):
        return len(self._batches)

class NoisyECGLoader:

    def __init__(self, base_loader, noise_fn, noise_args, rng):
        self.base_loader = base_loader
        self.noise_fn = noise_fn
        self.noise_args = noise_args
        self.rng = rng
        self._batches = []
        for x, y in self.base_loader:
            x_np = x.numpy()
            x_np = self.noise_fn(x_np, *self.noise_args, self.rng)
            self._batches.append((torch.from_numpy(x_np).float(), y))

    def __iter__(self):
        for x, y in self._batches:
            yield x, y

    def __len__(self):
        return len(self._batches)

NOISE_CONDITIONS = [

    ("clean", None, ()),
    ("additive_sigma_0.01", additive_noise, (0.01,)),
    ("additive_sigma_0.05", additive_noise, (0.05,)),
    ("additive_sigma_0.10", additive_noise, (0.10,)),
    ("subtractive_p_0.05", subtractive_noise, (0.05,)),
    ("subtractive_p_0.10", subtractive_noise, (0.10,)),
    ("subtractive_p_0.20", subtractive_noise, (0.20,)),
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

def _sync_module_device(model: nn.Module, device) -> None:
    dev_str = str(device)
    for m in model.modules():
        if hasattr(m, "device"):
            m.device = dev_str

def run_benchmark(dataset_name: str, config: dict, out_dir: str) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    sprif_model, asrnn_model = config["load_models"](config["task_dir"])
    sprif_model.to(device)
    asrnn_model.to(device)

    _sync_module_device(asrnn_model, device)

    base_loader = config["build_loader"](config["task_dir"])

    results = []
    rng = seed_generator(42)

    for label, noise_fn, args in NOISE_CONDITIONS:
        print(f"  Condition: {label}")
        if noise_fn is None:

            acc_sprif = config["evaluate"](sprif_model, base_loader, device, model_name="SPRiF", **config["loader_kwargs"])
            acc_asrnn = config["evaluate"](asrnn_model, base_loader, device, model_name="ASRNN", **config["loader_kwargs"])
        else:
            noisy_cls = config["noisy_loader_cls"]
            noisy_loader = noisy_cls(base_loader, noise_fn, args, rng)
            acc_sprif = config["evaluate"](sprif_model, noisy_loader, device, model_name="SPRiF", **config["loader_kwargs"])
            acc_asrnn = config["evaluate"](asrnn_model, noisy_loader, device, model_name="ASRNN", **config["loader_kwargs"])

        results.append({
            "dataset": dataset_name,
            "condition": label,
            "SPRiF_accuracy": round(acc_sprif, 4),
            "ASRNN_accuracy": round(acc_asrnn, 4),
        })
        print(f"    SPRiF: {acc_sprif:.4f}  |  ASRNN: {acc_asrnn:.4f}")

    return results

def plot_benchmark(all_results: List[dict], out_dir: str):
    datasets = sorted(set(r["dataset"] for r in all_results))

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    conditions_order = [c[0] for c in NOISE_CONDITIONS]
    condition_labels = [c.replace("_", "\n") for c in conditions_order]

    colors = {"SPRiF": "#2b83ba", "ASRNN": "#4daf4a"}
    bar_width = 0.35

    for ax, ds in zip(axes, datasets):
        ds_results = [r for r in all_results if r["dataset"] == ds]
        by_cond = {r["condition"]: r for r in ds_results}

        x = np.arange(len(conditions_order))
        sprif_vals = [by_cond[c]["SPRiF_accuracy"] for c in conditions_order]
        asrnn_vals = [by_cond[c]["ASRNN_accuracy"] for c in conditions_order]

        bars1 = ax.bar(x - bar_width / 2, sprif_vals, bar_width, label="SPRiF",
                       color=colors["SPRiF"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + bar_width / 2, asrnn_vals, bar_width, label="ASRNN",
                       color=colors["ASRNN"], alpha=0.85, edgecolor="white")

        clean_sprif = by_cond["clean"]["SPRiF_accuracy"]
        clean_asrnn = by_cond["clean"]["ASRNN_accuracy"]
        for i, c in enumerate(conditions_order):
            if c == "clean":
                continue
            d_sprif = clean_sprif - sprif_vals[i]
            d_asrnn = clean_asrnn - asrnn_vals[i]
            ax.text(x[i] - bar_width / 2, sprif_vals[i] + 0.01, f"-{d_sprif:.3f}",
                    ha="center", fontsize=6, color=colors["SPRiF"], fontweight="bold")
            ax.text(x[i] + bar_width / 2, asrnn_vals[i] + 0.01, f"-{d_asrnn:.3f}",
                    ha="center", fontsize=6, color=colors["ASRNN"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(ds, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.suptitle("SPRiF vs ASRNN Noise Robustness Benchmark", y=1.01, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "robustness_benchmark.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

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

    json_path = os.path.join(RESULTS_DIR, "robustness_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("Plotting...")
    plot_benchmark(all_results, out_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()

