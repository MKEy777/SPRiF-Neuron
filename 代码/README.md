# SPRiF: Slow-Path Resonance with Intrinsic Fast-flow Spiking Neuron Networks

This repository contains the official implementation of SPRiF neuron networks for multiple benchmark tasks.

## Architecture

SPRiF (Slow-Path Resonance with Intrinsic Fast-flow) is a bio-inspired spiking neuron model that combines:

- **Slow dynamics** (3-state): One leaky integrator + one damped oscillatory pair
- **Fast dynamics** (2-state): Driven by slow-to-fast coupling with per-neuron leakage
- **Surrogate gradient**: Multi-Gaussian surrogate for differentiable spike training

## Directory Structure

```
SPRiF_Paper/
├── core_algorithm/          # Shared algorithm library (copied into each task)
│   ├── sprif_layer.py       # SPRiFNeuronLayer + surrogate gradient
│   └── utils.py             # set_seed, dump_json, load_json, convert_dataset_wtime
│
├── Task_GSC/                # Google Speech Commands (12-class keyword spotting)
├── Task_ECG/                # QTDB ECG classification (6-class heartbeat)
├── Task_S-MNIST/            # Sequential MNIST (10-digit classification, row-by-row order)
├── Task_pSMNIST/            # Permuted Sequential MNIST (10-digit classification, random pixel order)
└── Task_SHD/                # Spiking Heidelberg Digits (20-class digit recognition)
```

Each task directory is **fully self-contained** and can be run independently.

## Requirements

```bash
pip install -r requirements.txt
```

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- scipy >= 1.5.0
- librosa >= 0.8.0

## Running Experiments

### Task GSC (Google Speech Commands)

```bash
cd Task_GSC
python train.py --data-root /path/to/speech_commands_v0.02
```

Optional: `--cache-root /path/to/cache` to specify a cache directory.

### Task ECG (QTDB ECG)

```bash
cd Task_ECG
python train.py --train-mat ./data/QTDB_train.mat --test-mat ./data/QTDB_test.mat
```

### Task pSMNIST (Permuted Sequential MNIST)

```bash
cd Task_pSMNIST
python train.py
```

MNIST dataset will be downloaded automatically on first run.

### Task S-MNIST (Sequential MNIST)

```bash
cd Task_S-MNIST
python train.py
```

Pixels are read in natural row-by-row order (no permutation). Same hyperparameters as pSMNIST.
Ablation variants: `python train_ablation_{a,b,c}.py`.

### Task SHD (Spiking Heidelberg Digits)

**Step 1**: Preprocess the SHD dataset (HDF5 → NPY):
```bash
cd Task_SHD
python generate_data.py
```

**Step 2**: Train the model:
```bash
python train.py --train-dir /path/to/train_1ms --test-dir /path/to/test_1ms
```

## Model Checkpoints

Models are automatically saved with the naming convention:

```
[NetworkPrefix]_[KeyHParams]_seed[N]_acc[XX.XX].pth
```

Example: `SPRiFGSCNet_hs300_bs200_lr0.003_seed42_acc92.50.pth`

## Citation

If you use this code in your research, please cite the paper.
