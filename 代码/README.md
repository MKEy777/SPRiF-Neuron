# SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron Networks

Official implementation of SPRiF neuron networks across multiple benchmark tasks.

## Architecture

SPRiF (Spectral Projective Reset Integrate-and-Fire) is a bio-inspired spiking neuron model combining:

- **Slow dynamics (3-state)**: one leaky integrator + one damped oscillatory pair
- **Fast dynamics (2-state)**: slow-to-fast coupling driving per-neuron leakage
- **Surrogate gradient**: multi-Gaussian surrogate for differentiable spike training

## Directory Structure

```
├── Task_GSC/                # Google Speech Commands (12-class keyword spotting)
├── Task_ECG/                # QTDB ECG classification (6-class heartbeat)
├── Task_S-MNIST/            # Sequential MNIST (10-digit classification, row-by-row)
├── Task_pSMNIST/            # Permuted Sequential MNIST (10-digit, random pixel order)
├── Task_SHD/                # Spiking Heidelberg Digits (20-class digit recognition)
├── experiments/             # Analysis scripts for paper figures
│   ├── impulse_analysis/    # Impulse response kernel analysis
│   ├── reset_analysis/      # Learned reset direction analysis
│   ├── robustness/          # Noise robustness benchmarking
│   ├── trajectory_analysis/ # Slow-state trajectory visualization
│   ├── loss_landscape/      # Loss landscape visualization
│   └── SI-DMS/              # Spike-intervention delayed match-to-sample
├── requirements.txt
├── LICENSE
└── README.md
```

Each task directory is **self-contained** and includes its own `core_algorithm/` with the shared SPRiF layer implementation.

## Dependencies

```bash
pip install -r requirements.txt
```

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- scipy >= 1.5.0
- librosa >= 0.8.0
- torchaudio >= 0.10.0
- tables >= 3.6.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- PyYAML >= 5.4
- wfdb >= 4.1.0 (for QTDB preprocessing)
- soundfile >= 0.12.0

## Running Experiments

### Google Speech Commands (GSC)

```bash
cd Task_GSC
python download_GSC.py
python train.py
```

### ECG Classification (QTDB)

```bash
cd Task_ECG
python generate_qtdb_dataset.py --download
python train.py --train-mat ./data/QTDB_train.mat --test-mat ./data/QTDB_test.mat
```

### Permuted Sequential MNIST (pSMNIST)

```bash
cd Task_pSMNIST
python train.py
```

### Sequential MNIST (S-MNIST)

```bash
cd Task_S-MNIST
python train.py
```

### Spiking Heidelberg Digits (SHD)

Download the SHD dataset (shd_train.h5, shd_test.h5) from https://zenodo.org/records/7969702 and place them under `data/SHD/`:

```bash
mkdir -p data/SHD
# Place shd_train.h5 and shd_test.h5 in data/SHD/

cd Task_SHD
python generate_data.py
python train.py --train-dir ../data/SHD/train_1ms --test-dir ../data/SHD/test_1ms
```

### Analysis Experiments

```bash
# Impulse response kernel analysis
cd experiments/impulse_analysis
python impulse_analyze.py

# Learned reset direction analysis
cd experiments/reset_analysis
python reset_analyze.py

# Noise robustness benchmarking
cd experiments/robustness
python noise_benchmark.py

# Slow-state trajectory visualization
cd experiments/trajectory_analysis
python trajectory_analyze.py

# Loss landscape visualization
cd experiments/loss_landscape
python run_all.py

# SI-DMS: spike-intervention delayed match-to-sample
cd experiments/SI-DMS
python train.py --model sprif_full
python run_all.py           # train all models with all seeds
python evaluate.py          # evaluate trained models
python aggregate.py         # aggregate results into table/figure data
```

## Model Checkpoints

Saved model naming format: `[NetworkPrefix]_[KeyHParams]_seed[N]_acc[XX.XX].pth`

Checkpoints are saved in the task root directory during training.

## Citation

If you find this work useful for your research, please cite our paper.
