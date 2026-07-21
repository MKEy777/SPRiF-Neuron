# SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron Networks

Official implementation of SPRiF neuron networks across multiple benchmark tasks.

## Architecture

SPRiF (Spectral Projective Reset Integrate-and-Fire) is a bio-inspired spiking neuron model combining:

- **Slow dynamics (3-state)**: one leaky integrator + one damped oscillatory pair
- **Fast dynamics (2-state)**: slow-to-fast coupling driving per-neuron leakage
- **Surrogate gradient**: multi-Gaussian surrogate for differentiable spike training

## Directory Structure

```
├── core_algorithm/          # Shared algorithm library (copied to each task)
│   ├── sprif_layer.py       # SPRiFNeuronLayer + surrogate gradient
│   └── utils.py             # set_seed, dump_json, load_json, convert_dataset_wtime
│
├── Task_GSC/                # Google Speech Commands (12-class keyword spotting)
├── Task_ECG/                # QTDB ECG classification (6-class heartbeat)
├── Task_S-MNIST/            # Sequential MNIST (10-digit classification, row-by-row)
├── Task_pSMNIST/            # Permuted Sequential MNIST (10-digit, random pixel order)
├── Task_SHD/                # Spiking Heidelberg Digits (20-class digit recognition)
└── experiments/             # Analysis scripts for paper figures
    ├── impulse_analysis/    # Impulse response kernel analysis
    ├── reset_analysis/      # Learned reset direction analysis
    ├── robustness/          # Noise robustness benchmarking
    ├── trajectory_analysis/ # Slow-state trajectory visualization
    ├── loss_landscape/      # Loss landscape visualization
    └── SI-DMS/              # Spike-intervention delayed match-to-sample
```

Each task directory is **self-contained** and can be run independently.

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

```bash
cd Task_SHD
python generate_data.py
python train.py --train-dir ../../data/SHD/train_1ms --test-dir ../../data/SHD/test_1ms
```

## Model Checkpoints

Saved model naming format: `[NetworkPrefix]_[KeyHParams]_seed[N]_acc[XX.XX].pth`

Checkpoints are saved in the task root directory during training.

## Citation

If you find this work useful for your research, please cite our paper.
