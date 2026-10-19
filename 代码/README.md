# SPRiF Neuron Networks (TNNLS Scalar-Reset Model)

Official implementation of SPRiF neuron networks across multiple benchmark tasks.

## TNNLS branch (2026-10-05)

The five primary `train.py` entries now use the scalar-reset SPRiF model:
three spectral coordinates, one auxiliary filter, and one membrane. The neuron
learns eleven local coefficients; the second row of `bar_G` absorbs the former
coupling scale. Only the membrane is reset, and its drive uses the previous filter
state. New checkpoints have the `TNNLS11_` prefix.

Dataset membership, preprocessing, readout, training budget, and checkpoint
selection are retained. S-MNIST, PS-MNIST, ECG, and SHD use train/test;
the supplied GSC entry uses train/valid and selects on valid. No third split is
introduced. SHD reads NPY files from the supplied train/test directories.

Independent sequences start at zero, as defined by the current manuscript.
The threshold convention is `H(0)=1`; surrogate/reset gradients are retained.
Historical ablation files remain historical models. The reset-direction analysis
requires the original implementation and its checkpoints. New benchmark results
must come from new runs.

SI-DMS registers the new model as `sprif_scalar`, and its default queue contains
that model. Historical IDs retain their original meaning. The planned capacity
controls and paired SI-DMS evaluation protocol remain separate implementation work.

Run synthetic behavior checks from this directory:

```powershell
python -m pytest tests/test_sprif_scalar.py experiments/SI-DMS/tests -q
```

## Architecture

The current SPRiF spiking neuron model combines:

- **Slow dynamics (3-state)**: one leaky integrator + one damped oscillatory pair
- **Auxiliary filter (1-state)**: unreset linear filter with a delayed membrane drive
- **Membrane (1-state)**: scalar voltage with subtractive spike reset
- **Surrogate gradient**: multi-Gaussian surrogate for differentiable spike training

## Directory Structure

```
├── Task_GSC/                # Google Speech Commands (12-class keyword spotting)
├── Task_ECG/                # QTDB ECG classification (6-class heartbeat)
├── Task_S-MNIST/            # Sequential MNIST (784 scalar pixel inputs)
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
