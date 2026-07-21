# SPRiF Robustness Experiments

Inspired by DGN (ICLR 2026) methodology: train on clean data, test on noisy data.

## Datasets
- **GSC** (Google Speech Commands v2) — 12-class keyword spotting, 101-frame mel-spectrograms
- **QTDB** (QT Database) — 6-class ECG heartbeat classification

## Experiments

### R1: Fixed-Length Noise Robustness Benchmark (`noise_benchmark.py`)

Compares SPRiF vs ASRNN accuracy under fixed-length training and noisy testing. Seven perturbation conditions:

- **Additive Gaussian** noise: σ ∈ {0.01, 0.05, 0.10}
- **Subtractive dropout**: p ∈ {0.05, 0.10, 0.20}
- **Mixed**: σ=0.05 + p=0.10

Output: `robustness/robustness_benchmark.png` + `experiment-design-20260606/results/robustness_benchmark.json`

### R3: Sinusoidal Frequency-Perturbation Sensitivity (`frequency_selectivity.py`)

SPRiF-unique experiment. Injects sinusoidal perturbations at:
- 5 frequencies: 0.01π, 0.05π, 0.10π, 0.25π, 0.50π (normalized)
- 3 amplitudes: low (0.02), medium (0.05), high (0.10)

Output: accuracy change per condition.

Output: `robustness/frequency_selectivity.png` + `experiment-design-20260606/results/frequency_selectivity.json`

## ASRNN Baseline

The ASRNN (Adaptive Spiking RNN) neuron is used as the comparison baseline for robustness experiments.

All robustness scripts include training fallback: if no checkpoint is found, the model is trained automatically.

## Usage

```bash
cd 代码/experiments
python robustness/noise_benchmark.py
python robustness/frequency_selectivity.py
```

## Key Design Choices

- Same noise generator seed (42) for SPRiF and ASRNN — fair comparison
- Noise applied to raw features before model-specific preprocessing
- All evaluations run on the entire test set
