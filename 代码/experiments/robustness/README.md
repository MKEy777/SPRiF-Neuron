# SPRiF Robustness Experiments

Inspired by DGN (ICLR 2026) methodology: train on clean data, test on noisy data.

## Datasets
- **GSC** (Google Speech Commands v2) — 12-class keyword spotting, 101-frame mel-spectrograms
- **QTDB** (QT Database) — 6-class ECG heartbeat classification

## Experiments

### R1: Noise Robustness Benchmark (`noise_benchmark.py`)

Compares SPRiF vs LIF accuracy under:
- **Additive Gaussian** noise: σ ∈ {0.01, 0.05, 0.10}
- **Subtractive dropout**: p ∈ {0.05, 0.10, 0.20}
- **Mixed**: σ=0.05 + p=0.10

Output: `robustness/robustness_benchmark.png` + `experiment-design-20260606/results/robustness_benchmark.json`

### R2: Sequence Length x Noise (`sequence_noise.py`)

Varies sequence length with fixed additive noise (σ=0.05):
- GSC: truncate to 50, 75, 101 frames
- QTDB: resample to 150, 300, 600 steps + original

Output: `robustness/sequence_noise.png` + `experiment-design-20260606/results/sequence_noise.json`

### R3: Frequency Selectivity (`frequency_selectivity.py`)

SPRiF-unique experiment. Injects sinusoidal perturbations at:
- 5 frequencies: 0.01π, 0.05π, 0.10π, 0.25π, 0.50π (normalized)
- 3 amplitudes: low (0.02), medium (0.05), high (0.10)

Output: `robustness/frequency_selectivity.png` + `experiment-design-20260606/results/frequency_selectivity.json`

## LIF Baseline

The LIF neuron layer was implemented alongside these experiments and is available for training:

- `代码/Task_GSC/core_algorithm/lif_layer.py` + `model_lif.py` + `train_lif.py`
- `代码/Task_ECG/core_algorithm/lif_layer.py` + `model_lif.py` + `train_lif.py`

All robustness scripts include training fallback: if no checkpoint is found, the model is trained automatically.

## Usage

```bash
cd 代码/experiments
python robustness/noise_benchmark.py
python robustness/sequence_noise.py
python robustness/frequency_selectivity.py
```

## Key design choices

- Same noise generator seed (42) for SPRiF and LIF — fair comparison
- Noise applied to raw features before model-specific preprocessing
- All evaluations run on the entire test set
- LIF uses the same architecture, hyperparameters, and tau_m range (matching SPRiF alpha range) for fair comparison
