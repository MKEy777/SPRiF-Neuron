# Effective Temporal Kernel Analysis

## Objective
Demonstrate that SPRiF learns diverse temporal filters — unlike LIF's single exponential decay kernel, SPRiF's spectral parameterization enables each neuron to learn distinct impulse responses (fast/slow decay, oscillations).

## Method
1. Load trained SPRiF models (pSMNIST / GSC / ECG)
2. Inject a unit impulse via the `input_current` parameter (bypasses `input_linear`, ensuring each neuron receives the same 1.0 pulse)
3. Record T=100 steps of slow state x_t and membrane potential u⁰
4. Differences arise purely from each neuron's intrinsic spectral parameters (α, ρ, ω)

## Datasets
- PS-MNIST (2-layer, analyze Layer 0 only)
- GSC (1-layer)
- ECG / QTDB (1-layer)

## Outputs
- `impulse_response_gallery.png` — impulse response line plots for 8 sampled neurons per task (x_real main line + x_osc1/x_osc2 dashed lines), sorted by α
- `frequency_response.png` — |FFT(x_real)| frequency-domain plot, color-coded by α, showing low-pass/band-pass diversity
- `lif_comparison.png` — per-task selection of 3 neurons (slow/medium/fast), SPRiF vs equivalent LIF exp(-t/τ) comparison + oscillation amplitude

## Interpretation Guide
- **α large (~0.9+)** → fast decay, short memory, analogous to LIF small τ
- **α small (~0.1-)** → slow decay, long memory
- **ω large** → high oscillation frequency, sensitive to periodic inputs
- **|FFT| non-monotonic decay** → band-pass filtering behavior, not present in LIF
- **Large SPRiF vs LIF difference** → spectral parameterization produces temporal structure unattainable by LIF
