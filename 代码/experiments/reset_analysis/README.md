# Reset Direction Analysis (λ)

## Objective
Analyze the learned projective reset directions λ_j of SPRiF — verify that λ is genuinely learned (non-trivial distribution) and exhibits meaningful correlation with firing rates and spectral parameters.

## Method
1. Load trained SPRiF models (pSMNIST / GSC / ECG)
2. Extract per-neuron `lambda_reset` values via `get_spectral_parameters()["lambda_reset"]`
3. Run forward pass on test set, record per-neuron firing rate
4. Analyze λ distribution + correlation with firing rate / α / ω

## Datasets
- **PS-MNIST** (2-layer [64, 256]) — PermutedMNIST test set
- **GSC** (1-layer [300]) — SpeechCommands test set
- **ECG / QTDB** (1-layer [36]) — QTDB test set

## Outputs
- `lambda_distribution.png` — λ histogram per task, overlaid by layer, with λ=0 reference line
- `lambda_vs_firing_rate.png` — λ vs firing rate scatter, colored by layer (with correlation coefficient r)
- `lambda_vs_alpha.png` — λ vs α scatter (with r)
- `lambda_vs_omega.png` — λ vs ω scatter (with r)
- `lambda_stats.csv` — full data table: task, layer, neuron, α, ρ, ω, η₀, η₁, λ, firing_rate

## Interpretation Guide
- **λ concentrated near 0** → projective reset degenerates to scalar reset; C3 ablation should cause no significant loss
- **λ widely distributed (positive and negative)** → projective reset actively learned; C3 ablation should degrade performance
- **λ positively correlated with firing rate** → high-firing neurons learn larger reset magnitude in the u₁ dimension
- **λ correlated with α** → reset strategy coupled with temporal filtering characteristics
- **λ correlated with ω** → reset strategy coupled with frequency selectivity
- **λ < 0 neurons** → reset direction negative in u₁ dimension, possibly linked to specific firing patterns
