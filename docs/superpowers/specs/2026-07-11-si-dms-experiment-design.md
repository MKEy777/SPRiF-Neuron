# SI-DMS Experiment Design

## Goal

Implement a self-contained Spike-Intervention Delayed Match-to-Sample experiment that tests exactly two SPRiF claims: functional slow/fast separation protects memory from spike-triggered reset, and learned projective reset improves over scalar reset.

## Protocol

Each sample contains two 100 ms Poisson cues separated by a configurable delay. Ten channels encode a left cue, ten encode a right cue, and ten carry continuous background noise. The target is match for left-left/right-right and mismatch otherwise. At controlled times during the delay, a label-independent intervention raises selected hidden neurons just above their current threshold so the same mask produces actual spike/reset events in every model.

The primary architecture is input -> one feedforward spiking hidden layer -> two non-spiking leaky readout neurons. Recurrent hidden connectivity is optional and disabled in the main protocol. The classifier never reads SPRiF's slow state directly.

## Models

- `sprif_full`: 3D spectral slow state, 2D fast state, learned reset direction `[1, lambda]`.
- `sprif_merged`: one 3D spectral state, first coordinate is membrane and receives scalar reset; no fast state.
- `sprif_lambda0`: same as full with fixed reset direction `[1, 0]`.
- `lif`: trainable membrane time constant and scalar soft reset, adapted from the BRF reproduction repository.
- `asrnn`: adaptive threshold/membrane dynamics and multi-Gaussian surrogate adapted from the local ASRNN implementation.
- `brf`: balanced resonate-and-fire dynamics with refractory state and stability boundary adapted from the referenced MIT-licensed reproduction repository.

The first three are the only mechanism ablations. LIF, ASRNN, and BRF are external baselines.

## Training and Evaluation

Training minimizes final-step cross entropy plus a firing-rate target penalty computed only on non-intervention spikes. Training samples delays and intervention counts from configured sets. Evaluation uses identical deterministic batches for all models and saves accuracy, loss, natural/forced firing rates, forced-hit rate, parameter counts, reset residuals, and a complete delay-by-intervention grid.

Primary outputs are per-run JSON/CSV histories, checkpoints, aggregate mean/std tables, stress-AUC summaries, accuracy curves, heatmaps, and an auditable trace around a controlled reset.

## Integrity Gates

- Intervention masks are independent of cue identity and target.
- Every selected intervention must produce a spike.
- SPRiF full reset must satisfy `u_pre - u_post = spike * theta * [1, lambda]`.
- SPRiF slow state is updated before intervention and is not altered by it.
- `lambda0` differs from full only by fixing lambda to zero.
- `merged` follows the existing project ablation definition.
- All models share task batches, hidden width, readout type, optimizer budget, and intervention masks.
- Smoke tests use tiny settings only; no unrun result is represented as evidence.
