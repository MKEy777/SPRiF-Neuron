# Spike-Intervention Delayed Match-to-Sample (SI-DMS)

## Purpose

SI-DMS replaces the former trajectory-visualization experiment. It is a controlled mechanism experiment, not a reproduction of HetSyn or BRF. The task tests whether a neuron retains a sample cue across a delay when selected hidden neurons are forced to emit real spikes during that delay.

The experiment addresses exactly two claims:

1. **Slow/fast separation protects memory.** Compare `sprif_full` with `sprif_merged`.
2. **Learned projective reset is better than scalar reset.** Compare `sprif_full` with `sprif_lambda0`.

LIF, ASRNN, and BRF are external baselines. They are not SPRiF mechanism ablations.

## Task protocol

- Input channels: 10 left-cue channels, 10 right-cue channels, and 10 background-noise channels.
- Timeline: pre-period → sample cue → delay → test cue.
- Target: binary match/non-match classification.
- Training delays: 200, 400, 800, and 1600 ms.
- Evaluation delays: 200, 400, 800, 1600, and 2500 ms.
- Training intervention counts: `K ∈ {0,1,2,4}`.
- Evaluation intervention counts: `K ∈ {0,1,2,4,8}`.
- At each intervention time, randomly select 10% of hidden neurons.
- Intervention masks are generated without reading the class label.

For a selected neuron, intervention is applied after intrinsic state flow and before thresholding. It adds the minimum nonnegative increment required to reach `threshold + margin`. The selected neuron therefore emits a real spike in that forward pass, after which the model's actual reset rule executes.

## Fair architecture

All models use the same input encoding, hidden width, optimizer, training batches, and two-dimensional nonspiking leaky-integrator readout. The main setting uses one feed-forward spiking hidden layer. The readout receives hidden spikes only and has no direct access to SPRiF's slow state.

## Model matrix

| Role | Model | Controlled change | Claim tested |
| --- | --- | --- | --- |
| Full | `sprif_full` | 3-D slow + 2-D fast + learned `[1,λ]` reset | Reference |
| Mechanism ablation | `sprif_merged` | One 3-D state; scalar reset on membrane coordinate | Slow/fast separation |
| Mechanism ablation | `sprif_lambda0` | Full dynamics; reset fixed to `[1,0]` | Projective reset |
| External baseline | `lif` | Leaky integration and scalar reset | Context only |
| External baseline | `asrnn` | Adaptive threshold state | Context only |
| External baseline | `brf` | Balanced resonant dynamics | Context only |

Shuffled λ, no-reset, merged-plus-extra variants, and ω=0 are excluded because they do not directly answer the two declared claims.

## Metrics

- Primary: accuracy for every `delay × K` cell.
- Robustness: accuracy drop from `K=0`, stress-area score over K, and degradation slope.
- Integrity: forced-spike hit rate, natural firing rate excluding intervention sites, and exact reset-residual checks.
- Statistics: at least three seeds for final paper results; report mean and standard deviation or confidence interval.

## Claim–evidence matrix

| Claim | Reviewer question | Required comparison | Evidence | Status |
| --- | --- | --- | --- | --- |
| Slow/fast separation protects memory | Does isolating memory from discharge/reset improve robustness? | full vs merged on identical grids | Heatmaps, K curves, paired seed statistics | Running / TBD |
| Projective reset improves recovery | Is learned `[1,λ]` better than `[1,0]` after actual spikes? | full vs lambda0 on identical masks | Heatmaps, K curves, paired statistics, forced-hit audit | Running / TBD |

## Integrity gates

1. `forced_hit_rate == 1` for every `K>0` condition.
2. Full and lambda0 leave the slow state unchanged by reset.
3. Full reset residual equals `spike × threshold × [1,λ]`.
4. Lambda0 reset residual equals `spike × threshold × [1,0]`.
5. Merged reset affects only its membrane coordinate.
6. No pre/post-reset arrow is drawn without a real spike and nonzero reset residual.

## Paper presentation

- **Figure SI-DMS-A:** protocol, full/merged/lambda0 heatmaps, and paired robustness curves.
- **Figure SI-DMS-B:** forced-hit audit, natural-rate control, reset-residual verification, and failure conditions.

External baselines should be visually separated from the mechanism ablations.

## Execution source

Run `../Spike-Intervention Delayed Match-to-Sample（SI-DMS）/`. Copy outputs into `results/si_dms/` only after verification. Smoke-test values must never enter paper tables.

## No-fabrication status

No SI-DMS performance result is asserted here. All result cells remain `TBD` until produced by the formal multi-seed run.
