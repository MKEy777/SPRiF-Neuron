# Spike-Intervention Delayed Match-to-Sample (SI-DMS)

This directory implements a **novel mechanism verification experiment**, not a reproduction of HetSyn or BRF paper experiments. The task applies controlled, label-independent spike interventions between the first and second cue, testing whether the model can maintain match/non-match memory despite transient spike perturbations.

## Experimental Questions

Mechanism ablation tests only two claims:

1. `sprif_full` vs `sprif_merged`: whether slow/fast state separation protects memory.
2. `sprif_full` vs `sprif_lambda0`: whether learned projective reset `[1, λ]` outperforms scalar reset `[1, 0]`.

`lif`, `asrnn`, `brf` are external baselines, not part of SPRiF mechanism ablation. Shuffled λ, no-reset, or ω=0 are excluded to keep the ablation table focused on the two core claims.

## Intervention Definition and Evidence Boundaries

Within each delay, uniformly sample `K` time steps; at each step, randomly select a fixed proportion of hidden units. Mask generation does not read labels. For selected units, apply a minimal positive increment before threshold evaluation to reach `threshold + margin`. Recorded reset events thus come from spikes that genuinely occur in this forward pass; trajectories with `spike.sum() == 0` are not plotted as pre/post reset evidence.

The main experiment uses one recurrent 64-unit spiking hidden layer with a non-spiking leaky-integrator readout, without directly reading the SPRiF slow state. The default protocol trains for 3,000 steps with batch size 256, Adam at learning rate 0.003, delays in `{200, 400, 800, 1600}` ms, intervention counts in `{0, 1, 2, 4, 8}`, and a 15% intervention fraction. Evaluation additionally includes delay 2,500 ms, intervention counts `{16, 32, 40}`, and fractions `{5, 10, 15, 20, 30, 50}%`.

## Quick Start

```powershell
python -m pytest -q
python run_all.py --config config/default.yaml --output results
python aggregate.py --input results/all_metrics.json --curve results/fraction_response.csv
python plot_results.py --input results/all_metrics.json --fraction 0.15
```

Single model run:

```powershell
python train.py --model sprif_full --config config/default.yaml --seed 1
python evaluate.py --checkpoint results/sprif_full/seed_1/checkpoint.pt --config config/default.yaml --seed 1
```

End-to-end smoke test (not for publication results):

```powershell
python run_all.py --config config/smoke.yaml --output smoke_results --eval-batches 1
```

## Outputs

- `checkpoint.pt`: model parameters, model name, and config snapshot.
- `train_history.json`: per-step loss, accuracy, natural firing rate, and intervention conditions.
- `eval_metrics.json/csv`: accuracy, natural firing rate, and forced-hit rate per `delay × K` condition.
- `all_metrics.json`: aggregated results across all models and random seeds.
- `summary.csv`: clean accuracy, max intervention accuracy, and stress drop, with explicit distinction between mechanism ablation and external baselines.
- `figures/*_robustness.png`: delay × K robustness curves per model.

For publication-quality results, use at least the default 3 seeds and report mean and confidence intervals for each delay × K grid cell. `forced_hit_rate` should be 1; if not, first debug the intervention implementation before interpreting reset mechanism results.
