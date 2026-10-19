# Spike-Intervention Delayed Match-to-Sample (SI-DMS)

This directory implements a **novel mechanism verification experiment**, not a reproduction of HetSyn or BRF paper experiments. The task applies controlled, label-independent spike interventions between the first and second cue, testing whether the model can maintain match/non-match memory despite transient spike perturbations.

## Experimental Questions

The TNNLS primary model is `sprif_scalar`: a three-coordinate spectral state,
a scalar auxiliary filter, and a scalar membrane with subtractive reset. It has
eleven local learned coefficients, including the absorbed second readout row.
The membrane uses the previous filter state; spikes do not directly reset the
spectral or filter state.

`sprif_full`, `sprif_merged`, and `sprif_lambda0` are preserved as historical
models. They cannot serve as the planned TNNLS capacity controls. MEMORY_RESET5,
the two training conditions, and paired clean/stress evaluation still require
their planned implementation before formal results. `lif`, `asrnn`, and `brf`
remain available through `--models`; the default queue contains `sprif_scalar`.

## Intervention Definition and Evidence Boundaries

Within each delay, uniformly sample `K` time steps; at each step, randomly select a fixed proportion of hidden units. Mask generation does not read labels. For selected units, overwrite the membrane candidate with `threshold + 0.05`, then emit a spike and apply the ordinary reset. This operation may lower a candidate that was already above that value. Recorded reset events come from spikes in this forward pass.

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
python train.py --model sprif_scalar --config config/default.yaml --seed 1
python evaluate.py --checkpoint results/sprif_scalar/seed_1/checkpoint.pt --config config/default.yaml --seed 1
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

The supplied YAML still has seeds 1, 2, and 3; formal TNNLS runs use the five
seeds specified in the revision protocol. No training has been run for the new
model here. Summary files identify their reference model explicitly: new-model
files use `sprif_scalar`, while legacy-only files retain `sprif_full`.
`forced_hit_rate` should be 1 before interpreting intervention results.
