# Spike-Intervention Delayed Match-to-Sample (SI-DMS)

This directory implements a **novel mechanism verification experiment**, not a reproduction of HetSyn or BRF paper experiments. The task applies controlled, label-independent spike interventions between the first and second cue, testing whether the model can maintain match/non-match memory despite transient spike perturbations.

## Experimental Questions

The TNNLS primary model is `sprif_scalar`: a three-coordinate spectral state,
a scalar auxiliary filter, and a scalar membrane with subtractive reset. It has
eleven local learned coefficients, including the absorbed second readout row.
The membrane uses the previous filter state; spikes do not directly reset the
spectral or filter state.

`sprif_full`, `sprif_merged`, and `sprif_lambda0` are preserved as historical
models. They cannot serve as the planned TNNLS capacity controls. The new
`memory_reset5` and `real_modes3` cells are registered; the dedicated
`run_tnnls.py` entry point supports two training conditions, five seeds and
paired clean/stress inputs. These changes require runtime validation in the
training environment before being used for formal results. The legacy
`run_all.py` entry point retains its previous defaults.

## Intervention Definition and Evidence Boundaries

Within each delay, uniformly sample `K` time steps; at each step, randomly select a fixed proportion of hidden units. Mask generation does not read labels. For selected units, overwrite the membrane candidate with `threshold + 0.05`, then emit a spike and apply the ordinary reset. This operation may lower a candidate that was already above that value. Recorded reset events come from spikes in this forward pass.

The main experiment uses one recurrent 64-unit spiking hidden layer with a non-spiking leaky-integrator readout, without directly reading the SPRiF slow state. The default protocol trains for 2,000 steps with batch size 256, Adam at learning rate 0.003, delays in `{200, 400, 800, 1600}` ms, intervention counts in `{0, 1, 2, 4, 8}`, and a 15% intervention fraction. Evaluation additionally includes delay 2,500 ms, intervention counts `{16, 32, 40}`, and fractions `{5, 10, 15, 20, 30, 50}%`.

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


## TNNLS controls and training entry point

`real_modes3` implements three independently driven real decays with scalar membrane reset, five states and eleven local learned coefficients. It retains the primary SI-DMS alpha initialization (raw alpha=3) for the first mode and samples the other two independently with time constants uniform in [2,40] steps. This explicit initialization choice must be recorded with the run; it differs from the primary model's fixed rho initialization. `memory_reset5` applies the fixed normalized-input-direction reset to the spectral state after both readouts, preserving h and scalar membrane reset.

In a prepared training environment, first run `python -m pytest -q`. For a small development-only check, use a separate output path:

```powershell
python run_tnnls.py --models real_modes3 --training Clean Intervention --seeds 0 --steps 2 --eval-batches 1 --smoke --output results_tnnls_smoke
```

The smoke flag labels every checkpoint and evaluation row. The default configuration still has the full evaluation grid, so this is a functionality check, not an accuracy result. Formal run, in a fresh output directory:

```powershell
python run_tnnls.py --models real_modes3 --training Clean Intervention --output results_tnnls_real_modes3
```

Formal mode fixes seeds 0,1,2,42,1111, 2000 optimizer steps and 20 evaluation batches. Results are tagged `tnnls-independent-streams-v2-steps2000` and must not be mixed with the earlier 3000-step runs. Clean training sets K=0 throughout; Intervention training uses K=0/1/2/4/8. Evaluation uses shared input and separate intervention-mask random streams across K/f, model, and training condition for each seed, delay and batch. Each run saves checkpoint, history and per-condition JSON; root evaluations.json uses plotting model names and carries training, model_id and smoke fields. One-model outputs must be combined with genuine outputs for the other five models before using the composite plotter; no numbers are filled or inferred here. This implementation has only been statically checked and must be validated on the intended PyTorch version before formal training.


## 同一张默认显卡运行两个进程（TNNLS v2）

使用两个终端，各自运行一条命令；保持进程在不同输出目录，不需要指定显卡编号或安装 GPU 锁。`device: auto` 使用当前默认可见 CUDA 显卡，CUDA 不可用时回退 CPU。两进程共享该显卡显存，如果显存不足会发生 CUDA OOM；同时运行也不保证速度翻倍。

进程 1：Clean 训练，六个模型：

```bash
python run_tnnls.py --models sprif_scalar memory_reset5 real_modes3 asrnn brf lif --training Clean --output results_tnnls_clean
```

进程 2：Intervention 训练，六个模型：

```bash
python run_tnnls.py --models sprif_scalar memory_reset5 real_modes3 asrnn brf lif --training Intervention --output results_tnnls_intervention
```

两条命令各跑 6 模型 × 5 种子 = 30 次；只跑 REAL_MODES3 时删去各自 `--models` 后的其他五个模型。每个输出目录必须为空或不存在；出现中断时保留已完成数据，当前入口不会自动续跑，也不会覆盖已有目录。训练结束后：

```bash
python merge_tnnls.py --inputs results_tnnls_clean/evaluations.json results_tnnls_intervention/evaluations.json --output results_tnnls_combined/evaluations.json
python aggregate.py --input results_tnnls_combined/evaluations.json --output results_tnnls_combined/summary.csv --curve results_tnnls_combined/fraction_curve.csv
```

合并器拒绝重复记录、smoke 数据及缺失或不同协议版本。两个输出都完成后才能运行；已有服务器文件如果没有协议版本，应先核查代码版本和随机流，不能自行补版本标签。新版汇总按训练条件和模型分组，每种子先对五延迟等权平均，再计算五种子的均值与样本标准差；单独报告 2500 ms，清洁/干预准确率单位为百分数、差值单位为百分点。fraction_curve.csv 的准确率为0–1。汇总会拒绝缺失种子或关键条件，不从其他模型补数据。

正式模式在运行前比对完整任务、网络及训练协议，拒绝使用 smoke 或不一致的配置；仅历史 YAML 的种子列表由五种子 CLI 固定值覆盖。训练延迟、训练干预次数和训练掩码使用独立随机流，配对测试共享输入。结果记录 `protocol_version=tnnls-independent-streams-v2`。此协议的随机流不同于之前服务器训练版本，可能导致结果变化；不要直接当作旧运行的逐项续跑。旧 `run_all.py`、`evaluate.py`、`plot_results.py` 仍是历史流程，不应作为上述新版混合训练条件的正式入口。
