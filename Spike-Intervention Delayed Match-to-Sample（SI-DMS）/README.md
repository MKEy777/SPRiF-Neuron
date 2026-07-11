# Spike-Intervention Delayed Match-to-Sample (SI-DMS)

该目录实现一个**新设计的机制验证实验**，不是对 HetSyn 或 BRF 论文实验的复现。任务在第一线索和第二线索之间施加与标签无关的受控脉冲干预，考察模型能否在瞬态放电扰动后保持 match/non-match 记忆。

## 实验问题

机制消融只检验两个主张：

1. `sprif_full` 对比 `sprif_merged`：slow/fast 状态分离是否保护记忆。
2. `sprif_full` 对比 `sprif_lambda0`：learned projective reset `[1, λ]` 是否优于 scalar reset `[1, 0]`。

`lif`、`asrnn`、`brf` 是外部基线，不属于 SPRiF 机制消融。未包含 shuffled λ、no-reset 或 ω=0，避免让消融表偏离两项核心主张。

## 干预定义与证据边界

每个 delay 内均匀抽取 `K` 个时刻，每次随机抽取固定比例隐藏单元。掩码生成不读取标签。对选中单元，在阈值判断前施加最小正增量，使其到达 `threshold + margin`。因此记录的 reset 事件来自本次前向计算中真实发生的 spike；代码不会把 `spike.sum()==0` 的轨迹画成 pre/post reset 证据。

主实验使用 feed-forward hidden layer 与非放电 leaky-integrator readout，不直接读取 SPRiF slow state。这样分类性能只能通过之后的正常放电传到 readout，而不能绕过 reset 机制。

## 快速开始

```powershell
python -m pytest -q
python run_all.py --config config/default.yaml --output results
python aggregate.py --input results/all_metrics.json
python plot_results.py --input results/all_metrics.json
```

单模型运行：

```powershell
python train.py --model sprif_full --config config/default.yaml --seed 1
python evaluate.py --checkpoint results/sprif_full/seed_1/checkpoint.pt --config config/default.yaml --seed 1
```

端到端链路检查（不能作为论文结果）：

```powershell
python run_all.py --config config/smoke.yaml --output smoke_results --eval-batches 1
```

## 输出

- `checkpoint.pt`：模型参数、模型名和配置快照。
- `train_history.json`：逐步 loss、accuracy、自然放电率和采样条件。
- `eval_metrics.json/csv`：每个 `delay × K` 条件的准确率、自然放电率和强制放电命中率。
- `all_metrics.json`：所有模型与随机种子的合并结果。
- `summary.csv`：clean accuracy、最大干预准确率与 stress drop，并显式区分机制消融和外部基线。
- `figures/*_robustness.png`：各模型 delay×K 鲁棒性曲线。

正式结果建议至少使用默认的 3 个种子，并报告每个 delay×K 网格的均值与置信区间。`forced_hit_rate` 应为 1；若不是，应先排查干预实现，不能继续解释 reset 机制。
