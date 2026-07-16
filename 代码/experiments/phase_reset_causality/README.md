# SPRiF 相位轨迹因果实验

该目录实现独立的相位轨迹回归与 reset 路由干预实验，不读取或覆盖旧
`trajectory_visualization`、SI-DMS 结果，也不修改论文 TeX。

## 核心问题

模型在 100 ms cue 中接收初始相位 `phi` 和频率 `omega` 的 Poisson 编码，
随后在没有相位输入的 delay 中持续回归：

```text
[cos(phi + omega * t), sin(phi + omega * t)]
```

所有模型只从 filtered spike train 读出二维轨迹。SPRiF 在同一 checkpoint、
同一样本、同一干预时刻和同一强制脉冲 mask 下比较：

- `clean`
- `forced_no_reset`
- `fast_reset`
- `slow_reset`
- `both_reset`

强制操作只做非负膜电位抬升：

```text
u_forced = max(u, threshold + margin)
```

因此不会复现旧 SI-DMS `_force` 将高膜电位降低到固定目标值的问题。慢状态
reset 的方向由 `G^T d_u` 给出，并缩放到与快速 reset 相同的 L2 范数。

## 运行

安装依赖：

```powershell
pip install -r requirements.txt
```

快速完整检查：

```powershell
python run_all.py --config config/smoke.yaml
```

正式实验：

```powershell
python run_all.py --config config/default.yaml
```

如需在两张 GPU 上并行运行正式实验，两个分片必须使用同一个输出目录；它们只
训练和评估，不会并发写入汇总文件。两边结束后再聚合和画图：

```powershell
python run_sprif_lif.py --config config/default.yaml --device cuda:0
python run_asrnn_brf.py --config config/default.yaml --device cuda:1
python aggregate.py --config config/default.yaml
python plot_results.py --config config/default.yaml
```

单独训练或评价：

```powershell
python train.py --config config/default.yaml --model sprif --seed 1
python evaluate.py --config config/default.yaml --model sprif --seed 1
python aggregate.py --config config/default.yaml
python plot_results.py --config config/default.yaml
```

可使用 `--device cpu` 或 `--device cuda`；`run_all.py --skip-train` 会复用现有
checkpoint，`--no-sensitivity` 只运行主事件设置。

## 评价与统计

主实验分别在配置中的每个 `event_step` 进行单事件配对评价。主要指标包括：

- delay MSE 与输出半径；
- circular phase error；
- 事件后 1--5 步 phase jump；
- 相对 `forced_no_reset` 的 excess-error AUC；
- 10% 峰值阈值、连续保持窗口定义的 recovery time；
- forced-hit、新阈值越过、自然脉冲重叠率和自然 firing rate；
- SPRiF fast/slow 状态相对 no-reset 分支的状态差。

Sensitivity 模式额外运行 K、reset 强度、多事件累计，并将训练频率（ID）与
未见频率（OOD）逐周期分开评价。主统计在每个 seed 内保留样本—事件级
`slow_reset - fast_reset` 效应，再对 seed 和样本两层重采样，输出分层 bootstrap
置信区间和精确符号置换检验。外部模型只运行 `clean`、`forced_no_reset` 和
`native_reset`，不将其内部状态硬映射为 SPRiF 的 fast/slow 状态。

若任一批次中不能找到足够阈下单元的样本比例超过 1%，评价会直接失败，而不
会静默删除这些样本。`clean_gate_passed` 仅说明 checkpoint 是否满足预注册的
clean MSE、输出半径和非零 firing 条件；未通过的原始结果仍保存，但 reset 行
不会进入汇总表、因果统计或论文级代表轨迹，不能解释为 reset 鲁棒性。原始
seed 目录仍保留诊断轨迹；根目录的 `representative_traces.npz` 只从通过门槛的
SPRiF checkpoint 中选择。

## 输出

默认输出到：

```text
experiment-design-20260606/results/phase_reset_causality/
```

主要文件：

- `raw/<model>/seed_<n>/checkpoint.pt`
- `raw/<model>/seed_<n>/train_history.json`
- `raw/<model>/seed_<n>/eval_metrics.json`
- `summary.csv`
- `paired_effects.csv`
- `paired_statistics.json`
- `all_metrics.json`
- `representative_traces.npz`
- `main_causal_figure.{png,pdf}`
- `baseline_robustness.{png,pdf}`
- `appendix_sensitivity.{png,pdf}`

代表轨迹同时保存 phase failure、输出半径塌缩和重复放电三类样本索引、目标、
分支输出与脉冲；主图只读取通过 clean gate 后提升到结果根目录的版本。

旧轨迹 NPZ 中没有真实 spike，因此不会被本实验加载，也不能作为事件级 reset
证据。是否将新结果写入论文，必须在正式运行、数值一致性检查和图表审核后另行
决定。

## 测试

```powershell
python -m pytest tests -q
```

测试覆盖数据目标、Delay 输入隔离、非下降强制脉冲、reset 范数与路由、统一
spike readout、配对事件、相位指标、多事件压力、结果键去重以及 CLI smoke
全流程。
