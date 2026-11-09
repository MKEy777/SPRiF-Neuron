# 时间核组合图：原始数据与运行方法

推荐先按 `04-训练代码/README.md` 完成最少四个模型训练，再从代码根目录直接执行：

```bash
python "03-时间核/plot_kernels.py" --seed 0
```

脚本自动读取 PS-MNIST、GSC、QTDB 的 SPRiF checkpoint，提取参数并出图。对于每个 ASRNN 对比，优先读取已完成的同任务、同种子 ASRNN checkpoint；不存在时复用本目录 `legacy_asrnn_tau.npz` 中对应的历史时间常数，不自动补训。历史数据的原始 checkpoint、种子和 ECG/QTDB 对应关系在正式使用前仍需核实。下文保留手动导出和自定义数据清单的用法。

原版布局需要三个 SPRiF 数据组，以及其中两个组各自对应的 ASRNN 基线。示例名称是 `pSMNIST`、`GSC`、`QTDB`，可以改成真正有数据的任务或种子，但名称必须与数据来源一致，不能把同一个结果冒充三个任务。

## 1. 导出新版 SPRiF 参数

在 `新版绘图代码` 目录执行，分别对三个真实 checkpoint 运行一次。例如：

```bash
python export_parameters.py --checkpoint "/path/to/pSMNIST-checkpoint.pt" --threshold 1 --output "03-时间核/原始数据/pSMNIST_parameters.npz"
python export_parameters.py --checkpoint "/path/to/GSC-checkpoint.pt" --threshold 0.8 --output "03-时间核/原始数据/GSC_parameters.npz"
python export_parameters.py --checkpoint "/path/to/QTDB-checkpoint.pt" --threshold 0.6 --output "03-时间核/原始数据/QTDB_parameters.npz"
```

`--threshold` 必须与该模型训练时一致；SI-DMS 的检查点内有阈值配置时可省略。每份文件会附同名 JSON，记录模型、层前缀和 checkpoint SHA-256。一个 checkpoint 存在多个可选层时，加上明确的 `--prefix`，例如 `--prefix layers.0.`。旧版 Full/λ 权重和 REAL_MODES3 不能直接作为 SPRiF 使用。

## 2. 导出 ASRNN 基线

对于当前 SI-DMS ASRNN 的正式 checkpoint：

```bash
python "03-时间核/export_asrnn_parameters.py" --checkpoint "/path/to/asrnn/checkpoint.pt" --output "03-时间核/原始数据/对应组_asrnn.npz"
```

它提取训练出的膜电位时间常数，构造**无脉冲时的亚阈值响应**；不是完整 ASRNN 的自适应发放或循环响应。必须使用同一任务且时间步长匹配的 SPRiF 与 ASRNN。两个基线分别执行一次。

旧版保存的 `raw_impulse_responses.npz` 中也有 GSC 和 ECG 的 ASRNN 时间常数。若已核对其来源，可只提取基线（不会使用旧 SPRiF 响应）：

```bash
python "03-时间核/export_asrnn_parameters.py" --legacy-raw "/path/to/raw_impulse_responses.npz" --task GSC --output "03-时间核/原始数据/GSC_asrnn.npz"
python "03-时间核/export_asrnn_parameters.py" --legacy-raw "/path/to/raw_impulse_responses.npz" --task ECG --output "03-时间核/原始数据/QTDB_asrnn.npz"
```

历史文件没有完整的新版训练来源记录，JSON 会标记为来源待核实；用于论文前须确认历史 ECG 与当前 QTDB 的任务定义及基线 checkpoint。

## 3. 填写数据清单并绘图

复制本目录的 `manifest.example.json` 为 `manifest.json`，将每个文件名和标签改成真实对应关系。相对路径相对于 `manifest.json` 所在目录。必须填写三个不同标签的 SPRiF 组和两个不同组的 ASRNN 基线。

```bash
python "03-时间核/plot_kernels.py" --manifest "03-时间核/原始数据/manifest.json" --steps 200
```

上排显示按 α 的 10%、40%、70%、90% 分位选择的神经元：实线为实模态对膜驱动核的贡献，点线为旋转模态贡献；这两种贡献都包含一步延迟的辅助滤波路径。下排曲线是每个神经元完整驱动核的频谱先各自归一化，再计算中位数和 10%–90% 分位带；零响应不可归一化，会排除并在来源记录中报告。

右侧将 SPRiF 中 α 最接近中位数的神经元，与 ASRNN 中 α 最近的神经元比较，展示零状态、无发放、单位电流下的膜电位响应，各自按峰值绝对值归一化。这是选择规则明确的示例，不是完整网络性能比较。

输出 `temporal_kernels_composite.pdf/.svg/.png`、对应 `.data.npz`、来源 JSON 和对齐检查 JSON。结果来源标记为 smoke 的输入默认被拒绝，`--allow-smoke` 只用于测试。正式输入出图后仍需检查版面及原始数据来源。
