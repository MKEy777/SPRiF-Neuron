# 两张组合图：最少训练与直接绘图

默认只训练一个随机种子的四个模型：

| 用途 | 数据任务 | 模型 | 数量 |
|---|---|---|---|
| 机制图 | PS-MNIST | `sprif_scalar`、`memory_reset5` | 2 |
| 时间核图 | GSC、QTDB | 各一个 `sprif_scalar` | 2 |

时间核图第三组直接复用机制图的 PS-MNIST SPRiF，不重复训练。ASRNN 优先使用已存在的 checkpoint，否则使用随包提供的历史参数；默认不训练 ASRNN，也不训练 SI-DMS、REAL_MODES3、LIF、BRF 或 SHD。四个模型可以用 `--job all` 顺序运行，或用下面两个不重叠的任务并行运行。

**历史基线的来源需要核实：**历史数据提供了 GSC 和 ECG 的 ASRNN 时间常数，但尚未确认全部原始 checkpoint 及 ECG/QTDB 的对应关系。出论文正式图之前必须核实；如果无法核实且确实需要 ASRNN 对比，才需要补训对应 ASRNN 模型，不能把旧数据当成已验证结果。

## 服务器上两条训练命令

把整个代码包解压并进入 `SPRiF-figures` 目录。分别在两个终端运行：

```bash
python "04-训练代码/train_figures.py" --job mechanism --seed 0
```

```bash
python "04-训练代码/train_figures.py" --job kernel --seed 0 \
  --gsc-data-root "/你的路径/speech_commands_v0.02" \
  --qtdb-train-mat "/你的路径/QTDB_train.mat" \
  --qtdb-test-mat "/你的路径/QTDB_test.mat"
```

将第二条命令中的路径改成服务器上真实数据的位置。两个进程使用 PyTorch 默认设备，不在命令中选择 GPU，可以同时使用同一张可见显卡；是否能同时放入显存取决于服务器配置。

PS-MNIST 默认数据目录在 `benchmarks/Task_pSMNIST/data`，数据缺失时原训练脚本会尝试下载 MNIST。已有数据可给第一条命令增加 `--psmnist-data-root "/你的路径/MNIST目录"`。GSC 根目录需包含 `testing_list.txt`、`validation_list.txt` 和 `silence_validation_list.txt`，以及对应的音频；QTDB 使用已有预处理 MAT 文件。时间核任务会在启动训练前检查 GSC 和 QTDB 路径。

PS-MNIST、GSC、QTDB 分别沿用项目已有的 150、150、250 个 epoch；之前的 **2000 步只针对 SI-DMS**，不适用于这三个基准任务。依赖训练环境已有的 PyTorch、torchvision、numpy 等，脚本不会安装依赖或在本机自动训练。

## 训练后直接画图

两个任务结束后，在 `SPRiF-figures` 目录运行：

```bash
python "01-机制分析/plot_mechanism.py" --seed 0
python "03-时间核/plot_kernels.py" --seed 0
```

绘图会直接读取本包训练出的 checkpoint，自动提取真实测试样本、轨迹和参数到各自的 `原始数据/`，再输出组合图；无需手动运行中间导出脚本。如果 PS-MNIST 的第 0 个测试样本没有两个模型同时发放的事件，机制图会报错，可改用 `--sample-index 1` 等实际样本，选择规则会记录。

正式 checkpoint 位于 `04-训练代码/checkpoints/任务/模型/seed_0/checkpoint.pt`。训练过程中可能出现 `checkpoint.training.pt`，它不是完成的结果；训练正常结束后才会生成正式文件。已存在且元信息匹配的正式 checkpoint 会复用，避免重复训练；中断的临时文件会保留，不自动覆盖或续训。

若 ASRNN 历史来源无法确认，确需重新训练基线时，分别使用 `--job asrnn_gsc` 或 `--job asrnn_qtdb`，只训练缺少的基线，并提供相应任务的数据路径。绘图默认优先使用已完成的对应 ASRNN checkpoint，缺少的任务才回退到历史参数并显示来源待核实提示。`--asrnn-source trained` 则要求两个任务的 ASRNN checkpoint 都存在。默认流程不会执行 ASRNN 训练。

## 证据范围

每个模型只训练一个种子用于示例图，不构成五种子准确率统计。PS-MNIST 的 `MEMORY_RESET5` 使用固定的输入方向光谱复位，不增加局部可学习参数；代码会记录任务、模型、种子、训练配置和选中轮次。机制轨迹由这两个 checkpoint 以及测试样本重新回放，不能冒称训练期间存档的轨迹。所有正式图仍需根据真实数据进行端到端检查和版面核验。
