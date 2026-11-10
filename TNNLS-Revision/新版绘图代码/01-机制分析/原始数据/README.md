# 机制组合图：原始数据与运行方法

推荐直接使用 `04-训练代码/train_figures.py --job mechanism --seed 0` 产生的两个 PS-MNIST checkpoint，再从代码根目录执行：

```bash
python "01-机制分析/plot_mechanism.py" --seed 0
```

绘图脚本会自动读取 checkpoint、取真实 MNIST 测试样本并回放；若数据集放在其他位置，可附 `--psmnist-data-root`。指定已有预处理样本可用 `--sample /path/to/sample.npy`。完整训练流程见 `04-训练代码/README.md`；下文保留手动导出与 SI-DMS 轨迹的用法。

## 输入数据

将同一训练条件、同一种子、同一实验配置和协议版本的两个**正式** SI-DMS 检查点准备好：

- `sprif_scalar/.../checkpoint.pt`
- `memory_reset5/.../checkpoint.pt`

运行 `export_trajectories.py` 会用两个网络处理**相同的一条生成试次和同一干预掩码**，逐步记录两个网络各自的 `x`、`h`、膜电位的复位前后值和脉冲。每个模型使用自己训练出的权重及自己的循环脉冲反馈。输出 `trajectories.npz` 和同名 JSON 来源记录。这里不训练、不重采样模型权重；一条试次不是跨种子统计。

自动选择的示例事件是有完整时间窗的**第一个强制脉冲**，不是按效果挑选。两模型中的相同神经元编号仅为展示方便，不代表功能匹配。

## 服务器命令

先将整个 `新版绘图代码` 目录上传服务器，从该目录执行以下命令；将四处 `/path/to/...` 换成你的真实路径。以下路径里的引号是必要的。

```bash
python "01-机制分析/export_trajectories.py" \
  --code-root "/path/to/SI-DMS-训练代码" \
  --sprif-checkpoint "/path/to/Intervention/sprif_scalar/seed_0/checkpoint.pt" \
  --reset-checkpoint "/path/to/Intervention/memory_reset5/seed_0/checkpoint.pt" \
  --output "01-机制分析/原始数据/trajectories.npz"

python "01-机制分析/plot_mechanism.py"
```

`--code-root` 是**含 `sidms/` 文件夹**的训练代码目录，必须与生成 checkpoint 的实现一致。两检查点须来自相同的训练条件、种子和训练协议。不同路径请按服务器实际位置修改。默认延迟 1600 ms、干预 8 次、比例 0.15；可用 `--delay-ms`、`--count`、`--fraction`、`--input-seed` 和 `--mask-seed` 指定，参数会记录在 JSON 中。

要指定某次强制脉冲，例如事件位于第 80 步、神经元 3，可加 `--event-step 80 --neuron 3`；该位置必须实际受到干预且两边具有完整窗口。默认 `--radius 15`，即展示前后各 15 步。

## 五个面板

- a：一个 SPRiF 神经元的光谱状态和复位前/后膜电位；虚线为选定强制发放时刻。
- b：两个独立模型对应编号的旋转平面轨迹；圆点为发放前、叉号为发放后。
- c：选定事件的光谱状态复位方向。
- d：这一条试次中所有被强制发放的神经元事件的光谱复位范数。
- e：这些事件的五个状态分量平均复位变化，按阈值归一化。

d/e 的事件不是独立训练重复，不能当作跨种子标准差或显著性检验。曲线反映模型的局部复位和各自循环网络反馈；该图本身不能证明表 2 中 PS-MNIST / SHD 的性能差异。要直接展示表 2 的轨迹，还需这两个任务的实际新版网络、测试输入和对应检查点。

## 用 PS-MNIST / SHD 的真实测试样本

`export_benchmark_trajectories.py` 可从两个当前模型 checkpoint 重建隐藏层轨迹。它使用 `--code-root/core_algorithm/sprif_layer.py` 的新版层实现，并在 MEMORY_RESET5 一侧**显式应用论文中的光谱复位规则**。必须确认控制组 checkpoint 确实由相同规则和架构训练得到；纯 state_dict 本身不能证明这一点。两个 checkpoint 必须是独立训练结果，并指定正确的训练种子和阈值。

准备一个实际的测试样本为 `sample.npy`，形状 `[时间步, 输入特征]`，或含 `input` 数组的 `.npz`；预处理必须与原任务一致，例如 PS-MNIST 使用训练时相同的固定像素排列、SHD 使用训练时的时间分箱。脚本不下载数据、不执行训练。

```bash
python "01-机制分析/export_benchmark_trajectories.py" \
  --code-root "/path/to/Task_pSMNIST" \
  --sprif-checkpoint "/path/to/sprif-seed-0.pt" \
  --reset-checkpoint "/path/to/memory-reset5-seed-0.pt" \
  --input "/path/to/real-preprocessed-sample.npy" \
  --task pSMNIST --sample-id "test-sample-id" --seed 0 --threshold 1 \
  --output "01-机制分析/原始数据/benchmark_trajectories.npz"

python "01-机制分析/plot_mechanism.py" --input "01-机制分析/原始数据/benchmark_trajectories.npz"
```

SHD 使用其任务目录并改为 `--task SHD`。默认记录第 0 隐藏层；可用 `--layer` 指定其他隐藏层。两个网络分别使用自己的权重、前层脉冲和循环反馈。图中选取两个模型在**同一时间、同一神经元编号同时自然发放**的事件；这不是功能匹配，且只包含同时发放的子集。如果该样本没有此类事件，脚本报错，应提供另一条真实样本。自动示例为有完整时间窗的首个此类事件。

这种由 checkpoint 和已声明复位规则重建出的单样本轨迹用于解释神经元机制；它不是已经存档的训练期实测轨迹，不能替代表 2 的多种子准确率统计。

输出为 `mechanism_composite.pdf/.svg/.png`、对齐检查 JSON 和来源 JSON。默认拒绝 `smoke` 测试 checkpoint；`--allow-smoke` 只供程序测试，不能作为论文数据。
