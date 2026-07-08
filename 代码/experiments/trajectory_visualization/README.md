# 相位轨迹可视化实验

证明 SPRiF 核心 claim：**spike 不重置慢状态**。

## 实验设计

**任务**: 合成相位轨迹任务（Cue-Delay 范式）
- Cue 阶段（100ms）：20 个 phase channel 编码初始相位 φ
- Delay 阶段（800ms）：无相位输入，模型靠内部状态维持旋转轨迹
- 6 个 probe 时刻（180, 300, 420, 540, 660, 780ms）：注入受控扰动诱发 spike
- 目标：ŷ_t = [cos(φ+ωt), sin(φ+ωt)]

**对照**: SPRiF（慢状态 3D，连续）vs ASRNN（膜电位 1D，被 reset）

## 文件结构

```
trajectory_visualization/
├── config.py              # 全局配置
├── generate_data.py       # 数据生成
├── models.py              # SPRiF + ASRNN 模型
├── train.py               # 训练脚本
├── record_forward.py      # 记录前向传播
├── plot_main_figure.py    # 绘制 5-panel 主图
├── run_all.py             # 编排器
├── checkpoints/           # 训练好的模型
└── README.md              # 本文档
```

## 运行命令

```bash
cd 代码/experiments

# 一键运行（训练 + 记录 + 绘图）
python trajectory_visualization/run_all.py

# 或分步运行:
python trajectory_visualization/train.py --model both --epochs 100
python trajectory_visualization/record_forward.py
python trajectory_visualization/plot_main_figure.py
```

## 输出

```
experiment-design-20260606/results/figures/trajectory_visualization/
├── main_figure_5panel.png          # 5-panel 主图
├── appendix_phi0.png               # φ=0 附录图
├── appendix_phi2.png               # φ=π 附录图
├── appendix_phi3.png               # φ=3π/2 附录图
└── trajectory_data_phi*.npz        # 记录的轨迹数据
```

## 核心验证

1. **SPRiF 慢状态连续**: x_t 在 probe 时刻相邻步差分小（不被 reset）
2. **ASRNN 膜电位跳变**: mem 在 spike 处显著下降（被 reset）
3. **输出对比**: SPRiF 2D 轨迹近似单位圆，ASRNN 在 spike 处可见畸变

## 参数来源

严格按 `SPRiF 状态轨迹可视化实验.md` 设计文档实现。

## 依赖

- `代码/Task_pSMNIST/core_algorithm/sprif_layer.py` — SPRiFNeuronLayer
- `代码/Task_GSC/SRNN_layers/spike_dense.py` — spike_dense
- `代码/Task_GSC/SRNN_layers/spike_neuron.py` — mem_update_adp
