# SPRiF 状态轨迹可视化实验

**Reset-Protected Spectral Trajectory Visualization**

---

## 目的

为 SPRiF 的 **Claim C2（"脉冲不重置记忆"）** 提供直接视觉证据。

核心实验命题：
> 让读者直观看到：慢状态负责连续时间轨迹，快状态负责 spike 和 reset；
> spike/reset 发生时，快状态跳变，但慢状态轨迹保持连续。

---

## 实验范式

```
Cue (100ms, 相位编码输入) → Delay (800ms, 无相位输入, 含受控扰动探针)
```

- **Cue 阶段**：20 个 phase channels 以 cosine-tuned Poisson rate 编码初始相位 φ
- **Delay 阶段**：Phase channels 关闭，模型须靠内部状态维持连续旋转轨迹
- **Perturbation Probes**：在 delay 阶段 6 个固定时间点（180, 300, 420, 540, 660, 780ms）向 hidden neurons 注入受控去极化电流，诱导 spike

SPRiF 和 LIF 在**完全相同**的输入、任务和训练设置下对比。

| 对比维度 | SPRiF | LIF |
|---------|-------|-----|
| 记忆载体 | 慢状态 x_t（不被 reset） | 膜电位 v_t（被 reset） |
| Readout 来源 | Slow state x_t | Membrane v_t |
| Spike 后记忆 | 连续 | 断裂 |

---

## 文件结构

```
trajectory_visualization/
├── README.md              # 本文件
├── config.py              # 全局配置（与设计文档保持一致）
├── generate_data.py       # 合成相位轨迹数据集生成
├── models.py              # SPRiF / LIF 模型定义
├── train.py               # 训练脚本（支持独立运行）
├── record_forward.py      # 状态轨迹记录
├── plot_main_figure.py    # AAAI 5-panel 主图
├── plot_appendix.py       # 附录 panels
├── run_all.py             # 端到端运行脚本
├── checkpoints/           # 模型权重
└── output/                # 输出图像
```

---

## 快速开始

### 完整运行

```bash
cd 代码/experiments/trajectory_visualization
python run_all.py
```

首次运行预计耗时：
- 数据生成：~30s
- SPRiF 训练（100 epochs, CPU）：~15min
- LIF 训练（100 epochs, CPU）：~5min
- 状态记录 + 绘图：~30s

GPU 上训练将显著加速。

### 分步运行

```bash
# 仅训练 SPRiF
python train.py --model sprif --epochs 100

# 仅训练 LIF
python train.py --model lif --epochs 100

# 使用已有 checkpoint，仅绘图
python run_all.py --skip-train --skip-record
```

---

## 输出

### AAAI 主文 Figure（5 panels）

`output/main_figure.png`:

| Panel | 内容 | 核心信息 |
|-------|------|---------|
| (a) | 任务示意图 | 让读者 1 秒内理解范式 |
| (b) | 慢状态相平面 (x¹, x²) | ★ 核心视觉：spike 不打断慢状态旋转 |
| (c) | 快状态投影式 reset (u⁰, u¹) | ★ 创新点证据：方向性、可学习的 reset |
| (d) | 时间域三层对照 | SPRiF 慢状态连续 vs 膜电位 reset vs LIF 膜电位 reset |
| (e) | 输出轨迹验证 | SPRiF 维持相位圆 vs LIF 轨迹畸变 |

### 附录 Panels

| Panel | 文件 | 移入附录理由 |
|-------|------|-------------|
| A | `appendix_a_input_structure.png` | 已被 main (a) 简洁版替代 |
| B | `appendix_b_input_raster.png` | 原始数据展示，无科学洞察 |
| C | `appendix_c_hidden_raster.png` | Spike 时刻已标记在 (b)(c)(d) |
| H | `appendix_h_probe_zoom_p*.png` | 过于微观；适合 rebuttal 或 oral PPT |
| J | `appendix_j_parameters.png` | 属于 C6 参数分析，另开 Figure |
| 多样本 | `appendix_multi_sample.png` | 证明非 cherry-picking |

---

## 设计文档对齐

所有参数、术语和 panel 布局与 `SPRiF 状态轨迹可视化实验.md` (Section 1-10) 保持一致：

- 时间参数：Section 1.1
- 输入通道：Section 1.2
- Cue 生成：Section 2
- Perturbation probes：Section 3
- 网络结构：Section 5（含 LIF 对照）
- Loss：Section 6
- 训练设置：Section 7
- 可视化：Section 8-10

---

## 依赖

- Python ≥ 3.8
- PyTorch ≥ 1.13
- NumPy, Matplotlib

SPRiF 核心层导入自 `代码/Task_ECG/core_algorithm/sprif_layer.py`。
