# SPRiF 实验双机并行运行指南

## 任务分配

| 机器 | 数据集 | 任务 | 预计时间 |
|------|--------|------|----------|
| **机器1** | pSMNIST | 训练 + trajectory_analysis | ~20 分钟 |
| **机器2** | GSC + ECG | 训练 + robustness 实验 | ~90 分钟 |

---

## 机器1 运行步骤（pSMNIST）

```bash
# ============================================================
# 机器1: pSMNIST 任务
# ============================================================

# --- 步骤1: 训练 pSMNIST SPRiF ---
cd 代码/Task_pSMNIST
python train.py --lr 1e-2 --epochs 150 --batch-size 512 --seed 0 --hidden-sizes 64 256 --mode srnn

# 等待训练完成...
# 输出: SPRiFpSMNISTNet_acc*.pth

# --- 步骤2: 运行轨迹分析实验 ---
cd 代码/experiments
python trajectory_analysis/trajectory_analyze.py

# 输出: trajectory_comparison.png, trajectory_data.npz

# --- 步骤3: 等待机器2完成，复制模型文件 ---
# 从机器2复制以下文件到 代码/Task_GSC/ 和 代码/Task_ECG/:
# - Task_GSC/SPRiFGSCNet_acc*.pth
# - Task_GSC/ASRNNGSCNet_acc*.pth
# - Task_ECG/SPRiFECGModel_acc*.pth
# - Task_ECG/ASRNNECGModel_acc*.pth

# 使用 scp 复制（在机器1上执行）:
# scp user@机器2:/path/to/代码/Task_GSC/*.pth 代码/Task_GSC/
# scp user@机器2:/path/to/代码/Task_ECG/*.pth 代码/Task_ECG/

# --- 步骤4: 运行需要多任务模型的实验 ---
cd 代码/experiments

# 冲激响应分析
python impulse_analysis/impulse_analyze.py

# Reset 方向分析
python reset_analysis/reset_analyze.py

# 参数可视化
python param_visualization/param_visualize.py

echo "机器1 全部完成！"
```

---

## 机器2 运行步骤（GSC + ECG）

```bash
# ============================================================
# 机器2: GSC + ECG 任务
# ============================================================

# --- 步骤1: 数据准备 ---

# GSC 数据已复制:
# autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/speech_commands_v0.02
# 缓存目录:
# autodl-tmp/A-sprif/Task_GSC/dataset/SpeechCommands/cache_power_to_db

# ECG 数据准备（手动放置）
# 确保以下文件存在:
# - 代码/Task_ECG/data/QTDB_train.mat
# - 代码/Task_ECG/data/QTDB_test.mat

# --- 步骤2: 训练 GSC 模型 ---

# GSC SPRiF (最佳参数已配置为默认值)
cd 代码/Task_GSC
python train.py --epochs 150 --batch-size 200 --seed 42 --hidden-sizes 300

# GSC ASRNN (源码参数)
python train_asrnn.py --epochs 30 --batch-size 32 --seed 0 --hidden-size 256

# --- 步骤3: 训练 ECG 模型 ---

# ECG SPRiF (最佳参数已配置为默认值)
cd 代码/Task_ECG
python train.py --epochs 250 --batch-size 64 --seed 1111 --hidden-sizes 36

# ECG ASRNN
python train_asrnn.py --epochs 250 --batch-size 64 --seed 1111 --hidden-size 36

# --- 步骤4: 运行鲁棒性实验 ---
cd 代码/experiments

# R1: 噪声鲁棒性基准测试
python robustness/noise_benchmark.py

# R2: 序列长度 × 噪声
python robustness/sequence_noise.py

# R3: 频率选择性
python robustness/frequency_selectivity.py

echo "机器2 鲁棒性实验完成！"

# --- 步骤5: 复制模型到机器1（如果需要） ---
# 将以下文件复制到机器1:
# scp Task_GSC/*.pth user@机器1:/path/to/代码/Task_GSC/
# scp Task_ECG/*.pth user@机器1:/path/to/代码/Task_ECG/

echo "机器2 全部完成！"
```

---

## 时间线

```
时间    机器1 (pSMNIST)              机器2 (GSC+ECG)
────────────────────────────────────────────────────────────
0min    开始训练 pSMNIST             下载数据 + 开始训练 GSC SPRiF
15min   pSMNIST 训练完成             GSC SPRiF 完成，开始 GSC ASRNN
16min   trajectory_analysis 完成     GSC ASRNN 训练中...
30min   等待机器2...                 GSC ASRNN 完成，开始 ECG SPRiF
55min   等待机器2...                 ECG SPRiF 完成，开始 ECG ASRNN
80min   等待机器2...                 ECG ASRNN 完成
81min   收到模型，开始分析实验        开始 robustness 实验
85min   分析实验完成！               robustness 实验完成！
```

---

## 模型文件交换

### 从机器2复制到机器1

训练完成后，机器2需要将模型文件复制到机器1，以便运行 impulse/reset/param 实验：

```bash
# 在机器1上执行（从机器2拉取）:
scp user@机器2:/path/to/代码/Task_GSC/SPRiFGSCNet_*.pth 代码/Task_GSC/
scp user@机器2:/path/to/代码/Task_GSC/ASRNNGSCNet_*.pth 代码/Task_GSC/
scp user@机器2:/path/to/代码/Task_ECG/SPRiFECGModel_*.pth 代码/Task_ECG/
scp user@机器2:/path/to/代码/Task_ECG/ASRNNECGModel_*.pth 代码/Task_ECG/
```

---

## 输出文件位置

### 机器1 输出
```
代码/experiments/
└── experiment-design-20260606/results/figures/
    └── trajectory_analysis/
        ├── trajectory_comparison.png
        └── trajectory_data.npz
```

### 机器2 输出
```
代码/experiments/
├── robustness/
│   ├── robustness_benchmark.png
│   ├── sequence_noise.png
│   └── frequency_selectivity.png
└── experiment-design-20260606/results/
    ├── robustness_benchmark.json
    ├── sequence_noise.json
    └── frequency_selectivity.json
```

### 机器1 完成模型交换后输出
```
代码/experiments/
└── experiment-design-20260606/results/figures/
    ├── trajectory_analysis/
    ├── impulse_analysis/
    │   ├── impulse_response_gallery.png
    │   ├── frequency_response.png
    │   └── impulse_kernel_stats.csv
    ├── reset_analysis/
    │   ├── lambda_distribution.png
    │   ├── lambda_vs_*.png
    │   └── lambda_stats.csv
    └── param_visualization/
        ├── param_distributions.png
        └── param_per_task.csv
```

---

## 快速启动命令

### 机器1 一键启动
```bash
cd 代码/Task_pSMNIST && \
python train.py --lr 1e-2 --epochs 150 --batch-size 512 --seed 0 --hidden-sizes 64 256 --mode srnn && \
cd ../experiments && \
python trajectory_analysis/trajectory_analyze.py
```

### 机器2 一键启动（数据已准备）
```bash
cd 代码/Task_GSC && \
python train.py --epochs 150 --batch-size 200 --seed 42 --hidden-sizes 300 && \
python train_asrnn.py --epochs 30 --batch-size 32 --seed 0 --hidden-size 256 && \
cd ../Task_ECG && \
python train.py --epochs 250 --batch-size 64 --seed 1111 --hidden-sizes 36 && \
python train_asrnn.py --epochs 250 --batch-size 64 --seed 1111 --hidden-size 36 && \
cd ../experiments && \
python robustness/noise_benchmark.py && \
python robustness/sequence_noise.py && \
python robustness/frequency_selectivity.py
```

---

## 注意事项

1. **GPU**: 两台机器都使用 GPU 训练
2. **数据同步**: 机器2训练完成后，及时将模型文件复制到机器1
3. **结果收集**: 最后从两台机器收集所有输出文件
4. **错误处理**: 如果某个实验失败，检查对应的模型 checkpoint 是否存在

---

## 实验 6: 相位轨迹可视化（trajectory_visualization）

**目的**: 证明 SPRiF 慢状态在 spike 时刻连续（Claim C2），对比 ASRNN 膜电位被 reset 的结构性差异。

**任务**: 合成相位轨迹任务（Cue-Delay 范式），Cue 阶段编码初始相位，Delay 阶段维持旋转轨迹，probe 时刻注入扰动诱发 spike。

**前置**: 无外部依赖，合成数据，自包含训练。可在任一机器上独立运行。

### 运行命令

```bash
cd 代码/experiments

# 一键运行（训练 SPRiF+ASRNN + 记录 + 绘图，约 10-15 分钟）
python trajectory_visualization/run_all.py

# 或分步运行:
# 步骤1: 训练（默认 100 epochs）
python trajectory_visualization/train.py --model both --epochs 100

# 步骤2: 记录前向传播（4 个样本 × 2 模型）
python trajectory_visualization/record_forward.py

# 步骤3: 绘制 5-panel 主图
python trajectory_visualization/plot_main_figure.py
```

### 输出

```
experiment-design-20260606/results/figures/trajectory_visualization/
├── main_figure_5panel.png          # 5-panel 主图（论文用）
├── appendix_phi0.png               # φ=0 附录图
├── appendix_phi2.png               # φ=π 附录图
├── appendix_phi3.png               # φ=3π/2 附录图
└── trajectory_data_phi*.npz        # 记录的轨迹数据（可复现）

代码/experiments/trajectory_visualization/checkpoints/
├── TrajectoryViz_SPRiF_mse*.pth    # SPRiF 最佳模型
└── TrajectoryViz_ASRNN_mse*.pth    # ASRNN 最佳模型
```

### 核心验证

1. **SPRiF 慢状态连续**: x_t 在 probe 时刻相邻步差分小（不被 reset）
2. **ASRNN 膜电位跳变**: mem 在 spike 处显著下降（被 reset）
3. **输出对比**: SPRiF 2D 轨迹近似单位圆，ASRNN 在 spike 处可见畸变
4. **Probe 诱发 spike**: probe window 内 hidden spike rate 显著高于非 probe 时段

### 参数说明

- 默认参数严格按 `SPRiF 状态轨迹可视化实验.md` 设计文档
- 可通过命令行覆盖：`--epochs`, `--lr`, `--batch-size`, `--hidden-size`, `--a-probe`
- 默认 probe 幅度 A_probe=1.0，若不诱发 spike 可调大（如 `--a-probe 2.0`）

### 预期时间

- 训练: ~10 分钟（GPU）
- 记录: ~1 分钟
- 绘图: ~30 秒
- 总计: ~12 分钟
