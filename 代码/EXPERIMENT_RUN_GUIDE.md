# SPRiF 实验运行指南

本文档包含所有实验的运行命令，适用于服务器部署。

---

## 目录结构

```
代码/
├── Task_pSMNIST/          # PS-MNIST 任务
├── Task_GSC/              # GSC 语音任务
├── Task_ECG/              # ECG 心电任务
├── experiments/           # 实验脚本
│   ├── trajectory_analysis/
│   ├── impulse_analysis/
│   ├── reset_analysis/
│   ├── param_visualization/
│   └── robustness/
└── ASRNN/                 # ASRNN 原始代码（参考）
```

---

## 数据集准备

### 1. PS-MNIST (自动下载)

```bash
# 无需手动下载，运行训练脚本时自动从 torchvision 下载
```

### 2. GSC (Google Speech Commands)

```bash
# 下载数据集
cd 代码/Task_GSC
python download_GSC.py

# 数据集位置: 代码/Task_GSC/data/SpeechCommands/
```

### 3. ECG (QTDB)

```bash
# 将 QTDB_train.mat 和 QTDB_test.mat 放到以下位置:
# 代码/Task_ECG/data/QTDB_train.mat
# 代码/Task_ECG/data/QTDB_test.mat

mkdir -p 代码/Task_ECG/data
# 手动下载并放置数据文件
```

---

## 模型训练命令

### PS-MNIST (SPRiF)

```bash
cd 代码/Task_pSMNIST
python train.py \
    --lr 1e-2 \
    --epochs 150 \
    --batch-size 512 \
    --seed 0 \
    --hidden-sizes 64 256 \
    --mode srnn
```

**输出**: `Task_pSMNIST/SPRiFpSMNISTNet_acc*.pth`

---

### GSC (SPRiF)

```bash
cd 代码/Task_GSC
python train.py --epochs 150 --batch-size 200 --seed 42 --hidden-sizes 300
```

**输出**: `Task_GSC/SPRiFGSCNet_acc*.pth`

---

### GSC (ASRNN)

```bash
cd 代码/Task_GSC
python train_asrnn.py \
    --lr 3e-3 \
    --epochs 30 \
    --batch-size 32 \
    --seed 0 \
    --hidden-size 256
```

**输出**: `Task_GSC/ASRNNGSCNet_acc*.pth`

**ASRNN 参数说明** (与原始论文一致):
- `hidden_size=256`: 隐藏层大小
- `lr=3e-3`: 学习率
- `tau_m=20, tau_adp=200`: 膜电位和自适应阈值时间常数

---

### ECG/QTDB (SPRiF)

```bash
cd 代码/Task_ECG
python train.py \
    --train-mat data/QTDB_train.mat \
    --test-mat data/QTDB_test.mat \
    --lr 1e-2 \
    --epochs 250 \
    --batch-size 64 \
    --seed 1111 \
    --hidden-sizes 36 \
    --neuron-threshold 0.6
```

**输出**: `Task_ECG/SPRiFECGModel_acc*.pth`

---

### ECG/QTDB (ASRNN)

```bash
cd 代码/Task_ECG
python train_asrnn.py \
    --train-mat data/QTDB_train.mat \
    --test-mat data/QTDB_test.mat \
    --lr 1e-2 \
    --epochs 250 \
    --batch-size 64 \
    --seed 1111 \
    --hidden-size 36
```

**输出**: `Task_ECG/ASRNNECGModel_acc*.pth`

**ASRNN 参数说明** (与原始论文一致):
- `hidden_size=36`: 隐藏层大小（论文使用36个循环神经元）
- `lr=1e-2`: 学习率
- `tau_m_h=20, tau_m_o=20`: 隐藏层和输出层膜电位时间常数
- `tau_adp_h=7, tau_adp_o=100`: 自适应阈值时间常数

---

## 实验运行命令

### 实验 1: 轨迹分析 (Trajectory Analysis)

**目的**: 证明 SPRiF 慢状态在脉冲时刻连续

**前置**: 需要训练 PS-MNIST SPRiF 模型

```bash
cd 代码/experiments
python trajectory_analysis/trajectory_analyze.py
```

**输出**: `experiment-design-20260606/results/figures/trajectory_analysis/`
- `trajectory_comparison.png`
- `trajectory_data.npz`

---

### 实验 2: 冲激响应分析 (Impulse Analysis)

**目的**: 展示 SPRiF 学到的多样化时域核

**前置**: 需要训练所有三个任务的 SPRiF 模型

```bash
cd 代码/experiments
python impulse_analysis/impulse_analyze.py
```

**输出**: `experiment-design-20260606/results/figures/impulse_analysis/`
- `impulse_response_gallery.png`
- `frequency_response.png`
- `lif_comparison.png`
- `impulse_kernel_stats.csv`

---

### 实验 3: Reset 方向分析 (Reset Analysis)

**目的**: 分析 λ 分布及其与谱参数的相关性

**前置**: 需要训练所有三个任务的 SPRiF 模型

```bash
cd 代码/experiments
python reset_analysis/reset_analyze.py
```

**输出**: `experiment-design-20260606/results/figures/reset_analysis/`
- `lambda_distribution.png`
- `lambda_vs_firing_rate.png`
- `lambda_vs_alpha.png`
- `lambda_vs_omega.png`
- `lambda_stats.csv`

---

### 实验 4: 参数可视化 (Param Visualization)

**目的**: 可视化 α/ρ/ω 参数分布

**前置**: 需要训练所有三个任务的 SPRiF 模型

```bash
cd 代码/experiments
python param_visualization/param_visualize.py
```

**输出**: `experiment-design-20260606/results/figures/param_visualization/`
- `param_distributions.png`
- `param_per_task.csv`

---

### 实验 5: 鲁棒性实验 (Robustness Experiments)

**目的**: 对比 SPRiF vs ASRNN 在噪声下的表现

**前置**: 需要训练 GSC 和 ECG 的 SPRiF 和 ASRNN 模型

#### R1: 噪声鲁棒性基准测试

```bash
cd 代码/experiments
python robustness/noise_benchmark.py
```

**输出**:
- `robustness/robustness_benchmark.png`
- `experiment-design-20260606/results/robustness_benchmark.json`

#### R2: 序列长度 × 噪声

```bash
cd 代码/experiments
python robustness/sequence_noise.py
```

**输出**:
- `robustness/sequence_noise.png`
- `experiment-design-20260606/results/sequence_noise.json`

#### R3: 频率选择性

```bash
cd 代码/experiments
python robustness/frequency_selectivity.py
```

**输出**:
- `robustness/frequency_selectivity.png`
- `experiment-design-20260606/results/frequency_selectivity.json`

---

## 完整运行流程

### 阶段 1: 数据准备

```bash
# GSC 数据下载
cd 代码/Task_GSC && python download_GSC.py

# ECG 数据放置（手动）
# 将 QTDB_train.mat 和 QTDB_test.mat 放到 代码/Task_ECG/data/
```

### 阶段 2: 训练所有模型

```bash
# PS-MNIST (SPRiF)
cd 代码/Task_pSMNIST && python train.py --lr 1e-2 --epochs 150 --batch-size 512 --seed 0 --hidden-sizes 64 256 --mode srnn

# GSC (SPRiF)
cd 代码/Task_GSC && python train.py --lr 5e-3 --epochs 150 --batch-size 200 --seed 42 --hidden-sizes 300 --neuron-threshold 0.8 --neuron-init-std 0.15 --tau-alpha-min 10.0 --tau-alpha-max 80.0 --tau-rho-min 4.0 --tau-rho-max 30.0 --tau-eta-min 0.8 --tau-eta-max 8.0 --omega-min 0.04 --omega-max 0.40

# GSC (ASRNN)
cd 代码/Task_GSC && python train_asrnn.py --data-root data/SpeechCommands --lr 3e-3 --epochs 150 --batch-size 32 --seed 42 --hidden-size 256

# ECG (SPRiF)
cd 代码/Task_ECG && python train.py --train-mat data/QTDB_train.mat --test-mat data/QTDB_test.mat --lr 1e-2 --epochs 250 --batch-size 64 --seed 1111 --hidden-sizes 36 --neuron-threshold 0.6

# ECG (ASRNN)
cd 代码/Task_ECG && python train_asrnn.py --train-mat data/QTDB_train.mat --test-mat data/QTDB_test.mat --lr 1e-2 --epochs 250 --batch-size 64 --seed 1111 --hidden-size 36
```

### 阶段 3: 运行分析实验

```bash
cd 代码/experiments

# 实验 1: 轨迹分析
python trajectory_analysis/trajectory_analyze.py

# 实验 2: 冲激响应分析
python impulse_analysis/impulse_analyze.py

# 实验 3: Reset 方向分析
python reset_analysis/reset_analyze.py

# 实验 4: 参数可视化
python param_visualization/param_visualize.py

# 实验 5: 鲁棒性实验
python robustness/noise_benchmark.py
python robustness/sequence_noise.py
python robustness/frequency_selectivity.py
```

---

## 模型参数总结

| 模型 | 任务 | Hidden Size | Epochs | 参数量 |
|------|------|-------------|--------|--------|
| SPRiF | PS-MNIST | [64, 256] | 150 | ~67K |
| SPRiF | GSC | [300] | 150 | ~130K |
| ASRNN | GSC | 256 | 150 | ~0.3M |
| SPRiF | ECG | [36] | 250 | ~1.8K |
| ASRNN | ECG | 36 | 250 | ~10K |

---

## 输出目录结构

```
experiment-design-20260606/results/
├── figures/
│   ├── trajectory_analysis/
│   ├── impulse_analysis/
│   ├── reset_analysis/
│   └── param_visualization/
├── table1_main_results.json
├── table2_ablation_results.json
├── robustness_benchmark.json
├── sequence_noise.json
└── frequency_selectivity.json
```

---

## 注意事项

1. **数据路径**: 确保所有数据文件放在正确位置
2. **GPU**: 建议在 GPU 上训练，CPU 训练会很慢
3. **Checkpoint**: 训练完成后会生成 `.pth` 文件，实验脚本会自动加载
4. **ASRNN 参数**: 使用论文中的默认参数，已验证与原始实现一致

---

## 数据集占位符

如果数据集尚未下载，请使用以下占位符路径：

| 数据集 | 占位符路径 |
|--------|-----------|
| GSC | `代码/Task_GSC/data/SpeechCommands/` |
| ECG | `代码/Task_ECG/data/QTDB_train.mat` |
| ECG | `代码/Task_ECG/data/QTDB_test.mat` |
| PS-MNIST | 自动下载（无需手动准备） |
