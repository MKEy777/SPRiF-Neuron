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
