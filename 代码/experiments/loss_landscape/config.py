"""
S-MNIST 损失景观与梯度可视化实验 — 全局配置。

对比 SPRiF 与 LIF 两种神经元：
  1. 2D filter-normalized 损失景观（Li et al. 2018）
  2. 3D 损失曲面
  3. BPTT 梯度范数随时间步/层传播曲线
"""
import os
import argparse

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 本文件所在目录：代码/experiments/loss_landscape
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上 2 级到 代码/
CODE_ROOT = os.path.dirname(os.path.dirname(SELF_DIR))
# 依赖的两个神经元层所在任务目录
SMNIST_DIR = os.path.join(CODE_ROOT, "Task_S-MNIST")
GSC_DIR = os.path.join(CODE_ROOT, "Task_GSC")

# checkpoint / 中间数据
CHECKPOINT_DIR = os.path.join(SELF_DIR, "checkpoints")
DATA_ROOT = os.path.join(SMNIST_DIR, "data")

# 图表输出（本项目目录下）
FIGURE_DIR = os.path.join(SELF_DIR, "figures")

# ---------------------------------------------------------------------------
# 模型参数（SPRiF 与 LIF 保持相同结构，参数量可比）
# ---------------------------------------------------------------------------
INPUT_SIZE = 1
HIDDEN_SIZES = [64, 256]
NUM_CLASSES = 10
MODE = "srnn"          # sfnn / srnn（recurrent）
WARMUP_STEPS = 0

# ---------------------------------------------------------------------------
# 训练参数
# ---------------------------------------------------------------------------
LR = 1e-2
EPOCHS = 60
BATCH_SIZE = 512
TBPTT_LEN = 262
SCHEDULER_STEP = 25
SCHEDULER_GAMMA = 0.1
SEED = 0

# ---------------------------------------------------------------------------
# 损失景观参数
# ---------------------------------------------------------------------------
GRID_RESOLUTION = 25       # 网格分辨率 N×N
GRID_RANGE = 1.0           # 扰动系数范围 [-GRID_RANGE, GRID_RANGE]
DIRECTION_SEED = 123       # 随机方向种子
LANDSCAPE_EVAL_BATCHES = 4 # 每个网格点用多少个 batch 估计 loss（子集加速）

# ---------------------------------------------------------------------------
# 梯度可视化参数
# ---------------------------------------------------------------------------
GRAD_BATCH_SIZE = 64       # 梯度分析用单 batch 大小
GRAD_NUM_TIMESTEPS = 40    # 采样多少个时间步记录梯度（沿 784 步均匀采样）


def get_args():
    """命令行参数解析，可覆盖默认值。"""
    parser = argparse.ArgumentParser(description="S-MNIST 损失景观与梯度可视化")
    parser.add_argument("--neuron", type=str, default="both",
                        choices=["sprif", "lif", "both"],
                        help="训练/分析哪个神经元")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-resolution", type=int, default=GRID_RESOLUTION)
    parser.add_argument("--grid-range", type=float, default=GRID_RANGE)
    parser.add_argument("--eval-batches", type=int, default=LANDSCAPE_EVAL_BATCHES,
                        help="每个网格点估计 loss 用的 batch 数")
    parser.add_argument("--subset", type=int, default=0,
                        help="仅用前 N 个训练样本（0=全量，用于快速验证）")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()