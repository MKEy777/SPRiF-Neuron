"""
相位轨迹可视化实验 — 全局配置。

严格按 `SPRiF 状态轨迹可视化实验.md` 中的参数设置。
"""
import math
import os
import argparse

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 向上 3 级到 代码/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 图表输出
FIGURE_DIR = os.path.join(
    os.path.dirname(ROOT),
    "experiment-design-20260606",
    "results",
    "figures",
    "trajectory_visualization",
)
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
)

# ---------------------------------------------------------------------------
# 时间参数
# ---------------------------------------------------------------------------
T_TOTAL = 900          # 总时长 ms (dt=1ms，900 步)
T_CUE = 100            # Cue 阶段 ms

# ---------------------------------------------------------------------------
# 通道配置
# ---------------------------------------------------------------------------
N_PHASE = 20           # phase channel 数量
N_PROBE = 10           # probe channel 数量
N_MARKER = 2           # marker channel 数量
N_CHANNELS = N_PHASE + N_PROBE + N_MARKER  # 32

# 通道索引
CH_PHASE = slice(0, N_PHASE)
CH_PROBE = slice(N_PHASE, N_PHASE + N_PROBE)
CH_MARKER_CUE = N_PHASE + N_PROBE        # 30
CH_MARKER_DELAY = N_PHASE + N_PROBE + 1  # 31

# ---------------------------------------------------------------------------
# 发放率参数 (Hz)
# ---------------------------------------------------------------------------
R0 = 30.0              # 基础发放率
R1 = 25.0              # 调制幅度
R_PROBE = 100.0        # probe 发放率

# ---------------------------------------------------------------------------
# Probe 参数
# ---------------------------------------------------------------------------
PROBE_TIMES = [180, 300, 420, 540, 660, 780]  # 6 个 probe 注入时刻 (ms)
T_PROBE = 10            # probe 窗口时长 ms
A_PROBE = 3.0           # probe 注入幅度（可调，需足够大以诱发 spike）

# ---------------------------------------------------------------------------
# omega 选择
# ---------------------------------------------------------------------------
OMEGA_CHOICES = [
    2.0 * math.pi / 50.0,   # 快: T=50ms
    2.0 * math.pi / 100.0,  # 中: T=100ms
    2.0 * math.pi / 200.0,  # 慢: T=200ms
]

# ---------------------------------------------------------------------------
# 模型参数
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 64

# ---------------------------------------------------------------------------
# 训练参数
# ---------------------------------------------------------------------------
BETA = 0             # firing rate 正则系数
LR = 2e-3
EPOCHS = 100
BATCH_SIZE = 256
TRAIN_N = 10000
VAL_N = 1000
JITTER_RANGE = (-20, 20)  # probe 时间抖动 U(-20,20) ms
GRAD_CLIP = 1.0

# ---------------------------------------------------------------------------
# 可视化参数
# ---------------------------------------------------------------------------
VIZ_PHIS = [0.0, math.pi / 2, math.pi, 3.0 * math.pi / 2]
VIZ_OMEGA = 2.0 * math.pi / 100.0  # 中间频率


RUN_TAG = os.environ.get("RUN_TAG", "").strip()


def tagged(basename: str) -> str:
    """给文件名插入 RUN_TAG 后缀。

    tagged("main_figure_5panel.png") ->
        RUN_TAG="fast_a4" -> "main_figure_5panel_fast_a4.png"
        RUN_TAG=""        -> "main_figure_5panel.png"
    """
    if not RUN_TAG:
        return basename
    root, ext = os.path.splitext(basename)
    return f"{root}_{RUN_TAG}{ext}"


def get_args():
    """命令行参数解析，可覆盖默认值。"""
    parser = argparse.ArgumentParser(description="相位轨迹可视化实验")
    parser.add_argument("--model", type=str, default="both",
                        choices=["sprif", "asrnn", "both"],
                        help="训练哪个模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--a-probe", type=float, default=A_PROBE,
                        help="probe 注入幅度")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，使用已有 checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker 进程数")
    parser.add_argument("--train-n", type=int, default=None,
                        help="覆盖训练样本数（默认 TRAIN_N=10000）")
    parser.add_argument("--val-n", type=int, default=None,
                        help="覆盖验证样本数（默认 VAL_N=1000）")
    return parser.parse_args()
