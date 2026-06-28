"""
SPRiF 状态轨迹可视化实验 — 全局配置
=============================================

本实验为 SPRiF 的 Claim C2（"脉冲不重置记忆"）提供直接视觉证据。
所有参数与 `SPRiF 状态轨迹可视化实验.md` 设计文档保持一致。

实验范式:
    Cue (100ms, 相位输入) -> Delay (800ms, 无相位输入, 含受控扰动探针)
    模型须靠内部状态维持连续相位轨迹。

Author: SPRiF Neuron Project
Date: 2026-06-20
"""

import math
import os

# ============================================================================
# 路径
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "output")
MODEL_DIR = os.path.join(EXP_DIR, "checkpoints")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 时间参数 (单位: ms)
# ============================================================================

DT = 1                     # 时间步长
T_CUE = 100                # Cue 阶段长度
T_DELAY = 800              # Delay 阶段长度
T_TOTAL = T_CUE + T_DELAY  # 总长度 = 900 ms

# ============================================================================
# 输入通道
# ============================================================================

N_PHASE_CH = 20            # 相位编码通道
N_PROBE_CH = 10            # 扰动探针通道
N_MARKER_CH = 2            # 标记通道 (cue / delay)
N_INPUT_CH = N_PHASE_CH + N_PROBE_CH + N_MARKER_CH  # = 32

# 通道索引
PHASE_CH_START = 0
PHASE_CH_END = 20
PROBE_CH_START = 20
PROBE_CH_END = 30
MARKER_CUE_CH = 30
MARKER_DELAY_CH = 31

# ============================================================================
# Cue 参数
# ============================================================================

R0 = 30.0                  # 基线发放率 (Hz)
R1 = 25.0                  # 调制幅度 (Hz)
OMEGAS = [
    2.0 * math.pi / 50.0,   # ω₁: T = 50ms
    2.0 * math.pi / 100.0,  # ω₂: T = 100ms
    2.0 * math.pi / 200.0,  # ω₃: T = 200ms
]

# ============================================================================
# Perturbation Probe 参数
# ============================================================================

T_PROBES = [180, 300, 420, 540, 660, 780]  # probe 绝对时间 (ms)
T_PROBE_DURATION = 10                       # 每个 probe 持续时长 (ms)
R_PROBE = 100.0                             # probe 期间输入通道发放率 (Hz)
A_PROBE = 1.0                               # 扰动电流幅值 (加到 input_current)
PROBE_JITTER = 20                           # 训练时 probe 位置随机 jitter (±ms)

# ============================================================================
# 网络参数
# ============================================================================

HIDDEN_SIZE = 64           # SPRiF/LIF hidden neurons
OUTPUT_SIZE = 2            # (cos, sin) 输出

# SPRiF 初始化范围
SPRIF_KWARGS = {
    "threshold": 1.0,
    "init_std": 0.05,
    "bias": False,
    "tau_alpha_range": (20.0, 120.0),
    "tau_rho_range": (4.0, 30.0),
    "tau_eta_range": (0.8, 8.0),
    "omega_range": (0.02 * math.pi, 0.20 * math.pi),
}

# LIF 参数
LIF_TAU_M = 20.0           # 膜时间常数 (ms)
LIF_THRESHOLD = 1.0

# ============================================================================
# 训练参数
# ============================================================================

TRAIN_SAMPLES = 10000
VAL_SAMPLES = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
BETA_FR = 1e-4             # firing-rate regularization 权重
GRAD_CLIP = 1.0
SEED = 42

# ============================================================================
# 可视化参数
# ============================================================================

VIS_OMEGA = 2.0 * math.pi / 100.0  # 可视化用固定频率
VIS_PHASES = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]

# 时间域放大窗口 (Panel d)
ZOOM_WINDOW = 30           # spike 前后 ±30ms
ZOOM_SPIKE_TIME = 420      # 默认聚焦的 spike 时间 (ms)

# 绘图样式
COLORS = {
    "sprif_slow": "#1b9e77",
    "sprif_fast": "#d95f02",
    "lif": "#7570b3",
    "spike": "#e41a1c",
    "target": "#333333",
    "cue_bg": "#f0f0f0",
    "probe_bg": "#fff3cd",
}
