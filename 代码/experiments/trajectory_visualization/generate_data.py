"""
相位轨迹可视化实验 — 数据生成。

合成相位轨迹任务：Cue 阶段（100ms）用 20 个 phase channel 编码初始相位 φ，
Delay 阶段（800ms）无相位输入，模型靠内部状态维持旋转轨迹。
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    T_TOTAL, T_CUE,
    N_PHASE, N_PROBE, N_CHANNELS,
    R0, R1, R_PROBE,
    PROBE_TIMES, T_PROBE,
    CH_PROBE, CH_MARKER_CUE, CH_MARKER_DELAY,
    OMEGA_CHOICES,
)


def _poisson_spike(rate_hz: float, dt_ms: float = 1.0) -> float:
    """单次 Poisson 采样概率。"""
    prob = rate_hz * dt_ms / 1000.0
    return float(np.random.rand() < prob)


def generate_sample(
    phi: float,
    omega: float,
    probe_times: list = None,
    jitter: bool = False,
    seed: int = None,
) -> tuple:
    """
    生成单个相位轨迹样本。

    Args:
        phi: 初始相位
        omega: 角频率
        probe_times: probe 注入时刻列表（会被 jitter 修改）
        jitter: 是否对 probe 时刻添加均匀抖动
        seed: 随机种子

    Returns:
        input_spikes: [T, 32] 输入脉冲序列
        probe_mask:   [T] probe 注入掩码（1=注入时刻）
        target:       [T, 2] 目标轨迹 [cos, sin]
    """
    if seed is not None:
        np.random.seed(seed)
    if probe_times is None:
        probe_times = list(PROBE_TIMES)

    # 应用 jitter
    if jitter:
        probe_times = [t + np.random.randint(-20, 21) for t in probe_times]
        probe_times = [max(T_CUE, min(T_TOTAL - 1, t)) for t in probe_times]

    # 初始化
    input_spikes = np.zeros((T_TOTAL, N_CHANNELS), dtype=np.float32)
    probe_mask = np.zeros(T_TOTAL, dtype=np.float32)
    target = np.zeros((T_TOTAL, 2), dtype=np.float32)

    # Phase channels: cue 阶段发放
    for t in range(T_CUE):
        for i in range(N_PHASE):
            phi_i = 2.0 * math.pi * i / N_PHASE
            rate = R0 + R1 * math.cos(omega * t + phi - phi_i)
            rate = max(0.0, min(100.0, rate))  # 限制在 [0, 100] Hz
            if np.random.rand() < rate / 1000.0:
                input_spikes[t, i] = 1.0

    # Marker channels
    input_spikes[0, CH_MARKER_CUE] = 1.0      # cue 标志
    input_spikes[T_CUE, CH_MARKER_DELAY] = 1.0  # delay 标志

    # Probe channels: probe 窗口内 100Hz
    for t_probe in probe_times:
        for dt in range(T_PROBE):
            t = t_probe + dt
            if 0 <= t < T_TOTAL:
                probe_mask[t] = 1.0
                for j in range(N_PROBE):
                    if np.random.rand() < R_PROBE / 1000.0:
                        input_spikes[t, N_PHASE + j] = 1.0

    # Target: 旋转轨迹
    for t in range(T_TOTAL):
        target[t, 0] = math.cos(phi + omega * t)
        target[t, 1] = math.sin(phi + omega * t)

    return input_spikes, probe_mask, target


def generate_dataset(n: int, jitter: bool = True, seed: int = None) -> list:
    """
    批量生成样本。

    Returns:
        samples: list of (input_spikes, probe_mask, target)
    """
    if seed is not None:
        np.random.seed(seed)

    samples = []
    for _ in range(n):
        phi = np.random.uniform(0, 2 * math.pi)
        omega = np.random.choice(OMEGA_CHOICES)
        sample = generate_sample(phi, omega, jitter=jitter)
        samples.append(sample)
    return samples


class PhaseTrajectoryDataset(Dataset):
    """PyTorch Dataset 包装。"""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_spikes, probe_mask, target = self.samples[idx]
        return (
            torch.from_numpy(input_spikes),
            torch.from_numpy(probe_mask),
            torch.from_numpy(target),
        )
