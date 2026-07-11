"""
相位轨迹可视化实验 — 数据生成（向量化加速版）。

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


def generate_sample(
    phi: float,
    omega: float,
    probe_times: list = None,
    jitter: bool = False,
    rng: np.random.Generator = None,
) -> tuple:
    """
    生成单个相位轨迹样本（向量化版）。

    Returns:
        input_spikes: [T, 32] 输入脉冲序列
        probe_mask:   [T] probe 注入掩码（1=注入时刻）
        target:       [T, 2] 目标轨迹 [cos, sin]
    """
    if rng is None:
        rng = np.random.default_rng()
    if probe_times is None:
        probe_times = list(PROBE_TIMES)

    # 应用 jitter
    if jitter:
        probe_times = [t + int(rng.integers(-20, 21)) for t in probe_times]
        probe_times = [max(T_CUE, min(T_TOTAL - 1, t)) for t in probe_times]

    # 初始化
    input_spikes = np.zeros((T_TOTAL, N_CHANNELS), dtype=np.float32)
    probe_mask = np.zeros(T_TOTAL, dtype=np.float32)

    # ---- Phase channels (cue 阶段) 向量化 ----
    t_cue = np.arange(T_CUE, dtype=np.float32)                       # [T_CUE]
    i_arr = np.arange(N_PHASE, dtype=np.float32)                     # [N_PHASE]
    phi_i = 2.0 * math.pi * i_arr / N_PHASE                          # [N_PHASE]
    # rate[t, i] = R0 + R1 * cos(omega*t + phi - phi_i)
    rate = R0 + R1 * np.cos(omega * t_cue[:, None] + phi - phi_i[None, :])
    rate = np.clip(rate, 0.0, 100.0)
    prob = rate / 1000.0                                             # dt=1ms
    rand_mat = rng.random(size=(T_CUE, N_PHASE))
    input_spikes[:T_CUE, :N_PHASE] = (rand_mat < prob).astype(np.float32)

    # ---- Marker channels ----
    input_spikes[0, CH_MARKER_CUE] = 1.0
    input_spikes[T_CUE, CH_MARKER_DELAY] = 1.0

    # ---- Probe channels 向量化 ----
    for t_probe in probe_times:
        t_start = max(0, t_probe)
        t_end = min(T_TOTAL, t_probe + T_PROBE)
        if t_end <= t_start:
            continue
        w = t_end - t_start
        probe_mask[t_start:t_end] = 1.0
        rand_probe = rng.random(size=(w, N_PROBE))
        input_spikes[t_start:t_end, N_PHASE:N_PHASE + N_PROBE] =(
            rand_probe < R_PROBE / 1000.0
        ).astype(np.float32)

    # ---- Target 向量化 ----
    t_all = np.arange(T_TOTAL, dtype=np.float32)
    target = np.stack(
        [np.cos(phi + omega * t_all), np.sin(phi + omega * t_all)],
        axis=1,
    ).astype(np.float32)

    return input_spikes, probe_mask, target


def generate_dataset(n: int, jitter: bool = True, seed: int = None) -> list:
    """批量生成样本（使用独立 Generator，避免全局种子冲突）。"""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        phi = float(rng.uniform(0, 2 * math.pi))
        omega = float(rng.choice(OMEGA_CHOICES))
        sample = generate_sample(phi, omega, jitter=jitter, rng=rng)
        samples.append(sample)
    return samples


class PhaseTrajectoryDataset(Dataset):
    """PyTorch Dataset 包装（预先转 tensor 减少 __getitem__ 开销）。"""

    def __init__(self, samples: list):
        # 预 stack 成大 tensor，__getitem__ 只做切片
        xs = np.stack([s[0] for s in samples], axis=0)
        pms = np.stack([s[1] for s in samples], axis=0)
        ts = np.stack([s[2] for s in samples], axis=0)
        self.input_spikes = torch.from_numpy(xs)
        self.probe_mask = torch.from_numpy(pms)
        self.target = torch.from_numpy(ts)

    def __len__(self):
        return self.input_spikes.shape[0]

    def __getitem__(self, idx):
        return (
            self.input_spikes[idx],
            self.probe_mask[idx],
            self.target[idx],
        )