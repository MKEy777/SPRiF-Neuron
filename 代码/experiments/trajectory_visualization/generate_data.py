"""
合成相位轨迹数据集生成器
================================

生成 Cue → Delay with perturbation probes 的 Poisson 脉冲输入。

每个样本:
    - Cue 阶段 (0-100ms): 20 个 phase channels 以 cosine-tuned Poisson rate 发放
    - Delay 阶段 (100-900ms): phase channels 关闭, probe channels 在固定窗口发放
    - 目标: 在 delay 阶段输出 (cos(φ+ωt), sin(φ+ωt))

Usage:
    from generate_data import generate_dataset
    train_loader = generate_dataset(n_samples=10000, train=True)
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DT, T_CUE, T_DELAY, T_TOTAL,
    N_PHASE_CH, N_PROBE_CH, N_MARKER_CH, N_INPUT_CH,
    R0, R1, OMEGAS,
    T_PROBES, T_PROBE_DURATION, R_PROBE, PROBE_JITTER,
    MARKER_CUE_CH, MARKER_DELAY_CH,
    SEED,
)


def _poisson_spikes(rates: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Generate Poisson spikes for given rates (Hz) with dt=1ms.

    Args:
        rates: [T, C] firing rates in Hz
        rng: numpy RandomState

    Returns:
        spikes: [T, C] binary spike trains
    """
    prob = rates * DT / 1000.0  # rate * dt in seconds
    prob = np.clip(prob, 0.0, 1.0)
    return (rng.random(rates.shape) < prob).astype(np.float32)


def _build_probe_mask(
    t_probes: np.ndarray, duration: int = T_PROBE_DURATION
) -> np.ndarray:
    """Build a binary mask indicating probe windows.

    Args:
        t_probes: [n_probes] probe start times (ms)
        duration: probe duration (ms)

    Returns:
        mask: [T_TOTAL] binary mask, 1 inside probe windows
    """
    mask = np.zeros(T_TOTAL, dtype=np.float32)
    for tp in t_probes:
        start = int(tp)
        end = min(start + duration, T_TOTAL)
        mask[start:end] = 1.0
    return mask


def generate_sample(
    phi: float,
    omega: float,
    t_probes: np.ndarray,
    rng: np.random.RandomState,
    jitter_probes: bool = False,
    jitter_range: int = PROBE_JITTER,
) -> dict:
    """Generate a single synthetic trajectory sample.

    Args:
        phi: initial phase (rad)
        omega: angular frequency (rad/ms)
        t_probes: probe times (ms), will be jittered if jitter_probes=True
        rng: numpy RandomState
        jitter_probes: whether to apply random jitter to probe positions
        jitter_range: max jitter magnitude (ms)

    Returns:
        dict with keys:
            input_spikes: [T_TOTAL, N_INPUT_CH] binary spike raster
            probe_mask:   [T_TOTAL] binary perturbation mask
            target:       [T_TOTAL, 2] target trajectory (cos, sin)
            phi:          float, initial phase
            omega:        float, angular frequency
    """
    T = T_TOTAL

    # ---- Jitter probes (training only) ----
    if jitter_probes:
        jitter = rng.randint(-jitter_range, jitter_range + 1, size=len(t_probes))
        t_probes_jittered = np.clip(
            t_probes.astype(float) + jitter, T_CUE + 1, T_TOTAL - T_PROBE_DURATION
        )
    else:
        t_probes_jittered = t_probes.astype(float)

    # ---- Phase channel rates ----
    # preferred phases evenly spaced over [0, 2π)
    phi_i = np.linspace(0.0, 2.0 * np.pi, N_PHASE_CH, endpoint=False)

    t_arr = np.arange(T, dtype=np.float32)
    rates = np.zeros((T, N_INPUT_CH), dtype=np.float32)

    # Cue period: r_i(t) = r0 + r1 * cos(ωt + φ - φ_i)
    cue_mask = t_arr < T_CUE
    for i in range(N_PHASE_CH):
        rates[cue_mask, i] = (
            R0 + R1 * np.cos(omega * t_arr[cue_mask] + phi - phi_i[i])
        )
    rates[cue_mask, :N_PHASE_CH] = np.clip(rates[cue_mask, :N_PHASE_CH], 0.0, None)

    # Delay period: phase channels off (rates already 0)

    # ---- Probe channels ----
    for tp in t_probes_jittered:
        start = int(tp)
        end = min(start + T_PROBE_DURATION, T_TOTAL)
        rates[start:end, N_PHASE_CH:N_PHASE_CH + N_PROBE_CH] = R_PROBE

    # ---- Marker channels ----
    rates[:T_CUE, MARKER_CUE_CH] = 1.0      # cue marker (deterministic)
    rates[T_CUE:, MARKER_DELAY_CH] = 1.0    # delay marker (deterministic)

    # ---- Generate Poisson spikes ----
    input_spikes = _poisson_spikes(rates, rng)

    # Marker channels are deterministic, not Poisson
    input_spikes[:T_CUE, MARKER_CUE_CH] = 1.0
    input_spikes[T_CUE:, MARKER_DELAY_CH] = 1.0

    # ---- Probe mask (for current injection) ----
    probe_mask = _build_probe_mask(t_probes_jittered)

    # ---- Target trajectory ----
    # y_t = [cos(φ+ωt), sin(φ+ωt)], but only defined during delay
    phase_t = phi + omega * t_arr
    target = np.stack([np.cos(phase_t), np.sin(phase_t)], axis=-1).astype(np.float32)

    return {
        "input_spikes": input_spikes,
        "probe_mask": probe_mask,
        "target": target,
        "phi": phi,
        "omega": omega,
    }


def generate_dataset(
    n_samples: int,
    omegas: list = OMEGAS,
    t_probes: list = T_PROBES,
    jitter_probes: bool = True,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a dataset of synthetic phase trajectory samples.

    Args:
        n_samples: number of samples
        omegas: list of possible angular frequencies
        t_probes: probe times (ms)
        jitter_probes: whether to jitter probe positions
        seed: random seed

    Returns:
        inputs:  [N, T_TOTAL, N_INPUT_CH] float32 tensor
        targets: [N, T_TOTAL, 2] float32 tensor
        masks:   [N, T_TOTAL] float32 tensor (probe perturbation mask)
    """
    rng = np.random.RandomState(seed)
    t_probes_arr = np.array(t_probes, dtype=np.float32)

    all_inputs = []
    all_targets = []
    all_masks = []

    for i in range(n_samples):
        # Random phase and frequency
        phi = rng.uniform(0.0, 2.0 * np.pi)
        omega = omegas[rng.randint(len(omegas))]

        sample = generate_sample(
            phi=phi,
            omega=omega,
            t_probes=t_probes_arr,
            rng=rng,
            jitter_probes=jitter_probes,
        )

        all_inputs.append(sample["input_spikes"])
        all_targets.append(sample["target"])
        all_masks.append(sample["probe_mask"])

    inputs = torch.from_numpy(np.stack(all_inputs, axis=0))
    targets = torch.from_numpy(np.stack(all_targets, axis=0))
    masks = torch.from_numpy(np.stack(all_masks, axis=0))

    return inputs, targets, masks


def create_dataloaders(
    n_train: int = 10000,
    n_val: int = 1000,
    batch_size: int = 64,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Returns:
        train_loader, val_loader
    """
    print(f"Generating {n_train} training samples...")
    train_inputs, train_targets, train_masks = generate_dataset(
        n_samples=n_train, jitter_probes=True, seed=seed,
    )

    print(f"Generating {n_val} validation samples...")
    val_inputs, val_targets, val_masks = generate_dataset(
        n_samples=n_val, jitter_probes=False, seed=seed + 10000,
    )

    train_dataset = TensorDataset(train_inputs, train_targets, train_masks)
    val_dataset = TensorDataset(val_inputs, val_targets, val_masks)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    print(f"  Train: {n_train} samples, {len(train_loader)} batches")
    print(f"  Val:   {n_val} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def generate_visualization_samples(
    phases: list,
    omega: float,
    t_probes: list = T_PROBES,
    seed: int = SEED,
) -> list:
    """Generate fixed samples for visualization (no probe jitter).

    Args:
        phases: list of initial phases (rad)
        omega: fixed angular frequency
        t_probes: fixed probe times
        seed: random seed

    Returns:
        list of dicts, one per phase
    """
    rng = np.random.RandomState(seed)
    t_probes_arr = np.array(t_probes, dtype=np.float32)
    samples = []

    for phi in phases:
        sample = generate_sample(
            phi=phi,
            omega=omega,
            t_probes=t_probes_arr,
            rng=rng,
            jitter_probes=False,
        )
        samples.append(sample)
        print(f"  Generated vis sample: φ={phi:.2f} ({phi*180/np.pi:.0f}°)")

    return samples


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing data generation...")
    train_loader, val_loader = create_dataloaders(n_train=100, n_val=20, batch_size=8)

    inputs, targets, masks = next(iter(train_loader))
    print(f"  Input shape:   {inputs.shape}")   # [8, 900, 32]
    print(f"  Target shape:  {targets.shape}")  # [8, 900, 2]
    print(f"  Mask shape:    {masks.shape}")    # [8, 900]

    # Check spike statistics
    total_spikes = inputs.sum()
    n_elements = inputs.numel()
    print(f"  Total spikes:  {total_spikes:.0f} / {n_elements} "
          f"({100 * total_spikes / n_elements:.2f}%)")

    # Check probe presence
    for tp in T_PROBES:
        t_idx = tp + 5  # middle of probe window
        avg_spikes = inputs[:, t_idx, 20:30].sum(dim=1).mean().item()
        print(f"  Probe @ t={tp}ms: avg {avg_spikes:.1f} spikes/sample "
              f"(expected ~{R_PROBE * T_PROBE_DURATION / 1000 * N_PROBE_CH:.1f})")

    print("Done.")
