"""
SPRiF & LIF 状态轨迹记录
==============================

训练完成后，对固定可视化样本做完整 forward pass，逐时间步记录所有内部状态。

记录内容:
    SPRiF: slow state x_t (3D), fast pre-state ũ_t (2D), fast post-state u_t (2D),
           membrane v_t, spikes z_t, readout ŷ_t
    LIF:   membrane potential v_t, spikes z_t, readout ŷ_t
    共用:  input spikes s_t, probe mask p_t, target y_t

Usage:
    from record_forward import record_all
    records = record_all(sprif_model, lif_model, vis_samples, device)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from config import (
    T_TOTAL, T_CUE, T_DELAY,
    N_INPUT_CH, HIDDEN_SIZE, A_PROBE,
    VIS_PHASES, VIS_OMEGA,
)
from models import SPRiFTrajectoryModel, LIFTrajectoryModel


# ============================================================================
# SPRiF recording forward pass
# ============================================================================

def record_sprif_forward(
    model: SPRiFTrajectoryModel,
    input_spikes: torch.Tensor,
    probe_mask: torch.Tensor,
    A_probe: float = A_PROBE,
    target: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Record all internal states for a SPRiF forward pass.

    Args:
        model: trained SPRiFTrajectoryModel
        input_spikes: [1, T, N_INPUT_CH] single sample
        probe_mask: [1, T] perturbation mask
        A_probe: perturbation current amplitude
        target: optional [1, T, 2] target trajectory

    Returns:
        dict with keys:
            slow_states:    [T, H, 3]  — slow state x_t
            fast_pre:       [T, H, 2]  — fast state before reset ũ_t
            fast_post:      [T, H, 2]  — fast state after reset u_t
            membranes:      [T, H]     — membrane potential v_t = ũ_t⁰
            spikes:         [T, H]     — output spikes z_t
            readouts:       [T, 2]     — predicted trajectory ŷ_t
            target:         [T, 2]     — target trajectory (if provided)
            input_spikes:   [T, 32]    — input spike raster
            probe_mask:     [T]        — probe perturbation mask
            spectral_params: dict      — learned α, ρ, ω, η, λ
    """
    model.eval()
    device = next(model.parameters()).device

    input_spikes = input_spikes.to(device)
    probe_mask = probe_mask.to(device)
    if target is not None:
        target = target.to(device)

    batch_size, T, _ = input_spikes.shape
    assert batch_size == 1, "record_sprif_forward expects single sample (batch=1)"

    layer = model.sprif_layer
    state = layer.init_state(1, device=device, dtype=input_spikes.dtype)
    runtime = layer._precompute_runtime_params()

    T_out = T
    H = model.hidden_size

    # Pre-allocate arrays
    slow_states = np.zeros((T_out, H, 3), dtype=np.float32)
    fast_pre = np.zeros((T_out, H, 2), dtype=np.float32)
    fast_post = np.zeros((T_out, H, 2), dtype=np.float32)
    membranes = np.zeros((T_out, H), dtype=np.float32)
    spikes = np.zeros((T_out, H), dtype=np.float32)
    readouts = np.zeros((T_out, 2), dtype=np.float32)
    input_spikes_rec = np.zeros((T_out, N_INPUT_CH), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            input_spikes_rec[t] = input_spikes[0, t].cpu().numpy()

            # Input current with perturbation
            input_current = layer.input_linear(input_spikes[:, t, :])
            input_current = input_current + A_probe * probe_mask[:, t].unsqueeze(-1)

            # Manual step-by-step to capture pre-reset state
            x_state = state["x"]
            u_state = state["u"]

            x_next = layer._slow_flow(x_state, input_current, runtime)
            u_tilde = layer._fast_flow(u_state, x_next, runtime)

            membrane = u_tilde[..., 0]
            spike = layer._spike_fn(membrane - layer.threshold)

            # Projective reset
            u_next = (
                u_tilde
                - spike.unsqueeze(-1)
                * runtime["reset_direction"].unsqueeze(0)
                * layer.threshold
            )

            state = {"x": x_next, "u": u_next, "prev_spike": spike}

            # Readout
            out = model.readout(x_next.reshape(1, -1))

            # Store
            slow_states[t] = x_next.squeeze(0).cpu().numpy()
            fast_pre[t] = u_tilde.squeeze(0).cpu().numpy()
            fast_post[t] = u_next.squeeze(0).cpu().numpy()
            membranes[t] = membrane.squeeze(0).cpu().numpy()
            spikes[t] = spike.squeeze(0).cpu().numpy()
            readouts[t] = out.squeeze(0).cpu().numpy()

    # Gather spectral parameters
    spectral_params = {}
    with torch.no_grad():
        for k, v in layer.get_spectral_parameters().items():
            spectral_params[k] = v.detach().cpu().numpy()

    result = {
        "slow_states": slow_states,
        "fast_pre": fast_pre,
        "fast_post": fast_post,
        "membranes": membranes,
        "spikes": spikes,
        "readouts": readouts,
        "input_spikes": input_spikes_rec,
        "probe_mask": probe_mask.squeeze(0).cpu().numpy(),
        "spectral_params": spectral_params,
    }
    if target is not None:
        result["target"] = target.squeeze(0).cpu().numpy()
    return result


# ============================================================================
# LIF recording forward pass
# ============================================================================

def record_lif_forward(
    model: LIFTrajectoryModel,
    input_spikes: torch.Tensor,
    probe_mask: torch.Tensor,
    A_probe: float = A_PROBE,
    target: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Record all internal states for a LIF forward pass.

    Args:
        model: trained LIFTrajectoryModel
        input_spikes: [1, T, N_INPUT_CH] single sample
        probe_mask: [1, T] perturbation mask
        A_probe: perturbation current amplitude

    Returns:
        dict with keys:
            membranes:      [T, H]     — membrane potential v_t (pre-reset)
            spikes:         [T, H]     — output spikes z_t
            readouts:       [T, 2]     — predicted trajectory ŷ_t
            input_spikes:   [T, 32]    — input spike raster
            probe_mask:     [T]        — probe perturbation mask
    """
    model.eval()
    device = next(model.parameters()).device

    input_spikes = input_spikes.to(device)
    probe_mask = probe_mask.to(device)
    if target is not None:
        target = target.to(device)

    batch_size, T, _ = input_spikes.shape
    assert batch_size == 1, "record_lif_forward expects single sample (batch=1)"

    layer = model.lif_layer
    state = layer.init_state(1, device=device, dtype=input_spikes.dtype)

    T_out = T
    H = model.hidden_size

    membranes = np.zeros((T_out, H), dtype=np.float32)
    spikes = np.zeros((T_out, H), dtype=np.float32)
    readouts = np.zeros((T_out, 2), dtype=np.float32)
    input_spikes_rec = np.zeros((T_out, N_INPUT_CH), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            input_spikes_rec[t] = input_spikes[0, t].cpu().numpy()

            input_current = layer.input_linear(input_spikes[:, t, :])
            input_current = input_current + A_probe * probe_mask[:, t].unsqueeze(-1)

            spike, v_pre, state = layer.forward_step(
                input_spikes[:, t, :], state, input_current=input_current,
            )

            out = model.readout(v_pre)

            membranes[t] = v_pre.squeeze(0).cpu().numpy()
            spikes[t] = spike.squeeze(0).cpu().numpy()
            readouts[t] = out.squeeze(0).cpu().numpy()

    result = {
        "membranes": membranes,
        "spikes": spikes,
        "readouts": readouts,
        "input_spikes": input_spikes_rec,
        "probe_mask": probe_mask.squeeze(0).cpu().numpy(),
    }
    if target is not None:
        result["target"] = target.squeeze(0).cpu().numpy()
    return result


# ============================================================================
# Combined recording
# ============================================================================

def record_all(
    sprif_model: SPRiFTrajectoryModel,
    lif_model: LIFTrajectoryModel,
    vis_samples: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Record state trajectories for all visualization samples.

    Args:
        sprif_model: trained SPRiF model
        lif_model: trained LIF model
        vis_samples: list of sample dicts from generate_visualization_samples()
        device: torch device

    Returns:
        {"sprif": [record_per_sample], "lif": [record_per_sample]}
    """
    sprif_records = []
    lif_records = []

    for i, sample in enumerate(vis_samples):
        phi = sample["phi"]
        print(f"\n  Recording sample {i+1}/{len(vis_samples)}: φ={phi:.2f}")

        # Convert to tensors
        input_t = torch.from_numpy(sample["input_spikes"]).unsqueeze(0).float()
        probe_t = torch.from_numpy(sample["probe_mask"]).unsqueeze(0).float()
        target_t = torch.from_numpy(sample["target"]).unsqueeze(0).float()

        # SPRiF
        sprif_rec = record_sprif_forward(sprif_model, input_t, probe_t, target=target_t)
        sprif_rec["phi"] = phi
        sprif_records.append(sprif_rec)

        # LIF
        lif_rec = record_lif_forward(lif_model, input_t, probe_t, target=target_t)
        lif_rec["phi"] = phi
        lif_records.append(lif_rec)

        # Print summary
        n_spikes_sprif = int(sprif_rec["spikes"].sum())
        n_spikes_lif = int(lif_rec["spikes"].sum())
        mse_sprif = np.mean((sprif_rec["readouts"][T_CUE:] - sprif_rec["target"][T_CUE:]) ** 2)
        mse_lif = np.mean((lif_rec["readouts"][T_CUE:] - lif_rec["target"][T_CUE:]) ** 2)
        print(f"    SPRiF: {n_spikes_sprif} spikes, delay MSE={mse_sprif:.6f}")
        print(f"    LIF:   {n_spikes_lif} spikes, delay MSE={mse_lif:.6f}")

    return {"sprif": sprif_records, "lif": lif_records}


# ============================================================================
# Utility: select representative neuron
# ============================================================================

def select_representative_neuron(
    sprif_record: Dict[str, np.ndarray],
    probe_times: List[int],
    min_spikes_at_probe: int = 1,
) -> int:
    """Select a neuron that spikes at probe times for visualization.

    Args:
        sprif_record: SPRiF recording for one sample
        probe_times: list of probe times (ms)
        min_spikes_at_probe: minimum number of probe windows where neuron spikes

    Returns:
        neuron_idx: index of selected neuron
    """
    spikes = sprif_record["spikes"]  # [T, H]
    H = spikes.shape[1]

    # Count spikes per neuron within ±5ms of each probe
    spike_counts = np.zeros(H, dtype=int)
    for tp in probe_times:
        start = max(0, tp)
        end = min(spikes.shape[0], tp + 10)
        spike_counts += (spikes[start:end].sum(axis=0) > 0).astype(int)

    # Find neuron with most probe-window spikes
    best_neuron = int(np.argmax(spike_counts))
    n_probes = spike_counts[best_neuron]
    print(f"  Selected neuron #{best_neuron}: spikes in {n_probes}/{len(probe_times)} probe windows")
    return best_neuron


def select_lif_neuron(
    lif_record: Dict[str, np.ndarray],
    probe_times: List[int],
    min_spikes_at_probe: int = 1,
) -> int:
    """Select a LIF neuron that spikes at probe times for visualization."""
    spikes = lif_record["spikes"]  # [T, H]
    H = spikes.shape[1]

    spike_counts = np.zeros(H, dtype=int)
    for tp in probe_times:
        start = max(0, tp)
        end = min(spikes.shape[0], tp + 10)
        spike_counts += (spikes[start:end].sum(axis=0) > 0).astype(int)

    best_neuron = int(np.argmax(spike_counts))
    n_probes = spike_counts[best_neuron]
    print(f"  Selected LIF neuron #{best_neuron}: spikes in {n_probes}/{len(probe_times)} probe windows")
    return best_neuron


# ============================================================================
# Save / Load records
# ============================================================================

def save_records(
    records: Dict,
    save_path: str,
):
    """Save recorded trajectories as compressed numpy archive."""
    np.savez_compressed(save_path, **records)
    print(f"  Saved records to: {save_path}")


def load_records(load_path: str) -> Dict:
    """Load recorded trajectories."""
    return dict(np.load(load_path, allow_pickle=True))
