from __future__ import annotations

import math

import torch


def circular_phase_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_phase = torch.atan2(prediction[..., 1], prediction[..., 0])
    target_phase = torch.atan2(target[..., 1], target[..., 0])
    delta = predicted_phase - target_phase
    return torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))


def delay_mse(prediction: torch.Tensor, target: torch.Tensor, cue_steps: int) -> torch.Tensor:
    return (prediction[:, cue_steps:] - target[:, cue_steps:]).pow(2).mean(dim=(1, 2))


def output_radius(prediction: torch.Tensor, cue_steps: int) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction[:, cue_steps:], dim=-1).mean(dim=1)


def event_metrics(
    phase_error: torch.Tensor,
    reference_error: torch.Tensor,
    event_step: int,
    auc_window: int = 50,
    recovery_window: int = 100,
    sustain_steps: int = 10,
) -> dict[str, torch.Tensor]:
    if phase_error.shape != reference_error.shape:
        raise ValueError("phase_error and reference_error must have identical shapes")
    pre_start = max(0, event_step - 5)
    pre = phase_error[:, pre_start:event_step].mean(dim=1)
    jump_slice = phase_error[:, event_step + 1 : event_step + 6]
    phase_jump = jump_slice.mean(dim=1) - pre

    excess = phase_error - reference_error
    auc_values = excess[:, event_step + 1 : event_step + 1 + auc_window]
    excess_auc = auc_values.sum(dim=1)

    recovery_values = excess[:, event_step + 1 : event_step + 1 + recovery_window].abs()
    peak = recovery_values.max(dim=1).values
    threshold = 0.1 * peak
    recovery_time = torch.full_like(peak, float(recovery_window))
    censored = torch.ones_like(peak, dtype=torch.bool)
    for batch_index in range(recovery_values.shape[0]):
        if peak[batch_index] <= 1e-12:
            recovery_time[batch_index] = 0.0
            censored[batch_index] = False
            continue
        values = recovery_values[batch_index]
        peak_index = int(torch.argmax(values).item())
        for start in range(peak_index + 1, max(0, values.numel() - sustain_steps + 1)):
            if torch.all(values[start : start + sustain_steps] <= threshold[batch_index]):
                recovery_time[batch_index] = float(start + 1)
                censored[batch_index] = False
                break
    return {
        "phase_jump": phase_jump,
        "excess_auc": excess_auc,
        "recovery_time": recovery_time,
        "recovery_censored": censored,
    }
