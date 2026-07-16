from __future__ import annotations

import torch


SPRiF_MODES = ("clean", "forced_no_reset", "fast_reset", "slow_reset", "both_reset")


def force_threshold_crossing(
    membrane: torch.Tensor,
    threshold: torch.Tensor | float,
    mask: torch.Tensor,
    margin: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold_tensor = torch.as_tensor(
        threshold, device=membrane.device, dtype=membrane.dtype
    )
    target = threshold_tensor + margin
    forced_values = torch.maximum(membrane, target)
    forced = torch.where(mask, forced_values, membrane)
    forced_hit = mask & (forced >= target - 1e-7)
    new_crossing = mask & (membrane < threshold_tensor) & forced_hit
    return forced, forced_hit, new_crossing


def matched_slow_reset(
    slow_to_fast: torch.Tensor,
    fast_delta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if fast_delta.dim() == 2:
        projected = torch.einsum("hpk,hp->hk", slow_to_fast, fast_delta)
    elif fast_delta.dim() == 3:
        projected = torch.einsum("hpk,bhp->bhk", slow_to_fast, fast_delta)
    else:
        raise ValueError("fast_delta must have shape [H,2] or [B,H,2]")
    target_norm = torch.linalg.vector_norm(fast_delta, dim=-1, keepdim=True)
    projected_norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    normalized = projected / projected_norm.clamp_min(eps)
    fallback = torch.zeros_like(projected)
    fallback[..., 0] = 1.0
    direction = torch.where(projected_norm > eps, normalized, fallback)
    return direction * target_norm


def apply_sprif_reset(
    mode: str,
    slow: torch.Tensor,
    fast_pre: torch.Tensor,
    spike: torch.Tensor,
    reset_direction: torch.Tensor,
    slow_to_fast: torch.Tensor,
    threshold: torch.Tensor | float,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in SPRiF_MODES:
        raise ValueError(f"unknown SPRiF intervention mode: {mode}")
    scale = torch.as_tensor(threshold, device=fast_pre.device, dtype=fast_pre.dtype)
    fast_delta = spike.unsqueeze(-1) * reset_direction.unsqueeze(0) * scale * gamma
    slow_delta = matched_slow_reset(slow_to_fast, fast_delta)

    slow_next = slow
    fast_next = fast_pre
    if mode in ("fast_reset", "both_reset"):
        fast_next = fast_next - fast_delta
    if mode in ("slow_reset", "both_reset"):
        slow_next = slow_next - slow_delta
    return slow_next, fast_next
