from dataclasses import dataclass

import torch

from .config import ExperimentConfig


@dataclass
class DMSBatch:
    x: torch.Tensor
    y: torch.Tensor
    intervention: torch.Tensor
    first_side: torch.Tensor
    second_side: torch.Tensor


def make_batch(cfg: ExperimentConfig, batch_size: int, delay_ms: int,
               intervention_count: int, hidden_size: int, seed: int | None = None,
               device: str | torch.device = "cpu",
               fraction: float | None = None) -> DMSBatch:
    task = cfg.task
    if delay_ms < task.dt_ms or delay_ms % task.dt_ms:
        raise ValueError("delay_ms must be a positive multiple of dt_ms")
    if intervention_count < 0 or intervention_count > delay_ms // task.dt_ms:
        raise ValueError("invalid intervention_count")
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    first = torch.randint(0, 2, (batch_size,), generator=g)
    is_match = torch.randint(0, 2, (batch_size,), generator=g)
    second = torch.where(is_match.bool(), first, 1 - first)
    pre = task.pre_ms // task.dt_ms
    cue = task.cue_ms // task.dt_ms
    delay = delay_ms // task.dt_ms
    total = pre + cue + delay + cue
    noise_p = task.noise_rate_hz * task.dt_ms / 1000.0
    x = (torch.rand((batch_size, total, task.input_size), generator=g) < noise_p).float()
    cue_p = task.cue_rate_hz * task.dt_ms / 1000.0
    for side, start in ((first, pre), (second, pre + cue + delay)):
        for b in range(batch_size):
            lo = int(side[b]) * task.cue_channels
            hi = lo + task.cue_channels
            x[b, start:start + cue, lo:hi] = (
                torch.rand((cue, task.cue_channels), generator=g) < cue_p).float()
    intervention = torch.zeros((batch_size, total, hidden_size), dtype=torch.bool)
    frac = task.intervention_fraction if fraction is None else fraction
    n_neurons = max(1, round(hidden_size * frac))
    for b in range(batch_size):
        if intervention_count:
            times = torch.randperm(delay, generator=g)[:intervention_count] + pre + cue
            for t in times:
                neurons = torch.randperm(hidden_size, generator=g)[:n_neurons]
                intervention[b, t, neurons] = True
    return DMSBatch(x.to(device), is_match.long().to(device), intervention.to(device),
                    first.to(device), second.to(device))
