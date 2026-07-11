from dataclasses import dataclass

import torch
from torch import nn

from .cells import build_cell
from .config import ExperimentConfig


@dataclass
class NetworkOutput:
    logits: torch.Tensor
    natural_rate_tensor: torch.Tensor
    total_rate_tensor: torch.Tensor
    forced_hit_rate_tensor: torch.Tensor

    @property
    def natural_rate(self): return float(self.natural_rate_tensor.detach())
    @property
    def total_rate(self): return float(self.total_rate_tensor.detach())
    @property
    def forced_hit_rate(self): return float(self.forced_hit_rate_tensor.detach())


class SIDMSNetwork(nn.Module):
    def __init__(self, name: str, cfg: ExperimentConfig):
        super().__init__()
        self.name, self.cfg = name, cfg
        h = cfg.model.hidden_size
        cell_input = cfg.task.input_size + (h if cfg.model.recurrent else 0)
        self.cell = build_cell(name, cell_input, h, cfg.task.dt_ms, cfg.model.threshold)
        self.readout = nn.Linear(h, 2)
        self.output_alpha = torch.exp(torch.tensor(
            -cfg.task.dt_ms / cfg.model.output_tau_ms))

    def forward(self, x: torch.Tensor, intervention: torch.Tensor) -> NetworkOutput:
        batch, steps, _ = x.shape
        state = self.cell.initial_state(batch, x.device)
        prev = torch.zeros(batch, self.cfg.model.hidden_size, device=x.device)
        out_mem = torch.zeros(batch, 2, device=x.device)
        spikes, hits = [], []
        for t in range(steps):
            cell_x = torch.cat((x[:, t], prev), -1) if self.cfg.model.recurrent else x[:, t]
            prev, state, diag = self.cell.step(cell_x, state, intervention[:, t])
            out_mem = self.output_alpha.to(x.device) * out_mem + (1 - self.output_alpha.to(x.device)) * self.readout(prev)
            spikes.append(prev)
            hits.append(diag["forced_hit"])
        spike_tensor = torch.stack(spikes, 1)
        hit_tensor = torch.stack(hits, 1)
        natural = (~intervention).to(x.dtype)
        natural_rate = (spike_tensor * natural).sum() / natural.sum().clamp_min(1)
        selected = intervention.sum()
        hit_rate = hit_tensor.sum().float() / selected.clamp_min(1)
        return NetworkOutput(out_mem, natural_rate, spike_tensor.mean(), hit_rate)
