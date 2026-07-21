from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from .surrogates import spike_fn

State = Dict[str, torch.Tensor]

def _force(mem: torch.Tensor, threshold: torch.Tensor | float, mask: torch.Tensor,
           margin: float = 0.05):
    target = torch.as_tensor(threshold, device=mem.device, dtype=mem.dtype) + margin
    forced = torch.where(mask, target.expand_as(mem), mem)
    return forced, mask & (forced >= target - 1e-7)

class BaseCell(nn.Module):
    def __init__(self, input_size, hidden_size, dt_ms, threshold, recurrent=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt_ms / 1000.0
        self.threshold = threshold
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = recurrent
        if recurrent:
            self.recurrent_weight = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.orthogonal_(self.recurrent_weight.weight, gain=0.5)

    def drive(self, x, prev_spike):
        d = self.input(x)
        if self.recurrent and prev_spike is not None:
            d = d + self.recurrent_weight(prev_spike)
        return d

    def zeros(self, batch, device, dims=()):
        return torch.zeros((batch, self.hidden_size, *dims), device=device)

class SPRiFFullCell(BaseCell):
    def __init__(self, *args, learned_reset=True, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden_size
        self.alpha_raw = nn.Parameter(torch.full((h,), 3.0))
        self.rho_raw = nn.Parameter(torch.full((h,), 2.0))
        self.omega_raw = nn.Parameter(torch.zeros(h))
        self.eta_raw = nn.Parameter(torch.ones(h, 2))
        self.fast_coupling = nn.Parameter(torch.zeros(h))
        self.G = nn.Parameter(torch.empty(h, 2, 3))
        nn.init.normal_(self.G, std=0.05)
        if learned_reset:
            self.reset_lambda = nn.Parameter(torch.full((h,), 0.5))

    def initial_state(self, batch, device):
        return {"slow": self.zeros(batch, device, (3,)), "fast": self.zeros(batch, device, (2,))}

    def step(self, x, state, intervention, prev_spike=None):
        drive = self.drive(x, prev_spike)
        alpha, rho = torch.sigmoid(self.alpha_raw), torch.sigmoid(self.rho_raw)
        omega = math.pi * torch.sigmoid(self.omega_raw)
        x0, x1, x2 = state["slow"].unbind(-1)
        slow = torch.stack((alpha * x0 + (1 - alpha) * drive,
                            rho * (torch.cos(omega) * x1 - torch.sin(omega) * x2) + (1 - rho) * drive,
                            rho * (torch.sin(omega) * x1 + torch.cos(omega) * x2)), -1)
        eta = torch.sigmoid(self.eta_raw)
        u0, u1 = state["fast"].unbind(-1)
        leak = torch.stack((eta[:, 0] * u0 + self.fast_coupling * u1,
                            eta[:, 1] * u1), -1)
        fast = leak + torch.einsum("bhk,hpk->bhp", slow, self.G)
        fast = fast.clone()
        fast0, hit = _force(fast[..., 0], self.threshold, intervention)
        fast[..., 0] = fast0
        z = spike_fn(fast[..., 0] - self.threshold)
        pre = fast
        lam = self.reset_lambda if hasattr(self, "reset_lambda") else torch.zeros(
            self.hidden_size, device=x.device)
        reset_vec = torch.stack((torch.ones_like(lam), lam), -1)
        fast = pre - z.unsqueeze(-1) * reset_vec * self.threshold
        return z, {"slow": slow, "fast": fast}, {
            "forced_hit": hit, "slow_pre_reset": slow, "fast_pre_reset": pre}

class SPRiFMergedCell(BaseCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden_size
        self.alpha_raw = nn.Parameter(torch.full((h,), 3.0))
        self.rho_raw = nn.Parameter(torch.full((h,), 2.0))
        self.omega_raw = nn.Parameter(torch.zeros(h))

    def initial_state(self, batch, device):
        return {"merged": self.zeros(batch, device, (3,))}

    def step(self, x, state, intervention, prev_spike=None):
        drive = self.drive(x, prev_spike)
        alpha, rho = torch.sigmoid(self.alpha_raw), torch.sigmoid(self.rho_raw)
        omega = math.pi * torch.sigmoid(self.omega_raw)
        x0, x1, x2 = state["merged"].unbind(-1)
        merged = torch.stack((alpha * x0 + (1 - alpha) * drive,
                              rho * (torch.cos(omega) * x1 - torch.sin(omega) * x2) + (1 - rho) * drive,
                              rho * (torch.sin(omega) * x1 + torch.cos(omega) * x2)), -1)
        merged = merged.clone()
        merged[..., 0], hit = _force(merged[..., 0], self.threshold, intervention)
        z = spike_fn(merged[..., 0] - self.threshold)
        pre = merged
        reset = torch.tensor([1.0, 0.0, 0.0], device=x.device)
        merged = pre - z.unsqueeze(-1) * reset * self.threshold
        return z, {"merged": merged}, {"forced_hit": hit, "merged_pre_reset": pre}

class LIFCell(BaseCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_tau = nn.Parameter(torch.full((self.hidden_size,), math.log(500e-3)))

    def initial_state(self, batch, device): return {"u": self.zeros(batch, device)}

    def step(self, x, state, intervention, prev_spike=None):
        alpha = torch.exp(-self.dt / self.log_tau.exp().clamp_min(self.dt))
        u = alpha * state["u"] + (1 - alpha) * self.drive(x, prev_spike)
        u, hit = _force(u, self.threshold, intervention)
        z = spike_fn(u - self.threshold)
        pre = u
        return z, {"u": pre - z * self.threshold}, {"forced_hit": hit, "mem_pre_reset": pre}

class ASRNNCell(BaseCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_tau_m = nn.Parameter(torch.full((self.hidden_size,), math.log(200e-3)))
        self.log_tau_adp = nn.Parameter(torch.full((self.hidden_size,), math.log(2000e-3)))
        self.beta = nn.Parameter(torch.full((self.hidden_size,), 1.8))

    def initial_state(self, batch, device):
        return {"mem": self.zeros(batch, device), "b": self.zeros(batch, device),
                "spike": self.zeros(batch, device)}

    def step(self, x, state, intervention, prev_spike=None):
        alpha = torch.exp(-self.dt / self.log_tau_m.exp().clamp_min(self.dt))
        rho = torch.exp(-self.dt / self.log_tau_adp.exp().clamp_min(self.dt))
        b = rho * state["b"] + (1 - rho) * state["spike"]
        threshold = self.threshold + self.beta * b
        mem = alpha * state["mem"] + (1 - alpha) * self.drive(x, prev_spike) - threshold * state["spike"] * self.dt
        mem, hit = _force(mem, threshold, intervention)
        z = spike_fn(mem - threshold)
        return z, {"mem": mem, "b": b, "spike": z}, {"forced_hit": hit, "mem_pre_reset": mem}

class BRFCell(BaseCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.omega = nn.Parameter(torch.full((self.hidden_size,), 10.0))
        self.b_offset = nn.Parameter(torch.empty(self.hidden_size).uniform_(0.5, 3.0))

    def initial_state(self, batch, device):
        return {"u": self.zeros(batch, device), "v": self.zeros(batch, device),
                "q": self.zeros(batch, device)}

    def step(self, x, state, intervention, prev_spike=None):
        omega = self.omega.abs().clamp(max=0.99 / self.dt)
        p = (-1 + torch.sqrt(1 - (self.dt * omega) ** 2)) / self.dt
        b = p - self.b_offset.abs() - state["q"]
        u = state["u"] + b * state["u"] * self.dt - omega * state["v"] * self.dt + self.drive(x, prev_spike) * self.dt
        v = state["v"] + omega * state["u"] * self.dt + b * state["v"] * self.dt
        u, hit = _force(u, self.threshold + state["q"], intervention)
        z = spike_fn(u - self.threshold - state["q"])
        q = 0.9 * state["q"] + z
        return z, {"u": u, "v": v, "q": q}, {"forced_hit": hit, "mem_pre_reset": u}

def build_cell(name: str, input_size: int, hidden_size: int, dt_ms: int, threshold: float,
               recurrent: bool = False):
    args = (input_size, hidden_size, dt_ms, threshold)
    kw = {"recurrent": recurrent}
    table = {"sprif_full": lambda: SPRiFFullCell(*args, learned_reset=True, **kw),
             "sprif_lambda0": lambda: SPRiFFullCell(*args, learned_reset=False, **kw),
             "sprif_merged": lambda: SPRiFMergedCell(*args, **kw), "lif": lambda: LIFCell(*args, **kw),
             "asrnn": lambda: ASRNNCell(*args, **kw), "brf": lambda: BRFCell(*args, **kw)}
    if name not in table:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(table)}")
    return table[name]()

