# SI-DMS Neuron Cell Implementations
# Paper Sec. 4.4, Appendix S2.2-S2.4.
# 6 models for Spike-Intervention Delayed Match-to-Sample task.
# All cells support forced-spike interventions via _force():
#   selected membrane coordinates placed above threshold before spike computation.
# Forced spikes follow the model's ordinary reset rule (Appendix S2.4).

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from .surrogates import spike_fn

State = Dict[str, torch.Tensor]

# Place selected membrane coordinates above threshold; spike follows model's reset rule
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

class SPRiFScalarCell(BaseCell):
    """TNNLS model: x in R3, delayed scalar filter h, and scalar-reset membrane."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden_size
        self.alpha_raw = nn.Parameter(torch.full((h,), 3.0))
        self.rho_raw = nn.Parameter(torch.full((h,), 2.0))
        self.omega_raw = nn.Parameter(torch.zeros(h))
        self.eta_raw = nn.Parameter(torch.ones(h, 2))
        self.bar_G = nn.Parameter(torch.empty(h, 2, 3))
        nn.init.normal_(self.bar_G, std=0.05)
        with torch.no_grad():
            # The former SI-DMS coupling initialized at zero; absorb that scale.
            self.bar_G[:, 1].zero_()

    def initial_state(self, batch, device):
        return {"x": self.bar_G.new_zeros(batch, self.hidden_size, 3, device=device),
                "h": self.bar_G.new_zeros(batch, self.hidden_size, device=device),
                "u": self.bar_G.new_zeros(batch, self.hidden_size, device=device)}

    def step(self, x, state, intervention, prev_spike=None):
        drive = self.drive(x, prev_spike)
        alpha, rho = torch.sigmoid(self.alpha_raw), torch.sigmoid(self.rho_raw)
        omega = math.pi * torch.sigmoid(self.omega_raw)
        x0, x1, x2 = state["x"].unbind(-1)
        spectral = torch.stack((alpha * x0 + (1 - alpha) * drive,
                                rho * (torch.cos(omega) * x1 - torch.sin(omega) * x2)
                                + (1 - rho) * drive,
                                rho * (torch.sin(omega) * x1 + torch.cos(omega) * x2)), -1)
        eta = torch.sigmoid(self.eta_raw)
        projection = torch.einsum("bhk,hpk->bhp", spectral, self.bar_G)
        h = eta[:, 1] * state["h"] + projection[..., 1]
        membrane = eta[:, 0] * state["u"] + state["h"] + projection[..., 0]
        membrane, hit = _force(membrane, self.threshold, intervention)
        z = spike_fn(membrane - self.threshold)
        return z, {"x": spectral, "h": h, "u": membrane - z * self.threshold}, {
            "forced_hit": hit, "slow_pre_reset": spectral,
            "h_pre_reset": h, "mem_pre_reset": membrane}


class SPRiFFullCell(BaseCell):
    # Full SPRiF: slow spectral state + fast discharge + projective reset [1,lambda_j]
    # Paper Sec. 3.2, Eq. (2)-(5); Appendix S2.2
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
        # Slow spectral state (Eq.3) -> fast discharge (Eq.4) -> projective reset [1,lambda_j] (Eq.5)
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
    # Merged ablation: single 3D state, membrane=x[0], scalar reset on coord 0
    # Paper Table 2, Appendix S2.1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden_size
        self.alpha_raw = nn.Parameter(torch.full((h,), 3.0))
        self.rho_raw = nn.Parameter(torch.full((h,), 2.0))
        self.omega_raw = nn.Parameter(torch.zeros(h))

    def initial_state(self, batch, device):
        return {"merged": self.zeros(batch, device, (3,))}

    def step(self, x, state, intervention, prev_spike=None):
        # Merged: single-state spectral dynamics; membrane=x[0]; scalar reset on coord 0
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
    # Standard LIF baseline: v = alpha*v + (1-alpha)*I; reset: v -= z*theta
    # Paper Eq. (1), Sec. 3.1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_tau = nn.Parameter(torch.full((self.hidden_size,), math.log(500e-3)))

    def initial_state(self, batch, device): return {"u": self.zeros(batch, device)}

    def step(self, x, state, intervention, prev_spike=None):
        # LIF update: leaky membrane + scalar reset; forced spikes via _force()
        alpha = torch.exp(-self.dt / self.log_tau.exp().clamp_min(self.dt))
        u = alpha * state["u"] + (1 - alpha) * self.drive(x, prev_spike)
        u, hit = _force(u, self.threshold, intervention)
        z = spike_fn(u - self.threshold)
        pre = u
        return z, {"u": pre - z * self.threshold}, {"forced_hit": hit, "mem_pre_reset": pre}

class ASRNNCell(BaseCell):
    # Adaptive Spike RNN (Bozkurt et al., Nat. Mach. Intell. 2021):
    # adaptive threshold b_t = rho*b_{t-1} + (1-rho)*z_{t-1}; theta_eff = theta + beta*b
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_tau_m = nn.Parameter(torch.full((self.hidden_size,), math.log(200e-3)))
        self.log_tau_adp = nn.Parameter(torch.full((self.hidden_size,), math.log(2000e-3)))
        self.beta = nn.Parameter(torch.full((self.hidden_size,), 1.8))

    def initial_state(self, batch, device):
        return {"mem": self.zeros(batch, device), "b": self.zeros(batch, device),
                "spike": self.zeros(batch, device)}

    def step(self, x, state, intervention, prev_spike=None):
        # ASRNN: adaptive threshold b_t; spike triggers reset AND adaptation recurrence
        alpha = torch.exp(-self.dt / self.log_tau_m.exp().clamp_min(self.dt))
        rho = torch.exp(-self.dt / self.log_tau_adp.exp().clamp_min(self.dt))
        b = rho * state["b"] + (1 - rho) * state["spike"]
        threshold = self.threshold + self.beta * b
        mem = alpha * state["mem"] + (1 - alpha) * self.drive(x, prev_spike) - threshold * state["spike"] * self.dt
        mem, hit = _force(mem, threshold, intervention)
        z = spike_fn(mem - threshold)
        return z, {"mem": mem, "b": b, "spike": z}, {"forced_hit": hit, "mem_pre_reset": mem}

class BRFCell(BaseCell):
    # Bimodal Resonate-and-Fire (Niebert et al., ICML 2024):
    # 2D resonant dynamics (u,v) with adaptive threshold q_t
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
    # Model registry: maps string names to cell constructors
    args = (input_size, hidden_size, dt_ms, threshold)
    kw = {"recurrent": recurrent}
    table = {"sprif_scalar": lambda: SPRiFScalarCell(*args, **kw),
             "sprif_full": lambda: SPRiFFullCell(*args, learned_reset=True, **kw),
             "sprif_lambda0": lambda: SPRiFFullCell(*args, learned_reset=False, **kw),
             "sprif_merged": lambda: SPRiFMergedCell(*args, **kw), "lif": lambda: LIFCell(*args, **kw),
             "asrnn": lambda: ASRNNCell(*args, **kw), "brf": lambda: BRFCell(*args, **kw)}
    if name not in table:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(table)}")
    return table[name]()

