from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig, TaskConfig
from .interventions import apply_sprif_reset, force_threshold_crossing


CODE_ROOT = Path(__file__).resolve().parents[3]
SPRiF_SOURCE = CODE_ROOT / "Task_pSMNIST"
if str(SPRiF_SOURCE) not in sys.path:
    sys.path.insert(0, str(SPRiF_SOURCE))

from core_algorithm.sprif_layer import SPRiFNeuronLayer, surrogate_spike


class FilteredSpikeReadout(nn.Module):
    def __init__(self, hidden_size: int, output_size: int = 2,
                 tau_ms: float = 20.0, dt_ms: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.register_buffer("alpha", torch.tensor(math.exp(-dt_ms / tau_ms)))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.dim() != 3:
            raise ValueError("spikes must have shape [batch,time,hidden]")
        state = torch.zeros(
            spikes.shape[0], self.linear.out_features,
            device=spikes.device, dtype=spikes.dtype,
        )
        outputs = []
        alpha = self.alpha.to(device=spikes.device, dtype=spikes.dtype)
        for step in range(spikes.shape[1]):
            drive = self.linear(spikes[:, step])
            state = alpha * state + (1.0 - alpha) * drive
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def _empty_mask(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference, dtype=torch.bool)


class SPRiFCellAdapter(nn.Module):
    valid_modes = {"clean", "forced_no_reset", "fast_reset", "slow_reset", "both_reset"}

    def __init__(self, input_size: int, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.threshold = cfg.threshold
        self.layer = SPRiFNeuronLayer(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            threshold=cfg.threshold,
            recurrent=cfg.recurrent,
            tau_alpha_range=(50.0, 1500.0),
            tau_rho_range=(300.0, 3000.0),
            tau_eta_range=(0.8, 10.0),
            omega_range=(2.0 * math.pi / 300.0, 2.0 * math.pi / 40.0),
        )

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        state = self.layer.init_state(batch, device=device, dtype=dtype)
        state["u"].zero_()
        return state

    def step(self, x_t, state, mode, mask, gamma, margin):
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported SPRiF mode: {mode}")
        runtime = self.layer._precompute_runtime_params()
        input_current = self.layer.input_linear(x_t)
        slow_pre = self.layer._slow_flow(state["x"], input_current, runtime)
        fast_pre = self.layer._fast_flow(state["u"], slow_pre, runtime)
        membrane_natural = fast_pre[..., 0]
        natural_spike = surrogate_spike(membrane_natural - self.threshold)

        fast_forced = fast_pre.clone()
        forced_membrane, forced_hit, new_crossing = force_threshold_crossing(
            membrane_natural, self.threshold, mask, margin
        )
        fast_forced[..., 0] = forced_membrane
        spike = surrogate_spike(forced_membrane - self.threshold)
        induced_spike = (spike - natural_spike).clamp(0.0, 1.0)

        reset_direction = runtime["reset_direction"]
        natural_delta = (
            natural_spike.unsqueeze(-1)
            * reset_direction.unsqueeze(0)
            * self.threshold
        )
        fast_after_natural = fast_forced - natural_delta
        slow_next, fast_next = apply_sprif_reset(
            mode,
            slow_pre,
            fast_after_natural,
            induced_spike,
            reset_direction,
            self.layer.G,
            self.threshold,
            gamma,
        )
        next_state = {"x": slow_next, "u": fast_next, "prev_spike": spike}
        return spike, next_state, {
            "membrane_pre": membrane_natural,
            "threshold": torch.full_like(membrane_natural, self.threshold),
            "forced_hit": forced_hit,
            "new_crossing": new_crossing,
            "natural_spike": natural_spike,
            "slow": slow_next,
            "fast": fast_next,
            "slow_pre_reset": slow_pre,
            "fast_pre_reset": fast_forced,
        }


class LIFCell(nn.Module):
    valid_modes = {"clean", "forced_no_reset", "native_reset"}

    def __init__(self, input_size: int, cfg: ModelConfig, dt_ms: float):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.threshold = cfg.threshold
        self.input = nn.Linear(input_size, cfg.hidden_size)
        self.log_tau_ms = nn.Parameter(torch.full((cfg.hidden_size,), math.log(5.0)))
        self.dt_ms = dt_ms

    def initial_state(self, batch, device, dtype):
        return {"u": torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)}

    def step(self, x_t, state, mode, mask, gamma, margin):
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported LIF mode: {mode}")
        alpha = torch.exp(-self.dt_ms / self.log_tau_ms.exp().clamp_min(self.dt_ms))
        membrane = alpha * state["u"] + (1.0 - alpha) * self.input(x_t)
        natural_spike = surrogate_spike(membrane - self.threshold)
        forced, forced_hit, new_crossing = force_threshold_crossing(
            membrane, self.threshold, mask, margin
        )
        spike = surrogate_spike(forced - self.threshold)
        induced = (spike - natural_spike).clamp(0.0, 1.0)
        reset_spike = natural_spike + (induced if mode == "native_reset" else 0.0)
        next_u = forced - reset_spike * self.threshold * gamma
        return spike, {"u": next_u}, {
            "membrane_pre": membrane,
            "threshold": torch.full_like(membrane, self.threshold),
            "forced_hit": forced_hit,
            "new_crossing": new_crossing,
            "natural_spike": natural_spike,
        }


class ASRNNCell(nn.Module):
    valid_modes = {"clean", "forced_no_reset", "native_reset"}

    def __init__(self, input_size: int, cfg: ModelConfig, dt_ms: float):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.base_threshold = cfg.threshold
        self.input = nn.Linear(input_size, cfg.hidden_size)
        nn.init.xavier_uniform_(self.input.weight, gain=3.0)
        self.log_tau_m_ms = nn.Parameter(torch.full((cfg.hidden_size,), math.log(5.0)))
        self.log_tau_adp_ms = nn.Parameter(torch.full((cfg.hidden_size,), math.log(200.0)))
        self.beta = nn.Parameter(torch.full((cfg.hidden_size,), 1.8))
        self.dt_ms = dt_ms

    def initial_state(self, batch, device, dtype):
        zeros = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        return {"mem": zeros.clone(), "adaptation": zeros.clone(), "prev_spike": zeros.clone()}

    def step(self, x_t, state, mode, mask, gamma, margin):
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported ASRNN mode: {mode}")
        alpha = torch.exp(-self.dt_ms / self.log_tau_m_ms.exp().clamp_min(self.dt_ms))
        rho = torch.exp(-self.dt_ms / self.log_tau_adp_ms.exp().clamp_min(self.dt_ms))
        adaptation = rho * state["adaptation"] + (1.0 - rho) * state["prev_spike"]
        threshold = self.base_threshold + torch.nn.functional.softplus(self.beta) * adaptation
        membrane = alpha * state["mem"] + (1.0 - alpha) * self.input(x_t)
        natural_spike = surrogate_spike(membrane - threshold)
        forced, forced_hit, new_crossing = force_threshold_crossing(
            membrane, threshold, mask, margin
        )
        spike = surrogate_spike(forced - threshold)
        induced = (spike - natural_spike).clamp(0.0, 1.0)
        reset_spike = natural_spike + (induced if mode == "native_reset" else 0.0)
        next_mem = forced - reset_spike * threshold * gamma
        return spike, {"mem": next_mem, "adaptation": adaptation, "prev_spike": spike}, {
            "membrane_pre": membrane,
            "threshold": threshold,
            "forced_hit": forced_hit,
            "new_crossing": new_crossing,
            "natural_spike": natural_spike,
        }


class BRFCell(nn.Module):
    valid_modes = {"clean", "forced_no_reset", "native_reset"}

    def __init__(self, input_size: int, cfg: ModelConfig, dt_ms: float):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.threshold = cfg.threshold
        self.input = nn.Linear(input_size, cfg.hidden_size)
        nn.init.xavier_uniform_(self.input.weight, gain=3.0)
        self.omega = nn.Parameter(torch.full((cfg.hidden_size,), 10.0))
        self.b_offset = nn.Parameter(torch.empty(cfg.hidden_size).uniform_(0.5, 3.0))
        self.dt = dt_ms / 1000.0

    def initial_state(self, batch, device, dtype):
        zeros = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        return {"u": zeros.clone(), "v": zeros.clone(), "q": zeros.clone()}

    def step(self, x_t, state, mode, mask, gamma, margin):
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported BRF mode: {mode}")
        omega = self.omega.abs().clamp(max=0.99 / self.dt)
        p = (-1.0 + torch.sqrt(1.0 - (self.dt * omega) ** 2)) / self.dt
        damping = p - self.b_offset.abs() - state["q"]
        u = state["u"] + (
            damping * state["u"] - omega * state["v"] + self.input(x_t)
        ) * self.dt
        v = state["v"] + (
            omega * state["u"] + damping * state["v"]
        ) * self.dt
        threshold = self.threshold + state["q"]
        natural_spike = surrogate_spike(u - threshold)
        forced, forced_hit, new_crossing = force_threshold_crossing(u, threshold, mask, margin)
        spike = surrogate_spike(forced - threshold)
        induced = (spike - natural_spike).clamp(0.0, 1.0)
        reset_spike = natural_spike + (induced if mode == "native_reset" else 0.0)
        q = 0.9 * state["q"] + reset_spike
        return spike, {"u": forced, "v": v, "q": q}, {
            "membrane_pre": u,
            "threshold": threshold,
            "forced_hit": forced_hit,
            "new_crossing": new_crossing,
            "natural_spike": natural_spike,
        }


class TrajectoryNetwork(nn.Module):
    def __init__(self, name: str, cell: nn.Module, task_cfg: TaskConfig, model_cfg: ModelConfig):
        super().__init__()
        self.name = name
        self.cell = cell
        self.task_cfg = task_cfg
        self.model_cfg = model_cfg
        self.readout = FilteredSpikeReadout(
            model_cfg.hidden_size, 2, model_cfg.readout_tau_ms, task_cfg.dt_ms
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mode: str = "clean",
        intervention_masks: dict[int, torch.Tensor] | None = None,
        gamma: float = 1.0,
        margin: float = 0.05,
        return_trace: bool = False,
    ):
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape [batch,time,features]")
        batch, steps, _ = inputs.shape
        state = self.cell.initial_state(batch, inputs.device, inputs.dtype)
        intervention_masks = intervention_masks or {}
        spikes, diagnostics = [], []
        for step in range(steps):
            mask = intervention_masks.get(step)
            if mask is None:
                mask = torch.zeros(
                    batch, self.model_cfg.hidden_size,
                    device=inputs.device, dtype=torch.bool,
                )
            else:
                mask = mask.to(device=inputs.device, dtype=torch.bool)
            spike, state, diag = self.cell.step(
                inputs[:, step], state, mode, mask, gamma, margin
            )
            spikes.append(spike)
            diagnostics.append(diag)
        spike_tensor = torch.stack(spikes, dim=1)
        output = self.readout(spike_tensor)
        if not return_trace:
            return output
        trace: dict[str, Any] = {
            "spikes": spike_tensor,
            "membrane_pre": torch.stack([item["membrane_pre"] for item in diagnostics], dim=1),
            "threshold": torch.stack([item["threshold"] for item in diagnostics], dim=1),
            "forced_hit": torch.stack([item["forced_hit"] for item in diagnostics], dim=1),
            "new_crossing": torch.stack([item["new_crossing"] for item in diagnostics], dim=1),
            "natural_spike": torch.stack([item["natural_spike"] for item in diagnostics], dim=1),
        }
        for key in ("slow", "fast", "slow_pre_reset", "fast_pre_reset"):
            if key in diagnostics[0]:
                trace[key] = torch.stack([item[key] for item in diagnostics], dim=1)
        return output, trace


def build_model(name: str, task_cfg: TaskConfig, model_cfg: ModelConfig) -> TrajectoryNetwork:
    builders = {
        "sprif": lambda: SPRiFCellAdapter(task_cfg.input_size, model_cfg),
        "lif": lambda: LIFCell(task_cfg.input_size, model_cfg, task_cfg.dt_ms),
        "asrnn": lambda: ASRNNCell(task_cfg.input_size, model_cfg, task_cfg.dt_ms),
        "brf": lambda: BRFCell(task_cfg.input_size, model_cfg, task_cfg.dt_ms),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(builders)}")
    return TrajectoryNetwork(name, builders[name](), task_cfg, model_cfg)


__all__ = ["FilteredSpikeReadout", "TrajectoryNetwork", "build_model"]
