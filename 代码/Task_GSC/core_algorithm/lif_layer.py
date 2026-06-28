"""
Standard LIF neuron layer with API compatible with SPRiFNeuronLayer.

Exposes the same signatures (init_state, forward_step, forward, forward_with_state,
get_spectral_parameters) so it can be used as a drop-in replacement in GSC models.

Dynamics:
    v_tilde = beta * v_{t-1} + (1 - beta) * input_current
    spike = Heaviside(v_tilde - threshold)
    v_next = v_tilde - spike * threshold   (soft reset)
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from core_algorithm.sprif_layer import surrogate_spike

StateDict = Dict[str, Tensor]


class LIFNeuronLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        recurrent: bool = False,
        bias: bool = False,
        init_std: float = 0.05,
        tau_m_range: Tuple[float, float] = (10.0, 80.0),
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.recurrent = recurrent

        self.input_linear = nn.Linear(input_size, hidden_size, bias=bias)
        self.recurrent_linear = (
            nn.Linear(hidden_size, hidden_size, bias=False) if recurrent else None
        )

        # Membrane decay: beta controls how much of the old voltage is retained.
        # Parameterized via raw logit to keep beta in (0, 1).
        # tau_m = -1 / log(beta)  →  beta = exp(-1 / tau_m)
        self.beta_raw = nn.Parameter(torch.empty(hidden_size))

        self._reset_parameters(init_std=init_std, tau_m_range=tau_m_range)

    @staticmethod
    def _safe_logit(x: Tensor, eps: float = 1e-4) -> Tensor:
        return torch.logit(x.clamp(eps, 1.0 - eps))

    def _reset_parameters(
        self,
        init_std: float,
        tau_m_range: Tuple[float, float],
    ) -> None:
        nn.init.xavier_uniform_(self.input_linear.weight)
        if self.input_linear.bias is not None:
            nn.init.zeros_(self.input_linear.bias)
        if self.recurrent_linear is not None:
            nn.init.orthogonal_(self.recurrent_linear.weight)

        # Initialize beta to match tau_m ~ Uniform[tau_min, tau_max] in log-space
        with torch.no_grad():
            tau_m = torch.exp(
                torch.empty(self.hidden_size).uniform_(
                    math.log(tau_m_range[0]),
                    math.log(tau_m_range[1]),
                )
            )
            beta = torch.exp(-1.0 / tau_m)
            self.beta_raw.copy_(self._safe_logit(beta))

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StateDict:
        if device is None:
            device = self.input_linear.weight.device
        if dtype is None:
            dtype = self.input_linear.weight.dtype

        return {
            "v": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            "prev_spike": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        }

    def _precompute_runtime_params(self) -> Dict[str, Tensor]:
        beta = torch.sigmoid(self.beta_raw)
        return {"beta": beta}

    def get_spectral_parameters(self) -> Dict[str, Tensor]:
        """Return membrane time constant for compatibility with SPRiF API."""
        beta = torch.sigmoid(self.beta_raw)
        tau_m = -1.0 / torch.log(beta + 1e-8)
        return {"tau_m": tau_m}

    def forward_step(
        self,
        x_t: Tensor,
        state: StateDict,
        runtime: Dict[str, Tensor],
        input_current: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, StateDict]:
        v_prev = state["v"]
        prev_spike = state["prev_spike"]

        if input_current is None:
            input_current = self.input_linear(x_t)
            if self.recurrent and self.recurrent_linear is not None:
                input_current = input_current + self.recurrent_linear(prev_spike)

        beta = runtime["beta"].unsqueeze(0)
        v_tilde = beta * v_prev + (1.0 - beta) * input_current

        theta = self.threshold
        spike = surrogate_spike(v_tilde - theta)

        if isinstance(theta, Tensor):
            reset_scale = theta
        else:
            reset_scale = torch.as_tensor(theta, device=v_tilde.device, dtype=v_tilde.dtype)

        v_next = v_tilde - spike * reset_scale

        next_state = {
            "v": v_next,
            "prev_spike": spike,
        }
        return spike, v_tilde, next_state

    def forward(
        self,
        x: Tensor,
        batch_first: bool = False,
    ) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Input must be [time, batch, features] or [batch, time, features]."
            )

        if batch_first:
            x = x.transpose(0, 1)

        time_steps, batch_size, feature_dim = x.shape
        if feature_dim != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {feature_dim}."
            )

        state = self.init_state(batch_size, device=x.device, dtype=x.dtype)
        runtime = self._precompute_runtime_params()
        spikes = []

        for t in range(time_steps):
            spike, _, state = self.forward_step(x[t], state, runtime)
            spikes.append(spike)

        spike_seq = torch.stack(spikes, dim=0)

        if batch_first:
            spike_seq = spike_seq.transpose(0, 1)

        return spike_seq

    def forward_with_state(
        self,
        x: Tensor,
        state: StateDict,
        batch_first: bool = False,
    ) -> Tuple[Tensor, StateDict]:
        if batch_first:
            x = x.transpose(0, 1)

        runtime = self._precompute_runtime_params()
        spikes = []

        for t in range(x.shape[0]):
            spike, _, state = self.forward_step(x[t], state, runtime)
            spikes.append(spike)

        spike_seq = torch.stack(spikes, dim=0)

        if batch_first:
            spike_seq = spike_seq.transpose(0, 1)

        return spike_seq, state


__all__ = ["LIFNeuronLayer"]
