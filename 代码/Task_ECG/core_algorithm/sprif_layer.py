# SPRiF TNNLS model: three spectral coordinates, scalar filter h, scalar membrane u.
# Scalar subtractive reset affects only u; h drives the membrane one step later.
# Eleven local learned coefficients: alpha, rho, omega, eta[2], bar_G[2,3].
# The second readout row is learned directly after absorbing its former scale.
# Task-specific decay/frequency initialization and network interfaces are retained.

import math
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor, nn

StateDict = Dict[str, Tensor]

# Multi-Gaussian surrogate; H(0)=1 matches the manuscript.
lens = 0.5
gamma = 0.5

def gaussian(x: Tensor, mu: float = 0.0, sigma: float = 0.5) -> Tensor:
    denom = torch.sqrt(2 * torch.tensor(math.pi, device=x.device, dtype=x.dtype)) * sigma
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / denom

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: Tensor) -> Tensor:
        ctx.save_for_backward(input_tensor)
        return input_tensor.ge(0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale, hight = 6.0, 0.15
        temp = (
            gaussian(input_tensor, mu=0.0, sigma=lens) * (1.0 + hight)
            - gaussian(input_tensor, mu=lens, sigma=scale * lens) * hight
            - gaussian(input_tensor, mu=-lens, sigma=scale * lens) * hight
        )
        return grad_input * temp.to(dtype=grad_input.dtype) * gamma

def surrogate_spike(input_tensor: Tensor) -> Tensor:
    return ActFun_adp.apply(input_tensor)

class SPRiFNeuronLayer(nn.Module):
    """Five-state scalar-reset neuron used by the TNNLS manuscript."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        recurrent: bool = False,
        bias: bool = False,
        init_std: float = 0.05,
        tau_alpha_range: Tuple[float, float] = (20.0, 120.0),
        tau_rho_range: Tuple[float, float] = (4.0, 30.0),
        tau_eta_range: Tuple[float, float] = (0.8, 8.0),
        omega_range: Tuple[float, float] = (0.02 * math.pi, 0.20 * math.pi),
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.recurrent = recurrent
        self.input_linear = nn.Linear(input_size, hidden_size, bias=bias)
        self.recurrent_linear = nn.Linear(hidden_size, hidden_size, bias=False) if recurrent else None

        self.alpha_raw = nn.Parameter(torch.empty(hidden_size))
        self.rho_raw = nn.Parameter(torch.empty(hidden_size))
        self.omega_raw = nn.Parameter(torch.empty(hidden_size))
        self.eta_raw = nn.Parameter(torch.empty(hidden_size, 2))
        self.bar_G = nn.Parameter(torch.empty(hidden_size, 2, 3))

        self._reset_parameters(
            init_std=init_std,
            tau_alpha_range=tau_alpha_range,
            tau_rho_range=tau_rho_range,
            tau_eta_range=tau_eta_range,
            omega_range=omega_range,
        )

    @staticmethod
    def _safe_logit(x: Tensor, eps: float = 1e-4) -> Tensor:
        return torch.logit(x.clamp(eps, 1.0 - eps))

    def _reset_parameters(
        self,
        init_std: float,
        tau_alpha_range: Tuple[float, float],
        tau_rho_range: Tuple[float, float],
        tau_eta_range: Tuple[float, float],
        omega_range: Tuple[float, float],
    ) -> None:
        # Xavier uniform for W_in; orthogonal init for W_rec
        nn.init.xavier_uniform_(self.input_linear.weight)
        if self.input_linear.bias is not None:
            nn.init.zeros_(self.input_linear.bias)
        if self.recurrent_linear is not None:
            nn.init.orthogonal_(self.recurrent_linear.weight)

        with torch.no_grad():
            tau_alpha = torch.exp(
                torch.empty(self.hidden_size).uniform_(
                    math.log(tau_alpha_range[0]),
                    math.log(tau_alpha_range[1]),
                )
            )
            alpha = torch.exp(-1.0 / tau_alpha)
            self.alpha_raw.copy_(self._safe_logit(alpha))

            tau_rho = torch.exp(
                torch.empty(self.hidden_size).uniform_(
                    math.log(tau_rho_range[0]),
                    math.log(tau_rho_range[1]),
                )
            )
            rho = torch.exp(-1.0 / tau_rho)
            self.rho_raw.copy_(self._safe_logit(rho))

            omega = torch.empty(self.hidden_size).uniform_(
                omega_range[0],
                omega_range[1],
            )
            self.omega_raw.copy_(self._safe_logit(omega / math.pi))

            tau_eta = torch.exp(
                torch.empty(self.hidden_size, 2).uniform_(
                    math.log(tau_eta_range[0]),
                    math.log(tau_eta_range[1]),
                )
            )
            eta = torch.exp(-1.0 / tau_eta)
            self.eta_raw.copy_(self._safe_logit(eta))

        nn.init.normal_(self.bar_G, mean=0.0, std=0.05)
        with torch.no_grad():
            # One temporary draw per neuron preserves the scalar-reset product distribution.
            # It is discarded here, leaving only the absorbed row as a learned parameter.
            init_scale = self.bar_G.new_empty(self.hidden_size).normal_(0.0, init_std)
            self.bar_G[:, 1, :].mul_(init_scale.unsqueeze(-1))

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
            "x": torch.zeros(batch_size, self.hidden_size, 3, device=device, dtype=dtype),
            "h": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            "u": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            "prev_spike": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        }

    def _precompute_runtime_params(self) -> Dict[str, Tensor]:
        alpha = torch.sigmoid(self.alpha_raw)
        rho = torch.sigmoid(self.rho_raw)
        omega = math.pi * torch.sigmoid(self.omega_raw)
        return {
            "alpha": alpha,
            "rho": rho,
            "omega": omega,
            "cos_w": torch.cos(omega),
            "sin_w": torch.sin(omega),
            "eta": torch.sigmoid(self.eta_raw).unsqueeze(0),
        }

    def get_spectral_parameters(self) -> Dict[str, Tensor]:
        runtime = self._precompute_runtime_params()
        return {
            "alpha": runtime["alpha"],
            "rho": runtime["rho"],
            "omega": runtime["omega"],
            "eta": runtime["eta"].squeeze(0),
            "bar_G": self.bar_G,
        }

    def _slow_flow(self, x_prev: Tensor, input_current: Tensor, runtime: Dict[str, Tensor]) -> Tensor:
        # Spectral state: x_t = A*x_{t-1} + B*I_t.
        # Scalar membrane reset does not directly change x.
        x_real = x_prev[..., 0]
        x_osc_1 = x_prev[..., 1]
        x_osc_2 = x_prev[..., 2]

        alpha = runtime["alpha"].unsqueeze(0)
        rho = runtime["rho"].unsqueeze(0)
        cos_w = runtime["cos_w"].unsqueeze(0)
        sin_w = runtime["sin_w"].unsqueeze(0)

        x_next_0 = alpha * x_real + (1.0 - alpha) * input_current
        x_next_1 = rho * (cos_w * x_osc_1 - sin_w * x_osc_2) + (1.0 - rho) * input_current
        x_next_2 = rho * (sin_w * x_osc_1 + cos_w * x_osc_2)

        return torch.stack((x_next_0, x_next_1, x_next_2), dim=-1)

    def _fast_flow(
        self, u_prev: Tensor, h_prev: Tensor, x_t: Tensor,
        runtime: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Both paths use the new spectral state; the membrane uses OLD h.
        projection = torch.einsum("bhk,hpk->bhp", x_t, self.bar_G)
        eta = runtime["eta"]
        membrane = eta[..., 0] * u_prev + h_prev + projection[..., 0]
        h_next = eta[..., 1] * h_prev + projection[..., 1]
        return membrane, h_next

    def _spike_fn(self, membrane_delta: Tensor) -> Tensor:
        return surrogate_spike(membrane_delta)

    def forward_step(
        self, x_t: Tensor, state: StateDict, runtime: Dict[str, Tensor],
        input_current: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, StateDict]:
        # A supplied current is already projected, including any recurrent drive.
        if input_current is None:
            input_current = self.input_linear(x_t)
            if self.recurrent and self.recurrent_linear is not None:
                input_current = input_current + self.recurrent_linear(state["prev_spike"])

        x_next = self._slow_flow(state["x"], input_current, runtime)
        membrane, h_next = self._fast_flow(state["u"], state["h"], x_next, runtime)
        theta = torch.as_tensor(self.threshold, device=membrane.device, dtype=membrane.dtype)
        spike = self._spike_fn(membrane - theta)
        next_state = {
            "x": x_next,
            "h": h_next,
            "u": membrane - spike * theta,
            "prev_spike": spike,
        }
        return spike, membrane, next_state

    def forward(
        self,
        x: Tensor,
        batch_first: bool = False,
    ) -> Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be [time, batch, features] or [batch, time, features].")

        if batch_first:
            x = x.transpose(0, 1)

        time_steps, batch_size, feature_dim = x.shape
        if feature_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {feature_dim}.")

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

__all__ = ["SPRiFNeuronLayer"]

