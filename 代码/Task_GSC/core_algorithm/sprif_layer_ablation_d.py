"""
SPRiF Ablation D: Free 3x3 A matrix — unconstrained slow-state recurrence.

Replaces the structured spectral block-diagonal slow dynamics
(alpha, rho*R(omega)) with a fully-learnable 3x3 matrix A_raw and a
per-neuron input vector B_vec.

Fast state (u, eta, G, fast_coupling, lambda_reset) and projective reset
are unchanged.  This ablation tests whether the spectral parameterisation
provides inductive bias beyond having more parameters.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn

StateDict = Dict[str, Tensor]

lens = 0.5
gamma = 0.5


def gaussian(x: Tensor, mu: float = 0.0, sigma: float = 0.5) -> Tensor:
    denom = torch.sqrt(2 * torch.tensor(math.pi, device=x.device, dtype=x.dtype)) * sigma
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / denom


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: Tensor) -> Tensor:
        ctx.save_for_backward(input_tensor)
        return input_tensor.gt(0).to(input_tensor.dtype)

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


SurrogateSpike = ActFun_adp


class SPRiFNeuronLayerAblationD(nn.Module):
    """SPRiF neuron with free 3x3 A matrix instead of spectral parameterisation.

    Slow state recurrence:  x_t = A_raw @ x_{t-1} + B_vec * input_current
    where A_raw is (H, 3, 3) and B_vec is (H, 3).

    Fast state (2D), projective reset, and all fast parameters are unchanged.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        recurrent: bool = False,
        bias: bool = False,
        init_std: float = 0.05,
        tau_eta_range: Tuple[float, float] = (1.5, 10.0),
        **unused_kwargs,
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

        # Free slow-state parameters (replaces alpha_raw, rho_raw, omega_raw)
        self.A_raw = nn.Parameter(torch.empty(hidden_size, 3, 3))   # (H, 3, 3)
        self.B_vec = nn.Parameter(torch.empty(hidden_size, 3))       # (H, 3)

        # Fast-state parameters (unchanged from full SPRiF)
        self.lambda_reset = nn.Parameter(torch.empty(hidden_size))
        self.eta_raw = nn.Parameter(torch.empty(hidden_size, 2))
        self.fast_coupling = nn.Parameter(torch.empty(hidden_size))
        self.G = nn.Parameter(torch.empty(hidden_size, 2, 3))

        self._reset_parameters(
            init_std=init_std,
            tau_eta_range=tau_eta_range,
        )

    @staticmethod
    def _safe_logit(x: Tensor, eps: float = 1e-4) -> Tensor:
        return torch.logit(x.clamp(eps, 1.0 - eps))

    def _reset_parameters(
        self,
        init_std: float,
        tau_eta_range: Tuple[float, float],
    ) -> None:
        nn.init.xavier_uniform_(self.input_linear.weight)
        if self.input_linear.bias is not None:
            nn.init.zeros_(self.input_linear.bias)
        if self.recurrent_linear is not None:
            nn.init.orthogonal_(self.recurrent_linear.weight)

        with torch.no_grad():
            # A_raw: near-scaled identity for training stability
            eye = torch.eye(3).unsqueeze(0).expand(self.hidden_size, -1, -1).clone()
            noise = torch.empty(self.hidden_size, 3, 3).normal_(0.0, 0.02)
            self.A_raw.copy_(eye * 0.9 + noise)

            # B_vec: mimics (1-alpha, 1-rho, 0) pattern from full model
            B_init = torch.zeros(self.hidden_size, 3)
            B_init[:, 0] = 0.1   # like 1-alpha
            B_init[:, 1] = 0.1   # like 1-rho
            B_init[:, 2] = 0.0   # x2 never directly driven in full model
            self.B_vec.copy_(B_init + torch.empty_like(B_init).normal_(0.0, 0.01))

            # Fast-state initialisation (identical to full SPRiF)
            tau_eta = torch.exp(
                torch.empty(self.hidden_size, 2).uniform_(
                    math.log(tau_eta_range[0]),
                    math.log(tau_eta_range[1]),
                )
            )
            eta = torch.exp(-1.0 / tau_eta)
            self.eta_raw.copy_(self._safe_logit(eta))
            self.lambda_reset.normal_(mean=0.0, std=init_std)

        nn.init.normal_(self.fast_coupling, mean=0.0, std=init_std)
        nn.init.normal_(self.G, mean=0.0, std=0.05)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

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
            "u": 0.1 * torch.rand(batch_size, self.hidden_size, 2, device=device, dtype=dtype),
            "prev_spike": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        }

    @staticmethod
    def detach_state(state: Optional[StateDict]) -> Optional[StateDict]:
        if state is None:
            return None
        return {k: v.detach() for k, v in state.items()}

    # ------------------------------------------------------------------
    # Runtime parameters
    # ------------------------------------------------------------------

    def _precompute_runtime_params(self) -> Dict[str, Tensor]:
        eta = torch.sigmoid(self.eta_raw).unsqueeze(0)

        lambda_reset = self.lambda_reset
        ones = torch.ones_like(lambda_reset)
        reset_direction = torch.stack((ones, lambda_reset), dim=-1)

        # --- Spectral normalisation: keep A eigenvalues inside unit circle ---
        # Without this, free 3x3 matrices can develop spectral radius > 1,
        # causing exponential growth over long sequences -> NaN.
        eigvals = torch.linalg.eigvals(self.A_raw)          # (H, 3)  complex
        spectral_radius = eigvals.abs().max(dim=-1).values   # (H,)     real
        # Scale factor: 1.0 if stable, < 1.0 if unstable
        scale = torch.clamp(1.0 / (spectral_radius + 1e-8), max=1.0)  # (H,)
        A_stable = self.A_raw * scale.unsqueeze(-1).unsqueeze(-1)      # (H, 3, 3)

        return {
            "eta": eta,
            "fast_coupling": self.fast_coupling,
            "lambda_reset": lambda_reset,
            "reset_direction": reset_direction,
            "A_stable": A_stable,
            "spectral_radius": spectral_radius,
        }

    def get_spectral_parameters(self) -> Dict[str, Tensor]:
        runtime = self._precompute_runtime_params()
        return {
            "eta": runtime["eta"].squeeze(0),
            "lambda_reset": runtime["lambda_reset"],
        }

    def get_reset_direction(self) -> Tensor:
        return self._precompute_runtime_params()["reset_direction"]

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def _slow_flow(
        self, x_prev: Tensor, input_current: Tensor, runtime: Dict[str, Tensor]
    ) -> Tensor:
        """Free 3x3 matrix recurrence with spectral normalisation."""
        # x_prev: (B, H, 3),  A_stable: (H, 3, 3),  B_vec: (H, 3)
        A_stable = runtime["A_stable"]                                   # (H, 3, 3)
        Ax = torch.einsum("bhi,hij->bhj", x_prev, A_stable)             # (B, H, 3)
        Bx = torch.einsum("hi,bh->bhi", self.B_vec, input_current)      # (B, H, 3)
        return Ax + Bx

    def _fast_flow(
        self, u_prev: Tensor, x_t: Tensor, runtime: Dict[str, Tensor]
    ) -> Tensor:
        """Fast 2D dynamics — identical to full SPRiF."""
        eta = runtime["eta"]
        fast_coupling = runtime["fast_coupling"].unsqueeze(0)

        slow_to_fast = torch.einsum("bhk,hpk->bhp", x_t, self.G)

        u0 = eta[..., 0] * u_prev[..., 0] + fast_coupling * u_prev[..., 1]
        u1 = eta[..., 1] * u_prev[..., 1]
        fast_leak = torch.stack((u0, u1), dim=-1)

        return fast_leak + slow_to_fast

    # ------------------------------------------------------------------
    # Spike
    # ------------------------------------------------------------------

    def _spike_fn(self, membrane_delta: Tensor) -> Tensor:
        return surrogate_spike(membrane_delta)

    # ------------------------------------------------------------------
    # Single time-step
    # ------------------------------------------------------------------

    def forward_step(
        self, x_t: Tensor, state: StateDict, runtime: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, StateDict]:
        x_state = state["x"]
        u_state = state["u"]
        prev_spike = state["prev_spike"]

        input_current = self.input_linear(x_t)
        if self.recurrent and self.recurrent_linear is not None:
            input_current = input_current + self.recurrent_linear(prev_spike)

        x_next = self._slow_flow(x_state, input_current, runtime)
        u_tilde = self._fast_flow(u_state, x_next, runtime)

        membrane = u_tilde[..., 0]
        theta = self.threshold
        spike = self._spike_fn(membrane - theta)

        if isinstance(theta, Tensor):
            reset_scale = theta
        else:
            reset_scale = torch.as_tensor(
                theta, device=u_tilde.device, dtype=u_tilde.dtype
            )

        # Projective reset (identical to full SPRiF)
        u_next = (
            u_tilde
            - spike.unsqueeze(-1)
            * runtime["reset_direction"].unsqueeze(0)
            * reset_scale.unsqueeze(-1)
        )

        next_state = {
            "x": x_next,
            "u": u_next,
            "prev_spike": spike,
        }
        return spike, membrane, next_state

    # ------------------------------------------------------------------
    # Full sequence forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        state: Optional[StateDict] = None,
        batch_first: bool = False,
        return_state: bool = False,
        return_voltages: bool = False,
        return_post_reset_voltages: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, StateDict], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, StateDict]]:
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

        if state is None:
            state = self.init_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            state = {
                k: v.to(device=x.device, dtype=x.dtype) for k, v in state.items()
            }

        runtime = self._precompute_runtime_params()
        spikes: List[Tensor] = []
        voltages: List[Tensor] = []

        for t in range(time_steps):
            spike, membrane, state = self.forward_step(x[t], state, runtime)
            spikes.append(spike)

            if return_voltages:
                if return_post_reset_voltages:
                    voltages.append(state["u"][..., 0])
                else:
                    voltages.append(membrane)

        spike_seq = torch.stack(spikes, dim=0)
        voltage_seq = torch.stack(voltages, dim=0) if return_voltages else None

        if batch_first:
            spike_seq = spike_seq.transpose(0, 1)
            if voltage_seq is not None:
                voltage_seq = voltage_seq.transpose(0, 1)

        if return_voltages and return_state:
            return spike_seq, voltage_seq, state
        if return_voltages:
            return spike_seq, voltage_seq
        if return_state:
            return spike_seq, state
        return spike_seq
