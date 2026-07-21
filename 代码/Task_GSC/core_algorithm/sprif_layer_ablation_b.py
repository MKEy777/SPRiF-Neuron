
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

class SPRiFNeuronLayerAblationB(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        recurrent: bool = False,
        bias: bool = False,
        init_std: float = 0.05,
        tau_alpha_range=(10.0, 80.0),
        tau_rho_range=(4.0, 30.0),
        omega_range: Tuple[float, float] = (0.04 * math.pi, 0.40 * math.pi),
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

        self.alpha_raw = nn.Parameter(torch.empty(hidden_size))
        self.rho_raw = nn.Parameter(torch.empty(hidden_size))
        self.omega_raw = nn.Parameter(torch.empty(hidden_size))

        self._reset_parameters(
            init_std=init_std,
            tau_alpha_range=tau_alpha_range,
            tau_rho_range=tau_rho_range,
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
        omega_range: Tuple[float, float],
    ) -> None:
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
                omega_range[0], omega_range[1],
            )
            self.omega_raw.copy_(self._safe_logit(omega / math.pi))

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
            "prev_spike": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),

        }

    @staticmethod
    def detach_state(state: Optional[StateDict]) -> Optional[StateDict]:
        if state is None:
            return None
        return {k: v.detach() for k, v in state.items()}

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
        }

    def get_spectral_parameters(self) -> Dict[str, Tensor]:
        runtime = self._precompute_runtime_params()
        return {
            "alpha": runtime["alpha"],
            "rho": runtime["rho"],
            "omega": runtime["omega"],
        }

    def _slow_flow(
        self, x_prev: Tensor, input_current: Tensor, runtime: Dict[str, Tensor]
    ) -> Tensor:
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

    def _spike_fn(self, membrane_delta: Tensor) -> Tensor:
        return surrogate_spike(membrane_delta)

    def forward_step(
        self, x_t: Tensor, state: StateDict, runtime: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, StateDict]:
        x_state = state["x"]
        prev_spike = state["prev_spike"]

        input_current = self.input_linear(x_t)
        if self.recurrent and self.recurrent_linear is not None:
            input_current = input_current + self.recurrent_linear(prev_spike)

        x_next = self._slow_flow(x_state, input_current, runtime)

        membrane = x_next[..., 0]
        theta = self.threshold
        spike = self._spike_fn(membrane - theta)

        if isinstance(theta, Tensor):
            reset_scale = theta
        else:
            reset_scale = torch.as_tensor(theta, device=x_next.device, dtype=x_next.dtype)

        x_next_reset = x_next.clone()
        x_next_reset[..., 0] = x_next[..., 0] - spike * reset_scale

        next_state = {
            "x": x_next_reset,
            "prev_spike": spike,
        }
        return spike, membrane, next_state

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
                    voltages.append(state["x"][..., 0])
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

