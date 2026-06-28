"""
SPRiF 和 LIF 轨迹模型
===========================

两个模型共享相同的输入/输出维度，关键差异在于:
    - SPRiF: readout 从慢状态 x_t 读取 (不被 reset)
    - LIF:   readout 从膜电位 v_t 读取 (被 reset)

Usage:
    from models import SPRiFTrajectoryModel, LIFTrajectoryModel
"""

import math
import sys
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Add 代码/ to path so we can import Task_ECG as a package
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from Task_ECG.core_algorithm.sprif_layer import SPRiFNeuronLayer, surrogate_spike


# ============================================================================
# SPRiF Trajectory Model
# ============================================================================

class SPRiFTrajectoryModel(nn.Module):
    """SPRiF network for phase trajectory maintenance task.

    Architecture:
        Input (32) → SPRiF Hidden Layer (64) → Readout from x_t (192→2)

    The readout operates on the slow state x_t [batch, 64, 3], which is
    structurally preserved across spike events. This is the key advantage
    over LIF: spike reset does not touch the memory state.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 2,
        neuron_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if neuron_kwargs is None:
            neuron_kwargs = {}

        self.hidden_size = hidden_size
        self.sprif_layer = SPRiFNeuronLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            recurrent=False,
            **neuron_kwargs,
        )

        # Readout from slow state: concat(x_{1},...,x_{N}) → ŷ
        self.readout = nn.Linear(hidden_size * 3, output_size)

    def forward(
        self,
        x: Tensor,
        probe_mask: Optional[Tensor] = None,
        A_probe: float = 1.0,
    ) -> Tensor:
        """Forward pass returning output trajectory.

        Args:
            x: [batch, time, features] input spikes (batch_first=True)
            probe_mask: [batch, time] binary mask for perturbation current
            A_probe: perturbation current amplitude

        Returns:
            outputs: [batch, time, 2] predicted (cos, sin) trajectory
        """
        batch_size, T, _ = x.shape
        state = self.sprif_layer.init_state(batch_size, device=x.device, dtype=x.dtype)
        runtime = self.sprif_layer._precompute_runtime_params()

        outputs = []

        for t in range(T):
            # Compute input current with optional perturbation
            input_current = self.sprif_layer.input_linear(x[:, t, :])
            if probe_mask is not None:
                input_current = input_current + A_probe * probe_mask[:, t].unsqueeze(-1)

            spike, membrane, state = self.sprif_layer.forward_step(
                x[:, t, :], state, runtime, input_current=input_current,
            )

            # Readout from slow state x_t (never reset)
            x_t = state["x"]                             # [batch, hidden, 3]
            out = self.readout(x_t.reshape(batch_size, -1))  # [batch, 2]
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # [batch, T, 2]

    def get_spectral_parameters(self) -> Dict[str, Tensor]:
        """Get learned spectral parameters for analysis."""
        return self.sprif_layer.get_spectral_parameters()


# ============================================================================
# LIF Neuron Layer (simple implementation)
# ============================================================================

class LIFNeuronLayer(nn.Module):
    """Standard Leaky Integrate-and-Fire neuron layer.

    Dynamics:
        v_pre[t] = β · v_post[t-1] + input_current[t]
        z[t] = H(v_pre[t] - θ)
        v_post[t] = v_pre[t] - θ · z[t]     (soft reset)

    where β = exp(-dt/τ_m) is the membrane decay factor.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        tau_m: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.tau_m = tau_m

        # Membrane decay: β = exp(-1/τ_m) since dt = 1ms
        self.beta = math.exp(-1.0 / tau_m)

        self.input_linear = nn.Linear(input_size, hidden_size, bias=bias)

        # Initialize weights
        nn.init.xavier_uniform_(self.input_linear.weight)
        if bias and self.input_linear.bias is not None:
            nn.init.zeros_(self.input_linear.bias)

    @property
    def _beta_tensor(self) -> Tensor:
        """Return beta as a scalar tensor on the correct device."""
        return torch.tensor(self.beta, device=self.input_linear.weight.device,
                            dtype=self.input_linear.weight.dtype)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Dict[str, Tensor]:
        """Initialize LIF state (membrane potential)."""
        if device is None:
            device = self.input_linear.weight.device
        if dtype is None:
            dtype = self.input_linear.weight.dtype

        return {
            "v": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        }

    def forward_step(
        self,
        x_t: Tensor,
        state: Dict[str, Tensor],
        input_current: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Single timestep forward.

        Args:
            x_t: [batch, input_size] input spikes
            state: LIF state dict
            input_current: optional pre-computed input current

        Returns:
            spike:    [batch, hidden] output spikes
            membrane: [batch, hidden] pre-reset membrane potential (v_pre)
            next_state: updated state dict
        """
        if input_current is None:
            input_current = self.input_linear(x_t)

        v_prev = state["v"]
        v_pre = self._beta_tensor * v_prev + input_current
        spike = surrogate_spike(v_pre - self.threshold)
        v_post = v_pre - self.threshold * spike

        next_state = {"v": v_post}
        return spike, v_pre, next_state

    def forward(self, x: Tensor, batch_first: bool = False) -> Tensor:
        """Full sequence forward (returns spikes only)."""
        if batch_first:
            x = x.transpose(0, 1)

        time_steps, batch_size, _ = x.shape
        state = self.init_state(batch_size, device=x.device, dtype=x.dtype)
        spikes = []

        for t in range(time_steps):
            spike, _, state = self.forward_step(x[t], state)
            spikes.append(spike)

        spike_seq = torch.stack(spikes, dim=0)

        if batch_first:
            spike_seq = spike_seq.transpose(0, 1)

        return spike_seq


# ============================================================================
# LIF Trajectory Model
# ============================================================================

class LIFTrajectoryModel(nn.Module):
    """LIF control network for phase trajectory maintenance task.

    Architecture:
        Input (32) → LIF Hidden Layer (64) → Readout from v_t (64→2)

    Readout operates on membrane potential v_t, which IS reset at each spike.
    This is the structural difference from SPRiF.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 2,
        threshold: float = 1.0,
        tau_m: float = 20.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.lif_layer = LIFNeuronLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            threshold=threshold,
            tau_m=tau_m,
            bias=False,
        )

        # Readout from membrane potential v_t
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,
        probe_mask: Optional[Tensor] = None,
        A_probe: float = 1.0,
    ) -> Tensor:
        """Forward pass returning output trajectory.

        Args:
            x: [batch, time, features] input spikes (batch_first=True)
            probe_mask: [batch, time] binary mask for perturbation current
            A_probe: perturbation current amplitude

        Returns:
            outputs: [batch, time, 2] predicted (cos, sin) trajectory
        """
        batch_size, T, _ = x.shape
        state = self.lif_layer.init_state(batch_size, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(T):
            input_current = self.lif_layer.input_linear(x[:, t, :])
            if probe_mask is not None:
                input_current = input_current + A_probe * probe_mask[:, t].unsqueeze(-1)

            spike, v_pre, state = self.lif_layer.forward_step(
                x[:, t, :], state, input_current=input_current,
            )

            # Readout from membrane potential (gets reset at spike)
            out = self.readout(v_pre)  # [batch, 2]
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # [batch, T, 2]


# ============================================================================
# Model factory
# ============================================================================

def create_models(
    input_size: int = 32,
    hidden_size: int = 64,
    output_size: int = 2,
    sprif_kwargs: Optional[dict] = None,
    lif_kwargs: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SPRiFTrajectoryModel, LIFTrajectoryModel]:
    """Create both SPRiF and LIF trajectory models.

    Returns:
        sprif_model, lif_model
    """
    sprif_model = SPRiFTrajectoryModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        neuron_kwargs=sprif_kwargs,
    ).to(device)

    lif_kwargs_default = {"threshold": 1.0, "tau_m": 20.0}
    if lif_kwargs is not None:
        lif_kwargs_default.update(lif_kwargs)

    lif_model = LIFTrajectoryModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        **lif_kwargs_default,
    ).to(device)

    return sprif_model, lif_model


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing models...")
    device = torch.device("cpu")
    batch, T = 4, 900

    # Dummy input
    x = torch.rand(batch, T, 32, device=device)
    probe_mask = torch.zeros(batch, T, device=device)
    for start in [180, 300, 420, 540, 660, 780]:
        probe_mask[:, start:start + 10] = 1.0

    # SPRiF
    sprif_model = SPRiFTrajectoryModel().to(device)
    sprif_out = sprif_model(x, probe_mask)
    print(f"  SPRiF output shape: {sprif_out.shape}")  # [4, 900, 2]
    n_params = sum(p.numel() for p in sprif_model.parameters())
    print(f"  SPRiF params: {n_params:,}")

    # Check spectral params
    spec = sprif_model.get_spectral_parameters()
    for k, v in spec.items():
        print(f"    {k}: shape={v.shape}, range=[{v.min().item():.3f}, {v.max().item():.3f}]")

    # LIF
    lif_model = LIFTrajectoryModel().to(device)
    lif_out = lif_model(x, probe_mask)
    print(f"  LIF output shape: {lif_out.shape}")  # [4, 900, 2]
    n_params = sum(p.numel() for p in lif_model.parameters())
    print(f"  LIF params: {n_params:,}")

    print("Done.")
