"""
ASRNN Model Wrapper for ECG (QTDB Heartbeat Classification).

This module wraps the original ASRNN implementation to provide a unified
interface compatible with the robustness experiment framework.

Reference: Bojian Yin et al., "Accurate and efficient time-domain classification
with adaptive spiking recurrent neural networks", Nature Machine Intelligence, 2021.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Add SRNN_layers to path
ASRNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ASRNN_DIR, "SRNN_layers"))

from spike_neuron import mem_update_adp, b_j0_value, beta_value, ActFun_adp

# Constants from original ECG code
b_j0 = 0.01
tau_m_init = 20
R_m = 1
dt = 1
gamma = 0.5
lens = 0.5
surrograte_type = 'MG'


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp_ecg(torch.autograd.Function):
    """Surrogate gradient function for ECG model."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = 0.15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type == 'linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma


act_fun_adp_ecg = ActFun_adp_ecg.apply


def mem_update_ecg(inputs, mem, spike, tau_m, tau_adp, b, isAdapt=1, dt=1, device='cpu'):
    """Membrane potential update for ECG model."""
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    ro = torch.exp(-1. * dt / tau_adp).to(device)

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp_ecg(inputs_)
    return mem, spike, B, b


class ASRNNECGNet(nn.Module):
    """
    ASRNN model for ECG heartbeat classification (QTDB, 6 classes).

    Architecture matches the original ASRNN ECG paper:
    - Input: ECG signal features (variable length, ~5 features)
    - Layer 1: Linear (input -> 36) + adaptive spiking
    - Layer 2: Linear (36 -> 6) + adaptive spiking with adaptation
    - Output: membrane potential readout (no spike)

    The model uses adaptive thresholds with learnable tau_m and tau_adp.
    """

    def __init__(
        self,
        input_size: int = 5,  # ECG feature dimension
        hidden_size: int = 36,
        num_classes: int = 6,
        sub_seq_length: int = 10,
        device: str = "cpu",
    ):
        super(ASRNNECGNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sub_seq_length = sub_seq_length
        self.device = device

        # Linear layers
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, num_classes)

        # Learnable time constants
        self.tau_adp_h = nn.Parameter(torch.Tensor(hidden_size))
        self.tau_adp_o = nn.Parameter(torch.Tensor(num_classes))
        self.tau_m_h = nn.Parameter(torch.Tensor(hidden_size))
        self.tau_m_o = nn.Parameter(torch.Tensor(num_classes))

        # Initialize weights
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2o.bias, 0)

        # Initialize time constants
        nn.init.constant_(self.tau_adp_h, 7)
        nn.init.constant_(self.tau_adp_o, 100)
        nn.init.constant_(self.tau_m_h, 20)
        nn.init.constant_(self.tau_m_o, 20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, input_dim)
               For ECG: (batch, ~300-600, 5)

        Returns:
            Log-softmax output of shape (batch, num_classes)
        """
        batch_size, seq_num, input_dim = x.shape

        # Initialize states
        hidden_mem = hidden_spike = (torch.rand(batch_size, self.hidden_size) * b_j0).to(self.device)
        output_mem = output_spike = (torch.rand(batch_size, self.num_classes) * b_j0).to(self.device)
        b_h = b_o = b_j0

        # Extended sequence with zeros for readout
        max_iters = seq_num + 500  # Extra steps for readout integration

        output_sum = torch.zeros(batch_size, self.num_classes).to(self.device)

        for i in range(max_iters):
            if i < seq_num:
                input_x = x[:, i, :]
            else:
                input_x = torch.zeros(batch_size, input_dim).to(self.device)

            # Update hidden layer
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, b_h = mem_update_ecg(
                h_input, hidden_mem, hidden_spike,
                self.tau_m_h, self.tau_adp_h, b_h,
                isAdapt=0, dt=dt, device=self.device
            )

            # Update output layer
            o_input = self.h2o(hidden_spike)
            output_mem, output_spike, theta_o, b_o = mem_update_ecg(
                o_input, output_mem, output_spike,
                self.tau_m_o, self.tau_adp_o, b_o,
                isAdapt=1, dt=dt, device=self.device
            )

            # Accumulate output after sub_seq_length
            if i >= self.sub_seq_length:
                output_sum += output_mem

        # Apply log-softmax
        output = F.log_softmax(output_sum, dim=1)
        return output

    def forward_step_by_step(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning per-timestep outputs for analysis.

        Returns:
            (final_output, per_step_outputs, features_dict)
        """
        batch_size, seq_num, input_dim = x.shape

        # Initialize states
        hidden_mem = hidden_spike = (torch.rand(batch_size, self.hidden_size) * b_j0).to(self.device)
        output_mem = output_spike = (torch.rand(batch_size, self.num_classes) * b_j0).to(self.device)
        b_h = b_o = b_j0

        max_iters = seq_num + 500

        outputs = []
        hidden_spikes = []
        output_mems = []

        for i in range(max_iters):
            if i < seq_num:
                input_x = x[:, i, :]
            else:
                input_x = torch.zeros(batch_size, input_dim).to(self.device)

            # Update hidden layer
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, b_h = mem_update_ecg(
                h_input, hidden_mem, hidden_spike,
                self.tau_m_h, self.tau_adp_h, b_h,
                isAdapt=0, dt=dt, device=self.device
            )

            # Update output layer
            o_input = self.h2o(hidden_spike)
            output_mem, output_spike, theta_o, b_o = mem_update_ecg(
                o_input, output_mem, output_spike,
                self.tau_m_o, self.tau_adp_o, b_o,
                isAdapt=1, dt=dt, device=self.device
            )

            if i >= self.sub_seq_length:
                outputs.append(F.log_softmax(output_mem, dim=1).detach().cpu())
                hidden_spikes.append(hidden_spike.detach().cpu())
                output_mems.append(output_mem.detach().cpu())

        final_output = F.log_softmax(output_mem, dim=1)
        features = {
            'hidden_spikes': torch.stack(hidden_spikes, dim=1),  # (B, T, H)
            'output_membrane': torch.stack(output_mems, dim=1),  # (B, T, C)
        }
        return final_output, outputs, features


if __name__ == "__main__":
    # Quick test
    model = ASRNNECGNet(device="cpu")
    x = torch.randn(2, 300, 5)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
