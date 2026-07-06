"""
ASRNN Model Wrapper for GSC (Google Speech Commands).

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
from spike_dense import spike_dense, readout_integrator
from spike_rnn import spike_rnn


class ASRNNGSCNet(nn.Module):
    """
    ASRNN model for Google Speech Commands (12-class keyword spotting).

    Architecture matches the original ASRNN paper:
    - Input: mel-spectrogram features (3 channels x 101 frames x 40 mels)
    - Layer 1: spike_dense (input -> 256)
    - Layer 2: spike_rnn (256 -> 256, recurrent)
    - Output: readout_integrator (256 -> 12)

    The model uses adaptive thresholds with learnable tau_m and tau_adp.
    """

    def __init__(
        self,
        input_size: int = 120,  # 3 * 40 (stacked mel-spectrogram)
        hidden_size: int = 256,
        num_classes: int = 12,
        device: str = "cpu",
    ):
        super(ASRNNGSCNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

        # Threshold parameter
        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr, 5e-2)

        # Network layers
        is_bias = True
        self.dense_1 = spike_dense(
            input_size, hidden_size,
            tauAdp_inital_std=50, tauAdp_inital=200,
            tauM=20, tauM_inital_std=5,
            device=device, bias=is_bias
        )
        self.rnn_1 = spike_rnn(
            hidden_size, hidden_size,
            tauAdp_inital_std=50, tauAdp_inital=200,
            tauM=20, tauM_inital_std=5,
            device=device, bias=is_bias
        )
        self.dense_2 = readout_integrator(
            hidden_size, num_classes,
            tauM=10, tauM_inital_std=1,
            device=device, bias=is_bias
        )

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)
        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

        if is_bias:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias, 0)
            torch.nn.init.constant_(self.dense_1.dense.bias, 0)
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_length, input_dim)
               For GSC: (batch, 3, 101, 40) -> reshaped to (batch, 3, 101, 40)

        Returns:
            Log-softmax output of shape (batch, num_classes)
        """
        b, channel, seq_length, input_dim = x.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        # Apply threshold to input (gating mechanism)
        thr_func = ActFun_adp.apply
        input_s = thr_func(x - self.thr) * 1.0 - thr_func(-self.thr - x) * 1.0

        output = 0
        for i in range(seq_length):
            input_x = input_s[:, :, i, :].reshape(b, channel * input_dim)
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2, spike_layer2 = self.rnn_1.forward(spike_layer1)
            mem_layer3 = self.dense_2.forward(spike_layer2)
            output += mem_layer3

        # Average over time and apply log-softmax
        output = F.log_softmax(output / seq_length, dim=1)
        return output

    def forward_with_features(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning intermediate features for analysis.

        Returns:
            (output, features_dict) where features_dict contains:
            - 'spikes_layer1': spike trains from layer 1
            - 'spikes_layer2': spike trains from layer 2
            - 'membrane_output': membrane potential from output layer
        """
        b, channel, seq_length, input_dim = x.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        thr_func = ActFun_adp.apply
        input_s = thr_func(x - self.thr) * 1.0 - thr_func(-self.thr - x) * 1.0

        spikes_layer1 = []
        spikes_layer2 = []
        output = 0

        for i in range(seq_length):
            input_x = input_s[:, :, i, :].reshape(b, channel * input_dim)
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2, spike_layer2 = self.rnn_1.forward(spike_layer1)
            mem_layer3 = self.dense_2.forward(spike_layer2)

            spikes_layer1.append(spike_layer1.detach().cpu())
            spikes_layer2.append(spike_layer2.detach().cpu())
            output += mem_layer3

        output = F.log_softmax(output / seq_length, dim=1)
        features = {
            'spikes_layer1': torch.stack(spikes_layer1, dim=1),  # (B, T, H)
            'spikes_layer2': torch.stack(spikes_layer2, dim=1),  # (B, T, H)
        }
        return output, features


if __name__ == "__main__":
    # Quick test
    model = ASRNNGSCNet(device="cpu")
    x = torch.randn(2, 3, 101, 40)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
