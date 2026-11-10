
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

ASRNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ASRNN_DIR, "SRNN_layers"))

from spike_neuron import mem_update_adp, b_j0_value, beta_value, ActFun_adp
from spike_dense import spike_dense, readout_integrator
from spike_rnn import spike_rnn

class ASRNNGSCNet(nn.Module):

    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 256,
        num_classes: int = 12,
        device: str = "cpu",
    ):
        super(ASRNNGSCNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr, 5e-2)

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

        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)
        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

        if is_bias:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias, 0)
            torch.nn.init.constant_(self.dense_1.dense.bias, 0)
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, channel, seq_length, input_dim = x.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        thr_func = ActFun_adp.apply
        input_s = thr_func(x - self.thr) * 1.0 - thr_func(-self.thr - x) * 1.0

        output = 0
        for i in range(seq_length):
            input_x = input_s[:, :, i, :].reshape(b, channel * input_dim)
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2, spike_layer2 = self.rnn_1.forward(spike_layer1)
            mem_layer3 = self.dense_2.forward(spike_layer2)
            output += mem_layer3

        output = F.log_softmax(output / seq_length, dim=1)
        return output

    def forward_with_features(self, x: torch.Tensor) -> tuple:
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
            'spikes_layer1': torch.stack(spikes_layer1, dim=1),
            'spikes_layer2': torch.stack(spikes_layer2, dim=1),
        }
        return output, features

if __name__ == "__main__":

    model = ASRNNGSCNet(device="cpu")
    x = torch.randn(2, 3, 101, 40)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

