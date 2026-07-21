
from typing import List, Optional

import torch
import torch.nn as nn

from core_algorithm.lif_layer import LIFNeuronLayer

class LIFSMNISTNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 10,
        mode: str = "sfnn",
        warmup_steps: int = 0,
        neuron_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        recurrent = mode.lower() == "srnn"
        self.warmup_steps = warmup_steps
        self.layers = nn.ModuleList()
        in_dim = input_size
        nkw = neuron_kwargs or {}
        for h in hidden_sizes:
            self.layers.append(
                LIFNeuronLayer(
                    input_size=in_dim,
                    hidden_size=h,
                    recurrent=recurrent,
                    **nkw,
                )
            )
            in_dim = h
        self.readout = nn.Linear(hidden_sizes[-1], num_classes)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.constant_(self.readout.bias, 0.0)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out, batch_first=True)
        logits_t = self.readout(out)
        if logits_t.size(1) <= self.warmup_steps:
            logits = logits_t.mean(dim=1)
        else:
            logits = logits_t[:, self.warmup_steps:, :].mean(dim=1)
        return logits

    def init_states(self, batch_size, device=None, dtype=None):
        return [layer.init_state(batch_size, device=device, dtype=dtype) for layer in self.layers]

    @staticmethod
    def detach_states(states):
        return [{k: v.detach() for k, v in state.items()} for state in states]

    def mean_spike_rate(self, x):
        out = x
        rates = []
        for layer in self.layers:
            out = layer(out, batch_first=True)
            rates.append(out.mean())
        return torch.stack(rates).mean()

