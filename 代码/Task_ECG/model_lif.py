"""ECG-specific network definition using LIF neuron layers."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from core_algorithm.lif_layer import LIFNeuronLayer


class LIFECGModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        mode: str = "srnn",
        neuron_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element.")

        if neuron_kwargs is None:
            neuron_kwargs = {}

        recurrent = mode.lower() == "srnn"

        self.layers = nn.ModuleList()
        in_features = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(
                LIFNeuronLayer(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    recurrent=recurrent,
                    **neuron_kwargs,
                )
            )
            in_features = hidden_size

        self.readout = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        out = x

        for layer in self.layers:
            out = layer(out, batch_first=True)   # [batch, time, hidden]

        logits = self.readout(out)               # [batch, time, classes]

        return logits.permute(0, 2, 1)           # [batch, classes, time]
