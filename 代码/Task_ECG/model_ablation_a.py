
from typing import List

import torch
import torch.nn as nn

from core_algorithm.sprif_layer_ablation_a import SPRiFNeuronLayerAblationA

class SPRiFECGModelAblationA(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        mode: str = "srnn",
        neuron_kwargs: dict = None,
    ) -> None:
        super().__init__()
        if neuron_kwargs is None:
            neuron_kwargs = {}
        recurrent = mode.lower() == "srnn"
        self.layers = nn.ModuleList()
        in_features = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(
                SPRiFNeuronLayerAblationA(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    recurrent=recurrent,
                    **neuron_kwargs,
                )
            )
            in_features = hidden_size
        self.readout = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out, batch_first=True)
        logits = self.readout(out)
        return logits.permute(0, 2, 1)

