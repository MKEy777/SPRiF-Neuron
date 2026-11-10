import torch
import torch.nn as nn
from core_algorithm.sprif_layer_ablation_b import SPRiFNeuronLayerAblationB

class SPRiFGSCNetAblationB(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes,
        num_classes: int,
        dropout: float,
        recurrent_flags,
        neuron_kwargs: dict,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        in_dim = input_size
        for h, rec in zip(hidden_sizes, recurrent_flags):
            self.layers.append(
                SPRiFNeuronLayerAblationB(
                    input_size=in_dim,
                    hidden_size=h,
                    recurrent=rec,
                    **neuron_kwargs,
                )
            )
            in_dim = h

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        out = x
        spike_rates = []

        for layer in self.layers:
            out = layer(out, batch_first=True)
            out = self.dropout(out)
            spike_rates.append(out.mean())

        pooled = out.mean(dim=1)
        log_probs = torch.log_softmax(self.classifier(pooled), dim=-1)
        return log_probs, {"spike_rate": torch.stack(spike_rates).mean()}

