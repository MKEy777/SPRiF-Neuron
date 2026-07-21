
from typing import List

import torch
import torch.nn as nn

from core_algorithm.sprif_layer import SPRiFNeuronLayer

class SPRiFSHDNet(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float,
        recurrent_flags,
        warmup_steps: int,
        neuron_kwargs: dict,
    ):
        super().__init__()

        assert len(hidden_sizes) == len(recurrent_flags), (
            f"hidden_sizes and recurrent_flags length mismatch: "
            f"{len(hidden_sizes)} vs {len(recurrent_flags)}"
        )

        self.warmup_steps = warmup_steps
        self.layers = nn.ModuleList()

        in_dim = input_size
        for h, recurrent in zip(hidden_sizes, recurrent_flags):
            self.layers.append(
                SPRiFNeuronLayer(
                    input_size=in_dim,
                    hidden_size=h,
                    recurrent=recurrent,
                    threshold=neuron_kwargs.get("threshold", 1.0),
                    init_std=neuron_kwargs.get("init_std", 0.2),
                )
            )
            in_dim = h

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.readout = SPRiFNeuronLayer(
            input_size=hidden_sizes[-1],
            hidden_size=num_classes,
            recurrent=False,
            threshold=neuron_kwargs.get("threshold", 1.0),
            init_std=neuron_kwargs.get("init_std", 0.2),
        )

    def forward(self, x):
        out = x
        spike_rates = []

        for layer in self.layers:
            out = layer(out, batch_first=True)
            out = self.dropout(out)
            spike_rates.append(out.mean())

        batch_size, time_steps, _ = out.shape

        readout_state = self.readout.init_state(
            batch_size, device=out.device, dtype=out.dtype,
        )
        readout_runtime = self.readout._precompute_runtime_params()

        logits = torch.zeros(
            batch_size, self.readout.hidden_size,
            device=out.device, dtype=out.dtype,
        )
        readout_spike_sum = torch.zeros((), device=out.device, dtype=out.dtype)

        use_all_steps = time_steps <= self.warmup_steps

        readout_projected = self.readout.input_linear(
            out.contiguous().view(batch_size * time_steps, out.size(-1))
        ).view(batch_size, time_steps, self.readout.hidden_size)

        for t in range(time_steps):
            readout_spike, readout_membrane, readout_state = self.readout.forward_step(
                out[:, t, :], readout_state, readout_runtime,
                input_current=readout_projected[:, t, :],
            )

            readout_spike_sum = readout_spike_sum + readout_spike.mean()

            if use_all_steps or t > self.warmup_steps:
                logits = logits + torch.nn.functional.softmax(readout_membrane, dim=-1)

        spike_rates.append(readout_spike_sum / max(time_steps, 1))

        aux = {"spike_rate": torch.stack(spike_rates).mean()}
        return logits, aux

