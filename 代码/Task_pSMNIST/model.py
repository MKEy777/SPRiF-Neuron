"""pSMNIST-specific network definition using SPRiF neuron layers."""

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from core_algorithm.sprif_layer import SPRiFNeuronLayer


class PermutedMNIST(Dataset):
    """Permuted MNIST dataset with fixed permutation."""

    def __init__(self, mnist: Dataset, perm: torch.Tensor):
        self.mnist = mnist
        self.perm = perm.long()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        unrolled = img.reshape(-1)
        permuted = unrolled[self.perm]
        permuted = permuted.reshape(-1, 1)
        return permuted, label


class SPRiFpSMNISTNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 10,
        mode: str = "sfnn",
        warmup_steps: int = 0,
    ):
        super().__init__()
        recurrent = mode.lower() == "srnn"
        self.warmup_steps = warmup_steps
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h in hidden_sizes:
            self.layers.append(
                SPRiFNeuronLayer(
                    input_size=in_dim,
                    hidden_size=h,
                    recurrent=recurrent,
                    threshold=1.0,
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
            logits = logits_t[:, self.warmup_steps :, :].mean(dim=1)
        return logits

    def mean_spike_rate(self, x):
        out = x
        rates = []
        for layer in self.layers:
            out = layer(out, batch_first=True)
            rates.append(out.mean())
        return torch.stack(rates).mean()
