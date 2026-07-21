import os
import sys
from typing import List

import torch
import torch.nn as nn

from config import SMNIST_DIR, GSC_DIR

import importlib.util

def _load_class(module_alias: str, file_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_alias, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

_sprif_path = os.path.join(SMNIST_DIR, "core_algorithm", "sprif_layer.py")
SPRiFNeuronLayer = _load_class("sprif_layer_smnist", _sprif_path, "SPRiFNeuronLayer")

if GSC_DIR in sys.path:
    sys.path.remove(GSC_DIR)
sys.path.insert(0, GSC_DIR)
_lif_path = os.path.join(GSC_DIR, "core_algorithm", "lif_layer.py")
LIFNeuronLayer = _load_class("lif_layer_gsc", _lif_path, "LIFNeuronLayer")

def _build_layer(neuron_type: str, in_dim: int, hidden: int, recurrent: bool):
    if neuron_type == "sprif":
        return SPRiFNeuronLayer(
            input_size=in_dim,
            hidden_size=hidden,
            recurrent=recurrent,
            threshold=1.0,
        )
    elif neuron_type == "lif":
        return LIFNeuronLayer(
            input_size=in_dim,
            hidden_size=hidden,
            recurrent=recurrent,
            threshold=1.0,
        )
    raise ValueError(f"unknown neuron_type: {neuron_type}")

class SMNISTNet(nn.Module):

    def __init__(
        self,
        neuron_type: str,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 10,
        mode: str = "srnn",
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.neuron_type = neuron_type
        recurrent = mode.lower() == "srnn"
        self.warmup_steps = warmup_steps
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h in hidden_sizes:
            self.layers.append(_build_layer(neuron_type, in_dim, h, recurrent))
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
        return [layer.init_state(batch_size, device=device, dtype=dtype)
                for layer in self.layers]

    @staticmethod
    def detach_states(states):
        return [{k: v.detach() for k, v in state.items()} for state in states]

def build_model(neuron_type: str, cfg) -> SMNISTNet:
    return SMNISTNet(
        neuron_type=neuron_type,
        input_size=cfg.INPUT_SIZE,
        hidden_sizes=list(cfg.HIDDEN_SIZES),
        num_classes=cfg.NUM_CLASSES,
        mode=cfg.MODE,
        warmup_steps=cfg.WARMUP_STEPS,
    )

__all__ = ["SMNISTNet", "build_model", "SPRiFNeuronLayer", "LIFNeuronLayer"]

