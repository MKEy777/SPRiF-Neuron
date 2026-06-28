"""ECG-specific network definition using SPRiF neuron layers."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from core_algorithm.sprif_layer import SPRiFNeuronLayer


class SPRiFECGModel(nn.Module):
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
                SPRiFNeuronLayer(
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


def build_neuron_kwargs(config: dict) -> dict:
    """Build neuron keyword arguments from a flat config dict."""
    kwargs: dict = {
        "threshold": config["neuron_threshold"],
        "init_std": config["neuron_init_std"],
        "bias": config.get("neuron_bias", False),
    }

    optional_keys = [
        "tau_alpha_range",
        "tau_rho_range",
        "tau_eta_range",
        "omega_range",
    ]

    for key in optional_keys:
        if key in config:
            kwargs[key] = config[key]

    return kwargs


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    """Compute summary statistics for a 1D or n-D tensor."""
    flat = t.detach().float().reshape(-1)

    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _collect_internal_stats(model: nn.Module) -> dict:
    """Collect per-layer and global spectral parameter statistics."""
    per_layer: list = []

    alpha_all: list = []
    rho_all: list = []
    omega_all: list = []
    eta_all: list = []
    lambda_all: list = []

    for layer_idx, layer in enumerate(model.layers):
        spectral = layer.get_spectral_parameters()
        lambda_reset = layer.lambda_reset.detach()

        layer_stats = {
            "layer_idx": layer_idx,
            "alpha": _tensor_stats(spectral["alpha"]),
            "rho": _tensor_stats(spectral["rho"]),
            "omega": _tensor_stats(spectral["omega"]),
            "eta": _tensor_stats(spectral["eta"]),
            "lambda_reset": _tensor_stats(lambda_reset),
        }

        per_layer.append(layer_stats)

        alpha_all.append(spectral["alpha"].reshape(-1))
        rho_all.append(spectral["rho"].reshape(-1))
        omega_all.append(spectral["omega"].reshape(-1))
        eta_all.append(spectral["eta"].reshape(-1))
        lambda_all.append(lambda_reset.reshape(-1))

    global_stats = {
        "alpha": _tensor_stats(torch.cat(alpha_all)),
        "rho": _tensor_stats(torch.cat(rho_all)),
        "omega": _tensor_stats(torch.cat(omega_all)),
        "eta": _tensor_stats(torch.cat(eta_all)),
        "lambda_reset": _tensor_stats(torch.cat(lambda_all)),
    }

    return {
        "global": global_stats,
        "per_layer": per_layer,
    }


__all__ = [
    "SPRiFECGModel",
    "build_neuron_kwargs",
    "_collect_internal_stats",
]
