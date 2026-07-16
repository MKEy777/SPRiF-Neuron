from __future__ import annotations

from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def output_root(config, override: str | None = None) -> Path:
    if override:
        return Path(override).resolve()
    configured = Path(config.output_dir)
    if configured.is_absolute():
        return configured
    return (EXPERIMENT_ROOT / configured).resolve()


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

