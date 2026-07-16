from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TaskConfig


def generate_sample(
    phi: float,
    omega: float,
    cfg: TaskConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = np.zeros((cfg.total_steps, cfg.input_size), dtype=np.float32)
    phase_offsets = 2.0 * math.pi * np.arange(cfg.phase_channels) / cfg.phase_channels
    cue_t = np.arange(cfg.cue_steps, dtype=np.float32)
    rates = cfg.base_rate_hz + cfg.modulation_hz * np.cos(
        omega * cue_t[:, None] + phi - phase_offsets[None, :]
    )
    probabilities = np.clip(rates, 0.0, 1000.0 / cfg.dt_ms) * cfg.dt_ms / 1000.0
    x[: cfg.cue_steps, : cfg.phase_channels] = (
        rng.random(probabilities.shape) < probabilities
    ).astype(np.float32)
    x[0, cfg.phase_channels] = 1.0
    x[cfg.cue_steps, cfg.phase_channels + 1] = 1.0

    t = np.arange(cfg.total_steps, dtype=np.float32)
    phase = phi + omega * t
    target = np.stack((np.cos(phase), np.sin(phase)), axis=-1).astype(np.float32)
    return x, target, {"phi": phi, "omega": omega}


def _cache_path(cache_dir: str, n_samples: int, seed: int,
                omega_choices: tuple[float, ...]) -> Path:
    import hashlib
    from pathlib import Path
    key = hashlib.md5(
        str(sorted(omega_choices)).encode()
    ).hexdigest()[:8]
    return Path(cache_dir) / f"data_{n_samples}_seed{seed}_{key}.pt"


class PhaseTrajectoryDataset(Dataset):
    def __init__(self, n_samples: int, cfg: TaskConfig, seed: int,
                 omega_choices: tuple[float, ...] | None = None,
                 cache_dir: str | None = None):
        if cache_dir is not None:
            cache_path = _cache_path(cache_dir, n_samples, seed, omega_choices or cfg.omega_choices)
            if cache_path.exists():
                data = torch.load(cache_path, weights_only=False)
                self.x = data["x"]
                self.target = data["target"]
                self.phi = data["phi"]
                self.omega = data["omega"]
                return
        rng = np.random.default_rng(seed)
        choices = omega_choices or cfg.omega_choices
        xs, targets, phis, omegas = [], [], [], []
        for _ in range(n_samples):
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            omega = float(rng.choice(choices))
            x, target, _ = generate_sample(
                phi, omega, cfg, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            )
            xs.append(torch.from_numpy(x))
            targets.append(torch.from_numpy(target))
            phis.append(torch.tensor(phi, dtype=torch.float32))
            omegas.append(torch.tensor(omega, dtype=torch.float32))
        self.x = torch.stack(xs)
        self.target = torch.stack(targets)
        self.phi = torch.stack(phis)
        self.omega = torch.stack(omegas)
        if cache_dir is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"x": self.x, "target": self.target, "phi": self.phi, "omega": self.omega},
                       cache_path)

    def __len__(self) -> int:
        return len(self.phi)

    def __getitem__(self, index: int):
        return self.x[index], self.target[index], self.phi[index], self.omega[index]
