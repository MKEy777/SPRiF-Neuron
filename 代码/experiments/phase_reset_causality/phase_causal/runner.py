from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .pipeline import evaluate_checkpoint, train_model_seed


def run_models(
    model_names: tuple[str, ...],
    cfg: ExperimentConfig,
    output_root: str | Path,
    device: torch.device,
    *,
    skip_train: bool = False,
    run_sensitivity: bool = True,
    seed: int = 1,
    cache_dir: str | None = None,
) -> None:
    """Train and evaluate a disjoint model shard without aggregating results."""
    root = Path(output_root)
    for model_name in model_names:
        for seed in (seed,):
            print(f"[{model_name} seed={seed}] device={device}")
            if not skip_train:
                train_model_seed(model_name, seed, cfg, root, device, cache_dir=cache_dir)
            evaluate_checkpoint(
                model_name, seed, cfg, root, device,
                run_sensitivity=(run_sensitivity and model_name == "sprif"),
                cache_dir=cache_dir,
            )
