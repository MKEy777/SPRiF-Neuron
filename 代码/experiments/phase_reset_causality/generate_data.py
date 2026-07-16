from __future__ import annotations

import argparse
import itertools

from phase_causal.config import load_config
from phase_causal.data import PhaseTrajectoryDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate and cache all datasets for phase-reset-causality experiments"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = args.cache_dir or f"{cfg.output_dir}/cached_data"

    seeds = set(cfg.sprif_seeds) | set(cfg.baseline_seeds)
    omegas = cfg.task.omega_choices

    specs = []
    for seed in seeds:
        specs.append((cfg.training.train_samples, 10_000 + seed, None))
        specs.append((cfg.training.val_samples, 20_000 + seed, None))
    specs.append((cfg.training.test_samples, cfg.evaluation.test_seed, None))
    for period in cfg.evaluation.ood_periods:
        specs.append((cfg.training.test_samples, cfg.evaluation.test_seed + 1, (2.0 * 3.141592653589793 / period,)))
    for pi, period in enumerate(cfg.task.omega_periods):
        specs.append((cfg.training.test_samples, cfg.evaluation.test_seed + 100_001 + pi, (2.0 * 3.141592653589793 / period,)))

    for n_samples, seed, omega_choices in specs:
        label = f"n={n_samples} seed={seed}"
        if omega_choices:
            label += f" omega={[round(w, 4) for w in omega_choices]}"
        print(f"Generating {label} ...")
        _ = PhaseTrajectoryDataset(
            n_samples, cfg.task, seed,
            omega_choices=omega_choices,
            cache_dir=cache_dir,
        )

    print(f"All datasets cached under: {cache_dir}")


if __name__ == "__main__":
    main()
