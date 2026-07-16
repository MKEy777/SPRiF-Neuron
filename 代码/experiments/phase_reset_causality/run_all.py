from __future__ import annotations

import argparse

from cli_utils import device_from_name, output_root
from phase_causal.config import load_config
from phase_causal.pipeline import aggregate_results
from phase_causal.plotting import plot_results
from phase_causal.runner import run_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete phase-reset-causality experiment")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--models", nargs="+", choices=("sprif", "lif", "asrnn", "brf"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = output_root(cfg, args.output)
    device = device_from_name(args.device)
    models = tuple(args.models or cfg.models)
    cache_dir = args.cache_dir or f"{cfg.output_dir}/cached_data"
    run_models(
        models, cfg, root, device,
        skip_train=args.skip_train, run_sensitivity=not args.no_sensitivity,
        cache_dir=cache_dir,
    )
    aggregate_results(root, cfg.evaluation.bootstrap_samples)
    plot_results(root)
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
