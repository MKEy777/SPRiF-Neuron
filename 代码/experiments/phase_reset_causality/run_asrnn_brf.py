from __future__ import annotations

import argparse

from cli_utils import device_from_name, output_root
from phase_causal.config import load_config
from phase_causal.runner import run_models


MODEL_NAMES = ("asrnn", "brf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ASRNN and BRF phase-reset shard"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = output_root(cfg, args.output)
    cache_dir = args.cache_dir or f"{cfg.output_dir}/cached_data"
    run_models(
        MODEL_NAMES, cfg, root, device_from_name(args.device),
        skip_train=args.skip_train, run_sensitivity=not args.no_sensitivity,
        seed=args.seed, cache_dir=cache_dir,
    )
    print(f"Shard complete: {root}")


if __name__ == "__main__":
    main()
