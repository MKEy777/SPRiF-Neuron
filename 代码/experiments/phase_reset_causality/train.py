from __future__ import annotations

import argparse

from cli_utils import device_from_name, output_root
from phase_causal.config import load_config
from phase_causal.pipeline import train_model_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one phase-reset-causality model seed")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=("sprif", "lif", "asrnn", "brf"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = train_model_seed(
        args.model, args.seed, cfg, output_root(cfg, args.output), device_from_name(args.device)
    )
    print(path)


if __name__ == "__main__":
    main()

