from __future__ import annotations

import argparse

from cli_utils import device_from_name, output_root
from phase_causal.config import load_config
from phase_causal.pipeline import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one trained phase-reset-causality seed")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=("sprif", "lif", "asrnn", "brf"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = evaluate_checkpoint(
        args.model, args.seed, cfg, output_root(cfg, args.output),
        device_from_name(args.device), run_sensitivity=not args.no_sensitivity,
    )
    print(path)


if __name__ == "__main__":
    main()

