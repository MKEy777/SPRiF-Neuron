from __future__ import annotations

import argparse

from cli_utils import output_root
from phase_causal.config import load_config
from phase_causal.pipeline import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate phase-reset-causality metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = aggregate_results(output_root(cfg, args.output), cfg.evaluation.bootstrap_samples)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()

