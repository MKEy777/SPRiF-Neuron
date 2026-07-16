from __future__ import annotations

import argparse

from cli_utils import output_root
from phase_causal.config import load_config
from phase_causal.plotting import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot phase-reset-causality results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for path in plot_results(output_root(cfg, args.output)).values():
        print(path)


if __name__ == "__main__":
    main()

