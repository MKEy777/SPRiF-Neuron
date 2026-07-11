import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default="results/all_metrics.json")
    p.add_argument("--output", default="results/figures"); args = p.parse_args()
    df = pd.DataFrame(json.loads(Path(args.input).read_text(encoding="utf-8")))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for model, group in df.groupby("model"):
        pivot = group.groupby(["delay_ms", "intervention_count"]).accuracy.mean().unstack()
        ax = pivot.plot(marker="o", title=f"SI-DMS robustness: {model}")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.figure.tight_layout()
        ax.figure.savefig(out / f"{model}_robustness.png", dpi=200); plt.close(ax.figure)


if __name__ == "__main__": main()
