"""Standalone SI-DMS heatmap figure (manuscript Fig. 5)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import legacy_layout as layout  # noqa: E402

GROUPS = [["SPRiF", "MEMORY_RESET5", "REAL_MODES3"], ["ASRNN", "BRF", "LIF"]]
DELAYS = [200, 400, 800, 1600, 2500]
KS = [0, 1, 2, 4, 8, 16, 32, 40]


def load_grid(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = {"training", "model", "delay_ms", "intervention_count", "mean"}
    if not needed.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(needed - set(df.columns))}")
    if set(df.training) != {"Clean", "Intervention"}:
        raise ValueError("expected Clean and Intervention rows")
    models = [m for group in GROUPS for m in group]
    if set(df.model) != set(models):
        raise ValueError("expected the six manuscript models")
    for col in ("delay_ms", "intervention_count", "mean"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"non-finite values in {col}")
    for (training, model), group in df.groupby(["training", "model"]):
        if set(group.delay_ms) != set(DELAYS) or set(group.intervention_count) != set(KS):
            raise ValueError(f"{training}/{model}: incomplete delay x K grid")
        if group.duplicated(["delay_ms", "intervention_count"]).any():
            raise ValueError(f"{training}/{model}: duplicate cells")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=BASE / "01和02重画_20261108_第二次" / "绘图数据" / "热图_五种子均值.csv")
    parser.add_argument("--output", type=Path,
                        default=BASE / "输出图片" / "sidms_heatmaps")
    args = parser.parse_args()

    df = load_grid(args.input)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
                         "axes.linewidth": .5, "legend.frameon": False})

    fig = plt.figure(figsize=(7.15, 2.70))
    outer = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .07],
                             left=.062, right=.936, top=.808, bottom=.181,
                             wspace=.18, hspace=.67)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    norm = Normalize(.4, 1.0)
    image = None
    axes = {}
    for r, group in enumerate(GROUPS):
        for c, model in enumerate(group):
            inner = outer[r, c].subgridspec(1, 2, wspace=.14)
            for j, training in enumerate(("Clean", "Intervention")):
                ax = fig.add_subplot(inner[0, j])
                axes[(r, c, j)] = ax
                sub = df[(df.model == model) & (df.training == training)]
                pivot = (sub.pivot(index="delay_ms", columns="intervention_count", values="mean")
                         .reindex(index=DELAYS, columns=KS))
                image = ax.imshow(pivot, aspect="auto", cmap=cmap, norm=norm,
                                  interpolation="nearest")
                ax.set_xticks(np.arange(len(KS)))
                ax.set_yticks(np.arange(len(DELAYS)))
                if r == 1:
                    ax.set_xticklabels(KS, fontsize=6.2)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_yticklabels(DELAYS, fontsize=6.2)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(length=1.2, pad=1)
                ax.set_title(training, fontsize=6.4, pad=2)
            left = axes[(r, c, 0)].get_position()
            right = axes[(r, c, 1)].get_position()
            fig.text((left.x0 + right.x1) / 2, left.y1 + .10, model, ha="center", va="center",
                     fontsize=7.0, color=layout.COLORS[model],
                     fontweight="bold" if model in {"SPRiF", "ASRNN"} else "normal")

    cax = fig.add_subplot(outer[:, 3])
    bar = fig.colorbar(image, cax=cax, ticks=[.4, .5, .6, .7, .8, .9, 1.0])
    bar.set_label("Accuracy", fontsize=6.7)
    bar.ax.tick_params(labelsize=6.2, length=2, pad=1)
    bar.outline.set_linewidth(.5)

    fig.text(.5, .035, "Intervention count $K$", ha="center", va="center", fontsize=7)
    fig.text(.013, .505, "Delay (ms)", ha="center", va="center", rotation=90, fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output) + ".pdf", facecolor="white")
    fig.savefig(str(args.output) + ".png", dpi=600, facecolor="white")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
