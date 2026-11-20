"""Standalone SI-DMS intervention-curve figure (manuscript Fig. 4)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import legacy_layout as layout  # noqa: E402

MODELS = ["SPRiF", "MEMORY_RESET5", "ASRNN", "BRF", "REAL_MODES3", "LIF"]
DRAW_ORDER = ["LIF", "BRF", "MEMORY_RESET5", "REAL_MODES3", "ASRNN", "SPRiF"]


def load_curve(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = {"training", "model", "intervention_count", "mean", "std"}
    if not needed.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(needed - set(df.columns))}")
    if set(df.training) != {"Clean", "Intervention"}:
        raise ValueError("expected Clean and Intervention rows")
    if set(df.model) != set(MODELS):
        raise ValueError("expected the six manuscript models")
    for col in ("intervention_count", "mean", "std"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"non-finite values in {col}")
    return df


def draw_panel(ax, data, title, handles):
    k_vals = sorted(data.intervention_count.unique())
    if 16 not in k_vals:
        raise ValueError("K=16 marks the beyond-training region")
    x = np.arange(len(k_vals))
    ax.axvspan(x[k_vals.index(16)] - .5, x[-1] + .5, color="#f4f4f4", zorder=-4)
    ax.axhline(.5, color="#9e9e9e", ls=":", lw=.7, zorder=0)
    for model in DRAW_ORDER:
        sub = data[data.model == model].sort_values("intervention_count")
        if list(sub.intervention_count) != k_vals:
            raise ValueError(f"{model}: incomplete K grid")
        mu = sub["mean"].to_numpy()
        sd = sub["std"].to_numpy()
        emph = model in {"SPRiF", "ASRNN"}
        ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=layout.COLORS[model], alpha=.075, lw=0)
        line, = ax.plot(x, mu, color=layout.COLORS[model], marker=layout.MARKERS[model],
                        ls=layout.LINESTYLES[model], ms=4.2 if emph else 3.7,
                        lw=1.55 if emph else .95,
                        markerfacecolor="white" if model == "REAL_MODES3" else layout.COLORS[model],
                        markeredgewidth=.9, zorder=4 if emph else 2, label=model)
        handles.setdefault(model, line)
    ax.set(xlim=(-.35, x[-1] + .35), ylim=(.45, 1.015), xticks=x, xticklabels=k_vals,
           yticks=[.5, .6, .7, .8, .9, 1.0], xlabel="Intervention count $K$")
    ax.set_title(title, loc="left", fontsize=7, pad=3)
    ax.grid(axis="y", ls=":", lw=.55, color="#d5d5d5")
    ax.tick_params(labelsize=6.5, length=2, pad=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=BASE / "01和02重画_20261108_第二次" / "绘图数据" / "干预曲线_均值及标准差.csv")
    parser.add_argument("--output", type=Path,
                        default=BASE / "输出图片" / "sidms_intervention_curves")
    args = parser.parse_args()

    df = load_curve(args.input)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": .7, "legend.frameon": False})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    fig.subplots_adjust(left=.072, right=.99, top=.883, bottom=.205, wspace=.16)
    handles = {}
    draw_panel(axes[0], df[df.training == "Clean"], "(a) Clean training", handles)
    draw_panel(axes[1], df[df.training == "Intervention"], "(b) Intervention training", handles)
    axes[0].set_ylabel("Accuracy")

    fig.legend([handles[m] for m in MODELS], MODELS, ncol=6, loc="lower center",
               bbox_to_anchor=(.5, .012), fontsize=6.2, handlelength=1.5,
               columnspacing=.9, handletextpad=.35, borderpad=0, labelspacing=.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output) + ".pdf", facecolor="white")
    fig.savefig(str(args.output) + ".png", dpi=600, facecolor="white")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
