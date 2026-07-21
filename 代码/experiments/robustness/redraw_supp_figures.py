
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 9
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["legend.frameon"] = False

C_SPRIF = "#0F4D92"
C_ASRNN = "#B64342"
GRID = "#D8D8D8"

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(
    ROOT, "..", "..", "..", "experiment-design-20260606", "results"))

FREQ_ORDER = ["f1_0.01pi", "f2_0.05pi", "f3_0.10pi", "f4_0.25pi", "f5_0.50pi"]
FREQ_LABELS = ["0.01" + "$\\pi$", "0.05" + "$\\pi$", "0.10" + "$\\pi$",
               "0.25" + "$\\pi$", "0.50" + "$\\pi$"]
AMP_ORDER = ["low", "med", "high"]
AMP_TITLES = ["Low amplitude", "Medium amplitude", "High amplitude"]

COND_ORDER = [
    "clean", "additive_sigma_0.01", "additive_sigma_0.05", "additive_sigma_0.10",
    "subtractive_p_0.05", "subtractive_p_0.10", "subtractive_p_0.20", "mixed",
]
COND_LABELS = ["Clean", "Gauss\n$\\sigma$=0.01", "Gauss\n$\\sigma$=0.05",
               "Gauss\n$\\sigma$=0.10", "Zero\n$p$=0.05", "Zero\n$p$=0.10",
               "Zero\n$p$=0.20", "Mixed"]

def _add_panel_label(ax, label, x=-0.12, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")

def draw_noise():
    with open(os.path.join(RESULTS, "robustness_benchmark.json")) as f:
        data = json.load(f)

    by_ds = {ds: {} for ds in ["GSC", "QTDB"]}
    for r in data:
        by_ds[r["dataset"]][r["condition"]] = (
            r["SPRiF_accuracy"], r["ASRNN_accuracy"])

    datasets = ["GSC", "QTDB"]

    fig, axes = plt.subplots(len(datasets), 1, figsize=(5.8, 7.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bar_w = 0.36
    x = np.arange(len(COND_ORDER))

    group_spans = [(-0.5, 0.5, "#F2F2F2"), (0.5, 3.5, "#FFFFFF"),
                   (3.5, 6.5, "#F2F2F2"), (6.5, 7.5, "#FFFFFF")]

    panel = 0
    leg_handles = []
    for ax, ds in zip(axes, datasets):
        for x0, x1, col in group_spans:
            ax.axvspan(x0, x1, color=col, zorder=0)
        vals = [by_ds[ds][c] for c in COND_ORDER]
        sprif = [v[0] for v in vals]
        asrnn = [v[1] for v in vals]
        b1 = ax.bar(x - bar_w / 2, sprif, width=bar_w, color=C_SPRIF,
                    label="SPRiF" if panel == 0 else None, edgecolor="white",
                    linewidth=0.6, zorder=3)
        b2 = ax.bar(x + bar_w / 2, asrnn, width=bar_w, color=C_ASRNN,
                    label="ASRNN" if panel == 0 else None, edgecolor="white",
                    linewidth=0.6, zorder=3)
        if panel == 0:
            leg_handles = [b1, b2]

        for i in range(len(COND_ORDER)):
            ax.text(x[i] - bar_w / 2, sprif[i] + 0.010, f"{sprif[i]:.3f}",
                    ha="center", va="bottom", fontsize=8, color=C_SPRIF)
            ax.text(x[i] + bar_w / 2, asrnn[i] + 0.010, f"{asrnn[i]:.3f}",
                    ha="center", va="bottom", fontsize=8, color=C_ASRNN)

        clean_val = by_ds[ds]["clean"][0]
        ax.axhline(clean_val, color="#606060", linestyle=":", linewidth=1.2,
                   zorder=2)
        ax.text(len(COND_ORDER) - 0.45, clean_val + 0.004, "clean",
                ha="right", va="bottom", fontsize=7.5, color="#606060")
        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=10, linespacing=1.15)
        ax.set_xlim(-0.6, len(COND_ORDER) - 0.4)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Accuracy", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.7, zorder=1)
        _add_panel_label(ax, f"{chr(ord('a') + panel)}  {ds}")
        panel += 1

    fig.legend(handles=leg_handles, labels=["SPRiF", "ASRNN"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=10, frameon=False, handlelength=1.6)
    fig.tight_layout(pad=1.4)
    _save(fig, os.path.join(ROOT, "robustness_benchmark"))

def _save(fig, base):
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base}.(svg|pdf|png)")

if __name__ == "__main__":
    draw_noise()
    print("Done.")

