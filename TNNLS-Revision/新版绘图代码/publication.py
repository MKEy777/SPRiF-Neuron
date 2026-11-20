"""Shared Matplotlib styles and mandatory rendered panel-alignment gate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = {"SPRiF": "#276b8c", "MEMORY_RESET5": "#a65328",
          "real": "#276b8c", "rotation": "#a65328", "membrane": "#5c4a85"}


def set_style(overrides=None):
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": .7, "axes.spines.right": False, "axes.spines.top": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    if overrides:
        matplotlib.rcParams.update(overrides)


def label(ax, letter):
    ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def save_figure(fig, base, *, formats=(".pdf", ".svg", ".png"), dpi=600):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(base.with_suffix(ext), dpi=dpi, facecolor="white")
    plt.close(fig)
