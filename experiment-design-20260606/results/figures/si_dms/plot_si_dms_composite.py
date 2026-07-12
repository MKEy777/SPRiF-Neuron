"""Figure 1: Controlled spike-intervention results (SI-DMS).

Layout: 1x2 asymmetric.
- Panel (a) — wide: accuracy vs intervention count K for 6 models
  (3 SPRiF mechanism variants + 3 external baselines), 3-seed mean +/- std.
- Panel (b) — two stacked heatmaps (sprif_full top, sprif_merged bottom)
  over delay (ms) x K, shared colorbar.

Inputs: results/{model}/seed_{N}/eval_metrics.json.
Outputs: si_dms_mechanism_composite.{pdf,svg,png}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

RESULTS_DIR = (
    Path(__file__).resolve().parents[4]
    / "Spike-Intervention Delayed Match-to-Sample\uFF08SI-DMS\uFF09"
    / "results"
)
OUT_DIR = Path(__file__).resolve().parent

SPRIF_MODELS = ["sprif_full", "sprif_merged", "sprif_lambda0"]
EXTERNAL_MODELS = ["lif", "asrnn", "brf"]
ALL_MODELS = SPRIF_MODELS + EXTERNAL_MODELS

COLORS = {
    "sprif_full": "#1f77b4",
    "sprif_merged": "#d62728",
    "sprif_lambda0": "#ff7f0e",
    "lif": "#bdbdbd",
    "asrnn": "#757575",
    "brf": "#424242",
}
MARKERS = {
    "sprif_full": "o",
    "sprif_merged": "v",
    "sprif_lambda0": "s",
    "lif": "D",
    "asrnn": "P",
    "brf": "X",
}
LINESTYLES = {m: "-" for m in SPRIF_MODELS}
LINESTYLES.update({m: "--" for m in EXTERNAL_MODELS})
LABELS = {
    "sprif_full": "SPRiF full",
    "sprif_merged": "SPRiF merged",
    "sprif_lambda0": r"SPRiF $\lambda{=}0$",
    "lif": "LIF",
    "asrnn": "ASRNN",
    "brf": "BRF",
}


def load_data():
    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if model_dir.name not in ALL_MODELS:
            continue
        for seed_dir in sorted(model_dir.glob("seed_*")):
            eval_path = seed_dir / "eval_metrics.json"
            if not eval_path.exists():
                continue
            with open(eval_path) as f:
                entries = json.load(f)
            for e in entries:
                rows.append({
                    "model": e["model"],
                    "seed": e["seed"],
                    "delay_ms": e["delay_ms"],
                    "intervention_count": e["intervention_count"],
                    "accuracy": float(e["accuracy"]),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No eval_metrics.json found under {RESULTS_DIR}")
    return df


def cell_text_color(val, vmin=0.40, vmax=1.00):
    t = (val - vmin) / (vmax - vmin + 1e-12)
    return "white" if t > 0.55 else "black"


def per_k_seed_stats(df, model, k_vals):
    """For one model: per-K mean and per-K seed-noise std.

    mean(K)  = mean over (3 seeds x 5 delays) of accuracy
    std(K)   = mean over the 5 delays of per-(delay) seed-std
               (i.e. the seed-noise level; the delay-driven spread
               is shown in panel (b), not as error bars here).
    """
    sub = df[df["model"] == model]
    means, stds = [], []
    for k in k_vals:
        per_delay_seed_std = (
            sub[sub["intervention_count"] == k]
            .groupby("delay_ms")["accuracy"].std().fillna(0)
        )
        per_k_mean = sub.loc[sub["intervention_count"] == k, "accuracy"].mean()
        means.append(per_k_mean)
        stds.append(per_delay_seed_std.mean())
    return np.array(means), np.array(stds)


def plot_panel_a(ax, df):
    k_vals = sorted(df["intervention_count"].unique())

    ax.axhline(0.5, color="#9e9e9e", linestyle=":", linewidth=0.8, zorder=1)

    handles, labels = [], []
    for m in SPRIF_MODELS + EXTERNAL_MODELS:
        mu, sd = per_k_seed_stats(df, m, k_vals)
        h = ax.errorbar(
            k_vals, mu, yerr=sd,
            color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
            markersize=5, linewidth=1.4, capsize=2.2, capthick=0.9,
            elinewidth=0.9, label=LABELS[m],
            zorder=3 if m in SPRIF_MODELS else 2,
            alpha=1.0 if m in SPRIF_MODELS else 0.95,
        )
        handles.append(h); labels.append(LABELS[m])

    ax.set_xlabel("Intervention Count ($K$)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.40, 1.02)
    ax.set_xticks(k_vals)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", linestyle=":", color="#d9d9d9", linewidth=0.6, zorder=0)

    ax.text(-0.13, 1.04, "(a)", transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="bottom", ha="left")


def plot_panel_b(fig, gs_b, df, cax):
    """3x2 grid of small heatmaps, one per model (delay x K).

    Rows are grouped by mechanism tier:
      row 1: SPRiF with full mechanism (full, lambda=0)
      row 2: degraded mechanism (merged, brf)
      row 3: pure baselines (lif, asrnn)
    """
    MODEL_ORDER = [
        ("sprif_full", LABELS["sprif_full"], "Full SPRiF"),
        ("sprif_lambda0", LABELS["sprif_lambda0"], "Full SPRiF"),
        ("sprif_merged", LABELS["sprif_merged"], "Degraded"),
        ("brf", LABELS["brf"], "Degraded"),
        ("lif", LABELS["lif"], "Baselines"),
        ("asrnn", LABELS["asrnn"], "Baselines"),
    ]

    delays = sorted(df["delay_ms"].unique())
    k_vals = sorted(df["intervention_count"].unique())
    n_delays, n_k = len(delays), len(k_vals)

    vmin, vmax = 0.40, 1.00
    cmap = plt.get_cmap("YlGn").copy()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    axes_b = []
    last_im = None
    for i, (model, label, tier) in enumerate(MODEL_ORDER):
        r, c = i // 2, i % 2
        ax = fig.add_subplot(gs_b[r, c])
        axes_b.append(ax)

        sub = df[df["model"] == model]
        pivot = sub.groupby(["delay_ms", "intervention_count"])["accuracy"].mean().unstack()
        pivot = pivot.reindex(index=delays, columns=k_vals)
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       interpolation="nearest", norm=norm)
        last_im = im

        ax.set_xticks(range(n_k))
        ax.set_yticks(range(n_delays))
        ax.set_xticklabels(k_vals if r == 2 else [], fontsize=6)
        ax.set_yticklabels(delays if c == 0 else [], fontsize=6)
        ax.tick_params(axis="both", labelsize=6, length=1.5)

        mean_val = pivot.values.mean()
        title_color = COLORS[model] if model in COLORS else "#333"
        ax.set_title(label, fontsize=7.5, fontweight="bold",
                     color=title_color, pad=2, loc="left")
        ax.text(0.98, 0.04, f"\u03bc={mean_val:.2f}",
                transform=ax.transAxes, fontsize=6.5, color="#666",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor="none", pad=1.2))

    axes_b[0].text(-0.18, 1.22, "(b)", transform=axes_b[0].transAxes,
                   fontweight="bold", fontsize=9, va="bottom", ha="left")
    axes_b[0].set_ylabel("Delay (ms)", fontsize=6.5)
    axes_b[2].set_ylabel("Delay (ms)", fontsize=6.5)
    axes_b[4].set_ylabel("Delay (ms)", fontsize=6.5)
    axes_b[4].set_xlabel("$K$", fontsize=6.5)
    axes_b[5].set_xlabel("$K$", fontsize=6.5)

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)


def main():
    df = load_data()
    fig = plt.figure(figsize=(7.0, 3.2))
    gs_main = fig.add_gridspec(
        1, 3,
        width_ratios=[1.15, 1.50, 0.045],
        wspace=0.28,
        left=0.075, right=0.96, top=0.93, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs_main[0, 0])
    gs_b = gs_main[0, 1].subgridspec(3, 2, hspace=0.30, wspace=0.20)
    cax = fig.add_subplot(gs_main[0, 2])

    plot_panel_a(ax_a, df)
    plot_panel_b(fig, gs_b, df, cax)

    handles, labels = ax_a.get_legend_handles_labels()
    ax_a_pos = ax_a.get_position()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(ax_a_pos.x0 + ax_a_pos.width / 2,
                                                    0.005),
               ncol=6, frameon=False, fontsize=6.5,
               handlelength=1.4, columnspacing=1.1, handletextpad=0.4,
               borderpad=0.3)

    for name in ["si_dms_mechanism_composite"]:
        fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
