"""Redraw Figure 2 v4 — redesigned panels b and c to better highlight conclusions.

Conclusions to highlight:
- Panel b: The slow-state phase survives probe/perturbation windows
  (the oscillatory slow state is NOT reset by external perturbations).
- Panel c: Learned reset directions [1, lambda] are diverse and task-heterogeneous;
  they do not collapse to a single shared direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

# ── Nature-style palette (NMI pastel inspired) ──────────────────────────────
TEAL = "#42949E"
BLUE = "#0F4D92"
BLUE_MID = "#3775BA"
ORANGE = "#E28E2C"
PURPLE = "#9A4D8E"
RED = "#B64342"
GRAY = "#606060"
GRAY_LIGHT = "#D8D8D8"
SPIKE_BAND = "#FCEAEA"
PROBE_BAND = "#FDE6CC"

TASK_COLORS = {"ECG": BLUE_MID, "GSC": PURPLE, "pSMNIST": ORANGE}
DISPLAY_TASK = {"ECG": "QTDB", "GSC": "GSC", "pSMNIST": "pSMNIST"}


def _set_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
    })


def _read_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(set(required) - set(source.files))
        if missing:
            raise ValueError(f"{path.name} missing keys: {', '.join(missing)}")
        return {key: source[key].copy() for key in source.files}


def _panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", fontfamily="sans-serif")


def _clean_grid(ax) -> None:
    ax.grid(True, color=GRAY_LIGHT, linewidth=0.3, alpha=0.6)
    ax.tick_params(length=2.5, width=0.4)


def _corr(frame: pd.DataFrame, x: str, y: str) -> float:
    return float(np.corrcoef(frame[x].to_numpy(float), frame[y].to_numpy(float))[0, 1])


def load_source_data(repo_root: Path) -> dict:
    figures = repo_root / "experiment-design-20260606" / "results" / "figures"
    real = _read_npz(
        figures / "trajectory_analysis" / "trajectory_data.npz",
        ["layer1_slow", "layer1_membrane", "layer1_spikes",
         "highlight_spike_time", "highlight_neuron"],
    )
    legacy = repo_root / "experiment-design-20260606" / "legacy"
    controlled = _read_npz(
        legacy / "trajectory_visualization" / "trajectory_data_phi1.npz",
        ["sprif_x_t", "sprif_u_pre", "sprif_u_post", "sprif_spike", "sprif_readout",
         "probe_mask", "spectral_omega", "omega", "phi"],
    )
    reset_path = figures / "reset_analysis" / "lambda_stats.csv"
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    reset = pd.read_csv(reset_path)
    required_cols = {"task", "lambda_reset", "firing_rate", "alpha", "omega"}
    missing = sorted(required_cols - set(reset.columns))
    if missing:
        raise ValueError(f"lambda_stats.csv missing columns: {', '.join(missing)}")
    return {"real_trajectory": real, "controlled_trajectory": controlled, "reset_stats": reset}


def build_mechanism_figure(data: dict):
    _set_style()
    fig = plt.figure(figsize=(7.2, 3.8))
    # Width ratios: panel a (hero) and d/e share width; b/c (mechanism) get more.
    outer = fig.add_gridspec(
        2, 3, left=0.06, right=0.99, bottom=0.13, top=0.95,
        hspace=0.55, wspace=0.5,
        width_ratios=[4, 4, 4],
    )

    # Panel a: 3 stacked rows (slow state, membrane, spike raster)
    gs_a = outer[0:2, 0].subgridspec(
        3, 1, height_ratios=[1.2, 1.0, 0.32], hspace=0.35
    )
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1], sharex=ax_a1)
    ax_a3 = fig.add_subplot(gs_a[2], sharex=ax_a1)

    # Panel b: 2 stacked sub-panels (phase plane + phase angle vs time)
    # Give the 2D phase plane (b1) more vertical room than the time series (b2)
    gs_b = outer[0, 1].subgridspec(2, 1, height_ratios=[1.5, 0.7], hspace=0.7)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])

    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[0, 2])
    ax_e = fig.add_subplot(outer[1, 2])

    # ── Panel (a): Recorded pSMNIST spike ────────────────────────────────
    real = data["real_trajectory"]
    spike_t = int(real["highlight_spike_time"])
    neuron = int(real["highlight_neuron"])
    start, end = spike_t - 15, spike_t + 16
    t = np.arange(start, end) - spike_t
    slow = real["layer1_slow"][start:end, neuron]
    membrane = real["layer1_membrane"][start:end, neuron]
    spikes = real["layer1_spikes"][start:end, neuron] > 0.5
    mem_rel = membrane - membrane[0]

    ax_a1.axvspan(-0.5, 0.5, color=SPIKE_BAND, alpha=0.7, zorder=0)
    ax_a2.axvspan(-0.5, 0.5, color=SPIKE_BAND, alpha=0.7, zorder=0)

    colors_slow = [TEAL, ORANGE, PURPLE]
    labels_slow = [r"$x^{\mathrm{real}}$", r"$x^{\mathrm{osc}}_1$", r"$x^{\mathrm{osc}}_2$"]
    for idx, (color, label) in enumerate(zip(colors_slow, labels_slow)):
        offset = idx * 0.04
        y = slow[:, idx] + offset
        ax_a1.plot(t, y, color=color, label=label, linewidth=1.4, zorder=3)
        if idx == 0:
            ax_a1.fill_between(t, offset, y, color=color, alpha=0.12, zorder=2)
    ax_a1.axvline(0, color=RED, linestyle="--", linewidth=0.9, alpha=0.8, zorder=4)
    ax_a1.set_ylabel("slow state (offset)")
    ax_a1.set_title("Recorded pSMNIST spike")
    ax_a1.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.99, 0.98),
                 handlelength=1.2, fontsize=5.2, frameon=True,
                 framealpha=0.85, edgecolor=GRAY_LIGHT)
    _panel_label(ax_a1, "a")

    ax_a2.plot(t, mem_rel, color=BLUE_MID, linewidth=1.4)
    spike_idx = np.where(spikes)[0]
    ax_a2.scatter(t[spike_idx], mem_rel[spike_idx], color=RED,
                  marker="v", s=18, zorder=3, clip_on=False, edgecolor="white",
                  linewidth=0.3)
    ax_a2.axvline(0, color=RED, linestyle="--", linewidth=0.9, alpha=0.8, zorder=4)
    ax_a2.axhline(0, color=GRAY_LIGHT, linewidth=0.5)
    ax_a2.set_ylabel(r"Fast $u^0$ (rel.)")

    ax_a3.vlines(t[spikes], 0, 1, color=RED, linewidth=1.2)
    ax_a3.axvline(0, color=RED, linestyle="--", linewidth=0.7, alpha=0.5)
    ax_a3.set_xlim(t[0], t[-1])
    ax_a3.set_ylim(0, 1)
    ax_a3.set_yticks([])
    ax_a3.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax_a3.set_xlabel("Steps relative to spike")
    ax_a3.spines["left"].set_visible(False)

    for ax in [ax_a1, ax_a2, ax_a3]:
        _clean_grid(ax)
    ax_a3.grid(False)
    plt.setp(ax_a1.get_xticklabels(), visible=False)
    plt.setp(ax_a2.get_xticklabels(), visible=False)

    # ── Panel (b): Controlled phase memory ───────────────────────────────
    # Goal: show that the slow-state oscillation SURVIVES the probe/perturbation windows.
    # Use two sub-panels:
    #   b1: 2D phase plane (the iconic slow-state trajectory)
    #   b2: x_osc_1(t) and x_osc_2(t) time series with probe windows
    #       The two components are smooth sinusoids; if probes broke the slow state
    #       we would see discontinuities / kinks in either trace inside the
    #       orange probe bands.  Direct visual evidence of phase memory.
    controlled = data["controlled_trajectory"]
    omega_arr = controlled["spectral_omega"]
    target_omega = float(controlled["omega"])
    selected = int(np.argmin(np.abs(omega_arr - target_omega)))
    xy = controlled["sprif_x_t"][:, selected, 1:3]
    probe = controlled["probe_mask"] > 0.5
    T = np.arange(len(xy))
    x1 = xy[:, 0]
    x2 = xy[:, 1]

    # ── b1: 2D phase plane ──
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_time = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap="viridis", linewidth=0.7, alpha=0.7)
    lc.set_array(norm_time)
    ax_b1.add_collection(lc)
    ax_b1.scatter(xy[probe, 0], xy[probe, 1], s=5, color=ORANGE,
                  edgecolor="white", linewidth=0.2, zorder=4)
    ax_b1.autoscale()
    ax_b1.set_aspect("equal", adjustable="datalim")
    ax_b1.set_xticks([-0.1, 0.0, 0.1])
    ax_b1.set_yticks([-0.1, 0.0, 0.1])
    ax_b1.tick_params(labelsize=5.0)
    _clean_grid(ax_b1)
    for spine in ax_b1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    _panel_label(ax_b1, "b")
    # Inline axis labels for compact subplot
    ax_b1.text(0.02, 0.95, r"$x^{\mathrm{osc}}_2$", transform=ax_b1.transAxes,
               fontsize=6, va="top", ha="left", color=GRAY)
    ax_b1.text(0.98, 0.02, r"$x^{\mathrm{osc}}_1$", transform=ax_b1.transAxes,
               fontsize=6, va="bottom", ha="right", color=GRAY)

    # ── b2: x_osc_1(t) and x_osc_2(t) time series with probe windows ──
    probe_idx = np.where(probe)[0]
    starts, ends = [], []
    if len(probe_idx) > 0:
        gaps = np.where(np.diff(probe_idx) > 1)[0]
        starts = [probe_idx[0]] + [probe_idx[g + 1] for g in gaps]
        ends = [probe_idx[g] for g in gaps] + [probe_idx[-1]]
        for s, e in zip(starts, ends):
            ax_b2.axvspan(s, e + 1, color=PROBE_BAND, alpha=0.7, zorder=0)
    ax_b2.plot(T, x1, color=PURPLE, linewidth=0.9, zorder=2)
    ax_b2.plot(T, x2, color=ORANGE, linewidth=0.9, zorder=2)
    ax_b2.axhline(0, color=GRAY_LIGHT, linewidth=0.4, zorder=1)
    ax_b2.set_xlim(0, len(T) - 1)
    # Modest headroom; rely on direct labels (not legend) to avoid overlap
    ymax = max(abs(x1).max(), abs(x2).max()) * 1.1
    ax_b2.set_ylim(-ymax, ymax)
    ax_b2.set_xlabel("step")
    ax_b2.set_ylabel(r"$x^{\mathrm{osc}}$", fontsize=6.5)
    ax_b2.tick_params(labelsize=5.0)
    _clean_grid(ax_b2)
    # Place the "unbroken through probes" annotation in the empty area at the
    # top of the b2 panel (axes coordinates) so it is clearly part of b2
    # but does not collide with the curves.
    ax_b2.text(0.02, 1.08, "unbroken through probes",
               transform=ax_b2.transAxes, fontsize=5.6,
               color=ORANGE, fontweight="bold", ha="left", va="bottom",
               zorder=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="none", alpha=0.85))
    # Direct labels at the right edge of each curve (cleaner than a legend)
    # Stagger vertically so the two labels do not overlap each other
    label_dy = ymax * 0.55  # vertical offset for staggering
    ax_b2.annotate(r"$x^{\mathrm{osc}}_1$",
                   xy=(T[-1], x1[-1]),
                   xytext=(T[-1] + 15, x1[-1] - label_dy),
                   fontsize=5.5, color=PURPLE, fontweight="bold",
                   ha="left", va="center", annotation_clip=False)
    ax_b2.annotate(r"$x^{\mathrm{osc}}_2$",
                   xy=(T[-1], x2[-1]),
                   xytext=(T[-1] + 15, x2[-1] + label_dy),
                   fontsize=5.5, color=ORANGE, fontweight="bold",
                   ha="left", va="center", annotation_clip=False)

    # ── Panel (c): Reset direction fans [1, λ] per task ──────────────────
    # Goal: show that learned [1, λ] directions are diverse within each task
    # and not collapsed to a single shared direction across tasks.
    # One row per task; each row shows 5 quantile arrows.
    reset = data["reset_stats"]
    tasks = ["ECG", "GSC", "pSMNIST"]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_counts = {task: len(reset[reset.task == task]) for task in tasks}

    # Use a smaller vertical range so panel c is more compact
    row_y = {0: 1.5, 1: 0.0, 2: -1.5}
    n_str = "  $n$=" + " / ".join(str(n_counts[t]) for t in tasks)

    for i, task in enumerate(tasks):
        lambdas = reset.loc[reset.task == task, "lambda_reset"].to_numpy(float)
        reps = np.quantile(lambdas, quantiles)
        color = TASK_COLORS[task]
        y_base = row_y[i]
        # row baseline
        ax_c.axhline(y_base, color=GRAY_LIGHT, linewidth=0.4, linestyle="-", alpha=0.4, zorder=0)
        # zero reference dashed vertical
        ax_c.axvline(0, color=GRAY, linewidth=0.4, linestyle=":", alpha=0.5, zorder=1)
        ax_c.axvline(1, color=GRAY, linewidth=0.4, linestyle=":", alpha=0.5, zorder=1)
        for lam in reps:
            ax_c.annotate(
                "",
                xy=(1.0, y_base + lam),
                xytext=(0.0, y_base),
                arrowprops={"arrowstyle": "->", "color": color,
                            "lw": 0.75, "alpha": 0.75,
                            "mutation_scale": 9},
            )
        # task label on left of row
        ax_c.text(-0.18, y_base, DISPLAY_TASK[task], ha="right", va="center",
                  fontsize=6.2, color=color, fontweight="bold")
        # n annotation on the right of row
        ax_c.text(1.22, y_base, f"$n$={n_counts[task]}", ha="left", va="center",
                  fontsize=5.0, color=GRAY)

    # Reference arrow showing [1, 0] (scalar reset) above the three rows
    ref_y = 3.0
    ax_c.annotate(
        "",
        xy=(1.0, ref_y), xytext=(0.0, ref_y),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.7,
                    "linestyle": "--", "alpha": 0.5, "mutation_scale": 8},
    )
    ax_c.text(1.04, ref_y, r"$\lambda=0$ (scalar reset)", fontsize=5.0,
              va="center", color="black", alpha=0.6)

    ax_c.set_xlim(-0.22, 1.55)
    ax_c.set_ylim(-2.3, 3.6)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([r"$u^0$", r"$u^0$"], fontsize=5.5)
    ax_c.set_yticks([])
    ax_c.set_xlabel(r"reset on $u^0$  $\rightarrow$", fontsize=6.0)
    # Place title at left (not centered) to keep clear of the c label
    ax_c.text(-0.20, 3.85,
              r"$\mathbf{c}$" + "   Reset direction fans $[1,\lambda]$  (Q10…Q90)",
              transform=ax_c.transData, fontsize=6.5, ha="left", va="bottom",
              color="black")
    ax_c.spines["left"].set_visible(False)
    ax_c.spines["bottom"].set_visible(False)
    ax_c.tick_params(axis="x", length=0)
    _clean_grid(ax_c)

    # ── Panel (d): Violin plots of λ per task ──────────────────────────────
    values = [reset.loc[reset.task == task, "lambda_reset"].to_numpy(float)
              for task in tasks]
    violins = ax_d.violinplot(values, positions=np.arange(3), widths=0.65,
                              showmedians=True, showextrema=False)
    for body, task in zip(violins["bodies"], tasks):
        body.set_facecolor(TASK_COLORS[task])
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, (v, task) in enumerate(zip(values, tasks)):
        jitter = rng.uniform(-0.12, 0.12, size=len(v))
        ax_d.scatter(i + jitter, v, color=TASK_COLORS[task],
                     s=2.5, alpha=0.35, rasterized=True, zorder=2)

    ax_d.axhline(0, color=RED, linestyle="--", linewidth=0.7, alpha=0.6)
    ax_d.set_xticks(range(3))
    ax_d.set_xticklabels([f"{DISPLAY_TASK[t]}\nn={len(v)}" for t, v in zip(tasks, values)])
    ax_d.set_ylabel(r"Reset direction $\lambda$")
    ax_d.set_title("Learned reset directions")
    _clean_grid(ax_d)
    _panel_label(ax_d, "d")

    # ── Panel (e): Pearson correlation heatmap ─────────────────────────────
    metrics = ["firing_rate", "alpha", "omega"]
    metric_labels = ["Firing rate", r"$\alpha$", r"$\omega$"]
    corr = np.array([[_corr(reset[reset.task == task], "lambda_reset", metric)
                       for metric in metrics] for task in tasks])
    vmax = max(abs(corr.min()), abs(corr.max()), 0.01)
    im = ax_e.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(3):
        for j in range(3):
            color = "white" if abs(corr[i, j]) > vmax * 0.6 else "black"
            ax_e.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                      fontsize=6, fontweight="bold", color=color)
    ax_e.set_xticks(range(3))
    ax_e.set_xticklabels(metric_labels, fontsize=5.8)
    ax_e.set_yticks(range(3))
    ax_e.set_yticklabels([DISPLAY_TASK[t] for t in tasks], fontsize=5.8)
    ax_e.set_title(r"Pearson $r$ with $\lambda$")
    cbar_e = fig.colorbar(im, ax=ax_e, fraction=0.06, pad=0.04, ticks=[-vmax, 0, vmax])
    cbar_e.ax.tick_params(labelsize=5)
    _panel_label(ax_e, "e")

    return fig


def export_figure(fig, output_base: Path, dpi: int = 600) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    fig.savefig(paths[0], facecolor="white", bbox_inches="tight")
    fig.savefig(paths[1], facecolor="white", bbox_inches="tight")
    fig.savefig(paths[2], dpi=dpi, facecolor="white", bbox_inches="tight")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "experiment-design-20260606" / "results" / "figures" / "combined_analysis"
    data = load_source_data(repo_root)
    fig = build_mechanism_figure(data)
    created = export_figure(fig, output_dir / "mechanism_composite_v5")
    plt.close(fig)
    print("created " + ", ".join(str(p) for p in created))


if __name__ == "__main__":
    main()
