"""Build two dense, publication-ready SPRiF analysis figures from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


WIDTH = 8.4
HEIGHT = 4.8
TEAL = "#1f4e5f"
BLUE = "#2a6f9e"
ORANGE = "#9c4d0e"
PURPLE = "#5d3a8e"
RED = "#8B0000"
GRAY = "#666666"
TASK_COLORS = {"ECG": "#0d3b5c", "GSC": "#5d3a8e", "pSMNIST": "#9c4d0e"}
DISPLAY_TASK = {"ECG": "QTDB", "GSC": "GSC", "pSMNIST": "pSMNIST"}


def _set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _read_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(set(required) - set(source.files))
        if missing:
            raise ValueError(f"{path.name} is missing keys: {', '.join(missing)}")
        return {key: source[key].copy() for key in source.files}


def load_source_data(repo_root: Path) -> dict:
    figures = repo_root / "experiment-design-20260606" / "results" / "figures"
    real = _read_npz(
        figures / "trajectory_analysis" / "trajectory_data.npz",
        ["layer1_slow", "layer1_membrane", "layer1_spikes", "highlight_spike_time", "highlight_neuron"],
    )
    controlled = _read_npz(
        repo_root / "experiment-design-20260606" / "legacy" / "trajectory_visualization" / "trajectory_data_phi1.npz",
        ["sprif_x_t", "sprif_u_pre", "sprif_u_post", "sprif_spike", "probe_mask", "spectral_omega", "omega"],
    )
    reset_path = figures / "reset_analysis" / "lambda_stats.csv"
    if not reset_path.exists():
        raise FileNotFoundError(reset_path)
    reset = pd.read_csv(reset_path)
    required_columns = {"task", "lambda_reset", "firing_rate", "alpha", "omega"}
    missing_columns = sorted(required_columns - set(reset.columns))
    if missing_columns:
        raise ValueError(f"lambda_stats.csv is missing columns: {', '.join(missing_columns)}")
    impulse = _read_npz(
        figures / "impulse_analysis" / "raw_impulse_responses.npz",
        ["pSMNIST_L0_slow_resp", "GSC_L0_slow_resp", "ECG_L0_slow_resp"],
    )
    asrnn = _read_npz(
        figures / "impulse_analysis" / "asrnn_comparison_data.npz",
        ["GSC_col2_sprif_x_real", "GSC_col2_asrnn_kernel", "GSC_col2_meta", "ECG_col2_meta"],
    )
    return {
        "real_trajectory": real,
        "controlled_trajectory": controlled,
        "reset_stats": reset,
        "impulse": impulse,
        "asrnn": asrnn,
        "source_dirs": {
            "trajectory_visualization",
            "trajectory_analysis",
            "reset_analysis",
            "impulse_analysis",
        },
    }


def _panel(ax, label: str) -> None:
    ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _clean(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.4, alpha=0.7)
    ax.tick_params(length=2, width=0.5)


def _corr(frame: pd.DataFrame, x: str, y: str) -> float:
    return float(np.corrcoef(frame[x].to_numpy(float), frame[y].to_numpy(float))[0, 1])


def build_mechanism_figure(data: dict):
    _set_style()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = fig.add_gridspec(2, 12, left=0.065, right=0.985, bottom=0.13, top=0.95, hspace=0.72, wspace=1.7)
    ax_a1 = fig.add_subplot(outer[0, 0:5])
    ax_a2 = fig.add_subplot(outer[1, 0:5], sharex=ax_a1)
    ax_b = fig.add_subplot(outer[0, 5:8])
    ax_c = fig.add_subplot(outer[1, 5:8])
    ax_d = fig.add_subplot(outer[0, 8:12])
    ax_e = fig.add_subplot(outer[1, 8:12])

    real = data["real_trajectory"]
    spike_t = int(real["highlight_spike_time"])
    neuron = int(real["highlight_neuron"])
    start, end = spike_t - 15, spike_t + 16
    t = np.arange(start, end) - spike_t
    slow = real["layer1_slow"][start:end, neuron]
    membrane = real["layer1_membrane"][start:end, neuron]
    spikes = real["layer1_spikes"][start:end, neuron] > 0.5
    for idx, (color, label) in enumerate([(TEAL, r"$x^{real}$"), (ORANGE, r"$x^{osc}_1$"), (PURPLE, r"$x^{osc}_2$")]):
        ax_a1.plot(t, slow[:, idx], color=color, label=label)
    ax_a1.axvline(0, color=RED, linestyle="--", linewidth=0.9)
    ax_a1.set_ylabel("slow state")
    ax_a1.set_title("Recorded pSMNIST spike")
    ax_a1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=0.8)
    mem_rel = membrane - membrane[0]
    ax_a2.plot(t, mem_rel, color=BLUE)
    ax_a2.scatter(t[spikes], mem_rel[spikes], color=RED, marker="v", s=13, zorder=3)
    ax_a2.axvline(0, color=RED, linestyle="--", linewidth=0.9)
    ax_a2.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax_a2.set_xlabel("steps relative to spike")
    ax_a2.set_ylabel(r"fast $u^0$ (relative)")
    _panel(ax_a1, "a")

    controlled = data["controlled_trajectory"]
    omega = controlled["spectral_omega"]
    selected = int(np.argmin(np.abs(omega - float(controlled["omega"]))))
    xy = controlled["sprif_x_t"][:, selected, 1:3]
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidth=0.9)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax_b.add_collection(lc)
    probe = controlled["probe_mask"] > 0.5
    ax_b.scatter(xy[probe, 0], xy[probe, 1], s=7, color=ORANGE, zorder=3)
    ax_b.autoscale()
    ax_b.set_aspect("equal", adjustable="datalim")
    ax_b.set_xlabel(r"$x^{osc}_1$")
    ax_b.set_ylabel(r"$x^{osc}_2$")
    ax_b.set_title("Controlled phase memory")
    _panel(ax_b, "b")

    reset = data["reset_stats"]
    tasks = ["ECG", "GSC", "pSMNIST"]
    for task in tasks:
        lambdas = reset.loc[reset.task == task, "lambda_reset"].to_numpy(float)
        representatives = np.quantile(lambdas, [0.1, 0.3, 0.5, 0.7, 0.9])
        for lam in representatives:
            ax_c.annotate(
                "",
                xy=(1.0, lam),
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "->", "color": TASK_COLORS[task], "lw": 0.75, "alpha": 0.72},
            )
        ax_c.plot([], [], color=TASK_COLORS[task], label=DISPLAY_TASK[task])
    ax_c.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax_c.axvline(0, color="#aaaaaa", linewidth=0.5)
    ax_c.set_xlim(-0.05, 1.08)
    ax_c.set_xlabel(r"reset on $u^0$")
    ax_c.set_ylabel(r"reset on $u^1$")
    ax_c.set_title(r"Learned directions $[1,\lambda]$")
    ax_c.legend(ncol=3, loc="upper left", handlelength=1.0, columnspacing=0.5)
    _panel(ax_c, "c")

    values = [reset.loc[reset.task == task, "lambda_reset"].to_numpy(float) for task in tasks]
    violins = ax_d.violinplot(values, positions=np.arange(3), widths=0.72, showmedians=True, showextrema=False)
    for body, task in zip(violins["bodies"], tasks):
        body.set_facecolor(TASK_COLORS[task])
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(0.8)
    ax_d.axhline(0, color=RED, linestyle="--", linewidth=0.7)
    ax_d.set_xticks(range(3), [f"{DISPLAY_TASK[t]}\n$n$={len(v)}" for t, v in zip(tasks, values)])
    ax_d.set_ylabel(r"reset direction $\lambda$")
    ax_d.set_title("Learned reset directions")
    _panel(ax_d, "d")

    metrics = ["firing_rate", "alpha", "omega"]
    corr = np.array([[_corr(reset[reset.task == task], "lambda_reset", metric) for metric in metrics] for task in tasks])
    ax_e.imshow(corr, cmap="coolwarm", vmin=-0.2, vmax=0.2, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax_e.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=5.5)
    ax_e.set_xticks(range(3), ["firing", r"$\alpha$", r"$\omega$"])
    ax_e.set_yticks(range(3), [DISPLAY_TASK[t] for t in tasks])
    ax_e.set_title(r"Pearson $r$ with $\lambda$")
    _panel(ax_e, "e")

    for ax in [ax_a1, ax_a2, ax_b, ax_c, ax_d]:
        _clean(ax)
    return fig


def _quantile_indices(alpha: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(alpha - np.quantile(alpha, q)))) for q in (0.1, 0.4, 0.7, 0.9)])


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(values), axis=1, keepdims=True) + 1e-12
    return values / denom


def build_temporal_figure(data: dict):
    _set_style()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    outer = fig.add_gridspec(2, 12, left=0.065, right=0.985, bottom=0.16, top=0.94, height_ratios=[1.05, 0.95], hspace=0.55, wspace=1.0)
    top = outer[0, :].subgridspec(1, 3, wspace=0.32)
    bottom_left = outer[1, 0:8].subgridspec(1, 3, wspace=0.38)
    top_axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    freq_axes = [fig.add_subplot(bottom_left[0, i]) for i in range(3)]
    ax_c = fig.add_subplot(outer[1, 8:12])

    impulse = data["impulse"]
    task_keys = [("pSMNIST", "pSMNIST_L0"), ("GSC", "GSC_L0"), ("ECG", "ECG_L0")]
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, 4))
    for col, (task, key) in enumerate(task_keys):
        ax = top_axes[col]
        slow = impulse[f"{key}_slow_resp"]
        alpha = impulse[f"{key}_alpha"]
        indices = _quantile_indices(alpha)
        tcolor = TASK_COLORS[task]
        for color, idx in zip(colors, indices):
            ax.plot(slow[idx, :, 0], color=color, linewidth=1.4, label=fr"$\alpha$={alpha[idx]:.2f}")
            ax.plot(slow[idx, :, 1], color=color, linestyle=":", linewidth=0.9, alpha=0.9)
        ax.axhline(0, color="#aaaaaa", linewidth=0.5)
        ax.set_xlim(0, slow.shape[1] - 1)
        ax.set_title(DISPLAY_TASK[task], color=tcolor, fontweight="bold")
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color(tcolor)
        ax.spines["top"].set_linewidth(1.8)
        ax.set_xlabel("step")
        if col == 0:
            ax.set_ylabel("impulse response")
            ax.legend(ncol=2, loc="upper right", columnspacing=0.7, handlelength=1.4, frameon=True, framealpha=0.92)
            _panel(ax, "a")
        _clean(ax)

        freq_ax = freq_axes[col]
        spectra = np.abs(np.fft.rfft(slow[:, :, 0], axis=1))
        spectra = _normalize_rows(spectra)
        freqs = np.fft.rfftfreq(slow.shape[1])
        median = np.median(spectra, axis=0)
        lo, hi = np.quantile(spectra, [0.1, 0.9], axis=0)
        task_color = TASK_COLORS[task]
        freq_ax.fill_between(freqs, lo, hi, color=task_color, alpha=0.22, linewidth=0)
        freq_ax.plot(freqs, median, color=task_color, linewidth=1.6)
        freq_ax.set_xlim(0, 0.5)
        freq_ax.set_ylim(0, 1.03)
        freq_ax.set_xlabel("cycles / step")
        if col == 0:
            freq_ax.set_ylabel("normalized |FFT|")
            _panel(freq_ax, "b")
        freq_ax.set_title(DISPLAY_TASK[task], color=task_color, fontweight="bold")
        _clean(freq_ax)

    asrnn = data["asrnn"]
    task_marker = {"GSC": "o", "ECG": "s"}
    styles = [("GSC", TASK_COLORS["GSC"]), ("ECG", TASK_COLORS["ECG"])]
    max_len = 0
    legend_handles = []
    legend_labels = []
    for task, color in styles:
        sprif = asrnn[f"{task}_col2_sprif_x_real"].astype(float)
        baseline = asrnn[f"{task}_col2_asrnn_kernel"].astype(float)
        meta = asrnn[f"{task}_col2_meta"]
        sprif = sprif / (np.max(np.abs(sprif)) + 1e-12)
        baseline = baseline / (np.max(np.abs(baseline)) + 1e-12)
        max_len = max(max_len, len(sprif), len(baseline))
        label = DISPLAY_TASK[task]
        mk = task_marker[task]
        h_sprif = ax_c.plot(sprif, color=color, linewidth=1.6, marker=mk, markevery=10,
                            markerfacecolor="white", markeredgewidth=1.0, markeredgecolor=color)[0]
        h_base = ax_c.plot(baseline, color=color, linestyle="--", linewidth=1.3, marker=mk, markevery=10,
                           markerfacecolor=color, markeredgewidth=1.0, markeredgecolor=color)[0]
        legend_handles.append(h_sprif)
        legend_labels.append(f"{label} SPRiF  τ={meta[1]:.0f}")
        legend_handles.append(h_base)
        legend_labels.append(f"{label} ASRNN  τ={meta[2]:.0f}")
    ax_c.set_xlim(0, max_len - 1)
    ax_c.set_ylim(-0.03, 1.03)
    ax_c.set_xlabel("step")
    ax_c.set_ylabel("normalized kernel")
    ax_c.set_title("SPRiF vs. ASRNN slow-decay kernels")
    leg = ax_c.legend(legend_handles, legend_labels,
                      loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                      handlelength=1.6, frameon=True, framealpha=0.95,
                      columnspacing=0.8, labelspacing=0.5, fontsize=7.5)
    _panel(ax_c, "c")
    _clean(ax_c)
    return fig


def export_figure(fig, output_base: Path, dpi: int = 600) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    fig.savefig(paths[0], facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(paths[1], facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(paths[2], dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "experiment-design-20260606" / "results" / "figures" / "combined_analysis"
    data = load_source_data(repo_root)
    figures = [
        (build_mechanism_figure(data), output_dir / "mechanism_composite"),
        (build_temporal_figure(data), output_dir / "temporal_kernels_composite"),
    ]
    for fig, base in figures:
        created = export_figure(fig, base)
        plt.close(fig)
        print(f"created {', '.join(str(path) for path in created)}")
    print("source directories: " + ", ".join(sorted(data["source_dirs"])))


if __name__ == "__main__":
    main()
