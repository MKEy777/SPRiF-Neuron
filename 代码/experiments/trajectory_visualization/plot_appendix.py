"""
附录 Panel 绘制脚本
========================

绘制移入附录的可视化 panel:
    Panel A: 输入时间结构图（完整版）
    Panel B: 输入 spike raster
    Panel C: Hidden spike raster (SPRiF + LIF)
    Panel H: Probe 局部放大
    Panel J: Learned spectral parameters 直方图

还包含多样本验证图（3 个额外相位，简化 2-panel 格式）。

Usage:
    python plot_appendix.py
"""

import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from config import (
    T_CUE, T_DELAY, T_TOTAL, T_PROBES, T_PROBE_DURATION,
    N_PHASE_CH, N_PROBE_CH, N_MARKER_CH, N_INPUT_CH,
    HIDDEN_SIZE, COLORS, OUT_DIR,
    VIS_PHASES, VIS_OMEGA,
    MARKER_CUE_CH, MARKER_DELAY_CH,
)
from record_forward import select_representative_neuron


# ============================================================================
# Panel A: 完整输入时间结构图
# ============================================================================

def plot_panel_a_input_structure(
    sample: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    phi: float = 0.0,
    omega: float = VIS_OMEGA,
):
    """Draw detailed input timeline with channel assignments."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})

    t_arr = np.arange(T_TOTAL)
    input_spikes = sample["input_spikes"]

    # ---- Subplot 1: Phase channel rates (theoretical) ----
    ax = axes[0]
    phi_i = np.linspace(0, 2 * np.pi, N_PHASE_CH, endpoint=False)
    for i in range(N_PHASE_CH):
        rate = np.where(
            t_arr < T_CUE,
            30 + 25 * np.cos(omega * t_arr + phi - phi_i[i]),
            0,
        )
        ax.plot(t_arr, rate, linewidth=0.5, alpha=0.6)

    for tp in T_PROBES:
        ax.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.1, color=COLORS["probe_bg"])

    ax.set_ylabel("Firing rate (Hz)", fontsize=9)
    ax.set_title("Appendix A: Phase Channel Firing Rates (20 channels)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15)

    # ---- Subplot 2: Probe channel input (spike raster) ----
    ax = axes[1]
    probe_data = input_spikes[:, N_PHASE_CH:N_PHASE_CH + N_PROBE_CH]
    for i in range(N_PROBE_CH):
        spike_t = np.where(probe_data[:, i] > 0.5)[0]
        if len(spike_t) > 0:
            ax.scatter(spike_t, np.full_like(spike_t, i, dtype=float),
                       marker="|", color=COLORS["sprif_fast"], s=20, alpha=0.7)

    ax.set_ylabel("Channel", fontsize=9)
    ax.set_title("Probe Channels (10 channels, 100Hz in window)", fontsize=10, loc="left")
    ax.set_ylim(-0.5, N_PROBE_CH - 0.5)
    ax.grid(True, alpha=0.15, axis="x")

    # ---- Subplot 3: Marker channels ----
    ax = axes[2]
    ax.plot(t_arr, input_spikes[:, MARKER_CUE_CH], color="blue", linewidth=0.8,
            label="Cue marker", drawstyle="steps-mid")
    ax.plot(t_arr, input_spikes[:, MARKER_DELAY_CH], color="green", linewidth=0.8,
            label="Delay marker", drawstyle="steps-mid")
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Active", fontsize=9)
    ax.set_title("Marker Channels", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_a_input_structure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Panel B: 输入 spike raster
# ============================================================================

def plot_panel_b_spike_raster(
    sample: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Draw full input spike raster for one sample."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    input_spikes = sample["input_spikes"]  # [T, 32]
    T = input_spikes.shape[0]

    for ch in range(N_INPUT_CH):
        spike_t = np.where(input_spikes[:, ch] > 0.5)[0]
        if len(spike_t) > 0:
            # Color-code by channel type
            if ch < N_PHASE_CH:
                color = "steelblue"
            elif ch < N_PHASE_CH + N_PROBE_CH:
                color = COLORS["sprif_fast"]
            else:
                color = "darkgreen"
            ax.scatter(spike_t, np.full_like(spike_t, ch, dtype=float),
                       marker="|", color=color, s=10, alpha=0.5)

    # Demarcate channel groups
    ax.axhline(y=N_PHASE_CH - 0.5, color="black", linewidth=1.0, linestyle="-")
    ax.axhline(y=N_PHASE_CH + N_PROBE_CH - 0.5, color="black", linewidth=1.0, linestyle="-")

    # Cue/Delay boundary
    ax.axvline(x=T_CUE, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

    # Probe windows
    for tp in T_PROBES:
        ax.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.08, color=COLORS["probe_bg"])

    # Labels
    ax.text(N_PHASE_CH / 2, -1.5, "Phase\n(20 ch)", ha="center", fontsize=8, fontweight="bold")
    ax.text(N_PHASE_CH + N_PROBE_CH / 2, -1.5, "Probe\n(10 ch)", ha="center", fontsize=8,
            fontweight="bold")
    ax.text(N_PHASE_CH + N_PROBE_CH + N_MARKER_CH / 2, -1.5, "Marker\n(2 ch)", ha="center",
            fontsize=8, fontweight="bold")

    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Input channel", fontsize=9)
    ax.set_title("Appendix B: Input Spike Raster", fontsize=12, fontweight="bold")
    ax.set_ylim(-3, N_INPUT_CH + 0.5)
    ax.set_xlim(0, T_TOTAL)

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_b_input_raster.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Panel C: Hidden spike raster
# ============================================================================

def plot_panel_c_hidden_raster(
    sprif_record: Dict[str, np.ndarray],
    lif_record: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    max_neurons: int = 64,
):
    """Draw hidden spike raster for both SPRiF and LIF."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    sprif_spikes = sprif_record["spikes"]  # [T, H]
    lif_spikes = lif_record["spikes"]      # [T, H]
    H = min(max_neurons, sprif_spikes.shape[1])

    # ---- SPRiF hidden raster ----
    ax = axes[0]
    for h in range(H):
        spike_t = np.where(sprif_spikes[:, h] > 0.5)[0]
        if len(spike_t) > 0:
            ax.scatter(spike_t, np.full_like(spike_t, h, dtype=float),
                       marker="|", color=COLORS["sprif_slow"], s=8, alpha=0.5)

    for tp in T_PROBES:
        ax.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.1, color=COLORS["probe_bg"])
    ax.axvline(x=T_CUE, color="red", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_ylabel("Neuron index", fontsize=9)
    ax.set_title("SPRiF Hidden Spikes", fontsize=10, fontweight="bold", loc="left",
                 color=COLORS["sprif_slow"])
    n_spikes_sprif = int(sprif_spikes.sum())
    ax.text(0.99, 0.95, f"Total spikes: {n_spikes_sprif}", transform=ax.transAxes,
            ha="right", fontsize=8, color=COLORS["sprif_slow"])
    ax.set_ylim(-0.5, H - 0.5)

    # ---- LIF hidden raster ----
    ax = axes[1]
    for h in range(H):
        spike_t = np.where(lif_spikes[:, h] > 0.5)[0]
        if len(spike_t) > 0:
            ax.scatter(spike_t, np.full_like(spike_t, h, dtype=float),
                       marker="|", color=COLORS["lif"], s=8, alpha=0.5)

    for tp in T_PROBES:
        ax.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.1, color=COLORS["probe_bg"])
    ax.axvline(x=T_CUE, color="red", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Neuron index", fontsize=9)
    ax.set_title("LIF Hidden Spikes", fontsize=10, fontweight="bold", loc="left",
                 color=COLORS["lif"])
    n_spikes_lif = int(lif_spikes.sum())
    ax.text(0.99, 0.95, f"Total spikes: {n_spikes_lif}", transform=ax.transAxes,
            ha="right", fontsize=8, color=COLORS["lif"])
    ax.set_ylim(-0.5, H - 0.5)

    fig.suptitle("Appendix C: Hidden Spike Rasters — SPRiF vs LIF",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_c_hidden_raster.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Panel H: Probe 局部放大
# ============================================================================

def plot_panel_h_probe_zoom(
    sprif_record: Dict[str, np.ndarray],
    neuron_idx: int,
    probe_idx: int = 2,  # default: 3rd probe (t=420ms)
    window_before: int = 15,
    window_after: int = 15,
    save_path: Optional[str] = None,
):
    """Zoom into a single probe window, showing state dynamics frame-by-frame."""
    tp = T_PROBES[probe_idx]
    t_start = max(0, tp - window_before)
    t_end = min(T_TOTAL, tp + T_PROBE_DURATION + window_after)
    t_rel = np.arange(t_start, t_end) - tp

    slow = sprif_record["slow_states"]
    fast_pre = sprif_record["fast_pre"]
    fast_post = sprif_record["fast_post"]
    membranes = sprif_record["membranes"]
    spikes = sprif_record["spikes"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # ---- Top: Slow state 3D ----
    ax = axes[0]
    labels = [r"$x^0$ (real)", r"$x^1$ (osc₁)", r"$x^2$ (osc₂)"]
    colors_3d = ["#1b9e77", "#d95f02", "#7570b3"]
    for d in range(3):
        ax.plot(t_rel, slow[t_start:t_end, neuron_idx, d],
                color=colors_3d[d], label=labels[d], linewidth=1.8, marker=".", markersize=4)
    # Mark probe window
    ax.axvspan(0, T_PROBE_DURATION, alpha=0.12, color=COLORS["probe_bg"])
    # Mark spikes
    spike_window = np.where(spikes[t_start:t_end, neuron_idx] > 0.5)[0]
    for si in spike_window:
        ax.axvline(t_rel[si], color=COLORS["spike"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("State value", fontsize=9)
    ax.set_title("Slow State — Continuous Through Perturbation", fontsize=10,
                 fontweight="bold", loc="left", color=COLORS["sprif_slow"])
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.15)

    # ---- Middle: Fast state 2D (pre and post) ----
    ax = axes[1]
    for d in range(2):
        label_pre = r"$\tilde{u}^" + str(d) + r"$ (pre)" if d == 0 else None
        ax.plot(t_rel, fast_pre[t_start:t_end, neuron_idx, d],
                color=colors_3d[d + 1], linewidth=1.5, marker=".", markersize=3,
                alpha=0.7, linestyle="-")
        label_post = r"$u^" + str(d) + r"$ (post)" if d == 0 else None
        ax.plot(t_rel, fast_post[t_start:t_end, neuron_idx, d],
                color=colors_3d[d + 1], linewidth=1.5, marker=".", markersize=3,
                alpha=0.4, linestyle=":")
    ax.axvspan(0, T_PROBE_DURATION, alpha=0.12, color=COLORS["probe_bg"])
    for si in spike_window:
        ax.axvline(t_rel[si], color=COLORS["spike"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Fast state", fontsize=9)
    ax.set_title("Fast State — Reset at Spike", fontsize=10,
                 fontweight="bold", loc="left", color=COLORS["sprif_fast"])
    ax.grid(True, alpha=0.15)

    # ---- Bottom: Membrane with threshold ----
    ax = axes[2]
    ax.plot(t_rel, membranes[t_start:t_end, neuron_idx],
            color="#2c7bb6", linewidth=2.0, marker=".", markersize=4, label=r"$v_t$")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="Threshold")
    ax.axvspan(0, T_PROBE_DURATION, alpha=0.12, color=COLORS["probe_bg"])
    for si in spike_window:
        ax.scatter(t_rel[si], membranes[t_start:t_end, neuron_idx][si],
                   color=COLORS["spike"], s=80, zorder=5, marker="v")
    ax.set_xlabel(f"Time relative to probe onset at t={tp}ms", fontsize=9)
    ax.set_ylabel("Membrane", fontsize=9)
    ax.set_title("Membrane Potential — Threshold Crossing", fontsize=10,
                 fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Appendix H: Probe Zoom — Neuron #{neuron_idx}, Probe at t={tp}ms",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_h_probe_zoom.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Panel J: Learned spectral parameters histogram
# ============================================================================

def plot_panel_j_parameters(
    sprif_record: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Draw histograms of learned SPRiF spectral parameters."""
    params = sprif_record["spectral_params"]
    H = len(params["alpha"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    param_configs = [
        ("alpha", r"$\alpha$ (decay)", "Blues", (0, 1)),
        ("rho", r"$\rho$ (rotation decay)", "Oranges", (0, 1)),
        ("omega", r"$\omega$ (angular freq, rad/ms)", "Greens", None),
        ("eta", r"$\eta$ (fast decay)", "Purples", (0, 1)),
        ("lambda_reset", r"$\lambda$ (reset direction)", "Reds", None),
    ]

    for idx, (key, label, cmap, xlim) in enumerate(param_configs):
        ax = axes_flat[idx]
        vals = params[key]
        if vals.ndim == 2:
            vals = vals.flatten()

        ax.hist(vals, bins=30, color=plt.cm.get_cmap(cmap)(0.6),
                edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        if xlim:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.15, axis="y")

        # Add statistics
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.text(0.95, 0.95, f"μ={mean_val:.3f}\nσ={std_val:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Hide extra subplot
    axes_flat[5].set_visible(False)

    fig.suptitle(
        f"Appendix J: Learned SPRiF Spectral Parameters ({H} neurons)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_j_parameters.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Multi-sample verification (simplified 2-panel for 3 extra phases)
# ============================================================================

def plot_multi_sample_verification(
    sprif_records: List[Dict[str, np.ndarray]],
    lif_records: List[Dict[str, np.ndarray]],
    phases: List[float],
    save_path: Optional[str] = None,
):
    """Plot simplified 2-panel figures for 3 extra phases (appendix).

    Only keeps panels (b) phase portrait + (e) output trajectory.
    """
    # Skip φ=0 (already in main figure), show the other 3
    n_extra = len(phases) - 1
    fig, axes = plt.subplots(n_extra, 2, figsize=(14, 4.5 * n_extra))

    if n_extra == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(n_extra):
        rec_idx = row_idx + 1  # skip φ=0
        sprif_rec = sprif_records[rec_idx]
        lif_rec = lif_records[rec_idx]
        phi = phases[rec_idx]

        # ---- Left: Slow state portrait ----
        ax = axes[row_idx, 0]
        slow = sprif_rec["slow_states"]
        spks = sprif_rec["spikes"]
        n_idx = select_representative_neuron(sprif_rec, T_PROBES)

        x1 = slow[T_CUE:, n_idx, 1]
        x2 = slow[T_CUE:, n_idx, 2]
        t_delay = np.arange(T_CUE, slow.shape[0])
        norm = plt.Normalize(T_CUE, slow.shape[0] - 1)
        colors_time = plt.cm.viridis(norm(t_delay))

        for i in range(len(t_delay) - 1):
            ax.plot(x1[i:i+2], x2[i:i+2], color=colors_time[i], linewidth=0.6, alpha=0.7)

        spike_t = np.where(spks[T_CUE:, n_idx] > 0.5)[0] + T_CUE
        if len(spike_t) > 0:
            ax.scatter(x1[spike_t - T_CUE], x2[spike_t - T_CUE],
                       c="red", s=30, zorder=10, marker="o")

        ax.set_xlabel(r"$x_t^1$", fontsize=8)
        ax.set_ylabel(r"$x_t^2$", fontsize=8)
        ax.set_title(f"SPRiF Slow State (φ={phi*180/np.pi:.0f}°)", fontsize=10,
                     fontweight="bold", color=COLORS["sprif_slow"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)

        # ---- Right: Output trajectory ----
        ax = axes[row_idx, 1]
        sprif_out = sprif_rec["readouts"]
        lif_out = lif_rec["readouts"]
        target = sprif_rec["target"]

        # Cos time series (delay)
        t_all = np.arange(target.shape[0])
        ax.plot(t_all[T_CUE:], target[T_CUE:, 0], color=COLORS["target"],
                linewidth=1.5, alpha=0.7, label="Target cos")
        ax.plot(t_all[T_CUE:], sprif_out[T_CUE:, 0], color=COLORS["sprif_slow"],
                linewidth=1.0, label="SPRiF")
        ax.plot(t_all[T_CUE:], lif_out[T_CUE:, 0], color=COLORS["lif"],
                linewidth=0.8, linestyle="--", alpha=0.6, label="LIF")
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel(r"$\hat{\cos}_t$", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Output Trajectory (SPRiF vs LIF)", fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.15)

        # Compute MSEs for annotation
        mse_sprif = np.mean((sprif_out[T_CUE:] - target[T_CUE:]) ** 2)
        mse_lif = np.mean((lif_out[T_CUE:] - target[T_CUE:]) ** 2)
        ax.text(0.02, 0.05, f"SPRiF MSE={mse_sprif:.4f}\nLIF MSE={mse_lif:.4f}",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle("Appendix: Multi-Sample Verification — Results Not Cherry-Picked",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, "appendix_multi_sample.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Plot all appendix panels
# ============================================================================

def plot_all_appendix(
    vis_samples: List[Dict[str, np.ndarray]],
    sprif_records: List[Dict[str, np.ndarray]],
    lif_records: List[Dict[str, np.ndarray]],
    out_dir: str = OUT_DIR,
    primary_idx: int = 0,
):
    """Generate all appendix panels.

    Args:
        vis_samples: list of sample dicts (4 phases)
        sprif_records: list of SPRiF recordings
        lif_records: list of LIF recordings
        out_dir: output directory
        primary_idx: index of the sample used in main figure (default 0 = φ=0)
    """
    sample = vis_samples[primary_idx]
    sprif_rec = sprif_records[primary_idx]
    lif_rec = lif_records[primary_idx]
    phi = VIS_PHASES[primary_idx]

    # Select neuron
    neuron_idx = select_representative_neuron(sprif_rec, T_PROBES)

    print("\nGenerating appendix panels...")

    # Panel A
    plot_panel_a_input_structure(
        sample,
        save_path=os.path.join(out_dir, "appendix_a_input_structure.png"),
        phi=phi,
    )

    # Panel B
    plot_panel_b_spike_raster(
        sample,
        save_path=os.path.join(out_dir, "appendix_b_input_raster.png"),
    )

    # Panel C
    plot_panel_c_hidden_raster(
        sprif_rec, lif_rec,
        save_path=os.path.join(out_dir, "appendix_c_hidden_raster.png"),
    )

    # Panel H (for each probe)
    for p_idx in range(len(T_PROBES)):
        plot_panel_h_probe_zoom(
            sprif_rec, neuron_idx, probe_idx=p_idx,
            save_path=os.path.join(out_dir, f"appendix_h_probe_zoom_p{p_idx}.png"),
        )

    # Panel J
    plot_panel_j_parameters(
        sprif_rec,
        save_path=os.path.join(out_dir, "appendix_j_parameters.png"),
    )

    # Multi-sample verification
    plot_multi_sample_verification(
        sprif_records, lif_records, VIS_PHASES,
        save_path=os.path.join(out_dir, "appendix_multi_sample.png"),
    )

    print("  All appendix panels generated.")


if __name__ == "__main__":
    print("This script is meant to be called from run_all.py.")
    print("For testing, run: python run_all.py")
