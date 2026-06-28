"""
AAAI 主文 5-Panel Figure 绘制脚本
====================================

Figure X: SPRiF Functional State Decomposition Under Controlled Perturbation

Panel layout:
    (a) Task schematic (horizontal strip)
    (b) Slow state portrait — (x¹, x²) phase plane ★ THE MONEY SHOT
    (c) Fast state projective reset — (u⁰, u¹) phase plane ★ NOVELTY EVIDENCE
    (d) Time-domain contrast — SPRiF slow vs membrane vs LIF membrane
    (e) Output verification — SPRiF vs LIF trajectory

Usage:
    python plot_main_figure.py
    # or from run_all.py
"""

import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

matplotlib.use("Agg")

from config import (
    T_CUE, T_DELAY, T_TOTAL, T_PROBES, T_PROBE_DURATION,
    HIDDEN_SIZE, COLORS, OUT_DIR, VIS_OMEGA,
    ZOOM_WINDOW, ZOOM_SPIKE_TIME,
)
from record_forward import (
    select_representative_neuron,
    select_lif_neuron,
)


# ============================================================================
# Figure-level layout
# ============================================================================

def plot_main_figure(
    sprif_record: Dict[str, np.ndarray],
    lif_record: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    phi: float = 0.0,
    omega: float = VIS_OMEGA,
    neuron_idx: Optional[int] = None,
    lif_neuron_idx: Optional[int] = None,
    zoom_spike_time: int = ZOOM_SPIKE_TIME,
    dpi: int = 300,
):
    """Generate the complete AAAI 5-panel main figure.

    Args:
        sprif_record: SPRiF recording for one sample
        lif_record: LIF recording for one sample
        save_path: output file path
        phi: initial phase (rad)
        omega: angular frequency
        neuron_idx: selected SPRiF neuron index (auto-select if None)
        lif_neuron_idx: selected LIF neuron index (auto-select if None)
        zoom_spike_time: time point for Panel (d) zoom
        dpi: output resolution
    """
    # ---- Auto-select neurons ----
    if neuron_idx is None:
        neuron_idx = select_representative_neuron(sprif_record, T_PROBES)
    if lif_neuron_idx is None:
        lif_neuron_idx = select_lif_neuron(lif_record, T_PROBES)

    # ---- Create figure with GridSpec ----
    fig = plt.figure(figsize=(18, 12))

    # Layout: 3 rows
    # Row 0: Panel (a) — task schematic (thin strip)
    # Row 1: Panel (b) left + Panel (c) right (main visual evidence)
    # Row 2: Panel (d) left + Panel (e) right (supporting evidence)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.6, 2.0, 2.0],
                          hspace=0.35, wspace=0.30)

    # ---- Panel (a): Task schematic ----
    ax_a = fig.add_subplot(gs[0, :])
    _plot_panel_a(ax_a, phi, omega)

    # ---- Panel (b): Slow state portrait ----
    ax_b = fig.add_subplot(gs[1, 0])
    _plot_panel_b(ax_b, sprif_record, lif_record, neuron_idx)

    # ---- Panel (c): Fast state projective reset ----
    ax_c = fig.add_subplot(gs[1, 1])
    _plot_panel_c(ax_c, sprif_record, neuron_idx)

    # ---- Panel (d): Time-domain contrast ----
    ax_d = fig.add_subplot(gs[2, 0])
    _plot_panel_d(ax_d, sprif_record, lif_record, neuron_idx, lif_neuron_idx, zoom_spike_time)

    # ---- Panel (e): Output verification ----
    ax_e = fig.add_subplot(gs[2, 1])
    _plot_panel_e(ax_e, sprif_record, lif_record, phi, omega)

    # ---- Figure title ----
    fig.suptitle(
        "SPRiF Functional State Decomposition Under Controlled Perturbation",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # ---- Save ----
    if save_path is None:
        save_path = os.path.join(OUT_DIR, "main_figure.png")

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved main figure: {save_path}")


# ============================================================================
# Panel (a): Task schematic
# ============================================================================

def _plot_panel_a(ax: plt.Axes, phi: float, omega: float):
    """Draw task timeline schematic."""
    ax.set_xlim(0, T_TOTAL)
    ax.set_ylim(-0.5, 2.5)

    # Timeline bar
    ax.axhline(y=1.5, xmin=0, xmax=1.0, color="black", linewidth=2)

    # Cue region
    ax.axvspan(0, T_CUE, alpha=0.15, color=COLORS["cue_bg"])
    ax.text(T_CUE / 2, 2.2, "Cue\n(100 ms)", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Delay region
    ax.axvspan(T_CUE, T_TOTAL, alpha=0.08, color="#e8e8e8")
    ax.text((T_CUE + T_TOTAL) / 2, 2.2, "Delay (800 ms)", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Probe markers
    for tp in T_PROBES:
        ax.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.25, color=COLORS["probe_bg"])
        ax.annotate("", xy=(tp + T_PROBE_DURATION / 2, 1.0),
                    xytext=(tp + T_PROBE_DURATION / 2, 0.5),
                    arrowprops=dict(arrowstyle="->", color=COLORS["spike"], lw=1.5))

    # Phase icon
    ax.text(-40, 1.5, f"φ={phi*180/np.pi:.0f}°\nω={omega*1000/(2*np.pi):.0f}Hz",
            ha="right", va="center", fontsize=8, color="gray")

    # Readout target
    ax.text(T_TOTAL + 40, 1.5, r"$\hat{\mathbf{y}}_t = (\cos,\sin)$",
            ha="left", va="center", fontsize=9, fontstyle="italic", color="gray")

    # Phase input annotation
    ax.annotate(
        "Phase channels\nencode φ", xy=(T_CUE / 2, 0.3),
        ha="center", fontsize=7, color="gray",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5),
    )

    ax.set_ylim(-0.5, 2.8)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_title("(a) Task Schematic — Phase Trajectory Maintenance", fontsize=11,
                 fontweight="bold", loc="left")

    # Clean up spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


# ============================================================================
# Panel (b): Slow state portrait — (x¹, x²) phase plane
# ============================================================================

def _plot_panel_b(
    ax: plt.Axes,
    sprif_record: Dict[str, np.ndarray],
    lif_record: Dict[str, np.ndarray],
    neuron_idx: int,
):
    """Draw slow state (x¹, x²) phase plane — the money shot."""
    slow = sprif_record["slow_states"]  # [T, H, 3]
    spikes = sprif_record["spikes"]     # [T, H]
    T = slow.shape[0]

    # Extract x¹, x² for the selected neuron during delay
    x1 = slow[T_CUE:, neuron_idx, 1]
    x2 = slow[T_CUE:, neuron_idx, 2]
    t_delay = np.arange(T_CUE, T)

    # Color gradient by time
    norm = plt.Normalize(T_CUE, T - 1)
    colors_time = plt.cm.viridis(norm(t_delay))

    # Draw trajectory with time-colored segments
    for i in range(len(t_delay) - 1):
        ax.plot(x1[i:i+2], x2[i:i+2], color=colors_time[i], linewidth=0.8, alpha=0.7)

    # Mark spike events with red dots
    spike_times = np.where(spikes[T_CUE:, neuron_idx] > 0.5)[0] + T_CUE
    if len(spike_times) > 0:
        ax.scatter(x1[spike_times - T_CUE], x2[spike_times - T_CUE],
                   c="red", s=40, zorder=10, marker="o", edgecolors="darkred",
                   linewidths=0.5, label=f"Spike events ({len(spike_times)})")

    # Optional: LIF membrane trajectory for comparison
    lif_mem = lif_record["membranes"]  # [T, H]
    lif_spikes_l = lif_record["spikes"]
    # For LIF we need 2D — use membrane of neuron_idx and another neuron
    lif_other = (neuron_idx + 1) % HIDDEN_SIZE
    lif_x = lif_mem[T_CUE:, neuron_idx]
    lif_y = lif_mem[T_CUE:, lif_other]
    # Normalize LIF to same scale for comparison
    lif_x_norm = (lif_x - lif_x.mean()) / (lif_x.std() + 1e-8) * x1.std() + x1.mean()
    lif_y_norm = (lif_y - lif_y.mean()) / (lif_y.std() + 1e-8) * x2.std() + x2.mean()
    ax.plot(lif_x_norm, lif_y_norm, color="gray", linewidth=0.5, alpha=0.4,
            linestyle="--", label="LIF membrane (rescaled)")

    # LIF spike events
    lif_spike_t = np.where(lif_spikes_l[T_CUE:, neuron_idx] > 0.5)[0] + T_CUE
    if len(lif_spike_t) > 0:
        ax.scatter(lif_x_norm[lif_spike_t - T_CUE], lif_y_norm[lif_spike_t - T_CUE],
                   c="gray", s=20, zorder=9, marker="x", alpha=0.6,
                   label=f"LIF spikes ({len(lif_spike_t)})")

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Time (ms)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(r"$x_t^1$ (slow oscillatory 1)", fontsize=9)
    ax.set_ylabel(r"$x_t^2$ (slow oscillatory 2)", fontsize=9)
    ax.set_title(
        "(b) Slow State Portrait — Continuous Rotation Through Spikes",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)


# ============================================================================
# Panel (c): Fast state projective reset — (u⁰, u¹) phase plane
# ============================================================================

def _plot_panel_c(
    ax: plt.Axes,
    sprif_record: Dict[str, np.ndarray],
    neuron_idx: int,
):
    """Draw fast state (u⁰, u¹) phase plane with projective reset vectors."""
    fast_pre = sprif_record["fast_pre"]    # [T, H, 2]
    fast_post = sprif_record["fast_post"]  # [T, H, 2]
    spikes = sprif_record["spikes"]        # [T, H]
    T = fast_pre.shape[0]

    # Background: full trajectory (faded)
    u0_full = fast_pre[T_CUE:, neuron_idx, 0]
    u1_full = fast_pre[T_CUE:, neuron_idx, 1]
    ax.plot(u0_full, u1_full, color="gray", linewidth=0.4, alpha=0.3)

    # Find spike events during delay
    spike_times = np.where(spikes[T_CUE:, neuron_idx] > 0.5)[0] + T_CUE

    # For each spike, draw pre → post with arrow
    colors_spikes = plt.cm.plasma(np.linspace(0.2, 0.9, len(spike_times)))

    for i, st in enumerate(spike_times):
        pre_u = fast_pre[st, neuron_idx]   # [2]
        post_u = fast_post[st, neuron_idx]  # [2]

        ax.scatter(*pre_u, c=[colors_spikes[i]], s=50, marker="o",
                   zorder=10, edgecolors="black", linewidths=0.5)
        ax.scatter(*post_u, c=[colors_spikes[i]], s=50, marker="v",
                   zorder=10, edgecolors="black", linewidths=0.5)

        # Reset vector arrow
        ax.annotate("", xy=post_u, xytext=pre_u,
                    arrowprops=dict(arrowstyle="->", color=colors_spikes[i],
                                    lw=2.0, alpha=0.8))

    # Legend for pre/post markers
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label=r"$\mathbf{u}^{\mathrm{pre}}$ (pre-spike)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="gray",
               markersize=8, label=r"$\mathbf{u}$ (post-reset)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.8)

    # Annotate reset direction
    ax.annotate(
        r"$\mathbf{u} = \mathbf{u}^{\mathrm{pre}} - \theta[1, \lambda_j]^T$",
        xy=(0.05, 0.95), xycoords="axes fraction",
        fontsize=8, fontstyle="italic", color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.7),
    )

    ax.set_xlabel(r"$u_t^0$ (membrane)", fontsize=9)
    ax.set_ylabel(r"$u_t^1$ (fast auxiliary)", fontsize=9)
    ax.set_title(
        "(c) Fast State Projective Reset — Directional, Learnable",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)


# ============================================================================
# Panel (d): Time-domain contrast
# ============================================================================

def _plot_panel_d(
    ax: plt.Axes,
    sprif_record: Dict[str, np.ndarray],
    lif_record: Dict[str, np.ndarray],
    neuron_idx: int,
    lif_neuron_idx: int,
    zoom_spike_time: int = ZOOM_SPIKE_TIME,
):
    """Three-tier time-domain contrast around a spike event."""
    window = ZOOM_WINDOW
    t_start = max(T_CUE, zoom_spike_time - window)
    t_end = min(T_TOTAL, zoom_spike_time + window)
    t_rel = np.arange(t_start, t_end) - zoom_spike_time

    # ---- Collect data ----
    sprif_slow = sprif_record["slow_states"]
    sprif_mem = sprif_record["membranes"]
    sprif_spikes = sprif_record["spikes"]
    lif_mem = lif_record["membranes"]
    lif_spikes = lif_record["spikes"]

    # SPRiF slow state x¹
    x1_slice = sprif_slow[t_start:t_end, neuron_idx, 1]
    # SPRiF membrane
    mem_slice = sprif_mem[t_start:t_end, neuron_idx]
    # LIF membrane
    lif_mem_slice = lif_mem[t_start:t_end, lif_neuron_idx]

    # Detect spike in window
    sprif_spike_in_window = sprif_spikes[t_start:t_end, neuron_idx] > 0.5
    lif_spike_in_window = lif_spikes[t_start:t_end, lif_neuron_idx] > 0.5

    # ---- Tier 1: SPRiF slow state ----
    ax1 = ax.inset_axes([0, 0.68, 1, 0.28])
    ax1.plot(t_rel, x1_slice, color=COLORS["sprif_slow"], linewidth=1.5, marker=".",
             markersize=2)
    # Mark spike
    for t_idx in np.where(sprif_spike_in_window)[0]:
        ax1.axvline(t_rel[t_idx], color=COLORS["spike"], linestyle="--",
                    linewidth=0.8, alpha=0.6)
    ax1.set_ylabel(r"$x_t^1$", fontsize=8, color=COLORS["sprif_slow"])
    ax1.tick_params(labelsize=7, colors=COLORS["sprif_slow"])
    ax1.set_title("SPRiF Slow State — Continuous", fontsize=8, fontweight="bold",
                  color=COLORS["sprif_slow"], loc="left")
    ax1.grid(True, alpha=0.15)

    # ---- Tier 2: SPRiF membrane ----
    ax2 = ax.inset_axes([0, 0.36, 1, 0.28])
    ax2.plot(t_rel, mem_slice, color=COLORS["sprif_fast"], linewidth=1.5, marker=".",
             markersize=2)
    # Threshold line
    ax2.axhline(y=1.0, color=COLORS["sprif_fast"], linestyle=":", linewidth=0.8, alpha=0.5)
    # Mark spike
    for t_idx in np.where(sprif_spike_in_window)[0]:
        ax2.axvline(t_rel[t_idx], color=COLORS["spike"], linestyle="--",
                    linewidth=0.8, alpha=0.6)
        ax2.scatter(t_rel[t_idx], mem_slice[t_idx], color=COLORS["spike"],
                    s=30, zorder=5, marker="v")
    ax2.set_ylabel(r"$v_t$", fontsize=8, color=COLORS["sprif_fast"])
    ax2.tick_params(labelsize=7, colors=COLORS["sprif_fast"])
    ax2.set_title("SPRiF Membrane — Reset at Spike", fontsize=8, fontweight="bold",
                  color=COLORS["sprif_fast"], loc="left")
    ax2.grid(True, alpha=0.15)

    # ---- Tier 3: LIF membrane ----
    ax3 = ax.inset_axes([0, 0.04, 1, 0.28])
    ax3.plot(t_rel, lif_mem_slice, color=COLORS["lif"], linewidth=1.5, marker=".",
             markersize=2)
    ax3.axhline(y=1.0, color=COLORS["lif"], linestyle=":", linewidth=0.8, alpha=0.5)
    for t_idx in np.where(lif_spike_in_window)[0]:
        ax3.axvline(t_rel[t_idx], color=COLORS["spike"], linestyle="--",
                    linewidth=0.8, alpha=0.6)
        ax3.scatter(t_rel[t_idx], lif_mem_slice[t_idx], color=COLORS["spike"],
                    s=30, zorder=5, marker="v")
    ax3.set_ylabel(r"$v_t$ (LIF)", fontsize=8, color=COLORS["lif"])
    ax3.set_xlabel("Time relative to spike (ms)", fontsize=8)
    ax3.tick_params(labelsize=7, colors=COLORS["lif"])
    ax3.set_title("LIF Membrane — Reset = Memory Loss", fontsize=8, fontweight="bold",
                  color=COLORS["lif"], loc="left")
    ax3.grid(True, alpha=0.15)

    # Shared vertical line at spike
    for ax_sub in [ax1, ax2, ax3]:
        ax_sub.axvline(0, color=COLORS["spike"], linestyle="-", linewidth=1.0, alpha=0.4)

    # Remove host axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"(d) Time-Domain Contrast — SPRiF vs LIF Around Spike (t={zoom_spike_time}ms)",
        fontsize=11, fontweight="bold", loc="left",
    )


# ============================================================================
# Panel (e): Output verification
# ============================================================================

def _plot_panel_e(
    ax: plt.Axes,
    sprif_record: Dict[str, np.ndarray],
    lif_record: Dict[str, np.ndarray],
    phi: float,
    omega: float,
):
    """Output trajectory verification — SPRiF vs LIF."""

    sprif_out = sprif_record["readouts"]   # [T, 2]
    lif_out = lif_record["readouts"]       # [T, 2]
    target = sprif_record["target"]        # [T, 2]
    T = target.shape[0]
    t_arr = np.arange(T)

    # ---- Left: cos time series (delay only) ----
    ax1 = ax.inset_axes([0, 0.55, 1, 0.40])
    delay_slice = slice(T_CUE, T)
    t_delay = t_arr[delay_slice]

    ax1.plot(t_delay, target[delay_slice, 0], color=COLORS["target"],
             linewidth=2.0, label="Target cos", alpha=0.8)
    ax1.plot(t_delay, sprif_out[delay_slice, 0], color=COLORS["sprif_slow"],
             linewidth=1.2, label="SPRiF ŷ_cos")
    ax1.plot(t_delay, lif_out[delay_slice, 0], color=COLORS["lif"],
             linewidth=1.0, linestyle="--", label="LIF ŷ_cos", alpha=0.7)

    # Mark probe windows
    for tp in T_PROBES:
        ax1.axvspan(tp, tp + T_PROBE_DURATION, alpha=0.08, color=COLORS["probe_bg"])

    ax1.set_ylabel(r"$\hat{\cos}_t$", fontsize=9)
    ax1.legend(loc="upper right", fontsize=7, ncol=3)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.15)

    # ---- Right: 2D circle trajectory ----
    ax2 = ax.inset_axes([0, 0.05, 1, 0.45])

    # Target unit circle
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle),
             color=COLORS["target"], linewidth=1.5, linestyle="--", alpha=0.6,
             label="Target circle")

    # SPRiF trajectory
    ax2.plot(sprif_out[delay_slice, 0], sprif_out[delay_slice, 1],
             color=COLORS["sprif_slow"], linewidth=1.2, label="SPRiF")
    # Mark endpoint
    ax2.scatter(*sprif_out[T - 1], color=COLORS["sprif_slow"], s=60, zorder=5, marker="*")
    # Mark start
    ax2.scatter(*sprif_out[T_CUE], color=COLORS["sprif_slow"], s=40, zorder=5, marker="o")

    # LIF trajectory
    ax2.plot(lif_out[delay_slice, 0], lif_out[delay_slice, 1],
             color=COLORS["lif"], linewidth=1.0, linestyle="--", alpha=0.7, label="LIF")
    ax2.scatter(*lif_out[T - 1], color=COLORS["lif"], s=60, zorder=5, marker="*")

    ax2.set_xlabel(r"$\hat{\cos}_t$", fontsize=9)
    ax2.set_ylabel(r"$\hat{\sin}_t$", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_aspect("equal")
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.15)

    # Remove host axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "(e) Output Verification — SPRiF Maintains Phase, LIF Drifts",
        fontsize=11, fontweight="bold", loc="left",
    )


# ============================================================================
# Standalone usage
# ============================================================================

if __name__ == "__main__":
    print("This script is meant to be called from run_all.py after training + recording.")
    print("For testing, run: python run_all.py")
