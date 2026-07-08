"""
相位轨迹可视化实验 — 主图绘制。

5-panel 主图：
(a) timeline 示意图
(b) 慢状态相平面 (x1, x2)
(c) 快状态投影重置 (u0, u1)
(d) 时间域对比 (t=420ms ±30ms)
(e) 输出验证
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from config import FIGURE_DIR, T_CUE, PROBE_TIMES, VIZ_PHIS


def _load_trajectory(phi_idx: int) -> dict:
    """加载记录的轨迹数据。"""
    path = os.path.join(FIGURE_DIR, f"trajectory_data_phi{phi_idx}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory data not found: {path}")
    data = np.load(path)
    return dict(data)


def plot_timeline(ax):
    """(a) Timeline 示意图。"""
    ax.set_xlim(0, 900)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Cue", "Delay"], fontsize=10)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_title("(a) Task Timeline", fontweight="bold", fontsize=12)

    # Cue 阶段
    ax.fill_between([0, T_CUE], 0, 2, alpha=0.3, color="blue", label="Cue (100ms)")
    # Delay 阶段
    ax.fill_between([T_CUE, 900], 0, 2, alpha=0.1, color="gray", label="Delay (800ms)")

    # Probe 时刻
    for t_probe in PROBE_TIMES:
        ax.axvline(t_probe, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.plot([], [], "r--", linewidth=1, alpha=0.6, label="Probes")

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)


def plot_slow_state_phase(ax, data):
    """(b) 慢状态相平面 (x1, x2)。"""
    x_t = data["sprif_x_t"]  # [T, H, 3]
    probe_mask = data["probe_mask"]  # [T]

    # 选一个 neuron（选 omega 接近 viz_omega 的）
    omega = data["spectral_omega"]
    target_omega = data["omega"]
    neuron_idx = np.argmin(np.abs(omega - target_omega))

    x1 = x_t[:, neuron_idx, 1]  # x_osc_1
    x2 = x_t[:, neuron_idx, 2]  # x_osc_2

    # 颜色编码时间
    T = len(x1)
    colors = np.linspace(0.2, 0.9, T)
    norm = Normalize(vmin=0, vmax=T)

    # 画轨迹
    points = np.array([x1, x2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=plt.cm.viridis(colors[:-1]), linewidth=1.5)
    ax.add_collection(lc)

    # Probe 时刻打点
    probe_steps = np.where(probe_mask > 0.5)[0]
    if len(probe_steps) > 0:
        ax.scatter(x1[probe_steps], x2[probe_steps], c="red", s=30, zorder=5,
                   marker="o", label="Probe times")

    # 起点
    ax.scatter(x1[0], x2[0], c="green", s=50, zorder=5, marker="*", label="Start")

    ax.set_xlabel(r"$x^{\mathrm{osc}}_1$", fontsize=11)
    ax.set_ylabel(r"$x^{\mathrm{osc}}_2$", fontsize=11)
    ax.set_title("(b) SPRiF Slow State Phase Plane", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def plot_fast_state_reset(ax, data):
    """(c) 快状态投影重置 (u0, u1)。"""
    u_pre = data["sprif_u_pre"]  # [T, H, 2]
    u_post = data["sprif_u_post"]  # [T, H, 2]
    spike = data["sprif_spike"]  # [T, H]
    probe_mask = data["probe_mask"]  # [T]
    lambda_reset = data["spectral_lambda"]  # [H]

    # 选一个 neuron（选有 spike 的）
    total_spikes = spike.sum(axis=0)
    neuron_idx = np.argmax(total_spikes > 0)

    u0_pre = u_pre[:, neuron_idx, 0]
    u1_pre = u_pre[:, neuron_idx, 1]
    u0_post = u_post[:, neuron_idx, 0]
    u1_post = u_post[:, neuron_idx, 1]

    # 画 pre-reset 轨迹
    ax.plot(u0_pre, u1_pre, "b-", linewidth=1, alpha=0.5, label="Pre-reset")

    # 画 post-reset 轨迹
    ax.plot(u0_post, u1_post, "r-", linewidth=1, alpha=0.5, label="Post-reset")

    # Probe 时刻画箭头
    probe_steps = np.where(probe_mask > 0.5)[0]
    spike_steps = np.where(spike[:, neuron_idx] > 0.5)[0]
    # 找 probe 时刻有 spike 的
    probe_spike_steps = np.intersect1d(probe_steps, spike_steps)

    if len(probe_spike_steps) > 0:
        for t in probe_spike_steps[:3]:  # 最多画 3 个箭头
            ax.annotate("",
                        xy=(u0_post[t], u1_post[t]),
                        xytext=(u0_pre[t], u1_pre[t]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.set_xlabel(r"$u_0$ (fast state)", fontsize=11)
    ax.set_ylabel(r"$u_1$ (fast state)", fontsize=11)
    ax.set_title("(c) SPRiF Fast State Reset", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)


def plot_time_domain_contrast(ax, data):
    """(d) 时间域对比 (t=420ms ±30ms)。"""
    T_total = len(data["sprif_membrane"])
    t_center = 420
    t_window = 30
    t_start = max(0, t_center - t_window)
    t_end = min(T_total, t_center + t_window + 1)
    t_rel = np.arange(t_start, t_end) - t_center

    # SPRiF slow state (x1)
    x1 = data["sprif_x_t"][t_start:t_end, :, 1].mean(axis=1)
    ax.plot(t_rel, x1, "b-", linewidth=2, label="SPRiF $x^{\mathrm{osc}}_1$")

    # SPRiF membrane
    sprif_mem = data["sprif_membrane"][t_start:t_end, :].mean(axis=1)
    ax.plot(t_rel, sprif_mem, "g-", linewidth=2, label="SPRiF membrane")

    # ASRNN mem
    asrnn_mem = data["asrnn_mem"][t_start:t_end, :].mean(axis=1)
    ax.plot(t_rel, asrnn_mem, "r--", linewidth=2, label="ASRNN mem")

    # 竖虚线标记 probe 时刻
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Time relative to probe (ms)", fontsize=11)
    ax.set_ylabel("State value", fontsize=11)
    ax.set_title("(d) Time-Domain Contrast", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)


def plot_output_verification(ax_top, ax_bottom, data):
    """(e) 输出验证。"""
    target = data["target"]  # [T, 2]
    sprif_readout = data["sprif_readout"]  # [T, 2]
    asrnn_readout = data["asrnn_readout"]  # [T, 2]

    # 上半：时间序列
    ax_top.plot(target[:, 0], "k-", linewidth=1.5, label="Target cos")
    ax_top.plot(sprif_readout[:, 0], "b-", linewidth=1.5, alpha=0.7, label="SPRiF ŷ_cos")
    ax_top.plot(asrnn_readout[:, 0], "r--", linewidth=1.5, alpha=0.7, label="ASRNN ŷ_cos")
    ax_top.set_ylabel("ŷ_cos", fontsize=11)
    ax_top.set_title("(e) Output Verification", fontweight="bold", fontsize=12)
    ax_top.legend(loc="upper right", fontsize=9, frameon=True)
    ax_top.grid(True, alpha=0.3)

    # 下半：2D 轨迹
    ax_bottom.plot(target[:, 0], target[:, 1], "k-", linewidth=2, label="Target")
    ax_bottom.plot(sprif_readout[:, 0], sprif_readout[:, 1], "b-", linewidth=1.5,
                   alpha=0.7, label="SPRiF")
    ax_bottom.plot(asrnn_readout[:, 0], asrnn_readout[:, 1], "r--", linewidth=1.5,
                   alpha=0.7, label="ASRNN")
    ax_bottom.set_xlabel("ŷ_cos", fontsize=11)
    ax_bottom.set_ylabel("ŷ_sin", fontsize=11)
    ax_bottom.legend(loc="upper right", fontsize=9, frameon=True)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_aspect("equal")


def plot_main_figure(phi_idx: int = 1):
    """绘制 5-panel 主图。"""
    data = _load_trajectory(phi_idx)
    phi = data["phi"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # (a) Timeline
    ax_a = fig.add_subplot(gs[0, 0])
    plot_timeline(ax_a)

    # (b) Slow state phase plane
    ax_b = fig.add_subplot(gs[0, 1])
    plot_slow_state_phase(ax_b, data)

    # (c) Fast state reset
    ax_c = fig.add_subplot(gs[1, 0])
    plot_fast_state_reset(ax_c, data)

    # (d) Time-domain contrast
    ax_d = fig.add_subplot(gs[1, 1])
    plot_time_domain_contrast(ax_d, data)

    # (e) Output verification
    ax_e_top = fig.add_subplot(gs[2, 0])
    ax_e_bottom = fig.add_subplot(gs[2, 1])
    plot_output_verification(ax_e_top, ax_e_bottom, data)

    fig.suptitle(f"Phase Trajectory Visualization — φ = {phi:.4f}",
                 y=0.98, fontweight="bold", fontsize=14)

    save_path = os.path.join(FIGURE_DIR, "main_figure_5panel.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_appendix_figures():
    """绘制附录图（其余 3 个 φ 的简化版）。"""
    for phi_idx in range(4):
        if phi_idx == 1:
            continue  # 主图已画

        data = _load_trajectory(phi_idx)
        phi = data["phi"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # (b) Slow state phase plane
        plot_slow_state_phase(axes[0, 0], data)

        # (e) Output verification
        plot_output_verification(axes[0, 1], axes[1, 0], data)

        # 隐藏多余子图
        axes[1, 1].axis("off")

        fig.suptitle(f"Appendix — φ = {phi:.4f}", y=0.98, fontweight="bold", fontsize=14)

        save_path = os.path.join(FIGURE_DIR, f"appendix_phi{phi_idx}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"Figure output: {FIGURE_DIR}")

    print("\nPlotting main figure...")
    plot_main_figure(phi_idx=1)

    print("\nPlotting appendix figures...")
    plot_appendix_figures()

    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
