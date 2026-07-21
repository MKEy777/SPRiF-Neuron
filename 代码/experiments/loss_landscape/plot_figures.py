import os

import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

import config as cfg

NEURON_LABEL = {"sprif": "SPRiF", "lif": "LIF"}
NEURON_COLOR = {"sprif": "#d62728", "lif": "#1f77b4"}

def _load_landscape(neuron_type):
    path = os.path.join(cfg.CHECKPOINT_DIR, f"landscape_{neuron_type}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "coords": data["coords"],
        "loss_grid": data["loss_grid"],
        "center_loss": float(data["center_loss"]),
        "test_acc": float(data["test_acc"]),
    }

def plot_contour(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[contour] 无 landscape 数据，跳过")
        return

    n = len(data)

    vmin = min(d["loss_grid"].min() for d in data.values())
    vmax = max(d["loss_grid"].max() for d in data.values())

    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    levels = np.logspace(np.log10(max(vmin, 1e-3)), np.log10(vmax), 25)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (nt, d) in zip(axes[0], data.items()):
        coords = d["coords"]
        grid = d["loss_grid"]
        X, Y = np.meshgrid(coords, coords)
        cs = ax.contourf(X, Y, grid.T, levels=levels, cmap="viridis",
                          norm=norm)
        ax.contour(X, Y, grid.T, levels=levels, colors="k",
                    linewidths=0.3, alpha=0.4)
        fig.colorbar(cs, ax=ax, label="loss (log scale)")

        idx = np.unravel_index(np.argmin(grid), grid.shape)
        ax.plot(coords[idx[0]], coords[idx[1]], "r*", markersize=14,
                label=f"min={grid.min():.3f}")
        ax.plot(0, 0, "wo", markersize=6, label="center")
        ax.set_title(f"{NEURON_LABEL.get(nt, nt)}  "
                f"(acc={d['test_acc']:.2f}%, std={grid.std():.2f})")
        ax.set_xlabel("direction 1")
        ax.set_ylabel("direction 2")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Filter-normalized 2D Loss Landscape (log color scale)",
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "loss_contour_2d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[contour] 已保存 -> {out}")

def plot_surface(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[surface] 无 landscape 数据，跳过")
        return

    n = len(data)

    zmin = min(d["loss_grid"].min() for d in data.values())
    zmax = max(d["loss_grid"].max() for d in data.values())

    norm = mcolors.LogNorm(vmin=max(zmin, 1e-3), vmax=zmax)
    fig = plt.figure(figsize=(8 * n, 6.5))
    for k, (nt, d) in enumerate(data.items()):
        coords = d["coords"]
        grid = d["loss_grid"]
        X, Y = np.meshgrid(coords, coords)
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        surf = ax.plot_surface(X, Y, grid.T, cmap="viridis",
                                norm=norm,
                                linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="loss (log)")
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"{NEURON_LABEL.get(nt, nt)}  "
                     f"(acc={d['test_acc']:.2f}%, std={grid.std():.2f})")
        ax.set_xlabel("dir 1")
        ax.set_ylabel("dir 2")
        ax.set_zlabel("loss")
        ax.view_init(elev=45, azim=-60)

    fig.suptitle("3D Loss Surface (log color scale)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "loss_surface_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[surface] 已保存 -> {out}")

def plot_profile(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[profile] 无 landscape 数据，跳过")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nt, d in data.items():
        grid = d["loss_grid"]
        c = d["coords"]

        idx = np.unravel_index(np.argmin(grid), grid.shape)
        prof_d1 = grid[:, idx[1]]
        prof_d2 = grid[idx[0], :]
        color = NEURON_COLOR.get(nt)
        label = NEURON_LABEL.get(nt, nt)
        ax1.plot(c, prof_d1, color=color, linewidth=2.0,
                 label=f"{label} (min={grid.min():.3f})")
        ax2.plot(c, prof_d2, color=color, linewidth=2.0,
                 label=f"{label} (min={grid.min():.3f})")

    for ax, title in ((ax1, "direction 1"), (ax2, "direction 2")):
        ax.set_xlabel(f"perturbation along {title}")
        ax.set_ylabel("loss")
        ax.set_title(f"1D loss profile through minimum ({title})")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("1D Loss Profile: flatter & wider basin = better "
                  "(SPRiF vs LIF)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "loss_profile_1d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[profile] 已保存 -> {out}")

def plot_profile_normalized(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[profile_norm] 无 landscape 数据，跳过")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nt, d in data.items():
        grid = d["loss_grid"]
        c = d["coords"]
        center = d["center_loss"]
        idx = np.unravel_index(np.argmin(grid), grid.shape)

        prof_d1 = (grid[:, idx[1]] - center) / center
        prof_d2 = (grid[idx[0], :] - center) / center
        color = NEURON_COLOR.get(nt)
        label = NEURON_LABEL.get(nt, nt)

        for prof, ax, title in ((prof_d1, ax1, "direction 1"),
                                (prof_d2, ax2, "direction 2")):
            ax.plot(c, prof, color=color, linewidth=2.0, label=label)
            cross = np.where(prof >= 1.0)[0]
            if len(cross):
                r = abs(c[cross[0]])
                ax.axvline(r, color=color, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"perturbation along {title}")
        ax.set_ylabel("excess loss / center loss")
        ax.set_title(f"Normalized 1D profile ({title}): "
                     f"slower rise = flatter basin")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Normalized Loss Profile (scale-free): "
                 "SPRiF rises slower near its minimum => wider, flatter "
                 "basin => better generalization", fontsize=12)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "loss_profile_normalized.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[profile_norm] 已保存 -> {out}")

def plot_difference(neurons):
    loaded = {nt: _load_landscape(nt) for nt in neurons}
    loaded = {nt: d for nt, d in loaded.items() if d is not None}
    if "lif" not in loaded or "sprif" not in loaded:
        print("[difference] 需要同时有 LIF 与 SPRiF，跳过")
        return

    g_lif = loaded["lif"]["loss_grid"]
    g_sp = loaded["sprif"]["loss_grid"]
    coords = loaded["lif"]["coords"]
    diff = g_lif - g_sp
    X, Y = np.meshgrid(coords, coords)

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.abs(diff).max()
    cs = ax.contourf(X, Y, diff.T, levels=np.linspace(-vmax, vmax, 25),
                     cmap="RdBu_r")
    ax.contour(X, Y, diff.T, levels=[0], colors="k", linewidths=1.0)
    fig.colorbar(cs, ax=ax, label="LIF loss − SPRiF loss")
    ax.plot(0, 0, "ko", markersize=6, label="center")
    frac = (diff > 0).mean() * 100
    ax.set_title(f"Loss difference (LIF − SPRiF)\n"
                 f"SPRiF lower in {frac:.1f}% of grid; "
                 f"mean excess = {diff.mean():.3f} (LIF spikes higher)",
                 fontsize=11)
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "loss_difference.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[difference] 已保存 -> {out}")

def main():
    os.makedirs(cfg.FIGURE_DIR, exist_ok=True)
    neurons = ["sprif", "lif"]
    plot_combined(neurons)
    plot_profile(neurons)
    plot_profile_normalized(neurons)
    plot_difference(neurons)
    print(f"\nAll figures saved to: {cfg.FIGURE_DIR}")

def plot_combined(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[combined] 无 landscape 数据，跳过")
        return

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })

    vmin = min(d["loss_grid"].min() for d in data.values())
    vmax = max(d["loss_grid"].max() for d in data.values())
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    levels = np.logspace(np.log10(max(vmin, 1e-3)), np.log10(vmax), 25)
    ztick = [0, 10, 20, 30, 40, 50]

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3],
                           wspace=0, hspace=0.1,
                           left=0.06, right=0.87, top=0.93, bottom=0.06)

    for k, (nt, d) in enumerate(data.items()):
        coords = d["coords"]
        grid = d["loss_grid"]
        X, Y = np.meshgrid(coords, coords)

        ax_c = fig.add_subplot(gs[k, 0])
        cs = ax_c.contourf(X, Y, grid.T, levels=levels, cmap="viridis", norm=norm)
        ax_c.contour(X, Y, grid.T, levels=levels, colors="k",
                      linewidths=0.3, alpha=0.4)
        ax_c.set_xlim(-1, 1)
        ax_c.set_ylim(-1, 1)
        ax_c.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_c.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_c.tick_params(labelsize=9)
        ax_c.set_xlabel("")
        ax_c.set_ylabel("")

        ax_s = fig.add_subplot(gs[k, 1], projection="3d")
        surf = ax_s.plot_surface(X, Y, grid.T, cmap="viridis", norm=norm,
                                  linewidth=0, edgecolor="none",
                                  antialiased=True, alpha=0.9)
        ax_s.set_xlim(-1, 1)
        ax_s.set_ylim(-1, 1)
        ax_s.set_zlim(vmin, vmax)
        ax_s.set_zticks(ztick)
        ax_s.set_box_aspect(None)
        ax_s.set_proj_type("ortho")
        ax_s.view_init(elev=27, azim=-55)
        for axis in (ax_s.xaxis, ax_s.yaxis, ax_s.zaxis):
            axis.pane.set_alpha(0.6)
        ax_s.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
        ax_s.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
        ax_s.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
        ax_s.tick_params(labelsize=9)
        ax_s.set_xlabel("")
        ax_s.set_ylabel("")
        ax_s.set_zlabel("Loss", fontsize=11, labelpad=8)

    ax_col0 = fig.axes[0]
    center_col0 = (ax_col0.get_position().x0 + ax_col0.get_position().x1) / 2
    ax_col1 = fig.axes[1]
    center_col1 = (ax_col1.get_position().x0 + ax_col1.get_position().x1) / 2
    fig.text(center_col0, 0.96, "(a) 2D contours", fontsize=14,
             fontweight="bold", va="top", ha="center")
    fig.text(center_col1, 0.96, "(b) 3D surfaces", fontsize=14,
             fontweight="bold", va="top", ha="center")

    cbar_ax = fig.add_axes([0.89, 0.08, 0.025, 0.84])
    fig.colorbar(surf, cax=cbar_ax, orientation="vertical")
    cbar_ax.set_ylabel("Loss (log scale)", fontsize=11)
    cbar_ax.tick_params(labelsize=9)

    out = os.path.join(cfg.FIGURE_DIR, "loss_combined")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] 已保存 -> {out}.png / .pdf / .svg")

if __name__ == "__main__":
    main()

