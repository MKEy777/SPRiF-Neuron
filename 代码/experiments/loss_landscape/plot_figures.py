"""
绘图脚本：读取 landscape_{neuron}.npz 与 grad_{neuron}.npz，生成三类图。

  1. 2D filter-normalized 损失等高线（SPRiF / LIF 并排，标注最小点）
  2. 3D 损失曲面
  3. BPTT 梯度范数随时间步传播曲线（log y 轴）

所有图保存到 config.FIGURE_DIR。
"""
import os

import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (注册 3d 投影)

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


def _load_grad(neuron_type):
    path = os.path.join(cfg.CHECKPOINT_DIR, f"grad_{neuron_type}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "timesteps": data["timesteps"],
        "per_step_norm": data["per_step_norm"],
        "layer_names": data["layer_names"],
        "layer_grad_norms": data["layer_grad_norms"],
    }


# ---------------------------------------------------------------------------
# 1. 2D 损失等高线
# ---------------------------------------------------------------------------
def plot_contour(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[contour] 无 landscape 数据，跳过")
        return

    n = len(data)
    # 统一色标范围，两模型用同一 loss 尺度公平对比
    vmin = min(d["loss_grid"].min() for d in data.values())
    vmax = max(d["loss_grid"].max() for d in data.values())
    # 对数色标：拉开近零区域，凸显 SPRiF 又深又低幅值的盆地
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
        # 标注最小点
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


# ---------------------------------------------------------------------------
# 2. 3D 损失曲面
# ---------------------------------------------------------------------------
def plot_surface(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[surface] 无 landscape 数据，跳过")
        return

    n = len(data)
    # 统一 z 轴与色标范围，避免各子图独立缩放造成"谁更平坦"的视觉错觉
    zmin = min(d["loss_grid"].min() for d in data.values())
    zmax = max(d["loss_grid"].max() for d in data.values())
    # 对数色标：凸显 SPRiF 低幅值、深而平的底部
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


# ---------------------------------------------------------------------------
# 3. 1D 损失剖面（过极小点的切片）—— 最直观展示盆地宽度/陡峭度
# ---------------------------------------------------------------------------
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
        # 过最小点做两条正交切片
        idx = np.unravel_index(np.argmin(grid), grid.shape)
        prof_d1 = grid[:, idx[1]]      # 沿 direction 1
        prof_d2 = grid[idx[0], :]      # 沿 direction 2
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


# ---------------------------------------------------------------------------
# 3b. 归一化 1D 剖面（超额损失 / 自身中心损失）—— 消除绝对尺度差异，
#     直接对比"相对盆地宽度/平坦度"。曲线上升越慢 = 盆地越平越宽。
# ---------------------------------------------------------------------------
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
        # 超额损失相对中心损失的倍数
        prof_d1 = (grid[:, idx[1]] - center) / center
        prof_d2 = (grid[idx[0], :] - center) / center
        color = NEURON_COLOR.get(nt)
        label = NEURON_LABEL.get(nt, nt)
        # 标注"损失翻倍"的扰动半径（相对涨幅达到 1.0 处）
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


# ---------------------------------------------------------------------------
# 3c. 差值等高线图：LIF − SPRiF（同一坐标网格）。几乎全正 = SPRiF 在整片
#     地形上都更低，单图直观凸显 SPRiF 优势。
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. BPTT 梯度传播曲线
# ---------------------------------------------------------------------------
def plot_gradient(neurons):
    data = {nt: _load_grad(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[gradient] 无 grad 数据，跳过")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左：逐时间步梯度范数（log y），x=时间步（越靠左越早，梯度需回传越远）
    for nt, d in data.items():
        t = d["timesteps"]
        norm = d["per_step_norm"]
        ax1.semilogy(t, norm, label=NEURON_LABEL.get(nt, nt),
                     color=NEURON_COLOR.get(nt), linewidth=1.8)
    ax1.set_xlabel("time step t")
    ax1.set_ylabel(r"$\|\partial L / \partial h_t\|$ (log)")
    ax1.set_title("Gradient norm over time steps (full BPTT)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # 右：各层参数梯度范数柱状对比
    layer_names = None
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(data)) if len(data) > 1 else [0]
    for k, (nt, d) in enumerate(data.items()):
        names = [str(x) for x in d["layer_names"]]
        vals = d["layer_grad_norms"]
        layer_names = names
        xpos = np.arange(len(names)) + offsets[k]
        ax2.bar(xpos, vals, width=width, label=NEURON_LABEL.get(nt, nt),
                color=NEURON_COLOR.get(nt))
    if layer_names is not None:
        ax2.set_xticks(np.arange(len(layer_names)))
        ax2.set_xticklabels(layer_names)
    ax2.set_ylabel("param grad norm")
    ax2.set_title("Per-layer parameter gradient norm")
    ax2.set_yscale("log")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.suptitle("BPTT Gradient Propagation (SPRiF vs LIF)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "gradient_propagation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradient] 已保存 -> {out}")

    # 打印衰减比对比
    for nt, d in data.items():
        norm = d["per_step_norm"]
        ratio = norm[0] / (norm[-1] + 1e-12)
        print(f"  {NEURON_LABEL.get(nt, nt)}: 首/末梯度衰减比 = {ratio:.2e}")


# ---------------------------------------------------------------------------
# 5. 原版 vs 压平 对比图（SPRiF 自身，演示 α 压平效果）
# ---------------------------------------------------------------------------
def plot_flat_compare():
    base = _load_landscape("sprif")
    flat_path = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_flat.npz")
    if base is None or not os.path.exists(flat_path):
        print("[flat_compare] 缺少 landscape_sprif / landscape_sprif_flat，跳过")
        return
    fd = np.load(flat_path)
    flat = {
        "coords": fd["coords"],
        "loss_grid": fd["loss_grid"],
        "center_loss": float(fd["center_loss"]),
        "test_acc": float(fd["test_acc"]),
    }
    for d in (base, flat):
        d["label"] = (f"std={d['loss_grid'].std():.2f}  "
                      f"max={d['loss_grid'].max():.2f}")

    grids = [base, flat]
    # 统一对数色标，公平对比
    vmin = min(d["loss_grid"].min() for d in grids)
    vmax = max(d["loss_grid"].max() for d in grids)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    levels = np.logspace(np.log10(max(vmin, 1e-3)), np.log10(vmax), 25)

    # 上排：等高线并排
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    coords = base["coords"]
    X, Y = np.meshgrid(coords, coords)
    for d, ax in ((base, axes[0, 0]), (flat, axes[0, 1])):
        g = d["loss_grid"]
        cs = ax.contourf(X, Y, g.T, levels=levels, cmap="viridis", norm=norm)
        ax.contour(X, Y, g.T, levels=levels, colors="k",
                    linewidths=0.3, alpha=0.4)
        idx = np.unravel_index(np.argmin(g), g.shape)
        ax.plot(coords[idx[0]], coords[idx[1]], "r*", markersize=14)
        ax.plot(0, 0, "wo", markersize=6)
        ax.set_title(f"SPRiF {'压平' if d is flat else '原版'}\n{d['label']}")
        ax.set_xlabel("direction 1")
        ax.set_ylabel("direction 2")
    fig.colorbar(cs, ax=axes[0, :], label="loss (log scale)", shrink=0.7)

    # 下排：1D 剖面（过最小点）并排
    for d, ax in ((base, axes[1, 0]), (flat, axes[1, 1])):
        g = d["loss_grid"]
        idx = np.unravel_index(np.argmin(g), g.shape)
        c = coords
        ax.plot(c, g[:, idx[1]], "b-", linewidth=2, label="dir 1")
        ax.plot(c, g[idx[0], :], "g--", linewidth=2, label="dir 2")
        ax.set_title(f"1D profile ({'压平' if d is flat else '原版'})")
        ax.set_xlabel("perturbation")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("SPRiF: original vs flattened (α<1 compresses basin walls)",
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "sprif_flat_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[flat_compare] 已保存 -> {out}")


def plot_flat_compare_3d():
    base = _load_landscape("sprif")
    flat_path = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_flat.npz")
    if base is None or not os.path.exists(flat_path):
        print("[flat_compare_3d] 缺少 landscape_sprif / landscape_sprif_flat，跳过")
        return
    fd = np.load(flat_path)
    flat = {
        "coords": fd["coords"],
        "loss_grid": fd["loss_grid"],
        "center_loss": float(fd["center_loss"]),
        "test_acc": float(fd["test_acc"]),
    }
    grids = [base, flat]
    zmin = min(d["loss_grid"].min() for d in grids)
    zmax = max(d["loss_grid"].max() for d in grids)
    norm = mcolors.LogNorm(vmin=max(zmin, 1e-3), vmax=zmax)
    coords = base["coords"]
    X, Y = np.meshgrid(coords, coords)

    fig = plt.figure(figsize=(13, 6))
    for k, (d, tag) in enumerate(((base, "original"), (flat, "flattened"))):
        g = d["loss_grid"]
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        surf = ax.plot_surface(X, Y, g.T, cmap="viridis", norm=norm,
                                linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, label="loss (log)")
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"SPRiF {tag}\nstd={g.std():.2f}  max={g.max():.2f}")
        ax.set_xlabel("dir 1")
        ax.set_ylabel("dir 2")
        ax.set_zlabel("loss")
        ax.view_init(elev=45, azim=-60)
    fig.suptitle("SPRiF 3D Loss Surface: original vs flattened", fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "sprif_flat_compare_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[flat_compare_3d] 已保存 -> {out}")


# ---------------------------------------------------------------------------
# 6. 原版 vs 削峰 对比图（SPRiF 自身，演示 x^p 削峰效果）
# ---------------------------------------------------------------------------
def _load_depeak():
    path = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_depeak.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {
        "coords": d["coords"],
        "loss_grid": d["loss_grid"],
        "center_loss": float(d["center_loss"]),
        "test_acc": float(d["test_acc"]),
    }


def plot_depeak_compare_3d():
    base = _load_landscape("sprif")
    de = _load_depeak()
    if base is None or de is None:
        print("[depeak] 缺少 landscape_sprif / landscape_sprif_depeak，跳过")
        return

    grids = [base, de]
    zmin = min(d["loss_grid"].min() for d in grids)
    zmax = max(d["loss_grid"].max() for d in grids)
    norm = mcolors.LogNorm(vmin=max(zmin, 1e-3), vmax=zmax)
    coords = base["coords"]
    X, Y = np.meshgrid(coords, coords)

    fig = plt.figure(figsize=(13, 6))
    for k, (d, tag) in enumerate(((base, "original"), (de, "de-peaked (keep min/max)"))):
        g = d["loss_grid"]
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        surf = ax.plot_surface(X, Y, g.T, cmap="viridis", norm=norm,
                                linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, label="loss (log)")
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"SPRiF {tag}\nstd={g.std():.2f}  max={g.max():.2f}"
                     f"  min={g.min():.4f}")
        ax.set_xlabel("dir 1")
        ax.set_ylabel("dir 2")
        ax.set_zlabel("loss")
        ax.view_init(elev=45, azim=-60)
    fig.suptitle("SPRiF 3D Loss Surface: original vs de-peaked (x^2)",
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "sprif_depeak_compare_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[depeak_3d] 已保存 -> {out}")


def plot_depeak_compare():
    base = _load_landscape("sprif")
    de = _load_depeak()
    if base is None or de is None:
        print("[depeak] 缺少 landscape_sprif / landscape_sprif_depeak，跳过")
        return

    grids = [base, de]
    vmin = min(d["loss_grid"].min() for d in grids)
    vmax = max(d["loss_grid"].max() for d in grids)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    levels = np.logspace(np.log10(max(vmin, 1e-3)), np.log10(vmax), 25)
    coords = base["coords"]
    X, Y = np.meshgrid(coords, coords)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for d, ax, tag in ((base, axes[0, 0], "original"),
                        (de, axes[0, 1], "de-peaked (x^2)")):
        g = d["loss_grid"]
        cs = ax.contourf(X, Y, g.T, levels=levels, cmap="viridis", norm=norm)
        ax.contour(X, Y, g.T, levels=levels, colors="k",
                    linewidths=0.3, alpha=0.4)
        idx = np.unravel_index(np.argmin(g), g.shape)
        ax.plot(coords[idx[0]], coords[idx[1]], "r*", markersize=14)
        ax.plot(0, 0, "wo", markersize=6)
        ax.set_title(f"SPRiF {tag}\nmax={g.max():.2f}  std={g.std():.2f}")
        ax.set_xlabel("direction 1")
        ax.set_ylabel("direction 2")
    fig.colorbar(cs, ax=axes[0, :], label="loss (log scale)", shrink=0.7)

    for d, ax, tag in ((base, axes[1, 0], "original"),
                        (de, axes[1, 1], "de-peaked (x^2)")):
        g = d["loss_grid"]
        idx = np.unravel_index(np.argmin(g), g.shape)
        c = coords
        ax.plot(c, g[:, idx[1]], "b-", linewidth=2, label="dir 1")
        ax.plot(c, g[idx[0], :], "g--", linewidth=2, label="dir 2")
        ax.set_title(f"1D profile ({tag})")
        ax.set_xlabel("perturbation")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("SPRiF: original vs de-peaked (x^2, keep min/max)",
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(cfg.FIGURE_DIR, "sprif_depeak_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[depeak] 已保存 -> {out}")


def main():
    os.makedirs(cfg.FIGURE_DIR, exist_ok=True)
    neurons = ["sprif", "lif"]
    plot_contour(neurons)
    plot_surface(neurons)
    plot_combined(neurons)
    plot_profile(neurons)
    plot_profile_normalized(neurons)
    plot_difference(neurons)
    plot_gradient(neurons)
    plot_flat_compare()
    plot_flat_compare_3d()
    plot_depeak_compare()
    plot_depeak_compare_3d()
    print(f"\nAll figures saved to: {cfg.FIGURE_DIR}")


# ---------------------------------------------------------------------------
# 7. 2D + 3D 联合图——论文用（简洁，无冗余标注）
# ---------------------------------------------------------------------------
def plot_combined(neurons):
    data = {nt: _load_landscape(nt) for nt in neurons}
    data = {nt: d for nt, d in data.items() if d is not None}
    if not data:
        print("[combined] 无 landscape 数据，跳过")
        return

    # ---- Figure contract ----
    # Core conclusion: SPRiF achieves lower loss and flatter minima than LIF
    # Evidence: (a) 2D contour magnitude comparison (b) 3D basin shape
    # Archetype: quantitative grid (2×2)
    # Backend: Python (exclusive)

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

        # --- 2D contour (column 0) ---
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

        # --- 3D surface (column 1) ---
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

    # 分栏标题（自动对齐到列中心）
    ax_col0 = fig.axes[0]  # 第一行第一列 = contour
    center_col0 = (ax_col0.get_position().x0 + ax_col0.get_position().x1) / 2
    ax_col1 = fig.axes[1]  # 第一行第二列 = 3D
    center_col1 = (ax_col1.get_position().x0 + ax_col1.get_position().x1) / 2
    fig.text(center_col0, 0.96, "(a) 2D contours", fontsize=14,
             fontweight="bold", va="top", ha="center")
    fig.text(center_col1, 0.96, "(b) 3D surfaces", fontsize=14,
             fontweight="bold", va="top", ha="center")

    # 色条（紧贴3D列，跨两行）
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