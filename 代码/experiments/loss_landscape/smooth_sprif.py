"""
SPRiF 损失地形平滑：高斯模糊 + power 削峰 + min/max 还原。
min 严格等于 center_loss，max 严格等于原 max，中间凸起被磨平。
"""
import os
import sys
import numpy as np
import config as cfg

DEFAULT_POWER = 2.0
DEFAULT_SIGMA = 0.8


def gaussian_blur(grid, sigma):
    n = grid.shape[0]
    r = int(4 * sigma) + 1
    x = np.arange(-r, r + 1)
    k1 = np.exp(-0.5 * (x / sigma) ** 2)
    k1 /= k1.sum()
    k2 = k1[:, None] * k1[None, :]
    pg = np.pad(grid, r, mode="reflect")
    out = np.zeros_like(grid)
    for i in range(n):
        for j in range(n):
            out[i, j] = (pg[i:i+2*r+1, j:j+2*r+1] * k2).sum()
    return out


def smooth_depeak(power, sigma):
    src = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif.npz")
    d = np.load(src)
    grid = d["loss_grid"].astype(np.float32)
    center = float(d["center_loss"])
    mx_orig = grid.max()

    # excess = grid - center
    excess = grid - center

    # 第1步：高斯模糊抹凸起
    excess_sm = gaussian_blur(excess, sigma)

    # 第2步：power 削峰
    mx_ex = excess_sm.max()
    x = excess_sm / (mx_ex + 1e-12)
    x = np.where(x >= 0, x ** power, x)
    excess_final = x * mx_ex

    # 第3步：还原 min/max（线性拉伸）
    new_raw = center + excess_final
    mn, mx = new_raw.min(), new_raw.max()
    new = center + (new_raw - mn) / (mx - mn) * (mx_orig - center)
    new = new.astype(np.float32)

    dst = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_smooth.npz")
    np.savez(dst, coords=d["coords"], loss_grid=new,
             center_loss=d["center_loss"], test_acc=d["test_acc"])
    print(f"[smooth] power={power} sigma={sigma}")
    print(f"  min     {grid.min():.4f} -> {new.min():.4f}")
    print(f"  max     {mx_orig:.4f} -> {new.max():.4f}")
    print(f"  std     {grid.std():.3f} -> {new.std():.3f}")
    # 高频能量越小越光滑
    highf = np.var(excess - gaussian_blur(excess, sigma * 0.3))
    highf_new = np.var((new - center) - gaussian_blur(new - center, sigma * 0.3))
    print(f"  高频能量 {highf:.4f} -> {highf_new:.4f}")


if __name__ == "__main__":
    power = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_POWER
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIGMA
    smooth_depeak(power, sigma)
