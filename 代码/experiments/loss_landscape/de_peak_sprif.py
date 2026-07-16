"""
削峰版 SPRiF：保持最小值/最大值不变，用 x^p 压缩中间峰值。
p>1 → 中间区域向最小值收缩，两端固定。
"""
import os
import numpy as np
import config as cfg

P = 2.0  # >1 压峰；=1 不变；越大中间压得越低


def de_peak(power):
    src = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif.npz")
    d = np.load(src)
    grid = d["loss_grid"].astype(np.float32)
    center = float(d["center_loss"])
    mx = grid.max()
    # 归一化 → 幂压缩 → 还原
    x = (grid - center) / (mx - center + 1e-12)
    x = np.where(x >= 0, x ** power, x)  # 理论上 x>=0 全成立
    new = center + x * (mx - center)
    new = new.astype(np.float32)

    dst = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_depeak.npz")
    np.savez(dst, coords=d["coords"], loss_grid=new,
             center_loss=d["center_loss"], test_acc=d["test_acc"])
    print(f"[de_peak] p={power} -> {dst}")
    print(f"  min   {grid.min():.4f} -> {new.min():.4f}  (理应不变)")
    print(f"  max   {grid.max():.4f} -> {new.max():.4f}  (理应不变)")
    print(f"  std   {grid.std():.3f} -> {new.std():.3f}")
    print(f"  mid   {grid[grid.shape[0]//2, grid.shape[1]//4]:.3f} -> "
          f"{new[grid.shape[0]//2, grid.shape[1]//4]:.3f}  (示例中间值)")


if __name__ == "__main__":
    de_peak(P)
