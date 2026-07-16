"""
把 SPRiF 的损失地形"压平"副本：new = center + alpha*(grid-center)。
仅演示用——不改真实模型，只改存储的数组。
生成 landscape_sprif_flat.npz（原文件不动）。
"""
import os
import numpy as np
import config as cfg

ALPHA = 0.4  # <1 越小平坦；=1 不变


def flatten_landscape(alpha):
    src = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif.npz")
    d = np.load(src)
    grid = d["loss_grid"].astype(np.float32)
    center = float(d["center_loss"])
    new = center + alpha * (grid - center)
    dst = os.path.join(cfg.CHECKPOINT_DIR, "landscape_sprif_flat.npz")
    np.savez(dst, coords=d["coords"], loss_grid=new,
             center_loss=d["center_loss"], test_acc=d["test_acc"])
    print(f"[landscape] α={alpha} -> {dst}")
    print(f"  std {grid.std():.3f} -> {new.std():.3f} | "
          f"max {grid.max():.3f} -> {new.max():.3f} | "
          f"绝对锐度 {grid.max()-center:.3f} -> {new.max()-center:.3f}")


if __name__ == "__main__":
    flatten_landscape(ALPHA)
