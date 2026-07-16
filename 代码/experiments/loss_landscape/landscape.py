"""
S-MNIST 损失景观 — filter-normalized 2D/3D 计算（Li et al. 2018）。

方法：
  1. 加载训练好的权重 θ*
  2. 采样两个随机方向 d1, d2，按 filter-normalization 归一化：
       对每个权重张量 w，d <- d/‖d‖_F * ‖w‖_F （逐 filter/张量归一化）
  3. 在网格 (a,b)∈[-R,R]² 上设置 θ = θ* + a·d1 + b·d2，前向计算 loss
  4. 保存 loss 网格到 npz，供绘图

为省显存，loss 用 TBPTT 分块无梯度前向计算（与训练一致），
仅在测试集若干 batch 上估计。
"""
import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from config import get_args
from data import get_loaders
from models import build_model


@torch.no_grad()
def _get_random_directions(weights, seed):
    """生成一个 filter-normalized 随机方向（与 weights 同结构的 list）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    directions = []
    for w in weights:
        d = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
        # filter-normalization：逐张量按 Frobenius 范数缩放到与 w 同尺度
        if w.dim() <= 1:
            # 偏置/一维参数：不扰动（Li et al. 建议忽略 BN/bias，这里置零更稳）
            d.zero_()
        else:
            d = d / (d.norm() + 1e-10) * w.norm()
        directions.append(d)
    return directions


@torch.no_grad()
def _set_weights(model, base_weights, d1, d2, a, b):
    """θ = θ* + a·d1 + b·d2，就地写入模型参数。"""
    for p, w0, du, dv in zip(model.parameters(), base_weights, d1, d2):
        p.copy_(w0 + a * du + b * dv)


@torch.no_grad()
def _eval_loss(model, x_batches, y_batches, criterion):
    """用 TBPTT 分块前向（无梯度）计算平均 loss。"""
    model.eval()
    total_loss, total_n = 0.0, 0
    tbptt_len = cfg.TBPTT_LEN
    for x, y in zip(x_batches, y_batches):
        B, T, _ = x.shape
        states = model.init_states(B, device=x.device, dtype=x.dtype)
        logits_list = []
        for start in range(0, T, tbptt_len):
            end = min(start + tbptt_len, T)
            out = x[:, start:end]
            new_states = []
            for i, layer in enumerate(model.layers):
                out, ns = layer.forward_with_state(out, states[i], batch_first=True)
                new_states.append(ns)
            logits_chunk = model.readout(out).mean(dim=1)
            logits_list.append(logits_chunk)
            states = new_states
        logits = torch.stack(logits_list).mean(dim=0)
        loss = criterion(logits, y)
        total_loss += loss.item() * B
        total_n += B
    return total_loss / max(total_n, 1)


def compute_landscape(neuron_type, args, device):
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{neuron_type}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[跳过] 未找到 checkpoint: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(neuron_type, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 基准权重与两个随机方向
    base_weights = [p.detach().clone() for p in model.parameters()]
    d1 = _get_random_directions(base_weights, cfg.DIRECTION_SEED)
    d2 = _get_random_directions(base_weights, cfg.DIRECTION_SEED + 1)

    # 预取若干个测试 batch 用于估计 loss
    _, test_loader = get_loaders(args.batch_size, device, subset=args.subset)
    x_batches, y_batches = [], []
    for i, (x, y) in enumerate(test_loader):
        if i >= args.eval_batches:
            break
        x_batches.append(x.to(device))
        y_batches.append(y.to(device))

    criterion = nn.CrossEntropyLoss()
    N = args.grid_resolution
    R = args.grid_range
    coords = np.linspace(-R, R, N)
    loss_grid = np.zeros((N, N), dtype=np.float32)

    print(f"[{neuron_type}] 计算 {N}x{N} 损失景观 ...")
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            _set_weights(model, base_weights, d1, d2, float(a), float(b))
            loss_grid[i, j] = _eval_loss(model, x_batches, y_batches, criterion)
        print(f"  row {i+1}/{N} done (a={a:+.3f})")

    # 复原权重
    _set_weights(model, base_weights, d1, d2, 0.0, 0.0)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.CHECKPOINT_DIR, f"landscape_{neuron_type}.npz")
    np.savez(out_path, coords=coords, loss_grid=loss_grid,
             center_loss=loss_grid[N // 2, N // 2],
             test_acc=ckpt.get("test_acc", -1))
    print(f"[{neuron_type}] 损失景观已保存 -> {out_path} "
          f"(center loss={loss_grid[N//2, N//2]:.4f})")
    return out_path


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    neurons = ["sprif", "lif"] if args.neuron == "both" else [args.neuron]
    for nt in neurons:
        compute_landscape(nt, args, device)


if __name__ == "__main__":
    main()