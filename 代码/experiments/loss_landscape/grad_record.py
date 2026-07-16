"""
S-MNIST 梯度可视化 — BPTT 梯度沿时间步传播分析。

核心 claim：SPRiF 的连续慢状态使梯度沿 784 步长序列传播更稳定，
而 LIF 因膜电位被 spike 重置，梯度随回溯步数快速衰减（梯度消失）。

方法：
  - 用小 batch 做 **全 BPTT**（非 TBPTT，才能看到跨全序列的梯度传播）
  - 记录第一隐层逐时间步隐状态 h_t 的梯度范数 ‖∂L/∂h_t‖
  - x 轴为时间步 t（越靠前=回溯越深），对比两模型衰减曲线

注意：梯度分析必须用全 BPTT；训练用 TBPTT，二者目的不同。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from config import get_args
from data import get_single_batch
from models import build_model

sys.path.insert(0, cfg.SMNIST_DIR)
from core_algorithm.utils import set_seed  # noqa: E402


def record_gradients(neuron_type, args, device):
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{neuron_type}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[跳过] 未找到 checkpoint: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(neuron_type, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.train()  # 需要梯度

    x, y = get_single_batch(cfg.GRAD_BATCH_SIZE, device)
    criterion = nn.CrossEntropyLoss()

    model.zero_grad(set_to_none=True)
    # 全 BPTT 前向，保留第一隐层逐步隐状态梯度
    logits, hidden_seq = model.forward_record_hidden(x)  # hidden_seq: (B, T, H1)
    loss = criterion(logits, y)
    loss.backward()

    # ∂L/∂h_t 的逐时间步范数（对 batch 与 hidden 维取 L2 后平均到每步）
    grad = hidden_seq.grad  # (B, T, H1)
    T = grad.shape[1]
    # 每个时间步：先对 hidden 维求范数，再对 batch 求均值
    per_step_norm = grad.norm(dim=2).mean(dim=0).detach().cpu().numpy()  # (T,)

    # 各层参数梯度范数（衡量整体梯度尺度）
    param_grad_norms = {}
    for i, layer in enumerate(model.layers):
        total = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        param_grad_norms[f"layer{i}"] = float(total ** 0.5)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.CHECKPOINT_DIR, f"grad_{neuron_type}.npz")
    np.savez(
        out_path,
        timesteps=np.arange(T),
        per_step_norm=per_step_norm.astype(np.float32),
        layer_names=np.array(list(param_grad_norms.keys())),
        layer_grad_norms=np.array(list(param_grad_norms.values()), dtype=np.float32),
    )
    print(f"[{neuron_type}] 梯度记录已保存 -> {out_path}")
    print(f"  首步(t=0)梯度范数={per_step_norm[0]:.3e}, "
          f"末步(t={T-1})={per_step_norm[-1]:.3e}, "
          f"衰减比={per_step_norm[0] / (per_step_norm[-1] + 1e-12):.2e}")
    print(f"  各层参数梯度范数: {param_grad_norms}")
    return out_path


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    neurons = ["sprif", "lif"] if args.neuron == "both" else [args.neuron]
    for nt in neurons:
        record_gradients(nt, args, device)


if __name__ == "__main__":
    main()