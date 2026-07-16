"""
S-MNIST 损失景观实验 — 统一网络封装。

复用现有神经元层：
  - SPRiF: 代码/Task_S-MNIST/core_algorithm/sprif_layer.py
  - LIF:   代码/Task_GSC/core_algorithm/lif_layer.py
两者接口一致（forward / init_state / forward_with_state），
因此可用同一网络壳 SMNISTNet 切换神经元类型，保证结构与参数量可比。
"""
import os
import sys
from typing import List

import torch
import torch.nn as nn

from config import SMNIST_DIR, GSC_DIR

# SPRiF 与 LIF 分别位于 Task_S-MNIST / Task_GSC 下，两个任务目录都存在同名的
# core_algorithm 包。若用 `import core_algorithm.xxx` 会因包名冲突而只解析到
# sys.path 中第一个 core_algorithm，导致加载到错误目录的旧版实现
# （报错 'SPRiFNeuronLayer' object has no attribute 'forward_with_state'）。
# 因此这里两者都用 importlib 按绝对文件路径显式加载，彻底��免命名冲突。
import importlib.util  # noqa: E402


def _load_class(module_alias: str, file_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_alias, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


_sprif_path = os.path.join(SMNIST_DIR, "core_algorithm", "sprif_layer.py")
SPRiFNeuronLayer = _load_class("sprif_layer_smnist", _sprif_path, "SPRiFNeuronLayer")

# LIF 内部有 `from core_algorithm.sprif_layer import surrogate_spike`，
# 需要 sys.path 中能解析到 GSC 自己的 core_algorithm 包。加载前把 GSC_DIR
# 放到 sys.path 首��，确保解析到 Task_GSC 版本而非 Task_S-MNIST。
if GSC_DIR in sys.path:
    sys.path.remove(GSC_DIR)
sys.path.insert(0, GSC_DIR)
_lif_path = os.path.join(GSC_DIR, "core_algorithm", "lif_layer.py")
LIFNeuronLayer = _load_class("lif_layer_gsc", _lif_path, "LIFNeuronLayer")


def _build_layer(neuron_type: str, in_dim: int, hidden: int, recurrent: bool):
    """按神经元类型构建一层。"""
    if neuron_type == "sprif":
        return SPRiFNeuronLayer(
            input_size=in_dim,
            hidden_size=hidden,
            recurrent=recurrent,
            threshold=1.0,
        )
    elif neuron_type == "lif":
        return LIFNeuronLayer(
            input_size=in_dim,
            hidden_size=hidden,
            recurrent=recurrent,
            threshold=1.0,
        )
    raise ValueError(f"unknown neuron_type: {neuron_type}")


class SMNISTNet(nn.Module):
    """统一 S-MNIST 网络壳，支持 sprif / lif 两种神经元。"""

    def __init__(
        self,
        neuron_type: str,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int = 10,
        mode: str = "srnn",
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.neuron_type = neuron_type
        recurrent = mode.lower() == "srnn"
        self.warmup_steps = warmup_steps
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h in hidden_sizes:
            self.layers.append(_build_layer(neuron_type, in_dim, h, recurrent))
            in_dim = h
        self.readout = nn.Linear(hidden_sizes[-1], num_classes)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.constant_(self.readout.bias, 0.0)

    def forward(self, x):
        """全 BPTT 前向（评估用）。x: (B, T, F)。"""
        out = x
        for layer in self.layers:
            out = layer(out, batch_first=True)
        logits_t = self.readout(out)
        if logits_t.size(1) <= self.warmup_steps:
            logits = logits_t.mean(dim=1)
        else:
            logits = logits_t[:, self.warmup_steps:, :].mean(dim=1)
        return logits

    def forward_record_hidden(self, x):
        """前向并返回逐时间步的第一隐层输出（用于梯度可视化）。

        为记录 ∂L/∂h_t，这里保留第一隐层每个时间步 spike 张量的计算图。
        返回 (logits, hidden_seq)，hidden_seq: (B, T, H1) requires_grad。
        """
        out = x
        hidden_seq = None
        for i, layer in enumerate(self.layers):
            out = layer(out, batch_first=True)
            if i == 0:
                out.retain_grad()
                hidden_seq = out
        logits_t = self.readout(out)
        logits = logits_t.mean(dim=1)
        return logits, hidden_seq

    def init_states(self, batch_size, device=None, dtype=None):
        return [layer.init_state(batch_size, device=device, dtype=dtype)
                for layer in self.layers]

    @staticmethod
    def detach_states(states):
        return [{k: v.detach() for k, v in state.items()} for state in states]


def build_model(neuron_type: str, cfg) -> SMNISTNet:
    """依据 config 模块构建模型。"""
    return SMNISTNet(
        neuron_type=neuron_type,
        input_size=cfg.INPUT_SIZE,
        hidden_sizes=list(cfg.HIDDEN_SIZES),
        num_classes=cfg.NUM_CLASSES,
        mode=cfg.MODE,
        warmup_steps=cfg.WARMUP_STEPS,
    )


__all__ = ["SMNISTNet", "build_model", "SPRiFNeuronLayer", "LIFNeuronLayer"]