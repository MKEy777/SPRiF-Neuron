"""
相位轨迹可视化实验 — 模型定义。

SPRiF 和 ASRNN 对照网络，均支持 perturbation 注入。
"""
import os
import sys
import torch
import torch.nn as nn

# 添加路径以导入 SPRiF 和 ASRNN
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_PSMNIST = os.path.join(ROOT, "Task_pSMNIST")
TASK_GSC = os.path.join(ROOT, "Task_GSC")
sys.path.insert(0, TASK_PSMNIST)
sys.path.insert(0, TASK_GSC)

from core_algorithm.sprif_layer import SPRiFNeuronLayer, surrogate_spike
from SRNN_layers.spike_dense import spike_dense
from SRNN_layers.spike_neuron import mem_update_adp

from config import HIDDEN_SIZE, A_PROBE


class SPRiFTrajectoryNet(nn.Module):
    """SPRiF 轨迹网络：慢状态 x_t 作为记忆载体，不被 spike 重置。"""

    def __init__(self, input_size: int = 32, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.layer = SPRiFNeuronLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            recurrent=False,
            threshold=1.0,
        )
        # readout 从慢状态 x_t [B,H,3] flatten 读
        self.readout = nn.Linear(hidden_size * 3, 2)

    def forward(self, x: torch.Tensor, probe_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，支持 perturbation 注入。

        Args:
            x: [B, T, 32] 输入脉冲
            probe_mask: [B, T] probe 注入掩码

        Returns:
            readout_seq: [B, T, 2] 输出轨迹
            spike_rate: scalar 平均发放率（用于正则）
        """
        B, T, _ = x.shape
        device = x.device
        state = self.layer.init_state(B, device=device, dtype=x.dtype)
        runtime = self.layer._precompute_runtime_params()

        readout_seq = []
        total_spikes = 0

        for t in range(T):
            x_t = x[:, t, :]  # [B, 32]
            probe_m = probe_mask[:, t]  # [B]

            # 计算 input_current（含 perturbation）
            input_current = self.layer.input_linear(x_t)
            # probe_mask 广播到 [B, H]
            input_current = input_current + A_PROBE * probe_m.unsqueeze(-1)

            spike, membrane, state = self.layer.forward_step(
                x_t, state, runtime, input_current=input_current
            )

            # readout 从慢状态 x_t 读
            x_state = state["x"]  # [B, H, 3]
            readout = self.readout(x_state.reshape(B, -1))  # [B, 2]
            readout_seq.append(readout)

            total_spikes += spike.sum()

        readout_seq = torch.stack(readout_seq, dim=1)  # [B, T, 2]
        spike_rate = total_spikes / (B * T * self.layer.hidden_size)

        return readout_seq, spike_rate

    def forward_step_record(
        self,
        x_t: torch.Tensor,
        state: dict,
        runtime: dict,
        probe_mask_t: torch.Tensor,
    ) -> tuple:
        """
        单步 forward，额外记录 u_tilde (pre-reset) 和 u_next (post-reset)。

        Returns:
            spike, membrane, next_state, u_tilde, u_next
        """
        x_state = state["x"]
        u_state = state["u"]
        prev_spike = state["prev_spike"]

        # 计算 input_current（含 perturbation）
        input_current = self.layer.input_linear(x_t)
        input_current = input_current + A_PROBE * probe_mask_t.unsqueeze(-1)

        # 慢状态更新
        x_next = self.layer._slow_flow(x_state, input_current, runtime)

        # 快状态更新（pre-reset）
        u_tilde = self.layer._fast_flow(u_state, x_next, runtime)

        # 膜电位 = u_tilde 的第一维
        membrane = u_tilde[..., 0]
        theta = self.layer.threshold
        spike = surrogate_spike(membrane - theta)

        # 投影重置（post-reset）
        if isinstance(theta, torch.Tensor):
            reset_scale = theta
        else:
            reset_scale = torch.as_tensor(theta, device=u_tilde.device, dtype=u_tilde.dtype)

        u_next = (
            u_tilde
            - spike.unsqueeze(-1)
            * runtime["reset_direction"].unsqueeze(0)
            * reset_scale.unsqueeze(-1)
        )

        next_state = {
            "x": x_next,
            "u": u_next,
            "prev_spike": spike,
        }

        return spike, membrane, next_state, u_tilde, u_next


class ASRNNTrajectoryNet(nn.Module):
    """ASRNN 轨迹网络：膜电位 mem 作为记忆载体，被 spike 重置。"""

    def __init__(self, input_size: int = 32, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.asrnn_layer = spike_dense(
            input_dim=input_size,
            output_dim=hidden_size,
            tauM=20,
            tauAdp_inital=200,
            is_adaptive=1,
            device="cpu",
        )
        # readout 从膜电位 mem [B, H] 读
        self.readout = nn.Linear(hidden_size, 2)

    def set_neuron_state(self, batch_size: int):
        """初始化 ASRNN 内部状态。"""
        self.asrnn_layer.set_neuron_state(batch_size)

    def forward(self, x: torch.Tensor, probe_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，支持 perturbation 注入。

        Args:
            x: [B, T, 32] 输入脉冲
            probe_mask: [B, T] probe 注入掩码

        Returns:
            readout_seq: [B, T, 2] 输出轨迹
            spike_rate: scalar 平均发放率（用于正则）
        """
        B, T, _ = x.shape
        device = x.device
        self.set_neuron_state(B)
        # 移动状态到正确设备
        self.asrnn_layer.mem = self.asrnn_layer.mem.to(device)
        self.asrnn_layer.spike = self.asrnn_layer.spike.to(device)
        self.asrnn_layer.b = self.asrnn_layer.b.to(device)

        readout_seq = []
        total_spikes = 0

        for t in range(T):
            x_t = x[:, t, :]  # [B, 32]
            probe_m = probe_mask[:, t]  # [B]

            # 计算 d_input（含 perturbation）
            d_input = self.asrnn_layer.dense(x_t.float())
            # probe_mask 广播到 [B, H]
            d_input = d_input + A_PROBE * probe_m.unsqueeze(-1)

            # 调用 mem_update_adp
            mem, spike, _, b = mem_update_adp(
                d_input,
                self.asrnn_layer.mem,
                self.asrnn_layer.spike,
                self.asrnn_layer.tau_adp,
                self.asrnn_layer.b,
                self.asrnn_layer.tau_m,
                device=device,
                isAdapt=1,
            )

            # 更新状态
            self.asrnn_layer.mem = mem
            self.asrnn_layer.spike = spike
            self.asrnn_layer.b = b

            # readout 从膜电位 mem 读
            readout = self.readout(mem)  # [B, 2]
            readout_seq.append(readout)

            total_spikes += spike.sum()

        readout_seq = torch.stack(readout_seq, dim=1)  # [B, T, 2]
        spike_rate = total_spikes / (B * T * self.asrnn_layer.output_dim)

        return readout_seq, spike_rate


__all__ = ["SPRiFTrajectoryNet", "ASRNNTrajectoryNet"]
