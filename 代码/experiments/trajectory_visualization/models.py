"""
相位轨迹可视化实验 — 模型定义。

SPRiF 和 ASRNN 对照网络，均支持 perturbation 注入。
"""
import os
import sys
import math
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

    def __init__(self, input_size: int = 32, hidden_size: int = HIDDEN_SIZE,
                 a_probe: float = A_PROBE):
        super().__init__()
        self.a_probe = a_probe
        self.layer = SPRiFNeuronLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            recurrent=False,
     threshold=0.3,
            # 任务需在 800ms delay 无输入下维持旋转：
            # tau_rho 需足够大使 rho≈1（rho=exp(-1/tau_rho)），否则振荡分量指数衰减为 0
            #   tau_rho=3000 -> rho=0.99967 -> rho^800≈0.77（维持振幅）
            tau_rho_range=(300.0, 3000.0),
            # omega 需覆盖任务频率 {2π/50, 2π/100, 2π/200}={0.126, 0.063, 0.031}
            omega_range=(2.0 * math.pi / 300.0, 2.0 * math.pi / 40.0),
        )
        # readout 从慢状态 x_t [B,H,3] flatten 读
        self.readout = nn.Linear(hidden_size * 3, 2)

    def forward(self, x: torch.Tensor, probe_mask: torch.Tensor,
                return_diag: bool = False) -> torch.Tensor:
        """
        前向传播，支持 perturbation 注入。

       Args:
            x: [B, T, 32] 输入脉冲
            probe_mask: [B, T] probe 注入掩码
            return_diag: 若为 True，额外返回诊断 dict

        Returns:
            readout_seq: [B, T, 2] 输出轨迹
            spike_rate: scalar 平均发放率（用于正则）
        """
        B, T, _ = x.shape
        device = x.device
        state = self.layer.init_state(B, device=device, dtype=x.dtype)
        runtime = self.layer._precompute_runtime_params()

        # 一次性算完整段 input_linear（避免每步启动开销）
        input_current_all = self.layer.input_linear(x)  # [B, T, H]
        input_current_all = input_current_all + self.a_probe * probe_mask.unsqueeze(-1)

        x_state_seq = []
        spikes_seq = []
        membrane_seq = []

        for t in range(T):
            x_t = x[:, t, :]
            spike, membrane, state = self.layer.forward_step(
                x_t, state, runtime, input_current=input_current_all[:, t, :]
            )
            x_state_seq.append(state["x"])
            spikes_seq.append(spike)
            if return_diag:
                membrane_seq.append(membrane)

        x_state_seq = torch.stack(x_state_seq, dim=1)  # [B, T, H, 3]
        # 一次性做 readout
        readout_seq = self.readout(x_state_seq.reshape(B, T, -1))  # [B, T, 2]

        spikes_all = torch.stack(spikes_seq, dim=1)  # [B, T, H]
        spike_rate = spikes_all.mean()

        if return_diag:
            membrane_all = torch.stack(membrane_seq, dim=1)  # [B, T, H]
            # 慢状态振荡分量振幅：验证 delay 末期是否衰减为 0
            osc = x_state_seq[..., 1:3]  # [B,T,H,2] 振荡两分量
            osc_amp = osc.pow(2).sum(-1).sqrt()  # [B,T,H] 振幅
            T_all = x_state_seq.shape[1]
            sp = self.layer.get_spectral_parameters()
            diag = {
                "threshold": float(self.layer.threshold),
                "a_probe": float(self.a_probe),
                "membrane_max": float(membrane_all.max()),
                "membrane_mean": float(membrane_all.mean()),
                "membrane_p99": float(membrane_all.flatten().quantile(0.99)),
                "input_current_max": float(input_current_all.max()),
                "input_current_mean": float(input_current_all.mean()),
                "osc_amp_early": float(osc_amp[:, 100:120].mean()),
                "osc_amp_late": float(osc_amp[:, T_all - 20:].mean()),
                "rho_min": float(sp["rho"].min()),
                "rho_max": float(sp["rho"].max()),
                "omega_min": float(sp["omega"].min()),
                "omega_max": float(sp["omega"].max()),
                "probe_mask_sum": float(probe_mask.sum()),
                "n_spikes": float(spikes_all.sum()),
            }
            return readout_seq, spike_rate, diag

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
        input_current = input_current + self.a_probe * probe_mask_t.unsqueeze(-1)

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

    def __init__(self, input_size: int = 32, hidden_size: int = HIDDEN_SIZE,
                 a_probe: float = A_PROBE):
        super().__init__()
        self.a_probe = a_probe
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

    def forward(self, x: torch.Tensor, probe_mask: torch.Tensor,
                return_diag: bool = False) -> torch.Tensor:
        """
        前向传播，支持 perturbation 注入。

        Args:
            x: [B, T, 32] 输入脉冲
            probe_mask: [B, T] probe 注入掩码
            return_diag: 若为 True，额外返回诊断 dict

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

        # 一次性算完整段 dense（避免每步启动开销）
        d_input_all = self.asrnn_layer.dense(x.float())  # [B, T, H]
        d_input_all = d_input_all + self.a_probe * probe_mask.unsqueeze(-1)

        mem_seq = []
        spikes_seq = []

        for t in range(T):
            d_input = d_input_all[:, t, :]

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
                b_j0=0.3,
            )

            # 更新状态
            self.asrnn_layer.mem = mem
            self.asrnn_layer.spike = spike
            self.asrnn_layer.b = b

            mem_seq.append(mem)
            spikes_seq.append(spike)

        mem_all = torch.stack(mem_seq, dim=1)          # [B, T, H]
        spikes_all = torch.stack(spikes_seq, dim=1)    # [B, T, H]

        readout_seq = self.readout(mem_all)             # [B, T, 2]
        spike_rate = spikes_all.mean()

        if return_diag:
            diag = {
                "a_probe": float(self.a_probe),
                "mem_max": float(mem_all.max()),
                "mem_mean": float(mem_all.mean()),
                "mem_p99": float(mem_all.flatten().quantile(0.99)),
                "d_input_max": float(d_input_all.max()),
                "d_input_mean": float(d_input_all.mean()),
                "probe_mask_sum": float(probe_mask.sum()),
                "n_spikes": float(spikes_all.sum()),
            }
            return readout_seq, spike_rate, diag

        return readout_seq, spike_rate


__all__ = ["SPRiFTrajectoryNet", "ASRNNTrajectoryNet"]
