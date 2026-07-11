"""
相位轨迹可视化实验 — 记录前向传播。

加载训练好的模型，记录逐步状态，用于可视化。
"""
import os
import glob
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "Task_pSMNIST"))
sys.path.insert(0, os.path.join(ROOT, "Task_GSC"))

from config import (
    FIGURE_DIR, CHECKPOINT_DIR,
    VIZ_PHIS, VIZ_OMEGA, PROBE_TIMES,
    RUN_TAG, tagged,
)
from generate_data import generate_sample
from models import SPRiFTrajectoryNet, ASRNNTrajectoryNet
from SRNN_layers.spike_neuron import mem_update_adp


def _find_best_checkpoint(model_name: str) -> str:
    """找到最佳 checkpoint（MSE 最小）。"""
    _tag_suffix = f"_{RUN_TAG}" if RUN_TAG else ""
    pattern = os.path.join(CHECKPOINT_DIR, f"TrajectoryViz_{model_name}{_tag_suffix}_mse*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")

    best = None
    best_mse = float("inf")
    for f in files:
        try:
            mse_str = os.path.basename(f).split("_mse")[1].replace(".pth", "")
            mse = float(mse_str)
            if mse < best_mse:
                best_mse = mse
                best = f
        except (ValueError, IndexError):
            continue

    if best is None:
        raise FileNotFoundError(f"No valid checkpoint found: {pattern}")
    return best


def record_sprif(
    model: SPRiFTrajectoryNet,
    input_spikes: np.ndarray,
    probe_mask: np.ndarray,
    target: np.ndarray,
    device: torch.device,
) -> dict:
    """
    记录 SPRiF 前向传播的逐步状态。

    Returns:
        dict with keys: x_t, u_pre, u_post, membrane, spike, readout, input_spikes, probe_mask, target
    """
    model.eval()
    T = input_spikes.shape[0]

    # 转换为 tensor
    x_tensor = torch.from_numpy(input_spikes).unsqueeze(0).to(device)  # [1, T, 32]
    probe_tensor = torch.from_numpy(probe_mask).unsqueeze(0).to(device)  # [1, T]

    # 初始化状态
    state = model.layer.init_state(1, device=device, dtype=x_tensor.dtype)
    runtime = model.layer._precompute_runtime_params()

    # 记录列表
    x_list = []
    u_pre_list = []
    u_post_list = []
    membrane_list = []
    spike_list = []
    readout_list = []

    with torch.no_grad():
        for t in range(T):
            x_t = x_tensor[:, t, :]  # [1, 32]
            probe_m = probe_tensor[:, t]  # [1]

            # 使用 forward_step_record 记录中间状态
            spike, membrane, state, u_tilde, u_next = model.forward_step_record(
                x_t, state, runtime, probe_m
            )

            x_list.append(state["x"].squeeze(0).cpu().numpy())  # [H, 3]
            u_pre_list.append(u_tilde.squeeze(0).cpu().numpy())  # [H, 2]
            u_post_list.append(u_next.squeeze(0).cpu().numpy())  # [H, 2]
            membrane_list.append(membrane.squeeze(0).cpu().numpy())  # [H]
            spike_list.append(spike.squeeze(0).cpu().numpy())  # [H]

            # readout
            x_state = state["x"]  # [1, H, 3]
            readout = model.readout(x_state.reshape(1, -1))  # [1, 2]
            readout_list.append(readout.squeeze(0).cpu().numpy())  # [2]

    return {
        "x_t": np.stack(x_list, axis=0),  # [T, H, 3]
        "u_pre": np.stack(u_pre_list, axis=0),  # [T, H, 2]
        "u_post": np.stack(u_post_list, axis=0),  # [T, H, 2]
        "membrane": np.stack(membrane_list, axis=0),  # [T, H]
        "spike": np.stack(spike_list, axis=0),  # [T, H]
        "readout": np.stack(readout_list, axis=0),  # [T, 2]
        "input_spikes": input_spikes,  # [T, 32]
        "probe_mask": probe_mask,  # [T]
        "target": target,  # [T, 2]
    }


def record_asrnn(
    model: ASRNNTrajectoryNet,
    input_spikes: np.ndarray,
    probe_mask: np.ndarray,
    target: np.ndarray,
    device: torch.device,
) -> dict:
    """
    记录 ASRNN 前向传播的逐步状态。

    Returns:
        dict with keys: mem, spike, readout, input_spikes, probe_mask, target
    """
    model.eval()
    T = input_spikes.shape[0]

    # 转换为 tensor
    x_tensor = torch.from_numpy(input_spikes).unsqueeze(0).to(device)  # [1, T, 32]
    probe_tensor = torch.from_numpy(probe_mask).unsqueeze(0).to(device)  # [1, T]

    # 初始化状态
    model.set_neuron_state(1)
    model.asrnn_layer.mem = model.asrnn_layer.mem.to(device)
    model.asrnn_layer.spike = model.asrnn_layer.spike.to(device)
    model.asrnn_layer.b = model.asrnn_layer.b.to(device)

    # 记录列表
    mem_list = []
    spike_list = []
    readout_list = []

    with torch.no_grad():
        for t in range(T):
            x_t = x_tensor[:, t, :]  # [1, 32]
            probe_m = probe_tensor[:, t]  # [1]

            # 计算 d_input（含 perturbation）
            d_input = model.asrnn_layer.dense(x_t.float())
            d_input = d_input + model.a_probe * probe_m.unsqueeze(-1)  # 使用模型训练时的 a_probe

            # 调用 mem_update_adp
            mem, spike, _, b = mem_update_adp(
                d_input,
                model.asrnn_layer.mem,
                model.asrnn_layer.spike,
                model.asrnn_layer.tau_adp,
                model.asrnn_layer.b,
                model.asrnn_layer.tau_m,
                device=device,
                isAdapt=1,
                b_j0=0.3,
            )

            # 更新状态
            model.asrnn_layer.mem = mem
            model.asrnn_layer.spike = spike
            model.asrnn_layer.b = b

            # 记录
            mem_list.append(mem.squeeze(0).cpu().numpy())  # [H]
            spike_list.append(spike.squeeze(0).cpu().numpy())  # [H]

            # readout
            readout = model.readout(mem)  # [1, 2]
            readout_list.append(readout.squeeze(0).cpu().numpy())  # [2]

    return {
        "mem": np.stack(mem_list, axis=0),  # [T, H]
        "spike": np.stack(spike_list, axis=0),  # [T, H]
        "readout": np.stack(readout_list, axis=0),  # [T, 2]
        "input_spikes": input_spikes,  # [T, 32]
        "probe_mask": probe_mask,  # [T]
        "target": target,  # [T, 2]
    }


def get_spectral_params(model: SPRiFTrajectoryNet) -> dict:
    """提取 SPRiF 频谱参数。"""
    model.eval()
    params = model.layer.get_spectral_parameters()
    return {
        "alpha": params["alpha"].detach().cpu().numpy(),
        "rho": params["rho"].detach().cpu().numpy(),
        "omega": params["omega"].detach().cpu().numpy(),
        "eta": params["eta"].detach().cpu().numpy(),
        "lambda_reset": params["lambda_reset"].detach().cpu().numpy(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Figure output: {FIGURE_DIR}")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 加载 SPRiF
    print("\nLoading SPRiF model...")
    sprif_checkpoint = _find_best_checkpoint("SPRiF")
    print(f"  Checkpoint: {os.path.basename(sprif_checkpoint)}")
    sprif_model = SPRiFTrajectoryNet().to(device)
    sprif_model.load_state_dict(torch.load(sprif_checkpoint, map_location=device, weights_only=True))
    sprif_model.eval()

    # 加载 ASRNN
    print("\nLoading ASRNN model...")
    asrnn_checkpoint = _find_best_checkpoint("ASRNN")
    print(f"  Checkpoint: {os.path.basename(asrnn_checkpoint)}")
    asrnn_model = ASRNNTrajectoryNet().to(device)
    asrnn_model.load_state_dict(torch.load(asrnn_checkpoint, map_location=device, weights_only=True))
    asrnn_model.eval()

    # 记录 4 个样本 (φ ∈ {0, π/2, π, 3π/2})
    for phi_idx, phi in enumerate(VIZ_PHIS):
        print(f"\n{'='*60}")
        print(f"Recording sample {phi_idx+1}/4: φ = {phi:.4f}")
        print(f"{'='*60}")

        # 生成样本
        input_spikes, probe_mask, target = generate_sample(
            phi=phi,
            omega=VIZ_OMEGA,
            probe_times=PROBE_TIMES,
            jitter=False,
            rng=np.random.default_rng(42 + phi_idx),
        )

        # 记录 SPRiF
        print("  Recording SPRiF...")
        sprif_rec = record_sprif(sprif_model, input_spikes, probe_mask, target, device)
        sprif_spectral = get_spectral_params(sprif_model)

        # 记录 ASRNN
        print("  Recording ASRNN...")
        asrnn_rec = record_asrnn(asrnn_model, input_spikes, probe_mask, target, device)

        # 保存
        save_path = os.path.join(FIGURE_DIR, tagged(f"trajectory_data_phi{phi_idx}.npz"))
        np.savez_compressed(
            save_path,
            phi=phi,
            omega=VIZ_OMEGA,
            sprif_x_t=sprif_rec["x_t"],
            sprif_u_pre=sprif_rec["u_pre"],
            sprif_u_post=sprif_rec["u_post"],
            sprif_membrane=sprif_rec["membrane"],
            sprif_spike=sprif_rec["spike"],
            sprif_readout=sprif_rec["readout"],
            asrnn_mem=asrnn_rec["mem"],
            asrnn_spike=asrnn_rec["spike"],
            asrnn_readout=asrnn_rec["readout"],
            input_spikes=sprif_rec["input_spikes"],
            probe_mask=sprif_rec["probe_mask"],
            target=sprif_rec["target"],
            spectral_alpha=sprif_spectral["alpha"],
            spectral_rho=sprif_spectral["rho"],
            spectral_omega=sprif_spectral["omega"],
            spectral_eta=sprif_spectral["eta"],
            spectral_lambda=sprif_spectral["lambda_reset"],
        )
        print(f"  Saved: {save_path}")

        # 统计信息
        sprif_total_spikes = sprif_rec["spike"].sum()
        asrnn_total_spikes = asrnn_rec["spike"].sum()
        print(f"  SPRiF total spikes: {sprif_total_spikes}")
        print(f"  ASRNN total spikes: {asrnn_total_spikes}")

        # 检查 probe 时刻的 spike
        probe_steps = np.where(probe_mask > 0.5)[0]
        if len(probe_steps) > 0:
            sprif_probe_spikes = sprif_rec["spike"][probe_steps].sum()
            asrnn_probe_spikes = asrnn_rec["spike"][probe_steps].sum()
            print(f"  SPRiF spikes during probe: {sprif_probe_spikes}")
            print(f"  ASRNN spikes during probe: {asrnn_probe_spikes}")

    print("\nRecording complete.")


if __name__ == "__main__":
    main()
