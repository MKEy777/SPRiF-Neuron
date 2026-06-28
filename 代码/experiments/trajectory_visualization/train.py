"""
SPRiF & LIF 相位轨迹训练脚本
==================================

训练 SPRiF 和 LIF 两个模型在相同的合成相位轨迹任务上。
MSE loss 只在 delay 阶段计算，可选 firing-rate regularization。

Usage:
    # 单独训练
    python train.py --model sprif
    python train.py --model lif

    # 在 run_all.py 中调用 train_both()
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Path setup — add 代码/ so we can import Task_ECG as a package
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from Task_ECG.core_algorithm.utils import set_seed

from config import (
    T_CUE, T_DELAY, T_TOTAL,
    N_INPUT_CH, HIDDEN_SIZE, OUTPUT_SIZE,
    SPRIF_KWARGS, LIF_TAU_M, LIF_THRESHOLD,
    TRAIN_SAMPLES, VAL_SAMPLES, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, BETA_FR, GRAD_CLIP, SEED,
    A_PROBE,
    MODEL_DIR,
)
from generate_data import create_dataloaders
from models import SPRiFTrajectoryModel, LIFTrajectoryModel


# ============================================================================
# Loss
# ============================================================================

def trajectory_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    spikes: Optional[torch.Tensor] = None,
    beta_fr: float = BETA_FR,
) -> Tuple[torch.Tensor, dict]:
    """MSE loss on delay period + optional firing-rate regularization.

    Args:
        outputs: [batch, T, 2] predicted trajectory
        targets: [batch, T, 2] target trajectory
        spikes:  [batch, T, H] hidden spikes (for FR reg)
        beta_fr: firing-rate regularization weight

    Returns:
        loss: scalar loss
        metrics: dict with mse, fr_reg, total
    """
    # Only compute MSE during delay period.
    # Design: L_traj = (1 / T_delay) Σ_t ||ŷ_t - y_t||²₂
    # F.mse_loss divides by B * T_delay * 2 (the 2 comes from the output dim).
    # We need division by B * T_delay only, so multiply by 2.0 to correct.
    delay_outputs = outputs[:, T_CUE:, :]
    delay_targets = targets[:, T_CUE:, :]

    mse = nn.functional.mse_loss(delay_outputs, delay_targets)

    loss = mse * 2.0

    metrics = {"mse": mse.item()}

    if spikes is not None and beta_fr > 0:
        # Penalize both too-low and too-high firing rates
        fr = spikes.mean(dim=(0, 1))  # [H] average over batch and time
        fr_target = 0.05  # target mean firing rate ~5%
        fr_reg = beta_fr * ((fr - fr_target) ** 2).mean()
        loss = loss + fr_reg
        metrics["fr_reg"] = fr_reg.item()
        metrics["fr_mean"] = fr.mean().item()

    metrics["loss"] = loss.item()

    return loss, metrics


# ============================================================================
# Training loop (SPRiF)
# ============================================================================

def train_sprif(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    save_path: Optional[str] = None,
) -> SPRiFTrajectoryModel:
    """Train SPRiF trajectory model."""
    print("\n" + "=" * 60)
    print("Training SPRiF Trajectory Model")
    print("=" * 60)

    model = SPRiFTrajectoryModel(
        input_size=N_INPUT_CH,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        neuron_kwargs=SPRIF_KWARGS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_metrics = {"mse": 0.0, "loss": 0.0, "fr_mean": 0.0, "n_batches": 0}

        for inputs, targets, probe_masks in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            probe_masks = probe_masks.to(device)

            optimizer.zero_grad()

            # Forward: need spikes for FR regularization
            # We do a custom forward to also get spikes
            batch_size, T, _ = inputs.shape
            state = model.sprif_layer.init_state(
                batch_size, device=device, dtype=inputs.dtype,
            )
            runtime = model.sprif_layer._precompute_runtime_params()

            outputs_list = []
            spikes_list = []

            for t in range(T):
                input_current = model.sprif_layer.input_linear(inputs[:, t, :])
                input_current = input_current + A_PROBE * probe_masks[:, t].unsqueeze(-1)

                spike, membrane, state = model.sprif_layer.forward_step(
                    inputs[:, t, :], state, runtime, input_current=input_current,
                )

                x_t = state["x"]
                out = model.readout(x_t.reshape(batch_size, -1))
                outputs_list.append(out)
                spikes_list.append(spike)

            outputs = torch.stack(outputs_list, dim=1)  # [B, T, 2]
            spikes = torch.stack(spikes_list, dim=1)    # [B, T, H]

            loss, metrics = trajectory_loss(outputs, targets, spikes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_metrics["mse"] += metrics["mse"]
            train_metrics["loss"] += metrics["loss"]
            train_metrics["fr_mean"] += metrics.get("fr_mean", 0.0)
            train_metrics["n_batches"] += 1

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_mse = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets, probe_masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                probe_masks = probe_masks.to(device)

                outputs = model(inputs, probe_masks)
                delay_outputs = outputs[:, T_CUE:, :]
                delay_targets = targets[:, T_CUE:, :]
                val_mse += nn.functional.mse_loss(delay_outputs, delay_targets).item()
                val_batches += 1

        val_mse /= val_batches
        n_b = train_metrics["n_batches"]

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {train_metrics['mse']/n_b:.6f} | "
                f"Val MSE: {val_mse:.6f} | "
                f"FR: {train_metrics['fr_mean']/n_b:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            if save_path:
                torch.save(model.state_dict(), save_path)

    print(f"  Best Val MSE: {best_val_loss:.6f}")

    # Load best model
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model


# ============================================================================
# Training loop (LIF)
# ============================================================================

def train_lif(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    save_path: Optional[str] = None,
) -> LIFTrajectoryModel:
    """Train LIF trajectory model."""
    print("\n" + "=" * 60)
    print("Training LIF Trajectory Model")
    print("=" * 60)

    model = LIFTrajectoryModel(
        input_size=N_INPUT_CH,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        threshold=LIF_THRESHOLD,
        tau_m=LIF_TAU_M,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_metrics = {"mse": 0.0, "loss": 0.0, "fr_mean": 0.0, "n_batches": 0}

        for inputs, targets, probe_masks in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            probe_masks = probe_masks.to(device)

            optimizer.zero_grad()

            batch_size, T, _ = inputs.shape
            state = model.lif_layer.init_state(
                batch_size, device=device, dtype=inputs.dtype,
            )

            outputs_list = []
            spikes_list = []

            for t in range(T):
                input_current = model.lif_layer.input_linear(inputs[:, t, :])
                input_current = input_current + A_PROBE * probe_masks[:, t].unsqueeze(-1)

                spike, v_pre, state = model.lif_layer.forward_step(
                    inputs[:, t, :], state, input_current=input_current,
                )

                out = model.readout(v_pre)
                outputs_list.append(out)
                spikes_list.append(spike)

            outputs = torch.stack(outputs_list, dim=1)
            spikes = torch.stack(spikes_list, dim=1)

            loss, metrics = trajectory_loss(outputs, targets, spikes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_metrics["mse"] += metrics["mse"]
            train_metrics["loss"] += metrics["loss"]
            train_metrics["fr_mean"] += metrics.get("fr_mean", 0.0)
            train_metrics["n_batches"] += 1

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_mse = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets, probe_masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                probe_masks = probe_masks.to(device)

                outputs = model(inputs, probe_masks)
                delay_outputs = outputs[:, T_CUE:, :]
                delay_targets = targets[:, T_CUE:, :]
                val_mse += nn.functional.mse_loss(delay_outputs, delay_targets).item()
                val_batches += 1

        val_mse /= val_batches
        n_b = train_metrics["n_batches"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {train_metrics['mse']/n_b:.6f} | "
                f"Val MSE: {val_mse:.6f} | "
                f"FR: {train_metrics['fr_mean']/n_b:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            if save_path:
                torch.save(model.state_dict(), save_path)

    print(f"  Best Val MSE: {best_val_loss:.6f}")

    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model


# ============================================================================
# Combined training entry point
# ============================================================================

def train_both(
    device: Optional[torch.device] = None,
    n_train: int = TRAIN_SAMPLES,
    n_val: int = VAL_SAMPLES,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
) -> Tuple[SPRiFTrajectoryModel, LIFTrajectoryModel]:
    """Train both SPRiF and LIF models on the trajectory task.

    Returns:
        sprif_model, lif_model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(SEED)

    # ---- Data ----
    train_loader, val_loader = create_dataloaders(
        n_train=n_train, n_val=n_val, batch_size=batch_size, seed=SEED,
    )

    # ---- SPRiF ----
    sprif_path = os.path.join(MODEL_DIR, "sprif_trajectory_best.pth")
    sprif_model = train_sprif(
        train_loader, val_loader, device,
        epochs=epochs, lr=lr, save_path=sprif_path,
    )

    # ---- LIF ----
    lif_path = os.path.join(MODEL_DIR, "lif_trajectory_best.pth")
    lif_model = train_lif(
        train_loader, val_loader, device,
        epochs=epochs, lr=lr, save_path=lif_path,
    )

    return sprif_model, lif_model


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SPRiF/LIF trajectory models")
    parser.add_argument("--model", type=str, default="both",
                        choices=["sprif", "lif", "both"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-train", type=int, default=TRAIN_SAMPLES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    train_loader, val_loader = create_dataloaders(
        n_train=args.n_train, n_val=VAL_SAMPLES, batch_size=args.batch_size,
    )

    if args.model in ("sprif", "both"):
        sprif_model = train_sprif(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_path=os.path.join(MODEL_DIR, "sprif_trajectory_best.pth"),
        )

    if args.model in ("lif", "both"):
        lif_model = train_lif(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_path=os.path.join(MODEL_DIR, "lif_trajectory_best.pth"),
        )
