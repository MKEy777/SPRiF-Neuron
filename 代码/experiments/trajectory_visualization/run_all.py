#!/usr/bin/env python
"""
SPRiF 状态轨迹可视化实验 — 完整运行脚本
=============================================

端到端执行:
    1. 生成合成相位轨迹数据集
    2. 训练 SPRiF 模型 (readout from slow state)
    3. 训练 LIF 模型 (readout from membrane)
    4. 生成固定可视化样本
    5. 对 4 个样本记录完整内部状态轨迹
    6. 绘制 AAAI 5-panel 主图
    7. 绘制所有附录 panels

Usage:
    cd 代码/experiments/trajectory_visualization
    python run_all.py              # 完整运行
    python run_all.py --skip-train # 跳过训练 (使用已有 checkpoint)
    python run_all.py --skip-train --skip-record  # 仅绘图

Output:
    output/main_figure.png              — AAAI 主文 5-panel Figure
    output/appendix_a_input_structure.png
    output/appendix_b_input_raster.png
    output/appendix_c_hidden_raster.png
    output/appendix_h_probe_zoom_p*.png
    output/appendix_j_parameters.png
    output/appendix_multi_sample.png
    checkpoints/sprif_trajectory_best.pth
    checkpoints/lif_trajectory_best.pth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Path setup — add 代码/ so we can import Task_ECG as a package
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from Task_ECG.core_algorithm.utils import set_seed

from config import (
    T_CUE, T_DELAY, T_TOTAL, T_PROBES,
    HIDDEN_SIZE, OUT_DIR, MODEL_DIR, SEED,
    VIS_PHASES, VIS_OMEGA,
    SPRIF_KWARGS, LIF_TAU_M, LIF_THRESHOLD,
    TRAIN_SAMPLES, VAL_SAMPLES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
)
from generate_data import generate_visualization_samples
from models import SPRiFTrajectoryModel, LIFTrajectoryModel
from train import train_sprif, train_lif, create_dataloaders
from record_forward import (
    record_sprif_forward,
    record_lif_forward,
    record_all,
    select_representative_neuron,
    select_lif_neuron,
)
from plot_main_figure import plot_main_figure
from plot_appendix import plot_all_appendix


def main():
    parser = argparse.ArgumentParser(
        description="SPRiF State Trajectory Visualization Experiment",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing checkpoints")
    parser.add_argument("--skip-record", action="store_true",
                        help="Skip recording, use existing records")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    # ---- Setup ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 65)
    print("  SPRiF State Trajectory Visualization Experiment")
    print("  Cue → Delay with Controlled Perturbation Probes")
    print("=" * 65)
    print(f"  Device:     {device}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Model dir:  {MODEL_DIR}")
    print(f"  Seed:       {args.seed}")

    set_seed(args.seed)

    sph_path = os.path.join(MODEL_DIR, "sprif_trajectory_best.pth")
    lif_path = os.path.join(MODEL_DIR, "lif_trajectory_best.pth")

    # ====================================================================
    # Step 1: Train models
    # ====================================================================
    if not args.skip_train:
        print("\n" + "=" * 65)
        print("  Step 1: Training Models")
        print("=" * 65)

        # ---- Data ----
        print("\n[1a] Generating synthetic dataset...")
        t0 = time.time()
        train_loader, val_loader = create_dataloaders(
            n_train=TRAIN_SAMPLES, n_val=VAL_SAMPLES, batch_size=BATCH_SIZE,
            seed=args.seed,
        )
        print(f"  Data generation took {time.time() - t0:.1f}s")

        # ---- SPRiF ----
        print("\n[1b] Training SPRiF model...")
        t0 = time.time()
        sprif_model = train_sprif(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=LEARNING_RATE, save_path=sph_path,
        )
        print(f"  SPRiF training took {time.time() - t0:.1f}s")

        # ---- LIF ----
        print("\n[1c] Training LIF model...")
        t0 = time.time()
        lif_model = train_lif(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=LEARNING_RATE, save_path=lif_path,
        )
        print(f"  LIF training took {time.time() - t0:.1f}s")

    else:
        print("\n[1] Skipping training. Loading checkpoints...")

        sprif_model = SPRiFTrajectoryModel(
            input_size=32, hidden_size=HIDDEN_SIZE, output_size=2,
            neuron_kwargs=SPRIF_KWARGS,
        ).to(device)
        sprif_model.load_state_dict(
            torch.load(sph_path, map_location=device, weights_only=True),
        )

        lif_model = LIFTrajectoryModel(
            input_size=32, hidden_size=HIDDEN_SIZE, output_size=2,
            threshold=LIF_THRESHOLD, tau_m=LIF_TAU_M,
        ).to(device)
        lif_model.load_state_dict(
            torch.load(lif_path, map_location=device, weights_only=True),
        )

        print(f"  SPRiF loaded from: {sph_path}")
        print(f"  LIF loaded from:   {lif_path}")

    # ====================================================================
    # Step 2: Generate visualization samples
    # ====================================================================
    print("\n" + "=" * 65)
    print("  Step 2: Generating Visualization Samples")
    print("=" * 65)

    vis_samples = generate_visualization_samples(
        phases=VIS_PHASES,
        omega=VIS_OMEGA,
        seed=args.seed,
    )
    print(f"  Generated {len(vis_samples)} samples "
          f"(φ ∈ {{0°, 90°, 180°, 270°}}, ω={VIS_OMEGA*1000/(2*np.pi):.0f}Hz)")

    # ====================================================================
    # Step 3: Record state trajectories
    # ====================================================================
    if not args.skip_record:
        print("\n" + "=" * 65)
        print("  Step 3: Recording State Trajectories")
        print("=" * 65)

        t0 = time.time()
        records = record_all(sprif_model, lif_model, vis_samples, device)
        print(f"\n  Recording took {time.time() - t0:.1f}s")

        # Save records for reproducibility
        rec_dir = os.path.join(OUT_DIR, "records")
        os.makedirs(rec_dir, exist_ok=True)

        for i, (sr, lr) in enumerate(zip(records["sprif"], records["lif"])):
            phi_deg = int(VIS_PHASES[i] * 180 / np.pi)

            # SPRiF
            sr_save = {k: v for k, v in sr.items()
                       if k not in ("spectral_params",)}
            np.savez_compressed(
                os.path.join(rec_dir, f"sprif_record_phi{phi_deg:03d}.npz"),
                **sr_save,
            )

            # LIF
            lr_save = {k: v for k, v in lr.items()}
            np.savez_compressed(
                os.path.join(rec_dir, f"lif_record_phi{phi_deg:03d}.npz"),
                **lr_save,
            )

        print(f"  Saved records to: {rec_dir}/")
    else:
        print("\n[3] Skipping recording. Loading existing records...")

        rec_dir = os.path.join(OUT_DIR, "records")
        sprif_records = []
        lif_records = []

        for i, phi in enumerate(VIS_PHASES):
            phi_deg = int(phi * 180 / np.pi)
            sr = dict(np.load(
                os.path.join(rec_dir, f"sprif_record_phi{phi_deg:03d}.npz"),
                allow_pickle=True,
            ))
            lr = dict(np.load(
                os.path.join(rec_dir, f"lif_record_phi{phi_deg:03d}.npz"),
                allow_pickle=True,
            ))
            sprif_records.append(sr)
            lif_records.append(lr)

        records = {"sprif": sprif_records, "lif": lif_records}
        print(f"  Loaded {len(sprif_records)} SPRiF + {len(lif_records)} LIF records")

    # ====================================================================
    # Step 4: Plot main figure
    # ====================================================================
    print("\n" + "=" * 65)
    print("  Step 4: Generating AAAI Main Figure (5 panels)")
    print("=" * 65)

    # Use φ=0 as primary sample
    primary_idx = 0
    sprif_rec = records["sprif"][primary_idx]
    lif_rec = records["lif"][primary_idx]

    # Select neurons
    neuron_idx = select_representative_neuron(sprif_rec, T_PROBES)
    lif_neuron_idx = select_lif_neuron(lif_rec, T_PROBES)

    plot_main_figure(
        sprif_record=sprif_rec,
        lif_record=lif_rec,
        save_path=os.path.join(OUT_DIR, "main_figure.png"),
        phi=VIS_PHASES[primary_idx],
        neuron_idx=neuron_idx,
        lif_neuron_idx=lif_neuron_idx,
    )

    # ====================================================================
    # Step 5: Plot appendix panels
    # ====================================================================
    print("\n" + "=" * 65)
    print("  Step 5: Generating Appendix Panels")
    print("=" * 65)

    plot_all_appendix(
        vis_samples=vis_samples,
        sprif_records=records["sprif"],
        lif_records=records["lif"],
        out_dir=OUT_DIR,
        primary_idx=primary_idx,
    )

    # ====================================================================
    # Done
    # ====================================================================
    print("\n" + "=" * 65)
    print("  Experiment Complete!")
    print("=" * 65)
    print(f"\n  Output directory: {OUT_DIR}/")
    print(f"  ├── main_figure.png                ← AAAI 5-panel Figure")
    print(f"  ├── appendix_a_input_structure.png")
    print(f"  ├── appendix_b_input_raster.png")
    print(f"  ├── appendix_c_hidden_raster.png")
    print(f"  ├── appendix_h_probe_zoom_p*.png")
    print(f"  ├── appendix_j_parameters.png")
    print(f"  ├── appendix_multi_sample.png")
    print(f"  └── records/")

    # Print summary statistics
    print(f"\n  Summary:")
    for i, phi in enumerate(VIS_PHASES):
        sr = records["sprif"][i]
        lr = records["lif"][i]
        n_spikes_s = int(sr["spikes"].sum())
        n_spikes_l = int(lr["spikes"].sum())
        target = sr["target"]
        mse_s = np.mean((sr["readouts"][T_CUE:] - target[T_CUE:]) ** 2)
        mse_l = np.mean((lr["readouts"][T_CUE:] - target[T_CUE:]) ** 2)
        print(f"    φ={phi*180/np.pi:3.0f}° | "
              f"SPRiF: spikes={n_spikes_s:4d}, MSE={mse_s:.6f} | "
              f"LIF: spikes={n_spikes_l:4d}, MSE={mse_l:.6f}")


if __name__ == "__main__":
    main()
