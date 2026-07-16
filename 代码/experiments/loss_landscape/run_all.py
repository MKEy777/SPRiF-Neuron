"""
S-MNIST 损失景观与梯度可视化实验 — 编排器。

一键运行：训练 SPRiF + LIF (TBPTT) → 计算损失景观 → 记录 BPTT 梯度 → 绘图。

用法:
  python run_all.py                       # 全流程全量
  python run_all.py --skip-train          # 跳过训练，用已有 checkpoint
  python run_all.py --epochs 5 --subset 2000  # 小规模验证
"""
import os
import sys
import subprocess

import config as cfg


def run_command(cmd, cwd=None):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    args = cfg.get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 透传给子脚本的公共参数
    common = ["--neuron", args.neuron, "--seed", str(args.seed)]
    if args.subset > 0:
        common += ["--subset", str(args.subset)]

    # 步骤 1: TBPTT 训练
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 1: Training SPRiF & LIF (TBPTT)")
        print("=" * 60)
        run_command(
            [sys.executable, "train.py", *common,
             "--epochs", str(args.epochs),
             "--lr", str(args.lr),
             "--batch-size", str(args.batch_size)],
            cwd=script_dir,
        )
    else:
        print("\nSkipping training (--skip-train)")

    # 步骤 2: 损失景观
    print("\n" + "=" * 60)
    print("STEP 2: Computing loss landscape (filter-normalized)")
    print("=" * 60)
    run_command(
        [sys.executable, "landscape.py", *common,
         "--grid-resolution", str(args.grid_resolution),
         "--grid-range", str(args.grid_range),
         "--eval-batches", str(args.eval_batches)],
        cwd=script_dir,
    )

    # 步骤 3: BPTT 梯度记录
    print("\n" + "=" * 60)
    print("STEP 3: Recording BPTT gradient propagation")
    print("=" * 60)
    run_command([sys.executable, "grad_record.py", *common], cwd=script_dir)

    # 步骤 4: 绘图
    print("\n" + "=" * 60)
    print("STEP 4: Plotting figures")
    print("=" * 60)
    run_command([sys.executable, "plot_figures.py"], cwd=script_dir)

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"Checkpoints/npz -> {cfg.CHECKPOINT_DIR}")
    print(f"Figures         -> {cfg.FIGURE_DIR}")


if __name__ == "__main__":
    main()