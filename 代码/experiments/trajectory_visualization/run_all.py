"""
相位轨迹可视化实验 — 编排器。

一键运行：训练 SPRiF + ASRNN → 记录前向传播 → 绘制主图。
"""
import os
import sys
import subprocess


def run_command(cmd: list, cwd: str = None):
    """运行命令并打印输出。"""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    # 解析 --skip-train
    skip_train = "--skip-train" in sys.argv

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 步骤 1: 训练
    if not skip_train:
        print("\n" + "="*60)
        print("STEP 1: Training SPRiF and ASRNN")
        print("="*60)
        run_command([
            sys.executable, "train.py",
            "--model", "both",
            "--epochs", "100",
        ], cwd=script_dir)
    else:
        print("\nSkipping training (--skip-train)")

    # 步骤 2: 记录前向传播
    print("\n" + "="*60)
    print("STEP 2: Recording forward pass")
    print("="*60)
    run_command([sys.executable, "record_forward.py"], cwd=script_dir)

    # 步骤 3: 绘制主图
    print("\n" + "="*60)
    print("STEP 3: Plotting main figure")
    print("="*60)
    run_command([sys.executable, "plot_main_figure.py"], cwd=script_dir)

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    print(f"Results saved to: {os.path.join(script_dir, 'checkpoints')}")
    print(f"Figures saved to: {os.path.join(os.path.dirname(script_dir), 'experiment-design-20260606', 'results', 'figures', 'trajectory_visualization')}")


if __name__ == "__main__":
    main()
