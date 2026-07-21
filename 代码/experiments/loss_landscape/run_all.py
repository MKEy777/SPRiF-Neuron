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

    common = ["--neuron", args.neuron, "--seed", str(args.seed)]
    if args.subset > 0:
        common += ["--subset", str(args.subset)]

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

    print("\n" + "=" * 60)
    print("STEP 3: Plotting figures")
    print("=" * 60)
    run_command([sys.executable, "plot_figures.py"], cwd=script_dir)

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"Checkpoints/npz -> {cfg.CHECKPOINT_DIR}")
    print(f"Figures         -> {cfg.FIGURE_DIR}")

if __name__ == "__main__":
    main()

