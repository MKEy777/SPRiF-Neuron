import os
import argparse

SELF_DIR = os.path.dirname(os.path.abspath(__file__))

CODE_ROOT = os.path.dirname(os.path.dirname(SELF_DIR))

SMNIST_DIR = os.path.join(CODE_ROOT, "Task_S-MNIST")
GSC_DIR = os.path.join(CODE_ROOT, "Task_GSC")

CHECKPOINT_DIR = os.path.join(SELF_DIR, "checkpoints")
DATA_ROOT = os.path.join(SMNIST_DIR, "data")

FIGURE_DIR = os.path.join(SELF_DIR, "figures")

INPUT_SIZE = 1
HIDDEN_SIZES = [64, 210]
NUM_CLASSES = 10
MODE = "srnn"
WARMUP_STEPS = 0

LR = 1e-2
EPOCHS = 60
BATCH_SIZE = 512
TBPTT_LEN = 262
SCHEDULER_STEP = 25
SCHEDULER_GAMMA = 0.1
SEED = 0

GRID_RESOLUTION = 25
GRID_RANGE = 1.0
DIRECTION_SEED = 123
LANDSCAPE_EVAL_BATCHES = 4

def get_args():
    parser = argparse.ArgumentParser(description="S-MNIST 损失景观实验")
    parser.add_argument("--neuron", type=str, default="both",
                        choices=["sprif", "lif", "both"],
                        help="训练/分析哪个神经元")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-resolution", type=int, default=GRID_RESOLUTION)
    parser.add_argument("--grid-range", type=float, default=GRID_RANGE)
    parser.add_argument("--eval-batches", type=int, default=LANDSCAPE_EVAL_BATCHES,
                        help="每个网格点估计 loss 用的 batch 数")
    parser.add_argument("--subset", type=int, default=0,
                        help="仅用前 N 个训练样本（0=全量，用于快速验证）")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()

