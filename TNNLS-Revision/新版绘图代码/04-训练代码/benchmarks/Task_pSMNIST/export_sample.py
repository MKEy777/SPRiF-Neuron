"""Save one actual PS-MNIST test sequence using the training permutation."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = torchvision.datasets.MNIST(root=str(args.data_root), train=False, download=False, transform=transform)
    if not 0 <= args.index < len(dataset):
        raise ValueError(f"test sample index must be in [0, {len(dataset)})")
    image, label = dataset[args.index]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    permutation = torch.randperm(784, generator=generator)
    sequence = image.reshape(-1)[permutation].reshape(-1, 1).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input=sequence, seed=np.array(args.seed),
                        sample_index=np.array(args.index), label=np.array(label))
    meta = {"dataset": "MNIST test", "root": str(args.data_root), "seed": args.seed,
            "sample_index": args.index, "label": int(label),
            "preprocessing": "ToTensor, Normalize(0.1307,0.3081), torch.randperm(784) with run seed"}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved real test sequence to {args.output}")


if __name__ == "__main__":
    main()
