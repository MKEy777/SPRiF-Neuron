import argparse, csv, json
from pathlib import Path
import torch

from sidms.config import ExperimentConfig
from sidms.engine import evaluate_grid
from sidms.models import SIDMSNetwork


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config/default.yaml"); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batches", type=int, default=20); p.add_argument("--batch-size", type=int)
    args = p.parse_args(); cfg = ExperimentConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu"); model = SIDMSNetwork(ckpt["name"], cfg)
    model.load_state_dict(ckpt["model"]); rows = evaluate_grid(model, cfg, args.seed, args.batches, args.batch_size)
    out = Path(args.checkpoint).parent
    (out / "eval_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
