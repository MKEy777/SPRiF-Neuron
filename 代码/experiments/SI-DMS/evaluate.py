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
    model.load_state_dict(ckpt["model"])
    n_k0 = 1 if 0 in cfg.task.eval_intervention_counts else 0
    n_kpos = len([k for k in cfg.task.eval_intervention_counts if k != 0])
    total_cells = len(cfg.task.eval_delays_ms) * (
        n_k0 + n_kpos * len(cfg.task.eval_intervention_fractions)
    )
    completed = 0
    def progress(row):
        nonlocal completed
        completed += 1
        print(f"  eval {completed:>2}/{total_cells}  delay={row['delay_ms']}  "
              f"K={row['intervention_count']}  acc={row['accuracy']:.3f}", flush=True)
    rows = evaluate_grid(model, cfg, args.seed, args.batches, args.batch_size, progress=progress)
    out = Path(args.checkpoint).parent
    (out / "eval_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()

