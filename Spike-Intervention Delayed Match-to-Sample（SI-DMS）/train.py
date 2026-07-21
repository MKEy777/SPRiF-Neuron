import argparse, json
from pathlib import Path
import torch

from sidms.config import ExperimentConfig
from sidms.engine import train_model


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--config", default="config/default.yaml"); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="results"); p.add_argument("--steps", type=int)
    args = p.parse_args(); cfg = ExperimentConfig.from_yaml(args.config)
    if args.steps is not None: cfg.train.steps = args.steps
    report_every = max(1, cfg.train.steps // 20)
    def progress(row):
        if row["step"] == 1 or row["step"] % report_every == 0 or row["step"] == cfg.train.steps:
            print(f"  train {row['step']:>4}/{cfg.train.steps}  loss={row['loss']:.4f}  "
                  f"acc={row['accuracy']:.3f}  "
                  f"delay={row['delay_ms']}  K={row['intervention_count']}", flush=True)
    model, history = train_model(args.model, cfg, args.seed, progress=progress)
    out = Path(args.output) / args.model / f"seed_{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "name": args.model}, out / "checkpoint.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
