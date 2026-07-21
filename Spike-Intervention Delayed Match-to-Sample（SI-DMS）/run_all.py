import argparse, json
from pathlib import Path
import torch

from sidms.config import ExperimentConfig
from sidms.engine import evaluate_grid, train_model


MODELS = ["sprif_full", "sprif_merged", "sprif_lambda0", "lif", "asrnn", "brf"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output", default="results"); p.add_argument("--steps", type=int)
    p.add_argument("--eval-batches", type=int, default=20); p.add_argument("--models", nargs="*", default=MODELS)
    args = p.parse_args(); cfg = ExperimentConfig.from_yaml(args.config)
    if args.steps is not None: cfg.train.steps = args.steps
    root = Path(args.output); all_rows = []
    for name in args.models:
        for seed in cfg.train.seeds:
            out = root / name / f"seed_{seed}"; out.mkdir(parents=True, exist_ok=True)
            print(f"\n[{name} | seed={seed}] training {cfg.train.steps} steps", flush=True)
            report_every = max(1, cfg.train.steps // 20)
            def train_progress(row):
                if row["step"] == 1 or row["step"] % report_every == 0 or row["step"] == cfg.train.steps:
                    clean = row.get("clean_accuracy")
                    clean_str = f"  clean_acc={clean:.3f}" if clean is not None else ""
                    print(f"  train {row['step']:>4}/{cfg.train.steps}  loss={row['loss']:.4f}  "
                          f"acc={row['accuracy']:.3f}{clean_str}  delay={row['delay_ms']}  K={row['intervention_count']}", flush=True)
            model, history = train_model(name, cfg, seed, progress=train_progress)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "name": name}, out / "checkpoint.pt")
            (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            n_k0 = 1 if 0 in cfg.task.eval_intervention_counts else 0
            n_kpos = len([k for k in cfg.task.eval_intervention_counts if k != 0])
            total_cells = len(cfg.task.eval_delays_ms) * (
                n_k0 + n_kpos * len(cfg.task.eval_intervention_fractions))
            completed = 0
            print(f"[{name} | seed={seed}] evaluating {total_cells} delay x K cells "
                  f"({args.eval_batches} batches each)", flush=True)
            def eval_progress(row):
                nonlocal completed
                completed += 1
                print(f"  eval {completed:>2}/{total_cells}  delay={row['delay_ms']}  "
                      f"K={row['intervention_count']}  f={row.get('intervention_fraction', 0):.2f}  acc={row['accuracy']:.3f}", flush=True)
            rows = evaluate_grid(model, cfg, seed, args.eval_batches, progress=eval_progress)
            (out / "eval_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); all_rows += rows
            root.mkdir(parents=True, exist_ok=True)
            (root / "all_metrics.partial.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
            print(f"[{name} | seed={seed}] saved to {out}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "all_metrics.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
