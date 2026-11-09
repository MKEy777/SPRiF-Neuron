"""Replay two trained SI-DMS checkpoints on the same generated trial; no training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from checkpoint_io import file_sha256, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-root", type=Path, required=True, help="Directory containing sidms/")
    p.add_argument("--sprif-checkpoint", type=Path, required=True)
    p.add_argument("--reset-checkpoint", type=Path, required=True)
    p.add_argument("--delay-ms", type=int, default=1600)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--fraction", type=float, default=.15)
    p.add_argument("--input-seed", type=int, default=3000000000000)
    p.add_argument("--mask-seed", type=int, default=4000000000000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-smoke", action="store_true", help="Development only; never a formal result")
    p.add_argument("--output", type=Path, default=BASE / "原始数据" / "trajectories.npz")
    args = p.parse_args()
    if not (args.code_root / "sidms" / "models.py").is_file():
        raise FileNotFoundError(f"No sidms/models.py below {args.code_root}")
    if args.count < 1 or not (0 < args.fraction <= 1):
        raise ValueError("Use at least one intervention and a fraction in (0, 1]")
    sys.path.insert(0, str(args.code_root.resolve()))
    import torch
    from sidms.config import ExperimentConfig, TaskConfig, ModelConfig, TrainConfig
    from sidms.data import make_batch
    from sidms.models import SIDMSNetwork

    sources = [("sprif", "sprif_scalar", args.sprif_checkpoint),
               ("reset", "memory_reset5", args.reset_checkpoint)]
    checkpoints = []
    for label, expected_name, path in sources:
        obj, state = load_checkpoint(path)
        if obj.get("name") != expected_name or "config" not in obj or "model" not in obj:
            raise ValueError(f"{path} is not a {expected_name} SI-DMS checkpoint")
        if obj.get("smoke") is not False and not args.allow_smoke:
            raise ValueError(f"{path} is smoke or lacks formal-run metadata; use --allow-smoke only for testing")
        checkpoints.append((label, path, obj, state))
    a, b = (item[2] for item in checkpoints)
    for key in ("training", "seed", "protocol_version"):
        if a.get(key) is None or a.get(key) != b.get(key):
            raise ValueError(f"Checkpoints must have matching nonempty {key}")
    if a["config"] != b["config"]:
        raise ValueError("Checkpoint task, model, and training configurations must match")
    raw = a["config"]
    cfg = ExperimentConfig(TaskConfig(**raw["task"]), ModelConfig(**raw["model"]), TrainConfig(**raw["train"]))
    batch = make_batch(cfg, batch_size=1, delay_ms=args.delay_ms, intervention_count=args.count,
                       hidden_size=cfg.model.hidden_size, seed=args.input_seed,
                       mask_seed=args.mask_seed, fraction=args.fraction, device=args.device)
    arrays = {"input": batch.x[0].cpu().numpy(), "forced": batch.intervention[0].cpu().numpy(),
              "dt_ms": np.array(cfg.task.dt_ms), "threshold": np.array(cfg.model.threshold)}
    arrays["event_mask"] = arrays["forced"]
    with torch.no_grad():
        for label, path, obj, state_dict in checkpoints:
            model = SIDMSNetwork(obj["name"], cfg).to(args.device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            state = model.cell.initial_state(1, args.device)
            prev = torch.zeros(1, cfg.model.hidden_size, device=args.device)
            rows = {k: [] for k in ("x_pre", "x_post", "h_pre", "h_post", "u_pre", "u_post", "spike", "current")}
            for t in range(batch.x.shape[1]):
                current = model.cell.drive(batch.x[:, t], prev)
                spike, new, diag = model.cell.step(batch.x[:, t], state, batch.intervention[:, t], prev)
                fields = {"x_pre": diag["slow_pre_reset"], "x_post": new["x"],
                          "h_pre": diag["h_pre_reset"], "h_post": new["h"],
                          "u_pre": diag["mem_pre_reset"], "u_post": new["u"],
                          "spike": spike, "current": current}
                for key, value in fields.items():
                    rows[key].append(value[0].detach().cpu().numpy().copy())
                state, prev = new, spike
            for key, value in rows.items():
                arrays[f"{label}_{key}"] = np.stack(value)
            if not np.all(arrays[f"{label}_spike"][arrays["forced"]] > .5):
                raise RuntimeError(f"{label}: a forced intervention did not produce a spike")
    meta = {"format": "sidms-actual-network-trajectories-v1", "event_type": "forced",
            "scope": "one generated trial; independent trained networks",
            "seed": a["seed"], "training": a["training"], "protocol_version": a["protocol_version"],
            "smoke": bool(a.get("smoke") or b.get("smoke")), "delay_ms": args.delay_ms,
            "intervention_count": args.count, "intervention_fraction": args.fraction,
            "input_seed": args.input_seed, "mask_seed": args.mask_seed,
            "implementation_sha256": {name: file_sha256(args.code_root / "sidms" / name)
                                       for name in ("models.py", "cells.py", "data.py")},
            "checkpoints": {label: {"path": str(path), "sha256": file_sha256(path)}
                            for label, path, _, _ in checkpoints}}
    arrays["metadata_json"] = np.array(json.dumps(meta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: {arrays['forced'].sum()} forced neuron-events")


if __name__ == "__main__":
    main()
