"""Export SPRiF spectral parameters from a checkpoint without training."""
import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_io import file_sha256, find_prefix, load_checkpoint
from common import parameters


def export(args):
    import torch

    obj, state = load_checkpoint(args.checkpoint)
    model = obj.get("name")
    if isinstance(model, str) and model.lower() not in {"sprif_scalar", "memory_reset5", "sprif"}:
        raise ValueError(f"Checkpoint model is {model!r}, not a SPRiF spectral cell")
    keys = ["alpha_raw", "rho_raw", "omega_raw", "eta_raw", "bar_G"]
    prefix = find_prefix(state, keys, args.prefix)
    values = {k: state[prefix + k].detach().cpu() for k in keys}
    cfg = obj.get("config", {})
    threshold = args.threshold
    if threshold is None and isinstance(cfg, dict):
        threshold = cfg.get("model", {}).get("threshold")
    if threshold is None:
        raise ValueError("Threshold not recorded in checkpoint; provide --threshold explicitly")
    result = {k: torch.sigmoid(values[k + "_raw"]).numpy() for k in ("alpha", "rho", "eta")}
    result.update(omega=(np.pi * torch.sigmoid(values["omega_raw"])).numpy(),
                  bar_G=values["bar_G"].numpy(), threshold=np.array(threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    parameters(args.output)
    meta = {"checkpoint": str(args.checkpoint), "sha256": file_sha256(args.checkpoint),
            "layer_prefix": prefix, "threshold": float(threshold), "model": model,
            "seed": obj.get("seed"), "training": obj.get("training"),
            "dt_ms": cfg.get("task", {}).get("dt_ms") if isinstance(cfg, dict) else None,
            "protocol_version": obj.get("protocol_version"), "smoke": obj.get("smoke")}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {args.output} ({len(result['alpha'])} neurons; prefix {prefix!r})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prefix", default=None, help="Exact layer prefix; inferred only when unique")
    p.add_argument("--threshold", type=float, help="Required if absent from checkpoint config")
    p.add_argument("--output", type=Path, required=True)
    export(p.parse_args())


if __name__ == "__main__":
    main()
