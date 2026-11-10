"""Export no-spike subthreshold decay from SI-DMS or benchmark ASRNN."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from checkpoint_io import file_sha256, find_prefix, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--legacy-raw", type=Path, help="Historical raw ASRNN tau arrays, in steps")
    p.add_argument("--task", choices=["GSC", "ECG"], help="Required with --legacy-raw")
    p.add_argument("--prefix", default=None)
    p.add_argument("--tau-key", default=None,
                   help="Benchmark tau in steps, e.g. dense_1.tau_m or tau_m_h")
    p.add_argument("--dt-ms", type=float, default=None,
                   help="Time step if absent from checkpoint config")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.legacy_raw:
        if not args.task:
            raise ValueError("--task is required with --legacy-raw")
        key = f"{args.task}_asrnn_tau_m"
        with np.load(args.legacy_raw, allow_pickle=False) as z:
            if key not in z:
                raise ValueError(f"{args.legacy_raw} lacks {key}")
            tau = np.asarray(z[key], dtype=float)
        if tau.ndim != 1 or not tau.size or not np.all(np.isfinite(tau) & (tau > 0)):
            raise ValueError("Historical tau must be a positive finite array in steps")
        alpha = np.exp(-1.0 / tau)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, alpha=alpha)
        meta = {"legacy_raw": str(args.legacy_raw), "sha256": file_sha256(args.legacy_raw),
                "task": args.task, "source_key": key, "tau_unit": "steps",
                "seed": None,
                "provenance_status": "historical data; original checkpoint and training seed must be verified separately"}
        source_record = args.legacy_raw.with_suffix(".source.json")
        if source_record.is_file():
            meta["historical_source_record"] = json.loads(source_record.read_text(encoding="utf-8-sig"))
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved {args.output} ({len(alpha)} neurons; historical baseline provenance unverified)")
        return
    import torch
    obj, state = load_checkpoint(args.checkpoint)
    name = obj.get("name")
    if name is not None and str(name).lower() != "asrnn":
        raise ValueError(f"Expected an ASRNN checkpoint, got {name!r}")
    if args.tau_key:
        if args.tau_key not in state:
            raise ValueError(f"Missing ASRNN time-constant key {args.tau_key!r}")
        tau = state[args.tau_key].detach().cpu().numpy().astype(float)
        if tau.ndim != 1 or not tau.size or not np.all(np.isfinite(tau) & (tau > 0)):
            raise ValueError("Benchmark time constants must be positive and finite")
        alpha = np.exp(-1.0 / tau)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, alpha=alpha)
        meta = {"checkpoint": str(args.checkpoint), "sha256": file_sha256(args.checkpoint),
                "source_key": args.tau_key, "tau_unit": "steps", "model": name,
                "task": obj.get("task"), "seed": obj.get("seed"),
                "protocol_version": obj.get("protocol_version"), "smoke": obj.get("smoke")}
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved {args.output} ({len(alpha)} benchmark neurons)")
        return
    prefix = find_prefix(state, ["log_tau_m"], args.prefix)
    dt_ms = args.dt_ms
    if dt_ms is None:
        dt_ms = obj.get("config", {}).get("task", {}).get("dt_ms")
    if dt_ms is None or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("A positive dt in milliseconds is required")
    log_tau = state[prefix + "log_tau_m"].detach().cpu()
    dt = float(dt_ms) / 1000.0
    alpha = torch.exp(-dt / torch.exp(log_tau).clamp_min(dt)).numpy()
    if alpha.ndim != 1 or not np.all(np.isfinite(alpha) & (alpha > 0) & (alpha < 1)):
        raise ValueError("Invalid ASRNN decay values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, alpha=alpha, dt_ms=np.array(dt_ms))
    meta = {"checkpoint": str(args.checkpoint), "sha256": file_sha256(args.checkpoint),
            "prefix": prefix, "dt_ms": float(dt_ms), "model": name,
            "seed": obj.get("seed"), "training": obj.get("training"),
            "protocol_version": obj.get("protocol_version"), "smoke": obj.get("smoke")}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {args.output} ({len(alpha)} neurons)")


if __name__ == "__main__":
    main()
