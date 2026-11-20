"""Record one trained PS-MNIST SPRiF network without performing training."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from checkpoint_io import file_sha256, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="Real preprocessed sample: NPY or NPZ key input")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--threshold", type=float)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    code = args.code_root / "core_algorithm" / "sprif_layer.py"
    if not code.is_file():
        raise FileNotFoundError(f"Missing SPRiF model implementation: {code}")
    obj, weights = load_checkpoint(args.checkpoint)
    if obj.get("name") is not None and obj.get("name") != "sprif_scalar":
        raise ValueError("The mechanism plot requires the scalar-reset SPRiF model")
    if obj.get("task") is not None and obj.get("task") != "pSMNIST":
        raise ValueError("Checkpoint task is not pSMNIST")
    if obj.get("seed") is not None and obj.get("seed") != args.seed:
        raise ValueError("Checkpoint seed does not match --seed")
    if obj.get("smoke") is True:
        raise ValueError("Smoke checkpoints cannot be used as formal trajectory data")
    threshold = args.threshold
    recorded_threshold = obj.get("config", {}).get("model", {}).get("threshold")
    if threshold is None:
        threshold = recorded_threshold
    elif recorded_threshold is not None and not np.isclose(threshold, recorded_threshold):
        raise ValueError("Specified threshold does not match checkpoint metadata")
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Provide the checkpoint's positive threshold")
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            sequence = np.asarray(loaded["input"], dtype=np.float32)
        finally:
            loaded.close()
    else:
        sequence = np.asarray(loaded, dtype=np.float32)
    if sequence.ndim != 2 or not sequence.size or not np.isfinite(sequence).all():
        raise ValueError("Input must be a finite [time, feature] sequence")
    indices = sorted({int(match.group(1)) for key in weights
                      if (match := re.fullmatch(r"layers\.(\d+)\.alpha_raw", key))})
    if indices != list(range(len(indices))) or args.layer not in indices:
        raise ValueError(f"Requested hidden layer {args.layer} not present in checkpoint")
    sys.path.insert(0, str(args.code_root.resolve()))
    import torch
    from core_algorithm.sprif_layer import SPRiFNeuronLayer

    layers = []
    for index in range(args.layer + 1):
        prefix = f"layers.{index}."
        part = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        if "input_linear.weight" not in part:
            raise ValueError(f"Missing input weights for layer {index}")
        matrix = part["input_linear.weight"]
        layer = SPRiFNeuronLayer(int(matrix.shape[1]), int(matrix.shape[0]), threshold=float(threshold),
                                 recurrent="recurrent_linear.weight" in part,
                                 bias="input_linear.bias" in part).to(args.device)
        layer.load_state_dict(part, strict=True)
        layer.eval()
        layers.append(layer)
    if layers[0].input_size != sequence.shape[1] or any(a.hidden_size != b.input_size for a, b in zip(layers, layers[1:])):
        raise ValueError("Input features or hidden layer dimensions do not match the checkpoint")
    rows = {key: [] for key in ("x_pre", "x_post", "h", "u_pre", "u_post", "spike")}
    with torch.no_grad():
        states = [layer.init_state(1, device=args.device) for layer in layers]
        runtimes = [layer._precompute_runtime_params() for layer in layers]
        for step in sequence:
            out = torch.as_tensor(step, device=args.device).unsqueeze(0)
            for index, layer in enumerate(layers):
                old = states[index]
                current = layer.input_linear(out)
                if layer.recurrent:
                    current = current + layer.recurrent_linear(old["prev_spike"])
                x_pre = layer._slow_flow(old["x"], current, runtimes[index])
                spike, membrane, new = layer.forward_step(out, old, runtimes[index], input_current=current)
                if index == args.layer:
                    fields = {"x_pre": x_pre, "x_post": new["x"], "h": new["h"],
                              "u_pre": membrane, "u_post": new["u"], "spike": spike}
                    for key, value in fields.items():
                        rows[key].append(value[0].detach().cpu().numpy().copy())
                states[index] = new
                out = spike
    arrays = {key: np.stack(value) for key, value in rows.items()}
    arrays["threshold"] = np.array(float(threshold))
    meta = {"format": "psmnist-sprif-trajectory-v1", "model": "SPRiF", "task": "pSMNIST",
            "scope": "one real preprocessed test sample replayed through one trained network",
            "seed": args.seed, "sample_id": args.sample_id, "layer": args.layer,
            "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
            "input": {"path": str(args.input), "sha256": file_sha256(args.input)},
            "implementation_sha256": file_sha256(code), "threshold": float(threshold),
            "protocol_version": obj.get("protocol_version"), "smoke": obj.get("smoke")}
    arrays["metadata_json"] = np.array(json.dumps(meta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved SPRiF trajectory: {args.output}")


if __name__ == "__main__":
    main()
