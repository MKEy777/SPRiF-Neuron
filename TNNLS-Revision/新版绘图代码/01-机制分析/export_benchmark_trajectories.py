"""Replay current benchmark SPRiF/MEMORY_RESET5 hidden layers on one real input sequence."""
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
    p.add_argument("--code-root", type=Path, required=True, help="Task directory containing core_algorithm/sprif_layer.py")
    p.add_argument("--sprif-checkpoint", type=Path, required=True)
    p.add_argument("--reset-checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="Actual preprocessed sample: NPY [time, features] or NPZ key input")
    p.add_argument("--task", choices=["pSMNIST", "SHD"], required=True)
    p.add_argument("--sample-id", required=True, help="Identifier of the original test sample")
    p.add_argument("--seed", type=int, required=True, help="Training seed for the two checkpoints")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=BASE / "原始数据" / "benchmark_trajectories.npz")
    args = p.parse_args()
    if args.layer < 0 or not np.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError("layer must be nonnegative and threshold positive")
    code = args.code_root / "core_algorithm" / "sprif_layer.py"
    if not code.is_file():
        raise FileNotFoundError(f"Missing current benchmark layer implementation: {code}")
    if file_sha256(args.sprif_checkpoint) == file_sha256(args.reset_checkpoint):
        raise ValueError("Provide two independently trained checkpoints, not the same file twice")
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            sequence = np.array(loaded["input"], dtype=np.float32)
        finally:
            loaded.close()
    else:
        sequence = np.asarray(loaded, dtype=np.float32)
    if sequence.ndim != 2 or not sequence.size or not np.isfinite(sequence).all():
        raise ValueError("Preprocessed input must be finite [time, features]")
    sys.path.insert(0, str(args.code_root.resolve()))
    import torch
    from core_algorithm.sprif_layer import SPRiFNeuronLayer

    sources = [("sprif", args.sprif_checkpoint), ("reset", args.reset_checkpoint)]
    arrays = {"input": sequence, "threshold": np.array(args.threshold)}
    layer_shapes = []
    checkpoint_metadata = {}
    with torch.no_grad():
        for label, path in sources:
            obj, state_dict = load_checkpoint(path)
            if obj.get("smoke") is True:
                raise ValueError(f"Smoke checkpoint is not a benchmark result: {path}")
            expected_name = "sprif_scalar" if label == "sprif" else "memory_reset5"
            if isinstance(obj.get("name"), str) and obj["name"] != expected_name:
                raise ValueError(f"{path}: expected model {expected_name}, got {obj['name']}")
            if obj.get("seed") is not None and obj["seed"] != args.seed:
                raise ValueError(f"{path}: checkpoint seed disagrees with --seed")
            if obj.get("task") is not None and obj["task"] != args.task:
                raise ValueError(f"{path}: checkpoint task disagrees with --task")
            checkpoint_metadata[label] = {key: obj.get(key) for key in ("name", "seed", "task", "protocol_version", "training_args")}
            indices = sorted({int(match.group(1)) for key in state_dict
                              if (match := re.fullmatch(r"layers\.(\d+)\.alpha_raw", key))})
            if indices != list(range(len(indices))) or args.layer not in indices:
                raise ValueError(f"{path}: expected current-model layers.0, layers.1, ...")
            layers = []
            for index in range(args.layer + 1):
                prefix = f"layers.{index}."
                layer_state = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
                required = {"alpha_raw", "rho_raw", "omega_raw", "eta_raw", "bar_G", "input_linear.weight"}
                if not required.issubset(layer_state):
                    raise ValueError(f"{path}: layer {index} is not a current SPRiF spectral layer")
                weight = layer_state["input_linear.weight"]
                recurrent = "recurrent_linear.weight" in layer_state
                bias = "input_linear.bias" in layer_state
                layer = SPRiFNeuronLayer(int(weight.shape[1]), int(weight.shape[0]), threshold=args.threshold,
                                         recurrent=recurrent, bias=bias).to(args.device)
                layer.load_state_dict(layer_state, strict=True)
                layer.eval()
                layers.append(layer)
            shape = [(layer.input_size, layer.hidden_size, layer.recurrent) for layer in layers]
            layer_shapes.append(shape)
            if shape[0][0] != sequence.shape[1] or any(shape[i][1] != shape[i+1][0] for i in range(len(shape)-1)):
                raise ValueError(f"{path}: input or layer dimensions do not match")
            states = [layer.init_state(1, device=args.device) for layer in layers]
            runtimes = [layer._precompute_runtime_params() for layer in layers]
            rows = {key: [] for key in ("x_pre", "x_post", "h_pre", "h_post", "u_pre", "u_post", "spike", "current")}
            for step in sequence:
                out = torch.as_tensor(step, device=args.device).unsqueeze(0)
                for index, layer in enumerate(layers):
                    old = states[index]
                    current = layer.input_linear(out)
                    if layer.recurrent:
                        current = current + layer.recurrent_linear(old["prev_spike"])
                    spike, membrane, new = layer.forward_step(out, old, runtimes[index], input_current=current)
                    x_pre = new["x"].clone()
                    if label == "reset":
                        alpha = torch.sigmoid(layer.alpha_raw)
                        rho = torch.sigmoid(layer.rho_raw)
                        direction = torch.stack((1-alpha, 1-rho, torch.zeros_like(alpha)), dim=-1)
                        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
                        if not bool(torch.isfinite(norm).all()) or bool((norm == 0).any()):
                            raise ValueError("MEMORY_RESET5 input direction has an invalid norm")
                        direction = direction / norm
                        new["x"] = x_pre - args.threshold * spike.unsqueeze(-1) * direction
                    if index == args.layer:
                        fields = {"x_pre": x_pre, "x_post": new["x"], "h_pre": new["h"], "h_post": new["h"],
                                  "u_pre": membrane, "u_post": new["u"], "spike": spike, "current": current}
                        for key, value in fields.items():
                            rows[key].append(value[0].detach().cpu().numpy().copy())
                    states[index] = new
                    out = spike
            for key, value in rows.items():
                arrays[f"{label}_{key}"] = np.stack(value)
    if layer_shapes[0] != layer_shapes[1]:
        raise ValueError("Two checkpoints must have matching hidden-layer architecture")
    left, right = checkpoint_metadata["sprif"], checkpoint_metadata["reset"]
    if left["protocol_version"] is not None and right["protocol_version"] is not None and left["protocol_version"] != right["protocol_version"]:
        raise ValueError("Checkpoint protocol versions differ")
    if isinstance(left["training_args"], dict) and isinstance(right["training_args"], dict):
        excluded = {"model", "checkpoint_path"}
        a = {key: value for key, value in left["training_args"].items() if key not in excluded}
        b = {key: value for key, value in right["training_args"].items() if key not in excluded}
        if a != b:
            raise ValueError("Checkpoint training arguments differ beyond the model variant")
    arrays["event_mask"] = (arrays["sprif_spike"] > .5) & (arrays["reset_spike"] > .5)
    if not arrays["event_mask"].any():
        raise ValueError("No simultaneous natural spike at the same index; provide another real sample")
    meta = {"format": "benchmark-reconstructed-network-trajectories-v1", "event_type": "coincident_natural",
            "scope": "one preprocessed test sample; coincident natural spikes; model reconstructed from checkpoints",
            "task": args.task, "sample_id": args.sample_id, "seed": args.seed, "layer": args.layer,
            "threshold": args.threshold, "smoke": None,
            "model_rule": "SPRiF scalar reset; MEMORY_RESET5 adds -theta*z*B/||B|| after reading x",
            "control_provenance": "user must verify reset checkpoint was trained with that rule",
            "checkpoint_metadata": checkpoint_metadata,
            "input": {"path": str(args.input), "sha256": file_sha256(args.input)},
            "layer_source_sha256": file_sha256(code),
            "checkpoints": {label: {"path": str(path), "sha256": file_sha256(path)} for label, path in sources}}
    arrays["metadata_json"] = np.array(json.dumps(meta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: {int(arrays['event_mask'].sum())} simultaneous natural neuron-events")


if __name__ == "__main__":
    main()
