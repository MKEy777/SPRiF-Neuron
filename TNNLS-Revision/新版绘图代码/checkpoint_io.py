"""Read model weights from supported checkpoint wrappers; never trains a model."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint must be a dictionary: {path}")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(obj.get(key), dict):
            return obj, obj[key]
    return obj, obj


def find_prefix(state, keys, requested=None):
    if requested is not None:
        prefixes = [requested]
    else:
        first = keys[0]
        prefixes = sorted({key[:-len(first)] for key in state if key.endswith(first)})
    matches = [prefix for prefix in prefixes if all(prefix + key in state for key in keys)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one layer with {keys}; found prefixes {matches}. "
            "Specify its exact --prefix, including the trailing dot."
        )
    return matches[0]
