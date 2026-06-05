import json
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def dump_json(obj: Any, fdir: str, name: str):
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir: str, name: str):
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find json file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_dataset_wtime(mat_data):
    """
    ECG helper kept compatible with legacy Rhythm-SNN scripts.
    Input mat_data expected keys: x, y, t.
    """
    X = mat_data["x"]
    Y = mat_data["y"]
    t = mat_data["t"]
    Y = np.argmax(Y[:, :, :], axis=-1)
    d1, d2 = t.shape
    dt = np.zeros((d1, d2))
    for trace in range(d1):
        dt[trace, 0] = 1
        dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]
    return dt, X, Y


def load_max_i(mat_data):
    max_i = mat_data["max_i"]
    return np.array(max_i.squeeze(), dtype=np.float16)
