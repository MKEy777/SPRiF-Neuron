import json
import os
import random
import time
from typing import Any

import numpy as np
import scipy.signal as ssg
import torch
import torch.nn as nn

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

def convert_seq(x, threshold=0.03):
    l = len(x)
    x = ssg.savgol_filter(x, 5, 3)
    X = np.zeros((l, 2))
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] >= threshold:
            X[i, 0] = 1
        elif x[i] - x[i + 1] >= threshold:
            X[i, 1] = 1
    return X

def expand_dim(x, N):
    y = np.zeros((x.shape[0], x.shape[1], N))
    for i in range(x.shape[0]):
        y[i, :, :] = np.tile(x[i, :], (N, 1)).transpose()
    return y

def lbl_to_spike(prediction):
    N = len(prediction)
    detections = np.zeros(N)
    for i in range(1, N):
        if prediction[i] != prediction[i - 1]:
            detections[i] = prediction[i] + 1
    return detections

def calculate_stats(prediction, lbl, tol):
    decisions = lbl_to_spike(prediction)
    labs = lbl_to_spike(lbl)
    lbl_indices = np.nonzero(labs)
    lbl_indices = np.array(lbl_indices).flatten()
    dist = np.zeros((len(lbl_indices), 6))
    for i in range(len(lbl_indices)):
        index = lbl_indices[i]
        lab = int(labs[index])
        dec_indices = np.array(np.nonzero((decisions - lab) == 0)).flatten()
        if len(dec_indices) == 0:
            dist[i, lab - 1] = 250
            continue
        j = np.argmin(np.abs(dec_indices - index))
        dist[i, lab - 1] = abs(dec_indices[j] - index)
        if dist[i, lab - 1] <= tol:
            decisions[dec_indices[j]] = 0
    mean_error = np.mean(dist, axis=0)
    TP = np.sum(dist <= tol, axis=0)
    FN = np.sum(dist > tol, axis=0)
    FP = np.zeros(6)
    for i in decisions[(decisions > 0)]:
        FP[int(i - 1)] += 1
    return mean_error, TP, FN, FP

def count_parameters(model):
    param_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            param_sum += p.numel()
    return param_sum

def save_checkpoint(epoch, model, optimizer, ckp_dir, best=True):
    if not os.path.isdir(ckp_dir):
        os.mkdir(ckp_dir)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(
        state,
        os.path.join(ckp_dir, "{0}.pt.tar".format("best" if best else "last")),
    )

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def get_logger(
    name,
    format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    file=False,
):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler() if not file else logging.FileHandler(name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def percentile(t: torch.Tensor, q: float):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

