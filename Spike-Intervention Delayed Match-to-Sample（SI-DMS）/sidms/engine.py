from __future__ import annotations

import random

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import make_batch
from .models import SIDMSNetwork


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(value: str):
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def train_model(name: str, cfg: ExperimentConfig, seed: int, device: str | None = None,
                progress=None):
    seed_everything(seed)
    device = resolve_device(device or cfg.train.device)
    model = SIDMSNetwork(name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    history = []
    rng = random.Random(seed)
    model.train()
    for step in range(cfg.train.steps):
        delay = rng.choice(cfg.task.train_delays_ms)
        count = rng.choice(cfg.task.train_intervention_counts)
        batch = make_batch(cfg, cfg.train.batch_size, delay, count,
                           cfg.model.hidden_size, seed=seed * 1_000_003 + step, device=device)
        out = model(batch.x, batch.intervention)
        ce = F.cross_entropy(out.logits, batch.y)
        rate_loss = (out.natural_rate_tensor - cfg.train.rate_target).square()
        loss = ce + cfg.train.rate_weight * rate_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        row = {"step": step + 1, "loss": float(loss.detach()),
               "accuracy": float((out.logits.argmax(-1) == batch.y).float().mean()),
               "natural_rate": out.natural_rate, "delay_ms": delay,
               "intervention_count": count}
        history.append(row)
        if progress is not None: progress(row)
    return model, history


@torch.no_grad()
def evaluate_grid(model: SIDMSNetwork, cfg: ExperimentConfig, seed: int, batches: int = 20,
                  batch_size: int | None = None, device: str | None = None, progress=None):
    device = resolve_device(device or cfg.train.device)
    model.to(device).eval(); batch_size = batch_size or cfg.train.batch_size
    rows = []
    for d_i, delay in enumerate(cfg.task.eval_delays_ms):
        for k_i, count in enumerate(cfg.task.eval_intervention_counts):
            correct = total = 0; rates = []; hits = []
            for b in range(batches):
                batch_seed = seed * 10_000_019 + d_i * 100_003 + k_i * 1009 + b
                batch = make_batch(cfg, batch_size, delay, count, cfg.model.hidden_size,
                                   seed=batch_seed, device=device)
                out = model(batch.x, batch.intervention)
                correct += int((out.logits.argmax(-1) == batch.y).sum()); total += batch_size
                rates.append(out.natural_rate); hits.append(out.forced_hit_rate)
            row = {"model": model.name, "seed": seed, "delay_ms": delay,
                   "intervention_count": count, "accuracy": correct / total,
                   "natural_rate": sum(rates) / len(rates),
                   "forced_hit_rate": sum(hits) / len(hits)}
            rows.append(row)
            if progress is not None: progress(row)
    return rows
