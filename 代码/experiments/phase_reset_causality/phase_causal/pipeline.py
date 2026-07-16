from __future__ import annotations

import itertools
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import PhaseTrajectoryDataset
from .engine import evaluate_multi_event_batch, evaluate_paired_batch
from .models import build_model
from .reporting import paired_effect_rows, validate_unique_records


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_dir(output_root: Path, model_name: str, seed: int) -> Path:
    path = output_root / "raw" / model_name / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _loader(dataset, cfg: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.training.num_workers > 0,
    )


def _epoch(model, loader, cue_steps, device, optimizer=None, grad_clip=1.0) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for inputs, target, _, _ in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = nn.functional.mse_loss(output[:, cue_steps:], target[:, cue_steps:])
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        batch = inputs.shape[0]
        total_loss += float(loss.detach()) * batch
        total_samples += batch
    return total_loss / max(total_samples, 1)


def train_model_seed(
    model_name: str,
    seed: int,
    cfg: ExperimentConfig,
    output_root: str | Path,
    device: torch.device,
    *,
    cache_dir: str | None = None,
) -> Path:
    _set_seed(seed)
    output_root = Path(output_root)
    run_dir = _seed_dir(output_root, model_name, seed)
    train_data = PhaseTrajectoryDataset(cfg.training.train_samples, cfg.task, seed=10_000 + seed, cache_dir=cache_dir)
    val_data = PhaseTrajectoryDataset(cfg.training.val_samples, cfg.task, seed=20_000 + seed, cache_dir=cache_dir)
    train_loader = _loader(train_data, cfg, shuffle=True)
    val_loader = _loader(val_data, cfg, shuffle=False)
    model = build_model(model_name, cfg.task, cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.training.lr_step_size, gamma=cfg.training.lr_gamma,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    checkpoint_path = run_dir / "checkpoint.pt"
    for epoch in range(1, cfg.training.epochs + 1):
        train_mse = _epoch(
            model, train_loader, cfg.task.cue_steps, device,
            optimizer=optimizer, grad_clip=cfg.training.grad_clip,
        )
        with torch.no_grad():
            val_mse = _epoch(model, val_loader, cfg.task.cue_steps, device)
        history.append({"epoch": epoch, "train_delay_mse": train_mse, "val_delay_mse": val_mse})
        current_lr = scheduler.get_last_lr()[0]
        print(f"  epoch {epoch:3d}  train={train_mse:.6f}  val={val_mse:.6f}  lr={current_lr:.6f}")
        scheduler.step()
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_name": model_name,
                "seed": seed,
                "model_state": model.state_dict(),
                "best_val_delay_mse": best_val,
                "best_epoch": best_epoch,
                "config": asdict(cfg),
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= cfg.training.patience:
                break

    (run_dir / "train_history.json").write_text(
        json.dumps({
            "model": model_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_delay_mse": best_val,
            "epochs": history,
        }, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def _merge_batch_rows(batch_rows: list[list[dict]]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for rows in batch_rows:
        for row in rows:
            key = (row["mode"], row["event_step"], row["k"], row["gamma"])
            grouped.setdefault(key, []).append(row)
    merged = []
    identifiers = {"mode", "event_step", "k", "gamma"}
    count_fields = {"valid_samples", "invalid_samples"}
    for key, rows in grouped.items():
        output = dict(zip(("mode", "event_step", "k", "gamma"), key))
        valid_total = sum(row["valid_samples"] for row in rows)
        for field in count_fields:
            output[field] = sum(row[field] for row in rows)
        numeric_fields = set().union(*(row.keys() for row in rows)) - identifiers - count_fields
        for field in numeric_fields:
            values = [(float(row[field]), row["valid_samples"]) for row in rows if field in row]
            weight = sum(item[1] for item in values)
            output[field] = sum(value * n for value, n in values) / max(weight, 1)
        merged.append(output)
    return merged


def _save_representative(path: Path, artifacts: dict, event_step: int) -> None:
    mask = artifacts["mask"].bool()
    forced_modes = ("forced_no_reset", "fast_reset", "slow_reset", "both_reset")
    reference_spikes = artifacts["traces"]["forced_no_reset"]["spikes"][:, event_step]
    for mode in forced_modes:
        trace = artifacts["traces"][mode]
        if not torch.all(trace["forced_hit"][:, event_step][mask]):
            raise RuntimeError(f"representative {mode} trace has a missed forced spike")
        if not torch.all(trace["new_crossing"][:, event_step][mask]):
            raise RuntimeError(f"representative {mode} trace lacks a new threshold crossing")
        if not torch.equal(trace["spikes"][:, event_step], reference_spikes):
            raise RuntimeError("forced event spike tensors differ across SPRiF branches")

    no_reset = artifacts["traces"]["forced_no_reset"]
    fast_residual = torch.linalg.vector_norm(
        artifacts["traces"]["fast_reset"]["fast"][:, event_step]
        - no_reset["fast"][:, event_step], dim=-1,
    ).mean()
    slow_residual = torch.linalg.vector_norm(
        artifacts["traces"]["slow_reset"]["slow"][:, event_step]
        - no_reset["slow"][:, event_step], dim=-1,
    ).mean()
    if fast_residual <= 0 or slow_residual <= 0:
        raise RuntimeError("representative trace must contain non-zero reset residuals")

    sample_effect = (
        artifacts["sample_metrics"]["slow_reset"]["excess_auc"]
        - artifacts["sample_metrics"]["fast_reset"]["excess_auc"]
    )
    failure_index = int(torch.argmax(torch.abs(sample_effect)).item())
    post_stop = min(artifacts["target"].shape[1], event_step + 51)
    slow_output = artifacts["outputs"]["slow_reset"][:, event_step + 1 : post_stop]
    slow_radius = torch.linalg.vector_norm(slow_output, dim=-1).mean(dim=1)
    radius_collapse_index = int(torch.argmin(slow_radius).item())
    slow_spikes = artifacts["traces"]["slow_reset"]["spikes"][:, event_step + 1 : post_stop]
    repeat_firing_index = int(torch.argmax(slow_spikes.sum(dim=(1, 2))).item())
    failure_indices = {
        "phase_failure": failure_index,
        "radius_collapse": radius_collapse_index,
        "repeat_firing": repeat_firing_index,
    }
    arrays = {
        "event_step": np.asarray(event_step),
        "mask": mask[0].numpy(),
        "target": artifacts["target"][0].numpy(),
        "forced_event_spikes_identical": np.asarray(True),
        "fast_reset_residual_norm": np.asarray(float(fast_residual)),
        "slow_reset_residual_norm": np.asarray(float(slow_residual)),
        "failure_sample_index": np.asarray(
            int(artifacts["valid_indices"][failure_index].item())
        ),
        "failure_target": artifacts["target"][failure_index].numpy(),
    }
    for label, index in failure_indices.items():
        arrays[f"{label}_sample_index"] = np.asarray(
            int(artifacts["valid_indices"][index].item())
        )
        arrays[f"{label}_target"] = artifacts["target"][index].numpy()
    for mode, output in artifacts["outputs"].items():
        arrays[f"{mode}_output"] = output[0].numpy()
        arrays[f"{mode}_spikes"] = artifacts["traces"][mode]["spikes"][0].numpy()
        arrays[f"failure_{mode}_output"] = output[failure_index].numpy()
        for label, index in failure_indices.items():
            arrays[f"{label}_{mode}_output"] = output[index].numpy()
            arrays[f"{label}_{mode}_spikes"] = artifacts["traces"][mode]["spikes"][index].numpy()
        arrays[f"{mode}_forced_hit"] = artifacts["traces"][mode]["forced_hit"][0].numpy()
        arrays[f"{mode}_new_crossing"] = artifacts["traces"][mode]["new_crossing"][0].numpy()
        for key in ("slow", "fast", "slow_pre_reset", "fast_pre_reset"):
            if key in artifacts["traces"][mode]:
                arrays[f"{mode}_{key}"] = artifacts["traces"][mode][key][0].numpy()
    np.savez_compressed(path, **arrays)


def evaluate_checkpoint(
    model_name: str,
    seed: int,
    cfg: ExperimentConfig,
    output_root: str | Path,
    device: torch.device,
    run_sensitivity: bool = True,
    cache_dir: str | None = None,
) -> Path:
    output_root = Path(output_root)
    run_dir = _seed_dir(output_root, model_name, seed)
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, cfg.task, cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state"])

    test_data = PhaseTrajectoryDataset(
        cfg.training.test_samples, cfg.task, cfg.evaluation.test_seed, cache_dir=cache_dir
    )
    loader = _loader(test_data, cfg, shuffle=False)
    settings = [
        ("main", event_step, cfg.intervention.k, cfg.intervention.gamma)
        for event_step in cfg.intervention.event_steps
    ]
    if run_sensitivity:
        center = cfg.intervention.event_steps[len(cfg.intervention.event_steps) // 2]
        settings.extend(
            ("k_sweep", center, k, cfg.intervention.gamma)
            for k in cfg.intervention.k_values if k != cfg.intervention.k
        )
        settings.extend(
            ("gamma_sweep", center, cfg.intervention.k, gamma)
            for gamma in cfg.intervention.gamma_values if gamma != cfg.intervention.gamma
        )

    all_rows = []
    sample_effects = []
    representative_saved = False
    for sweep, event_step, k, gamma in settings:
        batch_rows = []
        sample_offset = 0
        for batch_index, (inputs, target, _, _) in enumerate(loader):
            rows, artifacts = evaluate_paired_batch(
                model, inputs.to(device), target.to(device), event_step, k, gamma,
                cfg.intervention.margin,
                mask_seed=seed * 1_000_003 + event_step * 101 + batch_index,
                cue_steps=cfg.task.cue_steps,
                evaluation_cfg=cfg.evaluation,
            )
            batch_rows.append(rows)
            if model_name == "sprif" and sweep == "main":
                slow = artifacts["sample_metrics"]["slow_reset"]["excess_auc"].numpy()
                fast = artifacts["sample_metrics"]["fast_reset"]["excess_auc"].numpy()
                for local_index, effect in zip(
                    artifacts["valid_indices"].numpy(), slow - fast
                ):
                    sample_effects.append({
                        "sample_index": int(sample_offset + local_index),
                        "event_step": int(event_step),
                        "slow_minus_fast_auc": float(effect),
                    })
            if model_name == "sprif" and not representative_saved and sweep == "main":
                _save_representative(run_dir / "representative_traces.npz", artifacts, event_step)
                representative_saved = True
            sample_offset += inputs.shape[0]
        for row in _merge_batch_rows(batch_rows):
            row.update({
                "model": model_name,
                "seed": seed,
                "dataset": "in_distribution",
                "sweep": sweep,
                "event_count": 1,
            })
            all_rows.append(row)

    if run_sensitivity and model_name == "sprif":
        start = max(cfg.task.cue_steps + 1, int(round(cfg.task.total_steps * 0.20)))
        end = min(
            cfg.task.total_steps - 1 - cfg.evaluation.recovery_window,
            int(round(cfg.task.total_steps * 0.87)),
        )
        for event_count in cfg.intervention.multi_event_counts:
            if event_count == 0:
                event_steps = ()
            elif event_count == 1:
                event_steps = ((start + end) // 2,)
            else:
                event_steps = tuple(np.linspace(start, end, event_count, dtype=int).tolist())
            batch_rows = []
            for batch_index, (inputs, target, _, _) in enumerate(loader):
                rows, _ = evaluate_multi_event_batch(
                    model, inputs.to(device), target.to(device), event_steps,
                    cfg.intervention.k, cfg.intervention.gamma, cfg.intervention.margin,
                    mask_seed=seed * 1_000_003 + event_count * 997 + batch_index,
                    cue_steps=cfg.task.cue_steps,
                    evaluation_cfg=cfg.evaluation,
                )
                batch_rows.append(rows)
            for row in _merge_batch_rows(batch_rows):
                row.update({
                    "model": model_name,
                    "seed": seed,
                    "dataset": "in_distribution",
                    "sweep": "multi_event",
                    "event_count": int(event_count),
                })
                all_rows.append(row)

        center = cfg.intervention.event_steps[len(cfg.intervention.event_steps) // 2]
        for period_index, period in enumerate(cfg.task.omega_periods):
            id_data = PhaseTrajectoryDataset(
                cfg.training.test_samples, cfg.task,
                cfg.evaluation.test_seed + 100_001 + period_index,
                omega_choices=(2.0 * np.pi / period,),
                cache_dir=cache_dir,
            )
            id_loader = _loader(id_data, cfg, shuffle=False)
            batch_rows = []
            for batch_index, (inputs, target, _, _) in enumerate(id_loader):
                rows, _ = evaluate_paired_batch(
                    model, inputs.to(device), target.to(device), center,
                    cfg.intervention.k, cfg.intervention.gamma, cfg.intervention.margin,
                    mask_seed=(
                        seed * 1_000_003 + center * 103
                        + 100_003 + period_index * 10_007 + batch_index
                    ),
                    cue_steps=cfg.task.cue_steps,
                    evaluation_cfg=cfg.evaluation,
                )
                batch_rows.append(rows)
            for row in _merge_batch_rows(batch_rows):
                row.update({
                    "model": model_name,
                    "seed": seed,
                    "dataset": f"seen_period_{period}",
                    "sweep": "id_frequency",
                    "event_count": 1,
                    "frequency_period": float(period),
                })
                all_rows.append(row)

        for period_index, period in enumerate(cfg.evaluation.ood_periods):
            ood_data = PhaseTrajectoryDataset(
                cfg.training.test_samples, cfg.task,
                cfg.evaluation.test_seed + 1 + period_index,
                omega_choices=(2.0 * np.pi / period,),
                cache_dir=cache_dir,
            )
            ood_loader = _loader(ood_data, cfg, shuffle=False)
            batch_rows = []
            for batch_index, (inputs, target, _, _) in enumerate(ood_loader):
                rows, _ = evaluate_paired_batch(
                    model, inputs.to(device), target.to(device), center,
                    cfg.intervention.k, cfg.intervention.gamma, cfg.intervention.margin,
                    mask_seed=(
                        seed * 1_000_003 + center * 103
                        + period_index * 10_007 + batch_index
                    ),
                    cue_steps=cfg.task.cue_steps,
                    evaluation_cfg=cfg.evaluation,
                )
                batch_rows.append(rows)
            for row in _merge_batch_rows(batch_rows):
                row.update({
                    "model": model_name,
                    "seed": seed,
                    "dataset": f"unseen_period_{period}",
                    "sweep": "ood_frequency",
                    "event_count": 1,
                    "frequency_period": float(period),
                })
                all_rows.append(row)

    clean_main = [row for row in all_rows if row["sweep"] == "main" and row["mode"] == "clean"]
    clean_mse = float(np.mean([row["delay_mse"] for row in clean_main]))
    clean_radius = float(np.mean([row["output_radius"] for row in clean_main]))
    clean_rate = float(np.mean([row["firing_rate"] for row in clean_main]))
    clean_gate = (
        clean_mse < cfg.evaluation.clean_mse_max
        and clean_radius > cfg.evaluation.clean_radius_min
        and clean_rate > 0.0
    )
    payload = {
        "model": model_name,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "clean_gate_passed": clean_gate,
        "clean_gate": {
            "delay_mse": clean_mse,
            "output_radius": clean_radius,
            "firing_rate": clean_rate,
        },
        "rows": all_rows,
        "sample_effects": sample_effects,
    }
    metrics_path = run_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path


def _exact_sign_flip_p(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float((values * np.asarray(signs)).mean()))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def _hierarchical_bootstrap(
    sample_effects_by_seed: dict[int, np.ndarray],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    seed_ids = sorted(sample_effects_by_seed)
    if not seed_ids:
        return np.asarray([]), np.asarray([])
    within_seed_bootstrap = np.empty((len(seed_ids), bootstrap_samples), dtype=np.float64)
    chunk_size = 100
    for seed_index, seed in enumerate(seed_ids):
        values = np.asarray(sample_effects_by_seed[seed], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"seed {seed} has no sample-level effects")
        for start in range(0, bootstrap_samples, chunk_size):
            stop = min(start + chunk_size, bootstrap_samples)
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            within_seed_bootstrap[seed_index, start:stop] = values[indices].mean(axis=1)
    selected_seeds = rng.integers(
        0, len(seed_ids), size=(bootstrap_samples, len(seed_ids))
    )
    replicate_index = np.arange(bootstrap_samples)[:, None]
    boot = within_seed_bootstrap[selected_seeds, replicate_index].mean(axis=1)
    seed_means = np.asarray([
        np.asarray(sample_effects_by_seed[seed], dtype=np.float64).mean()
        for seed in seed_ids
    ])
    return seed_means, boot


def aggregate_results(
    output_root: str | Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2027,
) -> dict[str, Path]:
    output_root = Path(output_root)
    payloads = []
    all_rows = []
    rows = []
    sample_effects_by_seed: dict[int, np.ndarray] = {}
    representative_sources: list[Path] = []
    for path in sorted((output_root / "raw").glob("*/seed_*/eval_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        payload_rows = payload["rows"]
        all_rows.extend(payload_rows)
        if payload.get("clean_gate_passed", False):
            rows.extend(payload_rows)
            if payload.get("model") == "sprif":
                representative_source = path.parent / "representative_traces.npz"
                if representative_source.exists():
                    representative_sources.append(representative_source)
                effects = [
                    float(item["slow_minus_fast_auc"])
                    for item in payload.get("sample_effects", [])
                ]
                if effects:
                    sample_effects_by_seed[int(payload["seed"])] = np.asarray(effects)
        else:
            rows.extend(row for row in payload_rows if row.get("mode") == "clean")
    if not all_rows:
        raise FileNotFoundError("no eval_metrics.json files found")
    validate_unique_records(all_rows)
    frame = pd.DataFrame(rows)
    group_columns = [
        "model", "dataset", "sweep", "mode", "k", "gamma", "event_step", "event_count"
    ]
    numeric_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "k", "gamma", "event_step", "event_count"}
    ]
    summary = frame.groupby(group_columns, dropna=False)[numeric_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    paired_columns = [
        "seed", "k", "gamma", "event_step", "fast_excess_auc",
        "slow_excess_auc", "slow_minus_fast_auc", "dataset", "sweep", "event_count",
    ]
    paired = pd.DataFrame(paired_effect_rows(rows)).reindex(columns=paired_columns)
    rng = np.random.default_rng(bootstrap_seed)
    seed_effect, boot = _hierarchical_bootstrap(
        sample_effects_by_seed, bootstrap_samples, rng
    )
    statistics = {
        "bootstrap_level": "hierarchical_seed_sample",
        "n_seeds": int(len(seed_effect)),
        "n_sample_event_effects": int(sum(len(v) for v in sample_effects_by_seed.values())),
        "mean_slow_minus_fast_auc": float(seed_effect.mean()) if len(seed_effect) else None,
        "std_slow_minus_fast_auc": (
            float(seed_effect.std(ddof=1)) if len(seed_effect) > 1
            else (0.0 if len(seed_effect) == 1 else None)
        ),
        "bootstrap_ci95": (
            [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if len(boot) else None
        ),
        "exact_sign_flip_p": _exact_sign_flip_p(seed_effect) if len(seed_effect) else None,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_root / "summary.csv",
        "paired_effects": output_root / "paired_effects.csv",
        "all_metrics": output_root / "all_metrics.json",
        "paired_statistics": output_root / "paired_statistics.json",
    }
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired_effects"], index=False)
    paths["all_metrics"].write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    paths["paired_statistics"].write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    curated_representative = output_root / "representative_traces.npz"
    if curated_representative.exists():
        curated_representative.unlink()
    if representative_sources:
        shutil.copy2(sorted(representative_sources)[0], curated_representative)
    return paths
