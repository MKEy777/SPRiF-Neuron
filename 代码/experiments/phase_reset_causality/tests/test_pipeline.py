from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from phase_causal.config import load_config
from phase_causal.pipeline import (
    _save_representative,
    aggregate_results,
    evaluate_checkpoint,
    train_model_seed,
)
from phase_causal.plotting import plot_results


def test_smoke_config_trains_evaluates_and_aggregates(tmp_path):
    experiment_root = Path(__file__).resolve().parents[1]
    cfg = load_config(experiment_root / "config" / "smoke.yaml")
    device = torch.device("cpu")

    checkpoint = train_model_seed("sprif", 1, cfg, tmp_path, device)
    assert checkpoint.exists()
    assert (tmp_path / "raw" / "sprif" / "seed_1" / "train_history.json").exists()

    metrics_path = evaluate_checkpoint(
        "sprif", 1, cfg, tmp_path, device, run_sensitivity=True
    )
    assert metrics_path.exists()
    payload = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))
    assert {"multi_event", "id_frequency", "ood_frequency"}.issubset(
        {row["sweep"] for row in payload["rows"]}
    )
    assert payload["sample_effects"]

    outputs = aggregate_results(tmp_path, bootstrap_samples=50, bootstrap_seed=2)
    assert outputs["summary"].exists()
    assert outputs["paired_effects"].exists()
    assert (tmp_path / "representative_traces.npz").exists() == payload["clean_gate_passed"]
    statistics = __import__("json").loads(outputs["paired_statistics"].read_text(encoding="utf-8"))
    assert statistics["bootstrap_level"] == "hierarchical_seed_sample"
    summary = pd.read_csv(outputs["summary"])
    if payload["clean_gate_passed"]:
        assert {"fast_reset", "slow_reset"}.issubset(set(summary["mode"]))
    else:
        assert set(summary["mode"]) == {"clean"}

    figures = plot_results(tmp_path)
    assert {"main", "baselines", "sensitivity"} == set(figures)
    assert all(path.exists() for path in figures.values())


def _metric_row(seed, mode, excess_auc):
    return {
        "model": "sprif", "seed": seed, "dataset": "in_distribution",
        "sweep": "main", "mode": mode, "k": 8, "gamma": 1.0,
        "event_step": 200, "event_count": 1, "valid_samples": 2,
        "invalid_samples": 0, "delay_mse": 0.1, "excess_auc": excess_auc,
    }


def _write_payload(root, seed, clean_gate_passed, effects):
    run_dir = root / "raw" / "sprif" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    rows = [
        _metric_row(seed, "clean", 0.0),
        _metric_row(seed, "fast_reset", 1.0),
        _metric_row(seed, "slow_reset", 2.0),
    ]
    payload = {
        "model": "sprif", "seed": seed,
        "clean_gate_passed": clean_gate_passed,
        "rows": rows,
        "sample_effects": [
            {"sample_index": index, "event_step": 200,
             "slow_minus_fast_auc": value}
            for index, value in enumerate(effects)
        ],
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_uses_hierarchical_seed_sample_bootstrap(tmp_path):
    _write_payload(tmp_path, 1, True, [1.0, 2.0])
    _write_payload(tmp_path, 2, True, [3.0, 4.0])

    outputs = aggregate_results(tmp_path, bootstrap_samples=100, bootstrap_seed=7)
    statistics = json.loads(outputs["paired_statistics"].read_text(encoding="utf-8"))

    assert statistics["bootstrap_level"] == "hierarchical_seed_sample"
    assert statistics["n_seeds"] == 2
    assert statistics["n_sample_event_effects"] == 4
    assert statistics["mean_slow_minus_fast_auc"] == 2.5


def test_failed_clean_gate_keeps_only_clean_rows_in_summary(tmp_path):
    _write_payload(tmp_path, 1, False, [1.0, 2.0])
    raw_representative = tmp_path / "raw" / "sprif" / "seed_1" / "representative_traces.npz"
    np.savez(raw_representative, target=np.zeros((2, 2)))
    np.savez(tmp_path / "representative_traces.npz", target=np.ones((2, 2)))

    outputs = aggregate_results(tmp_path, bootstrap_samples=20, bootstrap_seed=3)
    summary = pd.read_csv(outputs["summary"])
    statistics = json.loads(outputs["paired_statistics"].read_text(encoding="utf-8"))

    assert set(summary["mode"]) == {"clean"}
    assert statistics["n_seeds"] == 0
    assert not (tmp_path / "representative_traces.npz").exists()


def test_passed_clean_gate_promotes_representative_trace(tmp_path):
    _write_payload(tmp_path, 1, True, [1.0, 2.0])
    raw_representative = tmp_path / "raw" / "sprif" / "seed_1" / "representative_traces.npz"
    np.savez(raw_representative, marker=np.asarray(7))

    aggregate_results(tmp_path, bootstrap_samples=20, bootstrap_seed=3)

    promoted = np.load(tmp_path / "representative_traces.npz")
    assert promoted["marker"].item() == 7


def test_representative_trace_records_intervention_integrity(tmp_path):
    from phase_causal.config import EvaluationConfig, ModelConfig, TaskConfig
    from phase_causal.engine import evaluate_paired_batch
    from phase_causal.models import build_model

    task = TaskConfig(total_steps=16, cue_steps=4, phase_channels=4)
    model = build_model("sprif", task, ModelConfig(hidden_size=4, threshold=0.3))
    inputs = torch.zeros(2, 16, task.input_size)
    t = torch.arange(16, dtype=torch.float32)
    target = torch.stack((torch.cos(0.1 * t), torch.sin(0.1 * t)), dim=-1)
    target = target.unsqueeze(0).repeat(2, 1, 1)
    _, artifacts = evaluate_paired_batch(
        model, inputs, target, event_step=8, k=2, gamma=1.0, margin=0.05,
        mask_seed=3, cue_steps=4,
        evaluation_cfg=EvaluationConfig(auc_window=3, recovery_window=5,
                                         recovery_sustain=2),
    )

    path = tmp_path / "representative.npz"
    _save_representative(path, artifacts, 8)
    data = np.load(path)

    assert data["forced_event_spikes_identical"].item()
    assert data["fast_reset_residual_norm"].item() > 0
    assert data["slow_reset_residual_norm"].item() > 0
    assert "failure_slow_reset_output" in data
    assert "phase_failure_sample_index" in data
    assert "radius_collapse_sample_index" in data
    assert "repeat_firing_sample_index" in data
