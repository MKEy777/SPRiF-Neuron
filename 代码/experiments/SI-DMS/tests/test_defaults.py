from pathlib import Path

import pytest

from sidms.config import ExperimentConfig
from sidms.reporting import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_publication_protocol():
    cfg = ExperimentConfig.from_yaml(ROOT / "config" / "default.yaml")
    matched = ExperimentConfig.from_yaml(ROOT / "config" / "default_matched.yaml")
    assert cfg.to_dict() == matched.to_dict()
    assert cfg.model.recurrent is True
    assert cfg.train.steps == 3000
    assert cfg.train.batch_size == 256
    assert cfg.train.learning_rate == 3e-3
    assert cfg.task.train_intervention_fraction == 0.15
    assert cfg.task.train_intervention_counts == [0, 1, 2, 4, 8]
    assert cfg.task.eval_intervention_counts == [0, 1, 2, 4, 8, 16, 32, 40]


def test_summary_uses_publication_fraction():
    rows = []
    for seed in [1, 2, 3]:
        rows.extend([
            {"model": "sprif_full", "seed": seed, "delay_ms": 1600,
             "intervention_count": 0, "intervention_fraction": 0.0,
             "accuracy": 1.0},
            {"model": "sprif_full", "seed": seed, "delay_ms": 1600,
             "intervention_count": 40, "intervention_fraction": 0.15,
             "accuracy": 0.9},
            {"model": "sprif_full", "seed": seed, "delay_ms": 1600,
             "intervention_count": 40, "intervention_fraction": 0.5,
             "accuracy": 0.1},
        ])
    result = summarize(rows)[0]
    assert result["max_stress_accuracy"] == 0.9
    assert result["abs_drop"] == pytest.approx(0.1)
