from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskConfig:
    total_steps: int = 900
    cue_steps: int = 100
    phase_channels: int = 20
    base_rate_hz: float = 30.0
    modulation_hz: float = 25.0
    dt_ms: float = 1.0
    omega_periods: tuple[int, ...] = (50, 100, 200)

    @property
    def input_size(self) -> int:
        return self.phase_channels + 2

    @property
    def omega_choices(self) -> tuple[float, ...]:
        import math

        return tuple(2.0 * math.pi / period for period in self.omega_periods)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 64
    threshold: float = 0.3
    recurrent: bool = False
    readout_tau_ms: float = 20.0


@dataclass(frozen=True)
class TrainingConfig:
    train_samples: int = 10_000
    val_samples: int = 2_000
    test_samples: int = 2_000
    batch_size: int = 256
    epochs: int = 150
    patience: int = 15
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    lr_step_size: int = 20
    lr_gamma: float = 0.5


@dataclass(frozen=True)
class InterventionConfig:
    k: int = 8
    gamma: float = 1.0
    margin: float = 0.05
    event_steps: tuple[int, ...] = (200, 350, 500, 650, 800)
    k_values: tuple[int, ...] = (1, 2, 4, 8, 16)
    gamma_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    multi_event_counts: tuple[int, ...] = (0, 1, 2, 4, 8)


@dataclass(frozen=True)
class EvaluationConfig:
    auc_window: int = 50
    recovery_window: int = 100
    recovery_sustain: int = 10
    clean_mse_max: float = 0.25
    clean_radius_min: float = 0.5
    bootstrap_samples: int = 10_000
    test_seed: int = 2027
    ood_periods: tuple[int, ...] = (75, 150)


@dataclass(frozen=True)
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sprif_seeds: tuple[int, ...] = tuple(range(1, 11))
    baseline_seeds: tuple[int, ...] = tuple(range(1, 6))
    models: tuple[str, ...] = ("sprif", "lif", "asrnn", "brf")
    output_dir: str = "../../../experiment-design-20260606/results/phase_reset_causality"


def _build_dataclass(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(
        task=_build_dataclass(TaskConfig, raw.get("task")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        intervention=_build_dataclass(InterventionConfig, raw.get("intervention")),
        evaluation=_build_dataclass(EvaluationConfig, raw.get("evaluation")),
        sprif_seeds=tuple(raw.get("sprif_seeds", range(1, 11))),
        baseline_seeds=tuple(raw.get("baseline_seeds", range(1, 6))),
        models=tuple(raw.get("models", ("sprif", "lif", "asrnn", "brf"))),
        output_dir=raw.get("output_dir", ExperimentConfig.output_dir),
    )
