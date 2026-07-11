from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    dt_ms: int = 5
    pre_ms: int = 50
    cue_ms: int = 100
    input_size: int = 30
    cue_channels: int = 10
    cue_rate_hz: float = 40.0
    noise_rate_hz: float = 10.0
    intervention_fraction: float = 0.10
    intervention_margin: float = 0.05
    train_delays_ms: list[int] = field(default_factory=lambda: [200, 400, 800, 1600])
    eval_delays_ms: list[int] = field(default_factory=lambda: [200, 400, 800, 1600, 2500])
    train_intervention_counts: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    eval_intervention_counts: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])


@dataclass
class ModelConfig:
    hidden_size: int = 64
    recurrent: bool = False
    threshold: float = 1.0
    output_tau_ms: float = 20.0


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 5e-3
    rate_target: float = 0.10
    rate_weight: float = 0.05
    grad_clip: float = 1.0
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    device: str = "auto"


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(TaskConfig(**raw.get("task", {})), ModelConfig(**raw.get("model", {})),
                   TrainConfig(**raw.get("train", {})))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
