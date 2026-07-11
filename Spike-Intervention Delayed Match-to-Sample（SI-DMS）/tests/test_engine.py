import json

import torch

from sidms.config import ExperimentConfig
from sidms.data import make_batch
from sidms.engine import evaluate_grid, train_model
from sidms.models import SIDMSNetwork


def test_network_forward_returns_logits_and_intervention_metrics():
    cfg = ExperimentConfig()
    cfg.model.hidden_size = 12
    batch = make_batch(cfg, 5, 200, 2, 12, seed=1)
    model = SIDMSNetwork("sprif_full", cfg)
    out = model(batch.x, batch.intervention)
    assert out.logits.shape == (5, 2)
    assert 0 <= out.natural_rate <= 1
    assert out.forced_hit_rate == 1.0


def test_train_checkpoint_and_grid_evaluation_are_reproducible(tmp_path):
    cfg = ExperimentConfig()
    cfg.model.hidden_size = 8
    cfg.train.steps = 2
    cfg.train.batch_size = 4
    cfg.task.train_delays_ms = [50]
    cfg.task.eval_delays_ms = [50, 100]
    cfg.task.train_intervention_counts = [0, 1]
    cfg.task.eval_intervention_counts = [0, 1]
    model, history = train_model("sprif_lambda0", cfg, seed=4, device="cpu")
    assert len(history) == 2
    rows1 = evaluate_grid(model, cfg, seed=9, batches=1, batch_size=4, device="cpu")
    rows2 = evaluate_grid(model, cfg, seed=9, batches=1, batch_size=4, device="cpu")
    assert rows1 == rows2
    assert len(rows1) == 4
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(rows1))
    assert json.loads(path.read_text()) == rows1


def test_training_progress_callback_receives_each_step():
    cfg = ExperimentConfig(); cfg.model.hidden_size = 4; cfg.train.steps = 3; cfg.train.batch_size = 2
    cfg.task.train_delays_ms = [20]; cfg.task.train_intervention_counts = [0]
    seen = []
    train_model("lif", cfg, seed=1, device="cpu", progress=lambda row: seen.append(row["step"]))
    assert seen == [1, 2, 3]


def test_evaluation_progress_callback_receives_each_grid_cell():
    cfg = ExperimentConfig(); cfg.model.hidden_size = 4
    cfg.task.eval_delays_ms = [20, 40]; cfg.task.eval_intervention_counts = [0, 1]
    model = SIDMSNetwork("lif", cfg); seen = []
    evaluate_grid(model, cfg, seed=1, batches=1, batch_size=2, device="cpu",
                  progress=lambda row: seen.append((row["delay_ms"], row["intervention_count"])))
    assert seen == [(20, 0), (20, 1), (40, 0), (40, 1)]
