import torch

from sidms.config import ExperimentConfig
from sidms.data import make_batch


def test_batch_encodes_match_labels_and_has_expected_timing():
    cfg = ExperimentConfig()
    batch = make_batch(cfg, batch_size=64, delay_ms=200, intervention_count=2,
                       hidden_size=20, seed=7)
    expected_t = (cfg.task.pre_ms + 2 * cfg.task.cue_ms + 200) // cfg.task.dt_ms
    assert batch.x.shape == (64, expected_t, 30)
    assert batch.y.shape == (64,)
    assert torch.equal(batch.y, (batch.first_side == batch.second_side).long())
    assert batch.intervention.shape == (64, expected_t, 20)


def test_interventions_are_inside_delay_and_label_independent():
    cfg = ExperimentConfig()
    batch = make_batch(cfg, batch_size=256, delay_ms=400, intervention_count=4,
                       hidden_size=40, seed=11)
    delay_start = (cfg.task.pre_ms + cfg.task.cue_ms) // cfg.task.dt_ms
    delay_end = delay_start + 400 // cfg.task.dt_ms
    assert not batch.intervention[:, :delay_start].any()
    assert not batch.intervention[:, delay_end:].any()
    active_times = batch.intervention.any(-1).sum(-1)
    assert torch.equal(active_times, torch.full_like(active_times, 4))
    per_event = batch.intervention.sum(-1)[batch.intervention.any(-1)]
    assert torch.equal(per_event, torch.full_like(per_event, 4))
    # Same RNG seed and task settings must produce the same intervention mask,
    # regardless of subsequently changing labels.
    clone = make_batch(cfg, 256, 400, 4, 40, seed=11)
    clone.y = 1 - clone.y
    assert torch.equal(batch.intervention, clone.intervention)


def test_zero_intervention_count_produces_empty_mask():
    cfg = ExperimentConfig()
    batch = make_batch(cfg, 8, 200, 0, 16, seed=3)
    assert batch.intervention.sum().item() == 0
