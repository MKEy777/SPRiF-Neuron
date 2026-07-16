import torch

from phase_causal.config import EvaluationConfig, ModelConfig, TaskConfig
from phase_causal.engine import (
    evaluate_multi_event_batch,
    evaluate_paired_batch,
    select_eligible_mask,
    validate_invalid_rate,
)
from phase_causal.models import build_model


def test_select_eligible_mask_uses_only_subthreshold_units():
    membrane = torch.tensor([[0.10, 0.50, 0.20], [0.40, 0.10, 0.50]])

    mask, valid = select_eligible_mask(membrane, threshold=0.3, k=2, seed=9)

    assert mask[0].tolist() == [True, False, True]
    assert valid.tolist() == [True, False]
    assert torch.all(membrane[mask] < 0.3)


def test_invalid_trial_rate_above_one_percent_is_rejected():
    valid = torch.tensor([True] * 98 + [False] * 2)

    try:
        validate_invalid_rate(valid, maximum_invalid_rate=0.01)
    except RuntimeError as exc:
        assert "invalid" in str(exc).lower()
    else:
        raise AssertionError("an invalid rate above 1% must be rejected")


def test_paired_batch_reports_all_sprif_modes_with_identical_hits():
    task_cfg = TaskConfig(total_steps=16, cue_steps=4, phase_channels=4)
    model_cfg = ModelConfig(hidden_size=4, threshold=0.3)
    eval_cfg = EvaluationConfig(auc_window=3, recovery_window=5, recovery_sustain=2)
    model = build_model("sprif", task_cfg, model_cfg)
    inputs = torch.zeros(2, 16, task_cfg.input_size)
    t = torch.arange(16, dtype=torch.float32)
    target = torch.stack((torch.cos(0.1 * t), torch.sin(0.1 * t)), dim=-1)
    target = target.unsqueeze(0).repeat(2, 1, 1)

    rows, artifacts = evaluate_paired_batch(
        model=model,
        inputs=inputs,
        target=target,
        event_step=8,
        k=2,
        gamma=1.0,
        margin=0.05,
        mask_seed=3,
        cue_steps=task_cfg.cue_steps,
        evaluation_cfg=eval_cfg,
    )

    assert {row["mode"] for row in rows} == {
        "clean", "forced_no_reset", "fast_reset", "slow_reset", "both_reset"
    }
    forced_rows = [row for row in rows if row["mode"] != "clean"]
    assert all(row["forced_hit_rate"] == 1.0 for row in forced_rows)
    assert all(row["new_crossing_rate"] == 1.0 for row in forced_rows)
    assert all(row["natural_spike_overlap_rate"] == 0.0 for row in forced_rows)
    assert all(row["valid_samples"] == 2 for row in forced_rows)
    assert artifacts["sample_metrics"]["slow_reset"]["excess_auc"].shape == (2,)
    reference = artifacts["traces"]["forced_no_reset"]
    for mode in ("fast_reset", "slow_reset", "both_reset"):
        trace = artifacts["traces"][mode]
        torch.testing.assert_close(trace["spikes"][:, :8], reference["spikes"][:, :8])
        torch.testing.assert_close(trace["slow"][:, :8], reference["slow"][:, :8])
        torch.testing.assert_close(trace["fast"][:, :8], reference["fast"][:, :8])


def test_multi_event_batch_reports_accumulated_reset_conditions():
    task_cfg = TaskConfig(total_steps=18, cue_steps=4, phase_channels=4)
    model_cfg = ModelConfig(hidden_size=4, threshold=0.3)
    eval_cfg = EvaluationConfig(auc_window=3, recovery_window=4, recovery_sustain=2)
    model = build_model("sprif", task_cfg, model_cfg)
    inputs = torch.zeros(2, 18, task_cfg.input_size)
    t = torch.arange(18, dtype=torch.float32)
    target = torch.stack((torch.cos(0.1 * t), torch.sin(0.1 * t)), dim=-1)
    target = target.unsqueeze(0).repeat(2, 1, 1)

    rows, artifacts = evaluate_multi_event_batch(
        model, inputs, target, event_steps=(7, 12), k=2, gamma=1.0,
        margin=0.05, mask_seed=4, cue_steps=task_cfg.cue_steps,
        evaluation_cfg=eval_cfg,
    )

    assert {row["mode"] for row in rows} == {
        "clean", "forced_no_reset", "fast_reset", "slow_reset", "both_reset"
    }
    assert all(row["event_count"] == 2 for row in rows)
    assert set(artifacts["masks"]) == {7, 12}
    for step, mask in artifacts["masks"].items():
        reference_spikes = artifacts["traces"]["forced_no_reset"]["spikes"][:, step]
        for mode in ("forced_no_reset", "fast_reset", "slow_reset", "both_reset"):
            selected_crossings = artifacts["traces"][mode]["new_crossing"][:, step][mask]
            selected_spikes = artifacts["traces"][mode]["spikes"][:, step][mask]
            assert torch.all(selected_crossings)
            assert torch.all(selected_spikes == 1)
            torch.testing.assert_close(
                artifacts["traces"][mode]["spikes"][:, step], reference_spikes
            )
    forced_rows = [row for row in rows if row["mode"] != "clean"]
    assert all(row["natural_spike_overlap_rate"] == 0.0 for row in forced_rows)
