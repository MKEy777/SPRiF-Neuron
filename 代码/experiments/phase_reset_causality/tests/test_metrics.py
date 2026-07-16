import math

import torch

from phase_causal.metrics import circular_phase_error, event_metrics


def test_circular_phase_error_wraps_across_pi_boundary():
    target_angle = torch.tensor([math.pi - 0.05])
    predicted_angle = torch.tensor([-math.pi + 0.05])
    prediction = torch.stack((predicted_angle.cos(), predicted_angle.sin()), dim=-1)
    target = torch.stack((target_angle.cos(), target_angle.sin()), dim=-1)

    error = circular_phase_error(prediction, target)

    torch.testing.assert_close(error, torch.tensor([0.10]), atol=1e-5, rtol=0)


def test_event_metrics_compute_jump_auc_and_recovery():
    reference = torch.zeros(1, 30)
    error = reference.clone()
    error[:, 10:15] = torch.tensor([1.0, 0.8, 0.4, 0.05, 0.05])

    result = event_metrics(error, reference, event_step=9, auc_window=5,
                           recovery_window=10, sustain_steps=2)

    torch.testing.assert_close(result["phase_jump"], torch.tensor([0.46]))
    torch.testing.assert_close(result["excess_auc"], torch.tensor([2.30]))
    torch.testing.assert_close(result["recovery_time"], torch.tensor([4.0]))
    assert result["recovery_censored"].tolist() == [False]


def test_phase_jump_uses_five_steps_strictly_before_the_event():
    reference = torch.zeros(1, 30)
    error = reference.clone()
    error[:, 10] = 5.0
    error[:, 11:16] = 1.0

    result = event_metrics(
        error, reference, event_step=10, auc_window=5,
        recovery_window=10, sustain_steps=2,
    )

    torch.testing.assert_close(result["phase_jump"], torch.tensor([1.0]))


def test_recovery_search_starts_after_the_error_peak():
    reference = torch.zeros(1, 30)
    error = reference.clone()
    error[:, 10:15] = torch.tensor([0.01, 0.01, 1.0, 0.05, 0.05])

    result = event_metrics(
        error, reference, event_step=9, auc_window=5,
        recovery_window=10, sustain_steps=2,
    )

    torch.testing.assert_close(result["recovery_time"], torch.tensor([4.0]))
