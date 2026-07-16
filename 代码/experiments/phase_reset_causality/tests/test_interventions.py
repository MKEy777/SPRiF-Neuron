import torch

from phase_causal.interventions import (
    apply_sprif_reset,
    force_threshold_crossing,
    matched_slow_reset,
)


def test_force_threshold_crossing_never_lowers_membrane():
    membrane = torch.tensor([[0.20, 1.20, 1.80]])
    mask = torch.tensor([[True, True, False]])

    forced, forced_hit, new_crossing = force_threshold_crossing(
        membrane, threshold=1.0, mask=mask, margin=0.05
    )

    torch.testing.assert_close(forced, torch.tensor([[1.05, 1.20, 1.80]]))
    assert torch.all(forced >= membrane)
    assert forced_hit.tolist() == [[True, True, False]]
    assert new_crossing.tolist() == [[True, False, False]]


def test_matched_slow_reset_has_same_norm_as_fast_reset():
    g = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    fast_delta = torch.tensor([[1.0, 2.0]])

    slow_delta = matched_slow_reset(g, fast_delta)

    torch.testing.assert_close(
        torch.linalg.vector_norm(slow_delta, dim=-1),
        torch.linalg.vector_norm(fast_delta, dim=-1),
    )


def test_matched_slow_reset_keeps_norm_when_projection_is_degenerate():
    g = torch.zeros(1, 2, 3)
    fast_delta = torch.tensor([[1.0, 2.0]])

    slow_delta = matched_slow_reset(g, fast_delta)

    torch.testing.assert_close(
        torch.linalg.vector_norm(slow_delta, dim=-1),
        torch.linalg.vector_norm(fast_delta, dim=-1),
    )


def test_reset_modes_change_only_the_routed_state():
    slow = torch.zeros(1, 1, 3)
    fast = torch.tensor([[[1.1, 0.4]]])
    spike = torch.ones(1, 1)
    reset_direction = torch.tensor([[1.0, 0.5]])
    g = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    slow_fast, fast_fast = apply_sprif_reset(
        "fast_reset", slow, fast, spike, reset_direction, g, threshold=1.0
    )
    slow_slow, fast_slow = apply_sprif_reset(
        "slow_reset", slow, fast, spike, reset_direction, g, threshold=1.0
    )

    torch.testing.assert_close(slow_fast, slow)
    assert not torch.equal(fast_fast, fast)
    assert not torch.equal(slow_slow, slow)
    torch.testing.assert_close(fast_slow, fast)
