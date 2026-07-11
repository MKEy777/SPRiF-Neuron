import torch

from sidms.cells import build_cell


def _step(name, intervention=None):
    torch.manual_seed(2)
    cell = build_cell(name, input_size=5, hidden_size=8, dt_ms=5, threshold=1.0)
    state = cell.initial_state(3, torch.device("cpu"))
    x = torch.randn(3, 5) * 0.2
    if intervention is None:
        intervention = torch.zeros(3, 8, dtype=torch.bool)
    return cell, state, cell.step(x, state, intervention)


def test_controlled_intervention_forces_exact_selected_spikes():
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[:, [1, 6]] = True
    for name in ["sprif_full", "sprif_merged", "sprif_lambda0", "lif", "asrnn", "brf"]:
        _, _, (spike, _, diag) = _step(name, mask)
        assert spike[mask].eq(1).all(), name
        assert diag["forced_hit"][mask].all(), name


def test_full_projective_reset_changes_both_fast_coordinates():
    cell, state, (_, new, diag) = _step("sprif_full", torch.ones(3, 8, dtype=torch.bool))
    assert torch.allclose(new["slow"], diag["slow_pre_reset"])
    reset = diag["fast_pre_reset"] - new["fast"]
    assert torch.allclose(reset[..., 0], torch.ones_like(reset[..., 0]), atol=1e-5)
    assert reset[..., 1].abs().sum() > 0
    assert cell.reset_lambda.requires_grad


def test_lambda0_is_scalar_reset_with_preserved_slow_state():
    cell, _, (_, new, diag) = _step("sprif_lambda0", torch.ones(3, 8, dtype=torch.bool))
    reset = diag["fast_pre_reset"] - new["fast"]
    assert torch.allclose(new["slow"], diag["slow_pre_reset"])
    assert torch.allclose(reset[..., 0], torch.ones_like(reset[..., 0]), atol=1e-5)
    assert torch.allclose(reset[..., 1], torch.zeros_like(reset[..., 1]), atol=1e-6)
    assert not hasattr(cell, "reset_lambda")


def test_merged_has_one_state_and_resets_only_membrane_coordinate():
    _, _, (_, new, diag) = _step("sprif_merged", torch.ones(3, 8, dtype=torch.bool))
    assert set(new) == {"merged"}
    reset = diag["merged_pre_reset"] - new["merged"]
    assert torch.allclose(reset[..., 0], torch.ones_like(reset[..., 0]), atol=1e-5)
    assert torch.allclose(reset[..., 1:], torch.zeros_like(reset[..., 1:]), atol=1e-6)
