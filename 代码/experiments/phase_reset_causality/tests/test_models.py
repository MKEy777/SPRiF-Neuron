import torch

from phase_causal.config import ModelConfig, TaskConfig
from phase_causal.models import BRFCell, FilteredSpikeReadout, build_model


def test_filtered_readout_depends_only_on_spikes():
    readout = FilteredSpikeReadout(hidden_size=3, output_size=2, tau_ms=20.0, dt_ms=1.0)
    spikes = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]])

    output_a = readout(spikes)
    output_b = readout(spikes.clone())

    torch.testing.assert_close(output_a, output_b)
    assert output_a.shape == (1, 2, 2)


def test_all_models_produce_shared_output_and_trace_shapes():
    task_cfg = TaskConfig(total_steps=12, cue_steps=4, phase_channels=4)
    model_cfg = ModelConfig(hidden_size=5, threshold=0.3)
    inputs = torch.zeros(2, task_cfg.total_steps, task_cfg.input_size)

    for name in ("sprif", "lif", "asrnn", "brf"):
        model = build_model(name, task_cfg, model_cfg)
        outputs, trace = model(inputs, return_trace=True)
        assert outputs.shape == (2, 12, 2)
        assert trace["spikes"].shape == (2, 12, 5)
        assert trace["membrane_pre"].shape == (2, 12, 5)
        assert torch.isfinite(outputs).all()


def test_sprif_intervention_branches_share_the_forced_spike_event():
    task_cfg = TaskConfig(total_steps=10, cue_steps=3, phase_channels=4)
    model_cfg = ModelConfig(hidden_size=4, threshold=0.3)
    model = build_model("sprif", task_cfg, model_cfg)
    inputs = torch.zeros(2, task_cfg.total_steps, task_cfg.input_size)
    mask = torch.tensor([[True, True, False, False], [False, True, True, False]])
    masks = {5: mask}

    event_spikes = []
    for mode in ("forced_no_reset", "fast_reset", "slow_reset", "both_reset"):
        _, trace = model(
            inputs,
            mode=mode,
            intervention_masks=masks,
            margin=0.05,
            return_trace=True,
        )
        event_spikes.append(trace["spikes"][:, 5])

    for spikes in event_spikes[1:]:
        torch.testing.assert_close(spikes, event_spikes[0])


def test_brf_uses_sidms_parameter_initialization_and_zero_state():
    torch.manual_seed(17)
    cfg = ModelConfig(hidden_size=8, threshold=0.3)
    cell = BRFCell(input_size=3, cfg=cfg, dt_ms=1.0)

    torch.testing.assert_close(cell.omega, torch.full((8,), 10.0))
    assert torch.all(cell.b_offset >= 0.5)
    assert torch.all(cell.b_offset <= 3.0)
    state = cell.initial_state(batch=2, device=torch.device("cpu"), dtype=torch.float32)
    assert set(state) == {"u", "v", "q"}
    assert all(torch.count_nonzero(value) == 0 for value in state.values())


def test_brf_clean_step_matches_sidms_update_equations():
    cfg = ModelConfig(hidden_size=2, threshold=100.0)
    cell = BRFCell(input_size=1, cfg=cfg, dt_ms=1.0)
    with torch.no_grad():
        cell.omega.copy_(torch.tensor([10.0, 25.0]))
        cell.b_offset.copy_(torch.tensor([0.5, 2.0]))
        cell.input.weight.zero_()
        cell.input.bias.zero_()

    state = {
        "u": torch.tensor([[0.2, -0.3]]),
        "v": torch.tensor([[-0.1, 0.4]]),
        "q": torch.tensor([[0.05, 0.2]]),
    }
    dt = 0.001
    omega = cell.omega.abs().clamp(max=0.99 / dt)
    p = (-1.0 + torch.sqrt(1.0 - (dt * omega) ** 2)) / dt
    damping = p - cell.b_offset.abs() - state["q"]
    expected_u = state["u"] + (
        damping * state["u"] - omega * state["v"]
    ) * dt
    expected_v = state["v"] + (
        omega * state["u"] + damping * state["v"]
    ) * dt

    spike, next_state, _ = cell.step(
        torch.zeros(1, 1), state, mode="clean",
        mask=torch.zeros(1, 2, dtype=torch.bool), gamma=1.0, margin=0.05,
    )

    torch.testing.assert_close(spike, torch.zeros_like(spike))
    torch.testing.assert_close(next_state["u"], expected_u)
    torch.testing.assert_close(next_state["v"], expected_v)
    torch.testing.assert_close(next_state["q"], 0.9 * state["q"])
