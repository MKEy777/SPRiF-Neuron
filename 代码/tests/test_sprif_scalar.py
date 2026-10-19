"""Behavior checks for the TNNLS scalar-reset, absorbed-readout neuron.

These checks use synthetic tensors; they do not generate benchmark data or train.
"""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = ["Task_S-MNIST", "Task_pSMNIST", "Task_ECG", "Task_GSC", "Task_SHD"]
SIDMS = ROOT / "experiments" / "SI-DMS"
sys.path.insert(0, str(SIDMS))


def load_file(path):
    spec = importlib.util.spec_from_file_location("test_source_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layer_type(task, filename="sprif_layer.py"):
    return load_file(ROOT / task / "core_algorithm" / filename).SPRiFNeuronLayer


def local_count(layer):
    return sum(p.numel() for name, p in layer.named_parameters()
               if "." not in name)


def require_scalar(layer):
    # An accidental extra reset/coupling coefficient changes the scientific model.
    assert local_count(layer) == 11 * layer.hidden_size
    state = layer.init_state(1)
    assert "h" in state, "The auxiliary filter must be a separate scalar state."
    assert state["u"].shape == (1, layer.hidden_size)


def set_hand_parameters(layer):
    with torch.no_grad():
        layer.alpha_raw.fill_(0)  # alpha = 1/2
        layer.rho_raw.fill_(0)  # rho = 1/2
        layer.omega_raw.fill_(0)  # omega = pi/2
        layer.eta_raw[:, 0].fill_(0)  # eta_0 = 1/2
        layer.eta_raw[:, 1].fill_(torch.logit(torch.tensor(0.25)))
        layer.bar_G.copy_(torch.tensor([[[1., 0., 0.], [0., 2., 0.]]]))


@pytest.mark.parametrize("task", TASKS)
def test_one_step_uses_previous_filter_and_resets_only_membrane(task):
    layer = layer_type(task)(1, 1)
    require_scalar(layer)
    set_hand_parameters(layer)
    state = {"x": torch.tensor([[[.2, .4, .6]]]), "h": torch.tensor([[.7]]),
             "u": torch.tensor([[.4]]), "prev_spike": torch.zeros(1, 1)}
    spike, membrane, after = layer.forward_step(
        torch.zeros(1, 1), state, layer._precompute_runtime_params(),
        input_current=torch.tensor([[2.]]))
    # Hand calculation: x=(1.1,.7,.2), h=.25*.7+2*.7=1.575,
    # membrane=.5*.4+.7+1.1=2; scalar reset leaves membrane=1.
    torch.testing.assert_close(after["x"], torch.tensor([[[1.1, .7, .2]]]))
    torch.testing.assert_close(after["h"], torch.tensor([[1.575]]))
    torch.testing.assert_close(membrane, torch.tensor([[2.]]))
    torch.testing.assert_close(spike, torch.ones(1, 1))
    torch.testing.assert_close(after["u"], torch.ones(1, 1))


@pytest.mark.parametrize("task", TASKS)
def test_recurrent_current_uses_previous_spike(task):
    layer = layer_type(task)(1, 1, recurrent=True, bias=True)
    require_scalar(layer)
    set_hand_parameters(layer)
    with torch.no_grad():
        layer.input_linear.weight.fill_(2)
        layer.input_linear.bias.fill_(.25)
        layer.recurrent_linear.weight.fill_(3)
    state = layer.init_state(1)
    state["prev_spike"].fill_(1)
    _, membrane, after = layer.forward_step(
        torch.tensor([[.5]]), state, layer._precompute_runtime_params())
    # I=2*.5+.25+3*1=4.25, x_0=.5*I=2.125.
    torch.testing.assert_close(membrane, torch.tensor([[2.125]]))
    torch.testing.assert_close(after["u"], torch.tensor([[1.125]]))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("coupling", [0., -.75, .5])
def test_absorption_preserves_fixed_parameter_scalar_reset_trajectory(task, coupling):
    torch.manual_seed(31)
    new = layer_type(task)(2, 3, recurrent=True, threshold=.4)
    require_scalar(new)
    # SHD has no archived scalar-reset file; the GSC cell has the same updates.
    reference_task = "Task_GSC" if task == "Task_SHD" else task
    reference = load_file(ROOT / reference_task / "core_algorithm" /
                          "sprif_layer_ablation_c.py").SPRiFNeuronLayerAblationC
    old = reference(2, 3, recurrent=True, threshold=.4)
    with torch.no_grad():
        old.fast_coupling.fill_(coupling)
        old.G.normal_(std=.8)
        for name in ("alpha_raw", "rho_raw", "omega_raw", "eta_raw"):
            getattr(new, name).copy_(getattr(old, name))
        new.input_linear.load_state_dict(old.input_linear.state_dict())
        new.recurrent_linear.load_state_dict(old.recurrent_linear.state_dict())
        new.bar_G[:, 0].copy_(old.G[:, 0])
        new.bar_G[:, 1].copy_(coupling * old.G[:, 1])
    old_state = old.init_state(2)
    old_state["x"] = torch.randn(2, 3, 3)
    old_state["u"] = torch.randn(2, 3, 2)
    new_state = {"x": old_state["x"].clone(),
                 "u": old_state["u"][..., 0].clone(),
                 "h": coupling * old_state["u"][..., 1],
                 "prev_spike": old_state["prev_spike"].clone()}
    old_runtime = old._precompute_runtime_params()
    new_runtime = new._precompute_runtime_params()
    for sample in torch.randn(12, 2, 2):
        z_old, v_old, old_state = old.forward_step(sample, old_state, old_runtime)
        z_new, v_new, new_state = new.forward_step(sample, new_state, new_runtime)
        torch.testing.assert_close(z_new, z_old)
        torch.testing.assert_close(v_new, v_old)
        torch.testing.assert_close(new_state["x"], old_state["x"])
        torch.testing.assert_close(new_state["h"], coupling * old_state["u"][..., 1])
        torch.testing.assert_close(new_state["u"], old_state["u"][..., 0])


@pytest.mark.parametrize("task", TASKS)
def test_auxiliary_readout_gradient_travels_through_one_step_delay(task):
    layer = layer_type(task)(1, 1)
    require_scalar(layer)
    set_hand_parameters(layer)
    with torch.no_grad():
        layer.bar_G[:, 0].zero_()
    runtime = layer._precompute_runtime_params()
    state = layer.init_state(1)
    _, _, state = layer.forward_step(torch.zeros(1, 1), state, runtime,
                                    input_current=torch.tensor([[2.]]))
    _, membrane, _ = layer.forward_step(torch.zeros(1, 1), state, runtime,
                                       input_current=torch.zeros(1, 1))
    membrane.sum().backward()
    # The delayed path reads x at the previous step: (1,1,0).
    torch.testing.assert_close(layer.bar_G.grad[:, 1], torch.tensor([[1., 1., 0.]]))


@pytest.mark.parametrize("task", TASKS)
def test_threshold_equality_follows_manuscript_heaviside_definition(task):
    layer = layer_type(task)(1, 1)
    require_scalar(layer)
    with torch.no_grad():
        layer.bar_G.zero_()
    state = layer.init_state(1)
    state["h"].fill_(1.)
    spike, membrane, after = layer.forward_step(
        torch.zeros(1, 1), state, layer._precompute_runtime_params(),
        input_current=torch.zeros(1, 1))
    torch.testing.assert_close(membrane, torch.ones(1, 1))
    torch.testing.assert_close(spike, torch.ones(1, 1))
    torch.testing.assert_close(after["u"], torch.zeros(1, 1))


@pytest.mark.parametrize("task", TASKS)
def test_new_sequences_start_at_zero_and_repeat_in_evaluation(task):
    layer = layer_type(task)(2, 3, recurrent=True).double().eval()
    require_scalar(layer)
    state = layer.init_state(2)
    assert state["x"].shape == (2, 3, 3)
    for value in state.values():
        assert value.dtype == torch.float64
        assert torch.count_nonzero(value).item() == 0
    inputs = torch.ones(2, 8, 2, dtype=torch.float64)
    torch.testing.assert_close(layer(inputs, batch_first=True),
                               layer(inputs, batch_first=True), rtol=0, atol=0)


@contextmanager
def task_imports(task):
    names = [n for n in sys.modules if n == "core_algorithm" or n.startswith("core_algorithm.")]
    saved = {n: sys.modules.pop(n) for n in names}
    sys.path.insert(0, str(ROOT / task))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "core_algorithm" or name.startswith("core_algorithm."):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path.remove(str(ROOT / task))


@pytest.mark.parametrize("task", TASKS)
def test_task_readout_and_backward_accept_new_state_layout(task):
    with task_imports(task):
        module = load_file(ROOT / task / "model.py")
        if task == "Task_S-MNIST":
            model = module.SPRiFSMNISTNet(1, [3, 4], num_classes=2, mode="srnn")
            features = 1
        elif task == "Task_pSMNIST":
            model = module.SPRiFpSMNISTNet(1, [3, 4], num_classes=2, mode="srnn")
            features = 1
        elif task == "Task_ECG":
            model = module.SPRiFECGModel(4, [3], 2, neuron_kwargs={"bias": True})
            features = 4
        elif task == "Task_GSC":
            model = module.SPRiFGSCNet(4, [3], 2, 0., (True,), {})
            features = 4
        else:
            model = module.SPRiFSHDNet(4, [3], 2, 0., (True,), 2, {})
            features = 4
        for layer in model.layers:
            require_scalar(layer)
        if task == "Task_SHD":
            require_scalar(model.readout)
        result = model(torch.rand(2, 6, features))
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == ((2, 2, 6) if task == "Task_ECG" else (2, 2))
        logits.sum().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        if task == "Task_ECG":
            stats = module._collect_internal_stats(model)
            assert "bar_G" in stats["global"]
            assert "lambda_reset" not in stats["global"]


def test_shd_directory_reader_keeps_existing_train_and_test_membership(tmp_path):
    dataset = load_file(ROOT / "Task_SHD" / "data.py").SHDDataset
    assert hasattr(dataset, "from_directory"), "Training must read the specified split directory."
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    np.save(train / "ID_1_3.npy", np.ones((2, 4)))
    np.save(train / "ID_0_2.npy", np.zeros((2, 4)))
    np.save(test / "ID_2_7.npy", np.full((2, 4), 2))
    (train / "notes.txt").write_text("not a sample")
    train_set, test_set = dataset.from_directory(train), dataset.from_directory(test)
    assert [Path(p).name for p in train_set.data_paths] == ["ID_0_2.npy", "ID_1_3.npy"]
    assert [Path(p).name for p in test_set.data_paths] == ["ID_2_7.npy"]
    sample, label = test_set[0]
    torch.testing.assert_close(sample, torch.full((2, 4), 2.))
    assert label.item() == 7


def test_sidms_forcing_preserves_spectral_and_filter_updates():
    from sidms.cells import build_cell
    cell = build_cell("sprif_scalar", 1, 1, 5, 1.)
    assert local_count(cell) == 11
    set_hand_parameters(cell)
    with torch.no_grad():
        cell.input.weight.fill_(2)
        cell.input.bias.zero_()
    state = {"x": torch.tensor([[[.2, .4, .6]]]), "h": torch.tensor([[.7]]),
             "u": torch.tensor([[.4]])}
    _, clean, _ = cell.step(torch.ones(1, 1), state, torch.zeros(1, 1, dtype=torch.bool))
    spikes, forced, diag = cell.step(torch.ones(1, 1), state, torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(forced["x"], clean["x"])
    torch.testing.assert_close(forced["h"], clean["h"])
    torch.testing.assert_close(forced["h"], torch.tensor([[1.575]]))
    torch.testing.assert_close(spikes, torch.ones(1, 1))
    torch.testing.assert_close(forced["u"], torch.tensor([[.05]]))
    assert diag["forced_hit"].all()


def test_sidms_summary_names_new_reference_without_relabeling_legacy():
    from sidms.reporting import summarize
    rows = []
    for model, stress in [("sprif_scalar", .9), ("asrnn", .7), ("sprif_full", .8)]:
        for seed in (1, 2):
            rows += [{"model": model, "seed": seed, "intervention_count": 0,
                      "intervention_fraction": 0., "accuracy": 1.},
                     {"model": model, "seed": seed, "intervention_count": 40,
                      "intervention_fraction": .15, "accuracy": stress}]
    output = {r["model"]: r for r in summarize(rows)}
    assert output["sprif_scalar"]["role"] == "primary_model"
    assert output["sprif_full"]["role"] == "legacy_model"
    assert output["asrnn"]["reference_model"] == "sprif_scalar"
    assert "p_vs_sprif_scalar" in output["asrnn"]
    assert "p_vs_sprif_full" not in output["asrnn"]
