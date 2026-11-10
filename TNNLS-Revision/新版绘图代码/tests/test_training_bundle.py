"""Test checkpoint-to-figure interfaces with untrained temporary models only."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIG = Path(__file__).resolve().parents[1]
TRAIN = FIG / "04-训练代码"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if result.returncode:
        raise AssertionError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def make_layer(path, offset=0):
    module = load_module(f"layer_fixture_{offset}_{path.parent.parent.name}", path)
    layer = module.SPRiFNeuronLayer(1, 4, recurrent=False)
    with torch.no_grad():
        layer.alpha_raw.zero_(); layer.rho_raw.zero_(); layer.omega_raw.zero_()
        layer.eta_raw.fill_(-2); layer.bar_G.zero_()
        layer.bar_G[:, 0, :2] = 1
        layer.input_linear.weight.fill_(8 + offset * .1)
    return layer


class TrainingBundleTests(unittest.TestCase):
    def test_only_required_models_are_selected(self):
        module = load_module("train_figures", TRAIN / "train_figures.py")
        expected = {("pSMNIST", "sprif_scalar"), ("pSMNIST", "memory_reset5"),
                    ("GSC", "sprif_scalar"), ("QTDB", "sprif_scalar")}
        self.assertEqual(set(module.JOBS["all"]), expected)
        self.assertTrue(set(module.JOBS["mechanism"]).isdisjoint(module.JOBS["kernel"]))
        output = run(TRAIN / "train_figures.py", "--job", "all", "--dry-run")
        self.assertIn("4 model(s)", output)
        self.assertNotIn("--device", output)

    def test_missing_qtdb_is_detected_before_training(self):
        module = load_module("train_figures_preflight", TRAIN / "train_figures.py")
        with tempfile.TemporaryDirectory(prefix="figure-data-preflight-") as temp:
            root = Path(temp)
            for name in ("testing_list.txt", "validation_list.txt", "silence_validation_list.txt"):
                (root / name).write_text("", encoding="utf-8")
            args = SimpleNamespace(gsc_data_root=root, qtdb_train_mat=root / "missing-train.mat",
                                   qtdb_test_mat=root / "missing-test.mat")
            with self.assertRaises(FileNotFoundError):
                module.validate_data_paths(args, module.JOBS["kernel"])

    def test_completed_checkpoints_are_reused_without_training(self):
        module = load_module("train_figures_reuse", TRAIN / "train_figures.py")
        with tempfile.TemporaryDirectory(prefix="figure-checkpoint-reuse-") as temp:
            root = Path(temp)
            layer_path = TRAIN / "benchmarks" / "Task_pSMNIST" / "core_algorithm" / "sprif_layer.py"
            for task, model in module.JOBS["all"]:
                path = module.expected_checkpoint(root, task, model, 0)
                path.parent.mkdir(parents=True)
                layer = make_layer(layer_path)
                state = {f"layers.0.{key}": value for key, value in layer.state_dict().items()}
                torch.save({"model": state, "name": model, "task": task,
                            "seed": 0, "protocol_version": "benchmark-spectral-reset-v1",
                            "training_args": {}, "selected_epoch": 1}, path)
            output = run(TRAIN / "train_figures.py", "--job", "all", "--checkpoint-root", root)
            self.assertEqual(output.count("Reusing completed checkpoint"), 4)
            self.assertFalse(list(root.rglob("*.training.pt")))

    def test_matched_reset_rule_without_training(self):
        for task in ("Task_pSMNIST", "Task_SHD"):
            module = load_module(f"layer_{task}", TRAIN / "benchmarks" / task / "core_algorithm" / "sprif_layer.py")
            plain = module.SPRiFNeuronLayer(1, 2, recurrent=False)
            reset = module.SPRiFNeuronLayer(1, 2, recurrent=False, memory_reset=True)
            reset.load_state_dict(plain.state_dict())
            with torch.no_grad():
                for layer in (plain, reset):
                    layer.alpha_raw.zero_(); layer.rho_raw.zero_(); layer.omega_raw.zero_()
                    layer.eta_raw.fill_(-2); layer.bar_G.zero_(); layer.bar_G[:, 0, :2] = 1
                signal = torch.full((1, 2), 4.0)
                z1, m1, s1 = plain.forward_step(torch.zeros(1, 1), plain.init_state(1), plain._precompute_runtime_params(), input_current=signal)
                z2, m2, s2 = reset.forward_step(torch.zeros(1, 1), reset.init_state(1), reset._precompute_runtime_params(), input_current=signal)
            torch.testing.assert_close(z1, torch.ones_like(z1))
            torch.testing.assert_close(z1, z2); torch.testing.assert_close(m1, m2)
            torch.testing.assert_close(s1["h"], s2["h"]); torch.testing.assert_close(s1["u"], s2["u"])
            b = torch.tensor([.5, .5, 0.]); expected = b / torch.linalg.vector_norm(b)
            torch.testing.assert_close(s1["x"] - s2["x"], expected.expand_as(s1["x"]))
            self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in reset.parameters()))

    def test_spectral_curves_match_current_neuron_forward(self):
        layer_path = TRAIN / "benchmarks" / "Task_pSMNIST" / "core_algorithm" / "sprif_layer.py"
        layer = make_layer(layer_path)
        layer.threshold = 1e9
        plot = load_module("kernel_forward_check", FIG / "03-时间核" / "plot_kernels.py")
        runtime = layer._precompute_runtime_params()
        params = {"alpha": runtime["alpha"].detach().numpy(),
                  "rho": runtime["rho"].detach().numpy(), "omega": runtime["omega"].detach().numpy(),
                  "eta": runtime["eta"][0].detach().numpy(), "bar_G": layer.bar_G.detach().numpy(),
                  "threshold": np.ones(4)}
        expected = plot.responses(params, 25)
        with torch.no_grad():
            state = layer.init_state(1)
            actual = []
            for t in range(25):
                pulse = torch.full((1, 4), float(t == 0))
                _, _, state = layer.forward_step(torch.zeros(1, 1), state, runtime, input_current=pulse)
                actual.append(state["x"][0].numpy().copy())
        actual = np.stack(actual)
        np.testing.assert_allclose(expected["x_real"].T, actual[:, :, 0], atol=2e-6)
        np.testing.assert_allclose(expected["x_rot1"].T, actual[:, :, 1], atol=2e-6)

    def test_two_figures_read_checkpoints_directly(self):
        layer_path = TRAIN / "benchmarks" / "Task_pSMNIST" / "core_algorithm" / "sprif_layer.py"
        with tempfile.TemporaryDirectory(prefix="figure-training-interface-test-") as temp:
            tmp = Path(temp); root = tmp / "checkpoints"
            for task, model, offset in (("pSMNIST", "sprif_scalar", 0), ("pSMNIST", "memory_reset5", 1),
                                        ("GSC", "sprif_scalar", 2), ("QTDB", "sprif_scalar", 3)):
                layer = make_layer(layer_path, offset)
                state = {f"layers.0.{k}": v for k, v in layer.state_dict().items()}
                path = root / task / model / "seed_0" / "checkpoint.pt"
                path.parent.mkdir(parents=True)
                args = {"model": model, "seed": 0, "data_root": "test-only", "checkpoint_path": str(path)}
                torch.save({"model": state, "name": model, "task": task, "seed": 0,
                            "config": {"model": {"threshold": 1.0}}, "training_args": args,
                            "protocol_version": "benchmark-spectral-reset-v1"}, path)
            sample = tmp / "sample.npy"
            np.save(sample, np.ones((50, 1), dtype=np.float32))
            legacy = tmp / "legacy.npz"
            curve_gsc = .1 * np.power(.9, np.arange(40))
            curve_ecg = .05 * np.power(.95, np.arange(40))
            np.savez(legacy, GSC_col2_asrnn_kernel=curve_gsc, ECG_col2_asrnn_kernel=curve_ecg,
                     GSC_col2_meta=np.array([.9, 9.5, 9.5, 1, 2]),
                     ECG_col2_meta=np.array([.95, 19.5, 19.5, 1, 2]))
            mechanism = tmp / "mechanism"; kernels = tmp / "kernels"
            run(FIG / "01-机制分析" / "plot_mechanism.py", "--checkpoint-root", root, "--sample", sample,
                "--seed", "0", "--radius", "5", "--data-dir", tmp / "mechanism-data", "--output", mechanism)
            run(FIG / "03-时间核" / "plot_kernels.py", "--checkpoint-root", root, "--seed", "0", "--steps", "40",
                "--historical-asrnn", legacy, "--data-dir", tmp / "kernel-data", "--output", kernels)
            for output, stem in ((mechanism, "mechanism_composite"), (kernels, "temporal_kernels_composite")):
                self.assertTrue((output / f"{stem}.pdf").is_file())
                report = json.loads((output / f"{stem}.alignment.json").read_text(encoding="utf-8"))
                self.assertEqual(report["verdict"], "PASS")
            source = json.loads((kernels / "temporal_kernels_composite.source.json").read_text(encoding="utf-8"))
            self.assertEqual([g["label"] for g in source["groups"]], ["pSMNIST", "GSC", "QTDB"])
            self.assertEqual(source["comparisons"][0]["baseline_kind"], "historical_curve")
            with np.load(kernels / "temporal_kernels_composite.data.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["comparison0_asrnn_raw"], curve_gsc)
                np.testing.assert_array_equal(values["comparison1_asrnn_raw"], curve_ecg)
            for task, key in (("GSC", "dense_1.tau_m"), ("QTDB", "tau_m_h")):
                path = root / task / "asrnn" / "seed_0" / "checkpoint.pt"
                path.parent.mkdir(parents=True)
                torch.save({"model": {key: torch.tensor([5., 10., 20.])},
                            "name": "asrnn", "task": task, "seed": 0}, path)
            plot_module = load_module("direct_kernel_test", FIG / "03-时间核" / "plot_kernels.py")
            groups, comparisons = plot_module.checkpoint_groups(root, 0, "trained", legacy, tmp / "trained-data")
            self.assertEqual(len(groups), 3)
            self.assertEqual(len(comparisons), 2)
            with np.load(comparisons[0]["asrnn"], allow_pickle=False) as exported:
                np.testing.assert_allclose(exported["alpha"], np.exp(-1 / np.array([5., 10., 20.])))
            (root / "QTDB" / "asrnn" / "seed_0" / "checkpoint.pt").unlink()
            _, mixed = plot_module.checkpoint_groups(root, 0, "auto", legacy, tmp / "mixed-data")
            gsc_meta = json.loads(mixed[0]["asrnn"].with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(gsc_meta["model"], "asrnn")
            self.assertEqual(mixed[1]["asrnn"], legacy)
            self.assertEqual(mixed[1]["legacy_task"], "ECG")


if __name__ == "__main__":
    unittest.main()
