"""No-training regression tests; all synthetic checkpoints remain in a temporary directory."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[1]
ROOT = FIG.parents[1]
SIDMS = ROOT / "代码" / "experiments" / "SI-DMS"
sys.path.insert(0, str(FIG))
sys.path.insert(0, str(SIDMS))


def run(*args, success=True):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", env=env)
    if success and result.returncode:
        raise AssertionError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    if not success and result.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {args}")
    return result


class CompositeTests(unittest.TestCase):
    def test_impulse_matches_model_replay_and_formula(self):
        from common import replay
        spec = importlib.util.spec_from_file_location("plot_kernels", FIG / "03-时间核" / "plot_kernels.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        p = {"alpha": np.array([.5, .8]), "rho": np.array([.7, .6]),
             "omega": np.array([.4, 1.1]), "eta": np.array([[.8, .6], [.5, .7]]),
             "bar_G": np.array([[[.3, .2, -.4], [.1, .5, .3]],
                                  [[.2, -.7, .1], [.4, .2, -.1]]]),
             "threshold": np.ones(2)}
        steps = 40
        current = np.zeros((steps, 2)); current[0] = 1
        actual = replay(p, current, spikes=False)
        out = module.responses(p, steps)
        np.testing.assert_allclose(out["total"].T, actual["drive"], atol=1e-12)
        np.testing.assert_allclose(out["membrane"].T, actual["u_pre"], atol=1e-12)
        np.testing.assert_allclose(out["x_real"].T, actual["x_pre"][:, :, 0], atol=1e-12)
        np.testing.assert_allclose(out["x_rot1"].T, actual["x_pre"][:, :, 1], atol=1e-12)
        b = np.array([1 - p["alpha"][0], 1 - p["rho"][0], 0])
        self.assertAlmostEqual(out["total"][0, 0], np.dot(p["bar_G"][0, 0], b))
        a, r, w = p["alpha"][0], p["rho"][0], p["omega"][0]
        matrix = np.array([[a, 0, 0], [0, r*np.cos(w), -r*np.sin(w)], [0, r*np.sin(w), r*np.cos(w)]])
        expected_one = p["bar_G"][0, 0] @ matrix @ b + p["bar_G"][0, 1] @ b
        self.assertAlmostEqual(out["total"][0, 1], expected_one)

    def test_checkpoint_to_both_composites_without_training(self):
        import torch
        from sidms.config import ExperimentConfig
        from sidms.models import SIDMSNetwork
        cfg = ExperimentConfig(); cfg.model.hidden_size = 8
        torch.manual_seed(7)
        keep = os.environ.get("SIDMS_FIGURE_TEST_KEEP")
        context = nullcontext(keep) if keep else tempfile.TemporaryDirectory(prefix="sidms-figure-test-")
        with context as tmp:
            tmp = Path(tmp)
            checkpoints = {}
            for name in ("sprif_scalar", "memory_reset5", "asrnn", "real_modes3"):
                network = SIDMSNetwork(name, cfg)  # random initialization only: no training
                path = tmp / f"{name}.pt"
                torch.save({"model": network.state_dict(), "config": cfg.to_dict(), "name": name,
                            "training": "Intervention", "seed": 0, "smoke": True,
                            "protocol_version": "synthetic-fixture"}, path)
                checkpoints[name] = path
            parameter = tmp / "sprif.npz"; baseline = tmp / "asrnn.npz"
            run(FIG / "export_parameters.py", "--checkpoint", checkpoints["sprif_scalar"], "--output", parameter)
            run(FIG / "export_parameters.py", "--checkpoint", checkpoints["real_modes3"], "--output", tmp / "bad.npz", success=False)
            run(FIG / "03-时间核" / "export_asrnn_parameters.py", "--checkpoint", checkpoints["asrnn"], "--output", baseline)
            traj = tmp / "trajectory.npz"
            run(FIG / "01-机制分析" / "export_trajectories.py", "--code-root", SIDMS,
                "--sprif-checkpoint", checkpoints["sprif_scalar"], "--reset-checkpoint", checkpoints["memory_reset5"],
                "--delay-ms", "200", "--count", "2", "--fraction", ".5", "--allow-smoke", "--output", traj)
            with np.load(traj, allow_pickle=False) as z:
                forced = z["forced"]
                self.assertGreater(forced.sum(), 0)
                np.testing.assert_allclose(z["sprif_x_pre"], z["sprif_x_post"], atol=1e-6)
                dx = z["reset_x_post"] - z["reset_x_pre"]
                np.testing.assert_allclose(np.linalg.norm(dx[forced], axis=-1), 1, atol=1e-6)
            mechanism_out = tmp / "mechanism"
            run(FIG / "01-机制分析" / "plot_mechanism.py", "--input", traj, "--output", mechanism_out,
                "--radius", "5", success=False)
            run(FIG / "01-机制分析" / "plot_mechanism.py", "--input", traj, "--output", mechanism_out,
                "--radius", "5", "--allow-smoke")
            self.assertTrue((mechanism_out / "mechanism_composite.pdf").is_file())
            manifest = tmp / "manifest.json"
            manifest.write_text(json.dumps({"groups": [{"label": f"group{i}", "parameters": str(parameter)} for i in range(3)],
                                            "comparisons": [{"group": f"group{i}", "asrnn": str(baseline)} for i in (1, 2)]}), encoding="utf-8")
            kernel_out = tmp / "kernels"
            run(FIG / "03-时间核" / "plot_kernels.py", "--manifest", manifest, "--steps", "40", "--output", kernel_out, success=False)
            run(FIG / "03-时间核" / "plot_kernels.py", "--manifest", manifest, "--steps", "40", "--allow-smoke", "--output", kernel_out)
            self.assertTrue((kernel_out / "temporal_kernels_composite.pdf").is_file())
            for output, stem in ((mechanism_out, "mechanism_composite"), (kernel_out, "temporal_kernels_composite")):
                result = json.loads((output / f"{stem}.alignment.json").read_text(encoding="utf-8"))
                self.assertEqual(result["verdict"], "PASS")

    def test_benchmark_paired_natural_spikes_without_training(self):
        import torch
        task_root = ROOT / "代码" / "Task_pSMNIST"
        sys.path.insert(0, str(task_root))
        from core_algorithm.sprif_layer import SPRiFNeuronLayer
        with tempfile.TemporaryDirectory(prefix="benchmark-figure-test-") as tmp:
            tmp = Path(tmp)
            paths = []
            for i in range(2):
                layer = SPRiFNeuronLayer(1, 4, recurrent=False)
                with torch.no_grad():
                    layer.alpha_raw.zero_(); layer.rho_raw.zero_(); layer.omega_raw.zero_()
                    layer.eta_raw.fill_(-2); layer.bar_G.zero_()
                    layer.bar_G[:, 0, :2] = 1
                    layer.input_linear.weight.fill_(8 + i*.1)
                path = tmp / f"model{i}.pt"
                torch.save({f"layers.0.{k}": v for k, v in layer.state_dict().items()}, path)
                paths.append(path)
            sample = tmp / "sample.npy"
            np.save(sample, np.ones((50, 1), dtype=np.float32))
            traj = tmp / "benchmark.npz"
            run(FIG / "01-机制分析" / "export_benchmark_trajectories.py", "--code-root", task_root,
                "--sprif-checkpoint", paths[0], "--reset-checkpoint", paths[1], "--input", sample,
                "--task", "pSMNIST", "--sample-id", "synthetic-test-only", "--seed", "0",
                "--threshold", "1", "--output", traj)
            with np.load(traj, allow_pickle=False) as z:
                self.assertGreater(z["event_mask"].sum(), 0)
                self.assertEqual(json.loads(str(z["metadata_json"]))["event_type"], "coincident_natural")
                dx = z["reset_x_post"] - z["reset_x_pre"]
                np.testing.assert_allclose(np.linalg.norm(dx[z["event_mask"]], axis=-1), 1, atol=2e-6)
            output = tmp / "figure"
            run(FIG / "01-机制分析" / "plot_mechanism.py", "--input", traj, "--output", output, "--radius", "5")
            self.assertTrue((output / "mechanism_composite.pdf").exists())


if __name__ == "__main__":
    unittest.main()
