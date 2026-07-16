import subprocess
import sys
from pathlib import Path


def test_run_all_cli_completes_smoke_pipeline(tmp_path):
    experiment_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(experiment_root / "run_all.py"),
        "--config", str(experiment_root / "config" / "smoke.yaml"),
        "--models", "sprif",
        "--device", "cpu",
        "--no-sensitivity",
        "--output", str(tmp_path),
    ]

    result = subprocess.run(command, cwd=experiment_root, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "main_causal_figure.png").exists()


def test_parallel_shard_clis_cover_all_models_without_aggregation_races(tmp_path):
    experiment_root = Path(__file__).resolve().parents[1]
    common = [
        "--config", str(experiment_root / "config" / "smoke.yaml"),
        "--device", "cpu", "--no-sensitivity", "--output", str(tmp_path),
    ]
    first = subprocess.run(
        [sys.executable, str(experiment_root / "run_sprif_lif.py"), *common],
        cwd=experiment_root, capture_output=True, text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert not (tmp_path / "summary.csv").exists()
    assert (tmp_path / "raw" / "sprif" / "seed_1" / "eval_metrics.json").exists()
    assert (tmp_path / "raw" / "lif" / "seed_1" / "eval_metrics.json").exists()

    second = subprocess.run(
        [sys.executable, str(experiment_root / "run_asrnn_brf.py"), *common],
        cwd=experiment_root, capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert not (tmp_path / "summary.csv").exists()
    for model in ("asrnn", "brf"):
        assert (tmp_path / "raw" / model / "seed_1" / "eval_metrics.json").exists()

    aggregate = subprocess.run(
        [sys.executable, str(experiment_root / "aggregate.py"), *common[:2],
         "--output", str(tmp_path)],
        cwd=experiment_root, capture_output=True, text=True,
    )
    assert aggregate.returncode == 0, aggregate.stdout + aggregate.stderr
    assert (tmp_path / "summary.csv").exists()
