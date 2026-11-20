"""Draw one SPRiF trajectory in the original three-row panel-a layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from checkpoint_io import file_sha256, load_checkpoint
from publication import require_matplotlib_panel_alignment, save_figure, set_style

STYLE = {"font.family": "sans-serif", "svg.fonttype": "none", "pdf.fonttype": 42,
         "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}
FORMATS = (".pdf", ".svg", ".png")
TEAL, ORANGE, PURPLE = "#42949E", "#E28E2C", "#9A4D8E"
BLUE, RED, GRID, SPIKE_BAND = "#3775BA", "#B64342", "#D8D8D8", "#FCEAEA"


def prepare_from_checkpoint(path, seed, layer, sample_index, sample_path, data_root, data_dir):
    code_root = BASE.parent / "04-训练代码" / "benchmarks" / "Task_pSMNIST"
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained SPRiF checkpoint: {path}")
    obj, _ = load_checkpoint(path)
    if obj.get("name") is not None and obj["name"] != "sprif_scalar":
        raise ValueError("Mechanism figure uses only the SPRiF model")
    if obj.get("task") is not None and obj["task"] != "pSMNIST":
        raise ValueError("Checkpoint task is not pSMNIST")
    if obj.get("seed") is not None and obj["seed"] != seed:
        raise ValueError("Checkpoint seed does not match --seed")
    if sample_path is None:
        raw_root = data_root or Path(obj.get("training_args", {}).get("data_root", "./data"))
        if not raw_root.is_absolute():
            raw_root = code_root / raw_root
        raw_root = raw_root.resolve()
        tag = hashlib.sha256(str(raw_root).encode("utf-8")).hexdigest()[:8]
        sample_path = data_dir / f"pSMNIST_seed{seed}_sample{sample_index}_{tag}.npz"
        if sample_path.exists():
            with np.load(sample_path, allow_pickle=False) as z:
                if int(z["seed"]) != seed or int(z["sample_index"]) != sample_index:
                    raise ValueError(f"Existing sample metadata does not match: {sample_path}")
        else:
            subprocess.run([sys.executable, str(code_root / "export_sample.py"), "--data-root", str(raw_root),
                            "--seed", str(seed), "--index", str(sample_index), "--output", str(sample_path)], check=True)
        sample_id = f"MNIST-test-{sample_index}-permutation-seed-{seed}"
    else:
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        sample_id = sample_path.stem
    checkpoint_hash, sample_hash = file_sha256(path), file_sha256(sample_path)
    trajectory = data_dir / f"pSMNIST_sprif_seed{seed}_layer{layer}_{checkpoint_hash[:8]}_{sample_hash[:8]}.npz"
    sidecar = trajectory.with_suffix(".json")
    if trajectory.exists() or sidecar.exists():
        if not (trajectory.is_file() and sidecar.is_file()):
            raise ValueError(f"Incomplete cached trajectory: {trajectory}")
        meta = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        expected = (meta.get("checkpoint", {}).get("sha256"), meta.get("input", {}).get("sha256"))
        if expected != (checkpoint_hash, sample_hash):
            raise ValueError(f"Cached trajectory provenance mismatch: {trajectory}")
    else:
        command = [sys.executable, str(BASE / "export_sprif_trajectory.py"), "--code-root", str(code_root),
                   "--checkpoint", str(path), "--input", str(sample_path), "--sample-id", sample_id,
                   "--seed", str(seed), "--layer", str(layer), "--output", str(trajectory)]
        if obj.get("config", {}).get("model", {}).get("threshold") is None:
            raise ValueError("Checkpoint has no threshold metadata; export a trajectory explicitly with --threshold")
        subprocess.run(command, check=True)
    return trajectory


def load_data(path, allow_smoke=False):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"]))
        if meta.get("format") == "psmnist-sprif-trajectory-v1":
            names = {k: k for k in ("x_pre", "x_post", "u_pre", "u_post", "spike")}
        elif meta.get("format") in {"sidms-actual-network-trajectories-v1", "benchmark-reconstructed-network-trajectories-v1"}:
            names = {k: "sprif_" + k for k in ("x_pre", "x_post", "u_pre", "u_post", "spike")}
        else:
            raise ValueError("Unsupported trajectory format")
        missing = sorted((set(names.values()) | {"threshold"}) - set(z.files))
        if missing:
            raise ValueError(f"Missing trajectory fields: {missing}")
        data = {key: np.array(z[source]) for key, source in names.items()}
        data["threshold"] = np.array(z["threshold"])
    if meta.get("smoke") and not allow_smoke:
        raise ValueError("Smoke data cannot be used as formal figure input")
    spike = data["spike"]
    if spike.ndim != 2 or spike.size == 0 or not np.isfinite(spike).all():
        raise ValueError("Spike trajectory must be finite [time, neuron]")
    for key in ("x_pre", "x_post", "u_pre", "u_post"):
        expected = (*spike.shape, 3) if key.startswith("x_") else spike.shape
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}: expected finite {expected}")
    theta = float(data["threshold"])
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("Invalid threshold")
    if not np.allclose(data["x_pre"], data["x_post"], atol=2e-5):
        raise ValueError("SPRiF spectral state changed during reset")
    if not np.allclose(data["u_post"] - data["u_pre"], -theta * (spike > .5), atol=2e-5):
        raise ValueError("SPRiF membrane reset does not match threshold")
    return data, meta


def select_event(data, radius, step=None, neuron=None):
    if radius < 1:
        raise ValueError("--radius must be positive")
    spikes = data["spike"] > .5
    n_time, n_neurons = spikes.shape
    if (step is None) != (neuron is None):
        raise ValueError("Provide --event-step and --neuron together")
    if step is not None:
        if not (radius <= step < n_time - radius and 0 <= neuron < n_neurons and spikes[step, neuron]):
            raise ValueError("Selected neuron/step must be a spike with a full window")
        return step, neuron, "explicit"
    scores = []
    for i in range(n_neurons):
        times = np.flatnonzero(spikes[:, i])
        valid = times[(times >= radius) & (times < n_time - radius)]
        if len(times) >= 3 and len(valid):
            membrane_range = float(np.ptp(data["u_pre"][:, i]))
            scores.append((membrane_range, i, valid))
    if scores:
        _, neuron, times = min(scores, key=lambda item: (item[0], item[1]))
        return int(times[len(times) // 2]), neuron, "smallest membrane range among neurons with at least 3 spikes; median valid spike"
    candidates = np.argwhere(spikes)
    candidates = candidates[(candidates[:, 0] >= radius) & (candidates[:, 0] < n_time - radius)]
    if not len(candidates):
        raise ValueError("No spike has the requested full window; try another test sample")
    step, neuron = map(int, candidates[0])
    return step, neuron, "first spike with a complete window"


def build(data, meta, step, neuron, radius):
    set_style(STYLE)
    fig = plt.figure(figsize=(3.5, 3.6))
    grid = fig.add_gridspec(3, 1, left=.22, right=.96, bottom=.16, top=.92,
                            height_ratios=[1.2, 1.0, .32], hspace=.35)
    state = fig.add_subplot(grid[0]); membrane = fig.add_subplot(grid[1], sharex=state)
    raster = fig.add_subplot(grid[2], sharex=state)
    for ax, gid in ((state, "state"), (membrane, "membrane"), (raster, "spikes")):
        ax.set_gid(gid)
    sl = slice(step-radius, step+radius+1)
    t = np.arange(-radius, radius+1)
    slow = data["x_post"][sl, neuron]
    voltage = data["u_pre"][sl, neuron]
    relative_voltage = voltage - voltage[0]
    spikes = data["spike"][sl, neuron] > .5
    for ax in (state, membrane):
        ax.axvspan(-.5, .5, color=SPIKE_BAND, alpha=.7, zorder=0)
        ax.axvline(0, color=RED, ls="--", lw=.9, alpha=.8, zorder=4)
    labels = [r"$x^{\mathrm{real}}$", r"$x^{\mathrm{osc}}_1$", r"$x^{\mathrm{osc}}_2$"]
    for i, (color, text) in enumerate(zip((TEAL, ORANGE, PURPLE), labels)):
        offset = i * .04
        values = slow[:, i] + offset
        state.plot(t, values, color=color, label=text, lw=1.4, zorder=3)
        if i == 0:
            state.fill_between(t, offset, values, color=color, alpha=.12, zorder=2)
    state.set_ylabel("slow state (offset)")
    task = meta.get("task", "pSMNIST")
    state.set_title(f"Recorded {task} spike")
    state.legend(loc="upper right", handlelength=1.2, frameon=True, framealpha=.85, edgecolor=GRID)
    membrane.plot(t, relative_voltage, color=BLUE, lw=1.4)
    membrane.scatter(t[spikes], relative_voltage[spikes], color=RED, marker="v", s=15,
                     zorder=3, edgecolor="white", linewidth=.3)
    membrane.axhline(0, color=GRID, lw=.5)
    membrane.set_ylabel(r"Fast $u^0$ (rel.)")
    raster.vlines(t[spikes], 0, 1, color=RED, lw=1.2)
    raster.axvline(0, color=RED, ls="--", lw=.7, alpha=.5)
    raster.set(xlim=(t[0], t[-1]), ylim=(0, 1), yticks=[], xlabel="Steps relative to spike")
    if radius == 15:
        raster.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    raster.spines["left"].set_visible(False)
    for ax in (state, membrane):
        ax.grid(True, color=GRID, linewidth=.3, alpha=.6)
        ax.tick_params(length=2.5, width=.4, labelbottom=False)
    raster.tick_params(length=2.5, width=.4)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Existing SPRiF trajectory NPZ")
    source.add_argument("--checkpoint", type=Path, help="A trained pSMNIST SPRiF checkpoint")
    source.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", type=int, default=1, help="Hidden layer, default 1 as in the original a panel")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--sample", type=Path, help="Existing real preprocessed PS-MNIST sequence")
    p.add_argument("--psmnist-data-root", type=Path)
    p.add_argument("--data-dir", type=Path, default=BASE / "原始数据")
    p.add_argument("--output", type=Path, default=BASE / "输出图片")
    p.add_argument("--event-step", type=int); p.add_argument("--neuron", type=int)
    p.add_argument("--radius", type=int, default=15)
    p.add_argument("--allow-smoke", action="store_true")
    args = p.parse_args()
    if args.input:
        input_path = args.input
    else:
        root = args.checkpoint_root or (BASE.parent / "04-训练代码" / "checkpoints")
        checkpoint = args.checkpoint or root / "pSMNIST" / "sprif_scalar" / f"seed_{args.seed}" / "checkpoint.pt"
        input_path = prepare_from_checkpoint(checkpoint, args.seed, args.layer, args.sample_index,
                                             args.sample, args.psmnist_data_root, args.data_dir)
    data, meta = load_data(input_path, args.allow_smoke)
    step, neuron, selection = select_event(data, args.radius, args.event_step, args.neuron)
    fig = build(data, meta, step, neuron, args.radius)
    base = args.output / "mechanism_sprif"
    require_matplotlib_panel_alignment(fig, json_out=str(base) + ".alignment.json", strict=True,
                                      row_groups=[], column_groups=[{"id": "three_rows", "panels": ["state", "membrane", "spikes"]}])
    save_figure(fig, base, formats=FORMATS, dpi=600)
    report = {"input": str(input_path), "source": meta, "model": "SPRiF", "event_step": step,
              "neuron": neuron, "selection": selection, "radius_steps": args.radius,
              "neurons_in_recording": int(data["spike"].shape[1]),
              "spike_events_in_recording": int((data["spike"] > .5).sum()),
              "plot_data": "stored spectral state, pre-reset membrane relative to window start, and spikes",
              "note": "One selected neuron from one test sample; not a multi-seed statistical comparison."}
    (args.output / "mechanism_sprif.source.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved SPRiF mechanism figure: {base}.pdf (step={step}, neuron={neuron})")


if __name__ == "__main__":
    main()
