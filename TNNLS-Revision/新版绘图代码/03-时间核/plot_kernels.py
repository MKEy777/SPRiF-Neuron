"""Old-layout temporal-kernel composite using current-model parameters."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from common import parameters
from checkpoint_io import file_sha256, load_checkpoint
from publication import label, save_figure, set_style, require_matplotlib_panel_alignment

STYLE = {"font.family": "sans-serif", "svg.fonttype": "none", "pdf.fonttype": 42}
FORMATS = (".pdf", ".svg", ".png")


def _run_export(command, output, expected_hash):
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError(f"Incomplete parameter export: {output}")
        meta = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        if meta.get("sha256") != expected_hash:
            raise ValueError(f"Cached parameter source hash does not match: {output}")
        return
    subprocess.run([sys.executable, *map(str, command)], check=True)


def checkpoint_groups(root, seed, asrnn_source, historical, data_dir):
    cache = data_dir / "checkpoint_exports" / f"seed_{seed}"
    groups = []
    for task in ("pSMNIST", "GSC", "QTDB"):
        path = root / task / "sprif_scalar" / f"seed_{seed}" / "checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed SPRiF checkpoint: {path}")
        obj, _ = load_checkpoint(path)
        if obj.get("name") != "sprif_scalar" or obj.get("task") != task or obj.get("seed") != seed:
            raise ValueError(f"Wrong model, task, or seed in {path}")
        digest = file_sha256(path)
        output = cache / f"{task}_sprif_{digest[:12]}.npz"
        _run_export([BASE.parent / "export_parameters.py", "--checkpoint", path,
                     "--prefix", "layers.0.", "--output", output], output, digest)
        groups.append({"label": task, "parameters": output})
    comparisons = []
    used_historical = []
    for task, legacy_task, tau_key in (("GSC", "GSC", "dense_1.tau_m"), ("QTDB", "ECG", "tau_m_h")):
        path = root / task / "asrnn" / f"seed_{seed}" / "checkpoint.pt"
        use_historical = asrnn_source == "historical" or (asrnn_source == "auto" and not path.is_file())
        if use_historical:
            if not historical.is_file():
                raise FileNotFoundError(f"Historical ASRNN baseline file missing: {historical}")
            output = historical
            used_historical.append(task)
        else:
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed ASRNN checkpoint: {path}")
            obj, _ = load_checkpoint(path)
            if obj.get("name") != "asrnn" or obj.get("task") != task or obj.get("seed") != seed:
                raise ValueError(f"Wrong ASRNN model, task, or seed in {path}")
            digest = file_sha256(path)
            output = cache / f"{task}_asrnn_{digest[:12]}.npz"
            command = [BASE / "export_asrnn_parameters.py", "--checkpoint", path,
                       "--tau-key", tau_key, "--output", output]
            _run_export(command, output, digest)
        comparisons.append({"group": task, "asrnn": output, "legacy_task": legacy_task})
    if used_historical:
        print(f"Using historical ASRNN values for {', '.join(used_historical)}: verify original checkpoint, seed, and ECG/QTDB task correspondence before publication.", flush=True)
    return groups, comparisons


def responses(p, steps):
    """Spectral-state responses and mode-resolved no-reset drive/membrane responses."""
    n = len(p["alpha"])
    x = np.zeros((n, 3)); hr = np.zeros(n); ho = np.zeros(n); u = np.zeros(n)
    real = np.empty((n, steps)); rotation = np.empty_like(real); membrane = np.empty_like(real)
    x_real = np.empty_like(real); x_rot1 = np.empty_like(real); x_rot2 = np.empty_like(real)
    a, r, w, eta, g = p["alpha"], p["rho"], p["omega"], p["eta"], p["bar_G"]
    for t in range(steps):
        impulse = 1.0 if t == 0 else 0.0
        x = np.stack((a*x[:, 0] + (1-a)*impulse,
                      r*(np.cos(w)*x[:, 1] - np.sin(w)*x[:, 2]) + (1-r)*impulse,
                      r*(np.sin(w)*x[:, 1] + np.cos(w)*x[:, 2])), axis=1)
        x_real[:, t], x_rot1[:, t], x_rot2[:, t] = x[:, 0], x[:, 1], x[:, 2]
        real[:, t] = g[:, 0, 0]*x[:, 0] + hr
        rotation[:, t] = np.sum(g[:, 0, 1:]*x[:, 1:], axis=1) + ho
        u = eta[:, 0]*u + real[:, t] + rotation[:, t]
        membrane[:, t] = u
        hr = eta[:, 1]*hr + g[:, 1, 0]*x[:, 0]
        ho = eta[:, 1]*ho + np.sum(g[:, 1, 1:]*x[:, 1:], axis=1)
    return {"real": real, "rotation": rotation, "total": real+rotation, "membrane": membrane,
            "x_real": x_real, "x_rot1": x_rot1, "x_rot2": x_rot2}


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    groups = manifest.get("groups", [])
    comparisons = manifest.get("comparisons", [])
    if len(groups) != 3 or len(comparisons) != 2:
        raise ValueError("Manifest requires exactly three groups and two ASRNN comparisons for the original layout")
    labels = [g.get("label") for g in groups]
    if any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != 3:
        raise ValueError("Each group needs a unique nonempty label")
    if len({c.get("group") for c in comparisons}) != 2 or any(c.get("group") not in labels for c in comparisons):
        raise ValueError("The two comparisons must reference distinct group labels")
    for item in [*groups, *comparisons]:
        key = "parameters" if "label" in item else "asrnn"
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"Missing {key} path")
        item[key] = (path.parent / item[key]).resolve()
        if not item[key].is_file():
            raise FileNotFoundError(f"Missing source data: {item[key]}")
    return groups, comparisons


def load_asrnn(path, legacy_task):
    with np.load(path, allow_pickle=False) as z:
        if "alpha" not in z:
            key, meta_key = f"{legacy_task}_col2_asrnn_kernel", f"{legacy_task}_col2_meta"
            if key not in z or meta_key not in z:
                raise ValueError(f"{path}: historical comparison requires {key} and {meta_key}")
            curve = np.asarray(z[key], dtype=float)
            meta = np.asarray(z[meta_key], dtype=float)
            if curve.ndim != 1 or curve.size < 2 or not np.isfinite(curve).all() or meta.shape != (5,) or not np.isfinite(meta).all():
                raise ValueError(f"Invalid historical ASRNN curve or metadata: {path}")
            return {"kind": "historical_curve", "curve": curve, "tau": float(meta[2]),
                    "neuron": int(meta[4]), "dt_ms": None}
        alpha = np.asarray(z["alpha"], dtype=float)
        dt = float(z["dt_ms"]) if "dt_ms" in z else None
    if alpha.ndim != 1 or not alpha.size or not np.all(np.isfinite(alpha) & (alpha > 0) & (alpha < 1)):
        raise ValueError(f"{path}: ASRNN alpha must be finite [N] and in (0, 1)")
    return {"kind": "learned_alpha", "alpha": alpha, "dt_ms": dt}


def peak_normalize(values):
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-12:
        raise ValueError("A selected response is zero and cannot be normalized")
    return values / (peak + 1e-12)


def spectrum_stats(total):
    spectrum = np.abs(np.fft.rfft(total, axis=1))
    maxima = spectrum.max(axis=1)
    normal = spectrum / (maxima[:, None] + 1e-12)
    return np.median(normal, axis=0), *np.quantile(normal, [.1, .9], axis=0), int((maxima == 0).sum())


def source_metadata(path, allow_smoke):
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {"provenance_status": "metadata sidecar missing; source not verified"}
    meta = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if meta.get("smoke") is True and not allow_smoke:
        raise ValueError(f"Smoke data cannot be plotted as formal data: {path}")
    return meta


def build(groups, comparisons, steps, allow_smoke=False):
    set_style(STYLE)
    fig = plt.figure(figsize=(8.4, 4.8))
    outer = fig.add_gridspec(2, 12, left=.075, right=.975, bottom=.19, top=.9,
                             height_ratios=[1.05, .95], hspace=.68, wspace=1.1)
    top = outer[0, :].subgridspec(1, 3, wspace=.38)
    bottom = outer[1, :8].subgridspec(1, 3, wspace=.48)
    axes_top = [fig.add_subplot(top[0, i]) for i in range(3)]
    axes_freq = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    ax_compare = fig.add_subplot(outer[1, 8:12])
    for i, ax in enumerate(axes_top): ax.set_gid(f"top{i}")
    for i, ax in enumerate(axes_freq): ax.set_gid(f"freq{i}")
    ax_compare.set_gid("compare")
    quantiles = (.1, .4, .7, .9)
    colors = plt.cm.viridis(np.linspace(.18, .88, 4))
    task_colors = ("#9c4d0e", "#5d3a8e", "#0d3b5c")
    outputs = {}; provenance = []; results = {}
    for i, item in enumerate(groups):
        source = source_metadata(item["parameters"], allow_smoke)
        p = parameters(item["parameters"])
        res = responses(p, steps)
        results[item["label"]] = (p, res, i, source)
        ax = axes_top[i]
        selected = [int(np.argmin(np.abs(p["alpha"] - np.quantile(p["alpha"], q)))) for q in quantiles]
        for q, index, color in zip(quantiles, selected, colors):
            ax.plot(res["x_real"][index], color=color, lw=1.4, label=fr"$\alpha$={p['alpha'][index]:.2f}")
            ax.plot(res["x_rot1"][index], color=color, ls=":", lw=.9, alpha=.9)
        ax.axhline(0, color="#bbbbbb", lw=.5)
        ax.set(title=item["label"], xlabel="step", xlim=(0, steps-1))
        ax.title.set_color(task_colors[i])
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color(task_colors[i])
        ax.spines["top"].set_linewidth(1.8)
        if i == 0:
            ax.set_ylabel("impulse response")
            ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=.92,
                      columnspacing=.7, handlelength=1.4)
            label(ax, "a")
        freq = np.fft.rfftfreq(steps)
        median, lo, hi, zero_count = spectrum_stats(res["x_real"])
        f = axes_freq[i]
        f.fill_between(freq, lo, hi, color=task_colors[i], alpha=.2, lw=0)
        f.plot(freq, median, color=task_colors[i], lw=1.2)
        f.set(title=item["label"], xlabel="cycles / step", xlim=(0, .5), ylim=(0, 1.03))
        f.title.set_color(task_colors[i])
        if i == 0:
            f.set_ylabel("normalized |FFT|")
            label(f, "b")
        for name, values in res.items(): outputs[f"group{i}_{name}"] = values
        outputs[f"group{i}_alpha"] = p["alpha"]
        outputs[f"group{i}_selected_indices"] = np.array(selected)
        outputs[f"group{i}_frequency"] = freq
        outputs[f"group{i}_fft_median"] = median
        outputs[f"group{i}_fft_q10"] = lo
        outputs[f"group{i}_fft_q90"] = hi
        provenance.append({"label": item["label"], "parameters": str(item["parameters"]), "source": source,
                           "neurons": len(p["alpha"]), "selected_quantiles": quantiles,
                           "selected_indices": selected, "zero_spectra": zero_count, "neurons_excluded": 0})
    comp_meta = []
    for comp in comparisons:
        p, res, group_index, sprif_source = results[comp["group"]]
        source = source_metadata(comp["asrnn"], allow_smoke)
        legacy_task = comp.get("legacy_task") or {"GSC": "GSC", "QTDB": "ECG", "ECG": "ECG"}.get(comp["group"])
        baseline_info = load_asrnn(comp["asrnn"], legacy_task)
        dt = baseline_info["dt_ms"]
        sprif_dt = sprif_source.get("dt_ms")
        if dt is not None and sprif_dt is not None and not np.isclose(float(dt), float(sprif_dt)):
            raise ValueError(f"{comp['group']}: SPRiF and ASRNN have different time steps")
        sorted_alpha = np.sort(p["alpha"])
        target_alpha = sorted_alpha[min(len(sorted_alpha)-1, 9*len(sorted_alpha)//10)]
        neuron = int(np.argmin(np.abs(p["alpha"] - target_alpha)))
        sprif_tau = float(-1.0 / np.log(p["alpha"][neuron]))
        if baseline_info["kind"] == "historical_curve":
            baseline_raw = baseline_info["curve"]
            base_neuron, baseline_tau = baseline_info["neuron"], baseline_info["tau"]
        else:
            taus = -1.0 / np.log(baseline_info["alpha"])
            base_neuron = int(np.argmin(np.abs(taus - sprif_tau)))
            baseline_tau = float(taus[base_neuron])
            a = baseline_info["alpha"][base_neuron]
            baseline_raw = (1-a) * np.power(a, np.arange(steps))
        window = min(steps, len(baseline_raw))
        sprif = peak_normalize(res["x_real"][neuron, :window])
        baseline = peak_normalize(baseline_raw[:window])
        color = task_colors[group_index]
        marker = "o" if comp["group"] == "GSC" else "s"
        ax_compare.plot(sprif, color=color, lw=1.6, marker=marker, markevery=10,
                        markerfacecolor="white", markeredgewidth=1, markeredgecolor=color,
                        label=f"{comp['group']} SPRiF  τ={sprif_tau:.0f}")
        ax_compare.plot(baseline, color=color, ls="--", lw=1.3, marker=marker, markevery=10,
                        markerfacecolor=color, markeredgewidth=1, markeredgecolor=color,
                        label=f"{comp['group']} ASRNN  τ={baseline_tau:.0f}")
        outputs[f"comparison{len(comp_meta)}_sprif_real_state"] = sprif
        outputs[f"comparison{len(comp_meta)}_asrnn_raw"] = baseline_raw
        outputs[f"comparison{len(comp_meta)}_asrnn_normalized"] = baseline
        comp_meta.append({"group": comp["group"], "asrnn": str(comp["asrnn"]), "dt_ms": dt, "source": source,
                          "sprif_neuron": neuron, "asrnn_neuron": base_neuron,
                          "sprif_tau_steps": sprif_tau, "asrnn_tau_steps": baseline_tau,
                          "baseline_kind": baseline_info["kind"], "displayed_steps": window,
                          "selection": "SPRiF original 90th-percentile index; historical ASRNN curve unchanged" if baseline_info["kind"] == "historical_curve" else "SPRiF original 90th-percentile index; nearest ASRNN time constant"})
    max_window = max(item["displayed_steps"] for item in comp_meta)
    ax_compare.set(title="SPRiF vs. ASRNN slow-decay kernels", xlabel="step", ylabel="normalized kernel",
                   xlim=(0, max_window-1), ylim=(-.03, 1.03))
    ax_compare.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=True,
                      framealpha=.95, fontsize=6, columnspacing=.8, handlelength=1.5)
    label(ax_compare, "c")
    return fig, outputs, {"groups": provenance, "comparisons": comp_meta,
                          "steps": steps, "frequency_band": "real spectral state, per-neuron normalized; median and 10th–90th percentiles",
                          "note": "Old panel meanings retained: spectral-state responses, FFT of real state, and selected real-state versus ASRNN curves. Historical ASRNN curves are unchanged; they are not re-matched to the new SPRiF time constant."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="Use previously exported parameter NPZ files")
    source.add_argument("--checkpoint-root", type=Path, help="Read benchmark checkpoints directly")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--asrnn-source", choices=["auto", "historical", "trained"], default="historical")
    p.add_argument("--historical-asrnn", type=Path, default=BASE / "原始数据" / "legacy_asrnn_comparison.npz")
    p.add_argument("--data-dir", type=Path, default=BASE / "原始数据", help="Location for exported parameter data")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output", type=Path, default=BASE / "输出图片")
    p.add_argument("--allow-smoke", action="store_true", help="Development only; not a formal figure")
    args = p.parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.manifest:
        groups, comparisons = load_manifest(args.manifest)
    else:
        root = args.checkpoint_root or (BASE.parent / "04-训练代码" / "checkpoints")
        groups, comparisons = checkpoint_groups(root, args.seed, args.asrnn_source, args.historical_asrnn, args.data_dir)
    fig, data, meta = build(groups, comparisons, args.steps, args.allow_smoke)
    base = args.output / "temporal_kernels_composite"
    require_matplotlib_panel_alignment(
        fig, json_out=str(base) + ".alignment.json", strict=True,
        row_groups=[{"id": "top", "panels": ["top0", "top1", "top2"]},
                    {"id": "bottom", "panels": ["freq0", "freq1", "freq2"]}],
        column_groups=[])
    save_figure(fig, base, formats=FORMATS, dpi=600)
    np.savez(args.output / "temporal_kernels_composite.data.npz", **data)
    (args.output / "temporal_kernels_composite.source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {args.output / 'temporal_kernels_composite.pdf'}")


if __name__ == "__main__":
    main()
