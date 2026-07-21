"""Publication composite for controlled spike-intervention DMS experiments.

The main composite combines a deterministic representative replay, seed-level
accuracy curves, and delay-by-intervention heatmaps.  It deliberately reads
only the 18 per-run ``eval_metrics.json`` files, never aggregate summaries.
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
})

SIDMS_DIR = Path(__file__).resolve().parents[4] / "Spike-Intervention Delayed Match-to-Sample（SI-DMS）"
RESULTS_DIR = SIDMS_DIR / "results"
OUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SIDMS_DIR / "config" / "default_matched.yaml"
SPRIF_MODELS = ["sprif_full", "sprif_merged", "sprif_lambda0"]
EXTERNAL_MODELS = ["lif", "asrnn", "brf"]
ALL_MODELS = SPRIF_MODELS + EXTERNAL_MODELS
PRIMARY_FRACTION = 0.15
FRACTION_RESPONSE_K = 40
RASTER_MAX_DELAY_OCCUPANCY = 0.40
LEFT_COLUMN_HSPACE = 0.24
MODEL_LEGEND_NCOL = 3
MODEL_LEGEND_X = 0.27
MODEL_LEGEND_Y = 0.008

COLORS = {"sprif_full": "#1f77b4", "sprif_merged": "#d62728",
          "sprif_lambda0": "#ff7f0e", "lif": "#4f8f83",
          "asrnn": "#4b4b4b", "brf": "#7b6ba8"}
MARKERS = {"sprif_full": "o", "sprif_merged": "v", "sprif_lambda0": "s",
           "lif": "D", "asrnn": "P", "brf": "X"}
LINESTYLES = {m: "-" for m in SPRIF_MODELS}
LINESTYLES.update({m: "--" for m in EXTERNAL_MODELS})
LINESTYLES["sprif_lambda0"] = "-."
LABELS = {"sprif_full": "SPRiF full", "sprif_merged": "SPRiF merged",
          "sprif_lambda0": r"SPRiF $\lambda{=}0$", "lif": "LIF",
          "asrnn": "ASRNN", "brf": "BRF"}
EVAL_DELAYS = (200, 400, 800, 1600, 2500)
EVAL_POSITIVE_K = (1, 2, 4, 8, 16, 32, 40)
EVAL_FRACTIONS = (.05, .10, .15, .20, .30, .50)


def validate_run_entries(entries, expected_model, expected_seed):
    """Assert the exact SI-DMS evaluation grid for one model/seed run."""
    if len(entries) != 215:
        raise RuntimeError(f"Expected 215 rows for {expected_model}/seed_{expected_seed}, found {len(entries)}")
    expected_keys = {(delay, 0, 0.0) for delay in EVAL_DELAYS}
    expected_keys |= {(delay, count, fraction) for delay in EVAL_DELAYS
                      for count in EVAL_POSITIVE_K for fraction in EVAL_FRACTIONS}
    keys = []
    for row_index, entry in enumerate(entries):
        if entry.get("model") != expected_model:
            raise RuntimeError(f"model mismatch at row {row_index}: {entry.get('model')!r} != {expected_model!r}")
        if entry.get("seed") != expected_seed:
            raise RuntimeError(f"seed mismatch at row {row_index}: {entry.get('seed')!r} != {expected_seed!r}")
        try:
            delay = int(entry["delay_ms"])
            count = int(entry["intervention_count"])
            fraction = round(float(entry["intervention_fraction"]), 8)
            accuracy = float(entry["accuracy"])
            hit_rate = float(entry["forced_hit_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid evaluation row {row_index}") from exc
        if not 0.0 <= accuracy <= 1.0:
            raise RuntimeError(f"accuracy outside [0, 1] at row {row_index}: {accuracy}")
        expected_hit_rate = 0.0 if count == 0 else 1.0
        if not np.isclose(hit_rate, expected_hit_rate):
            raise RuntimeError(f"forced_hit_rate at row {row_index}: {hit_rate} != {expected_hit_rate}")
        keys.append((delay, count, fraction))
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate evaluation key in {expected_model}/seed_{expected_seed}")
    if set(keys) != expected_keys:
        raise RuntimeError(f"evaluation grid mismatch in {expected_model}/seed_{expected_seed}")


def load_data():
    """Load exactly one eval file for each model/seed run (18 files total)."""
    rows, paths = [], []
    production = RESULTS_DIR == SIDMS_DIR / "results"
    expected_paths = {(model, f"seed_{seed}") for model in ALL_MODELS for seed in (1, 2, 3)}
    if production:
        found_paths = {(path.parent.parent.name, path.parent.name)
                       for path in RESULTS_DIR.glob("*/seed_*/eval_metrics.json")}
        if found_paths != expected_paths:
            raise RuntimeError("Expected exactly six model directories × three seed directories with eval_metrics.json")
    for model in ALL_MODELS:
        for seed in (1, 2, 3):
            eval_path = RESULTS_DIR / model / f"seed_{seed}" / "eval_metrics.json"
            # The small unit-test fixture intentionally contains one run; the
            # published data directory is still checked strictly below.
            if not eval_path.exists():
                continue
            paths.append(eval_path)
            with eval_path.open(encoding="utf-8") as fh:
                entries = json.load(fh)
            if production:
                validate_run_entries(entries, model, seed)
            rows.extend({"model": e["model"], "seed": e["seed"],
                         "delay_ms": e["delay_ms"],
                         "intervention_count": e["intervention_count"],
                         "intervention_fraction": float(e.get("intervention_fraction", 0.0)),
                         "accuracy": float(e["accuracy"])} for e in entries)
    if production and len(paths) != 18:
        raise RuntimeError(f"Expected 18 per-run evaluation files (6 models × 3 seeds), found {len(paths)}")
    return pd.DataFrame(rows)


def select_primary_slice(df, fraction=PRIMARY_FRACTION):
    clean = df["intervention_count"] == 0
    stressed = (df["intervention_count"] > 0) & np.isclose(df["intervention_fraction"], fraction)
    return df[clean | stressed].copy()


def per_k_seed_stats(df, model, k_vals):
    """Average delays within a seed, then calculate cross-seed mean ± SD."""
    sub = df[df["model"] == model]
    means, stds = [], []
    for k in k_vals:
        per_seed = sub[sub["intervention_count"] == k].groupby("seed")["accuracy"].mean()
        means.append(per_seed.mean())
        stds.append(per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0)
    return np.asarray(means), np.asarray(stds)


def replay_trial(intervention_count=8):
    """Replay seed-0/sample-0 DMS deterministically and retain plotting traces.

    This mirrors ``SIDMSNetwork.forward`` locally so model public outputs and
    checkpoint contents are untouched while per-time readout logits are kept.
    """
    if str(SIDMS_DIR) not in sys.path:
        sys.path.insert(0, str(SIDMS_DIR))
    from sidms.config import ExperimentConfig
    from sidms.data import make_batch
    from sidms.models import SIDMSNetwork

    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    checkpoint = torch.load(RESULTS_DIR / "sprif_full" / "seed_1" / "checkpoint.pt",
                            map_location="cpu", weights_only=False)
    model = SIDMSNetwork("sprif_full", cfg).eval()
    model.load_state_dict(checkpoint["model"])
    batch = make_batch(cfg, batch_size=1, delay_ms=1600,
                       intervention_count=intervention_count,
                       hidden_size=cfg.model.hidden_size, seed=0, device="cpu",
                       fraction=PRIMARY_FRACTION)
    state = model.cell.initial_state(1, "cpu")
    prev = torch.zeros(1, cfg.model.hidden_size)
    out_mem = torch.zeros(1, 2)
    spikes, hits, logits = [], [], []
    with torch.no_grad():
        for t in range(batch.x.shape[1]):
            prev, state, diag = model.cell.step(batch.x[:, t], state, batch.intervention[:, t], prev)
            alpha = model.output_alpha
            out_mem = alpha * out_mem + (1 - alpha) * model.readout(prev)
            spikes.append(prev.squeeze(0).cpu())
            hits.append(diag["forced_hit"].squeeze(0).cpu())
            logits.append(out_mem.squeeze(0).cpu())
    trace = {"spikes": torch.stack(spikes).numpy(), "hits": torch.stack(hits).numpy().astype(bool),
             "logits": torch.stack(logits).numpy(), "target": int(batch.y.item()),
             "pred": int(torch.stack(logits)[-1].argmax().item()), "dt": cfg.task.dt_ms,
             "pre": cfg.task.pre_ms, "cue": cfg.task.cue_ms, "delay": 1600,
             "first": int(batch.first_side.item()), "second": int(batch.second_side.item()),
             "input": batch.x.squeeze(0).cpu().numpy(),
             "intervention": batch.intervention.squeeze(0).cpu().numpy().astype(bool)}
    # Invariants make figure generation fail rather than silently depicting a wrong replay.
    assert trace["spikes"].shape[0] == 370, trace["spikes"].shape
    assert trace["hits"].sum() == trace["intervention"].sum() == intervention_count * 10
    assert trace["target"] == trace["pred"] == 1  # 1 denotes match in the experiment code.
    cue2_start = (trace["pre"] + trace["cue"] + trace["delay"]) // trace["dt"]
    assert int(trace["spikes"][cue2_start:].sum()) == 203
    return trace


def _time_axis(trace):
    return np.arange(trace["spikes"].shape[0]) * trace["dt"]


def select_raster_units(trace, n_units=10, max_delay_occupancy=RASTER_MAX_DELAY_OCCUPANCY):
    """Select active, non-tonic units while retaining post-delay evidence.

    Units above ``max_delay_occupancy`` are excluded before ranking.  Among
    the remaining units, cue-2 natural activity receives the largest weight,
    followed by distributed delay and cue-1 activity.
    """
    raw_times = _time_axis(trace)
    display_mask = raw_times >= trace["pre"]
    times = raw_times[display_mask] - trace["pre"]
    forced = trace["hits"][display_mask]
    natural = (trace["spikes"] > 0)[display_mask] & ~forced
    cue_end = trace["cue"]
    delay_end = cue_end + trace["delay"]
    cue1_mask = times < cue_end
    delay_mask = (times >= cue_end) & (times < delay_end)
    cue2_mask = times >= delay_end
    delay_occupancy = natural[delay_mask].mean(axis=0)
    eligible = np.flatnonzero(delay_occupancy <= max_delay_occupancy + 1e-12)
    if len(eligible) < n_units:
        raise RuntimeError(
            f"Only {len(eligible)} units satisfy delay occupancy <= {max_delay_occupancy:.2f}"
        )
    score = (8.0 * natural[cue2_mask].sum(axis=0)
             + 0.25 * natural[delay_mask].sum(axis=0)
             + 0.50 * natural[cue1_mask].sum(axis=0)
             + 0.01 * natural.sum(axis=0))
    ranked = eligible[np.argsort(score[eligible], kind="stable")[::-1]]
    return ranked[:n_units]


def select_paired_raster_units(clean_trace, stressed_trace, n_units=10,
                               max_delay_occupancy=RASTER_MAX_DELAY_OCCUPANCY):
    """Choose one non-tonic unit set shared by clean and stressed replays."""
    scores, occupancies = [], []
    for trace in (clean_trace, stressed_trace):
        raw_times = _time_axis(trace)
        display_mask = raw_times >= trace["pre"]
        times = raw_times[display_mask] - trace["pre"]
        natural = ((trace["spikes"] > 0)[display_mask]
                   & ~trace["hits"][display_mask])
        cue_end = trace["cue"]
        delay_end = cue_end + trace["delay"]
        cue1_mask = times < cue_end
        delay_mask = (times >= cue_end) & (times < delay_end)
        cue2_mask = times >= delay_end
        occupancies.append(natural[delay_mask].mean(axis=0))
        scores.append(8.0 * natural[cue2_mask].sum(axis=0)
                      + 0.25 * natural[delay_mask].sum(axis=0)
                      + 0.50 * natural[cue1_mask].sum(axis=0)
                      + 0.01 * natural.sum(axis=0))
    max_occupancy = np.maximum.reduce(occupancies)
    eligible = np.flatnonzero(max_occupancy <= max_delay_occupancy + 1e-12)
    if len(eligible) < n_units:
        raise RuntimeError(
            f"Only {len(eligible)} paired units satisfy delay occupancy <= {max_delay_occupancy:.2f}"
        )
    combined_score = np.sum(scores, axis=0)
    ranked = eligible[np.argsort(combined_score[eligible], kind="stable")[::-1]]
    return ranked[:n_units]


def plot_paired_trial(clean_ax, stress_ax, activity_ax, readout_ax,
                      clean_trace, stressed_trace):
    """Panel (a): clean and K=8 replays of the exact same A--A input."""
    if not np.array_equal(clean_trace["input"], stressed_trace["input"]):
        raise RuntimeError("Paired trial requires identical input spikes")
    units = select_paired_raster_units(clean_trace, stressed_trace)
    dt, pre = clean_trace["dt"], clean_trace["pre"]
    cue, delay = clean_trace["cue"], clean_trace["delay"]
    raw_times = _time_axis(clean_trace)
    display_mask = raw_times >= pre
    times = raw_times[display_mask] - pre
    cue1 = (0, cue); delay_span = (cue, cue + delay)
    cue2 = (cue + delay, 2 * cue + delay)

    for ax, trace, ylabel in ((clean_ax, clean_trace, "Clean"),
                              (stress_ax, stressed_trace, "$K=8$")):
        natural = ((trace["spikes"] > 0)[display_mask]
                   & ~trace["hits"][display_mask])
        forced = trace["hits"][display_mask]
        ax.axvspan(*cue1, color="#9ecae1", alpha=.22, zorder=-4)
        ax.axvspan(*cue2, color="#9ecae1", alpha=.22, zorder=-4)
        for row, unit in enumerate(units):
            ax.vlines(times[natural[:, unit]], row - .34, row + .34,
                      color="#2878b8", lw=.55, alpha=.9)
            ax.vlines(times[forced[:, unit]], row - .34, row + .34,
                      color="#d43d3d", lw=1.0, alpha=.95)
        ax.set(xlim=(0, times[-1] + dt), ylim=(-.65, 9.65), yticks=[])
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=3, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=6.2, length=1.5, pad=1)

    clean_ax.tick_params(labelbottom=False)
    clean_ax.add_patch(Rectangle((cue1[0], 9.9), cue, .72,
                                 facecolor="#6baed6", edgecolor="none", clip_on=False))
    clean_ax.add_patch(Rectangle((delay_span[0], 9.9), delay, .72,
                                 facecolor="#eeeeee", edgecolor="none", clip_on=False))
    clean_ax.add_patch(Rectangle((cue2[0], 9.9), cue, .72,
                                 facecolor="#6baed6", edgecolor="none", clip_on=False))
    for x, label, color in ((sum(cue1) / 2, "A", "white"),
                            (sum(delay_span) / 2, "delay", "#333333"),
                            (sum(cue2) / 2, "A", "white")):
        clean_ax.text(x, 10.26, label, ha="center", va="center", fontsize=6.4,
                      color=color, fontweight="bold" if label == "A" else "normal",
                      clip_on=False)
    clean_ax.text(-.15, 1.28, "(a)", transform=clean_ax.transAxes,
                  fontweight="bold", fontsize=9)
    stress_ax.set_xlabel("Time (ms)", fontsize=6.7, labelpad=1)
    event_mask = stressed_trace["intervention"][display_mask].any(axis=1)
    for event_time in times[event_mask]:
        stress_ax.axvline(event_time, color="#d43d3d", lw=.45, alpha=.25, zorder=-2)

    cue2_mask = (times >= cue2[0]) & (times < cue2[1])
    cue2_times = times[cue2_mask]
    line_specs = ((clean_trace, "Clean", "#2878b8", "-"),
                  (stressed_trace, "$K=8$", "#d43d3d", "--"))
    for trace, label, color, linestyle in line_specs:
        natural = ((trace["spikes"] > 0)[display_mask]
                   & ~trace["hits"][display_mask])
        activity_ax.plot(cue2_times, natural[cue2_mask].sum(axis=1),
                         label=label, color=color, ls=linestyle, lw=.9)
        logits = trace["logits"][display_mask]
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        readout_ax.plot(cue2_times, prob[cue2_mask, 1], label=label,
                        color=color, ls=linestyle, lw=.9)
    activity_ax.set(xlim=cue2, ylabel="Spikes / step")
    activity_ax.set_title("Cue-2 natural spikes", fontsize=6.3, loc="left", pad=1)
    readout_ax.set(xlim=cue2, ylim=(0, 1.02), yticks=[0, 1], ylabel="$P$(match)")
    readout_ax.set_title("Cue-2 match readout", fontsize=6.3, loc="left", pad=1)
    for ax in (activity_ax, readout_ax):
        ax.tick_params(labelsize=5.8, length=1.2, pad=1)
        ax.legend(loc="upper left", fontsize=5.3, frameon=True, facecolor="white",
                  framealpha=.60, edgecolor="none", handlelength=1.2,
                  handletextpad=.3, borderpad=.2, labelspacing=.15)


def plot_trial(ax, trace, probax, zoom):
    """Panel (a): raster plus separate readout and cue-2 detail axes."""
    raw_times = _time_axis(trace)
    dt, pre, cue, delay = trace["dt"], trace["pre"], trace["cue"], trace["delay"]
    # The pre-period is part of replay but is omitted from the presentation so
    # displayed time zero coincides with cue 1.
    display_mask = raw_times >= pre
    times = raw_times[display_mask] - pre
    cue1 = (0, cue); delay_span = (cue1[1], cue1[1] + delay)
    cue2 = (delay_span[1], delay_span[1] + cue)
    natural = (trace["spikes"] > 0)[display_mask]
    forced = trace["hits"][display_mask]
    # Exclude tonic delay units before ranking, then retain visible cue-2 spiking.
    units = select_raster_units(trace)
    for row, unit in enumerate(units):
        spk_t = times[natural[:, unit] & ~forced[:, unit]]
        hit_t = times[forced[:, unit]]
        ax.vlines(spk_t, row - .34, row + .34, color="#2878b8", lw=.55, alpha=.9)
        ax.vlines(hit_t, row - .34, row + .34, color="#d43d3d", lw=1.0, alpha=.95)
    cue_palette = {0: "#9ecae1", 1: "#fdae6b"}
    first_side = int(trace.get("first", 0)); second_side = int(trace.get("second", 0))
    first_identity = "A" if first_side == 0 else "B"
    second_identity = "A" if second_side == 0 else "B"
    ax.axvspan(*cue1, color=cue_palette[first_side], alpha=.25, zorder=-4)
    ax.axvspan(*cue2, color=cue_palette[second_side], alpha=.25, zorder=-3)
    ax.add_patch(Rectangle((cue1[0], 10.2), cue, .58,
                           facecolor=cue_palette[first_side], edgecolor="none"))
    ax.add_patch(Rectangle((delay_span[0], 10.2), delay, .58, facecolor="#eeeeee", edgecolor="none"))
    ax.add_patch(Rectangle((cue2[0], 10.2), cue, .58,
                           facecolor=cue_palette[second_side], edgecolor="none"))
    for x, label in [(sum(cue1) / 2, f"cue 1 ({first_identity})"),
                     (sum(delay_span) / 2, "delay"),
                     (sum(cue2) / 2, f"cue 2 ({second_identity})")]:
        ax.text(x, 10.49, label, ha="center", va="center", fontsize=6.3)
    ax.set(xlim=(0, times[-1] + dt), ylim=(-.7, 11.05), yticks=[], xlabel="Time (ms)")
    ax.xaxis.set_label_coords(.50, -.075)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.set_ylabel("Hidden unit", fontsize=6.7, labelpad=1)
    ax.text(-.15, 1.055, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    for row, unit in enumerate(units):
        zoom.vlines(times[natural[:, unit] & ~forced[:, unit]], row - .3, row + .3, color="#2878b8", lw=.65)
        zoom.vlines(times[forced[:, unit]], row - .3, row + .3, color="#d43d3d", lw=1.0)
    zoom.set(xlim=(cue2[0] - 10, cue2[1] + 10), ylim=(-.6, 9.6), yticks=[0, 4, 9],
             yticklabels=["1", "5", "10"])
    zoom.tick_params(labelsize=6.0, length=1, pad=1)
    zoom.set_ylabel("neuron", fontsize=6.0, labelpad=1)
    zoom.set_title(f"Cue 2 ({second_identity}) zoom", fontsize=6.3, loc="left", pad=1)

    logits = trace["logits"][display_mask]
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    probax.plot(times, prob[:, 1], color="#2d7f5e", lw=.9, label="match")
    probax.plot(times, prob[:, 0], color="#555555", lw=.85, label="non-match")
    probax.set(xlim=(0, times[-1] + dt), ylim=(0, 1.02), yticks=[0, 1], xticks=[])
    probax.tick_params(labelsize=6.0, length=1, pad=1)
    probax.set_title("Readout probability", fontsize=6.3, loc="left", pad=1)
    probax.legend(loc="upper right", fontsize=5.8, frameon=True, facecolor="white",
                  framealpha=.55, edgecolor="none", handlelength=1.25,
                  handletextpad=.35, borderpad=.25, labelspacing=.2)


def plot_accuracy(ax, df):
    k_vals = sorted(df["intervention_count"].unique())
    x = np.arange(len(k_vals))
    ax.axvspan(x[k_vals.index(16)] - .5, x[-1] + .5, color="#f4f4f4", zorder=-4)
    ax.axhline(.5, color="#9e9e9e", ls=":", lw=.7, zorder=0)
    order = ["lif", "brf", "sprif_merged", "sprif_lambda0", "asrnn", "sprif_full"]
    handles = {}
    for model in order:
        mu, sd = per_k_seed_stats(df, model, k_vals)
        emph = model in {"sprif_full", "asrnn"}
        ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=COLORS[model], alpha=.075, lw=0)
        h, = ax.plot(x, mu, color=COLORS[model], marker=MARKERS[model], ls=LINESTYLES[model],
                     ms=4.2 if emph else 3.7, lw=1.55 if emph else .95,
                     markerfacecolor="white" if model == "sprif_lambda0" else COLORS[model],
                     markeredgewidth=.9, zorder=4 if emph else 2, label=LABELS[model])
        handles[model] = h
    ax.set(xlim=(-.35, x[-1] + .35), ylim=(.45, 1.015), xticks=x, xticklabels=k_vals,
           yticks=[.5, .6, .7, .8, .9, 1.0], ylabel="Accuracy", xlabel="Intervention count $K$")
    ax.grid(axis="y", ls=":", lw=.55, color="#d5d5d5")
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.text(-.15, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    return handles


def plot_heatmaps(fig, gs, df, cax):
    order = ["sprif_full", "asrnn", "sprif_lambda0", "sprif_merged", "brf", "lif"]
    delays = sorted(df["delay_ms"].unique()); k_vals = sorted(df["intervention_count"].unique())
    cmap = mpl.colormaps["YlGnBu"].copy(); norm = mpl.colors.Normalize(.5, 1.0)
    image = None
    for idx, model in enumerate(order):
        r, c = divmod(idx, 2); ax = fig.add_subplot(gs[r, c])
        pivot = (df[df["model"] == model].groupby(["delay_ms", "intervention_count"])["accuracy"]
                 .mean().unstack().reindex(index=delays, columns=k_vals))
        image = ax.imshow(pivot, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set(xticks=np.arange(len(k_vals)), yticks=np.arange(len(delays)))
        ax.set_xticklabels(k_vals if r == 2 else [], fontsize=6.2)
        ax.set_yticklabels(delays if c == 0 else [], fontsize=6.2)
        ax.tick_params(length=1.2, pad=1)
        ax.set_title(LABELS[model], loc="left", fontsize=7.0, pad=1.0,
                     color=COLORS[model], fontweight="bold" if model in {"sprif_full", "asrnn"} else "normal")
        if c == 0: ax.set_ylabel("Delay (ms)", fontsize=6.3, labelpad=1)
        if r == 2: ax.set_xlabel("$K$", fontsize=6.5)
        if idx == 0: ax.text(-.22, 1.10, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    bar = fig.colorbar(image, cax=cax, ticks=[.5, .6, .7, .8, .9, 1.0])
    bar.set_label("Accuracy", fontsize=6.7); bar.ax.tick_params(labelsize=6.2, length=2, pad=1); bar.outline.set_linewidth(.5)


def plot_fraction_response(df):
    """Preserved supplementary response plot at unseen K=40."""
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    fractions = sorted(df.loc[df["intervention_count"] == FRACTION_RESPONSE_K, "intervention_fraction"].unique())
    labels, x = ["0"] + [f"{int(round(f * 100))}" for f in fractions], np.arange(len(fractions) + 1)
    for model in ["sprif_lambda0", "sprif_full", "sprif_merged"] + EXTERNAL_MODELS:
        sub = df[df["model"] == model]; values = [sub[sub["intervention_count"] == 0].groupby("seed")["accuracy"].mean()]
        values += [sub[(sub["intervention_count"] == 40) & np.isclose(sub["intervention_fraction"], f)].groupby("seed")["accuracy"].mean() for f in fractions]
        mu = np.array([v.mean() for v in values]); sd = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in values])
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[model], alpha=.055, lw=0)
        ax.plot(x, mu, color=COLORS[model], marker=MARKERS[model], ls=LINESTYLES[model], lw=1.5, ms=4.8,
                markerfacecolor="white" if model == "sprif_lambda0" else COLORS[model], label=LABELS[model])
    ax.axhline(.5, color="#9e9e9e", ls=":", lw=.8); ax.set(ylim=(.38, 1.015), yticks=np.arange(.4, 1.01, .1), xticks=x, xticklabels=labels,
        xlabel=r"Intervened neurons per event (%) at $K=40$", ylabel="Accuracy")
    ax.grid(axis="y", ls=":", color="#d9d9d9", lw=.6)
    handles, labels = ax.get_legend_handles_labels(); fig.legend(handles, labels, ncol=3, fontsize=7.2, loc="upper center", bbox_to_anchor=(.5, .995))
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.82)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"si_dms_fraction_response.{suffix}", bbox_inches="tight", **({"dpi": 600} if suffix == "png" else {}))
    plt.close(fig)


def main():
    all_df = load_data(); primary = select_primary_slice(all_df)
    clean_trace = replay_trial(intervention_count=0)
    stressed_trace = replay_trial(intervention_count=8)
    fig = plt.figure(figsize=(7.0, 4.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[.44, .56], left=.065, right=.965, top=.95, bottom=.14, wspace=.25)
    left = outer[0, 0].subgridspec(3, 1, height_ratios=[1.18, .44, .96], hspace=LEFT_COLUMN_HSPACE)
    trial_grid = left[0].subgridspec(2, 1, hspace=.08)
    clean_ax = fig.add_subplot(trial_grid[0])
    stress_ax = fig.add_subplot(trial_grid[1], sharex=clean_ax)
    details = left[1].subgridspec(1, 2, width_ratios=[1, 1.08], wspace=.36)
    activity_ax = fig.add_subplot(details[0]); readout_ax = fig.add_subplot(details[1])
    curve_ax = fig.add_subplot(left[2])
    right = outer[0, 1].subgridspec(1, 2, width_ratios=[1, .055], wspace=.18)
    heat_grid = right[0, 0].subgridspec(3, 2, hspace=.27, wspace=.21)
    plot_paired_trial(clean_ax, stress_ax, activity_ax, readout_ax,
                      clean_trace, stressed_trace)
    handles = plot_accuracy(curve_ax, primary)
    plot_heatmaps(fig, heat_grid, primary, fig.add_subplot(right[0, 1]))
    # Matplotlib fills legend columns top-to-bottom.  This order yields the
    # visual rows full--ASRNN--lambda0 and merged--BRF--LIF.
    ordered = ["sprif_full", "sprif_merged", "asrnn", "brf", "sprif_lambda0", "lif"]
    fig.legend([handles[m] for m in ordered], [LABELS[m] for m in ordered], ncol=MODEL_LEGEND_NCOL,
               loc="lower center", bbox_to_anchor=(MODEL_LEGEND_X, MODEL_LEGEND_Y),
               fontsize=5.35, handlelength=1.2, columnspacing=.62,
               handletextpad=.35, borderpad=.15, labelspacing=.28)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"si_dms_mechanism_composite.{suffix}", bbox_inches="tight", **({"dpi": 600} if suffix == "png" else {}))
    plt.close(fig); plot_fraction_response(all_df)
    print("Saved SI-DMS composite; paired replay shares identical A-A input; "
          "clean and K=8 both predict match.")


if __name__ == "__main__":
    main()
