"""Original SI-DMS panel geometry, adapted only for the current model roster."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
RASTER_MAX_DELAY_OCCUPANCY = .40
LEFT_COLUMN_HSPACE = .24
MODEL_LEGEND_NCOL = 3
MODEL_LEGEND_X = .27
MODEL_LEGEND_Y = .008
COLORS = {'SPRiF':'#1f77b4','MEMORY_RESET5':'#d62728','REAL_MODES3':'#ff7f0e','ASRNN':'#4b4b4b','BRF':'#7b6ba8','LIF':'#4f8f83'}
MARKERS = {'SPRiF':'o','MEMORY_RESET5':'v','REAL_MODES3':'s','ASRNN':'P','BRF':'X','LIF':'D'}
LINESTYLES = {m:('-' if m in {'SPRiF','MEMORY_RESET5','REAL_MODES3'} else '--') for m in COLORS}
LABELS = {m:m for m in COLORS}


def _time_axis(trace):
    return np.arange(trace["spikes"].shape[0]) * trace["dt"]

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

def per_k_seed_stats(df, model, k_vals):
    """Average delays within a seed, then calculate cross-seed mean ± SD."""
    sub = df[df["model"] == model]
    means, stds = [], []
    for k in k_vals:
        per_seed = sub[sub["intervention_count"] == k].groupby("seed")["accuracy"].mean()
        means.append(per_seed.mean())
        stds.append(per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0)
    return np.asarray(means), np.asarray(stds)

def plot_accuracy(ax, df):
    k_vals = sorted(df["intervention_count"].unique())
    x = np.arange(len(k_vals))
    ax.axvspan(x[k_vals.index(16)] - .5, x[-1] + .5, color="#f4f4f4", zorder=-4)
    ax.axhline(.5, color="#9e9e9e", ls=":", lw=.7, zorder=0)
    order = ["LIF", "BRF", "MEMORY_RESET5", "REAL_MODES3", "ASRNN", "SPRiF"]
    handles = {}
    for model in order:
        mu, sd = per_k_seed_stats(df, model, k_vals)
        emph = model in {"SPRiF", "ASRNN"}
        ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=COLORS[model], alpha=.075, lw=0)
        h, = ax.plot(x, mu, color=COLORS[model], marker=MARKERS[model], ls=LINESTYLES[model],
                     ms=4.2 if emph else 3.7, lw=1.55 if emph else .95,
                     markerfacecolor="white" if model == "REAL_MODES3" else COLORS[model],
                     markeredgewidth=.9, zorder=4 if emph else 2, label=LABELS[model])
        handles[model] = h
    ax.set(xlim=(-.35, x[-1] + .35), ylim=(.45, 1.015), xticks=x, xticklabels=k_vals,
           yticks=[.5, .6, .7, .8, .9, 1.0], ylabel="Accuracy", xlabel="Intervention count $K$")
    ax.grid(axis="y", ls=":", lw=.55, color="#d5d5d5")
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.text(-.15, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    return handles

def plot_heatmaps(fig, gs, df, cax):
    order = ["SPRiF", "ASRNN", "REAL_MODES3", "MEMORY_RESET5", "BRF", "LIF"]
    delays = sorted(df["delay_ms"].unique()); k_vals = sorted(df["intervention_count"].unique())
    cmap = mpl.colormaps["YlGnBu"].copy(); norm = mpl.colors.Normalize(.5, 1.0)
    image = None
    for idx, model in enumerate(order):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c])
        pivot = (df[df["model"] == model].groupby(["delay_ms", "intervention_count"])["accuracy"]
                 .mean().unstack().reindex(index=delays, columns=k_vals))
        image = ax.imshow(pivot, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set(xticks=np.arange(len(k_vals)), yticks=np.arange(len(delays)))
        ax.set_xticklabels(k_vals if r == 2 else [], fontsize=6.2)
        ax.set_yticklabels(delays if c == 0 else [], fontsize=6.2)
        ax.tick_params(length=1.2, pad=1)
        ax.set_title(LABELS[model], loc="left", fontsize=7.0, pad=1.0,
                     color=COLORS[model], fontweight="bold" if model in {"SPRiF", "ASRNN"} else "normal")
        if c == 0: ax.set_ylabel("Delay (ms)", fontsize=6.3, labelpad=1)
        if r == 2: ax.set_xlabel("$K$", fontsize=6.5)
        if idx == 0: ax.text(-.22, 1.10, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    bar = fig.colorbar(image, cax=cax, ticks=[.5, .6, .7, .8, .9, 1.0])
    bar.set_label("Accuracy", fontsize=6.7); bar.ax.tick_params(labelsize=6.2, length=2, pad=1); bar.outline.set_linewidth(.5)
