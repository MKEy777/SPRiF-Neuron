from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "clean": "#222222",
    "forced_no_reset": "#7f7f7f",
    "fast_reset": "#1f77b4",
    "slow_reset": "#d62728",
    "both_reset": "#9467bd",
    "native_reset": "#ff7f0e",
}


def sensitivity_rows(summary: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Select only the requested sweep plus its matched main-setting anchor."""
    base = summary[
        (summary["model"] == "sprif")
        & summary["mode"].isin(["fast_reset", "slow_reset"])
    ]
    if sweep == "frequency_all":
        return base[base["sweep"].isin(["id_frequency", "ood_frequency"])].copy()
    selected = base[base["sweep"] == sweep]
    if sweep not in {"k_sweep", "gamma_sweep"} or selected.empty:
        return selected.copy()
    event_step = selected["event_step"].iloc[0]
    anchor = base[(base["sweep"] == "main") & (base["event_step"] == event_step)]
    return pd.concat((anchor, selected), ignore_index=True)


def _phase_error(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.arctan2(output[:, 1], output[:, 0]) - np.arctan2(target[:, 1], target[:, 0])
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def _save(fig, root: Path, stem: str) -> Path:
    png = root / f"{stem}.png"
    pdf = root / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def _first_trace(root: Path):
    path = root / "representative_traces.npz"
    return np.load(path) if path.exists() else None


def _main_figure(root: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> Path:
    data = _first_trace(root)
    if data is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))
        for ax in axes.flat:
            ax.text(0.5, 0.5, "task not solved; causal panel suppressed",
                    ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return _save(fig, root, "main_causal_figure")
    target = data["target"]
    event_step = int(data["event_step"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))

    ax = axes[0, 0]
    ax.axvspan(0, event_step, color="#e8eef7", alpha=0.8)
    ax.axvline(event_step, color="#d62728", linestyle="--")
    ax.text(event_step, 0.85, "intervention", rotation=90, va="top", ha="right")
    ax.set(xlim=(0, len(target) - 1), ylim=(0, 1), yticks=[], title="(a) Cue-delay intervention")
    ax.set_xlabel("time step")

    ax = axes[0, 1]
    ax.plot(target[:, 0], target[:, 1], color="black", linewidth=2, label="target")
    for mode in ("fast_reset", "slow_reset"):
        key = f"{mode}_output"
        if key in data:
            output = data[key]
            ax.plot(output[:, 0], output[:, 1], color=COLORS[mode], label=mode)
    ax.set(title="(b) Output phase trajectory", xlabel="cos", ylabel="sin", aspect="equal")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    window = np.arange(max(0, event_step - 5), min(len(target), event_step + 16))
    for mode in ("forced_no_reset", "fast_reset", "slow_reset", "both_reset"):
        key = f"{mode}_output"
        if key in data:
            error = _phase_error(data[key], target)
            ax.plot(window - event_step, error[window], color=COLORS[mode], label=mode)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title="(c) Event-aligned phase error", xlabel="steps from event", ylabel="rad")

    ax = axes[1, 0]
    if not paired.empty:
        by_seed = paired.groupby("seed")["slow_minus_fast_auc"].mean()
        ax.scatter(by_seed.index, by_seed.values, color="#4c78a8")
        ax.axhline(0, color="black", linewidth=1)
    else:
        ax.text(0.5, 0.5, "paired effects unavailable", ha="center", va="center")
    ax.set(title="(d) Paired slow-fast effect", xlabel="seed", ylabel="AUC difference")

    ax = axes[1, 1]
    sensitivity = sensitivity_rows(summary, "k_sweep")
    if not sensitivity.empty and "excess_auc_mean" in sensitivity:
        for mode, group in sensitivity.groupby("mode"):
            ordered = group.groupby("k", as_index=False)["excess_auc_mean"].mean().sort_values("k")
            ax.plot(ordered["k"], ordered["excess_auc_mean"], marker="o",
                    color=COLORS[mode], label=mode)
        ax.legend(fontsize=8)
    ax.set(title="(e) Intervention-size response", xlabel="K", ylabel="excess-error AUC")

    ax = axes[1, 2]
    residuals = [
        float(data["fast_reset_residual_norm"]),
        float(data["slow_reset_residual_norm"]),
    ]
    ax.bar(["fast", "slow"], residuals, color=[COLORS["fast_reset"], COLORS["slow_reset"]])
    ax.set(title="(f) Matched reset residuals", ylabel="state-space norm")
    fig.tight_layout()
    return _save(fig, root, "main_causal_figure")


def _baseline_figure(root: Path, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    subset = summary[
        (summary["model"] != "sprif")
        & (summary["sweep"] == "main")
        & summary["mode"].isin(["clean", "native_reset"])
    ]
    if subset.empty:
        ax.text(0.5, 0.5, "External baselines not evaluated", ha="center", va="center")
    else:
        table = subset.groupby(["model", "mode"])["delay_mse_mean"].mean().unstack()
        table.plot(kind="bar", ax=ax, color=[COLORS.get(column, "gray") for column in table.columns])
        ax.legend(title="mode")
    ax.set(title="Baseline clean and native-reset performance", ylabel="delay MSE", xlabel="model")
    fig.tight_layout()
    return _save(fig, root, "baseline_robustness")


def _sensitivity_figure(root: Path, summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    for ax, sweep, x_column, title in (
        (axes[0, 0], "k_sweep", "k", "K sensitivity"),
        (axes[0, 1], "gamma_sweep", "gamma", "Reset-strength sensitivity"),
        (axes[0, 2], "multi_event", "event_count", "Repeated-reset stress"),
        (axes[1, 0], "frequency_all", "frequency_period_mean", "Seen vs unseen-frequency periods"),
    ):
        subset = sensitivity_rows(summary, sweep)
        if subset.empty or x_column not in subset:
            ax.text(0.5, 0.5, "not evaluated", ha="center", va="center")
        else:
            group_columns = ["mode", "sweep"] if sweep == "frequency_all" else ["mode"]
            for group_key, group in subset.groupby(group_columns):
                if sweep == "frequency_all":
                    mode, frequency_sweep = group_key
                    domain = "seen" if frequency_sweep == "id_frequency" else "unseen"
                    label = f"{mode} ({domain})"
                    linestyle = "-" if domain == "seen" else "--"
                else:
                    mode = group_key[0]
                    label = mode
                    linestyle = "-"
                ordered = group.groupby(x_column, as_index=False)["excess_auc_mean"].mean()
                ordered = ordered.sort_values(x_column)
                ax.plot(ordered[x_column], ordered["excess_auc_mean"], marker="o",
                        color=COLORS[mode], linestyle=linestyle, label=label)
            if ax.lines:
                ax.legend(fontsize=8)
        ax.set(title=title, xlabel=x_column, ylabel="excess-error AUC")

    ax = axes[1, 1]
    data = _first_trace(root)
    if data is None:
        ax.text(0.5, 0.5, "not available: clean gate failed", ha="center", va="center")
        ax.set(title="Representative hard case")
    else:
        target = data["failure_target"]
        event_step = int(data["event_step"])
        window = np.arange(max(0, event_step - 5), min(len(target), event_step + 31))
        for mode in ("fast_reset", "slow_reset"):
            output = data[f"failure_{mode}_output"]
            error = _phase_error(output, target)
            ax.plot(window - event_step, error[window], color=COLORS[mode], label=mode)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title="Representative hard case", xlabel="steps from event", ylabel="phase error (rad)")
        ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    fig.tight_layout()
    return _save(fig, root, "appendix_sensitivity")


def plot_results(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    summary = pd.read_csv(root / "summary.csv")
    paired_path = root / "paired_effects.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() and paired_path.stat().st_size else pd.DataFrame()
    return {
        "main": _main_figure(root, summary, paired),
        "baselines": _baseline_figure(root, summary),
        "sensitivity": _sensitivity_figure(root, summary),
    }
