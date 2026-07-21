import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from sidms.config import ExperimentConfig
from sidms.data import make_batch
from sidms.models import SIDMSNetwork

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", required=True, help="<model>/seed_x/checkpoint.pt")
    p.add_argument("--output", default="results/figures/raster.png")
    p.add_argument("--delay", type=int, default=1600)
    p.add_argument("--count", type=int, default=8, help="干预时刻数 K")
    p.add_argument("--fraction", type=float, default=0.15)
    p.add_argument("--sample", type=int, default=-1, help="batch 内样本索引（-1 自动选择 match 试次）")
    args = p.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    name = ckpt.get("name", "model")
    model = SIDMSNetwork(name, cfg)
    model.load_state_dict(ckpt["model"]); model.eval()

    batch = make_batch(cfg, batch_size=8, delay_ms=args.delay, intervention_count=args.count,
                       hidden_size=cfg.model.hidden_size, seed=0, fraction=args.fraction)
    with torch.no_grad():
        out = model(batch.x, batch.intervention)

    if args.sample >= 0:
        s = args.sample
    else:
        match_idx = (batch.y == 1).nonzero(as_tuple=True)[0]
        s = int(match_idx[0]) if len(match_idx) else 0
        print(f"auto-selected sample {s} (match={int(batch.y[s])})")

    spikes = out.spikes[s].numpy()
    forced = out.forced_hits[s].numpy()
    interv = batch.intervention[s].numpy()
    steps, hidden = spikes.shape

    dt = cfg.task.dt_ms
    pre = cfg.task.pre_ms // dt
    cue = cfg.task.cue_ms // dt
    delay = args.delay // dt

    T1 = pre; T2 = pre + cue; T3 = pre + cue + delay; T4 = pre + cue + delay + cue

    first_side = int(batch.first_side[s])
    second_side = int(batch.second_side[s])

    fig, (ax_in, ax) = plt.subplots(2, 1, figsize=(11, 5.5),
                                     gridspec_kw=dict(height_ratios=[1, 4], hspace=0.08))

    input_img = ax_in.imshow(
        batch.x[s].T, aspect="auto", cmap="gray_r", interpolation="nearest",
        vmin=0, vmax=1
    )
    for xpos in [T1, T2, T3, T4]:
        ax_in.axvline(xpos, color="gray", ls="--", lw=0.6)
    ax_in.set_ylabel("input\nch.", fontsize=7)
    ax_in.set_yticks([])
    ax_in.tick_params(axis="x", labelbottom=False)
    ax_in.set_xlim(0, steps)

    n_input = batch.x.shape[-1]
    n_cue = cfg.task.cue_channels
    ch_cue0 = list(range(n_cue))
    ch_cue1 = list(range(n_cue, min(2 * n_cue, n_input)))
    for ch in ch_cue0:
        ax_in.text(steps * 1.01, ch, "A", fontsize=5, va="center", color="blue")
    for ch in ch_cue1:
        ax_in.text(steps * 1.01, ch, "B", fontsize=5, va="center", color="orange")
    is_match = int(batch.y[s])
    ax_in.text(0.5, -0.45, f"cue1 = {'A' if first_side == 0 else 'B'}, "
               f"cue2 = {'A' if second_side == 0 else 'B'}",
               transform=ax_in.transAxes, ha="center", fontsize=8,
               color="green" if first_side == second_side else "red",
               fontweight="bold")

    nat_t, nat_n = ((spikes > 0.5) & (forced < 0.5)).nonzero()
    ax.scatter(nat_t, nat_n, s=4, c="tab:blue", marker="|", label="natural spike")
    frc_t, frc_n = (forced > 0.5).nonzero()
    ax.scatter(frc_t, frc_n, s=18, c="tab:red", marker="|", label="forced spike")

    interv_times = sorted({int(t) for t, _ in zip(*interv.nonzero())})
    for t in interv_times:
        ax.axvline(t, color="tab:red", alpha=0.12, lw=1)

    colors = {"pre": "#f5f5f5", "cue1": "#e3f2fd", "delay": "#fff3e0", "cue2": "#e3f2fd"}
    phases = [
        (0, T1, "pre"),
        (T1, T2, "cue1"),
        (T2, T3, "delay"),
        (T3, T4, "cue2"),
    ]
    for start, end, lab in phases:
        ax.axvspan(start, end, facecolor=colors[lab], alpha=0.3, zorder=0)
        ax.text((start + end) / 2, hidden * 1.04, lab, ha="center", fontsize=9,
                color="#555", fontweight="bold")

    for xpos in [T1, T2, T3, T4]:
        ax.axvline(xpos, color="gray", ls="--", lw=0.8)

    ax.set_xlim(0, steps)
    ax.set_ylim(-1, hidden)
    ax.set_xlabel(f"time step (dt={dt}ms)", fontsize=9)
    ax.set_ylabel("neuron", fontsize=9)
    match_str = "MATCH" if is_match else "NON-MATCH"
    ax.set_title(
        f"{name}  |  {match_str} trial: cue1={'A' if first_side==0 else 'B'} "
        f"cue2={'A' if second_side==0 else 'B'}  "
        f"(delay={args.delay}ms, K={args.count}, frac={args.fraction})",
        fontsize=10
    )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}  (干预时刻数={len(interv_times)})")

if __name__ == "__main__":
    main()

