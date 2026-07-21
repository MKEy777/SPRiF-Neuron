import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default="results/all_metrics.json")
    p.add_argument("--output", default="results/figures"); args = p.parse_args()
    df = pd.DataFrame(json.loads(Path(args.input).read_text(encoding="utf-8")))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for model, group in df.groupby("model"):
        pivot = group.groupby(["delay_ms", "intervention_count"]).accuracy.mean().unstack()
        ax = pivot.plot(marker="o", title=f"SI-DMS robustness: {model}")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.figure.tight_layout()
        ax.figure.savefig(out / f"{model}_robustness.png", dpi=200); plt.close(ax.figure)


    # 主图:fraction-response 曲线(x=fraction, 每模型一条线,取最长delay+最大K)
    if "intervention_fraction" in df.columns:
        max_k = df.intervention_count.max(); max_d = df.delay_ms.max()
        sub = df[(df.intervention_count == max_k) & (df.delay_ms == max_d)]
        clean = df[df.intervention_count == 0].groupby("model").accuracy.mean()
        fig, ax = plt.subplots()
        for model, g in sub.groupby("model"):
            curve = g.groupby("intervention_fraction").accuracy.mean()
            xs = [0.0] + list(curve.index)
            ys = [clean.get(model, curve.iloc[0])] + list(curve.values)
            ax.plot(xs, ys, marker="o", label=model)
        ax.set_xlabel("Intervention fraction"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.0); ax.set_title(f"SI-DMS fraction-response (delay={max_d}, K={max_k})")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(out / "fraction_response.png", dpi=200); plt.close(fig)


if __name__ == "__main__": main()
