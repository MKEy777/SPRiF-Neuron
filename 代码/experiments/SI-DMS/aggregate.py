import argparse, csv, json
from pathlib import Path
from sidms.reporting import summarize, summarize_curve

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default="results/all_metrics.json")
    p.add_argument("--output", default="results/summary.csv")
    p.add_argument("--curve", help="额外输出 fraction-response 曲线 CSV")
    p.add_argument("--fraction", type=float, default=0.15)
    args = p.parse_args()
    path = Path(args.input)
    if path.is_dir():
        all_rows = []
        for f in sorted(path.rglob("eval_metrics.json")):
            all_rows += json.loads(f.read_text(encoding="utf-8"))
    else:
        all_rows = json.loads(path.read_text(encoding="utf-8"))
    rows = summarize(all_rows, args.fraction)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    if args.curve:
        crows = summarize_curve(all_rows)
        cout = Path(args.curve); cout.parent.mkdir(parents=True, exist_ok=True)
        with cout.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=crows[0]); w.writeheader(); w.writerows(crows)

if __name__ == "__main__": main()

