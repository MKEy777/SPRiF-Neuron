import argparse, csv, json
from pathlib import Path
from sidms.reporting import summarize


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default="results/all_metrics.json")
    p.add_argument("--output", default="results/summary.csv"); args = p.parse_args()
    rows = summarize(json.loads(Path(args.input).read_text(encoding="utf-8")))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
