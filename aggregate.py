#!/usr/bin/env python3
"""
Build Ali's Language x Model matrix from results/summary.csv.

    python aggregate.py results/summary.csv --metric WER --out matrix_wer.csv
    python aggregate.py results/summary.csv --metric CER --out matrix_cer.csv

Unsupported cells are written as "n/a" so coverage gaps stay distinct from scores.
"""
import argparse, csv, collections

MODEL_ORDER = ["seamless", "owsm", "indicconformer"]   # Ali's slice
LANG_ORDER  = ["hindi", "tamil", "urdu", "bengali",
               "dogri", "kashmiri", "santali", "bodo"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--metric", choices=["WER", "CER"], default="WER")
    ap.add_argument("--out", default="matrix.csv")
    args = ap.parse_args()

    cell = collections.defaultdict(dict)
    seen = set()
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m, lg = r["model"].lower(), r["language"].lower()
                seen.add(m)
                cell[lg][m] = "n/a" if r["status"] == "unsupported" else r[args.metric]

    models = [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))
    langs  = [l for l in LANG_ORDER if l in cell] + sorted(set(cell) - set(LANG_ORDER))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language \\ model"] + models)
        for lg in langs:
            w.writerow([lg] + [cell[lg].get(m, "") for m in models])
    print(f"Wrote {args.metric} matrix ({len(langs)}x{len(models)}) -> {args.out}")

if __name__ == "__main__":
    main()
