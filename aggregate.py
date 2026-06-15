#!/usr/bin/env python3
"""
Build the Language x Model matrix from one or more summary.csv files.

    python aggregate.py results/summary.csv [teammate1.csv teammate2.csv ...] \
           --metric WER --out matrix_wer.csv

Unsupported cells are written as "n/a" so coverage gaps stay visually distinct
from real scores. Pass --metric CER for the character-error matrix.
"""
import argparse, csv, collections

MODEL_ORDER = ["parakeet", "nemotron", "sensevoice", "mms",
               "moonshine", "seamless", "speecht5", "whisper"]
LANG_ORDER  = ["hindi", "tamil", "urdu", "bengali",
               "dogri", "kashmiri", "santali", "bodo"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--metric", choices=["WER", "CER"], default="WER")
    ap.add_argument("--out", default="matrix.csv")
    args = ap.parse_args()

    cell = collections.defaultdict(dict)        # cell[lang][model] = value or "n/a"
    models_seen = set()
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m, lg = r["model"].lower(), r["language"].lower()
                models_seen.add(m)
                cell[lg][m] = "n/a" if r["status"] == "unsupported" else r[args.metric]

    models = [m for m in MODEL_ORDER if m in models_seen] + \
             sorted(models_seen - set(MODEL_ORDER))
    langs  = [l for l in LANG_ORDER if l in cell] + \
             sorted(set(cell) - set(LANG_ORDER))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language \\ model"] + models)
        for lg in langs:
            w.writerow([lg] + [cell[lg].get(m, "") for m in models])
    print(f"Wrote {args.metric} matrix ({len(langs)}x{len(models)}) -> {args.out}")

if __name__ == "__main__":
    main()
