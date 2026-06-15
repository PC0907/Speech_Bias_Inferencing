# ASR Fairness Eval — ali's slice (SeamlessM4T + Moonshine)

Runs each model on a **deterministic 2-hour subset** of each language, writes a
reference–prediction CSV per cell, and logs WER + CER to `summary.csv`. The
aggregator turns those into the Language × Model matrix from the planning sheet.

## What your two models can actually do

| Language | SeamlessM4T v2 | Moonshine |
|----------|:-:|:-:|
| Hindi, Tamil, Urdu, Bengali | ✅ real ASR | ❌ no model |
| Dogri, Kashmiri, Santali, Bodo | ❌ not a speech source | ❌ no model |

- **Moonshine** trains one monolingual model per language; the lineup is English +
  Arabic/Chinese/Japanese/Korean/Spanish/Ukrainian/Vietnamese. **None of your 8
  languages exist.** Every Moonshine cell is a coverage gap.
- **SeamlessM4T v2** does ASR for ~96 languages incl. Hindi/Tamil/Urdu/Bengali,
  but Dogri/Kashmiri/Santali/Bodo are not speech sources.

So you have **4 genuinely scoreable cells** (Seamless × Hi/Ta/Ur/Bn). The other 12
are coverage gaps — which is a finding, not a failure.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# scored cells (real numbers)
for L in hindi tamil urdu bengali; do
    python asr_eval.py --model seamless --lang $L --hours 2 --out results
done

# coverage gaps — logged as "unsupported" automatically, no GPU spent
for L in dogri kashmiri santali bodo; do
    python asr_eval.py --model seamless --lang $L --out results
done
for L in hindi tamil urdu bengali dogri kashmiri santali bodo; do
    python asr_eval.py --model moonshine --lang $L --out results
done

# OPTIONAL: force English Moonshine on the audio to record the garbage WER
# (good supplementary "what if you misapply an English model" data point)
python asr_eval.py --model moonshine --lang hindi --out results --force
```

Build the matrices once everyone's `summary.csv` files are collected:

```bash
python aggregate.py results/summary.csv [teammates...] --metric WER --out matrix_wer.csv
python aggregate.py results/summary.csv [teammates...] --metric CER --out matrix_cer.csv
```

## Three things to get right (they affect the paper, not just the code)

1. **Same 2-hour subset for every model.** Selection is deterministic (rows in
   dataset order until ≥2h), so your Seamless and the team's Whisper/MMS all score
   the *identical* clips. Don't shuffle — the comparison breaks otherwise.
2. **One normalization scheme everywhere.** `normalize()` does NFC + punctuation
   strip + whitespace collapse + lowercase, applied identically to every language.
   Agree on this with the team so numbers are comparable. Cross-script garbage
   (e.g. English output vs Devanagari reference) will read ~100% WER — that's
   correct, not a bug.
3. **WER is not clipped** and can exceed 100% on the coverage gaps. Report those as
   coverage, not as a quality gradient. Your real RQ1 signal lives in the 4 scored
   cells (and, across the full team, mainly in MMS, which is the only model that
   reaches the tribal tier at all).

## Where to plug in
- Dataset repo names live in `LANGS` in `asr_eval.py` — fix any that differ.
- If a dataset's text column isn't auto-detected, add its name to `TEXT_KEYS`.
- Verify the run with no GPU/network: `python asr_eval.py --selftest`.
