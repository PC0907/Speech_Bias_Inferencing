# ASR Eval — Ali's slice

Three models, eight languages, a deterministic 2-hour subset each. Writes a
reference–prediction CSV per cell and logs WER + CER to `summary.csv`.

## Coverage (what each model can actually do)

| Language | SeamlessM4T v2 | Moonshine | IndicConformer |
|----------|:-:|:-:|:-:|
| Hindi, Tamil, Urdu, Bengali | ✅ | ❌ no model | ✅ |
| Dogri, Kashmiri, Santali, Bodo | ❌ not a speech source | ❌ no model | ✅ |

- **Seamless** → real numbers only on Hi/Ta/Ur/Bn; the other four are coverage gaps.
- **Moonshine** → no model exists for any of the eight; every cell is a gap
  (use `--force` to run the English model anyway and record the garbage WER).
- **IndicConformer** → covers all eight (it's trained on the 22 scheduled
  languages). It is *not* a foundation model, though — see the note at the bottom.

## Text handling is RAW

No lowercasing, no punctuation stripping — nothing that could touch script-specific
cues. Only two things are applied, and neither removes content:

- **NFC** — canonical Unicode form, so identical-looking strings encoded with
  different code-point orders don't count as errors (this *protects* the
  low-resource scripts from false errors). Disable with `--byte-exact`.
- **whitespace collapse** — tokenisation hygiene; stops stray double/trailing
  spaces from inventing word errors.

The raw `reference` and `prediction` columns are always saved untouched, so you can
recompute under any scheme later without re-running inference.

## Install

```bash
pip install -r requirements.txt
# IndicConformer is gated:
huggingface-cli login        # after accepting the terms on its HF model page
```

## Run

```bash
# IndicConformer — the only model that gives you a full 8-language column
for L in hindi tamil urdu bengali dogri kashmiri santali bodo; do
    python asr_eval.py --model indicconformer --lang $L \
        --data /content/drive/MyDrive/test_$L --decoder ctc --out results
done

# Seamless — real on the first four, auto coverage-gap on the rest
for L in hindi tamil urdu bengali; do
    python asr_eval.py --model seamless --lang $L --data /content/drive/MyDrive/test_$L --out results
done
for L in dogri kashmiri santali bodo; do
    python asr_eval.py --model seamless --lang $L --out results   # logged unsupported, no GPU spent
done

# Moonshine — all eight are coverage gaps
for L in hindi tamil urdu bengali dogri kashmiri santali bodo; do
    python asr_eval.py --model moonshine --lang $L --out results
done
```

`--data` takes either a local `load_from_disk` folder (your Drive copies) or a HF
hub id (`XKaab/ASR-Hindi_7hrs`). Omit it to fall back to the sheet's hub id.

Build the matrices:

```bash
python aggregate.py results/summary.csv --metric WER --out matrix_wer.csv
python aggregate.py results/summary.csv --metric CER --out matrix_cer.csv
```

## Notes

- `--decoder rnnt` is usually a touch more accurate than `ctc`, a touch slower.
- If IndicConformer's custom forward throws a device error, run it on CPU
  (600M is slow but fine for 2 h) — remove the GPU; the adapter already falls back.
- Verify the logic with no GPU/network: `python asr_eval.py --selftest`.
- WER is not clipped and can exceed 100% on coverage gaps — that's correct;
  report those as coverage, not as a quality gradient.

**On IndicConformer and the paper:** it's a purpose-built, supervised Indic model,
not a web-scale foundation model like the others. It'll likely score *well* on all
eight — which makes it a counterpoint to the "foundation models exclude tribal
languages" thesis, not a test of it. Report it as a clearly-labelled
Indic-specialized reference / upper bound in its own row, not mixed into the
foundation-model group.
