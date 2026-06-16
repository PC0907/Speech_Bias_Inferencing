# ASR Eval — Ali's slice

Models: **seamless | owsm | indicconformer** (Moonshine retired — English-only,
covered none of the 8). Deterministic 2-hour subset per language; writes a
reference–prediction CSV per cell and logs WER + CER to `summary.csv`.

## Coverage

| Language | SeamlessM4T v2 | OWSM-CTC | IndicConformer |
|----------|:-:|:-:|:-:|
| Hindi, Tamil, Urdu, Bengali | ✅ | ✅ | ✅ |
| Dogri, Kashmiri, Santali, Bodo | ❌ | ❌ | ✅ |

- **Seamless / OWSM** → real numbers on Hi/Ta/Ur/Bn; the tribal/scheduled four
  are coverage gaps (auto-logged `unsupported`). Both are broad-multilingual
  *foundation* models; OWSM is the open, reproducible one (ESPnet).
- **IndicConformer** → all eight. Indic-*specialist*, so keep it in its own
  group in the paper (upper bound), not lumped with the foundation models.

## Install

```bash
pip install -r requirements.txt
huggingface-cli login          # for the gated IndicConformer (after accepting terms)
```

ESPnet (for OWSM) is a heavy install. If it fights the rest of the stack, install
it last, or run OWSM in a separate session: `pip install espnet espnet_model_zoo librosa`.

## Run

```bash
# IndicConformer — full 8-language column
for L in hindi tamil urdu bengali dogri kashmiri santali bodo; do
    python asr_eval.py --model indicconformer --lang $L --out results
done
# Seamless + OWSM — real on the four majors, auto coverage-gap on the rest
for M in seamless owsm; do
  for L in hindi tamil urdu bengali; do
      python asr_eval.py --model $M --lang $L --out results
  done
  for L in dogri kashmiri santali bodo; do
      python asr_eval.py --model $M --lang $L --out results
  done
done

python aggregate.py results/summary.csv --metric WER --out matrix_wer.csv
python aggregate.py results/summary.csv --metric CER --out matrix_cer.csv
```

## Notes

- Defaults: `--split valid`, `--text-field normalized`. Switch refs with
  `--text-field verbatim`; `--decoder rnnt` for a stronger IndicConformer number.
- `--data` takes a local `load_from_disk` folder or a HF hub id; omit it to use
  the sheet's hub id.
- OWSM model tag is `OWSM_MODEL` at the top of `asr_eval.py`
  (`espnet/owsm_ctc_v3.1_1B`; swap to `espnet/owsm_ctc_v4_1B` for the newer one).
- Text is RAW (NFC + whitespace only); raw `reference`/`prediction` columns are
  always saved so you can re-score under any scheme later.
- WER isn't clipped and can exceed 100% on coverage gaps — that's correct.
- Verify offline: `python asr_eval.py --selftest`.
