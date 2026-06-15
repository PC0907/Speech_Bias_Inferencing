#!/usr/bin/env python3
"""
ASR fairness eval harness  --  ali's slice: SeamlessM4T + Moonshine
-------------------------------------------------------------------
For each (model, language) it:
  1. streams the HF dataset, picks a DETERMINISTIC ~2h subset (shared across models),
  2. runs inference,
  3. writes a reference-prediction CSV,
  4. appends WER + CER to summary.csv.

Coverage is encoded explicitly: cells the model cannot serve are logged as
"unsupported" instead of being silently scored as ~100% WER. Use --force to run
an unsupported cell anyway (e.g. English Moonshine on Hindi audio) and record the
garbage number as a deliberate data point.

Metrics are self-contained (standard word/char Levenshtein, can exceed 1.0) so no
jiwer/torch is needed for scoring -> works on an offline HPC node.

Run a model:
    python asr_eval.py --model seamless --lang hindi --hours 2 --out results
    python asr_eval.py --model moonshine --lang hindi --hours 2 --out results --force

Verify the logic with no GPU / no network:
    python asr_eval.py --selftest
"""
import argparse, csv, json, os, re, sys, unicodedata

# ----------------------------------------------------------------------------
# Language registry. Edit the `hf` ids if your repo names differ slightly.
# `seamless` = Seamless source-speech code, or None if NOT a supported speech
# source. Moonshine has NO model for any of these 8, so it is handled separately.
# ----------------------------------------------------------------------------
LANGS = {
    "hindi":    {"hf": "XKaab/ASR-Hindi_7hrs",    "seamless": "hin", "tier": "HRL"},
    "tamil":    {"hf": "XKaab/ASR-Tamil_8hrs",    "seamless": "tam", "tier": "HRL"},
    "urdu":     {"hf": "XKaab/ASR-Urdu_6hrs",     "seamless": "urd", "tier": "MRL"},
    "bengali":  {"hf": "XKaab/ASR-Bengali_6hrs",  "seamless": "ben", "tier": "MRL"},
    "dogri":    {"hf": "XKaab/ASR-Dogri_4hrs",    "seamless": None,  "tier": "LRL-Scheduled"},
    "kashmiri": {"hf": "XKaab/ASR-Kashmiri_4hrs", "seamless": None,  "tier": "LRL-Scheduled"},
    "santali":  {"hf": "XKaab/ASR-Santali_4hrs",  "seamless": None,  "tier": "LRL-Tribal"},
    "bodo":     {"hf": "XKaab/ASR-Bodo_5hrs",     "seamless": None,  "tier": "LRL-Tribal"},
}

# Candidate column names to auto-detect inside a HF dataset row.
TEXT_KEYS  = ("sentence", "text", "transcription", "transcript", "normalized_text", "target")
AUDIO_KEYS = ("audio", "speech", "wav")

# ============================================================================
# Metrics + normalization  (no external deps -- testable offline)
# ============================================================================
_PUNCT = re.compile(r"[।॥.,;:!?\"'`~@#$%^&*()\[\]{}<>/\\|_=+\u2013\u2014\u2026“”‘’]+")
_WS    = re.compile(r"\s+")

def normalize(text, lower=True):
    """NFC -> strip punctuation (Latin + Devanagari danda etc.) -> collapse ws."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _PUNCT.sub(" ", t)
    if lower:
        t = t.lower()
    return _WS.sub(" ", t).strip()

def _lev(ref, hyp):
    """Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return float(bool(h))
    return _lev(r, h) / len(r)          # NOT clipped -- can exceed 1.0

def cer(ref, hyp):
    r, h = list(ref), list(hyp)         # chars incl. single spaces (jiwer convention)
    if not r:
        return float(bool(h))
    return _lev(r, h) / len(r)

def score(ref_raw, hyp_raw):
    r, h = normalize(ref_raw), normalize(hyp_raw)
    return wer(r, h), cer(r, h), r, h

# ============================================================================
# Deterministic 2-hour subset (shared across all models -> fair comparison)
# ============================================================================
def select_subset(examples, target_seconds, audio_key):
    """Take rows IN ORDER until accumulated audio >= target. Deterministic, so
    every model (yours and the rest of the team's) scores the identical clips."""
    selected, total = [], 0.0
    for ex in examples:
        a = ex[audio_key]
        dur = len(a["array"]) / a["sampling_rate"]
        selected.append(ex)
        total += dur
        if total >= target_seconds:
            break
    return selected, total

def detect_key(row, candidates, kind):
    for k in candidates:
        if k in row:
            return k
    raise KeyError(f"No {kind} column found. Row has: {list(row.keys())}. "
                   f"Add the right name to {'TEXT_KEYS' if kind=='text' else 'AUDIO_KEYS'}.")

# ============================================================================
# Model adapters (need torch + transformers; imported lazily so --selftest
# and offline scoring work without them)
# ============================================================================
def load_seamless(device):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    def transcribe(audio16k, tgt_lang):
        inputs = proc(audios=audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            toks = model.generate(**inputs, tgt_lang=tgt_lang)
        return proc.batch_decode(toks, skip_special_tokens=True)[0]
    return transcribe

def load_moonshine(device):
    import torch
    from transformers import AutoProcessor, MoonshineForConditionalGeneration
    # English model -- Moonshine has no Indic checkpoint. Used only with --force.
    proc = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-base").to(device).eval()

    def transcribe(audio16k, tgt_lang=None):
        inputs = proc(audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            toks = model.generate(**inputs, max_new_tokens=256)
        return proc.batch_decode(toks, skip_special_tokens=True)[0]
    return transcribe

def resample_16k(array, sr):
    if sr == 16000:
        return array
    import numpy as np
    try:
        import librosa
        return librosa.resample(np.asarray(array, dtype="float32"), orig_sr=sr, target_sr=16000)
    except ImportError:
        # linear fallback if librosa is unavailable
        x = np.asarray(array, dtype="float32")
        n = int(round(len(x) * 16000 / sr))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")

# ============================================================================
# Supportedness
# ============================================================================
def supported(model, lang):
    if model == "seamless":
        return LANGS[lang]["seamless"] is not None
    if model == "moonshine":
        return False               # no Moonshine model exists for any of the 8
    raise ValueError(model)

# ============================================================================
# Main run
# ============================================================================
def run(model_name, lang, hours, out_dir, split, force, subset_file):
    os.makedirs(out_dir, exist_ok=True)
    info = LANGS[lang]
    is_supported = supported(model_name, lang)

    if not is_supported and not force:
        _append_summary(out_dir, model_name, lang, info["tier"],
                        n=0, hrs=0.0, w="", c="", status="unsupported")
        print(f"[{model_name}/{lang}] UNSUPPORTED -> logged as coverage gap "
              f"(use --force to run anyway and record the garbage WER).")
        return

    from datasets import load_dataset
    ds = load_dataset(info["hf"], split=split, streaming=True)
    first = next(iter(ds))
    akey = detect_key(first, AUDIO_KEYS, "audio")
    tkey = detect_key(first, TEXT_KEYS, "text")
    print(f"[{model_name}/{lang}] audio col='{akey}'  text col='{tkey}'")

    # deterministic subset (reuse a saved id list if provided)
    ds = load_dataset(info["hf"], split=split, streaming=True)
    rows, total_sec = select_subset(ds, hours * 3600, akey)
    print(f"[{model_name}/{lang}] selected {len(rows)} utts  ({total_sec/3600:.2f} h)")

    device = "cuda" if _cuda() else "cpu"
    transcribe = (load_seamless if model_name == "seamless" else load_moonshine)(device)
    tgt = info["seamless"]

    rows_out, W, C = [], [], []
    for i, ex in enumerate(rows):
        a = ex[akey]
        audio = resample_16k(a["array"], a["sampling_rate"])
        try:
            hyp = transcribe(audio, tgt)
        except Exception as e:                       # unsupported lang code etc.
            hyp = ""
            print(f"  utt {i}: inference error -> empty hyp ({e})")
        ref = ex.get(tkey, "")
        w, c, rn, hn = score(ref, hyp)
        W.append(w); C.append(c)
        rows_out.append({"idx": i, "reference": ref, "prediction": hyp,
                         "ref_norm": rn, "hyp_norm": hn,
                         "wer": round(w, 4), "cer": round(c, 4)})

    pred_path = os.path.join(out_dir, f"{model_name}__{lang}.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wtr.writeheader(); wtr.writerows(rows_out)

    agg_w = sum(W) / len(W); agg_c = sum(C) / len(C)
    status = "scored" if is_supported else "forced(unsupported)"
    _append_summary(out_dir, model_name, lang, info["tier"],
                    n=len(rows_out), hrs=total_sec/3600,
                    w=round(agg_w, 4), c=round(agg_c, 4), status=status)
    print(f"[{model_name}/{lang}] WER={agg_w:.4f}  CER={agg_c:.4f}  -> {pred_path}")

def _append_summary(out_dir, model, lang, tier, n, hrs, w, c, status):
    path = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        if new:
            wtr.writerow(["model", "language", "tier", "n_utts",
                          "audio_hours", "WER", "CER", "status"])
        wtr.writerow([model, lang, tier, n, round(hrs, 3), w, c, status])

def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except ImportError:
        return False

# ============================================================================
# Self-test (no GPU, no network, no torch)
# ============================================================================
def selftest():
    assert normalize("Hello,  World!") == "hello world"
    assert normalize("नमस्ते।") == "नमस्ते"
    assert abs(wer("the cat sat", "the cat sat") - 0.0)      < 1e-9
    assert abs(wer("the cat sat", "the dog sat") - 1/3)      < 1e-9
    assert abs(wer("a b c d", "") - 1.0)                     < 1e-9
    assert abs(wer("a", "a b c") - 2.0)                      < 1e-9   # >100% allowed
    assert abs(cer("abc", "abd") - 1/3)                      < 1e-9
    # subsetting determinism + duration cutoff
    fake = [{"audio": {"array": [0.0]*16000, "sampling_rate": 16000}} for _ in range(10)]
    sel, tot = select_subset(iter(fake), target_seconds=3, audio_key="audio")
    assert len(sel) == 3 and abs(tot - 3.0) < 1e-9, (len(sel), tot)
    # supportedness matrix
    assert supported("seamless", "hindi") and not supported("seamless", "santali")
    assert not supported("moonshine", "hindi")
    print("OK: all self-tests passed.")

# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["seamless", "moonshine"])
    ap.add_argument("--lang", choices=list(LANGS))
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="train")
    ap.add_argument("--force", action="store_true",
                    help="run an unsupported cell anyway and record the garbage WER")
    ap.add_argument("--subset-file", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); sys.exit(0)
    if not (args.model and args.lang):
        ap.error("--model and --lang are required (or use --selftest)")
    run(args.model, args.lang, args.hours, args.out, args.split, args.force, args.subset_file)
