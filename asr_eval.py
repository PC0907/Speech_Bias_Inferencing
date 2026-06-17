#!/usr/bin/env python3
"""
ASR eval harness -- Rimsha's slice (speecht5 | mms).

SpeechT5  -> English-only (LibriSpeech fine-tune); all 8 languages are
             coverage gaps. Framing: exclusion by design, not omission.
MMS       -> Claims 1107 languages via per-adapter architecture. THE key
             "prove it" model: adapters are tried natively for all 8,
             including Dogri (dgo), Kashmiri (kas), Santali (sat), Bodo (brx).
             If an adapter doesn't exist -> logged as "no_adapter" (a finding).
             If it exists but WER is catastrophic -> logged as "scored".

Per (model, language): load -> deterministic ~2h subset -> inference ->
reference-prediction CSV (raw text kept) -> WER+CER row in summary.csv.

Text handling is RAW: NFC + whitespace collapse only. Metrics self-contained.

Examples:
  python asr_eval_rimsha.py --model mms --lang santali
  python asr_eval_rimsha.py --model mms --lang hindi
  python asr_eval_rimsha.py --model speecht5 --lang hindi   # coverage gap
  python asr_eval_rimsha.py --selftest
"""
import argparse, csv, os, re, sys, unicodedata

MODELS = ["speecht5", "mms"]

LANGS = {
    "hindi":    {"hub": "XKaab/ASR-Hindi_7hrs",    "mms": "hin", "tier": "HRL"},
    "tamil":    {"hub": "XKaab/ASR-Tamil_8hrs",    "mms": "tam", "tier": "HRL"},
    "urdu":     {"hub": "XKaab/ASR-Urdu_6hrs",     "mms": "urd", "tier": "MRL"},
    "bengali":  {"hub": "XKaab/ASR-Bengali_6hrs",  "mms": "ben", "tier": "MRL"},
    "dogri":    {"hub": "XKaab/ASR-Dogri_4hrs",    "mms": "dgo", "tier": "LRL-Scheduled"},
    "kashmiri": {"hub": "XKaab/ASR-Kashmiri_4hrs", "mms": "kas", "tier": "LRL-Scheduled"},
    "santali":  {"hub": "XKaab/ASR-Santali_4hrs",  "mms": "sat", "tier": "LRL-Tribal"},
    "bodo":     {"hub": "XKaab/ASR-Bodo_5hrs",     "mms": "brx", "tier": "LRL-Tribal"},
}

AUDIO_KEYS = ("audio_filepath", "audio", "speech", "wav")
TEXT_KEYS  = ("normalized", "text", "verbatim", "sentence", "transcription", "transcript")
DUR_KEYS   = ("duration", "length", "secs")

# --------------------------------------------------------------------------
# Text prep -- RAW (NFC + whitespace only)
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")
def prep(t, nfc=True):
    t = str(t or "")
    if nfc:
        t = unicodedata.normalize("NFC", t)
    return _WS.sub(" ", t).strip()

# --------------------------------------------------------------------------
# Metrics (self-contained; not clipped, can exceed 1.0)
# --------------------------------------------------------------------------
def _lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return _lev(r, h) / len(r) if r else float(bool(h))

def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    return _lev(r, h) / len(r) if r else float(bool(h))

def score(ref_raw, hyp_raw, nfc=True):
    r, h = prep(ref_raw, nfc), prep(hyp_raw, nfc)
    return wer(r, h), cer(r, h), r, h

# --------------------------------------------------------------------------
def supported(model, lang):
    if model == "speecht5":
        return False      # English-only; every cell is a coverage gap
    if model == "mms":
        return True       # claims all 8 natively via adapters; adapter may still fail
    raise ValueError(model)

# --------------------------------------------------------------------------
# Audio: torchcodec AudioDecoder, classic dict, or a file path
# --------------------------------------------------------------------------
def extract_audio(a):
    import numpy as np
    if isinstance(a, dict) and "array" in a:
        arr, sr = np.asarray(a["array"], dtype="float32"), int(a["sampling_rate"])
    elif hasattr(a, "get_all_samples"):
        s = a.get_all_samples()
        arr, sr = s.data.detach().cpu().numpy(), int(s.sample_rate)
    elif isinstance(a, str):
        import soundfile as sf
        arr, sr = sf.read(a, dtype="float32")
    else:
        raise TypeError(f"Unrecognised audio type: {type(a)}")
    if getattr(arr, "ndim", 1) == 2:
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[1] else 1)
    return arr.astype("float32"), sr

def to_16k(array, sr):
    import numpy as np
    if sr == 16000:
        return np.asarray(array, dtype="float32")
    import torch, torchaudio
    wav = torch.tensor(np.asarray(array, dtype="float32"))
    return torchaudio.functional.resample(wav, sr, 16000).numpy()

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_any(data_ref, split):
    from datasets import load_from_disk, load_dataset
    if os.path.exists(data_ref):
        ds = load_from_disk(data_ref)
        if hasattr(ds, "column_names") and isinstance(ds.column_names, dict):
            ds = ds[split] if split in ds else ds[list(ds.keys())[0]]
        return ds
    return load_dataset(data_ref, split=split, streaming=True)

def detect_key(row, cands, kind):
    for k in cands:
        if k in row:
            return k
    raise KeyError(f"No {kind} column in {list(row.keys())}.")

def pick_text_key(row, preferred):
    if preferred and preferred in row:
        return preferred
    return detect_key(row, TEXT_KEYS, "text")

def select_subset(rows, target_seconds, akey, dur_key):
    out, total = [], 0.0
    for ex in rows:
        if dur_key and ex.get(dur_key) is not None:
            d = float(ex[dur_key])
        else:
            arr, sr = extract_audio(ex[akey]); d = len(arr) / sr
        out.append(ex); total += d
        if total >= target_seconds:
            break
    return out, total

# --------------------------------------------------------------------------
# Model adapters
# --------------------------------------------------------------------------
def load_speecht5(device):
    """English-only. Used only with --force to record the exclusion WER."""
    import torch
    from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
    proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
    model = SpeechT5ForSpeechToText.from_pretrained(
        "microsoft/speecht5_asr").to(device).eval()
    def transcribe(audio16k, lang):
        inp = proc(audio=audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inp, max_length=400)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]
    return transcribe

def load_mms(device):
    """
    MMS-1b-all: Wav2Vec2ForCTC + per-language adapter.
    Call sequence per utterance:
      processor.tokenizer.set_target_lang(lang_code)
      model.load_adapter(lang_code)
      -> greedy CTC decode
    Raises RuntimeError if the adapter doesn't exist -> caught as "no_adapter".
    """
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
    current_lang = [None]   # track loaded adapter to avoid redundant reloads

    def transcribe(audio16k, lang):
        code = lang["mms"]
        if current_lang[0] != code:
            proc.tokenizer.set_target_lang(code)
            model.load_adapter(code)    # raises if adapter missing
            current_lang[0] = code
        inp = proc(audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inp).logits
        ids = torch.argmax(logits, dim=-1)
        return proc.batch_decode(ids)[0]
    return transcribe

def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# --------------------------------------------------------------------------
def run(model_name, lang, hours, out_dir, split, force, data_ref, nfc, text_field):
    os.makedirs(out_dir, exist_ok=True)
    info = LANGS[lang]
    ok = supported(model_name, lang)

    # SpeechT5: every cell is a gap unless forced
    if not ok and not force:
        _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "unsupported")
        print(f"[{model_name}/{lang}] UNSUPPORTED (English-only) -> coverage gap.")
        return

    data_ref = data_ref or info["hub"]
    ds = load_any(data_ref, split)
    first = next(iter(ds))
    akey = detect_key(first, AUDIO_KEYS, "audio")
    tkey = pick_text_key(first, text_field)
    dkey = next((k for k in DUR_KEYS if k in first), None)

    ds = load_any(data_ref, split)
    rows, total = select_subset(ds, hours * 3600, akey, dkey)
    print(f"[{model_name}/{lang}] {len(rows)} utts ({total/3600:.2f} h)  "
          f"audio='{akey}' text='{tkey}'  data='{data_ref}'")

    device = "cuda" if _cuda() else "cpu"
    if model_name == "mms":
        transcribe = load_mms(device)
    else:
        transcribe = load_speecht5(device)

    # For MMS, try loading the adapter once up front to detect "no_adapter" early
    if model_name == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        try:
            # adapter is loaded inside transcribe() on first call; probe here
            import torch
            from transformers import AutoProcessor as AP
            _p = AP.from_pretrained("facebook/mms-1b-all")
            _p.tokenizer.set_target_lang(info["mms"])
        except Exception as e:
            if "adapter" in str(e).lower() or "no file" in str(e).lower():
                _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "no_adapter")
                print(f"[{model_name}/{lang}] NO ADAPTER for '{info['mms']}' -> "
                      f"logged as no_adapter (MMS doesn't cover this language). Error: {e}")
                return

    out_rows, W, C = [], [], []
    adapter_failed = False
    for i, ex in enumerate(rows):
        try:
            audio, sr = extract_audio(ex[akey])
            hyp = transcribe(to_16k(audio, sr), info)
        except Exception as e:
            err = str(e).lower()
            if "adapter" in err or "no file" in err or "404" in err:
                # adapter doesn't exist: log immediately, don't continue
                _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "no_adapter")
                print(f"[{model_name}/{lang}] NO ADAPTER for '{info['mms']}': {e}")
                adapter_failed = True
                break
            hyp = ""
            print(f"  utt {i}: error -> empty hyp ({e})")
        ref = ex.get(tkey, "")
        w, c, rp, hp = score(ref, hyp, nfc)
        W.append(w); C.append(c)
        out_rows.append({"idx": i, "reference": ref, "prediction": hyp,
                         "ref_prep": rp, "hyp_prep": hp,
                         "wer": round(w, 4), "cer": round(c, 4)})

    if adapter_failed or not out_rows:
        return

    p = os.path.join(out_dir, f"{model_name}__{lang}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w_.writeheader(); w_.writerows(out_rows)

    aw, ac = sum(W) / len(W), sum(C) / len(C)
    status = "scored" if ok else "forced(unsupported)"
    _summary(out_dir, model_name, lang, info["tier"], len(out_rows), total / 3600,
             round(aw, 4), round(ac, 4), status)
    print(f"[{model_name}/{lang}] WER={aw:.4f}  CER={ac:.4f}  (ref='{tkey}')  -> {p}")

def _summary(out_dir, model, lang, tier, n, hrs, w, c, status):
    p = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "language", "tier", "n_utts",
                         "audio_hours", "WER", "CER", "status"])
        wr.writerow([model, lang, tier, n, round(hrs, 3), w, c, status])

# --------------------------------------------------------------------------
def selftest():
    assert prep("Hello,  World!") == "Hello, World!"
    assert prep("क।") == "क।"
    assert prep("a\t b") == "a b"
    assert abs(wer("the cat sat", "the dog sat") - 1/3) < 1e-9
    assert abs(wer("a", "a b c") - 2.0) < 1e-9
    assert abs(cer("abc", "abd") - 1/3) < 1e-9
    fake = [{"audio_filepath": {"array": [0.0]*16000, "sampling_rate": 16000},
             "duration": 1.0} for _ in range(10)]
    sel, tot = select_subset(iter(fake), 3, "audio_filepath", "duration")
    assert len(sel) == 3 and abs(tot - 3.0) < 1e-9
    assert not supported("speecht5", "hindi")
    assert supported("mms", "santali")
    print("OK: all self-tests passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Rimsha's ASR eval: speecht5 (English-only) + mms (1107-lang claim).")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--lang", choices=list(LANGS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--text-field", default="normalized")
    ap.add_argument("--force", action="store_true",
                    help="run speecht5 anyway to record English-model-on-Indic exclusion WER")
    ap.add_argument("--byte-exact", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); sys.exit(0)
    if not (a.model and a.lang):
        ap.error("--model and --lang are required (or use --selftest)")
    run(a.model, a.lang, a.hours, a.out, a.split, a.force, a.data,
        not a.byte_exact, a.text_field)
