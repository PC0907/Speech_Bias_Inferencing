#!/usr/bin/env python3
"""
ASR eval harness -- Ali's slice (seamless | owsm | indicconformer).
Moonshine retired (English-only, covered none of the 8).

Coverage:
  Seamless        -> hi/ta/ur/bn only
  OWSM-CTC        -> hi/ta/ur/bn only (open Whisper-style foundation model)
  IndicConformer  -> all 8

Dataset is IndicVoices-style: audio in `audio_filepath` as a torchcodec
AudioDecoder, a `duration` field, references in `normalized`/`verbatim`/`text`.

Per (model, language): load -> deterministic ~2h subset (by `duration`) ->
inference -> reference-prediction CSV (raw text kept) -> WER+CER row.

Text handling is RAW: NFC + whitespace collapse only (--byte-exact drops NFC).
Metrics are self-contained, so scoring needs no jiwer/torch.

Examples:
  python asr_eval.py --model indicconformer --lang santali
  python asr_eval.py --model owsm --lang hindi
  python asr_eval.py --model seamless --lang bengali --text-field verbatim
  python asr_eval.py --selftest
"""
import argparse, csv, os, re, sys, unicodedata

MODELS = ["seamless", "owsm", "indicconformer"]
OWSM_MODEL = "espnet/owsm_ctc_v3.1_1B"          # or espnet/owsm_ctc_v4_1B (newer)

LANGS = {
    "hindi":    {"hub": "XKaab/ASR-Hindi_7hrs",    "seamless": "hin", "owsm": "hin", "ic": "hi",  "tier": "HRL"},
    "tamil":    {"hub": "XKaab/ASR-Tamil_8hrs",    "seamless": "tam", "owsm": "tam", "ic": "ta",  "tier": "HRL"},
    "urdu":     {"hub": "XKaab/ASR-Urdu_6hrs",     "seamless": "urd", "owsm": "urd", "ic": "ur",  "tier": "MRL"},
    "bengali":  {"hub": "XKaab/ASR-Bengali_6hrs",  "seamless": "ben", "owsm": "ben", "ic": "bn",  "tier": "MRL"},
    "dogri":    {"hub": "XKaab/ASR-Dogri_4hrs",    "seamless": None,  "owsm": None,  "ic": "doi", "tier": "LRL-Scheduled"},
    "kashmiri": {"hub": "XKaab/ASR-Kashmiri_4hrs", "seamless": None,  "owsm": None,  "ic": "ks",  "tier": "LRL-Scheduled"},
    "santali":  {"hub": "XKaab/ASR-Santali_4hrs",  "seamless": None,  "owsm": None,  "ic": "sat", "tier": "LRL-Tribal"},
    "bodo":     {"hub": "XKaab/ASR-Bodo_5hrs",     "seamless": None,  "owsm": None,  "ic": "brx", "tier": "LRL-Tribal"},
}

# When forcing a model onto a language it doesn't support, decode with a nearest
# major language's code (script-based default; override with --force-lang).
# Devanagari Indo-Aryan/Tibeto-Burman -> Hindi; Perso-Arabic Kashmiri -> Urdu.
PROXY = {"dogri": "hindi", "bodo": "hindi", "santali": "hindi", "kashmiri": "urdu"}

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
    if model == "seamless":
        return LANGS[lang]["seamless"] is not None
    if model == "owsm":
        return LANGS[lang]["owsm"] is not None
    if model == "indicconformer":
        return True
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
        raise TypeError(f"Unrecognised audio value of type {type(a)}")
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
# Model adapters (lazy heavy imports)
# --------------------------------------------------------------------------
def load_seamless(device, info):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    def transcribe(audio16k, lang):
        try:                                       # newer transformers: audio=
            inp = proc(audio=audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        except TypeError:                          # older transformers: audios=
            inp = proc(audios=audio16k, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            toks = model.generate(**inp, tgt_lang=lang["seamless"])
        return proc.batch_decode(toks, skip_special_tokens=True)[0]
    return transcribe

def load_owsm(device, info):
    import numpy as np
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
    s2t = Speech2TextGreedySearch.from_pretrained(
        OWSM_MODEL, device=device, use_flash_attn=False,
        lang_sym=f"<{info['owsm']}>", task_sym="<asr>",
    )
    def transcribe(audio16k, lang):
        # batch_decode pads <30s to 30s and splits longer audio automatically;
        # a single 1-D array returns a single string.
        return s2t.batch_decode(np.asarray(audio16k, dtype="float32"),
                                batch_size=1, context_len_in_secs=4)
    return transcribe

def load_indicconformer(device, info, decoder="ctc"):
    import torch, numpy as np
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True).eval()
    try:
        model = model.to(device)
    except Exception:
        device = "cpu"
    def transcribe(audio16k, lang):
        wav = torch.tensor(np.asarray(audio16k, dtype="float32")).unsqueeze(0)
        try:
            wav = wav.to(device)
        except Exception:
            pass
        with torch.no_grad():
            out = model(wav, lang["ic"], decoder)
        return out[0] if isinstance(out, (list, tuple)) else out
    return transcribe

def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# --------------------------------------------------------------------------
def run(model_name, lang, hours, out_dir, split, force, data_ref, decoder, nfc, text_field, force_lang):
    os.makedirs(out_dir, exist_ok=True)
    info = LANGS[lang]
    ok = supported(model_name, lang)

    if not ok and not force:
        _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "unsupported")
        print(f"[{model_name}/{lang}] UNSUPPORTED -> coverage gap (use --force to run anyway).")
        return

    # Forcing an unsupported language: decode with a proxy language's code.
    model_info, proxy_note = info, ""
    if not ok and force:
        pkey = force_lang or PROXY.get(lang)
        if pkey:
            p = LANGS[pkey]
            model_info = {**info, "seamless": p["seamless"], "owsm": p["owsm"], "ic": p["ic"]}
            proxy_note = pkey
            print(f"[{model_name}/{lang}] FORCED via '{pkey}' "
                  f"(transcribing {lang} audio with {pkey}'s decoder).")

    data_ref = data_ref or info["hub"]
    ds = load_any(data_ref, split)
    first = next(iter(ds))
    akey = detect_key(first, AUDIO_KEYS, "audio")
    tkey = pick_text_key(first, text_field)
    dkey = next((k for k in DUR_KEYS if k in first), None)

    ds = load_any(data_ref, split)
    rows, total = select_subset(ds, hours * 3600, akey, dkey)
    print(f"[{model_name}/{lang}] {len(rows)} utts ({total/3600:.2f} h)  "
          f"audio='{akey}' text='{tkey}' dur='{dkey}'  data='{data_ref}'")

    device = "cuda" if _cuda() else "cpu"
    if model_name == "indicconformer":
        transcribe = load_indicconformer(device, model_info, decoder)
    elif model_name == "owsm":
        transcribe = load_owsm(device, model_info)
    else:
        transcribe = load_seamless(device, model_info)

    out_rows, W, C = [], [], []
    for i, ex in enumerate(rows):
        try:
            audio, sr = extract_audio(ex[akey])
            hyp = transcribe(to_16k(audio, sr), model_info)
        except Exception as e:
            hyp = ""
            print(f"  utt {i}: error -> empty hyp ({e})")
        ref = ex.get(tkey, "")
        w, c, rp, hp = score(ref, hyp, nfc)
        W.append(w); C.append(c)
        out_rows.append({"idx": i, "reference": ref, "prediction": hyp,
                         "ref_prep": rp, "hyp_prep": hp,
                         "wer": round(w, 4), "cer": round(c, 4)})

    if not out_rows:
        print(f"[{model_name}/{lang}] no rows scored."); return

    p = os.path.join(out_dir, f"{model_name}__{lang}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w_.writeheader(); w_.writerows(out_rows)

    aw, ac = sum(W) / len(W), sum(C) / len(C)
    status = "scored" if ok else (f"forced(via {proxy_note})" if proxy_note else "forced")
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
    assert prep("a\t b\n c") == "a b c"
    assert abs(wer("the cat sat", "the dog sat") - 1/3) < 1e-9
    assert abs(wer("a", "a b c") - 2.0) < 1e-9
    assert abs(cer("abc", "abd") - 1/3) < 1e-9
    arr, sr = extract_audio({"array": [0.1, 0.2, 0.3], "sampling_rate": 16000})
    assert sr == 16000 and len(arr) == 3
    fake = [{"audio_filepath": {"array": [0.0]*16000, "sampling_rate": 16000},
             "duration": 1.0} for _ in range(10)]
    sel, tot = select_subset(iter(fake), 3, "audio_filepath", "duration")
    assert len(sel) == 3 and abs(tot - 3.0) < 1e-9
    assert pick_text_key({"normalized": "x"}, "verbatim") == "normalized"  # fallback
    assert supported("seamless", "hindi") and not supported("seamless", "santali")
    assert supported("owsm", "hindi") and not supported("owsm", "santali")
    assert supported("indicconformer", "santali")
    print("OK: all self-tests passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ali's ASR eval (seamless | owsm | indicconformer).")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--lang", choices=list(LANGS))
    ap.add_argument("--data", default=None, help="local path or HF hub id (default: sheet's hub id)")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--text-field", default="normalized",
                    help="reference column: normalized | verbatim | text")
    ap.add_argument("--decoder", choices=["ctc", "rnnt"], default="ctc",
                    help="IndicConformer only")
    ap.add_argument("--force", action="store_true",
                    help="run an unsupported cell anyway (decodes via a proxy language)")
    ap.add_argument("--force-lang", default=None, choices=list(LANGS),
                    help="proxy language whose code to use when forcing (default: script-based)")
    ap.add_argument("--byte-exact", action="store_true", help="disable NFC too")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); sys.exit(0)
    if not (a.model and a.lang):
        ap.error("--model and --lang are required (or use --selftest)")
    run(a.model, a.lang, a.hours, a.out, a.split, a.force, a.data,
        a.decoder, not a.byte_exact, a.text_field, a.force_lang)
