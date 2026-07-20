#!/usr/bin/env python3
"""
ASR eval harness -- speecht5 | mms | indicconformer | owsm.

SpeechT5       -> English-only (LibriSpeech fine-tune); all 8 languages are
                  coverage gaps. Framing: exclusion by design, not omission.
MMS            -> Claims 1107 languages via per-adapter architecture. Adapters
                  are tried natively for all 8; missing adapter -> "no_adapter".
IndicConformer -> AI4Bharat 600M multilingual; natively covers all 22 IN-22
                  scheduled languages, incl. the tribal LRLs (Santali, Bodo)
                  and scheduled LRLs (Dogri, Kashmiri). Loaded via HF AutoModel
                  (trust_remote_code) -- inference only, no NeMo needed.
OWSM           -> CMU/ESPnet Open Whisper-style model. NOT a transformers
                  model: loads via espnet2.bin.s2t_inference.Speech2Text.
                  Coverage is read off the checkpoint's own token list at load
                  time -- a language with no "<xxx>" token is logged as
                  "unsupported_lang", not scored as a bad WER.

Per (model, language): load -> deterministic ~2h subset -> inference ->
reference-prediction CSV (raw text kept) -> WER+CER row in summary.csv.

Text handling is RAW: NFC + whitespace collapse only. Metrics self-contained.

Kaggle setup for OWSM (needs internet ON; ESPnet pins older deps, so run OWSM
in a fresh session, not the same one as IndicConformer):
  !pip install -q espnet espnet_model_zoo librosa
  !python asr.py --model owsm --lang hindi --out /kaggle/working/results

Examples:
  python asr.py --model indicconformer --lang santali
  python asr.py --model indicconformer --lang hindi --decode rnnt
  python asr.py --model mms --lang santali
  python asr.py --model owsm --lang hindi
  python asr.py --model owsm --lang all --owsm-model espnet/owsm_ctc_v4_1B
  python asr.py --model speecht5 --lang hindi   # coverage gap
  python asr.py --selftest
"""
import argparse, csv, os, re, sys, time, unicodedata

SCRIPT_VERSION = "owsm-r3"   # bump when editing; printed at startup

MODELS = ["speecht5", "mms", "indicconformer", "owsm"]

# --- OWSM --- checkpoints selectable via --owsm-model
OWSM_CHECKPOINTS = [
    "espnet/owsm_v4_medium_1B",   # AED, 1B, 320k h  (default)
    "espnet/owsm_ctc_v4_1B",      # CTC, 1B, 320k h
    "espnet/owsm_v3.1_ebf",       # AED, 1B, 180k h
    "espnet/owsm_ctc_v3.1_1B",    # CTC, 1B, 180k h
]

# NOTE on the "owsm" codes: OWSM uses ISO-639-3. This deliberately does NOT
# reuse the "mms" field -- MMS spells Dogri "dgo", ISO-639-3 is "doi".
LANGS = {
    "hindi":    {"hub": "XKaab/ASR-Hindi_7hrs",    "mms": "hin", "ic": "hi",  "owsm": "hin", "tier": "HRL"},
    "tamil":    {"hub": "XKaab/ASR-Tamil_8hrs",    "mms": "tam", "ic": "ta",  "owsm": "tam", "tier": "HRL"},
    "urdu":     {"hub": "XKaab/ASR-Urdu_6hrs",     "mms": "urd", "ic": "ur",  "owsm": "urd", "tier": "MRL"},
    "bengali":  {"hub": "XKaab/ASR-Bengali_6hrs",  "mms": "ben", "ic": "bn",  "owsm": "ben", "tier": "MRL"},
    "dogri":    {"hub": "XKaab/ASR-Dogri_4hrs",    "mms": "dgo", "ic": "doi", "owsm": "doi", "tier": "LRL-Scheduled"},
    "kashmiri": {"hub": "XKaab/ASR-Kashmiri_4hrs", "mms": "kas", "ic": "ks",  "owsm": "kas", "tier": "LRL-Scheduled"},
    "santali":  {"hub": "XKaab/ASR-Santali_4hrs",  "mms": "sat", "ic": "sat", "owsm": "sat", "tier": "LRL-Tribal"},
    "bodo":     {"hub": "XKaab/ASR-Bodo_5hrs",     "mms": "brx", "ic": "brx", "owsm": "brx", "tier": "LRL-Tribal"},
}

AUDIO_KEYS = ("audio_filepath", "audio", "speech", "wav")
TEXT_KEYS  = ("normalized", "text", "verbatim", "sentence", "transcription", "transcript")
DUR_KEYS   = ("duration", "length", "secs")

# MMS signals a missing language adapter with a few different wordings; the
# tokenizer/adapter error dumps the entire ~1100-code list, so we must catch
# it *before* the per-utterance loop or it floods the notebook 1000+ times.
_NO_ADAPTER_SIGNS = ("adapter", "no file", "404", "does not exist", "choose one of")
def _is_missing_adapter(e):
    s = str(e).lower()
    return any(k in s for k in _NO_ADAPTER_SIGNS)

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
    if model == "indicconformer":
        return True       # all 8 are IN-22 scheduled langs -> native coverage
    if model == "owsm":
        return True       # provisional -- real coverage read from token list at load
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

def load_indicconformer(device, decode="ctc"):
    """
    AI4Bharat IndicConformer-600M-Multi: single Hybrid CTC+RNNT conformer that
    natively covers all 22 IN-22 scheduled languages. One model; language is
    selected per call -- no per-language reload. Inference only (no NeMo).

    Forward:  model(wav, lang_code, "ctc"|"rnnt") -> transcription (str)
    wav: float32 tensor [1, num_samples] @ 16 kHz.
    """
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    ).to(device).eval()

    def transcribe(audio16k, lang):
        code = lang["ic"]
        wav = torch.tensor(audio16k, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
        with torch.no_grad():
            out = model(wav, code, decode)
        if isinstance(out, (list, tuple)):      # custom code returns str or [str]
            out = out[0] if out else ""
        return str(out)
    return transcribe

# --- OWSM -----------------------------------------------------------------
OWSM_WINDOW = 30 * 16000        # OWSM is trained on fixed 30 s inputs

def _owsm_windows(x):
    """Split into 30 s chunks, right-padding the last one.

    Naive non-overlapping chunking. The OWSM model card uses an overlapped
    buffered decode for long-form audio, which handles boundary words better.
    Most utterance-level clips are well under 30 s so this rarely triggers,
    but long recordings will take a small WER hit at the seams.
    """
    import numpy as np
    x = np.asarray(x, dtype="float32")
    if len(x) <= OWSM_WINDOW:
        return [np.pad(x, (0, OWSM_WINDOW - len(x)))]
    out = []
    for i in range(0, len(x), OWSM_WINDOW):
        chunk = x[i:i + OWSM_WINDOW]
        out.append(np.pad(chunk, (0, max(0, OWSM_WINDOW - len(chunk)))))
    return out

def _owsm_text(result):
    """ESPnet returns [(text, tokens, token_ints, text_nospecial, hyp), ...].

    Field order has shifted between ESPnet releases, so prefer the
    special-token-stripped field but fall back rather than hard-indexing.
    """
    if not result:
        return ""
    first = result[0]
    cands = []
    if len(first) >= 4:
        cands.append(first[3])
    if len(first) >= 2:
        cands.append(first[-2])
    cands.append(first[0])
    for c in cands:
        if isinstance(c, str) and c.strip():
            return c
    return ""

def load_owsm(device, model_id="espnet/owsm_v4_medium_1B", beam_size=5,
              fp16=True):
    """
    OWSM via ESPnet. Three things differ from the transformers models above:

    1. The language token is baked in at Speech2Text construction time, so we
       build one instance per language. AT MOST ONE lives on the GPU at a time
       -- a 1B model is ~4 GB in fp32, and two of them plus beam-search
       activations will OOM a 16 GB card. Switching language evicts the
       previous instance.
    2. Coverage is read off the checkpoint's own token list. The probe used to
       read it is built on CPU and freed immediately, so it never occupies GPU
       memory. No hardcoded supported-language list.
    3. CTC checkpoints (owsm_ctc_*) use a different inference class.
    """
    import gc
    import torch

    is_ctc = "ctc" in model_id
    if is_ctc:
        from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch as S2T
        kw = {}
    else:
        from espnet2.bin.s2t_inference import Speech2Text as S2T
        kw = {"beam_size": beam_size}

    use_cuda = str(device).startswith("cuda")
    if fp16 and use_cuda:
        kw["dtype"] = "float16"

    # Read the token list on CPU, then drop it. Costs host RAM briefly
    # (Kaggle has ~29 GB) and zero GPU.
    probe = S2T.from_pretrained(model_id, lang_sym="<eng>", task_sym="<asr>",
                                device="cpu",
                                **{k: v for k, v in kw.items() if k != "dtype"})
    token_list = set(getattr(probe.converter, "token_list", []))
    del probe
    gc.collect()

    live = {"code": None, "s2t": None}

    def _for(code):
        if live["code"] == code:
            return live["s2t"]
        if live["s2t"] is not None:          # evict before allocating the next
            live["s2t"] = None
            live["code"] = None
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
        live["s2t"] = S2T.from_pretrained(model_id, lang_sym=f"<{code}>",
                                          task_sym="<asr>", device=device, **kw)
        live["code"] = code
        return live["s2t"]

    def transcribe(audio16k, lang):
        s2t = _for(lang["owsm"])
        try:
            out = " ".join(_owsm_text(s2t(w))
                           for w in _owsm_windows(audio16k)).strip()
        finally:
            # Beam-search hypotheses are freed by refcount, but the caching
            # allocator holds the blocks; without this the high-water mark
            # creeps up across utterances until it OOMs.
            if use_cuda:
                torch.cuda.empty_cache()
        return out

    transcribe.covers = lambda code: f"<{code}>" in token_list
    transcribe.model_id = model_id
    transcribe.n_langs = sum(
        1 for t in token_list
        if isinstance(t, str) and len(t) == 5 and t.startswith("<") and t.endswith(">"))
    return transcribe
# --- end OWSM -------------------------------------------------------------

def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# --------------------------------------------------------------------------
def run(model_name, lang, hours, out_dir, split, force, data_ref, nfc, text_field,
        decode="ctc", transcribe=None, owsm_model="espnet/owsm_v4_medium_1B",
        owsm_beam=5, owsm_fp16=True, progress_every=25, max_minutes=0):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== asr.py {SCRIPT_VERSION} | {model_name}/{lang} | "
          f"cuda={_cuda()} | beam={owsm_beam} fp16={owsm_fp16} "
          f"progress_every={progress_every} ===", flush=True)
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

    # Load the model once (or reuse a preloaded one passed in via transcribe=)
    if transcribe is None:
        device = "cuda" if _cuda() else "cpu"
        if model_name == "mms":
            transcribe = load_mms(device)
        elif model_name == "indicconformer":
            transcribe = load_indicconformer(device, decode)
        elif model_name == "owsm":                      # --- OWSM ---
            transcribe = load_owsm(device, owsm_model, owsm_beam, owsm_fp16)
        else:
            transcribe = load_speecht5(device)

    # For MMS, try loading the adapter once up front to detect "no_adapter" early
    if model_name == "mms":
        try:
            from transformers import AutoProcessor as AP
            _p = AP.from_pretrained("facebook/mms-1b-all")
            _p.tokenizer.set_target_lang(info["mms"])
        except Exception as e:
            if _is_missing_adapter(e):
                _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "no_adapter")
                print(f"[{model_name}/{lang}] NO ADAPTER for '{info['mms']}' -> "
                      f"logged as no_adapter (MMS doesn't cover this language).")
                return

    # --- OWSM --- coverage gap check, mirrors the MMS block above
    if model_name == "owsm" and hasattr(transcribe, "covers"):
        code = info["owsm"]
        if not transcribe.covers(code):
            _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "",
                     "unsupported_lang")
            print(f"[{model_name}/{lang}] '<{code}>' NOT in token list of "
                  f"{transcribe.model_id} ({transcribe.n_langs} lang tokens) -> "
                  f"coverage gap, not scored.")
            return

    print(f"[{model_name}/{lang}] model ready, starting inference on "
          f"{len(rows)} utts...", flush=True)
    out_rows, W, C = [], [], []
    adapter_failed = False
    n_err = 0
    scored_secs = 0.0
    budget_hit = False
    _t0 = time.time()
    for i, ex in enumerate(rows):
        if max_minutes and (time.time() - _t0) / 60 >= max_minutes:
            budget_hit = True
            print(f"  [{model_name}/{lang}] wall-clock budget {max_minutes}m hit "
                  f"at utt {i}/{len(rows)} -> stopping, logging as partial.")
            break
        try:
            audio, sr = extract_audio(ex[akey])
            a16 = to_16k(audio, sr)
            hyp = transcribe(a16, info)
            scored_secs += len(a16) / 16000
        except Exception as e:
            if _is_missing_adapter(e):
                # adapter doesn't exist: log immediately, don't continue
                _summary(out_dir, model_name, lang, info["tier"], 0, 0.0, "", "", "no_adapter")
                print(f"[{model_name}/{lang}] NO ADAPTER for '{info['mms']}' -> no_adapter.")
                adapter_failed = True
                break
            hyp = ""
            n_err += 1
            print(f"  utt {i}: error -> empty hyp ({str(e)[:200]})")   # truncated
        ref = ex.get(tkey, "")
        w, c, rp, hp = score(ref, hyp, nfc)
        W.append(w); C.append(c)
        out_rows.append({"idx": i, "reference": ref, "prediction": hyp,
                         "ref_prep": rp, "hyp_prep": hp,
                         "wer": round(w, 4), "cer": round(c, 4)})

        if i == 0 or (i + 1) % progress_every == 0 or (i + 1) == len(rows):
            el = time.time() - _t0
            rate = (i + 1) / el
            eta = (len(rows) - i - 1) / rate if rate else float("nan")
            print(f"  [{model_name}/{lang}] {i+1}/{len(rows)}  "
                  f"{rate:.2f} utt/s  elapsed {el/60:.1f}m  eta {eta/60:.1f}m  "
                  f"running WER={sum(W)/len(W):.4f}  errs={n_err}",
                  flush=True)

    if adapter_failed or not out_rows:
        return

    p = os.path.join(out_dir, f"{model_name}__{lang}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w_.writeheader(); w_.writerows(out_rows)

    aw, ac = sum(W) / len(W), sum(C) / len(C)
    status = "scored" if ok else "forced(unsupported)"
    if model_name == "indicconformer":
        status = f"{status}[{decode}]"
    if model_name == "owsm":                            # --- OWSM ---
        status = f"{status}[{owsm_model.split('/')[-1]}]"
    if budget_hit:
        # Do NOT report this as a clean 2 h row -- it is a different, smaller
        # sample than the languages that finished, and averaging it into a
        # tier comparison without the flag would be misleading.
        status = f"{status}[partial {len(out_rows)}/{len(rows)}]"
    # audio_hours = what was actually scored, not what was selected
    _summary(out_dir, model_name, lang, info["tier"], len(out_rows),
             scored_secs / 3600 if scored_secs else total / 3600,
             round(aw, 4), round(ac, 4), status)
    print(f"[{model_name}/{lang}] WER={aw:.4f}  CER={ac:.4f}  (ref='{tkey}')  "
          f"n={len(out_rows)}  scored={scored_secs/3600:.2f}h  errs={n_err}  -> {p}")

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
    assert supported("indicconformer", "santali")
    assert supported("indicconformer", "bodo")
    assert supported("owsm", "santali")
    assert LANGS["bodo"]["ic"] == "brx" and LANGS["dogri"]["ic"] == "doi"
    # --- OWSM --- every language carries an ISO-639-3 code, and Dogri does NOT
    # inherit the MMS spelling
    assert all("owsm" in v for v in LANGS.values())
    assert LANGS["dogri"]["owsm"] == "doi" and LANGS["dogri"]["mms"] == "dgo"
    # 30 s windowing: short clip -> 1 padded window; 70 s -> 3 windows
    import numpy as np
    assert len(_owsm_windows(np.zeros(16000, dtype="float32"))) == 1
    assert len(_owsm_windows(np.zeros(16000, dtype="float32"))[0]) == OWSM_WINDOW
    assert len(_owsm_windows(np.zeros(70 * 16000, dtype="float32"))) == 3
    assert all(len(w) == OWSM_WINDOW
               for w in _owsm_windows(np.zeros(70 * 16000, dtype="float32")))
    assert _owsm_text([]) == ""
    assert _owsm_text([("<eng><asr> hi", ["x"], [1], "hi", None)]) == "hi"
    print("OK: all self-tests passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ASR eval: speecht5 (English-only) + mms + indicconformer + owsm.")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--lang", choices=list(LANGS) + ["all"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--text-field", default="normalized")
    ap.add_argument("--decode", choices=["ctc", "rnnt"], default="ctc",
                    help="IndicConformer decoding strategy (ignored by other models)")
    ap.add_argument("--owsm-model", default="espnet/owsm_v4_medium_1B",
                    choices=OWSM_CHECKPOINTS,
                    help="OWSM checkpoint (ignored by other models)")
    ap.add_argument("--owsm-beam", type=int, default=5,
                    help="OWSM beam size; 1 = greedy, much lower peak memory")
    ap.add_argument("--owsm-fp32", action="store_true",
                    help="run OWSM in fp32 (default is fp16 on CUDA)")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print rate/ETA every N utterances")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="wall-clock budget per language; 0 = no limit. "
                         "Stops cleanly and flags the row as partial.")
    ap.add_argument("--force", action="store_true",
                    help="run speecht5 anyway to record English-model-on-Indic exclusion WER")
    ap.add_argument("--byte-exact", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); sys.exit(0)
    if not (a.model and a.lang):
        ap.error("--model and --lang are required (or use --selftest)")

    langs = list(LANGS) if a.lang == "all" else [a.lang]

    # Load a multilingual model once and reuse it across all languages.
    # OWSM is excluded on purpose: its language token is fixed at construction,
    # so load_owsm() caches one instance per language internally instead.
    shared = None
    if a.lang == "all" and a.model == "indicconformer":
        shared = load_indicconformer("cuda" if _cuda() else "cpu", a.decode)
    if a.lang == "all" and a.model == "owsm":
        shared = load_owsm("cuda" if _cuda() else "cpu", a.owsm_model,
                           a.owsm_beam, not a.owsm_fp32)

    for lg in langs:
        run(a.model, lg, a.hours, a.out, a.split, a.force, a.data,
            not a.byte_exact, a.text_field, a.decode, transcribe=shared,
            owsm_model=a.owsm_model, owsm_beam=a.owsm_beam,
            owsm_fp16=not a.owsm_fp32, progress_every=a.progress_every,
            max_minutes=a.max_minutes)
