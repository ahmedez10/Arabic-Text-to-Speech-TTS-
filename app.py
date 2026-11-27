import random
import sys
import io
import re
import wave
import json
import base64
import threading
import numpy as np
import torch
import librosa
from scipy.signal import lfilter
from pathlib import Path
import os
import tempfile
import shutil

# Force UTF-8 output so Arabic in print() doesn't crash Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("=" * 60)
    print("  Missing dependencies. Install with:")
    print("    pip install fastapi uvicorn python-multipart")
    print("=" * 60)
    sys.exit(1)

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# =============================
# Runtime / Device
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

MODEL = None
_MODEL_LOAD_ERROR = None
_MODEL_LOCK = threading.Lock()
_APP_DIR = Path(__file__).parent
DEFAULT_VOICE_PATH = _APP_DIR / "female_voice.wav"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB cap on uploaded reference audio

def default_audio_path():
    return str(DEFAULT_VOICE_PATH) if DEFAULT_VOICE_PATH.exists() else None

def get_or_load_model():
    global MODEL, _MODEL_LOAD_ERROR
    if MODEL is None:
        if _MODEL_LOAD_ERROR is not None:
            raise RuntimeError(f"Model unavailable: {_MODEL_LOAD_ERROR}")
        print("Loading model from checkpoint/ ...")
        MODEL = ChatterboxMultilingualTTS.from_local(_APP_DIR / "checkpoint", DEVICE)
        print("Model loaded successfully!")
    return MODEL

try:
    get_or_load_model()
except Exception as e:
    _MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"CRITICAL: Failed to load model. Error: {_MODEL_LOAD_ERROR}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def numpy_to_wav_bytes(sr: int, arr: np.ndarray) -> bytes:
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

# =============================
# Audio reader (WAV, MP3, M4A, OGG, etc.)
# =============================
def load_audio_float(path: str, max_duration: float = 30.0) -> tuple:
    """
    Load any supported audio file as (sample_rate, float32 mono array),
    capped at max_duration seconds to bound RAM/CPU usage.
    """
    data, sr = librosa.load(path, sr=None, mono=True, duration=max_duration)
    return int(sr), data.astype(np.float32, copy=False)

# =============================
# Reference Audio Preprocessing
# =============================
def preprocess_reference_audio(path: str, target_sr: int = 24000, target_duration: float = 8.0) -> str:
    """
    Prepare reference audio for best voice cloning quality:
    1. Load (capped to 30 s) + anti-aliased resample to model SR (24 kHz)
    2. Find the most voiced segment — skips silence-heavy regions
    3. Trim to 8 s (model clips at 10 s anyway; cleaner is better)
    4. Normalize to -3 dBFS
    Returns path to a preprocessed temp WAV (caller must delete it).
    """
    try:
        sr, data = load_audio_float(path, max_duration=30.0)
        if len(data) == 0:
            return path

        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32, copy=False)
            sr = target_sr

        target_samples = int(target_duration * sr)

        if len(data) <= target_samples:
            segment = data
        else:
            # Sliding-window search for the most voiced 8-second segment
            frame_size = int(sr * 0.025)   # 25 ms frames
            n_frames = len(data) // frame_size
            rms_per_frame = np.array([
                np.sqrt(np.mean(data[i * frame_size:(i + 1) * frame_size] ** 2))
                for i in range(n_frames)
            ])
            window_frames = min(int(target_duration * sr / frame_size), n_frames)
            best_start_frame = 0
            best_score = -1.0
            for i in range(n_frames - window_frames + 1):
                score = float(np.sum(rms_per_frame[i:i + window_frames]))
                if score > best_score:
                    best_score = score
                    best_start_frame = i
            start = best_start_frame * frame_size
            segment = data[start: start + target_samples]

        # Peak-normalize to -3 dBFS
        peak = np.max(np.abs(segment))
        if peak > 1e-6:
            segment = segment * (10 ** (-3.0 / 20.0) / peak)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(segment, -1, 1) * 32767).astype(np.int16).tobytes())
        return tmp.name

    except Exception as e:
        print(f"preprocess_reference_audio failed ({e}), using raw path")
        return path

# =============================
# Text Chunking
# =============================
def split_into_chunks(text: str, max_chars: int = 120) -> list:
    """
    Split Arabic text into short sentence chunks for per-sentence generation.

    WHY THIS MATTERS:
    Chatterbox produces significantly better quality on short inputs (<120 chars)
    than on long single-shot generations. Long text causes:
    - Timing drift (speech gets too fast or too slow)
    - Quality degradation in the second half of the utterance
    - Repetition/looping artifacts near the end

    Splits on Arabic sentence-ending punctuation: . ! ? ؟
    Then sub-splits long chunks on ، (Arabic comma).
    Merges very short fragments to avoid generating 1-2 word clips.
    """
    # Primary split: sentence-ending punctuation
    parts = re.split(r"(?<=[.!?؟])\s+", text.strip())
    refined = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            refined.append(part)
        else:
            # Secondary split on Arabic comma
            sub = re.split(r"،\s*", part)
            buf = ""
            for s in sub:
                s = s.strip()
                if not s:
                    continue
                if buf and len(buf) + len(s) + 2 > max_chars:
                    refined.append(buf)
                    buf = s
                else:
                    buf = (buf + "، " + s).strip() if buf else s
            if buf:
                refined.append(buf)

    # Merge very short trailing fragments into the previous chunk (preserving sentence punctuation)
    merged = []
    for chunk in refined:
        if merged and len(chunk) < 20 and len(merged[-1]) + len(chunk) + 1 <= max_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged if merged else [text.strip()]


def crossfade_concat(chunks: list, sr: int, fade_ms: int = 25) -> np.ndarray:
    """Concatenate audio chunks with a short crossfade to avoid clicks at joins."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    fade_len = int(sr * fade_ms / 1000)
    result = chunks[0].copy()
    for nxt in chunks[1:]:
        if len(result) < fade_len or len(nxt) < fade_len:
            result = np.concatenate([result, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in  = np.linspace(0.0, 1.0, fade_len)
        overlap  = result[-fade_len:] * fade_out + nxt[:fade_len] * fade_in
        result   = np.concatenate([result[:-fade_len], overlap, nxt[fade_len:]])
    return result

# =============================
# Audio Post-Processing
# =============================
def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0,
                 min_silence_ms: int = 80, pad_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing silence. Keeps a small pad to avoid clipped starts."""
    threshold  = 10 ** (threshold_db / 20.0)
    frame_len  = int(sr * 0.01)       # 10 ms frames
    min_frames = max(1, int(min_silence_ms / 10))
    pad_samples = int(sr * pad_ms / 1000)

    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    start_frame = 0
    for i in range(n_frames - min_frames):
        if all(rms[i + j] > threshold for j in range(min(min_frames, 3))):
            start_frame = i
            break

    end_frame = n_frames
    for i in range(n_frames - 1, min_frames - 1, -1):
        if all(rms[i - j] > threshold for j in range(min(min_frames, 3))):
            end_frame = i + 1
            break

    start_sample = max(0, start_frame * frame_len - pad_samples)
    end_sample   = min(len(audio), end_frame * frame_len + pad_samples)
    trimmed = audio[start_sample:end_sample]

    return trimmed if len(trimmed) >= sr * 0.1 else audio


def normalize_volume(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Peak-normalize audio to target_db below 0 dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return audio * (10 ** (target_db / 20.0) / peak)


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """
    1st-order IIR high-pass filter via scipy.signal.lfilter.
    Removes sub-80 Hz DC/rumble that makes TTS output sound muddy.
    """
    if len(audio) < 2:
        return audio
    rc    = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt    = 1.0 / sr
    alpha = rc / (rc + dt)
    return lfilter([alpha, -alpha], [1.0, -alpha], audio).astype(np.float32, copy=False)


def postprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """High-pass → trim silence → peak normalize."""
    audio = highpass_filter(audio, sr, cutoff_hz=80.0)
    audio = trim_silence(audio, sr)
    audio = normalize_volume(audio)
    return audio

# =============================
# Gender Detection
# =============================
def detect_voice_gender(audio_path: str) -> str:
    """
    Estimate gender from pitch (F0) using autocorrelation.
    Returns 'female', 'male', or 'unknown'.
    Female F0: ~160-300 Hz  |  Male F0: ~80-160 Hz
    """
    try:
        sr, data = load_audio_float(audio_path, max_duration=10.0)
        data = data[: sr * 5]
        rms = np.sqrt(np.mean(data ** 2))
        if rms < 1e-4:
            return "unknown"
        data = data / (rms + 1e-9)

        frame_len = int(sr * 0.05)
        hop = frame_len // 2
        f0_values = []
        min_lag = int(sr / 350)
        max_lag = int(sr / 70)

        for start in range(0, len(data) - frame_len, hop):
            frame = data[start: start + frame_len]
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            if max_lag >= len(corr):
                continue
            segment = corr[min_lag: max_lag]
            if segment.size == 0:
                continue
            peak_idx = int(np.argmax(segment)) + min_lag
            if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.3:
                f0_values.append(sr / peak_idx)

        if not f0_values:
            return "unknown"

        median_f0 = float(np.median(f0_values))
        print(f"Detected median F0: {median_f0:.1f} Hz")

        if median_f0 > 165:
            return "female"
        elif median_f0 < 155:
            return "male"
        else:
            return "unknown"

    except Exception as e:
        print(f"Gender detection failed: {e}")
        return "unknown"

# =============================
# Audio Quality Scoring
# =============================
def score_audio_quality(sr: int, audio: np.ndarray, text_len: int) -> dict:
    """
    Score generated audio on signal-level heuristics.
    Higher total = better. Used for Best-of-N selection.
    """
    scores = {}

    # 1. Silence ratio — Egyptian speech: ideal 15-30%
    frame_size   = int(sr * 0.025)
    rms_threshold = 0.015
    n_frames     = max(1, len(audio) // frame_size)
    silent_frames = sum(
        1 for i in range(n_frames)
        if np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2)) < rms_threshold
    )
    sil = silent_frames / n_frames
    if 0.15 <= sil <= 0.30:
        scores["silence"] = 1.0
    elif sil > 0.50:
        scores["silence"] = max(0.0, 1.0 - (sil - 0.30) * 3.0)
    elif sil < 0.08:
        scores["silence"] = max(0.5, 1.0 - (0.08 - sil) * 5.0)
    else:
        scores["silence"] = 0.75  # marginal range

    # 2. Clipping
    clip_ratio = int(np.sum(np.abs(audio) > 0.95)) / max(1, len(audio))
    scores["clipping"] = max(0.0, 1.0 - clip_ratio * 1000)

    # 3. Duration sanity
    duration = len(audio) / sr
    exp_min  = text_len / 8.0
    exp_max  = text_len / 1.5
    if exp_min <= duration <= exp_max:
        scores["duration"] = 1.0
    elif duration < exp_min * 0.5 or duration > exp_max * 2:
        scores["duration"] = 0.2
    else:
        scores["duration"] = 0.6

    # 4. RMS smoothness (low CoV = natural pacing)
    window   = int(sr * 0.1)
    n_win    = max(1, len(audio) // window)
    rms_vals = [np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2)) for i in range(n_win)]
    rms_arr  = np.array(rms_vals)
    if rms_arr.mean() > 0:
        cv = float(rms_arr.std() / rms_arr.mean())
        scores["smoothness"] = max(0.0, 1.0 - cv * 0.8)
    else:
        scores["smoothness"] = 0.0

    # 5. Spectral — ZCR proxy for speech vs. noise
    zcr_values = []
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        if len(frame) < 2:
            continue
        crossings = int(np.sum(np.abs(np.diff(np.sign(frame))) > 0))
        zcr_values.append(crossings / len(frame) * sr)
    if zcr_values:
        mzcr = float(np.median(zcr_values))
        if 500 < mzcr < 4000:
            scores["spectral"] = 1.0
        elif mzcr >= 4000:
            scores["spectral"] = max(0.0, 1.0 - (mzcr - 4000) / 8000)
        else:
            scores["spectral"] = 0.7
    else:
        scores["spectral"] = 0.5

    weights = {"silence": 0.30, "smoothness": 0.25, "duration": 0.20, "clipping": 0.15, "spectral": 0.10}
    scores["total"] = round(sum(scores[k] * weights[k] for k in weights), 4)
    return scores

# =============================
# Shared: save uploaded audio
# =============================
def _save_upload(audio: UploadFile, max_bytes: int | None = None) -> str:
    """Save an UploadFile to a temp file, enforcing a size cap."""
    limit = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0
    try:
        while True:
            chunk = audio.file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError(f"Upload exceeds {max(1, limit // (1024 * 1024))} MB limit")
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        _cleanup(tmp.name)
        raise

def _resolve_audio_path(audio: UploadFile, default_fn) -> tuple:
    """
    Returns (raw_tmp_path_or_None, audio_path_for_model, prep_tmp_path_or_None).
    Caller must delete both raw and prep paths when done.
    """
    raw_tmp = None
    prep_tmp = None

    if audio and audio.filename:
        raw_tmp = _save_upload(audio)
        raw_path = raw_tmp
    else:
        raw_path = default_fn()

    if raw_path:
        prep_tmp = preprocess_reference_audio(raw_path, target_sr=24000, target_duration=8.0)
        if prep_tmp == raw_path:
            prep_tmp = None   # failed gracefully, no extra file
        audio_path = prep_tmp if prep_tmp else raw_path
    else:
        audio_path = None

    return raw_tmp, audio_path, prep_tmp

def _cleanup(*paths):
    for p in paths:
        if p:
            try: os.unlink(p)
            except OSError: pass

# =============================
# Core generation helper
# =============================
def _generate_chunked(model, text: str, audio_path, gen_kwargs: dict) -> np.ndarray:
    """
    Split text into short chunks, generate each one, crossfade-concatenate.
    Assumes caller already holds `_MODEL_LOCK`.
    """
    chunks = split_into_chunks(text, max_chars=120)
    print(f"  Generating {len(chunks)} chunk(s): {[c[:25] + '…' if len(c) > 25 else c for c in chunks]}")

    audio_chunks = []
    for i, chunk in enumerate(chunks):
        kw = dict(gen_kwargs)
        if i == 0 and audio_path:
            kw["audio_prompt_path"] = audio_path
        wav = model.generate(chunk, language_id="ar", **kw)
        audio_chunks.append(wav.squeeze(0).numpy())

    return crossfade_concat(audio_chunks, model.sr, fade_ms=25)

def _error_json(msg: str, status_code: int = 400) -> Response:
    return Response(
        content=json.dumps({"error": msg}, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json",
    )

# =============================
# FastAPI
# =============================
app = FastAPI(title="IspeakMasri")

_STATIC_DIR = _APP_DIR / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"), media_type="text/html")

@app.get("/api/default-voice")
async def get_default_voice():
    p = default_audio_path()
    if p:
        return FileResponse(p, media_type="audio/wav")
    return Response(status_code=404)

@app.post("/api/detect-gender")
def api_detect_gender(audio: UploadFile = File(...)):
    """Detect gender from any supported uploaded audio file."""
    tmp_path = None
    try:
        tmp_path = _save_upload(audio)
        gender = detect_voice_gender(tmp_path)
        return {"gender": gender}
    except ValueError as e:
        return _error_json(str(e), 400)
    except Exception as e:
        print(f"detect-gender error: {e}")
        return {"gender": "unknown"}
    finally:
        _cleanup(tmp_path)


@app.post("/api/generate")
def api_generate(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    seed: int = Form(0),
    cfg_weight: float = Form(0.5),
    repetition_penalty: float = Form(2.5),
    min_p: float = Form(0.02),
    audio: UploadFile = File(None),
):
    """Single generation with preprocessed reference, chunking, and post-processing."""
    text = (text or "").strip()
    if not text:
        return _error_json("اكتب حاجة الأول", 400)
    if len(text) > 300:
        return _error_json("النص طويل شوية، الحد الأقصى 300 حرف", 400)

    raw_tmp = prep_tmp = None
    try:
        model = get_or_load_model()
        raw_tmp, audio_path, prep_tmp = _resolve_audio_path(audio, default_audio_path)

        actual_seed = seed if seed else random.randint(1, 999999)
        gen_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
        }

        with _MODEL_LOCK:
            set_seed(actual_seed)
            wav_np = _generate_chunked(model, text[:300], audio_path, gen_kwargs)

        wav_np = postprocess_audio(wav_np, model.sr)
        return Response(content=numpy_to_wav_bytes(model.sr, wav_np), media_type="audio/wav")

    except ValueError as e:
        return _error_json(str(e), 400)
    except RuntimeError as e:
        return _error_json(str(e), 503)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return _error_json("حصلت مشكلة، جرّب تاني", 500)
    finally:
        _cleanup(raw_tmp, prep_tmp)


@app.post("/api/generate-best")
def api_generate_best(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    seed: int = Form(0),
    cfg_weight: float = Form(0.5),
    count: int = Form(5),
    repetition_penalty: float = Form(2.5),
    min_p: float = Form(0.02),
    audio: UploadFile = File(None),
):
    """Generate N candidates with parameter jittering, score each, return ranked JSON."""
    text = (text or "").strip()
    if not text:
        return _error_json("اكتب حاجة الأول", 400)
    if len(text) > 300:
        return _error_json("النص طويل شوية", 400)

    count = min(max(count, 2), 7)
    raw_tmp = prep_tmp = None
    try:
        model = get_or_load_model()
        raw_tmp, audio_path, prep_tmp = _resolve_audio_path(audio, default_audio_path)
        base_seed    = seed if seed else random.randint(1, 999999)
        text_trimmed = text[:300]
        text_len     = len(text_trimmed)

        jitter_offsets = [
            {"exag":  0.00, "temp":  0.00, "cfg":  0.00},
            {"exag": -0.10, "temp": -0.08, "cfg": +0.10},
            {"exag": +0.12, "temp": +0.06, "cfg": -0.08},
            {"exag": -0.15, "temp": +0.10, "cfg": +0.05},
            {"exag": +0.18, "temp": -0.06, "cfg": -0.10},
            {"exag": -0.05, "temp": +0.15, "cfg": +0.12},
            {"exag": +0.08, "temp": -0.12, "cfg":  0.00},
        ]

        print(f"Generating {count} candidates for: '{text_trimmed[:40]}…'")

        candidates = []
        with _MODEL_LOCK:
            conds_ready = False
            for i in range(count):
                try:
                    set_seed(base_seed + i)
                    j = jitter_offsets[i] if i < len(jitter_offsets) else jitter_offsets[0]

                    gen_exag = max(0.1, min(2.0, exaggeration + j["exag"]))
                    gen_temp = max(0.05, min(5.0, temperature   + j["temp"]))
                    gen_cfg  = max(0.0,  min(1.0, cfg_weight   + j["cfg"]))

                    gen_kwargs = {
                        "exaggeration": gen_exag,
                        "temperature":  gen_temp,
                        "cfg_weight":   gen_cfg,
                        "repetition_penalty": repetition_penalty,
                        "min_p": min_p,
                    }

                    # Prepare speaker embeddings once, reuse across subsequent candidates
                    prompt_for_call = audio_path if not conds_ready else None
                    wav_np = _generate_chunked(model, text_trimmed, prompt_for_call, gen_kwargs)
                    conds_ready = True

                    wav_np = postprocess_audio(wav_np, model.sr)
                    scores = score_audio_quality(model.sr, wav_np, text_len)
                    wav_b64 = base64.b64encode(numpy_to_wav_bytes(model.sr, wav_np)).decode()

                    candidates.append({
                        "index": i,
                        "seed":  base_seed + i,
                        "params": {"exag": round(gen_exag, 3), "temp": round(gen_temp, 3), "cfg": round(gen_cfg, 3)},
                        "scores": scores,
                        "total_score": scores["total"],
                        "duration": round(len(wav_np) / model.sr, 2),
                        "audio_b64": wav_b64,
                    })
                    print(f"  #{i}: score={scores['total']:.3f} exag={gen_exag:.2f} temp={gen_temp:.2f} cfg={gen_cfg:.2f}")

                except Exception as e:
                    print(f"  #{i} failed: {e}")
                    candidates.append({"index": i, "seed": base_seed + i, "error": str(e), "total_score": 0})

        candidates.sort(key=lambda c: c.get("total_score", 0), reverse=True)
        for rank, c in enumerate(candidates):
            c["rank"] = rank
            c["recommended"] = (rank == 0 and c.get("total_score", 0) > 0.3)

        return Response(
            content=json.dumps({"candidates": candidates}, ensure_ascii=False),
            media_type="application/json",
        )

    except ValueError as e:
        return _error_json(str(e), 400)
    except RuntimeError as e:
        return _error_json(str(e), 503)
    except Exception as e:
        print(f"ERROR in generate-best: {type(e).__name__}: {e}")
        return _error_json("حصلت مشكلة، جرّب تاني", 500)
    finally:
        _cleanup(raw_tmp, prep_tmp)


@app.post("/api/calibrate-voice")
def api_calibrate_voice(
    audio: UploadFile = File(None),
    gender: str = Form("unknown"),
):
    """Quick voice calibration: 6 probes across param space on a short phrase."""
    CALIBRATION_TEXT = "أهلاً وسهلاً، أنا سعيد إني أتكلم معاك."

    CALIBRATION_PROBES = [
        {"exag": 0.30, "temp": 0.65, "cfg": 0.55, "label": "calm-guided"},
        {"exag": 0.40, "temp": 0.65, "cfg": 0.45, "label": "natural-mid"},
        {"exag": 0.55, "temp": 0.65, "cfg": 0.40, "label": "expressive-free"},
        {"exag": 0.35, "temp": 0.70, "cfg": 0.60, "label": "warm-guided"},
        {"exag": 0.50, "temp": 0.60, "cfg": 0.35, "label": "expressive-stable"},
        {"exag": 0.25, "temp": 0.70, "cfg": 0.65, "label": "gentle-tight"},
    ]

    if gender == "male":
        for p in CALIBRATION_PROBES:
            p["cfg"]  = min(1.0, p["cfg"]  + 0.08)
            p["exag"] = max(0.1, p["exag"] - 0.05)

    raw_tmp = prep_tmp = None
    try:
        model = get_or_load_model()
        raw_tmp, audio_path, prep_tmp = _resolve_audio_path(audio, default_audio_path)
        results = []
        base_seed = random.randint(1, 999999)

        with _MODEL_LOCK:
            conds_ready = False
            for i, probe in enumerate(CALIBRATION_PROBES):
                try:
                    set_seed(base_seed + i)
                    gen_kwargs = {
                        "exaggeration": probe["exag"],
                        "temperature":  probe["temp"],
                        "cfg_weight":   probe["cfg"],
                        "repetition_penalty": 2.5,
                        "min_p": 0.02,
                    }
                    prompt_for_call = audio_path if not conds_ready else None
                    wav_np = _generate_chunked(model, CALIBRATION_TEXT, prompt_for_call, gen_kwargs)
                    conds_ready = True

                    wav_np = postprocess_audio(wav_np, model.sr)
                    scores = score_audio_quality(model.sr, wav_np, len(CALIBRATION_TEXT))
                    results.append({
                        "label":  probe["label"],
                        "params": {"exag": probe["exag"], "temp": probe["temp"], "cfg": probe["cfg"]},
                        "score":  scores["total"],
                        "scores": scores,
                    })
                    print(f"  Calibration {probe['label']}: {scores['total']:.3f}")
                except Exception as e:
                    print(f"  Calibration probe {i} failed: {e}")

        if not results:
            return _error_json("كل التجارب فشلت", 500)

        best = max(results, key=lambda r: r["score"])
        print(f"  Best: {best['label']} (score={best['score']:.3f})")

        return Response(
            content=json.dumps({"best": best, "all": results, "gender": gender}, ensure_ascii=False),
            media_type="application/json",
        )

    except ValueError as e:
        return _error_json(str(e), 400)
    except RuntimeError as e:
        return _error_json(str(e), 503)
    except Exception as e:
        print(f"ERROR in calibrate-voice: {type(e).__name__}: {e}")
        return _error_json("حصلت مشكلة أثناء المعايرة", 500)
    finally:
        _cleanup(raw_tmp, prep_tmp)

# =============================
# Launch
# =============================
if __name__ == "__main__":
    print("Starting IspeakMasri on http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
