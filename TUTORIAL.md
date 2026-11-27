# IspeakMasri Tutorial & Technical Guide

This guide covers setting up, running, tuning, and understanding the **IspeakMasri** Egyptian Arabic TTS application as currently implemented in `app.py`, `static/index.html`, and `src/chatterbox/`.

---

## 1. Environment Setup (Windows PowerShell)

Use a virtual environment with Python 3.10–3.12. An NVIDIA GPU with CUDA is strongly recommended—CPU inference works (`DEVICE` switches automatically via `torch.cuda.is_available()`), but generation (especially Best-of-N or calibration) is much slower on CPU.

```powershell
cd "c:\work\projects\egyptian tts\MASRAWI tts"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch first using the command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for your CUDA version (or CPU-only if you don't have an NVIDIA GPU), then install project requirements:

```powershell
pip install -r requirements.txt
```

Verify PyTorch and GPU visibility:

```powershell
python "test cuda.py"
```

---

## 2. Offline Checkpoints & Startup

`app.py` sets `HF_HUB_OFFLINE=1` by default and loads the model locally via `ChatterboxMultilingualTTS.from_local(_APP_DIR / "checkpoint", DEVICE)`. Make sure `checkpoint/` contains:

- `ve.pt` (VoiceEncoder)
- `t3_mtl23ls_v2.safetensors` (T3 transformer weights)
- `s3gen.pt` (S3Gen flow-matching + HiFi-GAN vocoder)
- `grapheme_mtl_merged_expanded_v1.json` (multilingual tokenizer)
- `conds.pt` (optional built-in conditionals)

Start the FastAPI server:

```powershell
python app.py
```

When startup completes, open **http://localhost:7860** in your browser.

---

## 3. Using the Web UI (`static/index.html`)

### Text Input
- Type up to **300 characters** of Egyptian Arabic (e.g., `ازيك يا باشا، عامل ايه؟`), pick one of the built-in example chips, or click **جرّب مثال** (`Try one`).
- Press **Ctrl + Enter** or click **ولّد الصوت** (`Generate voice`).

### Reference Voice & Gender Detection
- By default, generation uses `female_voice.wav` (marked `♀ أنثى`).
- Upload an audio file (**ارفع ملف**), drag-and-drop onto the Voice card, or record from your microphone (**سجّل صوتك**).
- For `.wav` uploads, the frontend automatically calls `POST /api/detect-gender`, displays `♀ أنثى` or `♂ ذكر`, and switches the Style presets to gender-tuned values.

### Voice Auto-Calibration
- After uploading a new voice file, the **معايرة الصوت — Auto-tune params** button appears inside the Voice card.
- Clicking it calls `POST /api/calibrate-voice`, runs 6 quick probes across `exaggeration`, `cfg_weight`, and `temperature` on a fixed Egyptian calibration sentence (`أهلاً وسهلاً، أنا سعيد إني أتكلم معاك.`), and applies the highest-scoring parameters directly to your sliders.

### Style Presets & Sliders
Each gender profile adjusts the three presets automatically:

| Gender | Preset | Exaggeration | CFG / Pace | Temperature |
|---|---|---|---|---|
| **Female** | Calm (`هادي`) | `0.30` | `0.50` | `0.60` |
| **Female** | Natural (`طبيعي`) | `0.40` | `0.40` | `0.65` |
| **Female** | Expressive (`تعبيري`) | `0.65` | `0.35` | `0.72` |
| **Male** | Calm (`هادي`) | `0.25` | `0.55` | `0.55` |
| **Male** | Natural (`طبيعي`) | `0.35` | `0.50` | `0.60` |
| **Male** | Expressive (`تعبيري`) | `0.55` | `0.45` | `0.68` |
| **Unknown** | Calm / Natural / Expressive | `0.28` / `0.38` / `0.60` | `0.52` / `0.45` / `0.40` | `0.58` / `0.63` / `0.70` |

Open **إعدادات متقدمة — Advanced** to set a fixed `Seed` (`0` = random) or tune `Temperature` (`0.05`–`5.0`).

### Best-of-N Mode (`ولّد الأحسن`)
1. Toggle **ولّد الأحسن** next to the Generate button.
2. Pick how many candidates to generate: **2, 3, 5, or 7**.
3. Click Generate. The backend generates `count` takes using parameter jittering (varying `exag`, `temp`, and `cfg` around your current slider values), scores each clip, ranks them, and auto-selects the top-scoring take (`⭐` / `✓ الأحسن`).
4. You can preview any candidate individually or click **استخدم ده** to load it into the main player.

### Player & Recent Takes
- The player renders an 80-bar waveform (`AudioContext.decodeAudioData`), synced word-level karaoke highlighting, playback speed (`0.75×`, `1×`, `1.25×`), and WAV download.
- **آخر التوليدات (Recent takes)** automatically stores the **last 5 generated takes** in browser `IndexedDB` (`ispeakmasri_db`), so they remain playable and downloadable even after refreshing the page.

---

## 4. How the Synthesis Pipeline Works

For every generation request in `app.py`, audio flows through:

1. **Reference Preprocessing (`preprocess_reference_audio`)**
   - Reads uploaded audio (capped to 30 s via `librosa.load`), resamples to **24 kHz** with `librosa.resample`, scans with a 25 ms sliding window to locate the **highest-energy 8-second voiced segment**, and peak-normalizes to **-3 dBFS**. (Chatterbox clips encoder conditioning to 6 s and decoder conditioning to 10 s, so feeding a clean 8 s voiced window avoids silent/noisy segments.)
2. **Text Chunking (`split_into_chunks`)**
   - Splits input on sentence endings (`. ! ? ؟`) and secondary Arabic commas (`،`) so each chunk stays at **≤ 120 characters** (merging tiny trailing fragments `< 20` chars). Short chunks prevent pacing drift and end-of-utterance artifacts.
3. **Model Inference (`_generate_chunked` → `ChatterboxMultilingualTTS.generate`)**
   - On chunk `0`, `prepare_conditionals()` encodes the reference speaker (`VoiceEncoder`), decoder prompt features (`S3Gen.embed_ref`), and conditioning speech tokens (`S3Tokenizer`). Subsequent chunks reuse the loaded conditionals.
   - `punc_norm()` cleans punctuation; `MTLTokenizer` prepends the `[ar]` language context and maps text to tokens.
   - `T3` generates discrete speech tokens with `repetition_penalty=2.5` and `min_p=0.02`.
   - `S3Gen` synthesizes the 24 kHz waveform.
4. **Chunk Joining & Post-Processing (`crossfade_concat` → `postprocess_audio`)**
   - Multi-chunk outputs are joined with a **25 ms linear crossfade**.
   - Audio passes through an **80 Hz 1st-order high-pass filter**, **leading/trailing silence trimming** (`-40 dBFS` threshold, `30 ms` pad), and **-3 dBFS peak normalization**.
5. **Quality Scoring (`score_audio_quality`, Best-of-N & Calibration)**
   - Weighted heuristic score (`0.0–1.0`):
     - Silence ratio (`0.30` weight, sweet spot 15–30%)
     - RMS energy smoothness (`0.25` weight)
     - Duration-to-character sanity (`0.20` weight)
     - Clipping detection (`0.15` weight)
     - Zero-crossing spectral check (`0.10` weight)

---

## 5. FastAPI Endpoints Reference

- `GET /` — Serves `static/index.html`.
- `GET /api/default-voice` — Returns `female_voice.wav`.
- `POST /api/detect-gender` — Form field: `audio` (file). Returns `{"gender": "female" | "male" | "unknown"}`.
- `POST /api/calibrate-voice` — Form fields: `audio` (optional file), `gender` (`"female" | "male" | "unknown"`). Returns `{"best": {...}, "all": [...], "gender": ...}`.
- `POST /api/generate` — Form fields: `text`, `exaggeration` (0.5), `temperature` (0.8), `seed` (0), `cfg_weight` (0.5), `repetition_penalty` (2.5), `min_p` (0.02), `audio` (optional file, max 25 MB). Returns `audio/wav`.
- `POST /api/generate-best` — Same form fields plus `count` (clamped `2–7`, default `5`). Returns JSON with ranked `candidates` (`index`, `rank`, `recommended`, `total_score`, `scores`, `params`, `duration`, `audio_b64`).

---

## 6. Tips for Better Egyptian Arabic Output (No Training Needed)

- **Clean Egyptian reference clip**: Use 6–12 seconds of dry, single-speaker Egyptian speech with no background music, reverb, or long pauses.
- **Natural Egyptian spelling**: Write colloquial words consistently (e.g., `دلوقتي`, `إزاي`, `علشان`, `أوي`) and use punctuation (`.` / `،` / `؟`) to guide pauses cleanly.
- **Tune per speaker**: Run **Auto-tune** once after uploading or recording a new speaker, then use **Best-of-3** or **Best-of-5** for important lines. Male voices usually sound cleaner with slightly lower `exaggeration` (`0.25–0.45`) and higher `cfg_weight` (`0.45–0.60`).

---

## 7. Note on Fine-Tuning

This repository is an **inference and web-app codebase**—there are no training scripts, dataset manifests, or trainer loops included. `T3.loss()` in `src/chatterbox/models/t3/t3.py` provides the text/speech cross-entropy loss primitives if you build or attach a compatible fine-tuning pipeline externally.

---

## 8. Troubleshooting

- **`Missing dependencies. Install with: pip install fastapi uvicorn python-multipart`**: Run `pip install -r requirements.txt` inside your activated `.venv`.
- **`CUDA is False` / slow generation**: Ensure you installed a CUDA-enabled PyTorch build matching your NVIDIA driver (`python "test cuda.py"`).
- **Large or exotic audio uploads**: Uploads are capped at 25 MB and decoded via `librosa.load` (first 30 s). Browser mic recordings are encoded directly to 16-bit PCM `.wav` in the browser; compressed `.mp3`/`.m4a` uploads require system decoder support (`soundfile`/`audioread`).
- **Model failed to load**: Verify all required files exist in `checkpoint/` and are complete (~3 GB total). Note `HF_HUB_OFFLINE=1` prevents accidental network downloads; remove or override it if intentionally downloading missing base files via `ChatterboxMultilingualTTS.from_pretrained()`.
