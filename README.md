---
title: Egyptian Arabic TTS
emoji: 🌎
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: Chatterbox multilanguil Arabic TTS Demo
license: mit
---

# IspeakMasri — أنا بتكلم مصري

Egyptian Arabic Text-to-Speech & Voice Cloning studio built on **Chatterbox Multilingual TTS** (`ChatterboxMultilingualTTS`), served with a **FastAPI** backend and a custom RTL web interface.

Runs fully offline from the local `checkpoint/` directory (`HF_HUB_OFFLINE=1` by default).

## Features

- **Voice Cloning (Zero-Shot)**: Upload a WAV/audio clip, drag-and-drop, record from your microphone, or use the built-in default voice (`female_voice.wav`).
- **Automatic Gender Detection**: Estimates speaker pitch ($F_0$ via autocorrelation in `detect_voice_gender()`) and applies gender-tuned parameter presets (`female` / `male` / `unknown`).
- **Voice Auto-Calibration (`/api/calibrate-voice`)**: Runs 6 parameter probes on a short Egyptian phrase to find the best `exaggeration`, `cfg_weight`, and `temperature` for any newly uploaded voice.
- **Best-of-N Generation (`/api/generate-best`)**: Choose **2, 3, 5, or 7** candidates with parameter jittering around your settings, automatically scored on signal heuristics (silence ratio, RMS smoothness, duration sanity, clipping, spectral quality) and ranked with the top take auto-selected.
- **Quality & Audio Pipeline**:
  - **Reference preprocessing**: Resamples reference WAV to 24 kHz, selects the highest-energy 8 s voiced segment, and peak-normalizes to -3 dBFS.
  - **Sentence chunking & crossfade**: Splits longer Arabic text on punctuation (`. ! ? ؟ ،` up to 120 chars/chunk) and crossfade-joins chunks (25 ms) for steadier pacing.
  - **Post-processing**: 80 Hz high-pass filter, leading/trailing silence trim, and -3 dBFS peak normalization.
  - **Sampling defaults**: Uses `repetition_penalty=2.5` and `min_p=0.02` to reduce looping and keep delivery natural.
- **Interactive Player & Recent Takes**:
  - 80-bar decoded waveform with click-to-seek and playhead tracking.
  - Word-by-word karaoke highlighting, playback speed (`0.75×`, `1×`, `1.25×`), and WAV download.
  - Saves the **last 5 takes** locally in browser `IndexedDB` (`ispeakmasri_db`) across page reloads.

---

## Model Architecture

The synthesis core lives in [`src/chatterbox/mtl_tts.py`](src/chatterbox/mtl_tts.py) (`ChatterboxMultilingualTTS`) and loads from `checkpoint/`:

| Component | File | Role |
|---|---|---|
| **Voice Encoder** | `ve.pt` | Extracts speaker identity embedding from reference audio (16 kHz) |
| **Tokenizer** | `grapheme_mtl_merged_expanded_v1.json` | Normalizes and tokenizes Arabic text (`language_id="ar"`) |
| **T3** | `t3_mtl23ls_v2.safetensors` | Llama-backbone token-to-token transformer predicting discrete S3 speech tokens |
| **S3Gen** | `s3gen.pt` | Flow-matching + HiFi-GAN vocoder converting speech tokens + prompt into 24 kHz audio |
| **Built-in Conditionals** | `conds.pt` | Fallback precomputed speaker/prompt conditionals |

---

## Quick Start (Windows PowerShell)

1. **Create and activate a virtual environment** (Python 3.10–3.12):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. **Install PyTorch** (pick CUDA build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if using an NVIDIA GPU):
   ```powershell
   # Example (check pytorch.org for your CUDA version):
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify CUDA (optional)**:
   ```powershell
   python "test cuda.py"
   ```

5. **Run the server**:
   ```powershell
   python app.py
   ```
   Open **http://localhost:7860** in your browser.

---

## Project Structure

```text
.
├── app.py                   # FastAPI server, audio preprocessing/postprocessing, scoring & APIs
├── static/
│   └── index.html           # Custom RTL web UI (Cairo / Noto Naskh Arabic / Reem Kufi)
├── src/chatterbox/
│   ├── mtl_tts.py           # ChatterboxMultilingualTTS inference & conditioning
│   └── models/              # T3, S3Gen, S3Tokenizer, VoiceEncoder, MTLTokenizer
├── checkpoint/              # Local offline model weights (~3 GB, gitignored)
├── female_voice.wav         # Default reference voice
├── test cuda.py             # Quick PyTorch CUDA verification script
├── requirements.txt         # Python dependencies
└── TUTORIAL.md              # Detailed walkthrough, API docs & tuning guide
```

---

## API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Serves `static/index.html` |
| `/api/default-voice` | `GET` | Streams `female_voice.wav` |
| `/api/detect-gender` | `POST` | Estimates speaker gender (`female` / `male` / `unknown`) from uploaded WAV |
| `/api/calibrate-voice` | `POST` | Tests 6 parameter combos on an Egyptian phrase and returns best settings |
| `/api/generate` | `POST` | Generates single WAV clip (max 300 chars) |
| `/api/generate-best` | `POST` | Generates 2–7 jittered candidates, scores them, returns ranked base64 WAV JSON |

See [TUTORIAL.md](TUTORIAL.md) for the full guide, presets reference, and troubleshooting.
