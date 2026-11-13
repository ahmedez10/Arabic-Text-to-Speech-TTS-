---
title: Egyptian Arabic TTS
emoji: 🌎
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: Chatterbox multilingual Arabic TTS Demo
license: mit
---

# Arabic-Text-to-Speech-TTS- · IspeakMasri 🇪🇬

Egyptian Arabic (العامية المصرية) text-to-speech with optional **voice cloning** from a short reference recording.
Final project for **DEPI** (Digital Egypt Pioneers Initiative), Machine Learning Engineer track.

The repository contains:

1. a **Gradio demo** (`app.py`) that turns Egyptian Arabic text into speech;
2. the **Chatterbox Multilingual TTS** inference code it runs on (`src/chatterbox/`);
3. the **dataset preparation and quality-check work** for ~9.5k Egyptian Arabic utterances (`egyptian_tts_data_preparation.ipynb`, `P2-TTS/`).

> **TL;DR (بالعربي):** اكتب نص بالمصري، ارفع صوت مرجعي (اختياري)، والموديل يقرأ النص بنفس الصوت.
> الريبو فيه كود التشغيل (inference) وتجهيز الداتا فقط. **ملفات الموديل (`checkpoint/`) مش موجودة في الريبو** ولازم تتحط محلياً.
> مفيش كود training هنا.

---

## Table of contents

1. [What this project is (and is not)](#1-what-this-project-is-and-is-not)
2. [Repository layout](#2-repository-layout)
3. [How it works](#3-how-it-works)
4. [Generation parameters](#4-generation-parameters)
5. [Dataset and data preparation](#5-dataset-and-data-preparation)
6. [Installation and running](#6-installation-and-running)
7. [Using the model from Python](#7-using-the-model-from-python)
8. [Configuration and customisation](#8-configuration-and-customisation)
9. [Deploying to Hugging Face Spaces](#9-deploying-to-hugging-face-spaces)
10. [Known limitations and issues](#10-known-limitations-and-issues)
11. [Troubleshooting](#11-troubleshooting)
12. [Credits and licence](#12-credits-and-licence)

---

## 1. What this project is (and is not)

**Is**

- An inference demo built on [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (ResembleAI). It loads the multilingual T3 text-to-speech-token model, swaps in a checkpoint file named `t3_mtl23ls_v2.safetensors`, and always generates with `language_id="ar"`.
- A prepared Egyptian Arabic dataset split (train / validation / test) plus the notebooks used to clean the text and check transcript quality with Whisper.

**Is not**

- A training repository. There is no trainer, optimiser setup, batching code or checkpoint-saving loop. `T3.loss()` exists in `src/chatterbox/models/t3/t3.py`, but that is only the loss function, not a training recipe.
- A guaranteed *Egyptian-only* model. `language_id="ar"` selects generic Arabic. Dialect quality comes from the checkpoint weights and from the reference audio you provide. The UI examples are only sample prompts; they do not teach the model anything.
- Self-contained. The model weights (about 3 GB) and the dataset audio are **not** in this repository (see [section 6](#6-installation-and-running)).

---

## 2. Repository layout

```
.
├── app.py                               # Gradio UI + model loading + generate_tts_audio()
├── requirements.txt                     # Python dependencies (PyTorch NOT included)
├── female_voice.wav                     # Default reference voice (stored with Git LFS)
├── test cuda.py                         # Tiny script: is CUDA available, which GPU
├── TUTORIAL.md                          # Longer local-run guide and code walkthrough
├── egyptian_tts_data_preparation.ipynb  # Text cleaning + number/time/date -> Egyptian words
├── P2-TTS/
│   ├── metadata_train.csv               # 7,612 rows
│   ├── metadata_validation.csv          #   951 rows
│   ├── metadata_test.csv                #   952 rows
│   └── check_Identicality.ipynb         # Whisper transcription vs. metadata text (QA)
└── src/chatterbox/
    ├── mtl_tts.py                       # ChatterboxMultilingualTTS: loading, conditioning, generate()
    ├── tts.py, vc.py                    # English TTS and voice-conversion variants (not used by app.py)
    └── models/
        ├── t3/                          # Llama-based text -> speech-token model
        ├── s3gen/                       # Speech tokens -> 24 kHz waveform (flow matching + HiFi-GAN)
        ├── s3tokenizer/                 # Waveform -> discrete speech tokens (16 kHz input)
        ├── voice_encoder/               # Speaker embedding from reference audio
        └── tokenizers/tokenizer.py      # MTLTokenizer: text normalisation + token IDs
```

---

## 3. How it works

### 3.1 End-to-end pipeline

```
                      reference audio (optional, default: female_voice.wav)
                                   │
                ┌──────────────────┴───────────────────┐
                │ prepare_conditionals()               │
                │  • VoiceEncoder  → speaker embedding │
                │  • S3Tokenizer   → 150 prompt tokens │
                │  • S3Gen.embed_ref → decoder prompt  │
                └──────────────────┬───────────────────┘
                                   │ conditioning
 text ─► punc_norm ─► MTLTokenizer ─► T3 (Llama 520M) ─► speech tokens ─► S3Gen ─► 24 kHz waveform
        (cleanup)    ("[ar]" + IDs)    classifier-free      (discrete)     (token → mel → audio)
                                       guidance, sampling
```

In plain words: **T3 decides *what* is said** (as discrete speech tokens), **S3Gen turns those tokens into sound**, and the **reference clip decides *whose voice* and *what delivery* it has**.

### 3.2 Step by step

1. **`app.py`** picks `cuda` or `cpu`, loads the model once into a global (`get_or_load_model()`), and builds the Gradio interface.
2. **`generate_tts_audio()`** strips the text, rejects empty input, truncates to **300 characters**, chooses the uploaded reference (or falls back to `female_voice.wav`), and calls `model.generate(text, language_id="ar", ...)`. If a seed other than 0 is given it seeds `torch`, `random` and `numpy`.
3. **`prepare_conditionals()`** (in `mtl_tts.py`) loads the reference with librosa at 24 kHz and builds three things:
   - S3Gen reference features from the **first 10 s** (`DEC_COND_LEN = 10 × 24000` samples);
   - T3 **prompt speech tokens** (up to 150, `speech_cond_prompt_len`) from the **first 6 s** at 16 kHz (`ENC_COND_LEN = 6 × 16000`);
   - a **speaker embedding** from the voice encoder (256-dim, averaged over the clip).
   The `exaggeration` value is stored in the T3 conditioning as `emotion_adv`.
4. **`punc_norm()`** cleans punctuation: collapses whitespace; rewrites `...`, `…`, `:`, `;`, ` - ` to commas; normalises curly quotes and dashes; and appends `.` if the text does not end with sentence punctuation.
5. **`MTLTokenizer.encode()`** lowercases, applies Unicode **NFKD** normalisation, prepends the language token `[ar]`, replaces spaces with a `[SPACE]` token, and encodes with the vocabulary in `grapheme_mtl_merged_expanded_v1.json`. NFKD splits composed Arabic letters such as `أ`, `إ`, `آ` into a base letter plus a combining mark, so this is expected, not a bug.
6. **Start/stop tokens** are added (`start_text_token = 255`, `stop_text_token = 0`) and the sequence is **duplicated into a batch of two**. One copy is the conditional pass and the other is used for **classifier-free guidance** (CFG).
7. **T3 inference** autoregressively generates speech tokens (up to `max_new_tokens = 1000`) with temperature, CFG weight, repetition penalty, `min_p` and `top_p`.
8. Only the conditional sequence is kept; **invalid tokens are dropped** (`drop_invalid_tokens`).
9. **S3Gen** decodes the tokens into a waveform using the reference features. The result is mono audio at **24 kHz** (`S3GEN_SR`).
10. `app.py` returns `(sample_rate, numpy_array)` to the Gradio `Audio` component.

### 3.3 Model components

| Component | Code | What it does | Key facts from the code |
|---|---|---|---|
| **VoiceEncoder** | `models/voice_encoder/` | Converts reference audio to a speaker embedding | 256-dim embedding (`speaker_embed_size`) |
| **S3Tokenizer** | `models/s3tokenizer/` | Converts 16 kHz audio to discrete speech tokens | Used for the T3 prompt tokens |
| **T3** | `models/t3/t3.py` | Predicts speech tokens from text + conditioning | `Llama_520M` backbone, learned positional embeddings, perceiver resampler for the prompt, speech vocab 8,194, multilingual text vocab 2,454 |
| **S3Gen** | `models/s3gen/` | Speech tokens → mel → waveform | Transformer encoder, flow-matching decoder (a Matcha-TTS-style design), HiFi-GAN vocoder with an F0 predictor |
| **MTLTokenizer** | `models/tokenizers/tokenizer.py` | Text → token IDs | Handles 23 languages; Arabic uses only the generic path (no extra preprocessing) |

Supported language codes are listed in `SUPPORTED_LANGUAGES` in `mtl_tts.py`; this project only uses `ar`.

### 3.4 Watermark

`ChatterboxMultilingualTTS.__init__` creates a Perth watermarker, but the line that applies it in `generate()` is **commented out**, so output audio is **not** watermarked in this repository. Re-enable it if you need it.

---

## 4. Generation parameters

| Parameter | Default | UI control | Effect |
|---|---|---|---|
| `exaggeration` | 0.5 | Slider 0.25–2.0 | Expressiveness / emotion intensity of delivery. Higher is more dramatic and can become unstable. |
| `cfg_weight` | 0.5 | Slider 0.0–1.0 ("CFG / Pace") | Strength of text conditioning via classifier-free guidance; also affects pacing. Lower values tend to be slower and more relaxed. |
| `temperature` | 0.8 | Slider 0.05–5.0 (advanced) | Sampling randomness. Lower is more stable, higher is more varied. |
| `seed` | 0 | Number (advanced) | Non-zero makes runs reproducible. `0` means "do not set a seed". |
| `repetition_penalty` | 2.0 | not exposed | Discourages repeated speech tokens. |
| `min_p` | 0.05 | not exposed | Drops very unlikely tokens before sampling. |
| `top_p` | 1.0 | not exposed | Nucleus sampling cut-off (1.0 means disabled). |
| `max_new_tokens` | 1000 | not exposed | Hard cap on generated speech tokens per call (hard-coded in `generate()`). |

These are **generation controls, not dialect switches.** Change one at a time when comparing runs.

---

## 5. Dataset and data preparation

### 5.1 Splits in `P2-TTS/`

Pipe-separated (`|`) UTF-8 files with a BOM (`utf-8-sig`). Columns:

| Column | Meaning |
|---|---|
| `audio_file` | Relative path such as `wavs/sales_5_00122.wav` (the audio itself is **not** in the repo) |
| `text` | Transcript as written in the dataset |
| `text_normalized` | Transcript after the normalisation in the notebook (numbers, times, dates → Egyptian words) |
| `speaker_name` | `SPEAKER_01` or `SPEAKER_02` |

| Split | Rows | SPEAKER_01 | SPEAKER_02 | sales | general | medical |
|---|---:|---:|---:|---:|---:|---:|
| train | 7,612 | 3,888 | 3,724 | 3,988 | 2,398 | 1,226 |
| validation | 951 | 486 | 465 | 504 | 266 | 181 |
| test | 952 | 486 | 466 | 482 | 311 | 159 |
| **total** | **9,515** | | | | | |

The domain (sales / general / medical) is read from the file-name prefix. Average transcript length is about 180 characters (max 1,260 in train). There is **no file overlap** between the three splits.
The split was made 80 / 10 / 10 with `random_state=42`, stratified by speaker when possible (last cells of `check_Identicality.ipynb`).

### 5.2 `egyptian_tts_data_preparation.ipynb`

Cleans the transcripts and makes numbers readable aloud in Egyptian dialect, so the TTS does not have to guess how to pronounce digits.

| Step | What it does |
|---|---|
| Load and clean | Reads `metadata_train.csv` / `metadata_eval.csv` from the NileTTS dataset on Hugging Face, trims and collapses whitespace |
| Digit unification | Arabic-Indic digits (`١٢٣`) → `123`; `٫` → `.`; removes thousands commas (`1,500` → `1500`) |
| Numbers → words | Egyptian forms: `11` إحداشر, `14` أربعتاشر, `300` تلتمية, `2000` ألفين; negatives use "ناقص"; ≥ 1 billion falls back to `num2words(lang="ar")` |
| Decimals and percent | `7.5%` → سبعة فاصلة خمسة في المية |
| Time | `7:30` → سبعة ونص, `7:45` → تمانية إلا ربع, `7:20` → سبعة وتلت (12-hour clock) |
| Date | `15/10/2026` → day, month name, year in words. Invalid dates are left untouched |
| Save | Writes `cleaned_metadata_*.csv` with a new `text_normalized` column |

Note: the regex step leaves dates and times alone for their dedicated functions; the notebook's `normalize_egyptian_text()` itself only runs whitespace, digit and number expansion.

### 5.3 `P2-TTS/check_Identicality.ipynb`

A quality-assurance notebook: does the **audio actually say what the transcript says**?

1. Merge train + eval metadata and drop duplicate `audio_file` rows.
2. Transcribe each recording with **faster-whisper** (`large-v3` on CPU, int8, then `small` for the remainder), `language="ar"`, `beam_size=1`, `vad_filter=True`.
3. Save `original_text`, `predicted_text`, processing time, an approximate **confidence score** (derived from segment log-probabilities), status and error message. Runs are resumable: rows with `status == "completed"` are skipped.
4. List recordings with confidence below **0.5** and play them next to both texts for manual review.
5. Apply a global spelling fix (`بالزبط` → `بالظبط`) across metadata and result files.
6. Create the final stratified train / validation / test split.

The paths in this notebook (`D:\AI\speech_dataset\...`) are from the author's machine and must be edited before re-running.

---

## 6. Installation and running

### 6.1 Requirements

- Python **3.10 or 3.11**, in a virtual environment.
- **PyTorch**, installed first from <https://pytorch.org/get-started/locally/> (it is *not* in `requirements.txt`).
- An NVIDIA GPU is strongly recommended. CPU works but is slow.
- About **3 GB** of free disk for the checkpoint, plus RAM/VRAM for a 520M-parameter T3 model and S3Gen.
- **Git LFS**, because `female_voice.wav` is an LFS file. Without it you get a tiny text pointer instead of audio.

### 6.2 Set up

```bash
git lfs install
git clone https://github.com/ahmedez10/Arabic-Text-to-Speech-TTS-.git
cd Arabic-Text-to-Speech-TTS-

python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate

# 1) install PyTorch for your OS/CUDA from pytorch.org, then:
pip install -r requirements.txt
pip install huggingface_hub
```

`huggingface_hub` is imported by the code but is not listed in `requirements.txt`. The code also imports `tokenizers` (installed with `transformers`).

### 6.3 Add the model checkpoint

`app.py` loads everything from a local `checkpoint/` folder and sets `HF_HUB_OFFLINE=1`, so **it never downloads anything at run time**. Create the folder next to `app.py` with these files:

```
checkpoint/
├── ve.pt                                  # voice encoder
├── t3_mtl23ls_v2.safetensors              # T3 weights (the Egyptian checkpoint)
├── s3gen.pt                               # speech-token → waveform model
├── grapheme_mtl_merged_expanded_v1.json   # text tokenizer vocabulary
└── conds.pt                               # optional built-in voice conditioning
```

- `ve.pt`, `s3gen.pt` and the tokenizer JSON come from the base repo [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox). `ChatterboxMultilingualTTS.from_pretrained()` can download them for you.
- **TODO (project owner):** add the download link for your Egyptian `t3_mtl23ls_v2.safetensors`. This repository does not state where it is hosted.
- The tokenizer tries to download `Cangjie5_TC.json` (used only for Chinese) when it starts. With offline mode on, that fails and only logs a warning; Arabic is unaffected.

`checkpoint/` and `checkpoints/` are in `.gitignore`, so the weights never get committed by accident.

### 6.4 Run

```bash
python app.py
```

Open the printed address (usually <http://127.0.0.1:7860>). Enter Egyptian Arabic text, optionally upload or record a reference voice, adjust the sliders, and press **ولّد الصوت**. Buttons: **مثال مصري** inserts a random example sentence, **مسح** clears the text box.

Check the GPU first if in doubt:

```bash
python "test cuda.py"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

(`test cuda.py` crashes on a machine without a GPU because it calls `get_device_name(0)` unconditionally.)

---

## 7. Using the model from Python

```python
import soundfile as sf
import torch
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_local("checkpoint", device)

wav = model.generate(
    "امبارح كنت بتفرج على فيلم، وفجأة الكهربا قطعت.",
    language_id="ar",
    audio_prompt_path="female_voice.wav",  # omit to use conds.pt if present
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
)                                           # torch tensor, shape (1, samples)

sf.write("out.wav", wav.squeeze(0).numpy(), model.sr)   # 24 kHz
```

Tips

- Without `audio_prompt_path`, `generate()` needs either `conds.pt` in the checkpoint folder or a previous call to `prepare_conditionals()`.
- Keep inputs to a few sentences. The demo caps text at 300 characters, and generation is capped at 1000 speech tokens.
- Spell dialect words consistently (for example `دلوقتي`, `إزاي`). The model was shaped by transcripts with a specific spelling convention.

---

## 8. Configuration and customisation

| What | Where |
|---|---|
| Default text and default reference voice | `LANGUAGE_CONFIG["ar"]` in `app.py` |
| "مثال مصري" sentences | `EGYPTIAN_EXAMPLES` in `app.py` |
| Text length limit | `text_input[:300]` in `generate_tts_audio()` |
| Slider ranges and defaults | the `gr.Slider` calls in `app.py` |
| Look and feel (RTL layout, red/dark theme, Cairo font) | `CUSTOM_CSS` and `ISPEAKMASRI_THEME` in `app.py` |
| Punctuation rules | `punc_norm()` in `mtl_tts.py` |
| Hidden sampling parameters | `generate()` signature in `mtl_tts.py` |

For better dialect output **without** training: use a clean, dry, single-speaker Egyptian reference clip (no music, noise or long silences), keep spelling consistent, and test with a fixed list of Egyptian sentences covering pronouns, negation, questions, numbers and loanwords. Avoid editing the tokenizer vocabulary: the model's embedding rows correspond to it.

---

## 9. Deploying to Hugging Face Spaces

The YAML block at the top of this file is the Spaces configuration (`sdk: gradio`, `app_file: app.py`). `app.py` already wraps the generator with `spaces.GPU` when the `spaces` package is available and falls back to a no-op locally.
Before deploying you must supply the checkpoint (it is git-ignored and `HF_HUB_OFFLINE=1` is forced), for example by uploading it to the Space or changing `app.py` to download from your own model repository. Also align the Gradio version (see below).

---

## 10. Known limitations and issues

- **No weights or audio in the repo.** The demo cannot run until `checkpoint/` exists; the dataset metadata points to `wavs/` files that are not included.
- **No fine-tuning code.** If the project goal is fine-tuning XTTS-v2 or Chatterbox, that pipeline still has to be written. The data splits are ready for it.
- **Generic `ar` language ID.** The model does not distinguish Egyptian from Modern Standard Arabic by itself.
- **300-character input limit** in the UI; very long text would also hit the 1000-token cap.
- **Dependency pins.** `requirements.txt` leaves `gradio` unpinned, while the Spaces header pins `5.29.0`. `app.py` passes `css=` and `theme=` to `launch()`, which recent Gradio releases expect but older ones reject (the `except TypeError` fallback passes them too). Pin one working version. `mcp_server=True` needs the Gradio MCP extra and falls back automatically if unsupported. `faster-whisper` (used by the QA notebook) and `torch` are not listed.
- **Not tested here.** This README was written from the source code; the app was not executed because the checkpoint is unavailable.
- **Dataset provenance.** The data-prep notebook downloads metadata from `KickItLikeShika/NileTTS-dataset`. Confirm the dataset's licence and the recording consent terms before redistributing audio or generated voices.
- **Voice cloning ethics.** Only clone voices you have permission to use, and do not present synthetic speech as a real person.
- In `check_Identicality.ipynb`, one of the repeated "continue Whisper-small" cells contains a stray character (`encoding="utf-8-sig"z`) that causes a `SyntaxError`; the duplicate cells next to it are fine.

---

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `CRITICAL: Failed to load model` on start | `checkpoint/` missing or incomplete. Check all file names in [6.3](#63-add-the-model-checkpoint). |
| `FileNotFoundError ... ve.pt` | Same as above, or running from a different folder. The app resolves `checkpoint/` relative to `app.py`. |
| Reference audio fails to load / garbage voice | `female_voice.wav` is an LFS pointer. Run `git lfs install && git lfs pull`. |
| `ModuleNotFoundError: spaces` | Optional; `app.py` already falls back. For other missing modules install them (`huggingface_hub`, `tokenizers`). |
| `TypeError` from `demo.launch(css=..., theme=...)` | Gradio version mismatch (see section 10). |
| CUDA out of memory | Close other GPU programs, use shorter text, or run on CPU. No low-memory mode is implemented. |
| Output is Arabic but not Egyptian | Use consistent Egyptian spelling and a native Egyptian reference clip; stronger specialisation needs fine-tuning. |
| Words mispronounced | Write numbers as words, spell dialect words consistently, check tokenizer handling before touching the model. |
| Warnings about Cangjie / Chinese | Harmless for Arabic (see 6.3). |

---

## 12. Credits and licence

- **Chatterbox / Chatterbox Multilingual** by [ResembleAI](https://github.com/resemble-ai/chatterbox): model architecture and inference code under `src/chatterbox/`.
- **NileTTS dataset** (`KickItLikeShika/NileTTS-dataset`) as the source of the Egyptian Arabic recordings and transcripts used by the preparation notebook.
- **faster-whisper** (Whisper large-v3 / small) for transcript verification.
- Project front matter declares `license: mit`. Check the licences of the upstream code, model weights and dataset as well before reuse.
