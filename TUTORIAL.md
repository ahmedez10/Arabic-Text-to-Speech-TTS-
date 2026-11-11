# IspeakMasri Egyptian Arabic TTS: local guide

This guide explains how to run this repository, how audio is generated, and what is needed to make the voice more consistently Egyptian. It describes the code that is here; it does not assume the repository contains a training pipeline.

## What this project is

This is a Gradio demo built on Chatterbox's multilingual text-to-speech model. It loads pretrained model components from Hugging Face and then replaces the multilingual T3 text-to-speech model weights with the Egyptian checkpoint `t3_mtl23ls_v2.safetensors`. The app passes Arabic as the language (`language_id="ar"`) and uses an optional reference recording to condition the speaker and delivery.

There are no dataset manifests, audio preprocessing scripts, trainer, optimizer setup, checkpoint-saving loop, or evaluation scripts in this checkout. The `T3.loss()` method in `src/chatterbox/models/t3/t3.py` implements loss calculation, but that alone is not a training recipe. So you can run inference locally now; further training requires preparing data and adding/adopting a compatible training pipeline.

## 1. Prepare a local environment (Windows PowerShell)

Use Python 3.10 or 3.11 in a virtual environment. PyTorch installation varies by operating system and CUDA version, so install the appropriate PyTorch build from the official PyTorch install selector first. CPU inference is possible but can be very slow; an NVIDIA GPU with a compatible CUDA PyTorch build is recommended.

```powershell
cd "C:\work\projects\egyptian tts\MASRAWI tts"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the command generated at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Then install the project requirements and the Hugging Face packages the app imports:

```powershell
pip install -r requirements.txt
pip install huggingface_hub safetensors
```

The app also imports `spaces`, which is intended for Hugging Face Spaces GPU scheduling. If `import spaces` fails locally, remove the `import spaces` line and the `@spaces.GPU` decorator from `app.py`; they are deployment wrappers and are not needed for local generation. If your installed Gradio rejects `mcp_server=True`, change the final line to `demo.launch()`.

Check that PyTorch can see your GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. Start the app

The app loads everything from the local `checkpoint` folder, so it runs offline with no network access. That folder holds `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`, and `conds.pt` (about 3 GB).

```powershell
$env:HF_TOKEN = "hf_..."   # only if your access requires it
python app.py
```

Open the local address printed by Gradio (commonly `http://127.0.0.1:7860`). On the page, enter Arabic text, optionally upload a clean reference clip, adjust generation controls, and click generate. The app truncates input to 300 characters. `female_voice.wav` is the default voice reference; use a clear Egyptian Arabic reference when you want its accent and voice traits to guide the output. A reference clip affects voice conditioning; it does not retrain the model.

To switch between CPU and CUDA, `app.py` chooses automatically with `torch.cuda.is_available()`. If model loading fails, first check the full terminal traceback, Python/PyTorch compatibility, available RAM/VRAM, and whether the model files downloaded completely.

## 3. How the code works

The main path is:

1. **`app.py`** creates the Gradio interface, picks `cuda` or `cpu`, loads the checkpoint from `checkpoint/`, and keeps it in a global model once.
2. **`ChatterboxMultilingualTTS.from_local()`** in `src/chatterbox/mtl_tts.py` loads the voice encoder, multilingual T3, S3Gen audio generator, tokenizer, and conditionals from `checkpoint/`.
3. **`generate_tts_audio()`** validates the text, picks the uploaded or default reference audio, and calls `generate(text, language_id="ar", ...)`.
4. **`prepare_conditionals()`** converts the reference recording into speaker embeddings and reference speech/audio features. These features condition the current generation.
5. **`MTLTokenizer`** lowercases and normalizes text, adds the `[ar]` language marker, and turns characters into token IDs.
6. **T3** (the transformer in `src/chatterbox/models/t3/t3.py`) predicts discrete speech tokens from text and conditioning. **S3Gen** converts those tokens into waveform audio. The app returns the waveform at `MODEL.sr`.

Important source locations:

- UI, checkpoint loading, examples, and generation parameters: [app.py](app.py)
- Multilingual loading, Arabic selection, reference conditioning, and generation: [src/chatterbox/mtl_tts.py](src/chatterbox/mtl_tts.py)
- Text normalization and language tokenization: [src/chatterbox/models/tokenizers/tokenizer.py](src/chatterbox/models/tokenizers/tokenizer.py)
- T3 transformer and its text/speech losses: [src/chatterbox/models/t3/t3.py](src/chatterbox/models/t3/t3.py)
- T3 model dimensions and multilingual vocabulary setup: [src/chatterbox/models/t3/modules/t3_config.py](src/chatterbox/models/t3/modules/t3_config.py)
- Waveform generation network: [src/chatterbox/models/s3gen/s3gen.py](src/chatterbox/models/s3gen/s3gen.py)

The language code `ar` selects Arabic handling in a multilingual model; it does not distinguish Egyptian Arabic from Modern Standard Arabic or other Arabic dialects. The checkpoint and reference audio can influence dialect style, but the UI's Egyptian examples do not teach the model anything: they are just sample prompts.

## 4. Improve Egyptian dialect output without training

Start with the existing conditioning and input quality:

- Use natural Egyptian wording and consistent spelling conventions in Arabic script (for example, decide consistently how to write `دلوقتي`, `إزاي`, and colloquial contractions).
- Use a clean, dry, single-speaker Egyptian reference recording with no music, noise, or long silences. Keep the speaker consistent while comparing runs.
- Try several short representative sentences, then vary one generation control at a time. `temperature` changes sampling variation; `cfg_weight` changes text-conditioning strength; `exaggeration` changes expressive delivery. These are generation controls, not dialect switches.
- Build a fixed evaluation list of Egyptian phrases, including colloquial pronouns, common verbs, negation, questions, numbers, and loanwords. Listen for dialect, pronunciation, and intelligibility rather than judging from a few hand-picked examples.
- If the sound is right but some words are pronounced poorly, inspect tokenizer coverage and text normalization before changing the model. Avoid changing the deployed tokenizer vocabulary casually: the model's learned embedding rows correspond to that vocabulary.

## 5. What further training involves

To specialize the model further, assemble paired Egyptian Arabic recordings and exact transcripts. Have native Egyptian speakers review both the transcript and the audio. Use varied speakers and recording conditions if you want a general Egyptian voice; use one consistent speaker only if your goal is a single voice. Remove clipped, noisy, overlapping, or mismatched examples, and keep separate held-out validation and test speakers/sentences. Obtain consent and rights to use every recording.

A compatible training pipeline must at minimum:

1. Normalize transcripts consistently while preserving Egyptian vocabulary and pronunciation distinctions that matter.
2. Resample and segment recordings with their matching text, then tokenize text with the exact multilingual tokenizer used by the checkpoint.
3. Convert speech to the model's discrete speech tokens using the matching S3 tokenizer; construct speaker/reference conditioning and padded batches.
4. Fine-tune T3 first using its text and speech cross-entropy losses, with validation, checkpointing, and conservative learning rates. The loss implementation is in `T3.loss()` but batching, labels, condition preparation, optimizer, and training loop are not supplied here.
5. Evaluate on held-out material and listen to generated samples before considering S3Gen fine-tuning. S3Gen is a separate waveform generation model; changing it is more involved and unnecessary if the issue is mainly dialect wording or pronunciation.
6. Save a checkpoint in the expected format and load it in `app.py` in place of the checkpoint's T3 state. Keep the original checkpoint so you can revert.

For a first experiment, do not attempt to train the entire model from scratch. Start with a small, carefully reviewed dataset and a tested T3 fine-tuning recipe compatible with this Chatterbox revision. Validate that the recipe can overfit a tiny sample before spending time on a larger run. This repository alone does not provide that recipe, so the next engineering step is to choose/create the trainer and pin its exact model/data formats to these source files.

## 6. Safe ways to customize the demo

- Change `EGYPTIAN_EXAMPLES` and the default text in `app.py` to improve the examples shown in the UI.
- Change `LANGUAGE_CONFIG["ar"]["audio"]` to point to your own permitted reference recording.
- Change the text limit and sliders only after confirming the model handles the new range reliably.
- Keep trained checkpoints and datasets outside Git unless you intentionally want to version large artifacts. The `.gitignore` already ignores WAV files and `checkpoints/`.

## Troubleshooting

- **`spaces` import error:** remove its import and `@spaces.GPU` wrapper for local execution.
- **Hugging Face/network errors:** check connectivity and `HF_TOKEN`; first run requires model downloads.
- **CUDA out-of-memory:** close other GPU workloads and try CPU (slower), or use a machine with more VRAM. This code does not currently expose a memory-optimized loading mode.
- **Arabic generated but not Egyptian:** `language_id="ar"` is broad Arabic conditioning. Try consistent Egyptian transcripts and a native Egyptian reference; stronger specialization needs more suitable fine-tuning data and a training pipeline.
- **Import/version errors:** use a clean virtual environment and install PyTorch for your system before `requirements.txt`; this repository does not pin PyTorch or Python versions.




Absolutely — this project is a small but fairly advanced text-to-speech demo, and the main “brain” is in src/chatterbox/mtl_tts.py.

What this repo is

This is a Gradio-based Arabic TTS demo, tuned for Egyptian Arabic.
It is built on top of the ResembleAI Chatterbox model family.
The actual app is app.py, which launches a web UI and calls the model.
The project is mostly inference, not training: the docs in README.md and TUTORIAL.md explain that there is no full training pipeline here.
Folder map

app.py

Main UI and runtime logic
Downloads model files
Loads model once
Generates audio from text
src/chatterbox/

Core TTS code
Contains multilingual model wrapper and model components
src/chatterbox/mtl_tts.py

Main multilingual TTS class
This is the file you tagged
src/chatterbox/models

Lower-level ML pieces:
src/chatterbox/models/t3
src/chatterbox/models/s3gen
src/chatterbox/models/tokenizers
src/chatterbox/models/voice_encoder
src/chatterbox/models/s3tokenizer
requirements.txt

Python dependencies
female_voice.wav

Default speaker/reference audio used by the demo
.venv

Local virtualenv, not project logic itself
test cuda.py

Quick CUDA/device check script
How the app works at a high level

The flow in this project is:

app.py starts a Gradio page.
It decides whether to use CUDA or CPU.
It downloads model checkpoints from Hugging Face.
It creates a global model object and loads the Arabic/Egyptian checkpoint.
The user enters text and optionally a reference audio file.
The model turns text into speech tokens, then turns those into a waveform audio signal.
So the app is basically:

text -> tokenization
tokenization -> speech token generation
speech tokens -> waveform synthesis
What is inside src/chatterbox/mtl_tts.py

This file defines the class ChatterboxMultilingualTTS. It is the main wrapper around the underlying neural nets.

Important pieces:

punc_norm

Cleans text before synthesis.
Fixes weird punctuation and ensures sentence endings like “.” or “!” are present.
SUPPORTED_LANGUAGES

A dictionary of language codes and names.
Example: "ar" = Arabic.
Conditionals

A dataclass that stores conditioning data for the T3 and S3Gen models.
It holds:
the reference speaker embedding
prompt speech tokens
style/emotion conditioning
generated reference audio features
ChatterboxMultilingualTTS

The main inference class.
It owns:
the T3 text-to-speech model
the S3Gen waveform generator
the voice encoder
the tokenizer
the device (CPU/GPU)
the stored conditionals
The important class methods

from_local(...)

Loads a local checkpoint directory.
Reads:
ve.pt
t3_mtl23ls_v2.safetensors
s3gen.pt
grapheme_mtl_merged_expanded_v1.json
This is how the project loads a checkpoint from disk.
from_pretrained(...)

Downloads the base Chatterbox model files from the HF repo.
This is used to get the standard model components.
from_checkpoint(...)

Downloads a specific checkpoint version from the remote model repo.
Then loads the custom Arabic/Egyptian weights and conditionals.
prepare_conditionals(...)

This is one of the most important methods.
It takes a reference waveform (like a speaking sample), extracts:
speaker identity
short reference audio features
prompt tokens
Then it creates a “conditioning state” used to guide the generation.
That means the model is not just reading text — it is also mimicking the voice/style of the reference audio.
generate(...)

This is the core synthesis method.
Flow:
validate the language ID
if a reference audio path is supplied, prepare conditionals
normalize punctuation
tokenize text with the multilingual tokenizer
pad tokens for generation
run T3 inference to produce speech tokens
clean invalid tokens
feed speech tokens to S3Gen
convert output to a NumPy waveform
return a torch tensor containing the generated audio
In simple terms:

T3 decides what the speech should be
S3Gen turns that into real audio waveform
VoiceEncoder + reference audio influence the voice characteristics
Why the app uses Arabic and a reference voice

The project passes language_id="ar" in app.py, which tells the multilingual model to generate Arabic text. But the voice is strongly shaped by the reference audio prompt, not just by the language ID. That is why the app lets you upload a clean voice sample.

This is very important: the model is not a “pure Egyptian-only” model. It is a multilingual model that is being conditioned for Arabic/Egyptian-like style with the checkpoint and reference audio.

The core production path

The actual work path is roughly:

app.py calls generate_tts_audio()
generate_tts_audio() reads text and default/uploaded voice sample
calls MODEL.generate(...)
inside src/chatterbox/mtl_tts.py:
punc_norm cleans the text
tokenizer converts text to IDs
T3 model predicts speech tokens
S3Gen reconstructs waveform
tensor is returned to Gradio for playback
Why this repo feels unusual

This repo is not a typical “full ML project structure” with training scripts and notebooks. It is a deployment/inference project:

no dataset loading
no training loops
no validation pipeline in the repo root
just model runtime and UI
That’s also why the docs repeatedly say: this is a demo built around a checkpoint, not a training repo.

If you want, I can do one of these next:

a line-by-line explanation of src/chatterbox/mtl_tts.py
a “how the model pipeline works” diagram in plain English
a “what each file in src/chatterbox/models does” breakdown
or a very beginner-friendly explanation of the PyTorch flow from input text to output audio