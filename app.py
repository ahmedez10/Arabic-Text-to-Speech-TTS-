import random
import numpy as np
import torch
from pathlib import Path
import os

import gradio as gr
import spaces
from huggingface_hub import snapshot_download

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Based on your original demo script :contentReference[oaicite:0]{index=0}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

REPO_ID = "oddadmix/chatterbox-egyptian-v0"

ckpt_dir = Path(
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        revision="main",
        allow_patterns=[
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
        token=os.getenv("HF_TOKEN"),
    )
)

# --- Global Model Initialization ---
MODEL = None

# Egyptian Arabic (Masri) defaults for Arabic UI
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        # ✅ Egyptian Arabic (Masri) default text (instead of MSA)
        "text": "الشهر اللي فات وصلنا لإنجاز جديد وعدّينا اتنين مليار مشاهدة على قناتنا على يوتيوب.",
    }
}

# Optional: a few Egyptian examples for quick demo/testing
EGYPTIAN_EXAMPLES = [
    "أنا رايحة الشغل دلوقتي، وهكلمِك أول ما أوصل.",
    "لو سمحتي ابعتيلي الإيميل على الواتساب عشان أتابع.",
    "بصي، الخصم اتناشر ونص في المية بس لحد آخر الأسبوع.",
    "أنا كنت فاكرة إنك جاية بدري، اتأخرتي ليه؟",
    "إزيك؟ عامل إيه النهارده؟",
    "الخصم اتناشر ونص في المية بس لفترة محدودة.",
    "أنا مستنيك من بدري، متتأخرش تاني.",
    "هو الموضوع ده هيتحل إمتى؟ أنا محتاجه النهارده.",
]

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")

    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])

    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


def get_or_load_model():
    """Load model once and ensure it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxMultilingualTTS.from_checkpoint(str(ckpt_dir) + "/", DEVICE)
        if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    return MODEL


# Attempt to load at startup (optional; keep for demo responsiveness)
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")


def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@spaces.GPU
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:
    """
    Generate speech audio from text using Chatterbox Multilingual model.
    - If a reference audio is provided, the model will try to match the speaker/style.
    - If not provided, it uses the model's default voice.

    Note: For Arabic here, the demo text + examples are Egyptian Arabic (Masri).
    """

    language_id = "ar"
    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    text_input = (text_input or "").strip()
    if not text_input:
        raise gr.Error("Please enter text to synthesize.")

    print(f"Generating audio for language='{language_id}', text='{text_input[:60]}...'")

    # Keep same behavior: use uploaded/mic ref if provided, else default audio for language.
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": float(exaggeration_input),
        "temperature": float(temperature_input),
        "cfg_weight": float(cfgw_input),
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")

    wav = current_model.generate(
        text_input[:300],  # max chars
        language_id=language_id,
        **generate_kwargs,
    )

    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())


def pick_random_egyptian_example():
    return random.choice(EGYPTIAN_EXAMPLES)


with gr.Blocks() as demo:
    gr.Markdown(
        """
# Chatterbox Egyptian Arabic (Masri) TTS Demo 🇪🇬
Generate natural Egyptian Arabic speech from text, with optional reference audio styling.
"""
    )

    with gr.Row():
        with gr.Column():
            initial_lang = "ar"

            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                placeholder="اكتب نص بالمصري… مثال: أنا رايح الشغل دلوقتي وهكلمك بعدين",
                label="Text to synthesize (Egyptian Arabic – max chars 300)",
                max_lines=5,
            )

            with gr.Row():
                random_btn = gr.Button("🎲 Random Egyptian Example", variant="secondary")
                clear_btn = gr.Button("🧹 Clear", variant="secondary")


            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=default_audio_for_ui(initial_lang),
            )

            gr.Markdown(
                "💡 **Note**: Make sure the reference clip matches the selected language. "
                "If the reference clip has a different language/accent, outputs may inherit it. "
                "To mitigate accent transfer, try lowering CFG/Pace (or set it to 0 for language transfer).",
                elem_classes=["audio-note"],
            )

            exaggeration = gr.Slider(
                0.25, 2.0, step=0.05,
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                value=0.5,
            )
            cfg_weight = gr.Slider(
                0.0, 1.0, step=0.05,
                label="CFG/Pace (0 can increase language transfer; 0.2–1.0 typical)",
                value=0.5,
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5.0, step=0.05, label="Temperature", value=0.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")


   

        random_btn.click(fn=pick_random_egyptian_example, inputs=[], outputs=[text])
        clear_btn.click(fn=lambda: "", inputs=[], outputs=[text])

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True)
