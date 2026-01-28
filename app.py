import random
import numpy as np
import torch
from pathlib import Path
import os

import gradio as gr
import spaces
from huggingface_hub import snapshot_download

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# =============================
# Runtime / Device
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

REPO_ID = "NAMAA-Space/NAMAA-Saudi-TTS"

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

# =============================
# Global Model
# =============================
MODEL = None

# =============================
# Saudi Arabic UI Config 🇸🇦
# =============================
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "./refrence.flac",
        # Saudi dialect default
        "text": "آبي أروح البقالة أشتري كم غرض وأرجع بسرعة.",
    }
}

# Saudi Dialect Examples (General / Najdi-neutral)
SAUDI_EXAMPLES = [
    "آبي أروح البقالة أشتري كم غرض وأرجع بسرعة.",
    "آبي أطلع مشوار خفيف وبأرجع قبل المغرب.",
    "ترى الموضوع بسيط، خلّصه اليوم وارتاح.",
    "أنا بالطريق الحين، باقي عشر دقايق وأوصل.",
    "خلّنا نبدأ اليوم، والباقي يجي مع الوقت.",
    "وش رايك نخلص الشغل اليوم ونرتاح بكرة؟",
    "إذا احتجت شي، كلّمني وأنا أجيك.",
    "الوضع تمام، الأمور ماشية مثل ما نبي.",
]

# =============================
# UI Helpers
# =============================
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxMultilingualTTS.from_checkpoint(str(ckpt_dir) + "/", DEVICE)
        if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    return MODEL


try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Error: {e}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================
# TTS Generation (UNCHANGED)
# =============================
@spaces.GPU
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:

    language_id = "ar"
    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    text_input = (text_input or "").strip()
    if not text_input:
        raise gr.Error("الرجاء إدخال نص للتحويل إلى صوت.")

    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": float(exaggeration_input),
        "temperature": float(temperature_input),
        "cfg_weight": float(cfgw_input),
    }

    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt

    wav = current_model.generate(
        text_input[:300],
        language_id=language_id,
        **generate_kwargs,
    )

    return (current_model.sr, wav.squeeze(0).numpy())


def pick_random_saudi_example():
    return random.choice(SAUDI_EXAMPLES)


# =============================
# Gradio UI 🇸🇦✨
# =============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
<div style="text-align:center">

# 🇸🇦 NAMAA Saudi Dialect TTS
### تحويل النص السعودي إلى صوت طبيعي

**نموذج صوتي باللهجة السعودية**
<br/>
مدعوم من مجتمع **نماء** – Arabic AI Community

</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            initial_lang = "ar"

            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                placeholder="اكتب جملة باللهجة السعودية… مثال: آبي أروح البقالة وأرجع",
                label="النص (لهجة سعودية – حد أقصى 300 حرف)",
                max_lines=5,
            )

            with gr.Row():
                random_btn = gr.Button("🎲 مثال سعودي عشوائي", variant="secondary")
                clear_btn = gr.Button("🧹 مسح النص", variant="secondary")

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="ملف صوت مرجعي (اختياري)",
                value=default_audio_for_ui(initial_lang),
            )

            gr.Markdown(
                """
💡 **ملاحظة**  
يفضّل أن يكون الصوت المرجعي بنفس اللهجة لتفادي انتقال نبرة غير مرغوبة.
في حال عدم الحاجة، اتركه فارغًا.
"""
            )

            exaggeration = gr.Slider(
                0.25, 2.0, step=0.05,
                label="Exaggeration (القيمة الطبيعية ≈ 0.5)",
                value=0.5,
            )

            cfg_weight = gr.Slider(
                0.0, 1.0, step=0.05,
                label="CFG / Pace (0.2 – 1.0 مناسب)",
                value=0.5,
            )

            with gr.Accordion("⚙️ إعدادات متقدمة", open=False):
                seed_num = gr.Number(value=0, label="Seed (0 = عشوائي)")
                temp = gr.Slider(0.05, 5.0, step=0.05, label="Temperature", value=0.8)

            run_btn = gr.Button("🔊 توليد الصوت", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="الصوت الناتج")

    random_btn.click(fn=pick_random_saudi_example, inputs=[], outputs=[text])
    clear_btn.click(fn=lambda: "", inputs=[], outputs=[text])

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True)