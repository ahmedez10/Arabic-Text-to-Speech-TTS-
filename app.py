import random
import numpy as np
import torch
from pathlib import Path
import os

import gradio as gr
import spaces
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

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
        "text": "آبي أروح البقالة أشتري كم غرض وأرجع بسرعة.",
    }
}

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
# Helpers
# =============================
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Loading model using from_pretrained...")
        # Load base model
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        
        # Load and replace t3 state from safetensors
        t3_path = os.path.join(ckpt_dir, "t3_mtl23ls_v2.safetensors")
        print(f"Loading t3 state from: {t3_path}")
        t3_state = load_safetensors(t3_path, device=DEVICE)
        MODEL.t3.load_state_dict(t3_state)
        MODEL.t3.to(DEVICE).eval()
        
        # Ensure entire model is on correct device
        if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
            MODEL.to(DEVICE)
        
        print("Model loaded successfully!")
    return MODEL

try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# =============================
# TTS (UNCHANGED)
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

    current_model = get_or_load_model()

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    text_input = (text_input or "").strip()
    if not text_input:
        raise gr.Error("الرجاء إدخال نص.")

    chosen_prompt = audio_prompt_path_input or default_audio_for_ui("ar")

    generate_kwargs = {
        "exaggeration": float(exaggeration_input),
        "temperature": float(temperature_input),
        "cfg_weight": float(cfgw_input),
    }

    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt

    wav = current_model.generate(
        text_input[:300],
        language_id="ar",
        **generate_kwargs,
    )

    return (current_model.sr, wav.squeeze(0).numpy())

def pick_random_saudi_example():
    return random.choice(SAUDI_EXAMPLES)

# =============================
# Custom CSS (HTML Design)
# =============================
CUSTOM_CSS = """
.gradio-container {
  direction: rtl;
  font-family: 'Noto Naskh Arabic', 'Segoe UI', Tahoma, Arial;
}
.namaa-hero {
  background: linear-gradient(135deg, #0b7a3b, #16a34a);
  color: white;
  padding: 22px;
  border-radius: 18px;
  text-align: center;
  margin-bottom: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}
.namaa-hero h1 {
  font-size: 30px;
  margin-bottom: 6px;
  font-weight: 800;
}
.namaa-hero p {
  font-size: 15px;
  opacity: 0.95;
}
.namaa-card {
  background: white;
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.06);
}
.namaa-note {
  background: rgba(22,163,74,0.1);
  border: 1px solid rgba(22,163,74,0.3);
  border-radius: 12px;
  padding: 10px;
  font-size: 13px;
}
"""

# =============================
# UI
# =============================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
<div class="namaa-hero">
  <h1>🇸🇦 NAMAA Saudi Dialect TTS</h1>
  <p>تحويل النص باللهجة السعودية إلى صوت طبيعي<br/>
  مدعوم من مجتمع <b>نماء</b></p>
</div>
"""
    )

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["namaa-card"]):
                text = gr.Textbox(
                    value=default_text_for_ui("ar"),
                    label="النص (لهجة سعودية)",
                    max_lines=5,
                )

                with gr.Row():
                    random_btn = gr.Button("🎲 مثال سعودي")
                    clear_btn = gr.Button("🧹 مسح")

                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="صوت مرجعي (اختياري)",
                    value=default_audio_for_ui("ar"),
                )

                gr.HTML(
                    """
<div class="namaa-note">
<b>ملاحظة:</b> يفضّل أن يكون الصوت المرجعي سعودي لتجنّب انتقال لهجة مختلفة.
</div>
"""
                )

                exaggeration = gr.Slider(0.25, 2.0, value=0.5, label="Exaggeration")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, label="CFG / Pace")

                with gr.Accordion("⚙️ إعدادات متقدمة", open=False):
                    seed_num = gr.Number(value=0, label="Seed")
                    temp = gr.Slider(0.05, 5.0, value=0.8, label="Temperature")

                run_btn = gr.Button("🔊 توليد الصوت", variant="primary")

        with gr.Column():
            with gr.Group(elem_classes=["namaa-card"]):
                audio_output = gr.Audio(label="الصوت الناتج")

    random_btn.click(fn=pick_random_saudi_example, inputs=[], outputs=[text])
    clear_btn.click(fn=lambda: "", inputs=[], outputs=[text])

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True)