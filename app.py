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
        MODEL = ChatterboxMultilingualTTS.from_checkpoint(str(ckpt_dir) + "/", DEVICE)
        if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
            MODEL.to(DEVICE)
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
# Custom CSS – Creative UI
# =============================
CUSTOM_CSS = """
:root{
  --green1:#0b7a3b;
  --green2:#16a34a;
  --green3:#22c55e;
  --ink:#0f172a;
  --muted:#475569;
  --card:#ffffff;
  --glass: rgba(255,255,255,0.75);
}

.gradio-container {
  direction: rtl;
  font-family: "Noto Naskh Arabic", "Segoe UI", Tahoma, Arial;
  background:
    radial-gradient(1200px 400px at 90% -10%, rgba(34,197,94,.15), transparent),
    radial-gradient(800px 300px at 10% 10%, rgba(22,163,74,.12), transparent);
}

/* ===== HERO ===== */
.namaa-hero {
  background: linear-gradient(135deg, var(--green1), var(--green2));
  color: white;
  padding: 28px 24px;
  border-radius: 22px;
  margin-bottom: 18px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0,0,0,0.2);
}

.namaa-hero::after{
  content:"";
  position:absolute;
  inset:0;
  background:
    radial-gradient(300px 300px at 90% 10%, rgba(255,255,255,.25), transparent);
}

.namaa-hero h1{
  font-size: 32px;
  font-weight: 900;
  margin: 0 0 6px 0;
}

.namaa-hero p{
  margin: 0;
  font-size: 15px;
  opacity: .95;
}

/* Badges */
.badges{
  margin-top: 10px;
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
}
.badge{
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.35);
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
}

/* ===== Cards ===== */
.namaa-card{
  background: var(--glass);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 16px;
  border: 1px solid rgba(0,0,0,.08);
  box-shadow: 0 12px 30px rgba(2,6,23,.12);
}

/* Section titles */
.section-title{
  font-size: 15px;
  font-weight: 800;
  color: var(--ink);
  margin-bottom: 6px;
}

/* Notes */
.namaa-note{
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.35);
  border-radius: 14px;
  padding: 10px;
  font-size: 13px;
  color: var(--ink);
}

/* Buttons */
button.primary{
  background: linear-gradient(135deg, var(--green2), var(--green3)) !important;
  border: none !important;
  box-shadow: 0 12px 25px rgba(22,163,74,.35) !important;
}
button.secondary{
  border-radius: 14px !important;
}
"""

# =============================
# UI
# =============================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    # ===== HERO =====
   gr.HTML(
        """
    <div class="namaa-hero" style="text-align:center;">
      <h1 style="margin:0;">NAMAA-Space / NAMAA-Saudi-TTS</h1>
      <p>Saudi Dialect Text-to-Speech</p>
    
      <div class="badges" style="justify-content:center;">
        <span class="badge">AR-SA</span>
        <span class="badge">Saudi Arabic</span>
        <span class="badge">TTS</span>
        <span class="badge">NAMAA Community</span>
      </div>
    </div>
    """
   )
   with gr.Row():
        # ===== LEFT: INPUT =====
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["namaa-card"]):

                gr.HTML('<div class="section-title">📝 النص الصوتي</div>')

                text = gr.Textbox(
                    value=default_text_for_ui("ar"),
                    placeholder="اكتب جملة سعودية… مثال: آبي أروح البقالة وأرجع",
                    max_lines=5,
                )

                with gr.Row():
                    random_btn = gr.Button("🎲 مثال سعودي", variant="secondary")
                    clear_btn = gr.Button("🧹 مسح", variant="secondary")

                gr.HTML('<div class="section-title">🎙️ الصوت المرجعي</div>')

                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    value=default_audio_for_ui("ar"),
                )

                gr.HTML(
                    """
<div class="namaa-note">
يفضّل استخدام صوت سعودي مرجعي لتفادي انتقال لهجة مختلفة.
</div>
"""
                )

                exaggeration = gr.Slider(
                    0.25, 2.0, value=0.5, label="Exaggeration"
                )
                cfg_weight = gr.Slider(
                    0.0, 1.0, value=0.5, label="CFG / Pace"
                )

                with gr.Accordion("⚙️ إعدادات متقدمة", open=False):
                    seed_num = gr.Number(value=0, label="Seed")
                    temp = gr.Slider(0.05, 5.0, value=0.8, label="Temperature")

                run_btn = gr.Button("🔊 توليد الصوت", variant="primary")

        # ===== RIGHT: OUTPUT =====
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["namaa-card"]):
                gr.HTML('<div class="section-title">🎧 الناتج</div>')
                audio_output = gr.Audio()

    random_btn.click(fn=pick_random_saudi_example, inputs=[], outputs=[text])
    clear_btn.click(fn=lambda: "", inputs=[], outputs=[text])

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True)
