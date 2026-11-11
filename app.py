import random
import sys
import io
import numpy as np
import torch
from pathlib import Path
import os

# Force UTF-8 output so emoji in print() don't crash Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Everything loads from ./checkpoint — keep the hub library off the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
try:
    import spaces
    _gpu_decorator = spaces.GPU
except ImportError:
    def _gpu_decorator(fn):
        return fn
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# =============================
# Runtime / Device
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# =============================
# Global Model
# =============================
MODEL = None

# =============================
# Egyptian Arabic UI Config 🇪🇬
# =============================
# Use absolute path so it works regardless of CWD (e.g. when launched from VS Code)
_APP_DIR = Path(__file__).parent

LANGUAGE_CONFIG = {
    "ar": {
        "audio": str(_APP_DIR / "female_voice.wav"),
        "text": "انا سبت الشغل و راجع دلوقتي علي طول.",
    }
}

EGYPTIAN_EXAMPLES = [
    "امبارح كنت بتفرج على فيلم، وفجأة الكهربا قطعت.",
    "أول ما وصلت البيت، غيرت هدومي وقعدت أرتاح شوية.",
    "مش كل مرة الحاجة تمشي زي ما إحنا مخططين.",
    "الجو حر قوي.",
    "اللي فات كان حاجة ودلوقتي حاجة تانية خالص.",
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
        print("Loading model from checkpoint/ ...")
        MODEL = ChatterboxMultilingualTTS.from_local(_APP_DIR / "checkpoint", DEVICE)
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
@_gpu_decorator
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:
    try:
        current_model = get_or_load_model()

        if seed_num_input and int(seed_num_input) != 0:
            set_seed(int(seed_num_input))

        text_input = (text_input or "").strip()
        if not text_input:
            raise gr.Error("الرجاء إدخال نص.")

        # Validate the uploaded audio path — fall back to default if invalid/missing
        chosen_prompt = None
        if audio_prompt_path_input and Path(audio_prompt_path_input).exists():
            chosen_prompt = audio_prompt_path_input
        else:
            default_path = default_audio_for_ui("ar")
            if default_path and Path(default_path).exists():
                chosen_prompt = default_path

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
    except gr.Error:
        raise
    except Exception as e:
        print(f"ERROR during generation: {type(e).__name__}: {e}")
        raise gr.Error(f"خطأ أثناء توليد الصوت: {type(e).__name__}: {e}")

    return (current_model.sr, wav.squeeze(0).numpy())

def pick_random_egyptian_example():
    return random.choice(EGYPTIAN_EXAMPLES)

# =============================
# Custom CSS (HTML Design)
# =============================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=Inter:wght@400;600;700&display=swap');

.gradio-container {
  direction: rtl;
  font-family: 'Cairo', 'Segoe UI', Tahoma, Arial;
  background: #0d0f14 !important;
  min-height: 100vh;
}
.ispeakmasri-hero {
  background: linear-gradient(135deg, #c8181e 0%, #e63946 40%, #a01015 100%);
  color: white;
  padding: 32px 24px;
  border-radius: 20px;
  text-align: center;
  margin-bottom: 20px;
  box-shadow: 0 20px 60px rgba(198,24,30,0.35), 0 0 0 1px rgba(255,255,255,0.07);
  position: relative;
  overflow: hidden;
}
.ispeakmasri-hero::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle at 70% 30%, rgba(255,255,255,0.08) 0%, transparent 60%);
  pointer-events: none;
}
.ispeakmasri-hero .badge {
  display: inline-block;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 12px;
  backdrop-filter: blur(4px);
}
.ispeakmasri-hero h1 {
  font-family: 'Inter', sans-serif;
  font-size: 48px;
  font-weight: 900;
  margin: 0 0 10px 0;
  letter-spacing: -1px;
  text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.ispeakmasri-hero .subtitle {
  font-size: 17px;
  opacity: 0.9;
  margin: 0;
  font-weight: 400;
  line-height: 1.7;
}
.ispeakmasri-hero .tagline {
  font-size: 12px;
  opacity: 0.55;
  margin-top: 8px;
  font-style: italic;
  letter-spacing: 0.5px;
}
.ispeakmasri-card {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 18px !important;
  padding: 20px !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.ispeakmasri-note {
  background: rgba(198,24,30,0.1);
  border: 1px solid rgba(198,24,30,0.25);
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 13px;
  color: #fca5a5;
  margin-top: 6px;
}
.ispeakmasri-footer {
  text-align: center;
  color: rgba(255,255,255,0.25);
  font-size: 12px;
  margin-top: 16px;
  padding: 12px;
  letter-spacing: 0.5px;
}
"""

ISPEAKMASRI_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Cairo"),
).set(
    body_background_fill="#0d0f14",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(255,255,255,0.04)",
    block_border_color="rgba(255,255,255,0.08)",
    block_label_text_color="#94a3b8",
    input_background_fill="#1e2330",
    input_border_color="rgba(255,255,255,0.1)",
    button_secondary_background_fill="#1e2330",
    button_secondary_border_color="rgba(255,255,255,0.1)",
    button_secondary_text_color="#e2e8f0",
    button_primary_background_fill="linear-gradient(135deg, #c8181e, #e63946)",
    button_primary_background_fill_hover="linear-gradient(135deg, #e63946, #c8181e)",
    button_primary_text_color="white",
)

# =============================
# UI
# =============================
with gr.Blocks() as demo:
    gr.HTML(
        """
<div class="ispeakmasri-hero">
  <div class="badge">\U0001f1ea\U0001f1ec Egyptian Arabic TTS</div>
  <h1>IspeakMasri</h1>
  <p class="subtitle">\u062d\u0648\u0651\u0644 \u0646\u0635\u0648\u0635\u0643 \u0628\u0627\u0644\u0639\u0627\u0645\u064a\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629 \u0625\u0644\u0649 \u0635\u0648\u062a \u0637\u0628\u064a\u0639\u064a</p>
  <p class="tagline">Powered by Chatterbox \u00b7 Egyptian Checkpoint \u00b7 ResembleAI</p>
</div>
"""
    )

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["ispeakmasri-card"]):
                text = gr.Textbox(
                    value=default_text_for_ui("ar"),
                    label="\u0627\u0644\u0646\u0635 (\u0644\u0647\u062c\u0629 \u0645\u0635\u0631\u064a\u0629)",
                    placeholder="\u0627\u0643\u062a\u0628 \u0647\u0646\u0627 \u0628\u0627\u0644\u0639\u0627\u0645\u064a\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629...",
                    max_lines=5,
                )

                with gr.Row():
                    random_btn = gr.Button("\U0001f3b2 \u0645\u062b\u0627\u0644 \u0645\u0635\u0631\u064a")
                    clear_btn = gr.Button("\U0001f9f9 \u0645\u0633\u062d")

                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="\u0635\u0648\u062a \u0645\u0631\u062c\u0639\u064a (\u0627\u062e\u062a\u064a\u0627\u0631\u064a)",
                    value=default_audio_for_ui("ar"),
                )

                gr.HTML(
                    """
<div class="ispeakmasri-note">
\U0001f4a1 <b>\u062a\u0644\u0645\u064a\u062d:</b> \u0627\u0633\u062a\u062e\u062f\u0645 \u0635\u0648\u062a\u0627\u064b \u0645\u0631\u062c\u0639\u064a\u0627\u064b \u0628\u0627\u0644\u0639\u0627\u0645\u064a\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629 \u0644\u0644\u062d\u0635\u0648\u0644 \u0639\u0644\u0649 \u0623\u0641\u0636\u0644 \u0646\u062a\u064a\u062c\u0629.
</div>
"""
                )

                exaggeration = gr.Slider(0.25, 2.0, value=0.5, label="Exaggeration")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, label="CFG / Pace")

                with gr.Accordion("\u2699\ufe0f \u0625\u0639\u062f\u0627\u062f\u0627\u062a \u0645\u062a\u0642\u062f\u0645\u0629", open=False):
                    seed_num = gr.Number(value=0, label="Seed")
                    temp = gr.Slider(0.05, 5.0, value=0.8, label="Temperature")

                run_btn = gr.Button("\U0001f50a \u0648\u0644\u0651\u062f \u0627\u0644\u0635\u0648\u062a", variant="primary")

        with gr.Column():
            with gr.Group(elem_classes=["ispeakmasri-card"]):
                audio_output = gr.Audio(label="\u0627\u0644\u0635\u0648\u062a \u0627\u0644\u0646\u0627\u062a\u062c")

    gr.HTML('<div class="ispeakmasri-footer">IspeakMasri &middot; Egyptian Arabic TTS &middot; Built on Chatterbox + Egyptian checkpoint</div>')

    random_btn.click(fn=pick_random_egyptian_example, inputs=[], outputs=[text])
    clear_btn.click(fn=lambda: "", inputs=[], outputs=[text])

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        outputs=[audio_output],
    )

try:
    demo.launch(css=CUSTOM_CSS, theme=ISPEAKMASRI_THEME, mcp_server=True, allowed_paths=[str(_APP_DIR)])
except TypeError:
    demo.launch(css=CUSTOM_CSS, theme=ISPEAKMASRI_THEME, allowed_paths=[str(_APP_DIR)])