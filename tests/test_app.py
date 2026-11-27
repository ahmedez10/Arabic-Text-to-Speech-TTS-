"""Lightweight tests for IspeakMasri backend helpers and FastAPI routes."""

import io
import os
import sys
import types
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyModel:
    """Fast in-memory stub mimicking ChatterboxMultilingualTTS for tests."""

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.conds = object()
        self.calls = []
        self.prompt_paths = []

    def generate(self, text, language_id="ar", **kwargs):
        import torch

        self.calls.append((text, language_id, dict(kwargs)))
        if "audio_prompt_path" in kwargs:
            self.prompt_paths.append(kwargs["audio_prompt_path"])
        t = np.linspace(0, 0.8, int(self.sr * 0.8), endpoint=False, dtype=np.float32)
        sig = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return torch.from_numpy(sig).unsqueeze(0)


_DUMMY_MODEL = _DummyModel()


class _FakeMTLClass:
    @classmethod
    def from_local(cls, ckpt_dir, device):
        return _DUMMY_MODEL


# Stub chatterbox module before importing app so tests run offline without 3GB weights
_fake_mtl = types.ModuleType("src.chatterbox.mtl_tts")
_fake_mtl.ChatterboxMultilingualTTS = _FakeMTLClass
sys.modules.setdefault("src.chatterbox.mtl_tts", _fake_mtl)

from fastapi.testclient import TestClient  # noqa: E402
import app as masri_app  # noqa: E402


def _make_wav_bytes(freq_hz: float, duration_s: float = 1.0, sr: int = 24000, amp: float = 0.4) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sig = amp * np.sin(2 * np.pi * freq_hz * t) if freq_hz > 0 else np.zeros_like(t)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class TestAudioAndTextHelpers(unittest.TestCase):
    def test_split_into_chunks_respects_sentence_and_comma_limits(self):
        short = "إزيك يا باشا؟ عامل إيه!"
        chunks = masri_app.split_into_chunks(short, max_chars=120)
        self.assertEqual(chunks, ["إزيك يا باشا؟ عامل إيه!"])

        long_text = (
            "النهاردة الجو جميل جداً في القاهرة، والشمس طالعة من بدري، "
            "والناس كلها مبسوطة في الشوارع، وكمان القهوة طعمها مختلف خالص الصبح بدري."
        )
        long_chunks = masri_app.split_into_chunks(long_text, max_chars=80)
        self.assertGreater(len(long_chunks), 1)
        for ch in long_chunks:
            self.assertLessEqual(len(ch), 120)

    def test_highpass_filter_handles_empty_and_preserves_shape(self):
        empty = np.array([], dtype=np.float32)
        self.assertEqual(len(masri_app.highpass_filter(empty, 24000)), 0)

        t = np.linspace(0, 0.5, 12000, endpoint=False, dtype=np.float32)
        dc_plus_tone = 0.5 + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        filtered = masri_app.highpass_filter(dc_plus_tone, 24000, cutoff_hz=80.0)
        self.assertEqual(filtered.shape, dc_plus_tone.shape)
        self.assertLess(abs(float(np.mean(filtered[2000:]))), 0.05)

    def test_detect_voice_gender_male_female_and_silence(self):
        import tempfile

        for freq, expected in [(220.0, "female"), (110.0, "male"), (0.0, "unknown")]:
            wav_bytes = _make_wav_bytes(freq, duration_s=1.0, sr=16000)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            try:
                self.assertEqual(masri_app.detect_voice_gender(tmp_path), expected)
            finally:
                os.unlink(tmp_path)

    def test_preprocess_reference_audio_resamples_and_picks_voiced_segment(self):
        import tempfile

        sr = 48000
        silent = np.zeros(int(sr * 5), dtype=np.float32)
        t = np.linspace(0, 8.0, int(sr * 8), endpoint=False, dtype=np.float32)
        voiced = 0.3 * np.sin(2 * np.pi * 210.0 * t)
        full = np.concatenate([silent, voiced])
        pcm = (full * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
            raw_path = tmp.name

        prep_path = None
        try:
            prep_path = masri_app.preprocess_reference_audio(raw_path, target_sr=24000, target_duration=8.0)
            self.assertNotEqual(prep_path, raw_path)
            out_sr, out_data = masri_app.load_audio_float(prep_path, max_duration=15.0)
            self.assertEqual(out_sr, 24000)
            self.assertGreater(float(np.sqrt(np.mean(out_data ** 2))), 0.05)
        finally:
            os.unlink(raw_path)
            if prep_path and prep_path != raw_path and os.path.exists(prep_path):
                os.unlink(prep_path)


class TestApiRoutes(unittest.TestCase):
    def setUp(self):
        masri_app.MODEL = _DUMMY_MODEL
        masri_app._MODEL_LOAD_ERROR = None
        _DUMMY_MODEL.calls.clear()
        _DUMMY_MODEL.prompt_paths.clear()
        self.client = TestClient(masri_app.app)

    def test_generate_validation_and_success(self):
        bad = self.client.post("/api/generate", data={"text": "   "})
        self.assertEqual(bad.status_code, 400)

        too_long = self.client.post("/api/generate", data={"text": "ا" * 301})
        self.assertEqual(too_long.status_code, 400)

        wav_bytes = _make_wav_bytes(220.0, duration_s=0.6)
        ok = self.client.post(
            "/api/generate",
            data={"text": "إزيك يا باشا، عامل إيه؟"},
            files={"audio": ("voice.wav", wav_bytes, "audio/wav")},
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers["content-type"], "audio/wav")
        self.assertTrue(ok.content.startswith(b"RIFF"))

    def test_generate_best_reuses_speaker_conditionals(self):
        wav_bytes = _make_wav_bytes(210.0, duration_s=0.6)
        res = self.client.post(
            "/api/generate-best",
            data={"text": "صباح الفل يا معلم.", "count": "3"},
            files={"audio": ("voice.wav", wav_bytes, "audio/wav")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["candidates"]), 3)
        self.assertTrue(body["candidates"][0]["recommended"])
        # Speaker embedding path passed only once across all 3 candidates
        self.assertEqual(len(_DUMMY_MODEL.prompt_paths), 1)

    def test_detect_gender_and_upload_limit(self):
        wav_bytes = _make_wav_bytes(110.0, duration_s=1.0, sr=16000)
        res = self.client.post(
            "/api/detect-gender",
            files={"audio": ("male.wav", wav_bytes, "audio/wav")},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["gender"], "male")

        # Temporarily shrink upload cap to verify oversized upload rejection
        orig_cap = masri_app.MAX_UPLOAD_BYTES
        masri_app.MAX_UPLOAD_BYTES = 128
        try:
            res_big = self.client.post(
                "/api/generate",
                data={"text": "أهلاً"},
                files={"audio": ("big.wav", wav_bytes, "audio/wav")},
            )
            self.assertEqual(res_big.status_code, 400)
        finally:
            masri_app.MAX_UPLOAD_BYTES = orig_cap

    def test_returns_503_when_model_unavailable(self):
        masri_app.MODEL = None
        masri_app._MODEL_LOAD_ERROR = "Checkpoint missing"
        res = self.client.post("/api/generate", data={"text": "أهلاً"})
        self.assertEqual(res.status_code, 503)


if __name__ == "__main__":
    unittest.main()
