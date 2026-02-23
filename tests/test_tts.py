import sys
import types

import numpy as np
from unittest.mock import patch, MagicMock

# Pre-mock the kokoro module so the source file can import from it
_mock_kokoro = types.ModuleType("kokoro")
_mock_kokoro.KPipeline = MagicMock()
sys.modules.setdefault("kokoro", _mock_kokoro)

from live_chat.tts.kokoro import KokoroTTS  # noqa: E402
from live_chat.config import Config  # noqa: E402


def test_tts_synthesize():
    with patch("live_chat.tts.kokoro.KPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        fake_audio = np.random.randn(24000).astype(np.float32)
        mock_pipeline.return_value = iter([
            ("Hello.", "h@loU", fake_audio),
        ])
        mock_pipeline_cls.return_value = mock_pipeline

        tts = KokoroTTS(Config())
        chunks = list(tts.synthesize("Hello."))

        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32
        assert len(chunks[0]) == 24000


def test_tts_synthesize_multiple_sentences():
    with patch("live_chat.tts.kokoro.KPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        audio1 = np.random.randn(24000).astype(np.float32)
        audio2 = np.random.randn(12000).astype(np.float32)
        mock_pipeline.return_value = iter([
            ("Hello.", "h@loU", audio1),
            ("World.", "w3:ld", audio2),
        ])
        mock_pipeline_cls.return_value = mock_pipeline

        tts = KokoroTTS(Config())
        chunks = list(tts.synthesize("Hello. World."))

        assert len(chunks) == 2
