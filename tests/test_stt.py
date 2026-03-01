import numpy as np
from unittest.mock import patch

from live_chat.stt.whisper import WhisperSTT
from live_chat.config import Config


def test_whisper_transcribe():
    with patch("live_chat.stt.whisper.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {
            "text": " Hello world",
            "segments": [{"no_speech_prob": 0.01, "avg_logprob": -0.3, "compression_ratio": 1.2}],
        }

        stt = WhisperSTT(Config())
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        result = stt.transcribe(audio)

        assert result == "Hello world"
        mock_whisper.transcribe.assert_called_once()


def test_whisper_transcribe_empty_audio():
    with patch("live_chat.stt.whisper.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": "", "segments": []}

        stt = WhisperSTT(Config())
        audio = np.zeros(800, dtype=np.float32)  # very short
        result = stt.transcribe(audio)

        assert result == ""
