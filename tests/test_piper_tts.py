import pytest

from live_chat.config import Config
from live_chat.tts.piper_tts import PiperTTS


def test_piper_tts_uses_config_voice():
    config = Config(tts_voice="en_US-amy-medium")
    tts = PiperTTS(config)
    assert tts._model_name == "en_US-amy-medium"


def test_piper_tts_default_voice():
    config = Config()
    tts = PiperTTS(config)
    assert tts._model_name == "en_US-lessac-medium"


def test_piper_tts_missing_model_error():
    config = Config(tts_voice="nonexistent-voice")
    tts = PiperTTS(config)
    with pytest.raises(FileNotFoundError, match="nonexistent-voice"):
        tts._ensure_model()
