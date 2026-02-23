import sys
import types

from unittest.mock import MagicMock, patch

# Pre-mock the kokoro module so pipeline.py can import KokoroTTS
_mock_kokoro = types.ModuleType("kokoro")
_mock_kokoro.KPipeline = MagicMock()
sys.modules.setdefault("kokoro", _mock_kokoro)

from live_chat.pipeline import Pipeline, State  # noqa: E402
from live_chat.config import Config  # noqa: E402


def test_pipeline_initial_state():
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WakeWordDetector"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.KokoroTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation"):
        pipeline = Pipeline(Config())
        assert pipeline.state == State.WAITING_FOR_WAKE_WORD


def test_pipeline_state_enum():
    assert State.WAITING_FOR_WAKE_WORD.value == "waiting"
    assert State.LISTENING.value == "listening"
    assert State.THINKING.value == "thinking"
    assert State.SPEAKING.value == "speaking"
