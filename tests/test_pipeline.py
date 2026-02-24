from unittest.mock import MagicMock, patch

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


def test_pipeline_initial_state():
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WakeWordDetector"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS"), \
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
