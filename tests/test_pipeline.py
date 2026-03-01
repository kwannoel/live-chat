import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


def test_pipeline_initial_state():
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation"):
        pipeline = Pipeline(Config())
        assert pipeline.state == State.IDLE


def test_pipeline_state_enum():
    assert State.IDLE.value == "idle"
    assert State.LISTENING.value == "listening"
    assert State.THINKING.value == "thinking"
    assert State.SPEAKING.value == "speaking"


@pytest.mark.asyncio
async def test_speak_sentence_uses_async_wait():
    """_speak_sentence must use wait_async (non-blocking), not wait (blocking)."""
    config = Config()
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput") as mock_out_cls, \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation"):

        mock_out = mock_out_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050
        mock_out.wait_async = AsyncMock()

        pipeline = Pipeline(config)
        await pipeline._speak_sentence("Hello.")

        mock_out.play.assert_called_once()
        mock_out.wait_async.assert_called_once()
        mock_out.wait.assert_not_called()
