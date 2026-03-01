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


@pytest.mark.asyncio
async def test_auto_speak_calls_llm_and_tts():
    """When auto_speak is True, pipeline should generate and speak an opening."""
    config = Config(auto_speak=True, persona="You are Bob.")
    with patch("live_chat.pipeline.AudioInput") as mock_in_cls, \
         patch("live_chat.pipeline.AudioOutput") as mock_out_cls, \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_out = mock_out_cls.return_value
        mock_in = mock_in_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # Router returns a model
        mock_router.route = AsyncMock(return_value="claude-haiku-4-5-20251001")

        # LLM streams one sentence
        async def fake_stream(model, system, messages):
            yield "Hello there!"
        mock_llm.stream = fake_stream

        # Conversation.for_api returns system + messages
        mock_conv.for_api.return_value = ("You are Bob.", [{"role": "user", "content": "Start a conversation."}])
        mock_conv.messages = []

        # TTS returns audio
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050
        mock_out.wait_async = AsyncMock()

        pipeline = Pipeline(config)

        # Persona should be passed to Conversation
        mock_conv_cls.assert_called_once_with(persona="You are Bob.")

        await pipeline._auto_speak()

        # Should have muted mic, called LLM, spoken, added to conversation, unmuted
        mock_in.mute.assert_called()
        mock_conv.add_assistant.assert_called_once_with("Hello there!")
        mock_tts.synthesize.assert_called()
        mock_in.unmute.assert_called()


def test_pipeline_passes_persona_to_conversation():
    config = Config(persona="You are Alice.")
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:
        Pipeline(config)
        mock_conv_cls.assert_called_once_with(persona="You are Alice.")
