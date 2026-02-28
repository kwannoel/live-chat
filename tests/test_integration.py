import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


@pytest.mark.asyncio
async def test_full_pipeline_vad_to_response():
    """Simulate: activate -> VAD speech -> STT -> route -> LLM -> TTS -> listening."""
    config = Config()

    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain") as mock_gain_cls, \
         patch("live_chat.pipeline.VAD") as mock_vad_cls, \
         patch("live_chat.pipeline.WhisperSTT") as mock_stt_cls, \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_gain = mock_gain_cls.return_value
        mock_vad = mock_vad_cls.return_value
        mock_stt = mock_stt_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # AutoGain passes through as float32
        mock_gain.apply.side_effect = lambda c: c.astype(np.float32) / 32768.0

        pipeline = Pipeline(config)
        assert pipeline.state == State.IDLE

        # 1. Activate via Enter
        pipeline.activate()
        assert pipeline.state == State.LISTENING

        # 2. VAD detects speech start
        chunk = np.zeros(512, dtype=np.int16)
        mock_vad.process.return_value = {"start": 0}
        await pipeline._process_chunk(chunk)
        # Should be buffering now

        # 3. Mid-speech chunks (no event)
        mock_vad.process.return_value = None
        await pipeline._process_chunk(chunk)

        # 4. VAD detects speech end
        mock_vad.process.return_value = {"end": 512}
        mock_stt.transcribe.return_value = "What is consciousness?"
        mock_router.route = AsyncMock(return_value=config.deep_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "What is consciousness?"}])
        mock_conv.messages = [{"role": "user", "content": "What is consciousness?"}]

        async def fake_stream(model, system, messages):
            for token in ["That's ", "a deep ", "question."]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050

        await pipeline._process_chunk(chunk)

        # Verify full pipeline executed
        mock_stt.transcribe.assert_called_once()
        mock_router.route.assert_called_once()
        mock_conv.add_user.assert_called_with("What is consciousness?")
        mock_conv.add_assistant.assert_called_once()

        # Should return to LISTENING (not IDLE) for continuous conversation
        assert pipeline.state == State.LISTENING
