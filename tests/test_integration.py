import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


@pytest.mark.asyncio
async def test_full_pipeline_wake_to_response():
    """Simulate: wake word -> speech -> STT -> route -> LLM -> TTS -> done."""
    config = Config()

    with patch("live_chat.pipeline.AudioInput") as mock_ai, \
         patch("live_chat.pipeline.AudioOutput") as mock_ao, \
         patch("live_chat.pipeline.VAD") as mock_vad_cls, \
         patch("live_chat.pipeline.WakeWordDetector") as mock_ww_cls, \
         patch("live_chat.pipeline.WhisperSTT") as mock_stt_cls, \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        # Setup mocks
        mock_ww = mock_ww_cls.return_value
        mock_vad = mock_vad_cls.return_value
        mock_stt = mock_stt_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_conv = mock_conv_cls.return_value

        pipeline = Pipeline(config)

        # 1. Wake word detected
        mock_ww.detect.return_value = True
        chunk = np.zeros(1280, dtype=np.int16)
        await pipeline._process_chunk(chunk)
        assert pipeline.state == State.LISTENING

        # 2. Speech starts
        mock_ww.detect.return_value = False
        mock_vad.process.return_value = {"start": 0}
        await pipeline._process_chunk(chunk)

        # 3. Speech ends
        mock_vad.process.return_value = {"end": 512}
        mock_stt.transcribe.return_value = "What is consciousness?"
        mock_router.route = AsyncMock(return_value=config.deep_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "What is consciousness?"}])
        mock_conv.messages = [{"role": "user", "content": "What is consciousness?"}]

        async def fake_stream(model, system, messages):
            for token in ["That's ", "a deep ", "question."]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([
            np.zeros(22050, dtype=np.int16)
        ])
        mock_tts.sample_rate = 22050

        await pipeline._process_chunk(chunk)

        # Verify the pipeline processed through all states
        mock_stt.transcribe.assert_called_once()
        mock_router.route.assert_called_once()
        mock_conv.add_user.assert_called_with("What is consciousness?")
        mock_conv.add_assistant.assert_called_once()
