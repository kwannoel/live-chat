"""End-to-end test using real recorded audio through the full pipeline.

Uses a pre-recorded fixture of "Hello, nice to meet you" to verify:
  1. STT transcribes the audio correctly (real Whisper, no mocks)
  2. Router selects a model
  3. LLM streams a response
  4. TTS synthesizes audio output
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from live_chat.config import Config
from live_chat.pipeline import Pipeline, State
from live_chat.stt.whisper import WhisperSTT

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hello_nice_to_meet_you.npy"


@pytest.fixture
def recorded_audio():
    """Load the real recorded audio fixture."""
    assert FIXTURE_PATH.exists(), f"Missing fixture: {FIXTURE_PATH}"
    return np.load(FIXTURE_PATH)


def test_stt_transcribes_recorded_audio(recorded_audio):
    """Real Whisper STT produces the expected transcription."""
    config = Config()
    stt = WhisperSTT(config)

    audio_f32 = recorded_audio.astype(np.float32) / 32768.0
    text = stt.transcribe(audio_f32)

    assert "hello" in text.lower()
    assert "nice to meet you" in text.lower()


@pytest.mark.asyncio
async def test_e2e_recorded_audio_through_pipeline(recorded_audio):
    """Full pipeline: real STT + mocked Router/LLM/TTS produces a response."""
    config = Config()
    chunk_size = 512

    with patch("live_chat.pipeline.AudioInput") as mock_ai, \
         patch("live_chat.pipeline.AudioOutput") as mock_ao, \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_router = mock_router_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # Router returns fast model
        mock_router.route = AsyncMock(return_value=config.fast_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "Hello, nice to meet you."}])
        mock_conv.messages = [{"role": "user", "content": "Hello, nice to meet you."}]

        # LLM streams a greeting back
        async def fake_stream(model, system, messages):
            for token in ["Nice to ", "meet you ", "too!"]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050

        # Build pipeline (real STT, everything else mocked)
        with patch("live_chat.pipeline.WhisperSTT", wraps=WhisperSTT) as real_stt:
            pipeline = Pipeline(config)
            # Replace the mocked STT with the real one
            pipeline._stt = WhisperSTT(config)

        # 1. Activate recording
        pipeline.activate()
        assert pipeline.state == State.LISTENING

        # 2. Feed recorded audio in chunks (simulating mic input)
        for i in range(0, len(recorded_audio), chunk_size):
            chunk = recorded_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            await pipeline._process_chunk(chunk)

        # 3. Stop recording (second activate)
        pipeline.activate()
        # Process one more chunk to trigger the stop event
        await pipeline._process_chunk(np.zeros(chunk_size, dtype=np.int16))

        # 4. Verify the full pipeline ran
        mock_conv.add_user.assert_called_once()
        user_text = mock_conv.add_user.call_args[0][0]
        assert "hello" in user_text.lower(), f"STT produced: '{user_text}'"
        assert "nice to meet you" in user_text.lower(), f"STT produced: '{user_text}'"

        mock_router.route.assert_called_once()
        mock_conv.add_assistant.assert_called_once_with("Nice to meet you too!")
        mock_tts.synthesize.assert_called()
