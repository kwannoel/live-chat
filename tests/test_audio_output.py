import pytest
import numpy as np
from unittest.mock import patch

from live_chat.audio.output import AudioOutput


@patch("live_chat.audio.output.sd")
def test_audio_output_play(mock_sd):
    output = AudioOutput()
    audio = np.random.randn(22050).astype(np.float32)

    output.play(audio, sample_rate=22050)

    mock_sd.play.assert_called_once()
    args = mock_sd.play.call_args
    assert args[1]["samplerate"] == 22050


@patch("live_chat.audio.output.sd")
def test_audio_output_stop(mock_sd):
    output = AudioOutput()
    output.stop()
    mock_sd.stop.assert_called_once()


@pytest.mark.asyncio
@patch("live_chat.audio.output.sd")
async def test_audio_output_wait_async(mock_sd):
    """wait_async wraps sd.wait in asyncio.to_thread (non-blocking)."""
    output = AudioOutput()
    await output.wait_async()
    mock_sd.wait.assert_called_once()
