import numpy as np
from unittest.mock import patch, MagicMock, call

from live_chat.audio.output import AudioOutput
from live_chat.config import Config


@patch("live_chat.audio.output.sd")
def test_audio_output_play(mock_sd):
    output = AudioOutput(Config())
    audio = np.random.randn(24000).astype(np.float32)

    output.play(audio)

    mock_sd.play.assert_called_once()
    args = mock_sd.play.call_args
    assert args[1]["samplerate"] == 24000


@patch("live_chat.audio.output.sd")
def test_audio_output_stop(mock_sd):
    output = AudioOutput(Config())
    output.stop()
    mock_sd.stop.assert_called_once()
