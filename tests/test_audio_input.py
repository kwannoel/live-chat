import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.input import AudioInput
from live_chat.config import Config


def test_audio_input_init():
    config = Config()
    audio_input = AudioInput(config)
    assert audio_input.sample_rate == 16000
    assert audio_input.channels == 1
    assert audio_input.blocksize == 512  # 32ms at 16kHz, required by Silero VAD


@patch("live_chat.audio.input.sd")
def test_audio_input_callback_puts_raw_audio_to_queue(mock_sd):
    config = Config()
    audio_input = AudioInput(config)
    queue = asyncio.Queue()
    audio_input.set_queue(queue)

    # Simulate a callback with audio data
    fake_audio = np.ones((512, 1), dtype=np.int16) * 100
    audio_input._callback(fake_audio, 512, None, None)

    assert not queue.empty()
    chunk = queue.get_nowait()
    assert chunk.shape == (512,)
    assert chunk.dtype == np.int16
    # Raw audio — no gain applied
    assert np.all(chunk == 100)
