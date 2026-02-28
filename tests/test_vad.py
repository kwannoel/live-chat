import numpy as np
from unittest.mock import patch
from live_chat.audio.vad import VAD


@patch("live_chat.audio.vad.load_silero_vad")
def test_vad_init(mock_load):
    vad = VAD()
    mock_load.assert_called_once()


@patch("live_chat.audio.vad.load_silero_vad")
@patch("live_chat.audio.vad.VADIterator")
def test_vad_process_returns_none_for_silence(mock_iter_cls, mock_load):
    mock_iter = mock_iter_cls.return_value
    mock_iter.return_value = None
    vad = VAD()
    result = vad.process(np.zeros(512, dtype=np.float32))
    assert result is None


@patch("live_chat.audio.vad.load_silero_vad")
@patch("live_chat.audio.vad.VADIterator")
def test_vad_process_returns_start_event(mock_iter_cls, mock_load):
    mock_iter = mock_iter_cls.return_value
    mock_iter.return_value = {"start": 0}
    vad = VAD()
    result = vad.process(np.zeros(512, dtype=np.float32))
    assert result == {"start": 0}
