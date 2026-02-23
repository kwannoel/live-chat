import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.vad import VAD
from live_chat.config import Config


def test_vad_init():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load:
        mock_load.return_value = MagicMock()
        vad = VAD(Config())
        assert vad.silence_ms == 700
        mock_load.assert_called_once()


def test_vad_process_returns_none_for_silence():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load, \
         patch("live_chat.audio.vad.VADIterator") as mock_iter_cls:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_iter = MagicMock()
        mock_iter.return_value = None  # no speech detected
        mock_iter_cls.return_value = mock_iter

        vad = VAD(Config())
        chunk = np.zeros(512, dtype=np.int16)
        result = vad.process(chunk)
        assert result is None


def test_vad_process_returns_start_event():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load, \
         patch("live_chat.audio.vad.VADIterator") as mock_iter_cls:
        mock_load.return_value = MagicMock()
        mock_iter = MagicMock()
        mock_iter.return_value = {"start": 0}
        mock_iter_cls.return_value = mock_iter

        vad = VAD(Config())
        chunk = np.zeros(512, dtype=np.int16)
        result = vad.process(chunk)
        assert result == {"start": 0}
