import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.wakeword import WakeWordDetector
from live_chat.config import Config


def test_wakeword_init():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()
        detector = WakeWordDetector(Config())
        mock_model_cls.assert_called_once_with(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx",
        )


def test_wakeword_detect_returns_false_for_silence():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.0}
        mock_model_cls.return_value = mock_model

        detector = WakeWordDetector(Config())
        chunk = np.zeros(1280, dtype=np.int16)
        assert detector.detect(chunk) is False


def test_wakeword_detect_returns_true_above_threshold():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.85}
        mock_model_cls.return_value = mock_model

        detector = WakeWordDetector(Config())
        chunk = np.zeros(1280, dtype=np.int16)
        assert detector.detect(chunk) is True
