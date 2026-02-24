import numpy as np
from openwakeword.model import Model

from live_chat.config import Config


class WakeWordDetector:
    def __init__(self, config: Config):
        self._wake_word = config.wake_word
        self._threshold = config.wakeword_threshold
        self._model = Model(
            wakeword_models=[self._wake_word],
            inference_framework="onnx",
        )

    def detect(self, chunk: np.ndarray) -> bool:
        """Returns True if the wake word is detected in this chunk."""
        prediction = self._model.predict(chunk)
        score = prediction.get(self._wake_word, 0.0)
        return score > self._threshold
