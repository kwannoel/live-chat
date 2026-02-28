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
        # Warmup: model needs several frames before prediction buffer populates
        for _ in range(10):
            self._model.predict(np.zeros(1280, dtype=np.int16))

    def detect(self, chunk: np.ndarray) -> bool:
        """Returns True if the wake word is detected in this chunk."""
        # Amplify quiet mic input — models expect louder audio
        amplified = np.clip(chunk.astype(np.int32) * 10, -32768, 32767).astype(np.int16)
        prediction = self._model.predict(amplified)
        score = float(prediction.get(self._wake_word, 0.0))
        return score > self._threshold
