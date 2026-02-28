from collections import deque

import numpy as np


class AutoGain:
    """Automatic gain control using rolling RMS tracking."""

    def __init__(
        self,
        target_rms: float = 0.1,
        window_chunks: int = 31,
        max_gain: float = 100.0,
    ):
        self._target_rms = target_rms
        self._max_gain = max_gain
        self._rms_history: deque[float] = deque(maxlen=window_chunks)

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """Convert int16 chunk to gain-adjusted float32."""
        audio = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self._rms_history.append(rms)
        avg_rms = sum(self._rms_history) / len(self._rms_history)

        if avg_rms < 1e-6:
            return audio

        gain = min(self._target_rms / avg_rms, self._max_gain)
        return np.clip(audio * gain, -1.0, 1.0)
