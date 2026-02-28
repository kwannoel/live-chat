import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad


class VAD:
    def __init__(self, threshold: float = 0.5):
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            sampling_rate=16000,
            threshold=threshold,
        )

    def process(self, chunk: np.ndarray) -> dict | None:
        """Process a float32 audio chunk. Returns {'start': n}, {'end': n}, or None."""
        tensor = torch.from_numpy(chunk)
        return self._iterator(tensor)

    def reset(self):
        self._iterator.reset_states()
