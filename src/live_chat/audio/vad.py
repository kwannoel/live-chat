import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from live_chat.config import Config


class VAD:
    def __init__(self, config: Config):
        self.silence_ms = config.vad_silence_ms
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            sampling_rate=config.sample_rate,
        )

    def process(self, chunk: np.ndarray) -> dict | None:
        """Process an audio chunk. Returns {'start': n}, {'end': n}, or None."""
        tensor = torch.from_numpy(chunk.astype(np.float32) / 32768.0)
        return self._iterator(tensor)

    def reset(self):
        self._iterator.reset_states()
