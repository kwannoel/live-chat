import numpy as np
import mlx_whisper

from live_chat.config import Config

_DEFAULT_MODEL = "mlx-community/whisper-small"


class WhisperSTT:
    def __init__(self, config: Config):
        self._model_path = _DEFAULT_MODEL

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz audio to text."""
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
        )
        return result["text"].strip()
