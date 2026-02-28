import numpy as np
import mlx_whisper

from live_chat.config import Config

_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class WhisperSTT:
    def __init__(self, config: Config):
        self._model_path = _DEFAULT_MODEL

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz audio to text. Returns empty string if no speech detected."""
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            language="en",
        )
        # Filter out Whisper hallucinations on noise
        segments = result.get("segments", [])
        if segments and segments[0].get("no_speech_prob", 0) > 0.5:
            return ""
        return result["text"].strip()
