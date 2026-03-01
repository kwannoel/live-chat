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
        segments = result.get("segments", [])
        if not segments:
            return ""

        # Filter Whisper hallucinations (common on silence/echo):
        # 1. High no_speech_prob = Whisper itself thinks no speech
        # 2. Low avg_logprob = Whisper isn't confident in the transcription
        # 3. High compression_ratio = repetitive/degenerate output
        seg = segments[0]
        if seg.get("no_speech_prob", 0) > 0.3:
            return ""
        if seg.get("avg_logprob", 0) < -0.7:
            return ""
        if seg.get("compression_ratio", 0) > 2.4:
            return ""

        return result["text"].strip()
