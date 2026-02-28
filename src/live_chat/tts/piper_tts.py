from collections.abc import Iterator
from pathlib import Path

import numpy as np
from piper.voice import PiperVoice

from live_chat.config import Config

class PiperTTS:
    def __init__(self, config: Config):
        self._voice: PiperVoice | None = None
        self._model_name = config.tts_voice
        self._sample_rate: int = 22050

    def _ensure_model(self):
        """Load Piper voice model on first use."""
        if self._voice is not None:
            return

        model_dir = Path.home() / ".local" / "share" / "piper" / "voices"
        model_path = model_dir / f"{self._model_name}.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}\n"
                f"Download voices from: https://huggingface.co/rhasspy/piper-voices\n"
                f"Place the .onnx and .onnx.json files in: {model_dir}"
            )

        self._voice = PiperVoice.load(str(model_path))
        self._sample_rate = self._voice.config.sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks (int16, 22050Hz) per sentence."""
        self._ensure_model()
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_array
