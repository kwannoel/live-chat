from collections.abc import Iterator

import numpy as np
from kokoro import KPipeline

from live_chat.config import Config


class KokoroTTS:
    def __init__(self, config: Config):
        self._voice = config.tts_voice
        self._speed = config.tts_speed
        self._pipeline = KPipeline(lang_code="a")  # American English

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks (float32, 24kHz) per sentence."""
        for _gs, _ps, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            yield audio
