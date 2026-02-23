import numpy as np
import sounddevice as sd

from live_chat.config import Config


class AudioOutput:
    def __init__(self, config: Config):
        self._sample_rate = config.tts_sample_rate

    def play(self, audio: np.ndarray):
        """Play audio array (float32) through speakers."""
        sd.play(audio, samplerate=self._sample_rate)

    def wait(self):
        """Block until playback finishes."""
        sd.wait()

    def stop(self):
        """Stop any current playback immediately."""
        sd.stop()
