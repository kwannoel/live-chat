import numpy as np
import sounddevice as sd


class AudioOutput:
    def play(self, audio: np.ndarray, sample_rate: int):
        """Play audio array through speakers."""
        sd.play(audio, samplerate=sample_rate)

    def wait(self):
        """Block until playback finishes."""
        sd.wait()

    def stop(self):
        """Stop any current playback immediately."""
        sd.stop()
