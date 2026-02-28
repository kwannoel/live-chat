import asyncio

import numpy as np
import sounddevice as sd


class AudioOutput:
    def play(self, audio: np.ndarray, sample_rate: int):
        """Play audio array through speakers."""
        sd.play(audio, samplerate=sample_rate)

    def wait(self):
        """Block until playback finishes."""
        sd.wait()

    async def wait_async(self):
        """Wait for playback to finish without blocking the event loop."""
        await asyncio.to_thread(sd.wait)

    def stop(self):
        """Stop any current playback immediately."""
        sd.stop()
