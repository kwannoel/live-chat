import asyncio

import numpy as np
import sounddevice as sd

from live_chat.config import Config


class AudioInput:
    def __init__(self, config: Config):
        self.sample_rate = config.sample_rate
        self.channels = 1
        self.blocksize = 512  # 32ms at 16kHz, required by Silero VAD
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._muted = False

    def set_queue(self, queue: asyncio.Queue[np.ndarray]):
        self._queue = queue

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        if self._queue is not None and self._loop is not None and not self._muted:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
