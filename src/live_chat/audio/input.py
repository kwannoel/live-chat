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
        self._stream: sd.InputStream | None = None

    def set_queue(self, queue: asyncio.Queue[np.ndarray]):
        self._queue = queue

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        if self._queue is not None:
            raw = indata[:, 0].copy()
            # Boost quiet MacBook mic to usable levels for VAD/STT
            amplified = np.clip(raw.astype(np.int32) * 10, -32768, 32767).astype(np.int16)
            self._queue.put_nowait(amplified)

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
