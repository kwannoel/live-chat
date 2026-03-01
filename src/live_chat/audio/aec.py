"""Acoustic Echo Cancellation using pyaec (SpeexDSP-based).

Subtracts the known TTS playback signal from mic input so the pipeline
only hears the local speaker, not speaker echo.
"""

import numpy as np
from pyaec import Aec


# AEC works in 10ms frames at 16kHz = 160 samples
_FRAME_SIZE = 160
# Filter length: longer = better echo cancellation but more CPU
# 1600 samples = 100ms, covers typical speaker-to-mic delay
_FILTER_LENGTH = 1600
_SAMPLE_RATE = 16000


class EchoCanceller:
    def __init__(self):
        self._aec = Aec(
            frame_size=_FRAME_SIZE,
            filter_length=_FILTER_LENGTH,
            sample_rate=_SAMPLE_RATE,
        )
        self._ref_buffer: list[np.ndarray] = []
        self._active = False

    def set_reference(self, audio: np.ndarray, source_rate: int):
        """Register TTS audio being played as the echo reference.

        Resamples from source_rate to 16kHz and stores in chunks.
        """
        # Resample from TTS rate (e.g. 22050) to 16kHz
        if source_rate != _SAMPLE_RATE:
            ratio = _SAMPLE_RATE / source_rate
            indices = np.arange(0, len(audio), 1 / ratio).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        # Ensure int16
        if audio.dtype != np.int16:
            audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        # Split into _FRAME_SIZE chunks
        for i in range(0, len(audio), _FRAME_SIZE):
            chunk = audio[i:i + _FRAME_SIZE]
            if len(chunk) == _FRAME_SIZE:
                self._ref_buffer.append(chunk)

        self._active = True

    def cancel(self, chunk: np.ndarray) -> np.ndarray:
        """Process a mic chunk (int16, 512 samples at 16kHz).

        Splits into 160-sample frames, applies AEC with reference signal,
        returns cleaned chunk.
        """
        if not self._active:
            return chunk

        output_frames = []
        for i in range(0, len(chunk), _FRAME_SIZE):
            frame = chunk[i:i + _FRAME_SIZE]
            if len(frame) < _FRAME_SIZE:
                output_frames.append(frame)
                continue

            # Get reference frame (or silence if none available)
            if self._ref_buffer:
                ref_frame = self._ref_buffer.pop(0)
            else:
                ref_frame = np.zeros(_FRAME_SIZE, dtype=np.int16)

            result = self._aec.cancel_echo(frame.tobytes(), ref_frame.tobytes())
            # pyaec returns list of signed byte values
            byte_vals = bytearray((v + 256) % 256 for v in result)
            output_frames.append(np.frombuffer(byte_vals, dtype=np.int16))

        return np.concatenate(output_frames) if output_frames else chunk

    def clear(self):
        """Clear reference buffer (e.g. when playback is stopped)."""
        self._ref_buffer.clear()
        self._active = False
