import numpy as np
from live_chat.audio.gain import AutoGain


def test_autogain_amplifies_quiet_audio():
    gain = AutoGain(target_rms=0.1, window_chunks=31, max_gain=100.0)
    # Feed quiet audio (rms ~0.001)
    quiet = np.full(512, 30, dtype=np.int16)
    result = gain.apply(quiet)
    # Output should be significantly louder
    assert result.dtype == np.float32
    assert np.abs(result).max() > 0.05


def test_autogain_does_not_amplify_silence():
    gain = AutoGain(target_rms=0.1, window_chunks=31, max_gain=100.0)
    silent = np.zeros(512, dtype=np.int16)
    result = gain.apply(silent)
    # Should stay near zero — don't amplify dead silence
    assert np.abs(result).max() < 0.01


def test_autogain_clamps_to_max_gain():
    gain = AutoGain(target_rms=0.1, window_chunks=31, max_gain=5.0)
    # Very quiet audio
    quiet = np.full(512, 10, dtype=np.int16)
    result = gain.apply(quiet)
    # Gain should be clamped — output won't reach target
    raw_rms = np.sqrt(np.mean((quiet.astype(np.float32) / 32768.0) ** 2))
    assert np.abs(result).max() <= raw_rms * 5.0 + 0.001
