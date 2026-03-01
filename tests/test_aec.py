import numpy as np

from live_chat.audio.aec import EchoCanceller


def test_aec_passthrough_without_reference():
    """With no reference signal, AEC passes audio through unchanged."""
    aec = EchoCanceller()
    chunk = (np.random.randn(512) * 3000).astype(np.int16)

    # Not active — should return input unchanged
    result = aec.cancel(chunk)
    assert np.array_equal(result, chunk)


def test_aec_reduces_echo():
    """AEC reduces a known echo signal after convergence."""
    aec = EchoCanceller()

    # Create a repeating reference tone
    ref_tone = (np.sin(np.linspace(0, 40 * np.pi, 22050)) * 10000).astype(np.int16)
    aec.set_reference(ref_tone, source_rate=16000)

    # Simulate mic picking up the echo (attenuated reference)
    echo_mic = (ref_tone[:512] * 0.5).astype(np.int16)

    # Run several frames to let AEC adapt
    for i in range(0, len(ref_tone) - 512, 512):
        frame = (ref_tone[i:i + 512] * 0.5).astype(np.int16)
        aec.cancel(frame)

    # After convergence, echo should be reduced
    result = aec.cancel(echo_mic)
    echo_rms = float(np.sqrt(np.mean(echo_mic.astype(float) ** 2)))
    result_rms = float(np.sqrt(np.mean(result.astype(float) ** 2)))

    assert result_rms < echo_rms, f"AEC didn't reduce echo: {result_rms} >= {echo_rms}"


def test_aec_clear_stops_processing():
    """After clear(), AEC stops processing and passes through."""
    aec = EchoCanceller()
    ref = np.zeros(1600, dtype=np.int16)
    aec.set_reference(ref, source_rate=16000)
    assert aec._active is True

    aec.clear()
    assert aec._active is False

    chunk = (np.random.randn(512) * 3000).astype(np.int16)
    result = aec.cancel(chunk)
    assert np.array_equal(result, chunk)
