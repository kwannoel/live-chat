# Continuous Voice Chat Design

**Goal:** Replace push-to-talk with continuous voice conversation — VAD-driven speech detection, automatic turn-taking, and barge-in support.

## Audio Pipeline

```
Mic (512 samples @ 16kHz)
  → AudioInput (raw int16, no fixed gain)
  → AutoGain (rolling RMS over ~1s, scales float32 to target_rms=0.1, max gain 100x)
  → Silero VAD (threshold=0.5, produces start/end events)
  → Speech buffer → STT → Router → LLM → TTS → Speaker
```

**AutoGain** (`audio/gain.py`): Tracks rolling RMS over ~31 chunks (~1s). Computes `target_rms / current_rms`, clamped to max 100x. Applies gain in float32 space — no int16 clipping. Replaces the fixed 10x gain in AudioInput.

## State Machine

```
IDLE →(Enter)→ LISTENING →(VAD end)→ THINKING → SPEAKING → LISTENING
                                                    |
                                        (VAD start = barge-in)
                                                    ↓
                                                LISTENING
```

- `IDLE → LISTENING`: Single Enter press, then stays in voice mode.
- `LISTENING`: VAD detects speech start/end. Buffers audio between start and end, then sends to STT.
- `THINKING`: STT → Router → LLM streaming begins.
- `SPEAKING`: TTS plays sentence-by-sentence. VAD still runs on mic input.
- Barge-in: During SPEAKING, if VAD detects speech → stop TTS, transition to LISTENING with barge-in audio in buffer.
- After SPEAKING completes, returns to LISTENING (not IDLE).

## Echo Handling

During SPEAKING, the mic picks up TTS from speakers. Simple solution: raise VAD threshold to 0.7 during SPEAKING state (speaker output is quieter/distant vs. direct speech into mic).

## Files

| Action | File |
|--------|------|
| Create | `audio/gain.py` — AutoGain class |
| Re-create | `audio/vad.py` — Silero VAD wrapper |
| Modify | `audio/input.py` — remove fixed gain |
| Modify | `pipeline.py` — VAD, auto-gain, continuous listening, barge-in |
| Modify | `main.py` — update UI text |
| Modify | `pyproject.toml` — re-add silero-vad |
| Create | `tests/test_gain.py` |
| Create | `tests/test_vad.py` |
| Modify | `tests/test_pipeline.py`, `test_integration.py`, `test_audio_input.py` |
