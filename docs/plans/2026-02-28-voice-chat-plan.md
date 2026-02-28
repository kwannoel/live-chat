# Continuous Voice Chat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace push-to-talk with continuous VAD-driven voice conversation with barge-in support.

**Architecture:** Add an AutoGain normalizer to adapt to any mic level, re-add Silero VAD for speech boundary detection, and rewrite the pipeline state machine for continuous listening with barge-in during TTS playback.

**Tech Stack:** silero-vad, torch, numpy, existing pipeline components

---

### Task 1: AutoGain normalizer

**Files:**
- Create: `src/live_chat/audio/gain.py`
- Create: `tests/test_gain.py`

**Step 1: Write the failing tests**

```python
# tests/test_gain.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gain.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'live_chat.audio.gain'"

**Step 3: Write minimal implementation**

```python
# src/live_chat/audio/gain.py
from collections import deque

import numpy as np


class AutoGain:
    """Automatic gain control using rolling RMS tracking."""

    def __init__(
        self,
        target_rms: float = 0.1,
        window_chunks: int = 31,
        max_gain: float = 100.0,
    ):
        self._target_rms = target_rms
        self._max_gain = max_gain
        self._rms_history: deque[float] = deque(maxlen=window_chunks)

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """Convert int16 chunk to gain-adjusted float32."""
        audio = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self._rms_history.append(rms)
        avg_rms = sum(self._rms_history) / len(self._rms_history)

        if avg_rms < 1e-6:
            return audio

        gain = min(self._target_rms / avg_rms, self._max_gain)
        return audio * gain
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gain.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/live_chat/audio/gain.py tests/test_gain.py
git commit -m "feat: add AutoGain normalizer for adaptive mic levels"
```

---

### Task 2: Re-add Silero VAD

**Files:**
- Create: `src/live_chat/audio/vad.py`
- Create: `tests/test_vad.py`
- Modify: `pyproject.toml`

**Step 1: Add silero-vad dependency**

In `pyproject.toml`, add `"silero-vad>=5.1"` and `"torch>=2.0"` to `dependencies`.

**Step 2: Write the failing tests**

```python
# tests/test_vad.py
import numpy as np
from unittest.mock import patch, MagicMock
from live_chat.audio.vad import VAD


@patch("live_chat.audio.vad.load_silero_vad")
def test_vad_init(mock_load):
    vad = VAD()
    mock_load.assert_called_once()


@patch("live_chat.audio.vad.load_silero_vad")
@patch("live_chat.audio.vad.VADIterator")
def test_vad_process_returns_none_for_silence(mock_iter_cls, mock_load):
    mock_iter = mock_iter_cls.return_value
    mock_iter.return_value = None
    vad = VAD()
    result = vad.process(np.zeros(512, dtype=np.float32))
    assert result is None


@patch("live_chat.audio.vad.load_silero_vad")
@patch("live_chat.audio.vad.VADIterator")
def test_vad_process_returns_start_event(mock_iter_cls, mock_load):
    mock_iter = mock_iter_cls.return_value
    mock_iter.return_value = {"start": 0}
    vad = VAD()
    result = vad.process(np.zeros(512, dtype=np.float32))
    assert result == {"start": 0}
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_vad.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/live_chat/audio/vad.py
import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad


class VAD:
    def __init__(self, threshold: float = 0.5):
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            sampling_rate=16000,
            threshold=threshold,
        )

    def process(self, chunk: np.ndarray) -> dict | None:
        """Process a float32 audio chunk. Returns {'start': n}, {'end': n}, or None."""
        tensor = torch.from_numpy(chunk)
        return self._iterator(tensor)

    def reset(self):
        self._iterator.reset_states()
```

Note: this VAD now takes **float32** input (post-AutoGain), not int16 like before.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_vad.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/live_chat/audio/vad.py tests/test_vad.py pyproject.toml
git commit -m "feat: re-add Silero VAD with float32 input"
```

---

### Task 3: Remove fixed gain from AudioInput

**Files:**
- Modify: `src/live_chat/audio/input.py:20-27`
- Modify: `tests/test_audio_input.py`

**Step 1: Update the test**

Replace the callback test in `tests/test_audio_input.py`:

```python
@patch("live_chat.audio.input.sd")
def test_audio_input_callback_puts_raw_audio_to_queue(mock_sd):
    config = Config()
    audio_input = AudioInput(config)
    queue = asyncio.Queue()
    audio_input.set_queue(queue)

    # Simulate a callback with audio data
    fake_audio = np.ones((512, 1), dtype=np.int16) * 100
    audio_input._callback(fake_audio, 512, None, None)

    assert not queue.empty()
    chunk = queue.get_nowait()
    assert chunk.shape == (512,)
    assert chunk.dtype == np.int16
    # Raw audio — no gain applied
    assert np.all(chunk == 100)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_input.py::test_audio_input_callback_puts_raw_audio_to_queue -v`
Expected: FAIL (chunk values are 1000 due to 10x gain)

**Step 3: Update AudioInput to send raw audio**

Replace `_callback` in `src/live_chat/audio/input.py`:

```python
    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        if self._queue is not None:
            self._queue.put_nowait(indata[:, 0].copy())
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_input.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/live_chat/audio/input.py tests/test_audio_input.py
git commit -m "refactor: remove fixed gain from AudioInput (AutoGain replaces it)"
```

---

### Task 4: Rewrite pipeline for continuous voice chat

**Files:**
- Modify: `src/live_chat/pipeline.py` (full rewrite of state machine)
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_integration.py`

**Step 1: Update pipeline unit test**

```python
# tests/test_pipeline.py
from unittest.mock import patch

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


def test_pipeline_initial_state():
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation"):
        pipeline = Pipeline(Config())
        assert pipeline.state == State.IDLE


def test_pipeline_state_enum():
    assert State.IDLE.value == "idle"
    assert State.LISTENING.value == "listening"
    assert State.THINKING.value == "thinking"
    assert State.SPEAKING.value == "speaking"
```

**Step 2: Update integration test**

```python
# tests/test_integration.py
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


@pytest.mark.asyncio
async def test_full_pipeline_vad_to_response():
    """Simulate: activate -> VAD speech -> STT -> route -> LLM -> TTS -> listening."""
    config = Config()

    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain") as mock_gain_cls, \
         patch("live_chat.pipeline.VAD") as mock_vad_cls, \
         patch("live_chat.pipeline.WhisperSTT") as mock_stt_cls, \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_gain = mock_gain_cls.return_value
        mock_vad = mock_vad_cls.return_value
        mock_stt = mock_stt_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # AutoGain passes through as float32
        mock_gain.apply.side_effect = lambda c: c.astype(np.float32) / 32768.0

        pipeline = Pipeline(config)
        assert pipeline.state == State.IDLE

        # 1. Activate via Enter
        pipeline.activate()
        assert pipeline.state == State.LISTENING

        # 2. VAD detects speech start
        chunk = np.zeros(512, dtype=np.int16)
        mock_vad.process.return_value = {"start": 0}
        await pipeline._process_chunk(chunk)
        # Should be buffering now

        # 3. Mid-speech chunks (no event)
        mock_vad.process.return_value = None
        await pipeline._process_chunk(chunk)

        # 4. VAD detects speech end
        mock_vad.process.return_value = {"end": 512}
        mock_stt.transcribe.return_value = "What is consciousness?"
        mock_router.route = AsyncMock(return_value=config.deep_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "What is consciousness?"}])
        mock_conv.messages = [{"role": "user", "content": "What is consciousness?"}]

        async def fake_stream(model, system, messages):
            for token in ["That's ", "a deep ", "question."]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050

        await pipeline._process_chunk(chunk)

        # Verify full pipeline executed
        mock_stt.transcribe.assert_called_once()
        mock_router.route.assert_called_once()
        mock_conv.add_user.assert_called_with("What is consciousness?")
        mock_conv.add_assistant.assert_called_once()

        # Should return to LISTENING (not IDLE) for continuous conversation
        assert pipeline.state == State.LISTENING
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py tests/test_integration.py -v`
Expected: FAIL (AutoGain and VAD not imported in pipeline)

**Step 4: Rewrite pipeline**

```python
# src/live_chat/pipeline.py
import asyncio
import enum

import numpy as np

from live_chat.audio.gain import AutoGain
from live_chat.audio.input import AudioInput
from live_chat.audio.output import AudioOutput
from live_chat.audio.vad import VAD
from live_chat.config import Config
from live_chat.llm.client import LLMClient
from live_chat.llm.conversation import Conversation
from live_chat.llm.router import Router
from live_chat.stt.whisper import WhisperSTT
from live_chat.tts.piper_tts import PiperTTS


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.state = State.IDLE
        self._on_state_change: callable = None
        self._on_transcript: callable = None

        # Components
        self._audio_in = AudioInput(config)
        self._audio_out = AudioOutput()
        self._gain = AutoGain()
        self._vad = VAD()
        self._stt = WhisperSTT(config)
        self._tts = PiperTTS(config)
        self._llm = LLMClient(config)
        self._router = Router(config)
        self._conversation = Conversation()

        # Audio queue
        self._audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._audio_in.set_queue(self._audio_queue)

        # Speech buffer for accumulating audio during listening
        self._speech_buffer: list[np.ndarray] = []
        self._interrupted = False

    def on_state_change(self, callback: callable):
        self._on_state_change = callback

    def on_transcript(self, callback: callable):
        """callback(role, text, model) — called for user/assistant transcripts."""
        self._on_transcript = callback

    def _set_state(self, state: State):
        self.state = state
        if self._on_state_change:
            self._on_state_change(state)

    def activate(self):
        """Start continuous listening (called from keyboard input)."""
        if self.state == State.IDLE:
            self._set_state(State.LISTENING)
            self._speech_buffer.clear()
            self._vad.reset()

    async def run(self):
        """Main loop — process audio chunks from the mic."""
        self._audio_in.start()
        try:
            while True:
                chunk = await self._audio_queue.get()
                await self._process_chunk(chunk)
        finally:
            self._audio_in.stop()

    async def _process_chunk(self, chunk: np.ndarray):
        # Apply auto-gain to get normalized float32
        normalized = self._gain.apply(chunk)

        if self.state == State.IDLE:
            pass

        elif self.state == State.LISTENING:
            event = self._vad.process(normalized)

            if event and "start" in event:
                self._speech_buffer = [normalized]
            elif event and "end" in event:
                self._speech_buffer.append(normalized)
                await self._process_speech()
            elif self._speech_buffer:
                self._speech_buffer.append(normalized)

        elif self.state == State.SPEAKING:
            event = self._vad.process(normalized)
            if event and "start" in event:
                self._interrupted = True
                self._audio_out.stop()
                self._speech_buffer = [normalized]
                self._set_state(State.LISTENING)

    async def _process_speech(self):
        """Transcribe buffered speech, route, call LLM, speak response."""
        if not self._speech_buffer:
            return

        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()

        # STT (audio is already float32 from AutoGain)
        text = self._stt.transcribe(audio)
        if not text:
            return

        self._set_state(State.THINKING)
        if self._on_transcript:
            self._on_transcript("user", text, None)
        self._conversation.add_user(text)

        # Route
        model = await self._router.route(text, self._conversation.messages[:-1])
        system, messages = self._conversation.for_api()

        # Stream LLM response, buffer sentences, speak as they complete
        self._set_state(State.SPEAKING)
        full_response = []
        sentence_buffer = []

        async for token in self._llm.stream(model, system, messages):
            full_response.append(token)
            sentence_buffer.append(token)

            current = "".join(sentence_buffer)
            if any(current.rstrip().endswith(p) for p in ".!?"):
                sentence = current.strip()
                if sentence:
                    await self._speak_sentence(sentence)
                sentence_buffer.clear()

            if self._interrupted:
                self._interrupted = False
                break

        # Speak any remaining text
        remaining = "".join(sentence_buffer).strip()
        if remaining and not self._interrupted:
            await self._speak_sentence(remaining)

        response_text = "".join(full_response)
        self._conversation.add_assistant(response_text)
        if self._on_transcript:
            self._on_transcript("assistant", response_text, model)

        # Return to listening for continuous conversation
        if self.state == State.SPEAKING:
            self._set_state(State.LISTENING)
            self._vad.reset()

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence."""
        for audio_chunk in self._tts.synthesize(sentence):
            self._audio_out.play(audio_chunk, sample_rate=self._tts.sample_rate)
            self._audio_out.wait()
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/test_integration.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/live_chat/pipeline.py tests/test_pipeline.py tests/test_integration.py
git commit -m "feat: rewrite pipeline for continuous VAD-driven voice chat with barge-in"
```

---

### Task 5: Update main.py and e2e test

**Files:**
- Modify: `src/live_chat/main.py:22-37`
- Modify: `tests/test_e2e_voice.py`

**Step 1: Update main.py UI text**

Change `_STATE_DISPLAY` and the instructions:

```python
_STATE_DISPLAY = {
    State.IDLE: "[dim]Press Enter to start...[/dim]",
    State.LISTENING: "[bold green]Listening...[/bold green]",
    State.THINKING: "[bold yellow]Thinking...[/bold yellow]",
    State.SPEAKING: "[bold blue]Speaking...[/bold blue]",
}
```

Change the instructions line to:

```python
    console.print("Press [bold]Enter[/bold] to start. [bold]Ctrl+C[/bold] to quit.\n")
```

The keyboard loop stays the same — Enter calls `pipeline.activate()`, which only does something when state is IDLE.

**Step 2: Update e2e test to use VAD-driven flow**

In `tests/test_e2e_voice.py`, update `test_e2e_recorded_audio_through_pipeline`:

- Add patches for `AutoGain` and `VAD`
- Instead of calling `pipeline.activate()` twice (start/stop), call it once then simulate VAD start/end events
- Mock `AutoGain.apply` to pass through as float32
- Mock `VAD.process` to return `{"start": 0}` on first chunk, `None` for middle chunks, `{"end": 512}` on last chunk

```python
@pytest.mark.asyncio
async def test_e2e_recorded_audio_through_pipeline(recorded_audio):
    """Full pipeline: real STT + mocked Router/LLM/TTS produces a response."""
    config = Config()
    chunk_size = 512

    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain") as mock_gain_cls, \
         patch("live_chat.pipeline.VAD") as mock_vad_cls, \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_gain = mock_gain_cls.return_value
        mock_vad = mock_vad_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # AutoGain passes through as float32
        mock_gain.apply.side_effect = lambda c: c.astype(np.float32) / 32768.0

        # Router returns fast model
        mock_router.route = AsyncMock(return_value=config.fast_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "Hello, nice to meet you."}])
        mock_conv.messages = [{"role": "user", "content": "Hello, nice to meet you."}]

        async def fake_stream(model, system, messages):
            for token in ["Nice to ", "meet you ", "too!"]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050

        # Build pipeline with real STT
        pipeline = Pipeline(config)
        pipeline._stt = WhisperSTT(config)

        # Activate continuous listening
        pipeline.activate()
        assert pipeline.state == State.LISTENING

        # Feed audio: first chunk = VAD start, middle = None, last = VAD end
        chunks = []
        for i in range(0, len(recorded_audio), chunk_size):
            chunk = recorded_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.append(chunk)

        # VAD start on first chunk
        mock_vad.process.return_value = {"start": 0}
        await pipeline._process_chunk(chunks[0])

        # Middle chunks — no VAD event
        mock_vad.process.return_value = None
        for chunk in chunks[1:-1]:
            await pipeline._process_chunk(chunk)

        # VAD end on last chunk
        mock_vad.process.return_value = {"end": 512}
        await pipeline._process_chunk(chunks[-1])

        # Verify
        mock_conv.add_user.assert_called_once()
        user_text = mock_conv.add_user.call_args[0][0]
        assert "hello" in user_text.lower(), f"STT produced: '{user_text}'"
        assert "nice to meet you" in user_text.lower(), f"STT produced: '{user_text}'"

        mock_router.route.assert_called_once()
        mock_conv.add_assistant.assert_called_once_with("Nice to meet you too!")
        assert pipeline.state == State.LISTENING  # continuous mode
```

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/live_chat/main.py tests/test_e2e_voice.py
git commit -m "feat: update UI for continuous voice chat, update e2e test"
```

---

### Task 6: Manual test and version bump

**Step 1: Run the app**

```bash
source .venv/bin/activate
python -m live_chat.main
```

Press Enter. Speak a sentence. Verify:
- VAD detects speech start/end
- STT transcribes correctly
- LLM responds
- TTS speaks the response
- Returns to listening automatically

**Step 2: Test barge-in**

While the agent is speaking, say something. Verify it stops and starts listening.

**Step 3: Bump version**

In `pyproject.toml`, change `version = "1.0.0"` to `version = "1.1.0"`.

**Step 4: Commit and tag**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 1.1.0"
git tag -a v1.1.0 -m "v1.1.0 — continuous voice chat with VAD and barge-in"
```
