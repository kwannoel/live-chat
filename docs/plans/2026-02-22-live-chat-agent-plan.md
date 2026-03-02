# Live Chat Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a voice-first Python CLI agent that auto-routes between Haiku (fast) and Opus/Sonnet (deep) for intelligent idea discussion, using local STT/TTS with wake word + VAD.

**Architecture:** Async streaming pipeline — mic audio flows through wake word detection, VAD, Whisper STT, a Haiku-powered router, the chosen LLM, local Kokoro TTS, and out to speakers. All components communicate via asyncio queues.

**Tech Stack:** Python 3.12, sounddevice, silero-vad, openwakeword, mlx-whisper, kokoro, anthropic SDK, rich, asyncio.

**Design doc:** `docs/plans/2026-02-22-live-chat-agent-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/live_chat/__init__.py`
- Create: `src/live_chat/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "live-chat"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "sounddevice>=0.5",
    "silero-vad>=5.1",
    "openwakeword>=0.6",
    "mlx-whisper>=0.4",
    "kokoro>=0.9",
    "anthropic>=0.42",
    "rich>=14.0",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[project.scripts]
live-chat = "live_chat.main:main"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**Step 2: Create src/live_chat/__init__.py**

```python
"""Live Chat — voice-first conversational agent."""
```

**Step 3: Create tests/__init__.py**

Empty file.

**Step 4: Write the failing test for config**

Create `tests/test_config.py`:

```python
from live_chat.config import Config


def test_default_config():
    config = Config()
    assert config.wake_word == "hey_jarvis"
    assert config.vad_silence_ms == 700
    assert config.active_timeout_s == 30
    assert config.fast_model == "claude-haiku-4-5-20241022"
    assert config.deep_model == "claude-sonnet-4-5-20250929"
    assert config.tts_voice == "af_heart"
    assert config.sample_rate == 16000
    assert config.tts_sample_rate == 24000


def test_config_from_dict():
    config = Config.from_dict({"wake_word": "alexa", "deep_model": "claude-opus-4-6"})
    assert config.wake_word == "alexa"
    assert config.deep_model == "claude-opus-4-6"
    # defaults preserved
    assert config.fast_model == "claude-haiku-4-5-20241022"
```

**Step 5: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: FAIL (config module doesn't exist yet)

**Step 6: Implement config**

Create `src/live_chat/config.py`:

```python
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    wake_word: str = "hey_jarvis"
    vad_silence_ms: int = 700
    active_timeout_s: int = 30
    fast_model: str = "claude-haiku-4-5-20241022"
    deep_model: str = "claude-sonnet-4-5-20250929"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    wakeword_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or Path.home() / ".live-chat" / "config.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        return cls()
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

**Step 8: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with config module"
```

---

### Task 2: Audio Input Module

**Files:**
- Create: `src/live_chat/audio/__init__.py`
- Create: `src/live_chat/audio/input.py`
- Create: `tests/test_audio_input.py`

**Step 1: Write the failing test**

Create `tests/test_audio_input.py`:

```python
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.input import AudioInput
from live_chat.config import Config


def test_audio_input_init():
    config = Config()
    audio_input = AudioInput(config)
    assert audio_input.sample_rate == 16000
    assert audio_input.channels == 1
    assert audio_input.blocksize == 1280  # 80ms at 16kHz


@patch("live_chat.audio.input.sd")
def test_audio_input_callback_puts_to_queue(mock_sd):
    config = Config()
    audio_input = AudioInput(config)
    queue = asyncio.Queue()
    audio_input.set_queue(queue)

    # Simulate a callback with audio data
    fake_audio = np.zeros((1280, 1), dtype=np.int16)
    audio_input._callback(fake_audio, 1280, None, None)

    assert not queue.empty()
    chunk = queue.get_nowait()
    assert chunk.shape == (1280,)
    assert chunk.dtype == np.int16
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_input.py -v`
Expected: FAIL

**Step 3: Implement audio input**

Create `src/live_chat/audio/__init__.py` (empty).

Create `src/live_chat/audio/input.py`:

```python
import asyncio

import numpy as np
import sounddevice as sd

from live_chat.config import Config


class AudioInput:
    def __init__(self, config: Config):
        self.sample_rate = config.sample_rate
        self.channels = 1
        self.blocksize = 1280  # 80ms at 16kHz, good for openwakeword
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._stream: sd.InputStream | None = None

    def set_queue(self, queue: asyncio.Queue[np.ndarray]):
        self._queue = queue

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio input status: {status}")
        if self._queue is not None:
            self._queue.put_nowait(indata[:, 0].copy())

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_input.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/live_chat/audio/ tests/test_audio_input.py
git commit -m "feat: audio input module with mic capture"
```

---

### Task 3: VAD Module

**Files:**
- Create: `src/live_chat/audio/vad.py`
- Create: `tests/test_vad.py`

**Step 1: Write the failing test**

Create `tests/test_vad.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.vad import VAD
from live_chat.config import Config


def test_vad_init():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load:
        mock_load.return_value = MagicMock()
        vad = VAD(Config())
        assert vad.silence_ms == 700
        mock_load.assert_called_once()


def test_vad_process_returns_none_for_silence():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load, \
         patch("live_chat.audio.vad.VADIterator") as mock_iter_cls:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_iter = MagicMock()
        mock_iter.return_value = None  # no speech detected
        mock_iter_cls.return_value = mock_iter

        vad = VAD(Config())
        chunk = np.zeros(512, dtype=np.int16)
        result = vad.process(chunk)
        assert result is None


def test_vad_process_returns_start_event():
    with patch("live_chat.audio.vad.load_silero_vad") as mock_load, \
         patch("live_chat.audio.vad.VADIterator") as mock_iter_cls:
        mock_load.return_value = MagicMock()
        mock_iter = MagicMock()
        mock_iter.return_value = {"start": 0}
        mock_iter_cls.return_value = mock_iter

        vad = VAD(Config())
        chunk = np.zeros(512, dtype=np.int16)
        result = vad.process(chunk)
        assert result == {"start": 0}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad.py -v`
Expected: FAIL

**Step 3: Implement VAD**

Create `src/live_chat/audio/vad.py`:

```python
import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from live_chat.config import Config


class VAD:
    def __init__(self, config: Config):
        self.silence_ms = config.vad_silence_ms
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            sampling_rate=config.sample_rate,
        )

    def process(self, chunk: np.ndarray) -> dict | None:
        """Process an audio chunk. Returns {'start': n}, {'end': n}, or None."""
        tensor = torch.from_numpy(chunk.astype(np.float32) / 32768.0)
        return self._iterator(tensor)

    def reset(self):
        self._iterator.reset_states()
```

**Step 4: Run tests**

Run: `pytest tests/test_vad.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/live_chat/audio/vad.py tests/test_vad.py
git commit -m "feat: VAD module wrapping silero-vad"
```

---

### Task 4: Wake Word Module

**Files:**
- Create: `src/live_chat/audio/wakeword.py`
- Create: `tests/test_wakeword.py`

**Step 1: Write the failing test**

Create `tests/test_wakeword.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.audio.wakeword import WakeWordDetector
from live_chat.config import Config


def test_wakeword_init():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()
        detector = WakeWordDetector(Config())
        mock_model_cls.assert_called_once_with(
            wakeword_models=["hey_jarvis"],
        )


def test_wakeword_detect_returns_false_for_silence():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.0}
        mock_model_cls.return_value = mock_model

        detector = WakeWordDetector(Config())
        chunk = np.zeros(1280, dtype=np.int16)
        assert detector.detect(chunk) is False


def test_wakeword_detect_returns_true_above_threshold():
    with patch("live_chat.audio.wakeword.Model") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.85}
        mock_model_cls.return_value = mock_model

        detector = WakeWordDetector(Config())
        chunk = np.zeros(1280, dtype=np.int16)
        assert detector.detect(chunk) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wakeword.py -v`
Expected: FAIL

**Step 3: Implement wake word detector**

Create `src/live_chat/audio/wakeword.py`:

```python
import numpy as np
from openwakeword.model import Model

from live_chat.config import Config


class WakeWordDetector:
    def __init__(self, config: Config):
        self._wake_word = config.wake_word
        self._threshold = config.wakeword_threshold
        self._model = Model(wakeword_models=[self._wake_word])

    def detect(self, chunk: np.ndarray) -> bool:
        """Returns True if the wake word is detected in this chunk."""
        prediction = self._model.predict(chunk)
        score = prediction.get(self._wake_word, 0.0)
        return score > self._threshold
```

**Step 4: Run tests**

Run: `pytest tests/test_wakeword.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/live_chat/audio/wakeword.py tests/test_wakeword.py
git commit -m "feat: wake word detection module"
```

---

### Task 5: STT Module

**Files:**
- Create: `src/live_chat/stt/__init__.py`
- Create: `src/live_chat/stt/whisper.py`
- Create: `tests/test_stt.py`

**Step 1: Write the failing test**

Create `tests/test_stt.py`:

```python
import numpy as np
from unittest.mock import patch

from live_chat.stt.whisper import WhisperSTT
from live_chat.config import Config


def test_whisper_transcribe():
    with patch("live_chat.stt.whisper.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {
            "text": " Hello world",
            "segments": [],
        }

        stt = WhisperSTT(Config())
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        result = stt.transcribe(audio)

        assert result == "Hello world"
        mock_whisper.transcribe.assert_called_once()


def test_whisper_transcribe_empty_audio():
    with patch("live_chat.stt.whisper.mlx_whisper") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": "", "segments": []}

        stt = WhisperSTT(Config())
        audio = np.zeros(800, dtype=np.float32)  # very short
        result = stt.transcribe(audio)

        assert result == ""
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt.py -v`
Expected: FAIL

**Step 3: Implement STT**

Create `src/live_chat/stt/__init__.py` (empty).

Create `src/live_chat/stt/whisper.py`:

```python
import numpy as np
import mlx_whisper

from live_chat.config import Config

_DEFAULT_MODEL = "mlx-community/whisper-small"


class WhisperSTT:
    def __init__(self, config: Config):
        self._model_path = _DEFAULT_MODEL

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz audio to text."""
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
        )
        return result["text"].strip()
```

**Step 4: Run tests**

Run: `pytest tests/test_stt.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/live_chat/stt/ tests/test_stt.py
git commit -m "feat: Whisper STT module via mlx-whisper"
```

---

### Task 6: Conversation History & LLM Client

**Files:**
- Create: `src/live_chat/llm/__init__.py`
- Create: `src/live_chat/llm/conversation.py`
- Create: `src/live_chat/llm/client.py`
- Create: `tests/test_conversation.py`
- Create: `tests/test_llm_client.py`

**Step 1: Write the failing test for conversation**

Create `tests/test_conversation.py`:

```python
from live_chat.llm.conversation import Conversation


def test_conversation_starts_empty():
    conv = Conversation()
    assert conv.messages == []


def test_conversation_add_user_message():
    conv = Conversation()
    conv.add_user("Hello")
    assert conv.messages == [{"role": "user", "content": "Hello"}]


def test_conversation_add_assistant_message():
    conv = Conversation()
    conv.add_user("Hi")
    conv.add_assistant("Hello!")
    assert len(conv.messages) == 2
    assert conv.messages[1] == {"role": "assistant", "content": "Hello!"}


def test_conversation_for_api_includes_system():
    conv = Conversation()
    conv.add_user("Hi")
    system, messages = conv.for_api()
    assert "spoken" in system.lower() or "concise" in system.lower()
    assert messages == [{"role": "user", "content": "Hi"}]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation.py -v`
Expected: FAIL

**Step 3: Implement conversation**

Create `src/live_chat/llm/__init__.py` (empty).

Create `src/live_chat/llm/conversation.py`:

```python
SYSTEM_PROMPT = """\
You are a voice-first conversational partner for intelligent idea discussion. \
Respond in natural spoken language — concise, clear, no markdown, no bullet lists, no code blocks. \
For simple exchanges, keep it to 1-2 sentences. \
For deeper topics, you may use a few paragraphs but stay conversational. \
Build on the user's ideas rather than just answering. Ask clarifying questions when needed.\
"""


class Conversation:
    def __init__(self):
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def for_api(self) -> tuple[str, list[dict[str, str]]]:
        """Returns (system_prompt, messages) for the Anthropic API."""
        return SYSTEM_PROMPT, list(self.messages)
```

**Step 4: Run tests**

Run: `pytest tests/test_conversation.py -v`
Expected: 4 passed

**Step 5: Write the failing test for LLM client**

Create `tests/test_llm_client.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.llm.client import LLMClient
from live_chat.config import Config


@pytest.mark.asyncio
async def test_llm_client_stream():
    config = Config()

    with patch("live_chat.llm.client.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock the streaming context manager
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def fake_text_stream():
            for word in ["Hello", " there", "!"]:
                yield word

        mock_stream.text_stream = fake_text_stream()
        mock_stream.get_final_message = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="Hello there!")]
        ))

        mock_client.messages.stream.return_value = mock_stream

        client = LLMClient(config)
        chunks = []
        async for chunk in client.stream("Tell me something", "Hello", []):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello there!"
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL

**Step 7: Implement LLM client**

Create `src/live_chat/llm/client.py`:

```python
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from live_chat.config import Config


class LLMClient:
    def __init__(self, config: Config):
        self._client = AsyncAnthropic()
        self._config = config

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Stream text chunks from the LLM."""
        async with self._client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

**Step 8: Run tests**

Run: `pytest tests/test_llm_client.py -v`
Expected: 1 passed

**Step 9: Commit**

```bash
git add src/live_chat/llm/ tests/test_conversation.py tests/test_llm_client.py
git commit -m "feat: conversation history and streaming LLM client"
```

---

### Task 7: Router

**Files:**
- Create: `src/live_chat/llm/router.py`
- Create: `tests/test_router.py`

**Step 1: Write the failing test**

Create `tests/test_router.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.llm.router import Router
from live_chat.config import Config


@pytest.mark.asyncio
async def test_router_classifies_fast():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="FAST")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        result = await router.classify("Hey, how's it going?", [])

        assert result == "fast"


@pytest.mark.asyncio
async def test_router_classifies_deep():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="DEEP")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        result = await router.classify(
            "Compare the trade-offs between event sourcing and CQRS", []
        )

        assert result == "deep"


@pytest.mark.asyncio
async def test_router_returns_model_name():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="FAST")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        model = await router.route("Hi there", [])

        assert model == config.fast_model
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL

**Step 3: Implement router**

Create `src/live_chat/llm/router.py`:

```python
from anthropic import AsyncAnthropic

from live_chat.config import Config

_ROUTER_PROMPT = """\
Classify the user's message as FAST or DEEP.

FAST: casual conversation, greetings, acknowledgments, simple factual questions, short clarifications, follow-ups that don't need analysis.
DEEP: multi-step reasoning, analysis, comparisons, creative brainstorming, nuanced topics, complex questions, anything requiring sustained thought.

Consider the conversation history — a simple follow-up to a deep topic should stay DEEP.

Respond with exactly one word: FAST or DEEP\
"""


class Router:
    def __init__(self, config: Config):
        self._client = AsyncAnthropic()
        self._config = config

    async def classify(
        self, text: str, history: list[dict[str, str]]
    ) -> str:
        """Classify a message as 'fast' or 'deep'."""
        context = history[-6:] if history else []  # last 3 exchanges
        messages = [*context, {"role": "user", "content": text}]

        response = await self._client.messages.create(
            model=self._config.fast_model,
            max_tokens=4,
            system=_ROUTER_PROMPT,
            messages=messages,
        )

        result = response.content[0].text.strip().upper()
        return "deep" if "DEEP" in result else "fast"

    async def route(
        self, text: str, history: list[dict[str, str]]
    ) -> str:
        """Returns the model ID to use for this message."""
        classification = await self.classify(text, history)
        if classification == "deep":
            return self._config.deep_model
        return self._config.fast_model
```

**Step 4: Run tests**

Run: `pytest tests/test_router.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/live_chat/llm/router.py tests/test_router.py
git commit -m "feat: Haiku-powered query router (fast/deep)"
```

---

### Task 8: TTS Module

**Files:**
- Create: `src/live_chat/tts/__init__.py`
- Create: `src/live_chat/tts/kokoro.py`
- Create: `tests/test_tts.py`

**Step 1: Write the failing test**

Create `tests/test_tts.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock

from live_chat.tts.kokoro import KokoroTTS
from live_chat.config import Config


def test_tts_synthesize():
    with patch("live_chat.tts.kokoro.KPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        fake_audio = np.random.randn(24000).astype(np.float32)
        mock_pipeline.return_value = iter([
            ("Hello.", "h@loU", fake_audio),
        ])
        mock_pipeline_cls.return_value = mock_pipeline

        tts = KokoroTTS(Config())
        chunks = list(tts.synthesize("Hello."))

        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32
        assert len(chunks[0]) == 24000


def test_tts_synthesize_multiple_sentences():
    with patch("live_chat.tts.kokoro.KPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        audio1 = np.random.randn(24000).astype(np.float32)
        audio2 = np.random.randn(12000).astype(np.float32)
        mock_pipeline.return_value = iter([
            ("Hello.", "h@loU", audio1),
            ("World.", "w3:ld", audio2),
        ])
        mock_pipeline_cls.return_value = mock_pipeline

        tts = KokoroTTS(Config())
        chunks = list(tts.synthesize("Hello. World."))

        assert len(chunks) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL

**Step 3: Implement TTS**

Create `src/live_chat/tts/__init__.py` (empty).

Create `src/live_chat/tts/kokoro.py`:

```python
from collections.abc import Iterator

import numpy as np
from kokoro import KPipeline

from live_chat.config import Config


class KokoroTTS:
    def __init__(self, config: Config):
        self._voice = config.tts_voice
        self._speed = config.tts_speed
        self._pipeline = KPipeline(lang_code="a")  # American English

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks (float32, 24kHz) per sentence."""
        for _gs, _ps, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            yield audio
```

**Step 4: Run tests**

Run: `pytest tests/test_tts.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/live_chat/tts/ tests/test_tts.py
git commit -m "feat: Kokoro TTS module for local speech synthesis"
```

---

### Task 9: Audio Output Module

**Files:**
- Create: `src/live_chat/audio/output.py`
- Create: `tests/test_audio_output.py`

**Step 1: Write the failing test**

Create `tests/test_audio_output.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock, call

from live_chat.audio.output import AudioOutput
from live_chat.config import Config


@patch("live_chat.audio.output.sd")
def test_audio_output_play(mock_sd):
    output = AudioOutput(Config())
    audio = np.random.randn(24000).astype(np.float32)

    output.play(audio)

    mock_sd.play.assert_called_once()
    args = mock_sd.play.call_args
    assert args[1]["samplerate"] == 24000


@patch("live_chat.audio.output.sd")
def test_audio_output_stop(mock_sd):
    output = AudioOutput(Config())
    output.stop()
    mock_sd.stop.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_output.py -v`
Expected: FAIL

**Step 3: Implement audio output**

Create `src/live_chat/audio/output.py`:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_audio_output.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/live_chat/audio/output.py tests/test_audio_output.py
git commit -m "feat: audio output module for speaker playback"
```

---

### Task 10: Pipeline — Wiring Everything Together

**Files:**
- Create: `src/live_chat/pipeline.py`
- Create: `tests/test_pipeline.py`

This is the core orchestrator. It wires all components and manages state transitions.

**Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


def test_pipeline_initial_state():
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WakeWordDetector"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.KokoroTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation"):
        pipeline = Pipeline(Config())
        assert pipeline.state == State.WAITING_FOR_WAKE_WORD


def test_pipeline_state_enum():
    assert State.WAITING_FOR_WAKE_WORD.value == "waiting"
    assert State.LISTENING.value == "listening"
    assert State.THINKING.value == "thinking"
    assert State.SPEAKING.value == "speaking"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

**Step 3: Implement pipeline**

Create `src/live_chat/pipeline.py`:

```python
import asyncio
import enum
import time

import numpy as np

from live_chat.audio.input import AudioInput
from live_chat.audio.output import AudioOutput
from live_chat.audio.vad import VAD
from live_chat.audio.wakeword import WakeWordDetector
from live_chat.config import Config
from live_chat.llm.client import LLMClient
from live_chat.llm.conversation import Conversation
from live_chat.llm.router import Router
from live_chat.stt.whisper import WhisperSTT
from live_chat.tts.kokoro import KokoroTTS


class State(enum.Enum):
    WAITING_FOR_WAKE_WORD = "waiting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.state = State.WAITING_FOR_WAKE_WORD
        self._on_state_change: callable = None

        # Components
        self._audio_in = AudioInput(config)
        self._audio_out = AudioOutput(config)
        self._vad = VAD(config)
        self._wakeword = WakeWordDetector(config)
        self._stt = WhisperSTT(config)
        self._tts = KokoroTTS(config)
        self._llm = LLMClient(config)
        self._router = Router(config)
        self._conversation = Conversation()

        # Audio queue
        self._audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._audio_in.set_queue(self._audio_queue)

        # Speech buffer for accumulating audio during listening
        self._speech_buffer: list[np.ndarray] = []
        self._last_speech_time: float = 0
        self._interrupted = False

    def on_state_change(self, callback: callable):
        self._on_state_change = callback

    def _set_state(self, state: State):
        self.state = state
        if self._on_state_change:
            self._on_state_change(state)

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
        if self.state == State.WAITING_FOR_WAKE_WORD:
            if self._wakeword.detect(chunk):
                self._set_state(State.LISTENING)
                self._speech_buffer.clear()
                self._last_speech_time = time.monotonic()
                self._vad.reset()

        elif self.state == State.LISTENING:
            event = self._vad.process(chunk)

            if event and "start" in event:
                self._last_speech_time = time.monotonic()
                self._speech_buffer.append(chunk)
            elif event and "end" in event:
                self._speech_buffer.append(chunk)
                await self._process_speech()
            elif self._speech_buffer:
                # Mid-speech, keep buffering
                self._speech_buffer.append(chunk)
                self._last_speech_time = time.monotonic()
            else:
                # Silence, check timeout
                elapsed = time.monotonic() - self._last_speech_time
                if elapsed > self.config.active_timeout_s:
                    self._set_state(State.WAITING_FOR_WAKE_WORD)

        elif self.state == State.SPEAKING:
            # Check for interruption via VAD
            event = self._vad.process(chunk)
            if event and "start" in event:
                self._interrupted = True
                self._audio_out.stop()
                self._speech_buffer = [chunk]
                self._set_state(State.LISTENING)

    async def _process_speech(self):
        """Transcribe buffered speech, route, call LLM, speak response."""
        if not self._speech_buffer:
            return

        # Concatenate and convert to float32
        audio = np.concatenate(self._speech_buffer)
        audio_f32 = audio.astype(np.float32) / 32768.0
        self._speech_buffer.clear()

        # STT
        text = self._stt.transcribe(audio_f32)
        if not text:
            self._last_speech_time = time.monotonic()
            return

        self._set_state(State.THINKING)
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

            # Check for sentence boundary
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

        self._conversation.add_assistant("".join(full_response))
        self._set_state(State.LISTENING)
        self._last_speech_time = time.monotonic()

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence."""
        for audio_chunk in self._tts.synthesize(sentence):
            self._audio_out.play(audio_chunk)
            self._audio_out.wait()
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/live_chat/pipeline.py tests/test_pipeline.py
git commit -m "feat: async pipeline orchestrating all components"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `src/live_chat/main.py`

**Step 1: Implement main.py**

Create `src/live_chat/main.py`:

```python
import asyncio
import signal
import sys

from rich.console import Console
from rich.live import Live
from rich.text import Text

from live_chat.config import Config
from live_chat.pipeline import Pipeline, State

console = Console()

_STATE_DISPLAY = {
    State.WAITING_FOR_WAKE_WORD: ("[dim]Waiting for wake word...[/dim]", "dots"),
    State.LISTENING: ("[bold green]Listening...[/bold green]", "dots"),
    State.THINKING: ("[bold yellow]Thinking...[/bold yellow]", "dots"),
    State.SPEAKING: ("[bold blue]Speaking...[/bold blue]", "dots"),
}


async def run():
    config = Config.load()
    pipeline = Pipeline(config)

    console.print(f"[bold]Live Chat[/bold] — voice-first agent")
    console.print(f"Wake word: [cyan]{config.wake_word}[/cyan]")
    console.print(f"Fast model: [cyan]{config.fast_model}[/cyan]")
    console.print(f"Deep model: [cyan]{config.deep_model}[/cyan]")
    console.print(f"Press [bold]Ctrl+C[/bold] to quit.\n")

    def on_state_change(state: State):
        label, _ = _STATE_DISPLAY[state]
        console.print(f"  {label}")

    pipeline.on_state_change(on_state_change)

    try:
        await pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

**Step 2: Verify it's importable**

Run: `python -c "from live_chat.main import main; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/live_chat/main.py
git commit -m "feat: CLI entry point with rich status display"
```

---

### Task 12: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

This test verifies the pipeline can be constructed and the state machine transitions work, using mocks for all hardware/API dependencies.

Create `tests/test_integration.py`:

```python
import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.pipeline import Pipeline, State
from live_chat.config import Config


@pytest.mark.asyncio
async def test_full_pipeline_wake_to_response():
    """Simulate: wake word → speech → STT → route → LLM → TTS → done."""
    config = Config()

    with patch("live_chat.pipeline.AudioInput") as mock_ai, \
         patch("live_chat.pipeline.AudioOutput") as mock_ao, \
         patch("live_chat.pipeline.VAD") as mock_vad_cls, \
         patch("live_chat.pipeline.WakeWordDetector") as mock_ww_cls, \
         patch("live_chat.pipeline.WhisperSTT") as mock_stt_cls, \
         patch("live_chat.pipeline.KokoroTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        # Setup mocks
        mock_ww = mock_ww_cls.return_value
        mock_vad = mock_vad_cls.return_value
        mock_stt = mock_stt_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_conv = mock_conv_cls.return_value

        pipeline = Pipeline(config)

        # 1. Wake word detected
        mock_ww.detect.return_value = True
        chunk = np.zeros(1280, dtype=np.int16)
        await pipeline._process_chunk(chunk)
        assert pipeline.state == State.LISTENING

        # 2. Speech starts
        mock_ww.detect.return_value = False
        mock_vad.process.return_value = {"start": 0}
        await pipeline._process_chunk(chunk)

        # 3. Speech ends
        mock_vad.process.return_value = {"end": 512}
        mock_stt.transcribe.return_value = "What is consciousness?"
        mock_router.route = AsyncMock(return_value=config.deep_model)
        mock_conv.for_api.return_value = ("system", [{"role": "user", "content": "What is consciousness?"}])
        mock_conv.messages = [{"role": "user", "content": "What is consciousness?"}]

        async def fake_stream(model, system, messages):
            for token in ["That's ", "a deep ", "question."]:
                yield token

        mock_llm.stream = fake_stream
        mock_tts.synthesize.return_value = iter([
            np.zeros(24000, dtype=np.float32)
        ])

        await pipeline._process_chunk(chunk)

        # Verify the pipeline processed through all states
        mock_stt.transcribe.assert_called_once()
        mock_router.route.assert_called_once()
        mock_conv.add_user.assert_called_with("What is consciousness?")
        mock_conv.add_assistant.assert_called_once()
```

**Step 2: Run test**

Run: `pytest tests/test_integration.py -v`
Expected: 1 passed

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for full pipeline flow"
```

---

### Task 13: Documentation & First Run

**Files:**
- Modify: `docs/plans/2026-02-22-live-chat-agent-design.md` (add setup instructions section)

**Step 1: Add setup instructions to design doc**

Append to the design doc:

```markdown
## Setup

```bash
# Clone and install
cd live-chat
pip install -e ".[dev]"

# Install system dependency for TTS
brew install espeak-ng

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Download wake word models (one-time)
python -c "import openwakeword; openwakeword.utils.download_models()"

# Run
live-chat
```
```

**Step 2: Verify the CLI entry point is registered**

Run: `pip install -e ".[dev]" && live-chat --help || python -m live_chat.main`

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: add setup instructions"
```

---

## Summary

| Task | Component | Est. Steps |
|------|-----------|------------|
| 1 | Project scaffolding + config | 8 |
| 2 | Audio input (mic capture) | 5 |
| 3 | VAD (silero-vad) | 5 |
| 4 | Wake word (openwakeword) | 5 |
| 5 | STT (mlx-whisper) | 5 |
| 6 | Conversation + LLM client | 9 |
| 7 | Router (fast/deep) | 5 |
| 8 | TTS (Kokoro) | 5 |
| 9 | Audio output | 5 |
| 10 | Pipeline orchestrator | 5 |
| 11 | CLI entry point | 3 |
| 12 | Integration test | 4 |
| 13 | Docs & first run | 3 |

**Total: 13 tasks, ~67 steps**

Dependencies: Tasks 1 must be first. Tasks 2-9 can be done in any order. Task 10 depends on 2-9. Task 11 depends on 10. Task 12 depends on 10. Task 13 is last.
