# Two-Agent Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable two live-chat instances to converse autonomously via real audio hardware, with different personas, voices, and one agent auto-speaking on startup.

**Architecture:** Add `persona` and `auto_speak` config fields. Conversation accepts a custom system prompt. Pipeline generates and speaks an opening line when `auto_speak` is true. CLI accepts a `--config` flag.

**Tech Stack:** Python, existing pipeline (Piper TTS, mlx-whisper STT, Claude LLM), argparse for CLI.

---

### Task 1: Add `persona` and `auto_speak` to Config

**Files:**
- Modify: `src/live_chat/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
def test_default_config_new_fields():
    config = Config()
    assert config.persona is None
    assert config.auto_speak is False


def test_config_from_dict_persona():
    config = Config.from_dict({"persona": "You are Alice."})
    assert config.persona == "You are Alice."
    assert config.auto_speak is False


def test_config_from_dict_auto_speak():
    config = Config.from_dict({"auto_speak": True})
    assert config.auto_speak is True
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `Config` has no `persona` or `auto_speak` fields.

**Step 3: Implement**

In `src/live_chat/config.py`, add two fields to the `Config` dataclass:

```python
persona: str | None = None
auto_speak: bool = False
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/live_chat/config.py tests/test_config.py
git commit -m "feat: add persona and auto_speak config fields"
```

---

### Task 2: Conversation accepts custom persona

**Files:**
- Modify: `src/live_chat/llm/conversation.py`
- Modify: `tests/test_conversation.py`

**Step 1: Write failing tests**

Add to `tests/test_conversation.py`:

```python
def test_conversation_custom_persona():
    conv = Conversation(persona="You are Alice.")
    conv.add_user("Hi")
    system, messages = conv.for_api()
    assert system == "You are Alice."


def test_conversation_default_persona():
    conv = Conversation()
    conv.add_user("Hi")
    system, _ = conv.for_api()
    assert "voice-first" in system.lower() or "spoken" in system.lower()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_conversation.py -v`
Expected: FAIL — `Conversation.__init__()` got unexpected keyword argument `persona`.

**Step 3: Implement**

In `src/live_chat/llm/conversation.py`, update `Conversation.__init__`:

```python
class Conversation:
    def __init__(self, persona: str | None = None):
        self.messages: list[dict[str, str]] = []
        self._system = persona or SYSTEM_PROMPT

    # ... add_user, add_assistant unchanged ...

    def for_api(self) -> tuple[str, list[dict[str, str]]]:
        """Returns (system_prompt, messages) for the Anthropic API."""
        return self._system, list(self.messages)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_conversation.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/live_chat/llm/conversation.py tests/test_conversation.py
git commit -m "feat: conversation accepts custom persona"
```

---

### Task 3: Pipeline passes persona to Conversation and implements auto-speak

**Files:**
- Modify: `src/live_chat/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Step 1: Write failing tests**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_auto_speak_calls_llm_and_tts():
    """When auto_speak is True, pipeline should generate and speak an opening."""
    config = Config(auto_speak=True, persona="You are Bob.")
    with patch("live_chat.pipeline.AudioInput") as mock_in_cls, \
         patch("live_chat.pipeline.AudioOutput") as mock_out_cls, \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS") as mock_tts_cls, \
         patch("live_chat.pipeline.LLMClient") as mock_llm_cls, \
         patch("live_chat.pipeline.Router") as mock_router_cls, \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:

        mock_llm = mock_llm_cls.return_value
        mock_router = mock_router_cls.return_value
        mock_tts = mock_tts_cls.return_value
        mock_out = mock_out_cls.return_value
        mock_in = mock_in_cls.return_value
        mock_conv = mock_conv_cls.return_value

        # Router returns a model
        mock_router.route = AsyncMock(return_value="claude-haiku-4-5-20251001")

        # LLM streams one sentence
        async def fake_stream(model, system, messages):
            yield "Hello there!"
        mock_llm.stream = fake_stream

        # Conversation.for_api returns system + messages
        mock_conv.for_api.return_value = ("You are Bob.", [{"role": "user", "content": "Start a conversation."}])

        # TTS returns audio
        mock_tts.synthesize.return_value = iter([np.zeros(22050, dtype=np.int16)])
        mock_tts.sample_rate = 22050
        mock_out.wait_async = AsyncMock()

        pipeline = Pipeline(config)

        # Persona should be passed to Conversation
        mock_conv_cls.assert_called_once_with(persona="You are Bob.")

        await pipeline._auto_speak()

        # Should have muted mic, called LLM, spoken, added to conversation, unmuted
        mock_in.mute.assert_called()
        mock_conv.add_assistant.assert_called_once_with("Hello there!")
        mock_tts.synthesize.assert_called()
        mock_in.unmute.assert_called()


def test_pipeline_passes_persona_to_conversation():
    config = Config(persona="You are Alice.")
    with patch("live_chat.pipeline.AudioInput"), \
         patch("live_chat.pipeline.AudioOutput"), \
         patch("live_chat.pipeline.AutoGain"), \
         patch("live_chat.pipeline.VAD"), \
         patch("live_chat.pipeline.WhisperSTT"), \
         patch("live_chat.pipeline.PiperTTS"), \
         patch("live_chat.pipeline.LLMClient"), \
         patch("live_chat.pipeline.Router"), \
         patch("live_chat.pipeline.Conversation") as mock_conv_cls:
        Pipeline(config)
        mock_conv_cls.assert_called_once_with(persona="You are Alice.")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `Pipeline` doesn't pass `persona` to `Conversation`, `_auto_speak` doesn't exist.

**Step 3: Implement**

In `src/live_chat/pipeline.py`:

1. Pass persona to Conversation in `__init__`:

```python
self._conversation = Conversation(persona=config.persona)
```

2. Add `_auto_speak()` method:

```python
async def _auto_speak(self):
    """Generate and speak an opening line (used when auto_speak=True)."""
    self._set_state(State.THINKING)

    opener_prompt = (
        "Start a conversation. Introduce yourself briefly and "
        "bring up something interesting to talk about."
    )
    self._conversation.add_user(opener_prompt)
    model = await self._router.route(opener_prompt, [])
    system, messages = self._conversation.for_api()

    self._set_state(State.SPEAKING)
    self._audio_in.mute()

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

    remaining = "".join(sentence_buffer).strip()
    if remaining:
        await self._speak_sentence(remaining)

    response_text = "".join(full_response)

    # Remove the synthetic user message, keep only assistant response
    self._conversation.messages.clear()
    self._conversation.add_assistant(response_text)

    if self._on_transcript:
        self._on_transcript("assistant", response_text, model)

    self._audio_in.unmute()
    await asyncio.sleep(0.3)
    while not self._audio_queue.empty():
        try:
            self._audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    self._set_state(State.LISTENING)
    self._vad.reset()
```

3. Update `activate()` to return a coroutine flag, and in `run()` or `main.py`, call `_auto_speak()` after activation if `config.auto_speak` is true. Simplest: make `activate()` schedule the auto-speak:

```python
def activate(self):
    """Start continuous listening (called from keyboard input)."""
    if self.state == State.IDLE:
        self._set_state(State.LISTENING)
        self._speech_buffer.clear()
        self._vad.reset()
        if self.config.auto_speak:
            asyncio.ensure_future(self._auto_speak())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/live_chat/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline supports persona and auto-speak on startup"
```

---

### Task 4: CLI `--config` flag

**Files:**
- Modify: `src/live_chat/main.py`

**Step 1: Implement**

In `src/live_chat/main.py`, add argparse to `main()`:

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Live Chat — voice-first agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    args = parser.parse_args()

    try:
        asyncio.run(run(config_path=args.config))
    except KeyboardInterrupt:
        pass
```

Update `run()` to accept `config_path`:

```python
async def run(config_path: str | None = None):
    _load_dotenv()
    path = Path(config_path) if config_path else None
    config = Config.load(path)
    # ... rest unchanged ...
```

**Step 2: Test manually**

Run: `live-chat --help`
Expected: Shows `--config` option in help text.

**Step 3: Commit**

```bash
git add src/live_chat/main.py
git commit -m "feat: add --config CLI flag for custom config file"
```

---

### Task 5: Download second Piper voice

**Step 1: Download voice files**

```bash
cd ~/.local/share/piper/voices/
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx"
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json"
```

If `joe` doesn't exist, fall back to `ryan`:

```bash
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
```

**Step 2: Verify**

```bash
ls ~/.local/share/piper/voices/
```

Expected: Two sets of `.onnx` + `.onnx.json` files.

---

### Task 6: Create demo config files and test end-to-end

**Files:**
- Create: `demo/alice.yaml`
- Create: `demo/bob.yaml`

**Step 1: Create config files**

`demo/alice.yaml`:

```yaml
tts_voice: en_US-lessac-medium
persona: |
  You are Alice, a curious and thoughtful conversationalist.
  Respond in natural spoken language — concise, clear, no markdown, no bullet lists, no code blocks.
  Keep responses to 1-3 sentences. Be warm and engaging.
  Build on what the other person says. Ask follow-up questions.
```

`demo/bob.yaml` (use whichever second voice was downloaded):

```yaml
tts_voice: en_US-joe-medium
auto_speak: true
persona: |
  You are Bob, a witty and opinionated conversationalist.
  Respond in natural spoken language — concise, clear, no markdown, no bullet lists, no code blocks.
  Keep responses to 1-3 sentences. Be direct and playful.
  Build on what the other person says. Challenge ideas constructively.
```

**Step 2: Run all tests**

```bash
python -m pytest -v
```

Expected: All tests pass.

**Step 3: Manual end-to-end test**

Terminal 1:
```bash
live-chat --config demo/alice.yaml
```
Wait for "Ready!" message.

Terminal 2:
```bash
live-chat --config demo/bob.yaml
```
Bob should auto-speak an opening line. Alice should hear it via mic, respond, and the conversation should continue.

**Step 4: Commit**

```bash
git add demo/
git commit -m "feat: add demo config files for two-agent conversation"
```

---

### Task 7: Run full test suite

**Step 1: Run all tests**

```bash
python -m pytest -v
```

Expected: All tests pass, no regressions.
