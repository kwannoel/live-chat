# Two-Agent Demo Design

## Goal

Enable a demo where two separate instances of live-chat converse autonomously through real audio hardware (speaker → mic). No human in the loop during the conversation.

## Setup

Two terminal windows, same machine:

```
# Terminal 1 — passive agent (starts first, listens)
live-chat --config alice.yaml

# Terminal 2 — active agent (starts second, speaks first)
live-chat --config bob.yaml
```

Audio travels physically: Instance B TTS → speaker → mic → Instance A STT, and vice versa. Both agents share the same speaker output and take turns. Run until Ctrl+C.

## Config Changes

Two new fields in `Config`:

- `persona` (`str | None`, default `None`): Custom system prompt replacing the hardcoded default. Defines agent name and personality.
- `auto_speak` (`bool`, default `False`): When true, the agent generates and speaks an opening line on startup before entering the listen loop.

### Example: alice.yaml (passive)

```yaml
tts_voice: en_US-lessac-medium
persona: |
  You are Alice, a curious and thoughtful conversationalist.
  Respond in natural spoken language — concise, clear, no markdown.
  Keep responses to 1-3 sentences. Be warm and engaging.
```

### Example: bob.yaml (active)

```yaml
tts_voice: en_US-joe-medium
auto_speak: true
persona: |
  You are Bob, a witty and opinionated conversationalist.
  Respond in natural spoken language — concise, clear, no markdown.
  Keep responses to 1-3 sentences. Be direct and playful.
```

## Code Changes

Four files modified, no new files:

1. **`config.py`** — Add `persona: str | None = None` and `auto_speak: bool = False`.
2. **`conversation.py`** — Accept optional `persona` in `__init__`. Use it instead of hardcoded `SYSTEM_PROMPT` when provided.
3. **`pipeline.py`** — Pass `config.persona` to `Conversation`. Add `_auto_speak()` method called after `activate()` when `config.auto_speak` is true.
4. **`main.py`** — Add `--config` CLI argument, pass path to `Config.load()`.

## Auto-Speak Flow

```
activate()
  → _set_state(THINKING)
  → Call LLM with system prompt + synthetic user message:
    "Start a conversation. Introduce yourself briefly and
     bring up something interesting to talk about."
  → _set_state(SPEAKING)
  → Mute mic
  → Stream response, speak sentence-by-sentence (existing logic)
  → Add assistant response to conversation history (not the synthetic user message)
  → Unmute mic, drain queue
  → _set_state(LISTENING)
  → Normal loop begins
```

## Second Voice

Download a second Piper voice (e.g., `en_US-joe-medium`) to `~/.local/share/piper/voices/` so the two agents sound distinct.

## Running the Demo

### Prerequisites

1. Install the project: `pip install -e .`
2. Set `ANTHROPIC_API_KEY` in `.env` or environment
3. Ensure two Piper voices are installed in `~/.local/share/piper/voices/`:
   - `en_US-lessac-medium` (Alice — included by default)
   - `en_US-joe-medium` (Bob — download from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)):
     ```
     cd ~/.local/share/piper/voices/
     curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx
     curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json
     ```

### Steps

Open two terminal windows on the same machine.

**Terminal 1** — start Alice (passive, listens):
```
live-chat --config demo/alice.yaml
```
Wait for the "Ready!" message.

**Terminal 2** — start Bob (active, speaks first):
```
live-chat --config demo/bob.yaml
```
Bob auto-generates an opening line and speaks it. Alice hears it through the mic, responds, and the conversation continues autonomously.

Press **Ctrl+C** in either terminal to stop.

### Customization

Create your own YAML config with any combination of:

```yaml
tts_voice: en_US-joe-medium    # Piper voice name
auto_speak: true                # Speak first on startup
persona: |                      # Custom system prompt
  You are a pirate captain.
  Speak in pirate dialect. Keep it short.
```

## What Stays the Same

The existing audio pipeline, VAD, STT, TTS engine, LLM client, router, and state machine are all unchanged. The demo is just a configuration-level feature on top of the existing architecture.
