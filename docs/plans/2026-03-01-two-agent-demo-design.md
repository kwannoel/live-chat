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

## Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persona` | `str \| None` | `None` | Custom system prompt replacing the hardcoded default |
| `auto_speak` | `bool` | `False` | Generate and speak an opening line on startup |
| `backend` | `str` | `"api"` | LLM backend: `"api"` (Anthropic API) or `"cli"` (Claude CLI subscription) |
| `cli_path` | `str` | `"claude"` | Path to the Claude CLI binary |
| `min_silence_ms` | `int` | `600` | Milliseconds of silence before VAD fires end-of-speech |

### Example: alice.yaml (passive, CLI backend)

```yaml
tts_voice: en_US-lessac-medium
backend: cli
persona: |
  You are Alice, a curious and thoughtful conversationalist.
  Respond in natural spoken language — concise, clear, no markdown.
  Keep responses to 1-3 sentences. Be warm and engaging.
```

### Example: bob.yaml (active, CLI backend)

```yaml
tts_voice: en_US-joe-medium
auto_speak: true
backend: cli
persona: |
  You are Bob, a witty and opinionated conversationalist.
  Respond in natural spoken language — concise, clear, no markdown.
  Keep responses to 1-3 sentences. Be direct and playful.
```

## CLI Backend

Setting `backend: cli` uses the `claude` CLI (Claude Code) instead of the Anthropic API. This uses the user's Claude subscription at no additional cost.

- Spawns `claude` as a subprocess with `--output-format stream-json`
- Parses NDJSON stream events for text tokens
- Pre-spawns the next process after each response to hide startup latency
- Skips the router (no fast/deep model classification)
- Requires `claude` CLI installed and authenticated

## Code Changes

Files modified/added:

1. **`config.py`** — Config fields for persona, auto_speak, backend, cli_path, min_silence_ms.
2. **`conversation.py`** — Accept optional `persona` in `__init__`.
3. **`pipeline.py`** — Backend selection (CLIClient vs LLMClient), concurrent sentence queue for streaming+speaking, clause-boundary TTS breaks, configurable VAD threshold.
4. **`main.py`** — `--config` CLI flag, CLI backend validation, API key check only for API backend.
5. **`llm/cli_client.py`** (new) — Claude CLI subprocess client with pre-spawning.

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

## Latency Optimizations

Three optimizations reduce turn-taking latency:

1. **Clause-boundary TTS** — speaks at commas, colons, semicolons, and em-dashes (when 4+ words buffered) instead of waiting for sentence-ending punctuation. Gets first audio out faster.
2. **Concurrent streaming and speaking** — a producer task reads LLM tokens and queues sentences while a consumer task speaks them. Eliminates dead air between sentences.
3. **Pre-spawned CLI process** — after each response, the next `claude` subprocess is spawned immediately. Hides ~200ms startup overhead on subsequent turns.

## Running the Demo

### Prerequisites

1. Install the project: `pip install -e .`
2. For API backend: set `ANTHROPIC_API_KEY` in `.env` or environment
3. For CLI backend: install and authenticate the `claude` CLI
4. Ensure two Piper voices are installed in `~/.local/share/piper/voices/`:
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
backend: cli                    # Use Claude CLI (free) instead of API
min_silence_ms: 500             # Faster turn-taking (default 600)
persona: |                      # Custom system prompt
  You are a pirate captain.
  Speak in pirate dialect. Keep it short.
```
