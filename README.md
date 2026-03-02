# live-chat

Voice-first conversational AI agent. Listens through your mic, thinks with Claude, and speaks back using local TTS. Runs entirely from the terminal.

**[Demo recording](demo/recordings/two-agent-demo.mp4)** — two AI agents (Alice and Bob) having an autonomous conversation.

## How it works

```
Mic → VAD → Whisper STT → Claude LLM → Piper TTS → Speaker
```

- **VAD** (Silero) detects when you start and stop speaking
- **STT** (mlx-whisper) transcribes your speech locally on Apple Silicon
- **LLM** (Claude) generates a response, streamed token-by-token
- **TTS** (Piper) synthesizes speech locally, spoken as sentences complete
- A **router** classifies messages as fast/deep and picks the appropriate model (API backend only)

## Install

Requires Python 3.10+ and macOS (Apple Silicon for mlx-whisper).

```bash
git clone https://github.com/kwannoel/live-chat.git
cd live-chat
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Piper voice

Download at least one Piper voice:

```bash
mkdir -p ~/.local/share/piper/voices
cd ~/.local/share/piper/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### LLM backend

Choose one:

- **API backend** (default): create a `.env` file in the project root with your API key:
  ```
  ANTHROPIC_API_KEY='sk-ant-...'
  ```
- **CLI backend** (free with Claude subscription): install and authenticate the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code), then set `backend: cli` in your config

## Usage

```bash
source .venv/bin/activate

# Default config (API backend, reads key from .env)
live-chat

# Custom config
live-chat --config my-config.yaml
```

The agent listens immediately on startup. Speak naturally, and it responds. Press **Ctrl+C** to quit.

## Configuration

Create a YAML config file with any of these fields:

| Field | Default | Description |
|-------|---------|-------------|
| `fast_model` | `claude-haiku-4-5-20251001` | Model for simple messages (API backend) |
| `deep_model` | `claude-sonnet-4-6` | Model for complex messages (API backend) |
| `tts_voice` | `en_US-lessac-medium` | Piper voice name |
| `persona` | *(system default)* | Custom system prompt |
| `auto_speak` | `false` | Speak an opening line on startup |
| `backend` | `api` | `api` or `cli` |
| `cli_path` | `claude` | Path to Claude CLI binary |
| `min_silence_ms` | `600` | Silence before end-of-speech detection (ms) |
| `sample_rate` | `16000` | Mic sample rate (Hz) |

## Two-agent demo

Two instances converse autonomously through real audio hardware (speaker to mic):

**Terminal 1** — Alice (listens first):
```bash
live-chat --config demo/alice.yaml
```

**Terminal 2** — Bob (speaks first):
```bash
live-chat --config demo/bob.yaml
```

Bob auto-generates an opening line. Alice hears it, responds, and the conversation continues. For a second voice, download `en_US-joe-medium`:

```bash
cd ~/.local/share/piper/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json
```

## Architecture

```
src/live_chat/
  main.py              # CLI entrypoint
  config.py            # YAML config loading
  pipeline.py          # Audio loop, state machine, stream+speak
  audio/
    input.py           # Mic capture (sounddevice)
    output.py          # Speaker playback
    vad.py             # Voice activity detection (Silero)
    gain.py            # Auto-gain normalization
  llm/
    client.py          # Anthropic API client
    cli_client.py      # Claude CLI client (subprocess)
    router.py          # Fast/deep message classifier
    conversation.py    # Message history + system prompt
  stt/
    whisper.py         # mlx-whisper transcription
  tts/
    piper_tts.py       # Piper TTS synthesis
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
