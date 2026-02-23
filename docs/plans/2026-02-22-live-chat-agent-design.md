# Live Chat Agent — Design Document

**Date**: 2026-02-22
**Status**: Approved

## Summary

A Python CLI that listens for a wake word, transcribes speech locally via Whisper, auto-routes to Haiku (fast) or Opus/Sonnet (deep) for intelligent idea discussion, and speaks responses back via local TTS — all streaming end-to-end for low latency.

## Decisions

| Decision | Choice |
|----------|--------|
| Interface | Voice-first |
| Model routing | Automatic (agent classifies query) |
| Use case | Intelligent idea discussion |
| Fast model | Claude Haiku 4.5 |
| Deep models | Claude Opus 4.6 / Sonnet 4.6 (configurable) |
| Platform | CLI + voice |
| Voicebox | Inspiration only (own pipeline) |
| Conversation flow | Wake word + VAD |
| Language | Python |

## Architecture

Approach A: Streaming Pipeline — fully local voice pipeline, cloud only for LLMs.

```
Mic → Wake Word → VAD → Whisper STT → Router → LLM (Haiku/Opus) → TTS → Speaker
```

Everything streams. TTS starts generating audio as soon as the first sentence arrives from the LLM.

## Component Details

### Audio Input Pipeline

- **Wake word**: `openwakeword` running continuously on mic input. Configurable wake word. Once triggered, switches to active listening mode.
- **VAD**: `silero-vad` detects speech boundaries. Silence >700ms marks segment complete, sent to STT. Returns to wake-word mode after configurable timeout (~30s of no speech).
- **STT**: `mlx-whisper` on Apple Silicon for fast local transcription.

### Router

A Haiku call with a classification prompt (~100ms overhead). Classifies into:

- **Fast**: casual conversation, acknowledgments, simple factual questions, short clarifications
- **Deep**: multi-step reasoning, analysis, comparisons, creative brainstorming, nuanced topics

Receives conversation history so follow-ups to deep topics stay deep.

### LLM Layer

- **Fast**: Haiku 4.5 — low latency, streams tokens quickly
- **Deep**: Opus 4.6 or Sonnet 4.6 — configurable, defaults to Sonnet

Both share the same in-memory conversation history. System prompt tuned for spoken conversation — concise, natural phrasing, no markdown/code.

### TTS Output

- **Sentence-level streaming**: Buffer LLM tokens until sentence boundary (`. ! ?`), send to TTS immediately while buffering next sentence.
- **Model**: Kokoro TTS (open source, fast, good quality) or Piper as alternative. Runs locally.
- **Playback**: `sounddevice` with double-buffering — while one sentence plays, the next is being synthesized.

### Interruption Handling

If user speaks while agent is talking (detected via VAD on input stream):
1. Stop TTS playback immediately
2. Cancel in-flight LLM generation
3. Transcribe the interruption
4. Process the new input

## Conversation & State Management

- In-memory message history (user/assistant roles), shared by router and LLM
- Sessions ephemeral by default; optional save/load to disk as JSON
- System prompt tuned for spoken dialogue: concise (1-3 sentences fast, few paragraphs deep), natural language, builds on ideas rather than just answering

## CLI Interface

- Status line: `[listening]`, `[thinking...]`, `[speaking]`
- Transcript log scrolling in terminal
- Keyboard shortcuts: `q` quit, `m` mute/unmute, `d` force deep mode
- Config via `~/.live-chat/config.yaml`

## Project Structure

```
live-chat/
├── pyproject.toml
├── src/
│   └── live_chat/
│       ├── __init__.py
│       ├── main.py              # CLI entry point, async event loop
│       ├── audio/
│       │   ├── input.py         # Mic capture, audio stream
│       │   ├── output.py        # Speaker playback, double-buffering
│       │   ├── vad.py           # Silero VAD wrapper
│       │   └── wakeword.py      # openwakeword wrapper
│       ├── stt/
│       │   └── whisper.py       # MLX Whisper transcription
│       ├── tts/
│       │   └── kokoro.py        # Kokoro TTS synthesis
│       ├── llm/
│       │   ├── router.py        # Query classifier (Haiku)
│       │   ├── client.py        # Anthropic API streaming client
│       │   └── conversation.py  # History management
│       ├── pipeline.py          # Wires everything together
│       └── config.py            # YAML config loader
└── tests/
```

Components are standalone async modules. `pipeline.py` orchestrates via `asyncio` with async queues between components.

## Dependencies

| Component | Library | Why |
|-----------|---------|-----|
| Audio I/O | `sounddevice` | Low-latency, cross-platform, numpy-friendly |
| VAD | `silero-vad` | Small, accurate, runs on CPU |
| Wake word | `openwakeword` | Open source, customizable |
| STT | `mlx-whisper` | Fast on Apple Silicon via Metal |
| TTS | `kokoro` | Open source, good quality, fast locally |
| LLM | `anthropic` | Official SDK, streaming support |
| Async | `asyncio` | Standard library |
| Config | `pyyaml` | Simple config files |
| CLI display | `rich` | Status line, transcript formatting |

## Error Handling

- **Model failures**: Retry once, then fall back (Opus → Sonnet → Haiku). If all fail, speak error and wait.
- **STT confidence**: Low confidence → ask user to repeat.
- **Audio device issues**: Detect on startup, pause and notify on disconnect.
- **Wake word false positives**: No speech within 3s after trigger → silently return to wake-word mode.
- **Long responses**: System prompt guides conciseness; TTS streaming handles long output naturally.

## Out of Scope (v1)

- No GUI / web interface
- No voice cloning
- No multi-language support (English only)
- No tool use / agentic actions
- No persistent memory across sessions (save/load only)
- No multi-user / server mode

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
