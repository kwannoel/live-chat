import asyncio
import enum
import time

import numpy as np

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
        self._audio_out = AudioOutput(config)
        self._vad = VAD(config)
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
        self._last_speech_time: float = 0
        self._interrupted = False
        self._stop_event = asyncio.Event()

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
        """Toggle recording (called from keyboard input)."""
        if self.state == State.IDLE:
            self._set_state(State.LISTENING)
            self._speech_buffer.clear()
            self._last_speech_time = time.monotonic()
        elif self.state == State.LISTENING:
            # Second Enter press stops recording and triggers processing
            self._stop_event.set()

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
        if self.state == State.IDLE:
            pass  # waiting for keyboard activation

        elif self.state == State.LISTENING:
            self._speech_buffer.append(chunk)
            if self._stop_event.is_set():
                self._stop_event.clear()
                await self._process_speech()

        elif self.state == State.SPEAKING:
            pass

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

        response_text = "".join(full_response)
        self._conversation.add_assistant(response_text)
        if self._on_transcript:
            self._on_transcript("assistant", response_text, model)
        self._set_state(State.LISTENING)
        self._last_speech_time = time.monotonic()

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence."""
        for audio_chunk in self._tts.synthesize(sentence):
            self._audio_out.play(audio_chunk, sample_rate=self._tts.sample_rate)
            self._audio_out.wait()
