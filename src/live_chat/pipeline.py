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
        """Main loop -- process audio chunks from the mic."""
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
