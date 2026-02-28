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
        self._audio_in.set_loop(asyncio.get_event_loop())
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

    # Minimum speech duration to avoid noise triggering STT (0.3s at 16kHz)
    _MIN_SPEECH_SAMPLES = 4800

    async def _process_speech(self):
        """Transcribe buffered speech, route, call LLM, speak response."""
        if not self._speech_buffer:
            return

        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()

        # Skip very short segments (likely noise, not speech)
        if len(audio) < self._MIN_SPEECH_SAMPLES:
            return

        # Skip low-energy segments (AutoGain amplifies noise too)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.02:
            return

        # STT (audio is already float32 from AutoGain)
        text = self._stt.transcribe(audio)
        if not text:
            return

        self._set_state(State.THINKING)
        if self._on_transcript:
            self._on_transcript("user", text, None)
        self._conversation.add_user(text)

        try:
            # Route
            model = await self._router.route(text, self._conversation.messages[:-1])
            system, messages = self._conversation.for_api()

            # Stream LLM response, buffer sentences, speak as they complete
            self._set_state(State.SPEAKING)
            full_response = []
            spoken_sentences = []
            sentence_buffer = []
            was_interrupted = False

            async for token in self._llm.stream(model, system, messages):
                full_response.append(token)
                sentence_buffer.append(token)

                current = "".join(sentence_buffer)
                if any(current.rstrip().endswith(p) for p in ".!?"):
                    sentence = current.strip()
                    if sentence:
                        await self._speak_sentence(sentence)
                        spoken_sentences.append(sentence)
                    sentence_buffer.clear()

                if self._interrupted:
                    self._interrupted = False
                    was_interrupted = True
                    break

            # Speak any remaining text
            remaining = "".join(sentence_buffer).strip()
            if remaining and not was_interrupted:
                await self._speak_sentence(remaining)
                spoken_sentences.append(remaining)

            if was_interrupted:
                spoken_text = " ".join(spoken_sentences)
                response_text = f"{spoken_text} [interrupted by user]" if spoken_text else "[interrupted by user]"
            else:
                response_text = "".join(full_response)

            self._conversation.add_assistant(response_text)
            if self._on_transcript:
                self._on_transcript("assistant", response_text, model)

        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

        self._set_state(State.LISTENING)
        self._vad.reset()

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence."""
        for audio_chunk in self._tts.synthesize(sentence):
            self._audio_out.play(audio_chunk, sample_rate=self._tts.sample_rate)
            await self._audio_out.wait_async()
