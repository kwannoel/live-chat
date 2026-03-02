import asyncio
import enum

import numpy as np

from live_chat.audio.gain import AutoGain
from live_chat.audio.input import AudioInput
from live_chat.audio.output import AudioOutput
from live_chat.audio.vad import VAD
from live_chat.config import Config
from live_chat.llm.cli_client import CLIClient
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
        self._vad = VAD(min_silence_ms=config.min_silence_ms)
        self._stt = WhisperSTT(config)
        self._tts = PiperTTS(config)
        if config.backend == "cli":
            self._llm = CLIClient(config)
            self._router = None
        else:
            self._llm = LLMClient(config)
            self._router = Router(config)
        self._conversation = Conversation(persona=config.persona)

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
            if self.config.auto_speak:
                asyncio.ensure_future(self._auto_speak())

    async def run(self):
        """Main loop — process audio chunks from the mic."""
        # Pre-spawn CLI process so first LLM call is faster
        if hasattr(self._llm, "warm_up"):
            system, _ = self._conversation.for_api()
            await self._llm.warm_up(system)

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

    # Minimum speech duration to avoid noise/echo triggering STT (0.5s at 16kHz)
    _MIN_SPEECH_SAMPLES = 8000

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
            if self._router is not None:
                model = await self._router.route(text, self._conversation.messages[:-1])
            else:
                model = "cli"
            system, messages = self._conversation.for_api()

            # Stream LLM response and speak concurrently via sentence queue
            self._set_state(State.SPEAKING)
            self._audio_in.mute()

            response_text = await self._stream_and_speak(model, system, messages)
            self._conversation.add_assistant(response_text)
            if self._on_transcript:
                self._on_transcript("assistant", response_text, model)

        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

        # Unmute mic, drain any stale chunks, resume listening
        self._audio_in.unmute()
        await asyncio.sleep(0.3)
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._set_state(State.LISTENING)
        self._vad.reset()

    async def _auto_speak(self):
        """Generate and speak an opening line (used when auto_speak=True)."""
        self._set_state(State.THINKING)

        opener_prompt = (
            "Start a conversation. Introduce yourself briefly and "
            "bring up something interesting to talk about."
        )
        self._conversation.add_user(opener_prompt)
        if self._router is not None:
            model = await self._router.route(opener_prompt, [])
        else:
            model = "cli"
        system, messages = self._conversation.for_api()

        self._set_state(State.SPEAKING)
        self._audio_in.mute()

        response_text = await self._stream_and_speak(model, system, messages)

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

    async def _stream_and_speak(
        self, model: str, system: str, messages: list[dict]
    ) -> str:
        """Stream LLM tokens and speak sentences concurrently.

        A producer task reads from the LLM and pushes complete sentences
        into a queue. A consumer task speaks them. This eliminates dead
        air between sentences caused by waiting for the LLM.
        """
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_response: list[str] = []

        async def _produce():
            sentence_buffer: list[str] = []
            async for token in self._llm.stream(model, system, messages):
                full_response.append(token)
                sentence_buffer.append(token)

                current = "".join(sentence_buffer)
                stripped = current.rstrip()
                is_sentence_end = any(stripped.endswith(p) for p in ".!?")
                is_clause_break = (
                    len(current.split()) >= 4
                    and any(stripped.endswith(p) for p in ",;:\u2014")
                )
                if is_sentence_end or is_clause_break:
                    chunk = current.strip()
                    if chunk:
                        await sentence_queue.put(chunk)
                    sentence_buffer.clear()

                if self._interrupted:
                    self._interrupted = False
                    break

            remaining = "".join(sentence_buffer).strip()
            if remaining and not self._interrupted:
                await sentence_queue.put(remaining)
            await sentence_queue.put(None)  # sentinel

        async def _consume():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                await self._speak_sentence(sentence)

        producer = asyncio.create_task(_produce())
        consumer = asyncio.create_task(_consume())
        await asyncio.gather(producer, consumer)

        return "".join(full_response)

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence."""
        for audio_chunk in self._tts.synthesize(sentence):
            self._audio_out.play(audio_chunk, sample_rate=self._tts.sample_rate)
            await self._audio_out.wait_async()
