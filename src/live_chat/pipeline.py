import asyncio
import enum
import time

import numpy as np

from live_chat.audio.aec import EchoCanceller
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
        self._aec = EchoCanceller()
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
        self._tts_end_time = 0.0

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

    # Raw int16 RMS threshold to distinguish real speech from speaker echo.
    # Normal speech at laptop mic: ~3000-10000 RMS. Speaker echo: ~500-2000.
    _BARGE_IN_RMS = 3000

    async def _process_chunk(self, chunk: np.ndarray):
        # Post-TTS echo gate: ignore low-energy chunks shortly after speaking.
        # Check raw mic signal before AEC/AutoGain to avoid amplified echo.
        if self.state == State.LISTENING and self._tts_end_time > 0:
            elapsed = time.monotonic() - self._tts_end_time
            if elapsed < 1.5:
                raw_rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
                if raw_rms < self._BARGE_IN_RMS:
                    return
            else:
                self._tts_end_time = 0

        # Apply echo cancellation (subtracts TTS playback from mic signal)
        chunk = self._aec.cancel(chunk)
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
                self._aec.clear()
                self._speech_buffer = [normalized]
                self._set_state(State.LISTENING)

    # Minimum speech duration to avoid noise/echo triggering STT (0.5s at 16kHz)
    _MIN_SPEECH_SAMPLES = 8000

    async def _drain_queue(self):
        """Discard stale audio in queue (echo from TTS playback)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Brief cooldown: let residual echo decay, then drain again
        await asyncio.sleep(0.3)
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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

        was_interrupted = False
        try:
            # Route
            model = await self._router.route(text, self._conversation.messages[:-1])
            system, messages = self._conversation.for_api()

            # Stream LLM response, buffer sentences, speak as they complete
            self._set_state(State.SPEAKING)
            full_response = []
            spoken_sentences = []
            sentence_buffer = []

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

        if not was_interrupted:
            # Normal completion: drain echo chunks + set echo gate timer
            await self._drain_queue()
            self._tts_end_time = time.monotonic()

        self._set_state(State.LISTENING)
        self._vad.reset()

    async def _speak_sentence(self, sentence: str):
        """Synthesize and play a single sentence.

        Consumes mic chunks during playback to prevent queue buildup.
        Detects barge-in via raw mic energy (not VAD — VAD can't
        distinguish echo from speech reliably without hardware AEC).
        """
        for audio_chunk in self._tts.synthesize(sentence):
            self._aec.set_reference(audio_chunk, self._tts.sample_rate)
            self._audio_out.play(audio_chunk, sample_rate=self._tts.sample_rate)

            # Consume mic chunks while waiting for playback to finish
            playback_done = asyncio.ensure_future(self._audio_out.wait_async())
            while not playback_done.done() and not self._interrupted:
                try:
                    mic_chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.032
                    )
                    # Barge-in: only real speech (loud) triggers interruption.
                    # Use raw int16 energy — AutoGain would amplify echo too.
                    raw_rms = float(np.sqrt(np.mean(
                        mic_chunk.astype(float) ** 2
                    )))
                    if raw_rms > self._BARGE_IN_RMS:
                        self._interrupted = True
                        self._audio_out.stop()
                        self._aec.clear()
                        self._vad.reset()
                        self._speech_buffer.clear()
                        self._set_state(State.LISTENING)
                except asyncio.TimeoutError:
                    pass

            if not playback_done.done():
                playback_done.cancel()
                try:
                    await playback_done
                except asyncio.CancelledError:
                    pass

            if self._interrupted:
                return
