"""
Pipeline Coordinator

Orchestrates the complete translation pipeline:
Audio Input -> ASR -> Translation -> TTS -> Audio Output

Manages multiple language pipelines running in parallel.
"""

import os
import sys
import time
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from pathlib import Path
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio.input_stream import AudioInputStream, AudioChunk
from audio.output_stream import AudioOutputStream, SharedStereoOutput, ChannelOutputProxy
from pipeline.parakeet_asr import ParakeetASRBuffer
from pipeline.sentence_buffer import SentenceBuffer
from pipeline.translation import TranslationService, TranslationResult
from pipeline.tts import TTSService, SpeechResult
from web.bus import BUS as WEB_BUS

# A HEARTBEAT line this often, from the audio path, whether or not anyone is
# speaking. The scheduler's stall watchdog reads the log's mtime: without a
# heartbeat, a quiet room looked exactly like a hung pipeline, and 15 minutes
# of post-service silence restarted a healthy service (2026-09-06 20:30, and
# the pre-service silence that morning). With it, the watchdog measures the
# process — no chunks reaching this code means the audio thread is stuck.
HEARTBEAT_SEC = float(os.environ.get("PIPELINE_HEARTBEAT_SEC", "60"))


@dataclass
class PipelineConfig:
    """Configuration for a single language pipeline."""
    language_code: str
    language_name: str
    output_device: str = "default"
    output_channel: Optional[int] = None  # 0=left, 1=right, None=both/mono
    # A language carried only by the web page, with no room audio at all.
    # Not currently used: languages past the Behringer's two channels go out
    # the machine's own analog jack instead. Kept because a deployment without
    # a spare output needs it, and because it costs nothing to leave in.
    web_only: bool = False
    translation_model: Optional[str] = None
    tts_voice: str = "default"
    enabled: bool = True


@dataclass
class TranslationEvent:
    """Event representing a translation through the pipeline."""
    timestamp: float            # When the event was created
    source_text: str
    translated_text: str
    target_language: str
    audio_duration: float       # Duration of synthesized speech (seconds)
    total_latency: float        # End-to-end: chunk_start → playback_start
    chunk_start_time: float     # When the first audio sample of this chunk arrived
    chunk_duration: float       # Duration of the audio chunk sent to ASR
    asr_time: float             # Time spent in ASR
    translation_time: float     # Time spent in translation
    tts_time: float             # Time spent in TTS synthesis
    queue_depth: int            # Pipeline queue depth when text was submitted


class LanguagePipeline:
    """
    A single language translation pipeline.

    Handles: Translation -> TTS -> Audio Output for one target language.
    """

    def __init__(
        self,
        config: PipelineConfig,
        models_dir: Optional[str] = None,
        audio_output=None,  # Optional external audio output (for shared stereo)
    ):
        self.config = config
        self._models_dir = models_dir
        self._external_audio_output = audio_output

        # Pipeline components
        self._translator: Optional[TranslationService] = None
        self._tts: Optional[TTSService] = None
        self._audio_output = None
        self._owns_audio_output = False  # Whether we created the output ourselves

        # Processing queue
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_translation: Optional[Callable[[TranslationEvent], None]] = None

    def load(self) -> None:
        """Load all pipeline components."""
        print(f"Loading pipeline for {self.config.language_name}...")

        # Load translation model
        self._translator = TranslationService(
            source_language="en",
            target_language=self.config.language_code,
            model_name=self.config.translation_model,
            download_root=f"{self._models_dir}/translation" if self._models_dir else None,
        )
        self._translator.load()

        # Load TTS
        self._tts = TTSService(
            language=self.config.language_code,
            voice=self.config.tts_voice,
            download_root=f"{self._models_dir}/tts" if self._models_dir else None,
        )
        self._tts.load()

        # Web-only languages skip room playback entirely; their audio still
        # reaches phones through the bus.
        if self.config.web_only:
            self._audio_output = None
            self._owns_audio_output = False
            print(f"Pipeline loaded: {self.config.language_name}")
            print("  Output: web page only (no room channel available)")
            return

        # Use external audio output if provided, otherwise create our own
        if self._external_audio_output is not None:
            self._audio_output = self._external_audio_output
            self._owns_audio_output = False
        else:
            self._audio_output = AudioOutputStream(
                device=self.config.output_device,
                sample_rate=self._tts.sample_rate,
                stereo_channel=self.config.output_channel,
            )
            self._owns_audio_output = True

        print(f"Pipeline loaded: {self.config.language_name}")
        channel_str = ""
        if self.config.output_channel is not None:
            channel_str = f" ({'left' if self.config.output_channel == 0 else 'right'} channel)"
        print(f"  Output device: {self._audio_output.device.name} (index {self._audio_output.device.index}){channel_str}")

    def unload(self) -> None:
        """Unload all components."""
        if self._translator:
            self._translator.unload()
        if self._tts:
            self._tts.unload()
        # Only stop audio if we own it
        if self._audio_output and self._owns_audio_output:
            self._audio_output.stop()

    def start(self) -> None:
        """Start the pipeline processing thread."""
        if self._running:
            return

        self._running = True
        # Only start audio if we own it (shared outputs are started by coordinator)
        if self._owns_audio_output:
            self._audio_output.start()

        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"Pipeline-{self.config.language_code}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        if self._thread:
            # Put sentinel to unblock queue
            self._queue.put(None)
            self._thread.join(timeout=5.0)
        # Only stop audio if we own it (shared outputs are stopped by coordinator)
        if self._audio_output and self._owns_audio_output:
            self._audio_output.stop()

    def process(self, text: str, chunk_start_time: float = 0.0,
                chunk_duration: float = 0.0, asr_time: float = 0.0) -> None:
        """
        Queue text for translation and playback.

        Args:
            text: English text to translate and speak
            chunk_start_time: When the audio chunk started accumulating
            chunk_duration: Duration of the source audio chunk
            asr_time: Time spent in ASR transcription
        """
        if not text or not text.strip():
            return

        depth = self._queue.qsize()
        try:
            self._queue.put_nowait((time.time(), text, chunk_start_time, chunk_duration, asr_time, depth))
        except queue.Full:
            logger.warning(
                "DROP | lang={} | reason=queue_full | queue_depth={} | text={}",
                self.config.language_code, depth, text[:60]
            )

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break

                start_time, text, chunk_start_time, chunk_duration, asr_time, queue_depth = item
                self._process_text(text, start_time, chunk_start_time, chunk_duration, asr_time, queue_depth)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Pipeline error ({}) | {}", self.config.language_name, e)

    def _process_text(self, text: str, start_time: float, chunk_start_time: float = 0.0,
                      chunk_duration: float = 0.0, asr_time: float = 0.0,
                      queue_depth: int = 0) -> None:
        """Process a single text through translation and TTS."""
        # Translate
        translation = self._translator.translate(text)
        if translation.is_empty:
            return

        # Synthesize speech — adaptive speed: when the playback queue backs
        # up (continuous preaching, translated speech longer than source),
        # speak faster so live audio catches back up. Same thresholds the
        # legacy system used for years.
        speed = 1.0
        if queue_depth >= 4:
            speed = 1.35
        elif queue_depth >= 2:
            speed = 1.2
        speech = self._tts.synthesize(translation.translated_text, speed=speed)
        if speed != 1.0:
            logger.info("[{}] queue_depth={} → speaking at {:.2f}x",
                        self.config.language_code.upper(), queue_depth, speed)
        if speech.is_empty:
            return

        # Publish to the live web page before room playback so phones aren't
        # behind the PA. No-ops (beyond a ring append) when the server is off.
        WEB_BUS.translation(self.config.language_code, translation.translated_text,
                            t0=chunk_start_time)
        WEB_BUS.audio(self.config.language_code, speech.audio, speech.sample_rate,
                      t0=chunk_start_time)

        # Record when playback starts (this is the true end-to-end point)
        playback_start = time.time()
        e2e_latency = (playback_start - chunk_start_time) if chunk_start_time > 0 else (playback_start - start_time)

        # Play audio
        if self._audio_output is not None:
            self._audio_output.play(speech.audio, sample_rate=speech.sample_rate)

        # Notify callback with full timing breakdown
        if self._on_translation:
            event = TranslationEvent(
                timestamp=start_time,
                source_text=text,
                translated_text=translation.translated_text,
                target_language=self.config.language_code,
                audio_duration=speech.duration,
                total_latency=e2e_latency,
                chunk_start_time=chunk_start_time,
                chunk_duration=chunk_duration,
                asr_time=asr_time,
                translation_time=translation.processing_time,
                tts_time=speech.processing_time,
                queue_depth=queue_depth,
            )
            self._on_translation(event)

    def set_callback(self, callback: Callable[[TranslationEvent], None]) -> None:
        """Set callback for translation events."""
        self._on_translation = callback


class TranslationCoordinator:
    """
    Main coordinator for the translation system.

    Manages audio input, ASR, and multiple language pipelines.
    """

    def __init__(
        self,
        input_device: str = "default",
        languages: List[PipelineConfig] = None,
        models_dir: Optional[str] = None,
        parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3",
        input_file: Optional[str] = None,
        input_realtime: bool = True,
    ):
        self._input_device = input_device
        self._models_dir = models_dir or str(Path(__file__).parent.parent.parent / "models")
        # PARAKEET_MODEL env var overrides the default model id. Lets you
        # swap to e.g. nemo-parakeet-tdt-1.1b without touching code or
        # adding a CLI flag — same pattern as NLLB_MODEL for translation.
        env_parakeet = os.environ.get("PARAKEET_MODEL", "").strip()
        self._parakeet_model = env_parakeet or parakeet_model
        # PARAKEET_PATH points at a local directory of ONNX files for
        # community-exported / hand-prepared models (e.g. the 1.1B without
        # punctuation). When set, PARAKEET_MODEL must be the adapter type
        # (e.g. "nemo-conformer-tdt") rather than a HF repo id.
        self._parakeet_path = os.environ.get("PARAKEET_PATH", "").strip() or None
        self._parakeet_buffer = None
        # File-input mode reads audio from a WAV/MP3/etc. instead of the mic,
        # for reproducible offline tests. When set, the audio input thread
        # auto-stops at EOF and the run() loop drains the pipeline cleanly.
        self._input_file = input_file
        self._input_realtime = input_realtime
        # Sentence buffer sits between Parakeet's token-level commits and the
        # per-language translation pipelines.
        self._sentence_buffer: Optional[SentenceBuffer] = None
        # Set from the audio thread when the ASR backend is gone for good;
        # run() then drains what is queued and exits non-zero so systemd
        # restarts the service.
        self._fatal: Optional[str] = None
        self._chunks_seen = 0
        self._last_heartbeat = 0.0

        # Default languages if not specified
        if languages is None:
            languages = [
                PipelineConfig(
                    language_code="es",
                    language_name="Spanish",
                ),
                PipelineConfig(
                    language_code="ht",
                    language_name="Haitian Creole",
                ),
            ]

        self._language_configs = languages

        # Components
        self._audio_input: Optional[AudioInputStream] = None
        self._pipelines: Dict[str, LanguagePipeline] = {}
        self._shared_outputs: Dict[str, SharedStereoOutput] = {}  # device -> shared output

        # State
        self._running = False
        self._callbacks: List[Callable[[TranslationEvent], None]] = []
        self._session_start: float = 0.0

        # Statistics
        self._stats = {
            'transcriptions': 0,
            'translations': 0,
            'silent_chunks': 0,
            'dropped': 0,
            'total_latency': 0.0,
            'total_asr_time': 0.0,
            'total_translation_time': 0.0,
            'total_tts_time': 0.0,
            'forced_emits': 0,
        }

    def load(self) -> None:
        """Load all components."""
        print("=" * 60)
        print("Loading Translation Coordinator")
        print("=" * 60)

        # Load ASR: Parakeet streaming — the unified-remote NeMo subprocess in
        # production, or onnx-asr with token-level LocalAgreement-2 — fed 1.5 s
        # chunks by _on_audio_chunk_streaming. The Whisper batch path was
        # retired on 2026-09-06 together with the legacy program it served.
        print("\nLoading ASR service...")
        if self._parakeet_path:
            print(f"  streaming backend: parakeet model={self._parakeet_model} path={self._parakeet_path}")
        else:
            print(f"  streaming backend: parakeet model={self._parakeet_model}")
        self._parakeet_buffer = ParakeetASRBuffer(
            model_name=self._parakeet_model,
            cache_dir=f"{self._models_dir}/asr/parakeet",
            model_path=self._parakeet_path,
        )
        self._parakeet_buffer.load()
        # Tunable via env; defaults tuned for a formal speaker cadence.
        # SENTENCE_BUFFER_OFF=1 disables buffering entirely (fragments go
        # straight to translation, the old behavior — useful for A/B tests).
        if os.environ.get("SENTENCE_BUFFER_OFF", "").strip() == "1":
            self._sentence_buffer = None
            print("  sentence_buffer: DISABLED (fragments go direct to translate)")
        else:
            silence_to = float(os.environ.get("SENTENCE_SILENCE_TIMEOUT", "2.0"))
            hard_to = float(os.environ.get("SENTENCE_HARD_TIMEOUT", "15.0"))
            min_words = int(os.environ.get("SENTENCE_MIN_WORDS", "3"))
            max_chars = int(os.environ.get("SENTENCE_MAX_CHARS", "800"))
            max_words = int(os.environ.get("SENTENCE_MAX_WORDS", "60"))
            sil_min_words = int(os.environ.get("SENTENCE_SILENCE_MIN_WORDS", "1"))
            strip_lead = os.environ.get("SENTENCE_STRIP_LEAD_PUNCT", "1") != "0"
            # Close sentences on the ASR's own punctuation boundary rather
            # than on a timer. Measured on 10 min of sermon: sentences
            # ending properly 18.8% -> 96.0%, segments holding two
            # sentences 74% -> 5%, content silently dropped by NLLB
            # 10% -> 0%, and the median actually gets faster.
            punct_bnd = os.environ.get("SENTENCE_PUNCT_BOUNDARY", "1") != "0"
            self._sentence_buffer = SentenceBuffer(
                silence_timeout=silence_to,
                hard_timeout=hard_to,
                min_emit_words=min_words,
                max_buffer_chars=max_chars,
                max_emit_words=max_words,
                silence_min_words=sil_min_words,
                strip_lead_punct=strip_lead,
                punct_boundary=punct_bnd,
            )
            print(
                f"  sentence_buffer: silence={silence_to}s hard={hard_to}s "
                f"min_words={min_words} max_words={max_words} "
                f"silence_min_words={sil_min_words} max_chars={max_chars} "
                f"punct_boundary={punct_bnd}"
            )

        # Load language pipelines
        print("\nLoading language pipelines...")

        # First, identify which devices need shared stereo outputs
        # Group languages by device for those using channel assignment
        device_channels: Dict[str, List[PipelineConfig]] = {}
        for config in self._language_configs:
            if config.enabled and config.output_channel is not None:
                device_key = config.output_device
                if device_key not in device_channels:
                    device_channels[device_key] = []
                device_channels[device_key].append(config)

        # Create shared stereo outputs for devices with multiple channel users
        for device, configs in device_channels.items():
            if len(configs) >= 1:  # Any channel assignment needs stereo output
                print(f"  Creating shared stereo output for device {device}...")
                # Use 44100 as preferred rate - common for USB devices
                shared = SharedStereoOutput(device=device, sample_rate=44100)
                self._shared_outputs[device] = shared

        # Now create pipelines with appropriate audio outputs
        for config in self._language_configs:
            if config.enabled:
                audio_output = None

                # Check if this language uses a shared stereo output
                if config.output_channel is not None and config.output_device in self._shared_outputs:
                    shared = self._shared_outputs[config.output_device]
                    audio_output = ChannelOutputProxy(shared, config.output_channel)

                pipeline = LanguagePipeline(config, self._models_dir, audio_output=audio_output)
                pipeline.load()
                pipeline.set_callback(self._on_translation_event)
                self._pipelines[config.language_code] = pipeline

        # Initialize audio input — file-backed for reproducible tests, or
        # mic-backed for live use.
        print("\nInitializing audio input...")
        if self._input_file:
            # File mode: same 1.5 s chunks as the mic so downstream behaviour
            # is unchanged. EOF triggers a graceful pipeline drain in run().
            from audio.file_input_stream import FileInputStream
            self._audio_input = FileInputStream(
                file_path=self._input_file,
                sample_rate=16000,
                chunk_duration=1.5,
                realtime=self._input_realtime,
            )
        else:
            # 1.5 s chunks, time-sliced (not silence-split): the ASR decides
            # commit boundaries, the sentence buffer decides sentence ones.
            self._audio_input = AudioInputStream(
                device=self._input_device,
                sample_rate=16000,
                target_chunk_duration=1.5,
                max_chunk_duration=1.5,
                silence_threshold=0.02,
                min_silence_duration=10.0,
            )
        self._audio_input.add_callback(self._on_audio_chunk_streaming)

        print("\n" + "=" * 60)
        print("Translation Coordinator Ready")
        print(f"  Input device: {self._audio_input.device.name} (index {self._audio_input.device.index})")
        print(f"  Native sample rate: {self._audio_input.native_sample_rate}Hz -> resampled to {self._audio_input.sample_rate}Hz")
        print(f"  Mode: streaming ({self._parakeet_model})")
        print(f"  Languages: {', '.join(self._pipelines.keys())}")
        print("=" * 60)

    def unload(self) -> None:
        """Unload all components."""
        if self._audio_input:
            self._audio_input.stop()

        for pipeline in self._pipelines.values():
            pipeline.unload()

        # Stop shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.stop()

    def start(self) -> None:
        """Start the translation system."""
        if self._running:
            return

        self._session_start = time.time()
        print("\nStarting translation system...")

        # Accept chunks from the first one: the audio callback drops anything
        # that arrives before _running is set, and a file input with no pacing
        # delivers the whole file in the milliseconds the outputs below take
        # to open — every chunk of a --no-realtime run was lost that way.
        # (Pipelines queue what arrives before their threads start.)
        self._running = True

        # Start audio capture FIRST (USB devices often need input opened before output)
        self._audio_input.start()

        # Start shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.start()

        # Start language pipelines
        for pipeline in self._pipelines.values():
            pipeline.start()

        print("Translation system running. Press Ctrl+C to stop.")
        print("\nListening for speech... (dots = audio detected)")
        print("-" * 40)

    def stop(self) -> None:
        """Stop the translation system."""
        if not self._running:
            return

        print("\nStopping translation system...")
        self._running = False

        # Stop audio capture
        if self._audio_input:
            self._audio_input.stop()

        # Stop pipelines
        for pipeline in self._pipelines.values():
            pipeline.stop()

        # Stop shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.stop()

        print("Translation system stopped.")

    def _on_audio_chunk_streaming(self, chunk: AudioChunk) -> None:
        """Handle incoming audio in Parakeet streaming mode — feed into rolling buffer."""
        if not self._running:
            return

        self._chunks_seen += 1
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_SEC:
            self._last_heartbeat = now
            logger.info(
                "HEARTBEAT | chunks={} | fragments={} | asr_alive={} | queues=[{}]",
                self._chunks_seen, self._stats['transcriptions'],
                self._parakeet_buffer.alive(),
                ",".join(f"{l}:{p._queue.qsize()}" for l, p in self._pipelines.items()),
            )

        try:
            result = self._parakeet_buffer.feed(chunk.data, chunk.chunk_start_time)
        except Exception as e:
            logger.error("Streaming ASR callback error: {}", e)
            if not self._parakeet_buffer.alive() and not self._fatal:
                # The ASR backend is gone for good — it stalled and was
                # killed, or it crashed. Nothing in this process can bring it
                # back, so hand the restart to systemd (Restart=always, ~45 s
                # to reload) rather than sit silent until the scheduler's
                # 900 s watchdog notices. A transient decode error, or a
                # quiet room, never reaches here: the backend is still alive.
                self._fatal = f"ASR backend is not running ({e})"
                logger.error("FATAL | {}", self._fatal)
            return

        # Fan new Parakeet fragments into the sentence buffer (or straight
        # through if buffering is disabled). We always invoke tick() on the
        # sentence buffer — even when Parakeet has nothing new this chunk — so
        # that the silence-timeout path can fire after the speaker pauses.
        if result is not None:
            new_text, seg_start_wall, asr_time = result
            self._stats['transcriptions'] += 1
            self._stats['total_asr_time'] += asr_time

            logger.info(
                "[EN-frag] {} | mode=streaming | asr={:.3f}s",
                new_text, asr_time,
            )
            WEB_BUS.commit(new_text, t0=seg_start_wall)

            if self._sentence_buffer is not None:
                emit = self._sentence_buffer.feed(new_text, seg_start_wall, asr_time)
            else:
                emit = (new_text, seg_start_wall, asr_time)

            if emit is not None:
                self._emit_to_pipelines(*emit)
            return

        # No new fragment — still run a silence tick in case a sentence is
        # sitting in the buffer waiting for the speaker's pause to be long
        # enough.
        if self._sentence_buffer is not None:
            emit = self._sentence_buffer.tick()
            if emit is not None:
                self._emit_to_pipelines(*emit)

    def _emit_to_pipelines(self, text: str, start_wall: float, asr_time: float) -> None:
        """Dispatch a fully-formed utterance to every language pipeline."""
        logger.info(
            "[EN] {} | mode=streaming/sentence | asr={:.3f}s",
            text, asr_time,
        )
        WEB_BUS.sentence(text, t0=start_wall)
        for pipeline in self._pipelines.values():
            pipeline.process(
                text,
                chunk_start_time=start_wall,
                chunk_duration=0.0,
                asr_time=asr_time,
            )

    def _on_translation_event(self, event: TranslationEvent) -> None:
        """Handle translation event from pipeline."""
        self._stats['translations'] += 1
        self._stats['total_latency'] += event.total_latency
        self._stats['total_translation_time'] += event.translation_time
        self._stats['total_tts_time'] += event.tts_time

        lang_upper = event.target_language.upper()

        logger.info(
            "[{}] {} | e2e={:.2f}s | translate={:.3f}s | tts={:.3f}s | audio={:.2f}s | queue_was={}",
            lang_upper, event.translated_text,
            event.total_latency, event.translation_time, event.tts_time,
            event.audio_duration, event.queue_depth,
        )

        # Notify callbacks
        for callback in self._callbacks:
            callback(event)

    def add_callback(self, callback: Callable[[TranslationEvent], None]) -> None:
        """Add callback for translation events."""
        self._callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get translation statistics."""
        n = self._stats['translations']
        avg_latency = self._stats['total_latency'] / n if n > 0 else 0.0
        t = self._stats['transcriptions']
        avg_asr = self._stats['total_asr_time'] / t if t > 0 else 0.0
        avg_translate = self._stats['total_translation_time'] / n if n > 0 else 0.0
        avg_tts = self._stats['total_tts_time'] / n if n > 0 else 0.0
        duration = time.time() - self._session_start if self._session_start > 0 else 0.0

        return {
            'transcriptions': t,
            'translations': n,
            'silent_chunks': self._stats['silent_chunks'],
            'dropped': self._stats['dropped'],
            'forced_emits': self._stats['forced_emits'],
            'average_latency': avg_latency,
            'avg_asr_time': avg_asr,
            'avg_translation_time': avg_translate,
            'avg_tts_time': avg_tts,
            'session_duration': duration,
        }

    def _wait_for_queues_to_drain(self, timeout: float = 60.0) -> None:
        """
        Wait for all pipeline queues and audio outputs to drain.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if all pipeline queues are empty
            all_empty = True
            for pipeline in self._pipelines.values():
                if not pipeline._queue.empty():
                    all_empty = False
                    break

            # Check if shared audio outputs are still playing
            for shared in self._shared_outputs.values():
                if shared.is_playing():
                    all_empty = False
                    break

            # Check individual audio outputs (is_playing is a property for AudioOutputStream)
            for pipeline in self._pipelines.values():
                if pipeline._owns_audio_output and pipeline._audio_output:
                    if pipeline._audio_output.is_playing:
                        all_empty = False
                        break

            if all_empty:
                print("\nAll audio finished playing.")
                return

            # Show progress
            print(".", end="", flush=True)
            time.sleep(0.5)

        print("\nTimeout waiting for audio to finish.")

    def run(self) -> None:
        """Run the translation system (blocking)."""
        self.load()
        self.start()

        shutdown_phase = 0  # 0=running, 1=draining, 2=stopping

        try:
            while self._running:
                time.sleep(0.1)
                if self._fatal:
                    # Same drain-and-shutdown path as Ctrl+C, so whatever is
                    # already queued still reaches the room; then exit.
                    print("\n" + "-" * 40)
                    print(f"FATAL: {self._fatal}")
                    print("Draining pipeline, then exiting for restart.")
                    print("-" * 40)
                    raise KeyboardInterrupt()
                # File-input mode: when the audio file is exhausted, raise
                # KeyboardInterrupt to take the same drain-and-shutdown path
                # the live mode uses for Ctrl+C. is_finished() exists only on
                # FileInputStream — getattr keeps the mic path unaffected.
                if getattr(self._audio_input, "is_finished", None) is not None:
                    if self._audio_input.is_finished():
                        print("\n" + "-" * 40)
                        print("Audio file finished — draining pipeline.")
                        print("-" * 40)
                        raise KeyboardInterrupt()
        except KeyboardInterrupt:
            shutdown_phase = 1
            print("\n" + "-" * 40)
            print("Stopping audio capture... (press Ctrl+C again to stop immediately)")
            print("-" * 40)

            # Stop audio input but let pipelines finish
            if self._audio_input:
                self._audio_input.stop()

            # Flush any remaining text in the Parakeet streaming buffer
            # first (it may produce one last fragment), then drain the
            # sentence buffer so any in-progress sentence reaches translation.
            if self._parakeet_buffer:
                flush_result = self._parakeet_buffer.flush()
                if flush_result:
                    text, start_wall, asr_time = flush_result
                    logger.info("[EN-frag] {} | mode=streaming/flush | asr={:.3f}s", text, asr_time)
                    WEB_BUS.commit(text, t0=start_wall)
                    if self._sentence_buffer is not None:
                        emit = self._sentence_buffer.feed(text, start_wall, asr_time)
                        if emit is not None:
                            self._emit_to_pipelines(*emit)
                    else:
                        self._emit_to_pipelines(text, start_wall, asr_time)
            if self._sentence_buffer is not None:
                emit = self._sentence_buffer.flush()
                if emit is not None:
                    self._emit_to_pipelines(*emit)

            # Wait for pipelines to drain their queues
            try:
                self._wait_for_queues_to_drain()
            except KeyboardInterrupt:
                shutdown_phase = 2
                print("\nForce stopping...")

        finally:
            self.stop()
            self.unload()

            # Print stats
            stats = self.get_stats()
            print("\n" + "=" * 60)
            print("Session Statistics")
            print("=" * 60)
            print(f"  Duration:        {stats['session_duration']:.0f}s")
            print(f"  Chunks:          {stats['transcriptions'] + stats['silent_chunks']} ({stats['silent_chunks']} silent, {stats['forced_emits']} forced)")
            print(f"  Transcriptions:  {stats['transcriptions']}")
            print(f"  Translations:    {stats['translations']}")
            print(f"  Dropped:         {stats['dropped']}")
            print(f"  Avg e2e latency: {stats['average_latency']:.2f}s")
            print(f"  Avg ASR time:    {stats['avg_asr_time']:.2f}s")
            print(f"  Avg translate:   {stats['avg_translation_time']:.2f}s")
            print(f"  Avg TTS time:    {stats['avg_tts_time']:.2f}s")
            logger.info(
                "SESSION_END | duration={:.0f}s | chunks={} | silent={} | forced={} | "
                "transcriptions={} | translations={} | dropped={} | "
                "avg_e2e={:.2f}s | avg_asr={:.2f}s | avg_translate={:.2f}s | avg_tts={:.2f}s",
                stats['session_duration'],
                stats['transcriptions'] + stats['silent_chunks'],
                stats['silent_chunks'],
                stats['forced_emits'],
                stats['transcriptions'],
                stats['translations'],
                stats['dropped'],
                stats['average_latency'],
                stats['avg_asr_time'],
                stats['avg_translation_time'],
                stats['avg_tts_time'],
            )

        if self._fatal:
            # Non-zero exit: systemd restarts the unit; the scheduler's
            # MIN_UPTIME guard leaves the fresh process alone while it loads.
            raise SystemExit(f"fatal: {self._fatal}")

    def __enter__(self):
        """Context manager entry."""
        self.load()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.unload()
        return False


def main():
    """Run the translator from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Church Audio Translator")
    parser.add_argument("--input", "-i", default="default", help="Input audio device")
    parser.add_argument("--languages", "-l", nargs="+", default=["es", "ht"],
                        help="Target languages")
    args = parser.parse_args()

    # Build language configs
    language_names = {
        "es": "Spanish",
        "ht": "Haitian Creole",
        "ru": "Russian",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
    }

    configs = [
        PipelineConfig(
            language_code=lang,
            language_name=language_names.get(lang, lang),
        )
        for lang in args.languages
    ]

    # Create and run coordinator
    coordinator = TranslationCoordinator(
        input_device=args.input,
        languages=configs,
    )

    coordinator.run()


if __name__ == "__main__":
    main()
