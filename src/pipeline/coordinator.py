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
from pipeline.asr import ASRService, WhisperTransformersService, TranscriptionResult
from pipeline.asr_process import ASRProcess, ASRChunkMeta
from pipeline.parakeet_asr import ParakeetASRBuffer
from pipeline.sentence_buffer import SentenceBuffer
from pipeline.translation import TranslationService, TranslationResult
from pipeline.tts import TTSService, SpeechResult


@dataclass
class PipelineConfig:
    """Configuration for a single language pipeline."""
    language_code: str
    language_name: str
    output_device: str = "default"
    output_channel: Optional[int] = None  # 0=left, 1=right, None=both/mono
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
    asr_time: float             # Time spent in Whisper transcription
    translation_time: float     # Time spent in MarianMT translation
    tts_time: float             # Time spent in Piper TTS synthesis
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

        # Synthesize speech
        speech = self._tts.synthesize(translation.translated_text)
        if speech.is_empty:
            return

        # Record when playback starts (this is the true end-to-end point)
        playback_start = time.time()
        e2e_latency = (playback_start - chunk_start_time) if chunk_start_time > 0 else (playback_start - start_time)

        # Play audio
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
        asr_model: str = "large-v3",
        models_dir: Optional[str] = None,
        asr_device: str = "cuda",
        parakeet: bool = False,
        parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3",
    ):
        self._input_device = input_device
        self._asr_model = asr_model
        self._asr_device = asr_device
        self._models_dir = models_dir or str(Path(__file__).parent.parent.parent / "models")
        self._parakeet = parakeet
        self._parakeet_model = parakeet_model
        self._parakeet_buffer = None
        # Sentence buffer sits between Parakeet's token-level commits and the
        # per-language translation pipelines. Only used in parakeet streaming
        # mode; batch mode already delivers phrase/sentence-sized Whisper
        # segments to pipeline.process().
        self._sentence_buffer: Optional[SentenceBuffer] = None

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
        self._asr: Optional[ASRService] = None
        self._asr_proc: Optional[ASRProcess] = None          # batch-mode subprocess
        self._asr_result_thread: Optional[threading.Thread] = None
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

        # Load ASR.
        #   Parakeet: onnx-asr with token-level LocalAgreement-2, runs through
        #             _on_audio_chunk_streaming (1.5s chunks).
        #   Batch:    ASR subprocess (transformers) with its own GIL so the
        #             audio callback never stalls.
        print("\nLoading ASR service...")
        if self._parakeet:
            print(f"  streaming backend: parakeet (onnx-asr) model={self._parakeet_model}")
            self._parakeet_buffer = ParakeetASRBuffer(
                model_name=self._parakeet_model,
                cache_dir=f"{self._models_dir}/asr/parakeet",
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
                hard_to = float(os.environ.get("SENTENCE_HARD_TIMEOUT", "10.0"))
                self._sentence_buffer = SentenceBuffer(
                    silence_timeout=silence_to,
                    hard_timeout=hard_to,
                )
                print(f"  sentence_buffer: silence={silence_to}s hard={hard_to}s")
        else:
            download_root = (
                f"{self._models_dir}/asr/transformers"
                if self._asr_device == "cuda"
                else f"{self._models_dir}/asr"
            )
            self._asr_proc = ASRProcess(
                model_size=self._asr_model,
                device=self._asr_device,
                language="en",
                download_root=download_root,
            )
            self._asr_proc.start()

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

        # Initialize audio input
        print("\nInitializing audio input...")
        if self._parakeet:
            # Parakeet streaming: 1.5s chunks, time-sliced (not silence-split).
            # LocalAgreement-2 at the token level decides commit boundaries.
            self._audio_input = AudioInputStream(
                device=self._input_device,
                sample_rate=16000,
                target_chunk_duration=1.5,
                max_chunk_duration=1.5,
                silence_threshold=0.02,
                min_silence_duration=10.0,
            )
            self._audio_input.add_callback(self._on_audio_chunk_streaming)
        else:
            # Batch mode: silence-based chunking.
            self._audio_input = AudioInputStream(
                device=self._input_device,
                sample_rate=16000,
                target_chunk_duration=7.0,
                max_chunk_duration=12.0,
                silence_threshold=0.02,
                min_silence_duration=0.5,
            )
            self._audio_input.add_callback(self._on_audio_chunk)

        mode_str = "streaming (parakeet / onnx-asr)" if self._parakeet else "batch (silence-based)"
        print("\n" + "=" * 60)
        print("Translation Coordinator Ready")
        print(f"  Input device: {self._audio_input.device.name} (index {self._audio_input.device.index})")
        print(f"  Native sample rate: {self._audio_input.native_sample_rate}Hz -> resampled to {self._audio_input.sample_rate}Hz")
        print(f"  Mode: {mode_str}")
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

        if self._asr:
            self._asr.unload()
        if self._asr_proc:
            self._asr_proc.stop()
            self._asr_proc = None

    def start(self) -> None:
        """Start the translation system."""
        if self._running:
            return

        self._session_start = time.time()
        print("\nStarting translation system...")

        # Start audio capture FIRST (USB devices often need input opened before output)
        self._audio_input.start()

        # Start shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.start()

        # Start language pipelines
        for pipeline in self._pipelines.values():
            pipeline.start()

        # Start ASR result thread (batch mode only — subprocess was started in load())
        if not self._parakeet and self._asr_proc:
            self._asr_result_thread = threading.Thread(
                target=self._asr_result_loop,
                daemon=True,
                name="ASRResultLoop",
            )
            self._asr_result_thread.start()

        self._running = True

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

        # Join ASR result thread (batch mode) — must happen before stopping pipelines
        # so any in-flight results still get dispatched
        if self._asr_result_thread:
            self._asr_result_thread.join(timeout=5.0)
            self._asr_result_thread = None

        # Stop pipelines
        for pipeline in self._pipelines.values():
            pipeline.stop()

        # Stop shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.stop()

        print("Translation system stopped.")

    def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle incoming audio chunk."""
        if not self._running:
            return

        try:
            self._on_audio_chunk_inner(chunk)
        except Exception as e:
            logger.error("ASR callback error: {}", e)

    def _on_audio_chunk_inner(self, chunk: AudioChunk) -> None:
        """Inner handler — exceptions here are caught by _on_audio_chunk."""
        chunk_duration = chunk.duration
        queue_depths = {lang: p._queue.qsize() for lang, p in self._pipelines.items()}
        queue_str = ",".join(f"{l}:{d}" for l, d in queue_depths.items())

        if chunk.emit_reason == "max_duration":
            self._stats['forced_emits'] += 1

        logger.info(
            "CHUNK | duration={:.2f}s | emit={} | peak_rms={:.3f} | queues=[{}]",
            chunk_duration, chunk.emit_reason, chunk.peak_rms, queue_str
        )

        # Submit to ASR subprocess — non-blocking, audio callback returns immediately.
        # Results are delivered via _asr_result_loop running in a separate thread.
        meta = ASRChunkMeta(
            chunk_start_time=chunk.chunk_start_time,
            chunk_duration=chunk.duration,
            emit_reason=chunk.emit_reason,
            peak_rms=chunk.peak_rms,
            sample_rate=chunk.sample_rate,
        )
        submitted = self._asr_proc.submit(chunk.data, meta)
        if not submitted:
            self._stats['dropped'] += 1
            logger.warning(
                "DROP | reason=asr_queue_full | duration={:.2f}s | rms={:.3f}",
                chunk_duration, chunk.peak_rms,
            )

    def _asr_result_loop(self) -> None:
        """
        Background thread: drain ASR subprocess results and dispatch to pipelines.

        Runs for the lifetime of the session.  Exits when _running is False and
        no further results arrive within one polling interval.
        """
        while True:
            result = self._asr_proc.get_result(timeout=0.2)
            if result is None:
                if not self._running:
                    break
                continue

            transcription, meta = result
            asr_time = meta.asr_time
            chunk_duration = meta.chunk_duration

            if transcription.is_empty:
                self._stats['silent_chunks'] += 1
                logger.info(
                    "SILENT | duration={:.2f}s | asr_time={:.3f}s | rms={:.3f} | emit={}",
                    chunk_duration, asr_time, meta.peak_rms, meta.emit_reason,
                )
                continue

            self._stats['transcriptions'] += 1
            self._stats['total_asr_time'] += asr_time

            # Translate each Whisper segment individually instead of the joined blob.
            # Whisper segments are phrase/sentence-level boundaries — far more accurate
            # than our silence-based chunk boundaries for dispatching to TTS.
            for seg in transcription.segments:
                seg_start_wall = (
                    (meta.chunk_start_time + seg.start)
                    if meta.chunk_start_time > 0
                    else 0.0
                )

                logger.info(
                    "[EN] {} | chunk={:.2f}s | seg={:.2f}-{:.2f}s | asr={:.3f}s | confidence={:.3f} | lang_prob={:.3f}",
                    seg.text, chunk_duration, seg.start, seg.end, asr_time,
                    seg.confidence, transcription.language_probability,
                )

                for pipeline in self._pipelines.values():
                    pipeline.process(
                        seg.text,
                        chunk_start_time=seg_start_wall,
                        chunk_duration=seg.duration,
                        asr_time=asr_time,
                    )

    def _on_audio_chunk_streaming(self, chunk: AudioChunk) -> None:
        """Handle incoming audio in Parakeet streaming mode — feed into rolling buffer."""
        if not self._running:
            return

        try:
            result = self._parakeet_buffer.feed(chunk.data, chunk.chunk_start_time)
        except Exception as e:
            logger.error("Streaming ASR callback error: {}", e)
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
            if self._parakeet and self._parakeet_buffer:
                flush_result = self._parakeet_buffer.flush()
                if flush_result:
                    text, start_wall, asr_time = flush_result
                    logger.info("[EN-frag] {} | mode=streaming/flush | asr={:.3f}s", text, asr_time)
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
