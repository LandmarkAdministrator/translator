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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio.input_stream import AudioInputStream, AudioChunk
from audio.output_stream import AudioOutputStream, SharedStereoOutput, ChannelOutputProxy
from pipeline.asr import ASRService, TranscriptionResult
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
    timestamp: float
    source_text: str
    translated_text: str
    target_language: str
    audio_duration: float
    total_latency: float


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

    def process(self, text: str) -> None:
        """
        Queue text for translation and playback.

        Args:
            text: English text to translate and speak
        """
        if not text or not text.strip():
            return

        try:
            self._queue.put_nowait((time.time(), text))
        except queue.Full:
            print(f"Warning: {self.config.language_name} queue full, dropping text")

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break

                start_time, text = item
                self._process_text(text, start_time)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pipeline error ({self.config.language_name}): {e}")

    def _process_text(self, text: str, start_time: float) -> None:
        """Process a single text through translation and TTS."""
        # Translate
        translation = self._translator.translate(text)
        if translation.is_empty:
            return

        # Synthesize speech
        speech = self._tts.synthesize(translation.translated_text)
        if speech.is_empty:
            return

        # Play audio
        self._audio_output.play(speech.audio, sample_rate=speech.sample_rate)

        # Calculate latency
        total_latency = time.time() - start_time

        # Notify callback
        if self._on_translation:
            event = TranslationEvent(
                timestamp=start_time,
                source_text=text,
                translated_text=translation.translated_text,
                target_language=self.config.language_code,
                audio_duration=speech.duration,
                total_latency=total_latency,
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
        asr_model: str = "base.en",
        models_dir: Optional[str] = None,
    ):
        self._input_device = input_device
        self._asr_model = asr_model
        self._models_dir = models_dir or str(Path(__file__).parent.parent.parent / "models")

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
        self._pipelines: Dict[str, LanguagePipeline] = {}
        self._shared_outputs: Dict[str, SharedStereoOutput] = {}  # device -> shared output

        # State
        self._running = False
        self._callbacks: List[Callable[[TranslationEvent], None]] = []

        # Statistics
        self._stats = {
            'transcriptions': 0,
            'translations': 0,
            'total_latency': 0.0,
        }

    def load(self) -> None:
        """Load all components."""
        print("=" * 60)
        print("Loading Translation Coordinator")
        print("=" * 60)

        # Load ASR
        print("\nLoading ASR service...")
        self._asr = ASRService(
            model_size=self._asr_model,
            language="en",
            download_root=f"{self._models_dir}/asr",
        )
        self._asr.load()

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

        # Initialize audio input with silence-based chunking
        print("\nInitializing audio input...")
        self._audio_input = AudioInputStream(
            device=self._input_device,
            sample_rate=16000,
            # Silence-based chunking: wait for pauses in speech
            target_chunk_duration=7.0,   # Target 7 seconds of speech
            max_chunk_duration=12.0,     # Force emit after 12 seconds
            silence_threshold=0.02,      # RMS threshold for silence detection
            min_silence_duration=0.5,    # 500ms of silence triggers emit
        )
        self._audio_input.add_callback(self._on_audio_chunk)

        print("\n" + "=" * 60)
        print("Translation Coordinator Ready")
        print(f"  Input device: {self._audio_input.device.name} (index {self._audio_input.device.index})")
        print(f"  Native sample rate: {self._audio_input.native_sample_rate}Hz -> resampled to {self._audio_input.sample_rate}Hz")
        print(f"  Chunking: silence-based (target {self._audio_input.target_chunk_duration}s, max {self._audio_input.max_chunk_duration}s)")
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

    def start(self) -> None:
        """Start the translation system."""
        if self._running:
            return

        print("\nStarting translation system...")

        # Start audio capture FIRST (USB devices often need input opened before output)
        self._audio_input.start()

        # Start shared stereo outputs
        for shared in self._shared_outputs.values():
            shared.start()

        # Start language pipelines
        for pipeline in self._pipelines.values():
            pipeline.start()

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

        # Show chunk info
        chunk_duration = chunk.duration
        print(f"\n--- Chunk received: {chunk_duration:.1f}s ---")

        # Transcribe
        result = self._asr.transcribe(chunk.data, chunk.sample_rate)

        if result.is_empty:
            print("  (no speech detected)")
            return

        self._stats['transcriptions'] += 1

        # Send to all language pipelines
        for pipeline in self._pipelines.values():
            pipeline.process(result.text)

        # Print transcription
        print(f"[EN] {result.text}")

    def _on_translation_event(self, event: TranslationEvent) -> None:
        """Handle translation event from pipeline."""
        self._stats['translations'] += 1
        self._stats['total_latency'] += event.total_latency

        # Print translation
        lang_upper = event.target_language.upper()
        print(f"[{lang_upper}] {event.translated_text} ({event.total_latency:.2f}s)")

        # Notify callbacks
        for callback in self._callbacks:
            callback(event)

    def add_callback(self, callback: Callable[[TranslationEvent], None]) -> None:
        """Add callback for translation events."""
        self._callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get translation statistics."""
        avg_latency = 0.0
        if self._stats['translations'] > 0:
            avg_latency = self._stats['total_latency'] / self._stats['translations']

        return {
            'transcriptions': self._stats['transcriptions'],
            'translations': self._stats['translations'],
            'average_latency': avg_latency,
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
            print(f"  Transcriptions: {stats['transcriptions']}")
            print(f"  Translations: {stats['translations']}")
            print(f"  Average latency: {stats['average_latency']:.2f}s")

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
    parser.add_argument("--model", "-m", default="base.en", help="ASR model size")
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
        asr_model=args.model,
    )

    coordinator.run()


if __name__ == "__main__":
    main()
