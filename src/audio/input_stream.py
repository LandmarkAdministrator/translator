"""
Audio Input Stream

Handles capturing audio from an input device with a circular buffer
for continuous processing. Automatically resamples to target rate.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Deque
import numpy as np
import sounddevice as sd

from .device_manager import AudioDeviceManager, AudioDevice


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Resample audio from one sample rate to another.

    Args:
        audio: Input audio array
        from_rate: Source sample rate
        to_rate: Target sample rate

    Returns:
        Resampled audio array
    """
    if from_rate == to_rate:
        return audio

    duration = len(audio) / from_rate
    new_length = int(duration * to_rate)

    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, audio)

    return resampled.astype(np.float32)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    timestamp: float        # When the chunk was emitted (end of accumulation)
    sample_rate: int
    channels: int
    chunk_start_time: float = 0.0   # When the first sample of this chunk arrived
    emit_reason: str = "unknown"    # 'silence', 'max_duration', or 'stop'
    peak_rms: float = 0.0           # Peak RMS energy across the chunk

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def samples(self) -> int:
        """Number of samples."""
        return len(self.data)

    @property
    def accumulation_time(self) -> float:
        """How long it took to accumulate this chunk (wall clock)."""
        if self.chunk_start_time > 0:
            return self.timestamp - self.chunk_start_time
        return 0.0


class CircularAudioBuffer:
    """
    Thread-safe circular buffer for audio data.

    Stores audio samples and allows extraction of chunks with overlap.
    """

    def __init__(
        self,
        max_duration: float,
        sample_rate: int,
        channels: int = 1
    ):
        """
        Initialize the circular buffer.

        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_samples = int(max_duration * sample_rate)

        self._buffer: Deque[float] = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
        self._total_samples_received = 0

    def append(self, data: np.ndarray) -> None:
        """Append audio data to the buffer."""
        with self._lock:
            if data.ndim > 1:
                data = data.flatten()

            for sample in data:
                self._buffer.append(sample)
            self._total_samples_received += len(data)

    def get_chunk(
        self,
        duration: float,
        overlap: float = 0.0
    ) -> Optional[AudioChunk]:
        """Get a chunk of audio from the buffer."""
        samples_needed = int(duration * self.sample_rate)

        with self._lock:
            if len(self._buffer) < samples_needed:
                return None

            chunk_data = np.array(
                list(self._buffer)[-samples_needed:],
                dtype=np.float32
            )

            overlap_samples = int(overlap * self.sample_rate)
            samples_to_remove = samples_needed - overlap_samples

            for _ in range(min(samples_to_remove, len(self._buffer))):
                self._buffer.popleft()

        return AudioChunk(
            data=chunk_data,
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def get_latest(self, duration: float) -> Optional[AudioChunk]:
        """Get the latest audio without consuming it."""
        samples_needed = int(duration * self.sample_rate)

        with self._lock:
            if len(self._buffer) < samples_needed:
                return None

            chunk_data = np.array(
                list(self._buffer)[-samples_needed:],
                dtype=np.float32
            )

        return AudioChunk(
            data=chunk_data,
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self.max_samples

    @property
    def total_received(self) -> int:
        """Total samples received since creation."""
        return self._total_samples_received


class AudioInputStream:
    """
    Captures audio from an input device.

    Uses silence-based chunking to capture complete phrases/sentences.
    Waits for natural pauses in speech before emitting chunks, rather
    than using fixed time windows. This produces much better ASR results.

    Automatically resamples to target sample rate (16000Hz for Whisper ASR).
    """

    # Common sample rates to try (44100 first for maximum USB compatibility)
    SAMPLE_RATES = [44100, 48000, 96000, 22050, 16000]

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_duration: float = 30.0,
        # Silence-based chunking parameters
        target_chunk_duration: float = 7.0,  # Target chunk size (seconds)
        max_chunk_duration: float = 12.0,    # Max before forced emit
        silence_threshold: float = 0.02,     # RMS energy threshold for silence
        min_silence_duration: float = 0.5,   # Seconds of silence to trigger chunk
    ):
        """
        Initialize the audio input stream with silence-based chunking.

        Args:
            device: Device name, index, or 'default'
            sample_rate: Target sample rate (default 16000 for Whisper)
            channels: Number of channels (default 1 for mono)
            buffer_duration: Circular buffer size in seconds
            target_chunk_duration: Target chunk size before looking for silence
            max_chunk_duration: Maximum chunk duration (forced emit)
            silence_threshold: RMS energy below this = silence (0.02 = -34dB)
            min_silence_duration: How long silence must last to trigger emit
        """
        self.sample_rate = sample_rate  # Target rate for output
        self.channels = channels
        self.target_chunk_duration = target_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration

        # Resolve device
        self._device_manager = AudioDeviceManager()
        self._device = self._device_manager.resolve_device(
            device or 'default',
            direction='input'
        )

        if self._device is None:
            raise ValueError(f"Could not find input device: {device}")

        # Find a supported sample rate for capture
        self._native_sample_rate = self._find_supported_sample_rate(sample_rate)

        # Create buffer at TARGET sample rate (after resampling)
        self._buffer = CircularAudioBuffer(
            max_duration=buffer_duration,
            sample_rate=sample_rate,  # Buffer stores resampled audio
            channels=channels,
        )

        # Silence detection state
        self._chunk_buffer: list[np.ndarray] = []  # Accumulates audio for current chunk
        self._chunk_samples = 0  # Total samples in current chunk
        self._silence_samples = 0  # Consecutive silent samples
        self._min_silence_samples = int(min_silence_duration * sample_rate)
        self._target_samples = int(target_chunk_duration * sample_rate)
        self._max_samples = int(max_chunk_duration * sample_rate)
        self._chunk_lock = threading.Lock()
        self._chunk_start_time: float = 0.0  # When the current chunk started accumulating
        self._chunk_peak_rms: float = 0.0    # Peak RMS seen in current chunk
        self._emit_reason: str = "unknown"

        # Stream state
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._callbacks: list[Callable[[AudioChunk], None]] = []

        # Processing thread
        self._process_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Queue for ready chunks
        self._ready_chunks: Deque[AudioChunk] = deque()

    def _find_supported_sample_rate(self, preferred: int) -> int:
        """Find a supported sample rate for the device.

        Tries device default first for better compatibility with USB devices
        that require matching sample rates for full-duplex operation.
        """
        # Try device default first (best for USB full-duplex), then preferred, then common rates
        device_default = int(self._device.default_sample_rate)
        rates_to_try = [device_default, preferred] + [r for r in self.SAMPLE_RATES if r not in (device_default, preferred)]

        for rate in rates_to_try:
            try:
                sd.check_input_settings(
                    device=self._device.index,
                    samplerate=rate,
                    channels=self.channels,
                )
                return rate
            except Exception:
                continue

        # Fall back to device default
        return device_default

    @property
    def device(self) -> AudioDevice:
        """Get the audio device being used."""
        return self._device

    @property
    def native_sample_rate(self) -> int:
        """Get the device's native capture sample rate."""
        return self._native_sample_rate

    @property
    def buffer(self) -> CircularAudioBuffer:
        """Get the audio buffer."""
        return self._buffer

    @property
    def is_running(self) -> bool:
        """Check if the stream is currently running."""
        return self._running

    def add_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Add a callback to be called when a chunk is ready."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Remove a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _emit_chunk(self, reason: str = "unknown") -> None:
        """Emit the current chunk buffer to callbacks."""
        with self._chunk_lock:
            if not self._chunk_buffer:
                return

            # Concatenate all audio in the chunk buffer
            chunk_data = np.concatenate(self._chunk_buffer)

            # Create the chunk with full timing/quality metadata
            chunk = AudioChunk(
                data=chunk_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_start_time=self._chunk_start_time,
                emit_reason=reason,
                peak_rms=self._chunk_peak_rms,
            )

            # Add to ready queue
            self._ready_chunks.append(chunk)

            # Reset for next chunk
            self._chunk_buffer = []
            self._chunk_samples = 0
            self._silence_samples = 0
            self._chunk_start_time = 0.0
            self._chunk_peak_rms = 0.0

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Internal callback for sounddevice stream."""
        if status:
            # Log hardware issues (overflow = CPU too slow, underflow = buffer underrun)
            from loguru import logger
            logger.warning("Audio device status: {}", str(status))

        # Convert to mono if needed
        if indata.ndim > 1 and indata.shape[1] > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata.flatten()

        # Resample if needed
        if self._native_sample_rate != self.sample_rate:
            data = resample_audio(data, self._native_sample_rate, self.sample_rate)

        # Also add to circular buffer for compatibility
        self._buffer.append(data)

        # Silence-based chunking logic
        emit_reason = None
        with self._chunk_lock:
            # Record when this chunk started
            if self._chunk_samples == 0:
                self._chunk_start_time = time.time()

            # Add data to current chunk buffer
            self._chunk_buffer.append(data.copy())
            self._chunk_samples += len(data)

            # Calculate RMS energy of this block
            rms_energy = np.sqrt(np.mean(data ** 2))
            if rms_energy > self._chunk_peak_rms:
                self._chunk_peak_rms = rms_energy

            # Check if this block is silence
            if rms_energy < self.silence_threshold:
                self._silence_samples += len(data)
            else:
                self._silence_samples = 0

            # Case 1: Reached target duration AND detected silence
            if (self._chunk_samples >= self._target_samples and
                    self._silence_samples >= self._min_silence_samples):
                emit_reason = "silence"

            # Case 2: Reached max duration (forced emit)
            if self._chunk_samples >= self._max_samples:
                emit_reason = "max_duration"

        if emit_reason:
            self._emit_chunk(reason=emit_reason)

    def _process_loop(self) -> None:
        """Background thread for processing audio chunks."""
        while not self._stop_event.is_set():
            # Check for ready chunks
            chunk = None
            if self._ready_chunks:
                try:
                    chunk = self._ready_chunks.popleft()
                except IndexError:
                    pass

            if chunk is not None:
                for callback in self._callbacks:
                    try:
                        callback(chunk)
                    except Exception:
                        pass  # Log error but continue
            else:
                time.sleep(0.05)  # Short sleep when no chunks ready

    def start(self) -> None:
        """Start capturing audio."""
        if self._running:
            return

        self._stop_event.clear()

        # Reset chunking state
        with self._chunk_lock:
            self._chunk_buffer = []
            self._chunk_samples = 0
            self._silence_samples = 0
            self._chunk_start_time = 0.0
            self._chunk_peak_rms = 0.0
        self._ready_chunks.clear()

        # Create and start the audio stream at NATIVE rate
        self._stream = sd.InputStream(
            device=self._device.index,
            samplerate=self._native_sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self._native_sample_rate * 0.25),  # 250ms blocks — reduces GIL-stall overflows
        )
        self._stream.start()

        # Start processing thread if we have callbacks
        if self._callbacks:
            self._process_thread = threading.Thread(
                target=self._process_loop,
                daemon=True,
            )
            self._process_thread.start()

        self._running = True

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return

        self._stop_event.set()

        # Emit any remaining audio in buffer
        self._emit_chunk(reason="stop")

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        self._running = False

    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the next ready chunk (manual polling mode)."""
        if self._ready_chunks:
            try:
                return self._ready_chunks.popleft()
            except IndexError:
                pass
        return None

    def get_latest(self, duration: Optional[float] = None) -> Optional[AudioChunk]:
        """Get the latest audio without consuming it."""
        return self._buffer.get_latest(duration or self.target_chunk_duration)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
