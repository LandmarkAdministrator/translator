"""
File-backed audio input stream.

Reads a WAV/MP3/etc. file and emits AudioChunk callbacks at real-time pace,
matching the public interface of AudioInputStream so the rest of the
pipeline (coordinator, ASR, sentence buffer) is unchanged.

Designed for reproducible offline tests: no microphone, no acoustic loss,
no ambient noise, and bit-for-bit identical inputs across runs. Auto-signals
completion via is_finished() so the coordinator can shut down cleanly when
the file is exhausted.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .input_stream import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class _FakeDevice:
    """Minimal stand-in for AudioDevice so coordinator's banner prints work."""
    name: str
    index: int = -1


class FileInputStream:
    """Audio input that reads from a file and paces chunks in real time.

    Public surface matches AudioInputStream:
      - .device, .native_sample_rate, .sample_rate
      - .add_callback(cb), .remove_callback(cb)
      - .start(), .stop()

    Plus:
      - .is_finished(): True once the entire file has been emitted (incl. any
        pacing sleep on the last chunk). The coordinator's run() loop polls
        this and triggers a graceful drain at EOF.
    """

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
        realtime: bool = True,
    ):
        """
        Args:
            file_path: Path to an audio file (WAV, MP3, FLAC, ...). Read via
                soundfile if available, else wave (WAV-only).
            sample_rate: Target output sample rate (Hz). Resampled if needed.
            chunk_duration: Seconds of audio per emitted chunk. Should match
                the AudioInputStream config used by the same pipeline mode
                (1.5s for Parakeet streaming, 7-12s for batch).
            realtime: When True, sleep between chunks to match wall-clock
                playback. When False, fire chunks as fast as possible (useful
                for offline batch scoring).
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.realtime = realtime

        # Load + resample audio. Done eagerly so any I/O / decoding error is
        # raised before start() and the caller can see it before launching
        # the pipeline.
        self._audio, self._native_sample_rate = self._load_audio()
        self._chunk_samples = int(self.sample_rate * self.chunk_duration)
        self._total_samples = len(self._audio)

        self._device = _FakeDevice(name=f"<file:{self.file_path.name}>")
        self._callbacks: List[Callable[[AudioChunk], None]] = []

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._finished_event = threading.Event()

    # -------------------------------------------------------------- properties
    @property
    def device(self) -> _FakeDevice:
        return self._device

    @property
    def native_sample_rate(self) -> int:
        return self._native_sample_rate

    # ---------------------------------------------------------------- callbacks
    def add_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # ------------------------------------------------------------------- state
    def is_finished(self) -> bool:
        return self._finished_event.is_set()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._finished_event.clear()
        self._thread = threading.Thread(
            target=self._emit_loop,
            daemon=True,
            name=f"FileInput-{self.file_path.name}",
        )
        self._thread.start()
        print(
            f"  FileInputStream: {self.file_path.name} "
            f"({self._total_samples / self.sample_rate:.1f}s @ {self.sample_rate}Hz, "
            f"chunk={self.chunk_duration}s, realtime={self.realtime})"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ private
    def _load_audio(self) -> tuple[np.ndarray, int]:
        """Load the file, downmix to mono, and resample to the target rate.

        Tries soundfile first (handles WAV/FLAC/OGG natively, others via
        libsndfile). Falls back to the `wave` module for plain WAV files
        when soundfile isn't installed.
        """
        try:
            import soundfile as sf
            data, native_sr = sf.read(str(self.file_path), dtype="float32", always_2d=False)
        except ImportError:
            if self.file_path.suffix.lower() != ".wav":
                raise RuntimeError(
                    f"soundfile is not installed; only .wav files are supported "
                    f"via the fallback path (got {self.file_path.suffix})"
                )
            import wave
            with wave.open(str(self.file_path), "rb") as wf:
                native_sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            if sampwidth != 2:
                raise RuntimeError(
                    f"WAV fallback only supports 16-bit PCM (got {sampwidth*8}-bit). "
                    f"Install soundfile (pip install soundfile) for other formats."
                )
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                data = data.reshape(-1, channels)

        # Downmix multi-channel to mono.
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        elif data.ndim == 2:
            data = data.flatten()

        # Resample if needed.
        if native_sr != self.sample_rate:
            from .resample import resample_audio
            data = resample_audio(data, native_sr, self.sample_rate)

        return data.astype(np.float32, copy=False), native_sr

    def _emit_loop(self) -> None:
        """Slice audio into chunks and emit at real-time pace."""
        start_wall = time.time()
        chunk_idx = 0
        offset = 0
        n = self._total_samples

        try:
            while offset < n and not self._stop_event.is_set():
                end = min(offset + self._chunk_samples, n)
                samples = self._audio[offset:end]

                # Pad the final partial chunk with zeros so downstream
                # consumers (Parakeet's rolling buffer) see a consistent
                # chunk size on every callback.
                if len(samples) < self._chunk_samples:
                    samples = np.concatenate(
                        [samples, np.zeros(self._chunk_samples - len(samples), dtype=np.float32)]
                    )

                chunk_start_wall = start_wall + chunk_idx * self.chunk_duration
                emit_wall = chunk_start_wall + self.chunk_duration

                # Pace ourselves to emit at the wall-clock equivalent of the
                # source audio's progression, using absolute deadlines (no
                # drift over a 60-minute file).
                if self.realtime:
                    sleep_for = emit_wall - time.time()
                    if sleep_for > 0:
                        # Honor stop signals during long sleeps.
                        if self._stop_event.wait(timeout=sleep_for):
                            return

                # peak_rms helps the coordinator's batch-mode path classify
                # silence; harmless to compute for streaming too.
                peak_rms = float(np.sqrt(np.mean(samples * samples)))

                chunk = AudioChunk(
                    data=samples.copy(),
                    timestamp=time.time(),
                    sample_rate=self.sample_rate,
                    channels=1,
                    chunk_start_time=chunk_start_wall,
                    emit_reason="file",
                    peak_rms=peak_rms,
                )

                for cb in list(self._callbacks):
                    try:
                        cb(chunk)
                    except Exception as e:
                        logger.error("FileInputStream callback error: %s", e)

                offset = end
                chunk_idx += 1
        finally:
            self._finished_event.set()
