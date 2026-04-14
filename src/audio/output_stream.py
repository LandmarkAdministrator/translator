"""
Audio Output Stream

Handles playing audio to an output device with queue-based
streaming for low-latency playback.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import sounddevice as sd

from .device_manager import AudioDeviceManager, AudioDevice
from .resample import resample_audio  # re-exported for backwards compatibility


@dataclass
class AudioSegment:
    """Represents a segment of audio to be played."""
    data: np.ndarray
    sample_rate: int
    priority: int = 0  # Higher priority plays first

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate


def mono_to_stereo_channel(audio: np.ndarray, channel: int) -> np.ndarray:
    """
    Convert mono audio to stereo with audio on a specific channel.

    Args:
        audio: Mono audio array (1D)
        channel: Target channel (0=left, 1=right)

    Returns:
        Stereo audio array (2D, shape: samples x 2)
    """
    audio = audio.flatten().astype(np.float32)
    stereo = np.zeros((len(audio), 2), dtype=np.float32)
    stereo[:, channel] = audio
    return stereo


class AudioOutputStream:
    """
    Plays audio to an output device.

    Supports queue-based streaming with priority ordering for
    real-time audio playback. Automatically resamples audio to
    match the device's native sample rate.

    Supports stereo channel assignment: output mono audio to only
    the left (0) or right (1) channel of a stereo device. This allows
    multiple languages to share a single stereo output device.
    """

    # Common sample rates to try (44100 first for maximum USB compatibility)
    SAMPLE_RATES = [44100, 48000, 96000, 22050, 16000]

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 22050,
        channels: int = 1,
        queue_size: int = 100,
        stereo_channel: Optional[int] = None,
    ):
        """
        Initialize the audio output stream.

        Args:
            device: Device name, index, or 'default'
            sample_rate: Preferred sample rate (will use device native if not supported)
            channels: Number of channels
            queue_size: Maximum number of queued audio segments
            stereo_channel: If set, output mono audio to this stereo channel only
                           (0=left, 1=right). Device will be opened in stereo mode.
                           If None, uses channels parameter for mono/stereo output.
        """
        # If stereo_channel is specified, force stereo output
        if stereo_channel is not None:
            if stereo_channel not in (0, 1):
                raise ValueError("stereo_channel must be 0 (left) or 1 (right)")
            self._stereo_channel = stereo_channel
            self.channels = 2  # Force stereo output
        else:
            self._stereo_channel = None
            self.channels = channels

        # Resolve device
        self._device_manager = AudioDeviceManager()
        self._device = self._device_manager.resolve_device(
            device or 'default',
            direction='output'
        )

        if self._device is None:
            raise ValueError(f"Could not find output device: {device}")

        # Find a supported sample rate
        self._native_sample_rate = self._find_supported_sample_rate(sample_rate)
        self.sample_rate = self._native_sample_rate  # Use native rate for output

        # Audio queue
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self._sequence = 0  # For FIFO ordering within same priority

        # Stream state
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._stop_event = threading.Event()

        # Current playback state
        self._current_segment: Optional[AudioSegment] = None
        self._current_position = 0
        self._lock = threading.Lock()

        # Statistics
        self._segments_played = 0
        self._total_duration_played = 0.0

    def _find_supported_sample_rate(self, preferred: int) -> int:
        """Find a supported sample rate for the device."""
        # Try preferred rate first
        rates_to_try = [preferred] + [r for r in self.SAMPLE_RATES if r != preferred]

        for rate in rates_to_try:
            try:
                sd.check_output_settings(
                    device=self._device.index,
                    samplerate=rate,
                    channels=self.channels,
                )
                return rate
            except Exception:
                continue

        # Fall back to device default
        return int(self._device.default_sample_rate)

    @property
    def device(self) -> AudioDevice:
        """Get the audio device being used."""
        return self._device

    @property
    def native_sample_rate(self) -> int:
        """Get the device's native sample rate being used."""
        return self._native_sample_rate

    @property
    def is_running(self) -> bool:
        """Check if the stream is currently running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get the number of segments in the queue."""
        return self._queue.qsize()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            return self._current_segment is not None

    @property
    def stereo_channel(self) -> Optional[int]:
        """Get the stereo channel this stream outputs to (0=left, 1=right, None=both)."""
        return self._stereo_channel

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Internal callback for sounddevice stream."""
        if status:
            pass  # Could log audio issues

        filled = 0
        outdata.fill(0)  # Start with silence

        with self._lock:
            while filled < frames:
                # Get next segment if needed
                if self._current_segment is None:
                    try:
                        _, _, segment = self._queue.get_nowait()
                        self._current_segment = segment
                        self._current_position = 0
                    except queue.Empty:
                        break

                # Copy data from current segment
                segment = self._current_segment
                remaining_in_segment = len(segment.data) - self._current_position
                to_copy = min(frames - filled, remaining_in_segment)

                segment_data = segment.data[
                    self._current_position:self._current_position + to_copy
                ]

                # Handle channel output
                if self.channels == 1:
                    outdata[filled:filled + to_copy, 0] = segment_data[:to_copy]
                elif self._stereo_channel is not None:
                    # Output to specific stereo channel only
                    outdata[filled:filled + to_copy, self._stereo_channel] = segment_data[:to_copy]
                else:
                    # Output to all channels
                    for ch in range(self.channels):
                        outdata[filled:filled + to_copy, ch] = segment_data[:to_copy]

                filled += to_copy
                self._current_position += to_copy

                # Check if segment is complete
                if self._current_position >= len(segment.data):
                    self._segments_played += 1
                    self._total_duration_played += segment.duration
                    self._current_segment = None
                    self._current_position = 0

    def play(
        self,
        audio: Union[np.ndarray, AudioSegment],
        sample_rate: Optional[int] = None,
        priority: int = 0,
        block: bool = False,
    ) -> bool:
        """
        Queue audio for playback.

        Args:
            audio: Audio data as numpy array or AudioSegment
            sample_rate: Sample rate of input audio (required if audio is numpy array)
            priority: Playback priority (higher = play sooner)
            block: If True, wait until playback completes

        Returns:
            True if audio was queued successfully
        """
        if isinstance(audio, np.ndarray):
            if sample_rate is None:
                sample_rate = self._native_sample_rate
            audio_data = audio.astype(np.float32).flatten()
        else:
            audio_data = audio.data
            sample_rate = audio.sample_rate

        # Resample if needed
        if sample_rate != self._native_sample_rate:
            audio_data = resample_audio(audio_data, sample_rate, self._native_sample_rate)

        segment = AudioSegment(
            data=audio_data,
            sample_rate=self._native_sample_rate,
            priority=priority,
        )

        try:
            self._sequence += 1
            self._queue.put_nowait((-segment.priority, self._sequence, segment))

            if block:
                timeout = segment.duration + 1.0
                start = time.time()
                while time.time() - start < timeout:
                    if not self.is_playing and self._queue.empty():
                        break
                    time.sleep(0.01)

            return True

        except queue.Full:
            return False

    def play_sync(
        self,
        audio: Union[np.ndarray, AudioSegment],
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Play audio synchronously (blocking).

        Args:
            audio: Audio data as numpy array or AudioSegment
            sample_rate: Sample rate (required if audio is numpy array)
        """
        self.play(audio, sample_rate, block=True)

    def clear_queue(self) -> int:
        """
        Clear all queued audio.

        Returns:
            Number of segments cleared
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break

        with self._lock:
            self._current_segment = None
            self._current_position = 0

        return count

    def start(self) -> None:
        """Start the output stream."""
        if self._running:
            return

        self._stop_event.clear()

        self._stream = sd.OutputStream(
            device=self._device.index,
            samplerate=self._native_sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self._native_sample_rate * 0.05),  # 50ms blocks
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop the output stream."""
        if not self._running:
            return

        self._stop_event.set()

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class SharedStereoOutput:
    """
    A shared stereo output stream that multiple sources can write to.

    When multiple languages need to output to the same device on different
    stereo channels, they share this single output stream. Each source
    writes to its assigned channel (0=left, 1=right).

    This avoids the "device unavailable" error that occurs when trying
    to open multiple streams to the same audio device.
    """

    # Common sample rates to try
    SAMPLE_RATES = [44100, 48000, 96000, 22050, 16000]

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 44100,
        queue_size: int = 100,
    ):
        """
        Initialize the shared stereo output.

        Args:
            device: Device name, index, or 'default'
            sample_rate: Preferred sample rate (default 44100 for USB compatibility)
            queue_size: Maximum queued segments per channel
        """
        self.channels = 2  # Always stereo

        # Resolve device
        self._device_manager = AudioDeviceManager()
        self._device = self._device_manager.resolve_device(
            device or 'default',
            direction='output'
        )

        if self._device is None:
            raise ValueError(f"Could not find output device: {device}")

        # Use device's default sample rate first (most likely to work with USB devices)
        # USB audio interfaces often require matching sample rates for full-duplex
        device_default = int(self._device.default_sample_rate)
        self._native_sample_rate = self._find_supported_sample_rate(device_default, sample_rate)
        self.sample_rate = self._native_sample_rate

        # Separate queues for each channel
        self._queues = {
            0: queue.Queue(maxsize=queue_size),  # Left channel
            1: queue.Queue(maxsize=queue_size),  # Right channel
        }
        self._sequences = {0: 0, 1: 0}

        # Stream state
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._lock = threading.Lock()

        # Current playback state per channel
        self._current_segments = {0: None, 1: None}
        self._current_positions = {0: 0, 1: 0}

    def _find_supported_sample_rate(self, device_default: int, preferred: int) -> int:
        """Find a supported sample rate for the device."""
        # Try device default first (best for USB devices), then preferred, then common rates
        rates_to_try = [device_default, preferred] + [r for r in self.SAMPLE_RATES if r not in (device_default, preferred)]

        for rate in rates_to_try:
            try:
                sd.check_output_settings(
                    device=self._device.index,
                    samplerate=rate,
                    channels=self.channels,
                )
                return rate
            except Exception:
                continue

        return device_default

    @property
    def device(self) -> AudioDevice:
        """Get the audio device being used."""
        return self._device

    @property
    def native_sample_rate(self) -> int:
        """Get the device's native sample rate."""
        return self._native_sample_rate

    @property
    def is_running(self) -> bool:
        """Check if the stream is running."""
        return self._running

    def is_playing(self) -> bool:
        """Check if any audio is currently playing or queued."""
        with self._lock:
            # Check if any channel has a current segment or queued audio
            for channel in [0, 1]:
                if self._current_segments[channel] is not None:
                    return True
                if not self._queues[channel].empty():
                    return True
        return False

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Internal callback - mixes both channels."""
        outdata.fill(0)

        with self._lock:
            for channel in [0, 1]:
                filled = 0
                while filled < frames:
                    # Get next segment if needed
                    if self._current_segments[channel] is None:
                        try:
                            segment = self._queues[channel].get_nowait()
                            self._current_segments[channel] = segment
                            self._current_positions[channel] = 0
                        except queue.Empty:
                            break

                    # Copy data from current segment
                    segment = self._current_segments[channel]
                    remaining = len(segment.data) - self._current_positions[channel]
                    to_copy = min(frames - filled, remaining)

                    segment_data = segment.data[
                        self._current_positions[channel]:
                        self._current_positions[channel] + to_copy
                    ]

                    # Write to this channel only
                    outdata[filled:filled + to_copy, channel] = segment_data

                    filled += to_copy
                    self._current_positions[channel] += to_copy

                    # Check if segment complete
                    if self._current_positions[channel] >= len(segment.data):
                        self._current_segments[channel] = None
                        self._current_positions[channel] = 0

    def play(
        self,
        audio: np.ndarray,
        channel: int,
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        Queue audio for playback on a specific channel.

        Args:
            audio: Audio data as numpy array
            channel: Target channel (0=left, 1=right)
            sample_rate: Sample rate of input audio

        Returns:
            True if audio was queued successfully
        """
        if channel not in (0, 1):
            raise ValueError("channel must be 0 (left) or 1 (right)")

        if sample_rate is None:
            sample_rate = self._native_sample_rate

        audio_data = audio.astype(np.float32).flatten()

        # Resample if needed
        if sample_rate != self._native_sample_rate:
            audio_data = resample_audio(audio_data, sample_rate, self._native_sample_rate)

        segment = AudioSegment(
            data=audio_data,
            sample_rate=self._native_sample_rate,
        )

        try:
            self._queues[channel].put_nowait(segment)
            return True
        except queue.Full:
            return False

    def start(self) -> None:
        """Start the output stream."""
        if self._running:
            return

        self._stream = sd.OutputStream(
            device=self._device.index,
            samplerate=self._native_sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self._native_sample_rate * 0.05),
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop the output stream."""
        if not self._running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class ChannelOutputProxy:
    """
    A proxy that provides an AudioOutputStream-like interface
    but writes to a specific channel of a SharedStereoOutput.

    This allows language pipelines to use the same interface
    whether they have a dedicated stream or share a stereo output.
    """

    def __init__(self, shared_output: SharedStereoOutput, channel: int):
        """
        Initialize the channel proxy.

        Args:
            shared_output: The shared stereo output to write to
            channel: The channel this proxy writes to (0=left, 1=right)
        """
        self._shared = shared_output
        self._channel = channel

    @property
    def device(self) -> AudioDevice:
        return self._shared.device

    @property
    def sample_rate(self) -> int:
        return self._shared.sample_rate

    @property
    def native_sample_rate(self) -> int:
        return self._shared.native_sample_rate

    @property
    def is_running(self) -> bool:
        return self._shared.is_running

    @property
    def stereo_channel(self) -> int:
        return self._channel

    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio."""
        with self._shared._lock:
            if self._shared._current_segments[self._channel] is not None:
                return True
            if not self._shared._queues[self._channel].empty():
                return True
        return False

    def play(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        priority: int = 0,
        block: bool = False,
    ) -> bool:
        """Queue audio for playback on this channel."""
        result = self._shared.play(audio, self._channel, sample_rate)

        if block and result:
            # Wait for playback (approximate)
            duration = len(audio) / (sample_rate or self._shared.sample_rate)
            time.sleep(duration + 0.1)

        return result

    def start(self) -> None:
        """Start is handled by the shared output."""
        pass

    def stop(self) -> None:
        """Stop is handled by the shared output."""
        pass
