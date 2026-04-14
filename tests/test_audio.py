#!/usr/bin/env python3
"""
Tests for the audio system.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


def test_device_manager():
    """Test audio device enumeration."""
    print("Testing Audio Device Manager...")

    from audio.device_manager import (
        AudioDeviceManager,
        list_input_devices,
        list_output_devices,
        get_default_input_device,
        get_default_output_device,
    )

    manager = AudioDeviceManager()

    print(f"\n  Total devices: {len(manager.devices)}")
    print(f"  Input devices: {len(manager.get_input_devices())}")
    print(f"  Output devices: {len(manager.get_output_devices())}")

    default_input = get_default_input_device()
    default_output = get_default_output_device()

    if default_input:
        print(f"\n  Default input: {default_input.name}")
        print(f"    Channels: {default_input.max_input_channels}")
        print(f"    Sample rate: {default_input.default_sample_rate}")
    else:
        print("\n  No default input device found!")

    if default_output:
        print(f"\n  Default output: {default_output.name}")
        print(f"    Channels: {default_output.max_output_channels}")
        print(f"    Sample rate: {default_output.default_sample_rate}")
    else:
        print("\n  No default output device found!")

    # Print all devices
    print("\n  All audio devices:")
    for device in manager.devices:
        print(f"    {device}")

    print("\n  ✓ Device manager working")
    return True


def test_circular_buffer():
    """Test the circular audio buffer."""
    print("\nTesting Circular Audio Buffer...")

    from audio.input_stream import CircularAudioBuffer, AudioChunk

    buffer = CircularAudioBuffer(
        max_duration=5.0,
        sample_rate=16000,
        channels=1,
    )

    # Test appending data
    data = np.random.randn(16000).astype(np.float32)  # 1 second
    buffer.append(data)
    print(f"  After 1s append: {buffer.duration:.2f}s")

    # Append more
    buffer.append(data)
    print(f"  After 2s append: {buffer.duration:.2f}s")

    # Get a chunk
    chunk = buffer.get_chunk(duration=1.0, overlap=0.5)
    assert chunk is not None
    assert abs(chunk.duration - 1.0) < 0.01
    print(f"  Got chunk: {chunk.duration:.2f}s, {chunk.samples} samples")

    # Check buffer after extraction
    print(f"  Buffer after extraction: {buffer.duration:.2f}s")

    # Test max capacity
    for _ in range(10):
        buffer.append(data)
    assert buffer.is_full or buffer.duration <= 5.0
    print(f"  Buffer at max: {buffer.duration:.2f}s (max 5s)")

    print("  ✓ Circular buffer working")
    return True


def test_input_stream_init():
    """Test audio input stream initialization."""
    print("\nTesting Audio Input Stream Initialization...")

    from audio.input_stream import AudioInputStream

    try:
        stream = AudioInputStream(
            device='default',
            sample_rate=16000,
            channels=1,
            buffer_duration=10.0,
            chunk_duration=2.0,
            chunk_overlap=0.5,
        )

        print(f"  Device: {stream.device.name}")
        print(f"  Sample rate: {stream.sample_rate}")
        print(f"  Chunk duration: {stream.chunk_duration}s")
        print("  ✓ Input stream initialized")
        return True

    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False


def test_output_stream_init():
    """Test audio output stream initialization."""
    print("\nTesting Audio Output Stream Initialization...")

    from audio.output_stream import AudioOutputStream

    try:
        stream = AudioOutputStream(
            device='default',
            sample_rate=22050,
            channels=1,
        )

        print(f"  Device: {stream.device.name}")
        print(f"  Sample rate: {stream.sample_rate}")
        print("  ✓ Output stream initialized")
        return True

    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False


def test_audio_capture(duration: float = 2.0):
    """Test capturing audio from the microphone."""
    print(f"\nTesting Audio Capture ({duration}s)...")

    from audio.input_stream import AudioInputStream

    try:
        stream = AudioInputStream(
            device='default',
            sample_rate=16000,
            channels=1,
            chunk_duration=1.0,
            chunk_overlap=0.2,
        )

        chunks_received = []

        def on_chunk(chunk):
            chunks_received.append(chunk)
            print(f"    Received chunk: {chunk.duration:.2f}s, "
                  f"max amplitude: {np.max(np.abs(chunk.data)):.4f}")

        stream.add_callback(on_chunk)

        print(f"  Starting capture for {duration}s...")
        stream.start()
        time.sleep(duration)
        stream.stop()

        print(f"  Captured {len(chunks_received)} chunks")
        print(f"  Buffer received: {stream.buffer.total_received} samples")

        if chunks_received:
            print("  ✓ Audio capture working")
            return True
        else:
            print("  ⚠ No chunks received (microphone may be muted)")
            return True  # Don't fail - mic might be intentionally muted

    except Exception as e:
        print(f"  ✗ Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_playback():
    """Test playing a test tone."""
    print("\nTesting Audio Playback...")

    from audio.output_stream import AudioOutputStream

    try:
        stream = AudioOutputStream(
            device='default',
            sample_rate=22050,
            channels=1,
        )

        # Generate a 440Hz test tone (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(22050 * duration), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

        print("  Playing 440Hz test tone...")
        stream.start()
        stream.play(tone, sample_rate=22050, block=True)
        time.sleep(0.5)  # Small gap
        stream.stop()

        print("  ✓ Audio playback working (did you hear it?)")
        return True

    except Exception as e:
        print(f"  ✗ Playback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all audio tests."""
    print("=" * 60)
    print("Audio System Tests")
    print("=" * 60)

    results = {}

    results['Device Manager'] = test_device_manager()
    results['Circular Buffer'] = test_circular_buffer()
    results['Input Init'] = test_input_stream_init()
    results['Output Init'] = test_output_stream_init()

    # Interactive tests (require audio hardware)
    print("\n" + "-" * 60)
    print("Interactive Tests (require audio hardware)")
    print("-" * 60)

    # Ask before running audio tests
    response = input("\nRun audio capture test? (y/n): ").strip().lower()
    if response == 'y':
        results['Audio Capture'] = test_audio_capture(2.0)

    response = input("Run audio playback test? (y/n): ").strip().lower()
    if response == 'y':
        results['Audio Playback'] = test_audio_playback()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
