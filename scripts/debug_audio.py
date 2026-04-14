#!/usr/bin/env python3
"""
Debug script for audio input issues.
Tests audio capture from a specific device.
"""

import os
import sys
import time
import numpy as np

# Set GPU environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import sounddevice as sd
from audio.device_manager import AudioDeviceManager


def test_raw_capture(device_index: int, duration: float = 5.0):
    """Test raw audio capture using sounddevice directly."""
    print(f"\n=== Test 1: Raw sounddevice capture (device {device_index}) ===")

    try:
        # Get device info
        device_info = sd.query_devices(device_index)
        print(f"Device: {device_info['name']}")
        print(f"Max input channels: {device_info['max_input_channels']}")
        print(f"Default sample rate: {device_info['default_samplerate']}")

        sample_rate = int(device_info['default_samplerate'])
        channels = 1

        print(f"\nRecording {duration} seconds at {sample_rate}Hz...")

        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device_index,
            dtype=np.float32
        )

        # Show progress
        for i in range(int(duration)):
            time.sleep(1)
            print(f"  {i+1}/{int(duration)} seconds...")

        sd.wait()

        # Analyze
        print(f"\nRecording shape: {recording.shape}")
        print(f"Max amplitude: {np.abs(recording).max():.6f}")
        print(f"Mean amplitude: {np.abs(recording).mean():.6f}")
        print(f"RMS level: {np.sqrt(np.mean(recording**2)):.6f}")

        if np.abs(recording).max() < 0.001:
            print("\n⚠️  WARNING: Audio level is extremely low!")
            print("   - Check if microphone is connected to the Onyx Producer")
            print("   - Check input gain on the Onyx Producer")
            print("   - Make sure you're speaking into the mic during recording")
        else:
            print("\n✓ Audio captured successfully!")

        return recording, sample_rate

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_stream_capture(device_index: int, duration: float = 5.0):
    """Test stream-based audio capture with callback."""
    print(f"\n=== Test 2: Stream capture with callback (device {device_index}) ===")

    try:
        device_info = sd.query_devices(device_index)
        sample_rate = int(device_info['default_samplerate'])

        chunks_received = []
        callback_count = [0]

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"  Status: {status}")
            callback_count[0] += 1
            chunks_received.append(indata.copy())

            # Show audio level
            level = np.abs(indata).max()
            bars = int(level * 50)
            print(f"  [{callback_count[0]:3d}] {'█' * bars}{' ' * (50-bars)} {level:.4f}")

        print(f"Starting stream at {sample_rate}Hz for {duration} seconds...")
        print("Audio levels (speak into microphone):\n")

        with sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1)  # 100ms blocks
        ):
            time.sleep(duration)

        print(f"\nCallbacks received: {callback_count[0]}")

        if callback_count[0] == 0:
            print("\n✗ No audio callbacks received!")
            print("   The audio stream didn't trigger any callbacks.")
        elif all(np.abs(chunk).max() < 0.001 for chunk in chunks_received):
            print("\n⚠️  Callbacks received but audio level is near zero!")
        else:
            print("\n✓ Stream capture working!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_input_stream_class(device_index: int, duration: float = 5.0):
    """Test the AudioInputStream class."""
    print(f"\n=== Test 3: AudioInputStream class (device {device_index}) ===")

    try:
        from audio.input_stream import AudioInputStream

        chunks_received = [0]

        def on_chunk(chunk):
            chunks_received[0] += 1
            level = np.abs(chunk.data).max()
            print(f"  Chunk {chunks_received[0]}: duration={chunk.duration:.2f}s, level={level:.4f}")

        print(f"Creating AudioInputStream...")
        stream = AudioInputStream(
            device=str(device_index),
            sample_rate=16000,
            channels=1,
            chunk_duration=2.0,
            chunk_overlap=0.5,
        )

        print(f"  Device: {stream.device.name}")
        print(f"  Native rate: {stream.native_sample_rate}Hz")
        print(f"  Target rate: {stream.sample_rate}Hz")

        stream.add_callback(on_chunk)

        print(f"\nStarting stream for {duration} seconds...")
        print("(Chunks should appear every ~1.5 seconds after initial 2s buffer)\n")

        stream.start()
        time.sleep(duration)
        stream.stop()

        print(f"\nChunks received: {chunks_received[0]}")

        if chunks_received[0] == 0:
            print("\n⚠️  No chunks received!")
            print("   This could mean:")
            print("   - Audio buffer not filling up (no audio input)")
            print("   - Callback not being triggered")
        else:
            print("\n✓ AudioInputStream working!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("Audio Input Debugger")
    print("=" * 60)

    manager = AudioDeviceManager()

    print("\nAvailable INPUT devices:")
    for dev in manager.get_input_devices():
        default = " (default)" if dev.is_default_input else ""
        print(f"  [{dev.index}] {dev.name}{default}")

    # Find Onyx Producer
    onyx = None
    for dev in manager.get_input_devices():
        if "onyx" in dev.name.lower():
            onyx = dev
            break

    if onyx:
        print(f"\nFound Onyx Producer at index {onyx.index}")
        device_index = onyx.index
    else:
        device_index = int(input("\nEnter device index to test: "))

    # Run tests
    test_raw_capture(device_index, duration=5.0)
    test_stream_capture(device_index, duration=5.0)
    test_input_stream_class(device_index, duration=10.0)

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
