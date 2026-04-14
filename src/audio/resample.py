"""
Audio Resampling

Shared anti-aliased resampling helper used by both the audio I/O layer and
the ASR front-end.  Uses ``scipy.signal.resample_poly`` (polyphase FIR) so
downsampling doesn't alias high frequencies into the speech band — this
matters for 48kHz / 44.1kHz capture devices being fed into 16kHz Whisper.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Resample audio between sample rates with anti-aliasing.

    Args:
        audio: 1-D float audio (or 2-D with samples on axis 0).
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled audio as ``float32``.  Same shape except the sample axis.
        A no-op (cast only) when ``from_rate == to_rate`` or ``audio`` is empty.
    """
    if audio.size == 0 or from_rate == to_rate:
        return audio.astype(np.float32, copy=False)

    # Reduce ratio to smallest integer up/down factors for polyphase.
    divisor = gcd(int(from_rate), int(to_rate))
    up = int(to_rate) // divisor
    down = int(from_rate) // divisor

    resampled = resample_poly(audio, up, down, axis=0)
    return resampled.astype(np.float32, copy=False)
