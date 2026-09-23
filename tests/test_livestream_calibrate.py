"""Tests for livestream/calibrate.py — the correlation maths only.

    ./venv/bin/python tests/test_livestream_calibrate.py

The capture and ffmpeg-decode halves need hardware and a live feed, so
what is covered here is the part that can be wrong silently: whether a
known delay is recovered, with the right sign, through envelopes that
have been through different gain and noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livestream.calibrate import ENV_RATE, best_lag, envelope  # noqa: E402

RATE = 16000
FAILURES: list[str] = []


def ok(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILURES.append(f"{label}{': ' + detail if detail else ''}")


def source(seconds: float, claps: list[float], seed: int = 0) -> np.ndarray:
    """Speech-ish audio with sharp transients at the given offsets."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    sig = rng.normal(0, 0.02, n).astype(np.float32)
    # Irregular syllable-rate bursts, so nothing periodic dominates.
    at = 0.0
    while at < seconds:
        i = int(at * RATE)
        ln = int(rng.uniform(0.08, 0.25) * RATE)
        if i + ln < n:
            sig[i:i + ln] *= rng.uniform(3.0, 9.0)
        at += float(rng.uniform(0.15, 0.5))
    for c in claps:
        i = int(c * RATE)
        if 0 <= i < n - 400:
            burst = rng.normal(0, 1.0, 400).astype(np.float32)
            burst *= np.exp(-np.arange(400) / 60.0).astype(np.float32)
            sig[i:i + 400] += burst
    return sig


def through_atem(src: np.ndarray, delay: float, *, gain: float = 0.18,
                 noise: float = 0.01, seed: int = 7) -> np.ndarray:
    """The same audio, delayed and mangled the way the ATEM path mangles it.

    This is the realistic fixture: one source, two recordings of it. An
    earlier version built two INDEPENDENT signals from the same seed, so
    their base noise was identical and un-shifted, and zero lag always won.
    """
    rng = np.random.default_rng(seed)
    d = int(delay * RATE)
    out = np.zeros(len(src), dtype=np.float32)
    if d < len(src):
        out[d:] = src[: len(src) - d]
    out = out * gain + rng.normal(0, noise, len(src)).astype(np.float32)
    return out


def test_recovers_a_known_delay() -> None:
    for delay in (0.5, 1.2, 2.0):
        a = source(20, [3.0, 9.0, 15.0], seed=1)
        b = through_atem(a, delay)
        lag, conf = best_lag(envelope(a, RATE), envelope(b, RATE))
        ok(f"delay {delay}s recovered", abs(lag - delay) < 0.05,
           f"got {lag:.3f}")
        ok(f"delay {delay}s confident", conf > 0.25, f"conf {conf:.2f}")


def test_sign_convention() -> None:
    """Positive lag must mean the second signal is the delayed one, which
    is what makes `offset` addable to a t0."""
    a = source(20, [4.0], seed=2)
    b = through_atem(a, 1.5)
    lag, _ = best_lag(envelope(a, RATE), envelope(b, RATE))
    ok("delayed b gives positive lag", lag > 0, f"got {lag:.3f}")
    lag2, _ = best_lag(envelope(b, RATE), envelope(a, RATE))
    ok("reversed gives negative lag", lag2 < 0, f"got {lag2:.3f}")


def test_survives_different_gain_and_codec_noise() -> None:
    """The mixer feed and the ATEM path differ in gain, EQ and codec."""
    a = source(20, [3.0, 11.0], seed=3)
    b = through_atem(a, 1.4, gain=0.10, noise=0.008)
    lag, conf = best_lag(envelope(a, RATE), envelope(b, RATE))
    ok("gain mismatch tolerated", abs(lag - 1.4) < 0.08, f"got {lag:.3f}")
    ok("still confident", conf > 0.25, f"conf {conf:.2f}")


def test_unrelated_signals_are_not_confident() -> None:
    """The guard that stops a bogus offset being written to config."""
    a = source(20, [2.0, 8.0], seed=4)
    b = source(20, [5.0, 13.0], seed=99)   # a different service entirely
    _, conf = best_lag(envelope(a, RATE), envelope(b, RATE))
    ok("unrelated audio scores low", conf < 0.25, f"conf {conf:.2f}")


def test_envelope_properties() -> None:
    env = envelope(source(5, [1.0], seed=5), RATE)
    ok("envelope resampled to ENV_RATE", abs(len(env) - 5 * ENV_RATE) <= 1,
       f"len {len(env)}")
    ok("envelope normalized", abs(float(env.mean())) < 0.1)
    ok("stereo collapses", len(envelope(np.zeros((RATE, 2), np.float32), RATE)) > 0)
    ok("empty input is safe", len(envelope(np.zeros(0, np.float32), RATE)) == 0)


def test_short_input_is_safe() -> None:
    lag, conf = best_lag(np.zeros(3), np.zeros(3))
    ok("no crash on tiny input", (lag, conf) == (0.0, 0.0))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"OK — {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
