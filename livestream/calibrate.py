"""Measure the fixed offset between the translator's clock and the ATEM's.

Why this exists
---------------
`bus.py` stamps every ASR and TTS event with `t0` — the wall-clock moment
the first audio sample of that speech arrived from the capture device.
The ingest ffmpeg stamps every segment with EXT-X-PROGRAM-DATE-TIME from
its own system clock.  Both describe the same sound in the room, but they
arrive by different paths:

    sound desk ──USB (UCA202)──────────────► translator   (a few ms)
    sound desk ──HDMI──ATEM encode──RTMP──► packager      (1–2 s)

So the same moment carries two different timestamps, and every subtitle
cue and every placed dub sentence depends on knowing the difference.

    offset = (ATEM path timestamp) − (translator path timestamp)

A positive offset means the ATEM is behind, which it always is.  Add it
to a `t0` to get the media timeline position of that speech.

How it measures
---------------
Records from the translator's configured input device for a few seconds
while simultaneously reading the audio the ingest has just written, then
cross-correlates the two envelopes.  Cross-correlation over the whole
window is far more robust than picking a single transient, and it works
on ordinary speech as well as on a clap.

A clap still helps: it gives the correlator an unambiguous peak and makes
the confidence number meaningful.  Clap once, hard, in front of a live
mic, with the ATEM switched to a camera that sees you.

Run it with the ingest already running and a live feed, before a service:

    ./venv/bin/python -m livestream.calibrate
    ./venv/bin/python -m livestream.calibrate --seconds 12 --write

`--write` stores the result in config/livestream.yaml.  Re-run it if you
change the ATEM's streaming profile or the capture device; it is fixed
hardware latency and will not drift on its own.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as config_mod
from . import m3u8

log = logging.getLogger("livestream.calibrate")

# Envelope sample rate.  100 Hz gives 10 ms resolution, which is an order
# of magnitude finer than the tolerance that matters (dub placement is
# sentence-level; ±300 ms is inaudible).
ENV_RATE = 100
CAPTURE_RATE = 16000


@dataclass
class Result:
    offset_seconds: float
    confidence: float          # normalized correlation peak, 0..1
    samples_compared: int

    @property
    def trustworthy(self) -> bool:
        # Onset envelopes are sparse, so a good match peaks around 0.3-0.6
        # rather than near 1.  Unrelated audio sits below 0.15.  Below this
        # the two recordings do not share enough content to be measuring the
        # same thing — usually the ATEM is on a source that cannot hear the
        # room, or one feed is silent.
        return self.confidence >= 0.25

    def describe(self) -> str:
        verdict = "usable" if self.trustworthy else "NOT TRUSTWORTHY"
        return (f"offset {self.offset_seconds:+.3f} s "
                f"(confidence {self.confidence:.2f} — {verdict})")


def envelope(pcm: np.ndarray, rate: int) -> np.ndarray:
    """Onset-strength envelope, resampled to ENV_RATE.

    Two deliberate choices here, both learned the hard way:

    Energy is taken in the log domain, so the measurement is immune to the
    two paths having wildly different gain — one is a clean mixer feed, the
    other has been through the ATEM's H.264/AAC encoder.

    Then the envelope is differenced and half-wave rectified, keeping only
    the *increases*.  Correlating raw energy lets any slow component shared
    by both recordings — room tone, a hum, the general shape of a loud
    passage — dominate the peak and pin the lag at zero.  Onsets are what
    actually mark a moment in time, so onsets are what we correlate.
    """
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    pcm = pcm.astype(np.float32)
    hop = max(1, int(rate / ENV_RATE))
    n = len(pcm) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = pcm[: n * hop].reshape(n, hop)
    env = np.log1p(np.sqrt((frames ** 2).mean(axis=1) + 1e-12) * 1000.0)
    onset = np.diff(env, prepend=env[:1])
    np.maximum(onset, 0.0, out=onset)
    onset -= onset.mean()
    std = onset.std()
    return onset / std if std > 1e-9 else onset


def best_lag(a: np.ndarray, b: np.ndarray, max_lag_s: float = 6.0):
    """Lag, in seconds, that best aligns `b` onto `a`, plus its confidence.

    Positive result = `b` is delayed relative to `a`.
    """
    if len(a) < ENV_RATE or len(b) < ENV_RATE:
        return 0.0, 0.0
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(max_lag_s * ENV_RATE)
    corr = np.correlate(a, b, mode="full")
    centre = len(b) - 1
    lo = max(0, centre - max_lag)
    hi = min(len(corr), centre + max_lag + 1)
    window = corr[lo:hi]
    if not len(window):
        return 0.0, 0.0
    idx = int(np.argmax(window))
    # np.correlate's index runs the opposite way to the lag we want: a peak
    # LEFT of centre means `b` lags `a`.  Negating here is what makes a
    # positive result mean "b is the delayed one", which in turn makes the
    # offset directly addable to a t0.
    lag_samples = centre - (lo + idx)
    # Normalize by the self-correlation so the peak is comparable to 1.
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) or 1.0
    return lag_samples / ENV_RATE, float(window[idx]) / denom


def capture_input(device: Optional[str], seconds: float):
    """Record from the translator's input device, returning (pcm, start_utc).

    The start timestamp is taken immediately before the stream opens, which
    is the same convention `t0` uses in the pipeline.
    """
    import sounddevice as sd

    kwargs = {"samplerate": CAPTURE_RATE, "channels": 1, "dtype": "float32"}
    if device:
        kwargs["device"] = device
    started = datetime.now(timezone.utc)
    pcm = sd.rec(int(seconds * CAPTURE_RATE), **kwargs)
    sd.wait()
    return pcm.reshape(-1), started


def read_ingest_audio(cfg: config_mod.Config, start: datetime, seconds: float):
    """Decode the ingest's audio covering [start, start+seconds).

    Returns (pcm, actual_start_utc) or (None, None) when the window is not
    on disk yet — the ingest writes a segment only once it is complete, so
    the caller has to wait out one segment plus a little.
    """
    staging = cfg.staging_dir / "audio_en"
    parsed = m3u8.parse_media_playlist(staging / "playlist.m3u8")
    if not parsed.segments or not parsed.init_uri:
        return None, None

    end = start + timedelta(seconds=seconds)
    wanted = [s for s in parsed.segments
              if s.program_date_time is not None
              and s.end_time() > start and s.program_date_time < end]
    if not wanted:
        return None, None
    # The window must be fully covered, or the correlation compares
    # different spans of time and the lag is meaningless.
    if wanted[0].program_date_time > start or wanted[-1].end_time() < end:
        return None, None

    blob = bytearray((staging / parsed.init_uri).read_bytes())
    for seg in wanted:
        p = staging / seg.uri
        if not p.is_file():
            return None, None
        blob += p.read_bytes()

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1",
         "-ar", str(CAPTURE_RATE), "pipe:1"],
        input=bytes(blob), capture_output=True, timeout=120,
    )
    if proc.returncode != 0 or not proc.stdout:
        log.error("could not decode ingest audio: %s",
                  proc.stderr.decode(errors="replace")[-400:])
        return None, None
    return np.frombuffer(proc.stdout, dtype=np.float32), wanted[0].program_date_time


def measure(cfg: config_mod.Config, seconds: float,
            device: Optional[str] = None) -> Optional[Result]:
    """Record, wait for the ingest to catch up, correlate."""
    log.info("recording %.0fs from the capture device — clap once, hard", seconds)
    mic, mic_start = capture_input(device, seconds)

    # The ingest only finalizes a segment when it is complete, so the tail
    # of our window needs one segment plus slack before it is readable.
    wait = cfg.segment_seconds * 2 + 2
    log.info("waiting %.0fs for the ingest to finalize that window", wait)
    time.sleep(wait)

    atem, atem_start = read_ingest_audio(cfg, mic_start, seconds)
    if atem is None:
        log.error("the ingest has not written that window — is it running, "
                  "and is the ATEM sending?")
        return None

    env_mic = envelope(mic, CAPTURE_RATE)
    env_atem = envelope(atem, CAPTURE_RATE)

    # Align the two arrays to a common absolute start before correlating,
    # so the lag we measure is purely path latency and not the difference
    # in where each recording happened to begin.
    skew = (atem_start - mic_start).total_seconds()
    if skew > 0:
        env_mic = env_mic[int(skew * ENV_RATE):]
    elif skew < 0:
        env_atem = env_atem[int(-skew * ENV_RATE):]

    lag, conf = best_lag(env_mic, env_atem)
    return Result(offset_seconds=lag, confidence=conf,
                  samples_compared=min(len(env_mic), len(env_atem)))


def write_offset(path: Path, offset: float) -> None:
    """Record the measured offset in the config, preserving the rest."""
    line = f"atem_offset_seconds: {offset:.3f}"
    if not path.is_file():
        path.write_text(line + "\n", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    out, replaced = [], False
    for raw in text.splitlines():
        if raw.strip().startswith("atem_offset_seconds:"):
            out.append(line)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        out += ["", "# Measured by livestream.calibrate — the fixed latency of",
                "# the ATEM path relative to the translator's capture device.",
                line]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="livestream.calibrate", description=__doc__)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--device", default=None,
                   help="capture device (default: the system default)")
    p.add_argument("--repeat", type=int, default=3,
                   help="measurements to take; the median is reported")
    p.add_argument("--write", action="store_true",
                   help="store the result in config/livestream.yaml")
    p.add_argument("--json", action="store_true")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = config_mod.load(args.config)
    except ValueError as e:
        log.error("configuration: %s", e)
        return 2

    results: list[Result] = []
    for i in range(max(1, args.repeat)):
        log.info("--- measurement %d of %d ---", i + 1, args.repeat)
        r = measure(cfg, args.seconds, args.device)
        if r is None:
            return 1
        log.info("  %s", r.describe())
        results.append(r)

    usable = [r for r in results if r.trustworthy]
    if not usable:
        log.error("no measurement was trustworthy. Check that the ATEM is "
                  "switched to a source that hears the room, that the "
                  "capture device is the mixer feed, and clap louder.")
        return 1

    offsets = sorted(r.offset_seconds for r in usable)
    median = offsets[len(offsets) // 2]
    spread = offsets[-1] - offsets[0]

    if args.json:
        print(json.dumps({
            "offset_seconds": round(median, 3),
            "spread_seconds": round(spread, 3),
            "measurements": [
                {"offset": round(r.offset_seconds, 3),
                 "confidence": round(r.confidence, 3)} for r in results
            ],
        }, indent=2))
    else:
        print()
        print(f"  ATEM path offset : {median:+.3f} s")
        print(f"  spread           : {spread:.3f} s over {len(usable)} good runs")
        print()
        print("  Add this to a translator t0 to get its position on the")
        print("  ATEM media timeline.")
        if spread > 0.25:
            print()
            print("  WARNING: the spread is wide. That usually means one feed")
            print("  was quiet. Re-run with a clap in a quiet room.")

    if args.write:
        path = args.config or (config_mod.REPO_ROOT / "config" / "livestream.yaml")
        write_offset(path, median)
        print(f"\n  written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
