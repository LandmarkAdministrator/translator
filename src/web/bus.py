"""Thread-safe event bus between the translation pipeline and the web server.

The pipeline (worker threads) calls publish_* — cheap, never blocks, drops
rather than stalls. The asyncio web server drains through per-client queues.
Text events keep a ring buffer so late joiners get recent transcript context;
audio is live-only (no backlog replay into someone's headphones).
"""
from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

RING_SIZE = 200


def wav_bytes(samples, sample_rate: int) -> bytes:
    """PCM16 WAV from a float or int16 numpy array (mono)."""
    import numpy as np
    a = samples
    if a.dtype != np.int16:
        a = np.clip(a, -1.0, 1.0)
        a = (a * 32767.0).astype(np.int16)
    data = a.tobytes()
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16,
        1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return hdr + data


class LiveBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._ring: Deque[dict] = deque(maxlen=RING_SIZE)
        # sinks are callables installed by the server; called from pipeline
        # threads with (event_dict, binary_or_None) and must not block.
        self._sinks: List[Callable[[dict, Optional[bytes]], None]] = []

    def add_sink(self, sink: Callable[[dict, Optional[bytes]], None]) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def ring(self) -> List[dict]:
        with self._lock:
            return list(self._ring)

    def clear_ring(self) -> None:
        with self._lock:
            self._ring.clear()

    def inject(self, event: dict, binary: Optional[bytes] = None) -> None:
        """Re-publish an event that arrived from another process.

        Sequence numbers are reassigned locally so they stay monotonic for our
        own clients across pipeline restarts; the original timestamp survives,
        since it marks when the words were actually spoken.
        """
        kind = event.get("kind", "unknown")
        rest = {k: v for k, v in event.items() if k not in ("seq", "kind")}
        self._publish(kind, rest, binary)

    def _publish(self, kind: str, data: dict, binary: Optional[bytes] = None) -> None:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "kind": kind, "t": round(time.time(), 3), **data}
            # Ring only replayable text: commits are transient (already
            # superseded by their sentence) and audio is live-only.
            if binary is None and kind != "en.commit":
                self._ring.append(event)
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(event, binary)
            except Exception:
                pass

    # -- pipeline-facing helpers ------------------------------------------
    # Every helper takes t0: the wall-clock moment the first audio sample of
    # this speech arrived from the capture device. Carrying it to the client
    # is what makes end-to-end latency measurable at the point of delivery
    # rather than inferred from the server side.
    def commit(self, text: str, t0: float = 0.0) -> None:
        """Streaming ASR committed new English words (never revised)."""
        self._publish("en.commit", {"text": text, "t0": t0})

    def sentence(self, text: str, t0: float = 0.0) -> None:
        """A full English sentence was handed to translation."""
        self._publish("en.sentence", {"text": text, "t0": t0})

    def translation(self, lang: str, text: str, t0: float = 0.0) -> None:
        self._publish(f"{lang}.text", {"text": text, "t0": t0})

    def audio(self, lang: str, samples, sample_rate: int, t0: float = 0.0) -> None:
        try:
            payload = wav_bytes(samples, sample_rate)
        except Exception:
            return
        seconds = len(samples) / float(sample_rate or 1)
        self._publish(f"{lang}.audio",
                      {"bytes": len(payload), "t0": t0, "secs": round(seconds, 3)},
                      binary=payload)


def frame_binary(event: dict, payload: bytes) -> bytes:
    """Binary wire format: [u32 header_len][header JSON][WAV bytes]."""
    head = json.dumps(event).encode()
    return struct.pack("<I", len(head)) + head + payload


# Single process-wide bus; the coordinator publishes into it and the web
# server (when enabled) serves from it. Importing this module is cheap and
# publishing with no server attached is a no-op beyond the ring append.
BUS = LiveBus()
