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
    def commit(self, text: str) -> None:
        """Streaming ASR committed new English words (never revised)."""
        self._publish("en.commit", {"text": text})

    def sentence(self, text: str) -> None:
        """A full English sentence was handed to translation."""
        self._publish("en.sentence", {"text": text})

    def translation(self, lang: str, text: str) -> None:
        self._publish(f"{lang}.text", {"text": text})

    def audio(self, lang: str, samples, sample_rate: int) -> None:
        try:
            payload = wav_bytes(samples, sample_rate)
        except Exception:
            return
        self._publish(f"{lang}.audio", {"bytes": len(payload)}, binary=payload)


def frame_binary(event: dict, payload: bytes) -> bytes:
    """Binary wire format: [u32 header_len][header JSON][WAV bytes]."""
    head = json.dumps(event).encode()
    return struct.pack("<I", len(head)) + head + payload


# Single process-wide bus; the coordinator publishes into it and the web
# server (when enabled) serves from it. Importing this module is cheap and
# publishing with no server attached is a no-op beyond the ring append.
BUS = LiveBus()
