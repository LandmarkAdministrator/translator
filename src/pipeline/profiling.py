"""Span recording for benchmarking the live pipeline.

Off unless `TRANSLATE_PROFILE` names a file, and when off every call here
is a bool check — safe to leave in the production path.

Why spans and not just durations
--------------------------------
`TranslationEvent` already carries `translation_time`, `tts_time` and
`asr_time`, so per-stage *durations* have always been available.  What
they cannot tell you is whether the language threads actually overlap.

Each language runs in its own thread with its own queue, translator and
TTS instance (`LanguagePipeline` in `coordinator.py`), so in principle
three languages run concurrently.  In practice they contend — for the
GIL, for the GPU, and for whatever locking sits inside torch.  The only
way to know how much is to record when each stage *started* and *ended*
per thread, then compare the sum of the durations against the wall-clock
span they collectively occupy:

    overlap = sum(durations) / |union(spans)|

overlap ≈ N  → the N languages run truly in parallel; a language is free
overlap ≈ 1  → fully serialized; a language costs its full time
in between  → partial, which is what real hardware usually gives

That ratio is the number that decides how many languages a given box can
carry, and it is the number this module exists to produce.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

_PATH: Optional[str] = os.environ.get("TRANSLATE_PROFILE") or None
_ENABLED = bool(_PATH)
_LOCK = threading.Lock()
_ROWS: list[dict] = []

# Wall-clock anchor, so rows from a run can be related to anything else
# timestamped on the same machine (GPU samples, the ingest's PDT).
_T0_WALL = time.time()
_T0_PERF = time.perf_counter()


def enabled() -> bool:
    return _ENABLED


def _now() -> float:
    """Seconds since this process started profiling, monotonic."""
    return time.perf_counter() - _T0_PERF


@contextmanager
def span(stage: str, lang: str = "", **meta: Any):
    """Record one stage's start and end.  A no-op when profiling is off."""
    if not _ENABLED:
        yield
        return
    start = _now()
    try:
        yield
    finally:
        end = _now()
        row = {
            "kind": "span",
            "stage": stage,
            "lang": lang,
            "thread": threading.current_thread().name,
            "start": round(start, 6),
            "end": round(end, 6),
            "dur": round(end - start, 6),
        }
        if meta:
            row.update(meta)
        with _LOCK:
            _ROWS.append(row)


def record(kind: str, **fields: Any) -> None:
    """Record a point event (a completed sentence, a queue depth sample)."""
    if not _ENABLED:
        return
    row = {"kind": kind, "t": round(_now(), 6)}
    row.update(fields)
    with _LOCK:
        _ROWS.append(row)


def flush() -> Optional[str]:
    """Write everything recorded so far as JSONL.  Returns the path."""
    if not _ENABLED or _PATH is None:
        return None
    with _LOCK:
        rows = list(_ROWS)
        _ROWS.clear()
    header = {
        "kind": "meta",
        "t0_wall": _T0_WALL,
        "pid": os.getpid(),
        "written": time.time(),
    }
    mode = "a" if os.path.exists(_PATH) else "w"
    with open(_PATH, mode, encoding="utf-8") as fh:
        if mode == "w":
            fh.write(json.dumps(header) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return _PATH


if _ENABLED:
    import atexit
    atexit.register(flush)
