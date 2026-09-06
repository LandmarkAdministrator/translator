#!/usr/bin/env python3
"""Capture ASR fragments with real sub-second arrival times.

Log replay cannot validate flush changes: loguru timestamps have only second
resolution, so the gaps that drive the silence timeout are unrecoverable. This
runs the production streaming ASR over a wav and records (arrival_time,
fragment) at full resolution, so buffer settings can then be swept offline
without paying for ASR each time.

    UNIFIED_PYTHON=./venv/bin/python python3 tests/capture_fragments.py in.wav out.json
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def new_since(msg: dict, prev: int) -> tuple[list, list, int]:
    """(new tokens, their times, new total) from a reply of either protocol."""
    toks = msg.get("tokens") or []
    times = [float(x) for x in (msg.get("timestamps") or [])]
    if "count" in msg:                      # protocol 2: already a delta
        return toks, times[:len(toks)], int(msg["count"])
    if len(toks) <= prev:                   # protocol 1: cumulative
        return [], [], prev
    return toks[prev:], times[prev:len(toks)], len(toks)


def main() -> int:
    wav, out_path = sys.argv[1], sys.argv[2]
    push_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    realtime = os.environ.get("CAPTURE_REALTIME", "1") != "0"

    server = str(Path(__file__).resolve().parent.parent / "src" / "pipeline" / "unified_asr_server.py")
    python = os.environ.get("UNIFIED_PYTHON") or sys.executable
    if not Path(python).exists():
        sys.exit(f"interpreter not found: {python}")

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"

    proc = subprocess.Popen([python, server], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    t0 = time.time()
    ready = json.loads(proc.stdout.readline())
    if not ready.get("ready"):
        sys.exit(f"server failed: {ready}")
    print(f"[server ready in {time.time()-t0:.0f}s]", flush=True)

    step = int(push_secs * sr)
    frags = []
    chunks = []       # EVERY chunk, including those that produced no text —
                      # production only runs the silence tick on those, so a
                      # faithful replay needs to know where they are.
    prev = 0          # protocol 1 returned cumulative tokens; fragments are the
                      # diff. Protocol 2 returns only new tokens plus a count.
    start = time.monotonic()
    for i in range(0, len(audio), step):
        chunk = np.ascontiguousarray(audio[i:i + step], dtype=np.float32)
        if realtime:
            # Pace at wall-clock so the recorded gaps match a live service.
            due = start + (i / sr)
            slack = due - time.monotonic()
            if slack > 0:
                time.sleep(slack)
        data = chunk.tobytes()
        proc.stdin.write(struct.pack("<I", len(data)))
        proc.stdin.write(data)
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            break
        msg = json.loads(line)
        if "error" in msg:
            sys.exit(f"server error: {msg['error']}")
        new_toks, new_times, prev = new_since(msg, prev)
        txt = "".join(new_toks)
        now = round(time.monotonic() - start, 4)
        rec = {"t": now, "audio_t": round(i / sr, 4), "text": txt,
               "tokens": new_toks, "times": new_times}
        chunks.append(rec)
        if txt:
            frags.append(rec)

    proc.stdin.write(struct.pack("<I", 0xFFFFFFFF))
    proc.stdin.flush()
    line = proc.stdout.readline()
    if line:
        new_toks, new_times, prev = new_since(json.loads(line), prev)
        if new_toks:
            rec = {"t": round(time.monotonic() - start, 4),
                   "audio_t": round(len(audio) / sr, 4),
                   "text": "".join(new_toks), "tokens": new_toks, "times": new_times}
            chunks.append(rec); frags.append(rec)
    proc.stdin.close()
    proc.wait(timeout=30)

    json.dump({"wav": wav, "sr": sr, "seconds": len(audio) / sr,
               "realtime": realtime, "push_secs": push_secs,
               "chunks": chunks, "fragments": frags},
              open(out_path, "w"), indent=1)
    gaps = [round(b["t"] - a["t"], 3) for a, b in zip(frags, frags[1:])]
    empty = sum(1 for c in chunks if not c["text"])
    print(f"  {len(chunks)} chunks ({len(frags)} with text, {empty} empty) "
          f"over {len(audio)/sr:.0f}s -> {out_path}")
    print(f"  empty chunks are where production's silence tick can fire")
    if gaps:
        gs = sorted(gaps)
        print(f"  inter-fragment gap: median {gs[len(gs)//2]:.2f}s  "
              f"p90 {gs[int(len(gs)*0.9)]:.2f}s  max {max(gs):.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
