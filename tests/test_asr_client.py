#!/usr/bin/env python3
"""The ASR subprocess client, exercised against fake servers — no NeMo needed.

Covers the two things that matter about the pipe to unified_asr_server.py:
  * both reply protocols decode to the same fragments — protocol 2 (delta plus
    running count, current) and protocol 1 (cumulative, what an older server
    sends), so a mismatched deploy degrades to "works" rather than "silent"
  * a server that stops answering is detected in UNIFIED_REPLY_TIMEOUT and
    reported dead, and a server that crashes is reported dead — while a
    server that answers "nothing new" (silence) is never mistaken for either

    ./venv/bin/python tests/test_asr_client.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FAKE_SERVER = r'''
import json, os, struct, sys, time
mode = os.environ["FAKE_MODE"]
out = sys.stdout
stdin = sys.stdin.buffer
ready = {"ready": True}
if mode == "p2":
    ready["protocol"] = 2
print(json.dumps(ready), flush=True)
total, n = [], 0
while True:
    hdr = stdin.read(4)
    if len(hdr) < 4:
        break
    (size,) = struct.unpack("<I", hdr)
    if size == 0xFFFFFFFF:
        if mode == "p2":
            print(json.dumps({"tokens": [], "timestamps": [], "count": len(total), "eos": True}), flush=True)
        else:
            print(json.dumps({"tokens": total, "timestamps": [float(i) for i in range(len(total))], "eos": True}), flush=True)
        break
    buf = b""
    while len(buf) < size:
        part = stdin.read(size - len(buf))
        if not part:
            sys.exit(0)
        buf += part
    if mode == "stall":
        time.sleep(3600)
    if mode == "crash":
        sys.exit(3)
    n += 1
    new = [] if (mode.endswith("silent") and n % 2 == 0) else [f" w{n}"]
    total += new
    if mode.startswith("p2"):
        print(json.dumps({"tokens": new, "timestamps": [float(n)] * len(new), "count": len(total)}), flush=True)
    else:
        print(json.dumps({"tokens": total, "timestamps": [float(i) for i in range(len(total))]}), flush=True)
'''

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def client(mode: str, timeout: float = 2.0):
    from pipeline.parakeet_asr import ParakeetASRBuffer
    os.environ["FAKE_MODE"] = mode
    os.environ["UNIFIED_REPLY_TIMEOUT"] = str(timeout)
    buf = ParakeetASRBuffer(model_name="unified-remote", cache_dir=None)
    buf.load()
    return buf


CHUNK = np.zeros(24000, dtype=np.float32)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fake-asr-"))
    server = tmp / "fake_server.py"
    server.write_text(FAKE_SERVER)
    os.environ["UNIFIED_SERVER"] = str(server)
    os.environ["UNIFIED_PYTHON"] = sys.executable

    for mode, label in (("p2", "protocol 2 (delta + count)"), ("p1", "protocol 1 (cumulative)")):
        print(f"\n{label}")
        b = client(mode)
        check("server reports alive", b.alive())
        got = [b.feed(CHUNK, 0.0) for _ in range(3)]
        # Fragments keep their SentencePiece leading space: the sentence
        # buffer needs it to tell a new word from a subword continuation.
        texts = [g[0].strip() if g else None for g in got]
        check("three pushes -> three fragments w1, w2, w3",
              texts == ["w1", "w2", "w3"], repr(texts))
        check("committed count tracks the stream", b._committed_count == 3, str(b._committed_count))
        check("flush with nothing new returns None", b.flush() is None)
        b._model.stop()

    print("\nsilence: a reply with nothing new is not an error")
    b = client("p2silent")
    got = [b.feed(CHUNK, 0.0) for _ in range(4)]
    texts = [g[0].strip() if g else None for g in got]
    check("empty replies yield None, others their word",
          texts == ["w1", None, "w3", None], repr(texts))
    check("still alive after empty replies", b.alive())
    b._model.stop()

    print("\nstall: no reply at all is detected within the deadline")
    b = client("stall", timeout=2.0)
    t0 = time.monotonic()
    try:
        b.feed(CHUNK, 0.0)
        check("feed raises", False, "returned normally")
    except RuntimeError as e:
        check("feed raises", True)
        check("error names the stall", "stalled" in str(e), str(e))
    took = time.monotonic() - t0
    check("detected in about the timeout (not the 900 s watchdog)", 1.5 <= took < 10.0, f"{took:.1f}s")
    check("client reports the server dead", not b.alive())

    print("\ncrash: a server that exits is reported dead")
    b = client("crash", timeout=5.0)
    t0 = time.monotonic()
    try:
        b.feed(CHUNK, 0.0)
        check("feed raises", False, "returned normally")
    except RuntimeError:
        check("feed raises", True)
    check("detected promptly", time.monotonic() - t0 < 3.0)
    check("client reports the server dead", not b.alive())

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
