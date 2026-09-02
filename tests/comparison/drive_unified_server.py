#!/usr/bin/env python3
"""Standalone driver for unified_asr_server.py: stream a wav in realtime-sized
pushes (without sleeping), flush, save the transcript, report timing.

Runs with the PRODUCTION venv python (client side needs only numpy/soundfile);
the server subprocess uses ~/nemo-venv/bin/python.

Usage: drive_unified_server.py WAV OUT_TXT [chunk_secs=1.5]
"""
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

wav_path, out_txt = sys.argv[1], sys.argv[2]
push_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5

server = str(Path(__file__).parent.parent.parent / "src" / "pipeline" / "unified_asr_server.py")
python = str(Path.home() / "nemo-venv" / "bin" / "python")

audio, sr = sf.read(wav_path, dtype="float32")
assert sr == 16000, sr

proc = subprocess.Popen([python, server], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
t0 = time.time()
line = proc.stdout.readline()
msg = json.loads(line)
if not msg.get("ready"):
    print("SERVER FAILED:", msg)
    sys.exit(1)
print(f"[server ready in {time.time()-t0:.0f}s]")

step = int(push_secs * sr)
n_tokens = 0
t0 = time.time()
lat = []
for i in range(0, len(audio), step):
    seg = np.ascontiguousarray(audio[i:i+step], dtype=np.float32)
    data = seg.tobytes()
    ts = time.time()
    proc.stdin.write(struct.pack("<I", len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()
    msg = json.loads(proc.stdout.readline())
    lat.append(time.time() - ts)
    if "error" in msg:
        print("STEP ERROR:", msg["error"])
        sys.exit(1)
    n_tokens = len(msg["tokens"])
proc.stdin.write(struct.pack("<I", 0xFFFFFFFF))
proc.stdin.flush()
msg = json.loads(proc.stdout.readline())
text = "".join(msg["tokens"]).strip()
Path(out_txt).write_text(text + "\n")
dur = len(audio) / sr
print(f"[{dur:.0f}s audio in {time.time()-t0:.1f}s wall; per-push decode "
      f"avg {sum(lat)/len(lat)*1000:.0f}ms max {max(lat)*1000:.0f}ms; "
      f"{len(msg['tokens'])} tokens -> {out_txt}]")
print("first 200 chars:", text[:200])
