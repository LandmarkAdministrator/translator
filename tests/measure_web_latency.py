#!/usr/bin/env python3
"""Measure end-to-end latency from the capture device to the web page.

Three questions, all timed from the moment the first audio sample of an
utterance arrived at the USB codec (the pipeline's chunk_start_time, carried to
the client as t0):

  1. English text reaching the page
  2. Translated text reaching the page
  3. Translated audio actually *playing* on the page

(3) is not arrival time. The page schedules clips gaplessly — a clip that
arrives while the previous one is still playing waits its turn — so this
mirrors index.html's scheduler (nextStart = max(now + 0.05, nextStart)) to get
the moment sound would leave the speaker.

Run it ON the Translate PC against loopback so both timestamps come from one
clock; network delivery to a phone is measured separately and added.

    python3 tests/measure_web_latency.py [--host H] [--port P] [--tls] [--secs N]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import ssl
import struct
import sys
import time
from collections import defaultdict

TEXT_KINDS = ("en.sentence", "es.text", "ht.text")


def connect(host: str, port: int, tls: bool, path: str = "/ws"):
    sock = socket.create_connection((host, port), timeout=20)
    if tls:
        ctx = ssl._create_unverified_context()
        sock = ctx.wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    if b"101" not in buf.split(b"\r\n")[0]:
        raise SystemExit(f"handshake failed: {buf.split(chr(13).encode())[0]!r}")
    return sock, bytearray(buf.split(b"\r\n\r\n", 1)[1])


def frames(sock, pending, deadline):
    """Yield (opcode, payload, arrival_time) until the deadline."""
    sock.settimeout(5.0)
    while time.time() < deadline:
        try:
            while len(pending) < 2:
                pending.extend(sock.recv(65536))
            op = pending[0] & 0x0F
            n = pending[1] & 0x7F
            hdr = 2
            if n == 126:
                while len(pending) < 4:
                    pending.extend(sock.recv(65536))
                n = struct.unpack(">H", bytes(pending[2:4]))[0]; hdr = 4
            elif n == 127:
                while len(pending) < 10:
                    pending.extend(sock.recv(65536))
                n = struct.unpack(">Q", bytes(pending[2:10]))[0]; hdr = 10
            while len(pending) < hdr + n:
                pending.extend(sock.recv(65536))
            payload = bytes(pending[hdr:hdr + n]); del pending[:hdr + n]
            yield op, payload, time.time()
        except socket.timeout:
            continue
        except Exception:
            return


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--secs", type=float, default=120.0)
    args = ap.parse_args()

    sock, pending = connect(args.host, args.port, args.tls)
    print(f"connected to {args.host}:{args.port} — measuring for {args.secs:.0f}s\n")

    samples = defaultdict(list)
    # Per-language playback cursor, exactly as the page keeps it.
    next_start: dict[str, float] = {}
    deadline = time.time() + args.secs

    for op, payload, now in frames(sock, pending, deadline):
        if op == 1:
            ev = json.loads(payload)
            if ev.get("replay"):
                continue
            kind, t0 = ev.get("kind"), ev.get("t0") or 0
            if kind in TEXT_KINDS and t0:
                samples[kind].append(now - t0)
        elif op == 2:
            hl = struct.unpack("<I", payload[:4])[0]
            ev = json.loads(payload[4:4 + hl])
            kind, t0 = ev.get("kind"), ev.get("t0") or 0
            if not t0 or not kind:
                continue
            lang = kind.split(".")[0]
            # Mirror the page: start at arrival + 50 ms, or when the previous
            # clip for this language finishes, whichever is later.
            start = max(now + 0.05, next_start.get(lang, 0.0))
            next_start[lang] = start + float(ev.get("secs") or 0.0)
            samples[kind].append(start - t0)
            samples[kind + ".arrival"].append(now - t0)

    print(f"{'stage':28} {'n':>4} {'mean':>8} {'median':>8} {'p90':>8} {'max':>8}")
    print("-" * 70)
    order = ["en.sentence", "es.text", "ht.text",
             "es.audio.arrival", "es.audio", "ht.audio.arrival", "ht.audio"]
    label = {
        "en.sentence": "English text",
        "es.text": "Spanish text",
        "ht.text": "Creole text",
        "es.audio.arrival": "Spanish audio (arrives)",
        "es.audio": "Spanish audio (plays)",
        "ht.audio.arrival": "Creole audio (arrives)",
        "ht.audio": "Creole audio (plays)",
    }
    for k in order:
        v = samples.get(k)
        if not v:
            continue
        print(f"{label[k]:28} {len(v):>4} {sum(v)/len(v):>7.2f}s "
              f"{pct(v,50):>7.2f}s {pct(v,90):>7.2f}s {max(v):>7.2f}s")
    if not samples:
        print("no timestamped events seen — is the pipeline running and publishing t0?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
