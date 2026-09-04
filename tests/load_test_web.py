#!/usr/bin/env python3
"""How many listeners can the live page carry?

Opens N concurrent WebSocket clients, measures the bytes each one receives and
whether any fall behind. The interesting number is per-client bitrate: audio is
sent as uncompressed WAV and every client currently receives *every* language,
so bandwidth scales with listeners much faster than it needs to.

    python3 tests/load_test_web.py --clients 25 --secs 60 [--host H --port P --tls]
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
import threading
import time
from collections import defaultdict

stats_lock = threading.Lock()
stats: dict = defaultdict(lambda: {"bytes": 0, "text": 0, "audio": 0,
                                   "max_gap": 0.0, "err": None})


def run_client(idx: int, host: str, port: int, tls: bool, secs: float):
    try:
        sock = socket.create_connection((host, port), timeout=20)
        if tls:
            ctx = ssl._create_unverified_context()
            sock = ctx.wrap_socket(sock, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET /ws HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                      f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += sock.recv(4096)
        if b"101" not in buf.split(b"\r\n")[0]:
            with stats_lock:
                stats[idx]["err"] = "handshake"
            return
        pending = bytearray(buf.split(b"\r\n\r\n", 1)[1])
        sock.settimeout(5.0)
        end = time.time() + secs
        last = time.time()
        nbytes = ntext = naudio = 0
        maxgap = 0.0
        while time.time() < end:
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
                del pending[:hdr + n]
                nbytes += hdr + n
                if op == 1:
                    ntext += 1
                elif op == 2:
                    naudio += 1
                    gap = time.time() - last
                    maxgap = max(maxgap, gap)
                last = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                with stats_lock:
                    stats[idx]["err"] = type(e).__name__
                break
        with stats_lock:
            stats[idx].update(bytes=nbytes, text=ntext, audio=naudio, max_gap=maxgap)
        sock.close()
    except Exception as e:
        with stats_lock:
            stats[idx]["err"] = f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=25)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tls", action="store_true")
    args = ap.parse_args()

    print(f"opening {args.clients} clients against {args.host}:{args.port} "
          f"for {args.secs:.0f}s...")
    threads = [threading.Thread(target=run_client,
                                args=(i, args.host, args.port, args.tls, args.secs),
                                daemon=True)
               for i in range(args.clients)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.secs + 30)
    elapsed = time.time() - t0

    ok = [s for s in stats.values() if not s["err"]]
    bad = [s for s in stats.values() if s["err"]]
    if not ok:
        print("all clients failed:", {s["err"] for s in bad})
        return 1
    tot = sum(s["bytes"] for s in ok)
    per = tot / len(ok)
    print(f"\n  clients ok:        {len(ok)}/{args.clients}"
          + (f"   FAILED: {len(bad)} {[s['err'] for s in bad][:3]}" if bad else ""))
    print(f"  elapsed:           {elapsed:.1f}s")
    print(f"  per client:        {per/1024:.0f} KB  ->  {per*8/elapsed/1000:.0f} kbps")
    print(f"  server total:      {tot/1024/1024:.1f} MB  ->  "
          f"{tot*8/elapsed/1_000_000:.1f} Mbps for {len(ok)} clients")
    print(f"  audio frames each: min {min(s['audio'] for s in ok)} "
          f"max {max(s['audio'] for s in ok)}  (uneven = some clients dropped audio)")
    print(f"  text frames each:  min {min(s['text'] for s in ok)} "
          f"max {max(s['text'] for s in ok)}")
    rate = per * 8 / elapsed / 1000
    if rate > 0:
        for link in (50, 100, 500, 1000):
            print(f"    {link:>4} Mbps uplink supports ~{int(link*1000/rate*0.7):>4} "
                  f"listeners (70% headroom)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
