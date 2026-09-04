#!/usr/bin/env python3
"""Connect to the live page's WebSocket and print what a phone would receive.

Verifies the whole publish path — pipeline thread -> bus -> server -> client —
against the running service, which unit tests cannot cover.

Usage: ws_probe.py [host:port] [seconds]
"""
import base64
import json
import os
import socket
import struct
import sys
import time

target = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
host, port = target.split(":")

key = base64.b64encode(os.urandom(16)).decode()
s = socket.create_connection((host, int(port)), timeout=10)
s.sendall((f"GET /ws HTTP/1.1\r\nHost: {target}\r\nUpgrade: websocket\r\n"
           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
           f"Sec-WebSocket-Version: 13\r\n\r\n").encode())

buf = b""
while b"\r\n\r\n" not in buf:
    buf += s.recv(4096)
head, rest = buf.split(b"\r\n\r\n", 1)
print(head.decode(errors="replace").splitlines()[0])
assert b"101" in head.split(b"\r\n")[0], "handshake failed"

pending = bytearray(rest)


def read(n):
    while len(pending) < n:
        chunk = s.recv(65536)
        if not chunk:
            raise SystemExit("closed")
        pending.extend(chunk)
    out = bytes(pending[:n])
    del pending[:n]
    return out


counts, deadline = {}, time.time() + secs
s.settimeout(max(2.0, secs))
try:
    while time.time() < deadline:
        h = read(2)
        op, n = h[0] & 0x0F, h[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", read(8))[0]
        payload = read(n)
        if op == 0x1:
            ev = json.loads(payload)
            counts[ev["kind"]] = counts.get(ev["kind"], 0) + 1
            if not ev.get("replay"):
                print(f"  {ev['kind']:<12} {ev.get('text','')[:78]}")
        elif op == 0x2:
            hlen = struct.unpack("<I", payload[:4])[0]
            ev = json.loads(payload[4:4 + hlen])
            counts[ev["kind"]] = counts.get(ev["kind"], 0) + 1
            print(f"  {ev['kind']:<12} [{len(payload) - 4 - hlen} bytes of audio]")
        elif op == 0x9:
            print("  (ping)")
except (socket.timeout, TimeoutError):
    pass
print("\nevent counts:", counts if counts else "NONE — is audio reaching the Behringer?")
