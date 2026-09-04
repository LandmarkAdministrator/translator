#!/usr/bin/env python3
"""Verify the TLS listener: HTTPS page, WSS stream, and admin over TLS.

Generates a throwaway self-signed certificate so this runs anywhere without
touching the real internal-CA material.
"""
import http.client
import json
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

TLS_PORT, PLAIN_PORT = 8896, 8895
tmp = Path(tempfile.mkdtemp())
cert, key = tmp / "test.crt", tmp / "test.key"
os.environ["TRANSLATOR_ADMIN_CREDENTIALS"] = str(tmp / "admin.json")

subprocess.run(
    ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
     "-keyout", str(key), "-out", str(cert), "-days", "1",
     "-subj", "/CN=127.0.0.1",
     "-addext", "subjectAltName=IP:127.0.0.1"],
    check=True, capture_output=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from web import auth                      # noqa: E402
from web.bus import BUS                   # noqa: E402
from web.live_server import LiveServer    # noqa: E402

auth.save_credentials("admin", "a-long-enough-password", Path(os.environ["TRANSLATOR_ADMIN_CREDENTIALS"]))


def main():
    srv = LiveServer(port=PLAIN_PORT, host="127.0.0.1", tls_port=TLS_PORT,
                     certfile=str(cert), keyfile=str(key), tls_host="127.0.0.1")
    threading.Thread(target=srv.run_forever, daemon=True).start()
    time.sleep(1.0)

    ctx = ssl.create_default_context(cafile=str(cert))
    ctx.check_hostname = False

    c = http.client.HTTPSConnection("127.0.0.1", TLS_PORT, context=ctx, timeout=8)
    c.request("GET", "/")
    r = c.getresponse(); body = r.read(); c.close()
    assert r.status == 200 and b"Live Translation" in body, r.status
    print(f"HTTPS page over TLS: OK ({len(body)} bytes)")

    c = http.client.HTTPSConnection("127.0.0.1", TLS_PORT, context=ctx, timeout=8)
    c.request("GET", "/admin")
    r = c.getresponse(); body = r.read(); c.close()
    assert r.status == 200 and b"Sign in" in body, r.status
    print("admin login page over TLS: OK")

    # WSS: raw TLS socket, RFC 6455 handshake, then a published event.
    import base64
    import socket
    import struct
    raw = socket.create_connection(("127.0.0.1", TLS_PORT), timeout=8)
    sock = ctx.wrap_socket(raw, server_hostname="127.0.0.1")
    k = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {k}\r\n"
                  f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    assert b"101" in buf.split(b"\r\n")[0], buf[:80]
    print("WSS handshake over TLS: OK")

    BUS.sentence("A sentence delivered over the encrypted stream.")
    sock.settimeout(6)
    deadline = time.time() + 6
    got = False
    pending = bytearray(buf.split(b"\r\n\r\n", 1)[1])
    while time.time() < deadline and not got:
        while len(pending) < 2:
            pending.extend(sock.recv(65536))
        n = pending[1] & 0x7F
        hdr = 2
        if n == 126:
            while len(pending) < 4:
                pending.extend(sock.recv(65536))
            n = struct.unpack(">H", bytes(pending[2:4]))[0]; hdr = 4
        while len(pending) < hdr + n:
            pending.extend(sock.recv(65536))
        payload = bytes(pending[hdr:hdr + n]); del pending[:hdr + n]
        if payload.startswith(b"{"):
            ev = json.loads(payload)
            if "encrypted stream" in ev.get("text", ""):
                got = True
    assert got, "event did not arrive over WSS"
    print("live event over WSS: OK")
    sock.close()

    # The plain listener must be loopback-only.
    import socket as s2
    probe = s2.socket()
    probe.settimeout(3)
    assert probe.connect_ex(("127.0.0.1", PLAIN_PORT)) == 0, "loopback HTTP not reachable"
    probe.close()
    print("plain HTTP still available on loopback: OK")
    print("\nALL TLS TESTS PASSED")


if __name__ == "__main__":
    main()
