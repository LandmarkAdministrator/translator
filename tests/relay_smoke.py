#!/usr/bin/env python3
"""Verify the split web server: it outlives the pipeline and reports standby.

Starts the server, connects a publisher, streams text and audio through the
Unix socket, then drops the publisher and checks the server stays up and flips
clients to standby.
"""
import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

PORT = 8894
tmp = Path(tempfile.mkdtemp())
SOCK = str(tmp / "relay.sock")
os.environ["TRANSLATOR_ADMIN_CREDENTIALS"] = str(tmp / "admin.json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from web import auth                       # noqa: E402
from web.bus import BUS                    # noqa: E402
from web.live_server import LiveServer     # noqa: E402
from web.relay import RelayPublisher       # noqa: E402

auth.CRED_PATH = Path(os.environ["TRANSLATOR_ADMIN_CREDENTIALS"])
auth.save_credentials("admin", "a-long-enough-password", auth.CRED_PATH)

import base64  # noqa: E402
import socket  # noqa: E402


def ws_connect():
    s = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    k = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {k}\r\n"
               f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    assert b"101" in buf.split(b"\r\n")[0]
    return s, bytearray(buf.split(b"\r\n\r\n", 1)[1])


def read_events(s, pending, seconds=4.0):
    """Yield decoded events (text frames only) for a while."""
    s.settimeout(seconds)
    out, deadline = [], time.time() + seconds
    while time.time() < deadline:
        try:
            while len(pending) < 2:
                pending.extend(s.recv(65536))
            op = pending[0] & 0x0F
            n = pending[1] & 0x7F
            hdr = 2
            if n == 126:
                while len(pending) < 4:
                    pending.extend(s.recv(65536))
                n = struct.unpack(">H", bytes(pending[2:4]))[0]; hdr = 4
            elif n == 127:
                while len(pending) < 10:
                    pending.extend(s.recv(65536))
                n = struct.unpack(">Q", bytes(pending[2:10]))[0]; hdr = 10
            while len(pending) < hdr + n:
                pending.extend(s.recv(65536))
            payload = bytes(pending[hdr:hdr + n]); del pending[:hdr + n]
            if op == 1:
                out.append(json.loads(payload))
            elif op == 2:
                hl = struct.unpack("<I", payload[:4])[0]
                out.append({**json.loads(payload[4:4 + hl]), "_bin": len(payload) - 4 - hl})
        except socket.timeout:
            break
        except Exception:
            break
    return out


def main():
    srv = LiveServer(port=PORT, host="127.0.0.1", relay_path=SOCK)
    threading.Thread(target=srv.run_forever, daemon=True).start()
    time.sleep(1.0)

    # 1. Server is up with no pipeline at all — the whole point of the split.
    s, pend = ws_connect()
    evs = read_events(s, pend, 1.5)
    assert any(e.get("kind") == "status" and e.get("live") is False for e in evs), evs
    print("server serves clients with NO pipeline running, reports standby: OK")

    # 2. Pipeline connects.
    pub = RelayPublisher(SOCK).start()
    time.sleep(1.2)
    evs = read_events(s, pend, 1.5)
    assert any(e.get("kind") == "status" and e.get("live") is True for e in evs), evs
    print("pipeline connect flips clients to live: OK")

    # 3. Text and audio cross the process boundary.
    import numpy as np
    BUS.sentence("The word of God is living and powerful.")
    BUS.translation("es", "La palabra de Dios es viva y eficaz.")
    BUS.audio("es", np.zeros(8000, dtype=np.float32), 16000)
    evs = read_events(s, pend, 3.0)
    kinds = [e.get("kind") for e in evs]
    assert "en.sentence" in kinds, kinds
    assert "es.text" in kinds, kinds
    audio = [e for e in evs if e.get("kind") == "es.audio"]
    assert audio and audio[0]["_bin"] > 8000, audio
    print(f"text + audio relayed across processes: OK ({audio[0]['_bin']} bytes of WAV)")

    # 4. Pipeline stops. Server must survive and say so.
    pub.stop()
    time.sleep(2.5)
    evs = read_events(s, pend, 3.0)
    assert any(e.get("kind") == "status" and e.get("live") is False for e in evs), evs
    print("pipeline stop flips clients to standby: OK")

    s2, pend2 = ws_connect()
    evs2 = read_events(s2, pend2, 1.5)
    assert any(e.get("kind") == "status" and e.get("live") is False for e in evs2), evs2
    print("server still accepts NEW clients after pipeline died: OK")

    import http.client
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=8)
    c.request("GET", "/admin")
    r = c.getresponse(); body = r.read(); c.close()
    assert r.status == 200 and b"Sign in" in body
    print("admin panel reachable with translation stopped: OK")

    print("\nALL RELAY TESTS PASSED")


if __name__ == "__main__":
    main()
