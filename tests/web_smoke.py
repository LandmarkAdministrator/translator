#!/usr/bin/env python3
"""Smoke test for the live web server: start it, publish fake pipeline events,
connect as a raw-socket WebSocket client (masked frames per RFC), and verify:
HTTP page serve, handshake, ring replay, live text events, binary audio
framing. Stdlib + numpy only; safe to run anywhere (binds 127.0.0.1).
"""
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from web import ws as wsproto                      # noqa: E402
from web.bus import BUS                            # noqa: E402
from web.live_server import LiveServer             # noqa: E402

PORT = 8899


def masked_frame(opcode: int, payload: bytes) -> bytes:
    head = bytearray([0x80 | opcode])
    n = len(payload)
    assert n < 126
    head.append(0x80 | n)
    mask = os.urandom(4)
    body = bytes(payload[i] ^ mask[i & 3] for i in range(n))
    return bytes(head) + mask + body


PREBUF = bytearray()  # bytes received past the handshake terminator


def read_exact(sock, n):
    global PREBUF
    buf = b""
    if PREBUF:
        take = bytes(PREBUF[:n])
        del PREBUF[:len(take)]
        buf += take
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        assert chunk, "socket closed early"
        buf += chunk
    return buf


def read_server_frame(sock):
    head = read_exact(sock, 2)
    opcode = head[0] & 0x0F
    n = head[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", read_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", read_exact(sock, 8))[0]
    return opcode, read_exact(sock, n)


def main():
    server = LiveServer(PORT, host="127.0.0.1")
    threading.Thread(target=server.run_forever, daemon=True).start()
    time.sleep(0.5)

    # 1) HTTP page
    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    page = b""
    while b"</html>" not in page:
        chunk = s.recv(65536)
        if not chunk:
            break
        page += chunk
    assert b"200 OK" in page and b"Live Translation" in page, "page serve failed"
    s.close()
    print("HTTP page: OK")

    # 2) publish a pre-connection event (should arrive via ring replay)
    BUS.sentence("This sentence was published before the client connected.")

    # 3) WebSocket handshake
    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    s.sendall(
        b"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    head_end = resp.index(b"\r\n\r\n") + 4
    PREBUF.extend(resp[head_end:])  # first WS frames may already be here
    resp = resp[:head_end]
    assert b"101" in resp.split(b"\r\n")[0], resp[:80]
    expect = wsproto.accept_key("dGhlIHNhbXBsZSBub25jZQ==").encode()
    assert expect in resp, "bad accept key"
    print("WS handshake: OK")

    # 4) ring replay
    op, payload = read_server_frame(s)
    ev = json.loads(payload)
    assert op == wsproto.OP_TEXT and ev["kind"] == "en.sentence" and ev.get("replay"), ev
    print("ring replay: OK")

    # 5) live text + commit
    BUS.commit("hello congregation")
    BUS.translation("es", "Hola congregación.")
    kinds = set()
    for _ in range(2):
        op, payload = read_server_frame(s)
        kinds.add(json.loads(payload)["kind"])
    assert kinds == {"en.commit", "es.text"}, kinds
    print("live text events: OK")

    # 6) audio binary framing
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, 24000)).astype(np.float32)
    BUS.audio("es", tone, 24000)
    op, payload = read_server_frame(s)
    assert op == wsproto.OP_BINARY
    hlen = struct.unpack("<I", payload[:4])[0]
    head = json.loads(payload[4:4 + hlen])
    wav = payload[4 + hlen:]
    assert head["kind"] == "es.audio" and wav[:4] == b"RIFF" and len(wav) == 44 + 48000, (head, len(wav))
    print("audio frame: OK")

    # 7) client ping -> pong
    s.sendall(masked_frame(wsproto.OP_PING, b"hi"))
    deadline = time.time() + 5
    while time.time() < deadline:
        op, payload = read_server_frame(s)
        if op == wsproto.OP_PONG:
            assert payload == b"hi"
            print("ping/pong: OK")
            break
    else:
        raise AssertionError("no pong")

    s.sendall(masked_frame(wsproto.OP_CLOSE, struct.pack(">H", 1000)))
    s.close()
    print("ALL WEB SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
