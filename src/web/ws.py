"""Minimal RFC 6455 WebSocket support (server side), stdlib only.

Deliberately dependency-free so the production venv needs no new packages.
Scope: what the live-translation page needs — server-to-client text and
binary frames, client-to-server control frames and small text frames,
ping/pong, clean close. Fragmented *client* frames are answered with a
close (browsers don't fragment the tiny messages our client sends).
"""
from __future__ import annotations

import base64
import hashlib
import struct
from typing import Optional, Tuple

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def handshake_response(client_key: str) -> bytes:
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n"
        "\r\n"
    ).encode()


def encode_frame(opcode: int, payload: bytes) -> bytes:
    """Server frames are unmasked. Single unfragmented frame."""
    header = bytearray([0x80 | (opcode & 0x0F)])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    return bytes(header) + payload


def text_frame(s: str) -> bytes:
    return encode_frame(OP_TEXT, s.encode())


def binary_frame(b: bytes) -> bytes:
    return encode_frame(OP_BINARY, b)


def close_frame(code: int = 1000) -> bytes:
    return encode_frame(OP_CLOSE, struct.pack(">H", code))


async def read_frame(reader) -> Optional[Tuple[int, bytes]]:
    """Read one client frame; returns (opcode, payload) or None on EOF.

    Client frames are masked per RFC. Caps payloads at 64 KiB — the page only
    ever sends tiny JSON control messages.
    """
    head = await reader.readexactly(2)
    fin_op, mask_len = head[0], head[1]
    opcode = fin_op & 0x0F
    fin = bool(fin_op & 0x80)
    masked = bool(mask_len & 0x80)
    n = mask_len & 0x7F
    if n == 126:
        n = struct.unpack(">H", await reader.readexactly(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", await reader.readexactly(8))[0]
    if n > 65536 or not fin:
        return (OP_CLOSE, b"")  # oversized or fragmented: ask to close
    mask = await reader.readexactly(4) if masked else b"\x00" * 4
    payload = bytearray(await reader.readexactly(n))
    if masked:
        for i in range(n):
            payload[i] ^= mask[i & 3]
    return (opcode, bytes(payload))
