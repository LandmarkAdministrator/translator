"""Carries bus events from the translation process to a standalone web server.

The web server used to live inside the translation process, which meant the
page and the admin panel died whenever translation stopped — including outside
service windows, which is most of the week, and precisely when someone would
want the panel to start it again.

So the web server is now its own always-on process that owns the listening
ports, and the translation pipeline connects to it as a publisher over a Unix
socket. When no publisher is connected the server serves a standby page instead
of a dead port.

Wire format, publisher -> server:
    [u32 head_len][head JSON][u32 payload_len][payload]

Publishing must never slow the pipeline down, so the sink only enqueues; a
background thread owns the socket and drops audio rather than blocking when the
consumer falls behind.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import socket
import struct
import threading
import time
from typing import Optional

from ._log import logger
from .bus import BUS

# Audio frames are large and strictly live — dropping them is correct under
# backpressure. Text is small; the queue is sized so text effectively never
# drops even during an audio burst.
QUEUE_MAX = 256
RECONNECT_MIN, RECONNECT_MAX = 0.5, 10.0
# A publisher reconnecting after longer than this starts a new service rather
# than resuming one, so the stale transcript is cleared instead of replayed.
STALE_RING_SEC = 600.0


def default_socket_path() -> str:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache")
    return os.path.join(base, "translate-web.sock")


def _encode(event: dict, payload: Optional[bytes]) -> bytes:
    head = json.dumps(event).encode()
    body = payload or b""
    return struct.pack("<I", len(head)) + head + struct.pack("<I", len(body)) + body


class RelayPublisher:
    """Bus sink in the translation process. Fire-and-forget, self-reconnecting."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_socket_path()
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "RelayPublisher":
        self._thread = threading.Thread(target=self._run, name="web-relay", daemon=True)
        self._thread.start()
        BUS.add_sink(self.sink)
        return self

    def stop(self) -> None:
        BUS.remove_sink(self.sink)
        self._stop.set()

    # Called from pipeline threads. Must not block or raise.
    def sink(self, event: dict, binary: Optional[bytes]) -> None:
        try:
            self._q.put_nowait((event, binary))
        except queue.Full:
            # Shed the oldest item to make room. The policy is age, not kind:
            # audio is the bulk of the traffic, so the oldest item is almost
            # always an audio frame and text tends to survive, but nothing
            # guarantees it. If this races with the sender we drop this one.
            try:
                self._q.get_nowait()
                self._q.put_nowait((event, binary))
            except Exception:
                pass

    def _run(self) -> None:
        backoff = RECONNECT_MIN
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(self.path)
                logger.info("[relay] connected to web server at {}", self.path)
                backoff = RECONNECT_MIN
                while not self._stop.is_set():
                    try:
                        event, binary = self._q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    sock.sendall(_encode(event, binary))
            except Exception as e:
                if backoff == RECONNECT_MIN:
                    logger.warning("[relay] web server unavailable ({}); retrying", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


class RelayListener:
    """Unix-socket server inside the web process; feeds the local bus."""

    def __init__(self, server, path: Optional[str] = None):
        self.server = server          # LiveServer, for liveness signalling
        self.path = path or default_socket_path()

    async def start(self) -> None:
        # A stale socket file from an unclean shutdown would block the bind.
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except OSError:
            pass
        await asyncio.start_unix_server(self._handle, path=self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        logger.info("[relay] listening for the pipeline on {}", self.path)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        logger.info("[relay] pipeline connected — going live")
        ring = BUS.ring()
        if ring and time.time() - ring[-1].get("t", 0) > STALE_RING_SEC:
            BUS.clear_ring()
            logger.info("[relay] cleared stale transcript from a previous service")
        self.server.set_live(True)
        try:
            while True:
                raw = await reader.readexactly(4)
                head = json.loads(await reader.readexactly(struct.unpack("<I", raw)[0]))
                (n,) = struct.unpack("<I", await reader.readexactly(4))
                payload = await reader.readexactly(n) if n else None
                BUS.inject(head, payload)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as e:
            logger.warning("[relay] publisher stream failed: {}", e)
        finally:
            logger.info("[relay] pipeline disconnected — going to standby")
            self.server.set_live(False)
            try:
                writer.close()
            except Exception:
                pass
