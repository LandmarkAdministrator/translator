"""Live-translation web server: static page + WebSocket event stream.

Runs an asyncio loop on a daemon thread inside the translator process.
Serves:
  GET /      -> src/web/static/index.html
  GET /ws    -> WebSocket: JSON text frames for text events; binary frames
                ([u32 len][JSON header][WAV]) for per-sentence audio.

On connect a client receives the recent text ring (catch-up), then live
events. Slow clients drop audio before text and are disconnected only if
they stop reading entirely. Stdlib only — see ws.py.

Enable from the pipeline with:  start_in_thread(port)   (WEB_PORT env in
run.py). With no server running, publishing to the bus is a no-op.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
except ImportError:  # keep the web module stdlib-only (tests, other venvs)
    import logging as _logging

    class _BraceLogger:
        def __init__(self):
            self._log = _logging.getLogger("web")

        def _fmt(self, msg, args):
            return msg.format(*args) if args else msg

        def info(self, msg, *args):
            self._log.info(self._fmt(msg, args))

        def error(self, msg, *args):
            self._log.error(self._fmt(msg, args))

    logger = _BraceLogger()

from web import ws as wsproto
from web.bus import BUS, frame_binary

STATIC_DIR = Path(__file__).parent / "static"
QUEUE_MAX = 300
AUDIO_DROP_ABOVE = 150  # if a client is this far behind, stop sending it audio
PING_INTERVAL = 20.0


class _Client:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self.alive = True

    def offer(self, frame: bytes, is_audio: bool) -> None:
        if not self.alive:
            return
        if is_audio and self.queue.qsize() > AUDIO_DROP_ABOVE:
            return
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # drop; text ring lets the page resync visually


class LiveServer:
    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: set[_Client] = set()

    # -- bus sink (called from pipeline threads) --------------------------
    def _sink(self, event: dict, binary: Optional[bytes]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if binary is None:
            frame = wsproto.text_frame(json.dumps(event))
            is_audio = False
        else:
            frame = wsproto.binary_frame(frame_binary(event, binary))
            is_audio = True
        loop.call_soon_threadsafe(self._broadcast, frame, is_audio)

    def _broadcast(self, frame: bytes, is_audio: bool) -> None:
        for client in list(self._clients):
            client.offer(frame, is_audio)

    # -- connection handling ----------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
        except Exception:
            writer.close()
            return
        head = request.decode("latin-1", "replace")
        line = head.split("\r\n", 1)[0]
        path = line.split(" ")[1] if len(line.split(" ")) > 1 else "/"
        headers = {}
        for h in head.split("\r\n")[1:]:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        if path.split("?")[0] == "/ws" and "sec-websocket-key" in headers:
            await self._serve_ws(reader, writer, headers["sec-websocket-key"])
            return
        await self._serve_http(writer, path)

    async def _serve_http(self, writer: asyncio.StreamWriter, path: str):
        clean = path.split("?")[0]
        # Static assets for the installable app. Paths are resolved and checked
        # to stay inside STATIC_DIR so a crafted request cannot escape it.
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                 ".png": "image/png", ".webmanifest": "application/manifest+json",
                 ".json": "application/json", ".css": "text/css"}
        extra = ""
        if clean in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            ctype, status = types[".html"], "200 OK"
        else:
            candidate = (STATIC_DIR / clean.lstrip("/")).resolve()
            if (candidate.is_file() and STATIC_DIR.resolve() in candidate.parents
                    and candidate.suffix in types):
                body = candidate.read_bytes()
                ctype, status = types[candidate.suffix], "200 OK"
                if candidate.suffix == ".png":
                    extra = "Cache-Control: public, max-age=604800\r\n"
                elif candidate.name == "sw.js":
                    # Never cache the worker itself, or updates cannot land.
                    extra = "Cache-Control: no-cache\r\n"
            else:
                body, ctype, status = b"not found", "text/plain", "404 Not Found"
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n{extra}"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
        )
        try:
            await writer.drain()
        except Exception:
            pass
        writer.close()

    async def _serve_ws(self, reader, writer, key: str):
        writer.write(wsproto.handshake_response(key))
        await writer.drain()
        client = _Client(writer)
        self._clients.add(client)
        logger.info("[web] client connected ({} total)", len(self._clients))
        # catch-up: recent text events so the transcript isn't empty
        for event in BUS.ring():
            client.offer(wsproto.text_frame(json.dumps({**event, "replay": True})), False)

        async def sender():
            while client.alive:
                frame = await client.queue.get()
                writer.write(frame)
                await writer.drain()

        async def receiver():
            while client.alive:
                got = await wsproto.read_frame(reader)
                if got is None:
                    return
                opcode, payload = got
                if opcode == wsproto.OP_CLOSE:
                    return
                if opcode == wsproto.OP_PING:
                    client.offer(wsproto.encode_frame(wsproto.OP_PONG, payload), False)

        async def pinger():
            while client.alive:
                await asyncio.sleep(PING_INTERVAL)
                client.offer(wsproto.encode_frame(wsproto.OP_PING, b"hb"), False)

        tasks = [asyncio.ensure_future(c()) for c in (sender, receiver, pinger)]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            client.alive = False
            for t in tasks:
                t.cancel()
            self._clients.discard(client)
            try:
                writer.write(wsproto.close_frame())
                writer.close()
            except Exception:
                pass
            logger.info("[web] client disconnected ({} left)", len(self._clients))

    # -- lifecycle ---------------------------------------------------------
    async def _main(self):
        server = await asyncio.start_server(self._handle, self.host, self.port)
        logger.info("[web] live page on http://{}:{}/  (ws on /ws)", self.host, self.port)
        async with server:
            await server.serve_forever()

    def run_forever(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        BUS.add_sink(self._sink)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error("[web] server stopped: {}", e)
        finally:
            BUS.remove_sink(self._sink)


def start_in_thread(port: int) -> threading.Thread:
    server = LiveServer(port)
    thread = threading.Thread(target=server.run_forever, name="web-live", daemon=True)
    thread.start()
    return thread
