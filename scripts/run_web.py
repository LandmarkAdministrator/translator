#!/usr/bin/env python3
"""Always-on web server for the live translation page and admin panel.

Runs independently of translate.service so the page stays reachable outside
service windows — congregants arriving early get a standby notice instead of a
refused connection, and the admin panel can start a stopped service.

Environment (same names the pipeline launcher uses):
    WEB_PORT        plain HTTP port, bound to WEB_HOST (loopback by default)
    WEB_HOST        default 127.0.0.1
    WEB_TLS_PORT    HTTPS + WSS on all interfaces, for the reverse proxy
    WEB_TLS_CERT / WEB_TLS_KEY      internal-CA certificate and key
    WEB_RELAY_SOCKET                Unix socket the pipeline publishes to
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    from utils.logger import setup_logger
    setup_logger(log_dir="logs", log_level=os.environ.get("WEB_LOG_LEVEL", "INFO"))

    from web.live_server import LiveServer
    from web.relay import default_socket_path

    web_port = os.environ.get("WEB_PORT", "").strip()
    tls_port = os.environ.get("WEB_TLS_PORT", "").strip()
    if not web_port and not tls_port:
        print("Set WEB_PORT and/or WEB_TLS_PORT.", file=sys.stderr)
        return 1

    LiveServer(
        port=int(web_port) if web_port else None,
        host=os.environ.get("WEB_HOST", "127.0.0.1"),
        tls_port=int(tls_port) if tls_port else None,
        certfile=os.environ.get("WEB_TLS_CERT") or None,
        keyfile=os.environ.get("WEB_TLS_KEY") or None,
        relay_path=os.environ.get("WEB_RELAY_SOCKET") or default_socket_path(),
    ).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
