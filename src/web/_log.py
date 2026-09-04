"""Logger for the web module, which stays importable without loguru.

The web package is deliberately stdlib-only so it can run under a bare Python
(tests, the standalone server, other venvs). Loguru is used when the pipeline's
environment provides it and a compatible shim stands in when it does not.
"""
from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging as _logging

    class _BraceLogger:
        """Loguru-compatible subset: brace formatting, all the usual levels.

        Must cover every level the package uses — a missing one raises
        AttributeError deep inside a request handler, where it looks like a
        network fault rather than a typo.
        """

        def __init__(self):
            self._log = _logging.getLogger("web")

        def _fmt(self, msg, args):
            try:
                return msg.format(*args) if args else msg
            except Exception:
                return f"{msg} {args}"

        def debug(self, msg, *args):
            self._log.debug(self._fmt(msg, args))

        def info(self, msg, *args):
            self._log.info(self._fmt(msg, args))

        def warning(self, msg, *args):
            self._log.warning(self._fmt(msg, args))

        def error(self, msg, *args):
            self._log.error(self._fmt(msg, args))

        def exception(self, msg, *args):
            self._log.exception(self._fmt(msg, args))

    logger = _BraceLogger()

__all__ = ["logger"]
