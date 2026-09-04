"""Authentication for the admin page.

The admin surface is reachable from the public internet, so it is treated as
hostile-facing:

  * passwords are stored only as scrypt hashes, never recoverable
  * comparisons are constant-time
  * sessions are opaque random tokens held server-side, so a stolen cookie
    cannot be forged offline and every session dies on restart
  * repeated failures lock an address out with escalating delay
  * the cookie is HttpOnly + SameSite=Strict, and Secure whenever the request
    arrived over HTTPS (which behind a proxy means X-Forwarded-Proto)

Credentials live outside the repository in ~/.config/translator/admin.json —
never in git, never in the image.

Stdlib only, matching the rest of the web module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

CRED_PATH = Path(os.environ.get(
    "TRANSLATOR_ADMIN_CREDENTIALS",
    Path.home() / ".config" / "translator" / "admin.json"))

SESSION_TTL = 8 * 3600        # a login lasts a service, not forever
MAX_FAILURES = 5              # per address before lockout
LOCKOUT_BASE = 30             # seconds, doubling per further failure
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


def hash_password(password: str, salt: Optional[bytes] = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)
    return {"salt": salt.hex(), "hash": dk.hex(), **{k: v for k, v in SCRYPT.items()}}


def save_credentials(username: str, password: str, path: Path = CRED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"username": username, **hash_password(password)}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.chmod(tmp, 0o600)          # readable only by this user
    tmp.replace(path)


def load_credentials(path: Path = CRED_PATH) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def verify_password(username: str, password: str, path: Path = CRED_PATH) -> bool:
    rec = load_credentials(path)
    if not rec:
        return False
    params = {k: rec.get(k, SCRYPT[k]) for k in SCRYPT}
    dk = hashlib.scrypt(password.encode("utf-8"),
                        salt=bytes.fromhex(rec["salt"]), **params)
    # Compare both fields in constant time; check the name too so a wrong
    # username costs the same as a wrong password.
    ok_user = hmac.compare_digest(username.encode(), rec["username"].encode())
    ok_pass = hmac.compare_digest(dk.hex().encode(), rec["hash"].encode())
    return ok_user and ok_pass


class Sessions:
    """Server-side session store with per-address lockout."""

    def __init__(self):
        self._sessions: dict[str, float] = {}       # token -> expiry
        self._failures: dict[str, tuple[int, float]] = {}   # addr -> (count, until)

    # -- lockout ---------------------------------------------------------
    def locked_for(self, addr: str) -> float:
        count, until = self._failures.get(addr, (0, 0.0))
        return max(0.0, until - time.time())

    def record_failure(self, addr: str) -> None:
        count, _ = self._failures.get(addr, (0, 0.0))
        count += 1
        delay = 0.0
        if count >= MAX_FAILURES:
            delay = LOCKOUT_BASE * (2 ** (count - MAX_FAILURES))
        self._failures[addr] = (count, time.time() + delay)

    def clear_failures(self, addr: str) -> None:
        self._failures.pop(addr, None)

    # -- sessions --------------------------------------------------------
    def create(self) -> str:
        self._prune()
        token = secrets.token_urlsafe(32)
        self._sessions[token] = time.time() + SESSION_TTL
        return token

    def valid(self, token: Optional[str]) -> bool:
        if not token:
            return False
        self._prune()
        return token in self._sessions

    def destroy(self, token: Optional[str]) -> None:
        if token:
            self._sessions.pop(token, None)

    def _prune(self) -> None:
        now = time.time()
        for tok in [t for t, exp in self._sessions.items() if exp <= now]:
            self._sessions.pop(tok, None)


def parse_cookies(header: str) -> dict:
    out = {}
    for part in (header or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def client_address(headers: dict, peer: str) -> str:
    """Real client address, trusting the proxy's X-Forwarded-For when present.

    Only meaningful because this service is reached through our own reverse
    proxy; the value is used for rate limiting, never for authorisation.
    """
    fwd = headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else peer
