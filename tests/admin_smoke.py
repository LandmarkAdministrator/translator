#!/usr/bin/env python3
"""Security smoke test for the public admin surface.

Checks the things that matter when a control panel faces the internet:
unauthenticated access is refused, wrong credentials are rejected, a real
login works, the session cookie is defensively flagged, logout revokes, and
repeated failures lock the address out.

Runs entirely on 127.0.0.1 with a throwaway credentials file.
"""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

PORT = 8897
tmp = Path(tempfile.mkdtemp()) / "admin.json"
os.environ["TRANSLATOR_ADMIN_CREDENTIALS"] = str(tmp)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from web import auth              # noqa: E402
from web.live_server import LiveServer   # noqa: E402

auth.CRED_PATH = tmp
import web.admin                  # noqa: E402,F401

USER, PW = "admin", "a-long-enough-password"
auth.save_credentials(USER, PW, tmp)


def req(method, path, body=None, cookie=None, ctype="application/x-www-form-urlencoded"):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=8)
    headers = {}
    if body is not None:
        headers["Content-Type"] = ctype
    if cookie:
        headers["Cookie"] = cookie
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse()
    data = r.read()
    out = (r.status, dict(r.getheaders()), data)
    c.close()
    return out


def main():
    threading.Thread(target=LiveServer(PORT, host="127.0.0.1").run_forever,
                     daemon=True).start()
    time.sleep(0.6)

    status, _, body = req("GET", "/admin")
    assert status == 200 and b"Sign in" in body, "login page not shown"
    print("unauthenticated /admin shows login: OK")

    status, _, body = req("GET", "/admin/api/status")
    assert status == 401, f"status API not protected (got {status})"
    print("unauthenticated API refused: OK")

    status, _, _ = req("POST", "/admin/api/action", body=json.dumps({"action": "stop"}),
                       ctype="application/json")
    assert status == 401, f"action API not protected (got {status})"
    print("unauthenticated action refused: OK")

    status, _, body = req("POST", "/admin/login", body=f"username={USER}&password=wrong")
    assert status == 401, f"bad password accepted (got {status})"
    print("wrong password rejected: OK")

    status, hdrs, _ = req("POST", "/admin/login", body=f"username={USER}&password={PW}")
    assert status == 303, f"valid login failed (got {status})"
    setc = hdrs.get("Set-Cookie", "")
    assert "HttpOnly" in setc and "SameSite=Strict" in setc and "Path=/admin" in setc, setc
    cookie = setc.split(";")[0]
    print(f"valid login sets defensive cookie: OK ({setc.split(';',1)[1].strip()[:46]}…)")

    status, _, body = req("GET", "/admin/api/status", cookie=cookie)
    assert status == 200, f"status denied to session (got {status})"
    st = json.loads(body)
    assert "service" in st and "manual_override" in st, st
    print(f"authenticated status: OK (service={st['service']})")

    status, _, body = req("GET", "/admin", cookie=cookie)
    assert status == 200 and b"Translation Admin" in body and b"Sign in" not in body
    print("authenticated admin page: OK")

    status, hdrs, _ = req("GET", "/admin/logout", cookie=cookie)
    assert status == 303 and "Max-Age=0" in hdrs.get("Set-Cookie", "")
    status, _, _ = req("GET", "/admin/api/status", cookie=cookie)
    assert status == 401, "session survived logout"
    print("logout revokes session: OK")

    for _ in range(6):
        req("POST", "/admin/login", body=f"username={USER}&password=nope")
    status, _, body = req("POST", "/admin/login", body=f"username={USER}&password={PW}")
    assert status == 429, f"no lockout after repeated failures (got {status})"
    print("brute-force lockout engages, even for the CORRECT password: OK")

    print("\nALL ADMIN SECURITY TESTS PASSED")


if __name__ == "__main__":
    main()
