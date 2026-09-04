#!/usr/bin/env python3
"""Set the admin username and password for the translator web UI.

    ./scripts/set_admin_password.py            # prompts, hidden input
    ./scripts/set_admin_password.py --user bob

Stores only an scrypt hash, in ~/.config/translator/admin.json with 0600
permissions — outside the repository, so credentials are never committed.
Run it again to change the password.
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from web.auth import CRED_PATH, save_credentials, verify_password  # noqa: E402

MIN_LEN = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="admin")
    args = ap.parse_args()

    print(f"Setting admin credentials in {CRED_PATH}")
    pw = getpass.getpass("New password: ")
    if len(pw) < MIN_LEN:
        print(f"Too short — use at least {MIN_LEN} characters. This page is "
              f"reachable from the internet and controls a live service.")
        return 1
    if pw != getpass.getpass("Repeat password: "):
        print("Passwords did not match.")
        return 1

    save_credentials(args.user, pw)
    if not verify_password(args.user, pw):
        print("Stored, but verification failed — please report this.")
        return 1
    print(f"Saved. Username: {args.user}")
    print("Restart the translator service for it to take effect if it is running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
