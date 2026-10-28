"""Provision the initial AI-SOC administrator account (operator-run).

Usage:
    python scripts/provision_admin.py

Interactive: prompts for name, email, username and password (hidden
input, confirmation required). No credentials are hardcoded, echoed or
logged. Registration through the web UI always creates least-privileged
viewer accounts; this local script is the secure bootstrap path to the
administrator role.
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth import service  # noqa: E402
from src.auth import settings as auth_settings  # noqa: E402


def main() -> int:
    print("AI-SOC administrator provisioning")
    print(f"Account database: {auth_settings.AUTH_DB_PATH.name} "
          "(project-relative data directory)")
    print("Password policy: 12+ characters with upper, lower, number and "
          "special character.\n")
    full_name = input("Full name: ").strip()
    email = input("Email: ").strip()
    username = input("Username: ").strip()
    password = getpass.getpass("Password (hidden): ")
    confirm = getpass.getpass("Confirm password (hidden): ")
    if password != confirm:
        print("ERROR: the two passwords do not match. Nothing was created.")
        return 1
    result = service.provision_admin(
        full_name=full_name, email=email, username=username, password=password)
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(result.info)
    print(f"Created user {result.user['user_id']} "
          f"({result.user['username']}, role=administrator).")
    print("Sign in at the application URL with these credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())