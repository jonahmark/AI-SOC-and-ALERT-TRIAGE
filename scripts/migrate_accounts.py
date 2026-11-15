"""Inspect local accounts or explicitly import them into empty private PostgreSQL."""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth import settings as auth_settings
from src.auth.db import AuthStorageError
from src.auth.migration import MigrationError, inspect_accounts, migrate_accounts


def _private_input(variable: str, prompt: str) -> str:
    value = os.environ.get(variable, "").strip()
    if value:
        return value
    if not sys.stdin.isatty():
        raise MigrationError(f"Set {variable} in a protected environment or run interactively.")
    return getpass.getpass(prompt).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=auth_settings.AUTH_DB_PATH)
    parser.add_argument("--apply", action="store_true", help="Import into an empty private database over verified TLS.")
    args = parser.parse_args()
    try:
        counts = inspect_accounts(args.source)
        print(f"Source: {counts['users']} accounts, {counts['password_history']} password-history entries, "
              f"{counts['totp_credentials']} enabled MFA enrolments.")
        if not args.apply:
            print("Inspection only. No cloud connection or changes were made. Use --apply to import.")
            return 0
        database_url = _private_input(
            "AI_SOC_MIGRATION_DATABASE_URL", "Private PostgreSQL connection URL (hidden): ")
        destination_key = ""
        if counts["totp_credentials"]:
            print("The destination MFA key must also be configured as AI_SOC_TOTP_KEY in cloud secrets.")
            destination_key = _private_input(
                "AI_SOC_MIGRATION_TOTP_KEY", "Destination MFA encryption key (hidden): ")
        imported = migrate_accounts(
            args.source, database_url, source_totp_key=auth_settings.TOTP_KEY,
            destination_totp_key=destination_key,
        )
    except (MigrationError, AuthStorageError) as exc:
        print(f"Import could not be confirmed: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Import cancelled.", file=sys.stderr)
        return 1
    print(f"Imported and verified {imported['users']} accounts. Existing passwords were preserved.")
    print("No active sessions, recovery links, verification links, or pending MFA challenges were copied.")
    print("Set AI_SOC_AUTH_DATABASE_URL in cloud secrets to the same private destination, then verify sign-in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
