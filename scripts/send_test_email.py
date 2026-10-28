"""Send a real test email through the configured SMTP server (operator-run).

Usage:
    python scripts/send_test_email.py recipient@example.com

Uses the same src.auth.mail path as verification/recovery messages, so a
success here means those emails will deliver too. Reads SMTP settings from
.env / environment. No credential is printed or logged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth import mail  # noqa: E402
from src.auth import settings as auth_settings  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or "@" not in sys.argv[1]:
        print("Usage: python scripts/send_test_email.py recipient@example.com")
        return 2
    recipient = sys.argv[1].strip()
    print(f"Email mode : {auth_settings.EMAIL_MODE}")
    print(f"SMTP server: {auth_settings.SMTP_HOST}:{auth_settings.SMTP_PORT} "
          f"(STARTTLS={auth_settings.SMTP_STARTTLS})")
    print(f"From       : {auth_settings.SMTP_FROM}")
    print(f"To         : {recipient}")
    if auth_settings.EMAIL_MODE != "smtp":
        print("ERROR: AI_SOC_EMAIL_MODE is not 'smtp'; no real email will be "
              "sent. Set it in .env and retry.")
        return 1
    result = mail.send_mail(
        recipient,
        "AI-SOC test email",
        "This is a test message from AI-SOC confirming that real SMTP "
        "delivery is configured correctly.\n\nIf you received this in your "
        "inbox, verification and password-recovery emails will deliver too.")
    print(f"Delivered  : {result.delivered}")
    print(f"Note       : {result.note}")
    return 0 if result.delivered else 1


if __name__ == "__main__":
    raise SystemExit(main())