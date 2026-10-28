"""Auth configuration, read from environment variables.

A project-root `.env` file (KEY=VALUE lines, # comments) is loaded as
defaults only - real environment variables always win. No secret has a
usable default: SMTP credentials must be supplied by the operator.
"""
from __future__ import annotations

import os
from pathlib import Path

import config


def _load_dotenv_defaults() -> None:
    path = config.PROJECT_ROOT / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_defaults()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Storage. Tests point AI_SOC_AUTH_DB at a temporary file.
AUTH_DB_PATH = Path(os.environ.get(
    "AI_SOC_AUTH_DB", str(config.DATA_PROCESSED_DIR / "auth.db")))

# Sessions
SESSION_TTL_HOURS = _env_int("AI_SOC_SESSION_TTL_HOURS", 12)
REMEMBER_ME_DAYS = _env_int("AI_SOC_REMEMBER_ME_DAYS", 14)
# Inactivity expiry, enforced server-side on every session validation in
# addition to the absolute TTL. Applies to remember-me sessions too.
# 0 disables the idle check (absolute TTL still applies).
SESSION_IDLE_TIMEOUT_MINUTES = _env_int("AI_SOC_SESSION_IDLE_MINUTES", 60)
SESSION_COOKIE = "soc_session"

# Single-use tokens
RESET_TOKEN_TTL_MINUTES = _env_int("AI_SOC_RESET_TOKEN_TTL_MINUTES", 30)
VERIFY_TOKEN_TTL_HOURS = _env_int("AI_SOC_VERIFY_TOKEN_TTL_HOURS", 24)
VERIFY_RESEND_COOLDOWN_SECONDS = 120
RESET_REQUESTS_PER_WINDOW = 3

# Brute-force protection (per identifier, sliding window + lockout)
MAX_FAILED_ATTEMPTS = _env_int("AI_SOC_MAX_FAILED_ATTEMPTS", 5)
ATTEMPT_WINDOW_MINUTES = 15
LOCKOUT_MINUTES = 15
MAX_REGISTRATIONS_PER_HOUR = 20

# Password policy
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 256
# Number of recent passwords (including the current one) that cannot be
# reused on password change or recovery reset.
PASSWORD_HISTORY_SIZE = _env_int("AI_SOC_PASSWORD_HISTORY_SIZE", 5)

# Two-factor authentication (TOTP, RFC 6238 via pyotp). The shared secret is
# encrypted at rest with Fernet (cryptography library); the key must be
# supplied by the operator - generate one with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Without a valid key, enrolment is unavailable and the UI says so honestly.
TOTP_KEY = os.environ.get("AI_SOC_TOTP_KEY", "").strip()
TOTP_ISSUER = "AI-SOC"
TOTP_CHALLENGE_TTL_MINUTES = _env_int("AI_SOC_TOTP_CHALLENGE_TTL_MINUTES", 5)
TOTP_MAX_ATTEMPTS = _env_int("AI_SOC_TOTP_MAX_ATTEMPTS", 5)

# Breach-password screening (HIBP k-anonymity range API). Off by default;
# when on, only a 5-character SHA-1 prefix is ever transmitted. Outages
# fail open and are audited, never hidden.
BREACH_SCREENING = os.environ.get(
    "AI_SOC_BREACH_SCREENING", "0").strip().lower() not in ("", "0", "false", "off")
BREACH_CHECK_TIMEOUT_SECONDS = _env_int("AI_SOC_BREACH_CHECK_TIMEOUT", 5)

# Email delivery: off | outbox | smtp.
# "outbox" is an explicit development mode: messages are written as .eml
# files to results/mail_outbox/ instead of being sent. It is labelled as
# such everywhere in the UI - it is never presented as real delivery.
EMAIL_MODE = os.environ.get("AI_SOC_EMAIL_MODE", "outbox").lower()
MAIL_OUTBOX_DIR = config.RESULTS_DIR / "mail_outbox"
APP_URL = os.environ.get("AI_SOC_APP_URL", "http://localhost:8501").rstrip("/")

# Transactional email provider presets. Set AI_SOC_EMAIL_PROVIDER to one of
# these and the host/port/STARTTLS are filled in for you; you still supply the
# account credentials (SMTP_USER / SMTP_PASSWORD) and a verified sender
# (SMTP_FROM). An explicit AI_SOC_SMTP_HOST / PORT / STARTTLS always overrides
# the preset, so region-specific servers (e.g. Amazon SES) can be set by hand.
SMTP_PROVIDER = os.environ.get("AI_SOC_EMAIL_PROVIDER", "").strip().lower()

# name -> (host, port, starttls). SES hosts are region-specific; the preset
# uses us-east-1 and must be overridden with AI_SOC_SMTP_HOST for other regions.
_SMTP_PRESETS: dict[str, tuple[str, int, bool]] = {
    "sendgrid": ("smtp.sendgrid.net", 587, True),
    "brevo": ("smtp-relay.brevo.com", 587, True),
    "mailgun": ("smtp.mailgun.org", 587, True),
    "postmark": ("smtp.postmarkapp.com", 587, True),
    "ses": ("email-smtp.us-east-1.amazonaws.com", 587, True),
    "gmail": ("smtp.gmail.com", 587, True),
}
_preset = _SMTP_PRESETS.get(SMTP_PROVIDER, ("", 587, True))


def smtp_preset(provider: str) -> tuple[str, int, bool]:
    """Host/port/STARTTLS for a known provider; empty host when unknown."""
    return _SMTP_PRESETS.get((provider or "").strip().lower(), ("", 587, True))

SMTP_HOST = os.environ.get("AI_SOC_SMTP_HOST", _preset[0])
SMTP_PORT = _env_int("AI_SOC_SMTP_PORT", _preset[1])
SMTP_USER = os.environ.get("AI_SOC_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("AI_SOC_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("AI_SOC_SMTP_FROM", "AI-SOC <no-reply@localhost>")
SMTP_STARTTLS = os.environ.get(
    "AI_SOC_SMTP_STARTTLS", "1" if _preset[2] else "0") not in ("0", "false")