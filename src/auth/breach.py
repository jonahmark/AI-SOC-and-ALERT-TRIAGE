"""Optional breach-password screening via the Have I Been Pwned range API.

k-anonymity model: only the first 5 hex characters of the password's
SHA-1 digest are sent; the password and the full hash never leave the
machine. Disabled by default (AI_SOC_BREACH_SCREENING). When enabled but
the service is unreachable, the check fails open and the caller records
that honestly - account operations are not blocked by an outage, and the
limitation is audited rather than hidden.
"""
from __future__ import annotations

import hashlib

import requests

from src.auth import settings as auth_settings

_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"


def enabled() -> bool:
    return bool(auth_settings.BREACH_SCREENING)


def check_password(password: str) -> tuple[str, int]:
    """(status, occurrences). status is one of:
      disabled    - screening is off; no request made
      unreachable - service could not be reached (fail-open)
      clean       - password not found in the breach corpus
      breached    - found, with the number of occurrences
    """
    if not enabled():
        return "disabled", 0
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]
    try:
        response = requests.get(
            _RANGE_URL.format(prefix=prefix),
            timeout=auth_settings.BREACH_CHECK_TIMEOUT_SECONDS,
            headers={"User-Agent": "AI-SOC-Research-Prototype"})
        response.raise_for_status()
    except requests.RequestException:
        return "unreachable", 0
    for line in response.text.splitlines():
        hash_suffix, _, count = line.partition(":")
        if hash_suffix.strip().upper() == suffix:
            try:
                return "breached", int(count.strip())
            except ValueError:
                return "breached", 1
    return "clean", 0