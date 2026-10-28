"""Password hashing, token generation and server-side input validation.

Uses argon2id via argon2-cffi (no custom cryptography). Tokens are random
256-bit values; only their SHA-256 digests are persisted.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import string

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from src.auth import settings as auth_settings

_HASHER = PasswordHasher()  # argon2id, library defaults (m=64 MiB, t=3, p=4)
# Fixed dummy hash so a login attempt against an unknown account performs
# the same work as one against a real account (timing equalisation).
DUMMY_HASH = _HASHER.hash(secrets.token_urlsafe(24))

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,31}$")
_SPECIAL = set(string.punctuation)


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _HASHER.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_user_id() -> str:
    return "USR-" + secrets.token_hex(6).upper()


# --------------------------------------------------------------- validation
def valid_email(email: str) -> bool:
    return len(email) <= 254 and bool(_EMAIL_RE.match(email))


def valid_username(username: str) -> bool:
    return bool(_USERNAME_RE.match(username))


def password_policy_errors(password: str) -> list[str]:
    errors = []
    if len(password) < auth_settings.MIN_PASSWORD_LENGTH:
        errors.append(
            f"Password must be at least {auth_settings.MIN_PASSWORD_LENGTH} "
            "characters long.")
    if len(password) > auth_settings.MAX_PASSWORD_LENGTH:
        errors.append(
            f"Password must be at most {auth_settings.MAX_PASSWORD_LENGTH} "
            "characters long.")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter.")
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter.")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number.")
    if not any(c in _SPECIAL for c in password):
        errors.append("Password must contain at least one special character.")
    return errors


def password_checks(password: str) -> list[tuple[str, bool]]:
    """The five policy requirements, evaluated for the live strength UI."""
    return [
        (f"At least {auth_settings.MIN_PASSWORD_LENGTH} characters",
         len(password) >= auth_settings.MIN_PASSWORD_LENGTH),
        ("One uppercase letter", any(c.isupper() for c in password)),
        ("One lowercase letter", any(c.islower() for c in password)),
        ("One number", any(c.isdigit() for c in password)),
        ("One special character", any(c in _SPECIAL for c in password)),
    ]


def password_strength(password: str) -> tuple[int, str]:
    """(score 0-4, label) - a transparent heuristic, not a security claim."""
    if not password:
        return 0, "Enter a password"
    checks = password_checks(password)
    score = sum(met for _, met in checks[1:])  # four character classes
    if len(password) >= 16:
        score += 1
    if len(password) < auth_settings.MIN_PASSWORD_LENGTH:
        score = min(score, 1)
    score = max(0, min(4, score))
    label = ["Very weak", "Weak", "Fair", "Good", "Strong"][score]
    return score, label


def clean_text(value: str, max_length: int) -> str:
    """Trim, cap length and strip control characters from free text."""
    value = "".join(c for c in str(value) if c.isprintable()).strip()
    return value[:max_length]