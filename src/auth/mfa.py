"""TOTP two-factor authentication primitives (RFC 6238 via pyotp).

No custom cryptography: codes come from pyotp, the shared secret is
encrypted at rest with Fernet (cryptography library), and QR images are
rendered by qrcode. Without a valid operator key (AI_SOC_TOTP_KEY) MFA is
reported as unavailable - never silently degraded to plaintext storage.
"""
from __future__ import annotations

import io

import pyotp
import qrcode
from cryptography.fernet import Fernet, InvalidToken

from src.auth import settings as auth_settings


def _fernet() -> Fernet | None:
    key = auth_settings.TOTP_KEY
    if not key:
        return None
    try:
        return Fernet(key.encode("utf-8"))
    except (ValueError, TypeError):
        return None


def available() -> bool:
    return _fernet() is not None


def new_secret() -> str:
    return pyotp.random_base32()


def encrypt_secret(secret: str) -> str:
    fernet = _fernet()
    if fernet is None:
        raise RuntimeError("TOTP encryption key is not configured")
    return fernet.encrypt(secret.encode("utf-8")).decode("utf-8")


def decrypt_secret(blob: str) -> str | None:
    """None when the key is missing or the ciphertext was not produced by
    it (e.g. after key rotation); admin removal is then the recovery path."""
    fernet = _fernet()
    if fernet is None:
        return None
    try:
        return fernet.decrypt(blob.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None


def provisioning_uri(secret: str, account_email: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(
        name=account_email, issuer_name=auth_settings.TOTP_ISSUER)


def qr_png(uri: str) -> bytes:
    buf = io.BytesIO()
    qrcode.make(uri).save(buf, format="PNG")
    return buf.getvalue()


def verify_code(secret: str, code: str) -> bool:
    code = (code or "").strip().replace(" ", "")
    if len(code) != 6 or not code.isdigit():
        return False
    # One 30-second step of tolerance in each direction for clock drift.
    return pyotp.TOTP(secret).verify(code, valid_window=1)