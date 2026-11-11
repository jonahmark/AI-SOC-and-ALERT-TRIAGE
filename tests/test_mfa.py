"""TOTP two-factor authentication tests: enrolment, the two-stage login,
challenge hardening, disable/recovery paths and secret protection - all
against the real SQLite store with real pyotp codes and Fernet encryption.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pyotp
import pytest
from cryptography.fernet import Fernet

from src.auth import db, mfa, rbac, security, service
from src.auth import settings as auth_settings
from tests.conftest import ADMIN_PASSWORD, TEST_PASSWORD, latest_mail_link_token


def _register(email="analyst@example.test", username="analyst"):
    result = service.register(
        full_name="Test Analyst", email=email, username=username,
        password=TEST_PASSWORD, confirm_password=TEST_PASSWORD)
    assert result.ok, result.errors
    return result.user


def _enroll(user_id: str) -> str:
    """Complete a real enrolment and return the shared secret."""
    result = service.begin_totp_enrollment(user_id)
    assert result.ok, result.errors
    secret = result.payload["secret"]
    confirm = service.confirm_totp_enrollment(user_id, pyotp.TOTP(secret).now())
    assert confirm.ok, confirm.errors
    return secret


def _code(secret: str) -> str:
    return pyotp.TOTP(secret).now()


# -------------------------------------------------------------- primitives
def test_mfa_unavailable_without_a_valid_key(monkeypatch):
    user = _register()
    for key in ("", "not-a-fernet-key"):
        monkeypatch.setattr(auth_settings, "TOTP_KEY", key)
        assert mfa.available() is False
        result = service.begin_totp_enrollment(user["user_id"])
        assert not result.ok
        assert any("AI_SOC_TOTP_KEY" in e for e in result.errors)


def test_secret_roundtrips_through_fernet_and_rejects_other_keys(monkeypatch):
    secret = mfa.new_secret()
    blob = mfa.encrypt_secret(secret)
    assert blob != secret and secret not in blob
    assert mfa.decrypt_secret(blob) == secret
    monkeypatch.setattr(auth_settings, "TOTP_KEY",
                        Fernet.generate_key().decode("utf-8"))
    assert mfa.decrypt_secret(blob) is None  # rotated/foreign key


def test_verify_code_rejects_malformed_input():
    secret = mfa.new_secret()
    for bad in ("", "12345", "1234567", "abcdef", None):
        assert mfa.verify_code(secret, bad) is False
    assert mfa.verify_code(secret, _code(secret)) is True
    assert mfa.verify_code(secret, "000000" if _code(secret) != "000000"
                           else "111111") is False


def test_qr_and_uri_are_real_setup_material():
    secret = mfa.new_secret()
    uri = mfa.provisioning_uri(secret, "analyst@example.test")
    assert uri.startswith("otpauth://totp/AI-SOC:")
    assert f"secret={secret}" in uri and "issuer=AI-SOC" in uri
    png = mfa.qr_png(uri)
    assert png.startswith(b"\x89PNG") and len(png) > 500


# --------------------------------------------------------------- enrolment
def test_enrolment_stores_only_ciphertext_and_needs_confirmation():
    user = _register()
    result = service.begin_totp_enrollment(user["user_id"])
    assert result.ok, result.errors
    secret = result.payload["secret"]
    row = db.get_store().query_one(
        "SELECT * FROM totp_credentials WHERE user_id = ?", (user["user_id"],))
    assert row["enabled"] == 0 and row["enrolled_at"] is None
    assert row["secret_encrypted"] != secret
    assert secret not in row["secret_encrypted"]
    assert mfa.decrypt_secret(row["secret_encrypted"]) == secret
    # Pending enrolment does not affect login yet:
    assert service.authenticate("analyst", TEST_PASSWORD).token


def test_confirmation_requires_the_matching_code():
    user = _register()
    payload = service.begin_totp_enrollment(user["user_id"]).payload
    wrong = service.confirm_totp_enrollment(user["user_id"], "000000")
    assert not wrong.ok
    assert db.get_store().query_one(
        "SELECT enabled FROM totp_credentials")["enabled"] == 0
    assert service.confirm_totp_enrollment(
        user["user_id"], _code(payload["secret"])).ok
    row = db.get_store().query_one("SELECT * FROM totp_credentials")
    assert row["enabled"] == 1 and row["enrolled_at"]


def test_confirmation_without_pending_enrolment_fails():
    user = _register()
    assert not service.confirm_totp_enrollment(user["user_id"], "123456").ok


def test_reenrol_requires_disabling_first_and_cancel_cleans_up():
    user = _register()
    secret = _enroll(user["user_id"])
    again = service.begin_totp_enrollment(user["user_id"])
    assert not again.ok and any("already enabled" in e for e in again.errors)
    assert service.disable_totp(user["user_id"], _code(secret)).ok
    pending = service.begin_totp_enrollment(user["user_id"])
    assert pending.ok
    service.cancel_totp_enrollment(user["user_id"])
    assert db.get_store().query("SELECT * FROM totp_credentials") == []
    assert service.authenticate("analyst", TEST_PASSWORD).token


# ----------------------------------------------------------- two-stage login
def test_login_with_mfa_issues_a_challenge_not_a_session():
    user = _register()
    secret = _enroll(user["user_id"])
    result = service.authenticate("analyst", TEST_PASSWORD)
    assert result.ok and result.mfa_token and result.token is None
    assert db.get_store().query("SELECT * FROM sessions") == []
    done = service.complete_totp_login(result.mfa_token, _code(secret))
    assert done.ok, done.errors
    assert done.token and service.validate_session(done.token) is not None
    events = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'login_success'")
    assert events and "mfa=totp" in events[0]["detail"]


def test_wrong_code_keeps_the_challenge_but_no_session():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    bad = service.complete_totp_login(challenge, "999999"
                                      if _code(secret) != "999999"
                                      else "888888")
    assert not bad.ok and bad.token is None
    assert db.get_store().query("SELECT * FROM sessions") == []
    assert service.complete_totp_login(challenge, _code(secret)).ok


def test_challenge_is_single_use():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    assert service.complete_totp_login(challenge, _code(secret)).ok
    replay = service.complete_totp_login(challenge, _code(secret))
    assert not replay.ok


def test_expired_challenge_is_rejected_and_removed():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(
        timespec="seconds")
    db.get_store().execute(
        "UPDATE totp_challenges SET expires_at = ? WHERE token_hash = ?",
        (past, security.token_digest(challenge)))
    assert not service.complete_totp_login(challenge, _code(secret)).ok
    assert db.get_store().query("SELECT * FROM totp_challenges") == []


def test_challenge_attempt_cap(monkeypatch):
    monkeypatch.setattr(auth_settings, "TOTP_MAX_ATTEMPTS", 2)
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    wrong = "999999" if _code(secret) != "999999" else "888888"
    assert not service.complete_totp_login(challenge, wrong).ok
    assert not service.complete_totp_login(challenge, wrong).ok
    capped = service.complete_totp_login(challenge, _code(secret))
    assert not capped.ok
    assert any("too many incorrect codes" in e.lower() for e in capped.errors)
    assert db.get_store().query("SELECT * FROM totp_challenges") == []


def test_wrong_codes_count_towards_login_lockout(monkeypatch):
    monkeypatch.setattr(auth_settings, "MAX_FAILED_ATTEMPTS", 2)
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    wrong = "999999" if _code(secret) != "999999" else "888888"
    for _ in range(2):
        assert not service.complete_totp_login(challenge, wrong).ok
    locked = service.complete_totp_login(challenge, _code(secret))
    assert not locked.ok
    assert any("too many failed attempts" in e.lower() for e in locked.errors)
    assert not service.authenticate("analyst", TEST_PASSWORD).ok


def test_password_change_invalidates_a_pending_challenge():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    new_password = "N3w!Passphrase-Long"
    assert service.change_password(
        user["user_id"], TEST_PASSWORD, new_password, new_password).ok
    assert not service.complete_totp_login(challenge, _code(secret)).ok


def test_remember_choice_carries_through_the_challenge():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate(
        "analyst", TEST_PASSWORD, remember=True).mfa_token
    done = service.complete_totp_login(challenge, _code(secret))
    assert done.ok
    row = db.get_store().query_one(
        "SELECT * FROM sessions WHERE session_id = ?",
        (security.token_digest(done.token),))
    assert row["remember"] == 1


def test_invalid_challenge_tokens_are_rejected():
    for bad in (None, "", 123, "no-such-token"):
        assert not service.complete_totp_login(bad, "123456").ok


# ------------------------------------------------------- disable + recovery
def test_self_disable_requires_a_current_code():
    user = _register()
    secret = _enroll(user["user_id"])
    wrong = "999999" if _code(secret) != "999999" else "888888"
    assert not service.disable_totp(user["user_id"], wrong).ok
    assert service.totp_enabled(user["user_id"])
    assert service.disable_totp(user["user_id"], _code(secret)).ok
    assert not service.totp_enabled(user["user_id"])
    assert service.authenticate("analyst", TEST_PASSWORD).token


def test_admin_removal_is_permission_checked_and_audited(admin_user):
    user = _register()
    _enroll(user["user_id"])
    with pytest.raises(rbac.PermissionDenied):
        service.admin_disable_totp({"role": "viewer"}, user["user_id"])
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    result = service.admin_disable_totp(admin, user["user_id"])
    assert result.ok, result.errors
    assert service.authenticate("analyst", TEST_PASSWORD).token
    events = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'totp_disabled_by_admin'")
    assert len(events) == 1
    assert events[0]["actor_id"] == admin["user_id"]
    assert events[0]["user_id"] == user["user_id"]


def test_password_recovery_does_not_remove_the_second_factor():
    user = _register()
    secret = _enroll(user["user_id"])
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    new_password = "R3set!Passphrase-New"
    assert service.reset_password(token, new_password, new_password).ok
    result = service.authenticate("analyst", new_password)
    assert result.mfa_token and result.token is None
    assert service.complete_totp_login(result.mfa_token, _code(secret)).ok


def test_mfa_secrets_never_reach_the_audit_trail():
    user = _register()
    secret = _enroll(user["user_id"])
    challenge = service.authenticate("analyst", TEST_PASSWORD).mfa_token
    code = _code(secret)
    service.complete_totp_login(challenge, code)
    service.disable_totp(user["user_id"], _code(secret))
    for event in db.get_store().query("SELECT * FROM audit_events"):
        blob = f"{event['event']} {event['detail']}"
        assert secret not in blob and code not in blob
        assert challenge not in blob