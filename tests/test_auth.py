"""Authentication service tests: registration, login, sessions, recovery,
verification, RBAC and audit - all against the real SQLite store, argon2id
hasher and token machinery, redirected to a disposable per-test database.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from src.auth import db, mail, rbac, security, service
from src.auth import settings as auth_settings
from tests.conftest import ADMIN_PASSWORD, TEST_PASSWORD, latest_mail_link_token


def _register(email="analyst@example.test", username="analyst",
              password=TEST_PASSWORD, **overrides) -> service.AuthResult:
    fields = dict(full_name="Test Analyst", email=email, username=username,
                  password=password, confirm_password=password)
    fields.update(overrides)
    return service.register(**fields)


def _expire(table: str, token: str) -> None:
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
        timespec="seconds")
    db.get_store().execute(
        f"UPDATE {table} SET expires_at = ? WHERE token_hash = ?",
        (past, security.token_digest(token)))


# ------------------------------------------------------------- registration
def test_registration_creates_least_privileged_account():
    result = _register()
    assert result.ok, result.errors
    assert result.user["role"] == "viewer"
    assert result.user["email_verified"] is False
    assert "password_hash" not in result.user
    row = db.get_store().query_one(
        "SELECT * FROM users WHERE user_id = ?", (result.user["user_id"],))
    assert row["password_hash"].startswith("$argon2id$")
    assert TEST_PASSWORD not in row["password_hash"]
    assert security.verify_password(row["password_hash"], TEST_PASSWORD)


def test_registration_cannot_accept_a_role():
    assert "role" not in inspect.signature(service.register).parameters
    with pytest.raises(TypeError):
        service.register(full_name="Escalation", email="esc@example.test",
                         username="escalator", password=TEST_PASSWORD,
                         confirm_password=TEST_PASSWORD, role="administrator")
    assert db.get_store().query("SELECT * FROM users") == []


def test_duplicate_email_is_case_insensitive():
    assert _register().ok
    result = _register(email="ANALYST@example.test", username="other")
    assert not result.ok
    assert any("email" in e.lower() for e in result.errors)


def test_duplicate_username_is_case_insensitive():
    assert _register().ok
    result = _register(email="different@example.test", username="ANALYST")
    assert not result.ok
    assert any("username" in e.lower() for e in result.errors)


@pytest.mark.parametrize("overrides", [
    {"email": "not-an-email"},
    {"username": "x"},
    {"username": "-leading-hyphen"},
    {"password": "short", "confirm_password": "short"},
    {"password": "alllowercase123!", "confirm_password": "alllowercase123!"},
    {"confirm_password": "Something-Else-9!"},
    {"full_name": "   "},
])
def test_invalid_registration_is_rejected_and_creates_nothing(overrides):
    result = _register(**overrides)
    assert not result.ok
    assert result.errors
    assert db.get_store().query("SELECT * FROM users") == []


def test_registration_rate_limit(monkeypatch):
    monkeypatch.setattr(auth_settings, "MAX_REGISTRATIONS_PER_HOUR", 1)
    assert _register().ok
    result = _register(email="second@example.test", username="second")
    assert not result.ok
    assert any("too many" in e.lower() for e in result.errors)


def test_verification_link_is_emailed_and_token_stored_only_as_digest():
    result = _register()
    assert result.ok
    token = latest_mail_link_token()
    assert token, "development outbox must contain the real message"
    row = db.get_store().query_one("SELECT * FROM verify_tokens")
    assert row["token_hash"] == security.token_digest(token)
    assert row["token_hash"] != token
    files = list(auth_settings.MAIL_OUTBOX_DIR.glob("*.eml"))
    assert len(files) == 1
    # The raw token legitimately appears in the emailed link, but never in
    # the database - only its SHA-256 digest is stored:
    dump = str(db.get_store().query("SELECT * FROM verify_tokens"))
    assert token not in dump


def test_outbox_mode_is_labelled_honestly_not_as_real_delivery():
    result = _register()
    assert result.ok
    assert "not configured for real" in result.info
    assert "outbox" in result.info


def test_off_mode_documents_the_limitation(monkeypatch):
    monkeypatch.setattr(auth_settings, "EMAIL_MODE", "off")
    result = _register()
    assert result.ok
    assert "could not be sent" in result.info
    assert not list(auth_settings.MAIL_OUTBOX_DIR.glob("*.eml"))


# -------------------------------------------------------------------- login
def test_login_works_with_email_and_username():
    assert _register().ok
    for identifier in ("analyst@example.test", "analyst", "ANALYST"):
        result = service.authenticate(identifier, TEST_PASSWORD)
        assert result.ok, result.errors
        assert result.token
        user = service.validate_session(result.token)
        assert user and user["username"] == "analyst"


def test_wrong_password_and_unknown_account_give_identical_generic_error():
    assert _register().ok
    wrong = service.authenticate("analyst", "Wr0ng!Password-XX")
    unknown = service.authenticate("ghost@example.test", "Wr0ng!Password-XX")
    assert not wrong.ok and not unknown.ok
    assert wrong.errors == unknown.errors == [
        "Invalid email/username or password."]


def test_empty_credentials_are_rejected():
    result = service.authenticate("", "")
    assert not result.ok and result.errors


def test_disabled_account_gets_the_same_generic_error(admin_user):
    assert _register().ok
    target = db.get_store().query_one(
        "SELECT * FROM users WHERE username = 'analyst'")
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    assert service.set_account_status(admin, target["user_id"], "disabled").ok
    result = service.authenticate("analyst", TEST_PASSWORD)
    assert not result.ok
    assert result.errors == ["Invalid email/username or password."]


def test_lockout_after_repeated_failures_blocks_even_correct_password(
        monkeypatch):
    monkeypatch.setattr(auth_settings, "MAX_FAILED_ATTEMPTS", 3)
    assert _register().ok
    for _ in range(3):
        assert not service.authenticate("analyst", "Wr0ng!Password-XX").ok
    locked = service.authenticate("analyst", TEST_PASSWORD)
    assert not locked.ok
    assert any("too many failed attempts" in e.lower() for e in locked.errors)


def test_lockout_applies_to_unknown_identifiers_without_revealing_existence(
        monkeypatch):
    monkeypatch.setattr(auth_settings, "MAX_FAILED_ATTEMPTS", 2)
    for _ in range(2):
        service.authenticate("nobody@example.test", "Wr0ng!Password-XX")
    locked = service.authenticate("nobody@example.test", "Wr0ng!Password-XX")
    assert not locked.ok
    assert any("too many failed attempts" in e.lower() for e in locked.errors)


def test_remember_me_session_lives_longer():
    assert _register().ok
    short = service.authenticate("analyst", TEST_PASSWORD, remember=False)
    long = service.authenticate("analyst", TEST_PASSWORD, remember=True)
    rows = {r["remember"]: r for r in
            db.get_store().query("SELECT * FROM sessions")}
    assert rows[1]["expires_at"] > rows[0]["expires_at"]
    assert short.ok and long.ok


# ----------------------------------------------------------------- sessions
@pytest.mark.parametrize("bad", [None, "", 123, b"bytes", "no-such-token"])
def test_validate_session_rejects_invalid_tokens(bad):
    assert service.validate_session(bad) is None


def test_expired_session_is_rejected_and_removed():
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(
        timespec="seconds")
    db.get_store().execute(
        "UPDATE sessions SET expires_at = ? WHERE session_id = ?",
        (past, security.token_digest(token)))
    assert service.validate_session(token) is None
    assert db.get_store().query_one(
        "SELECT * FROM sessions WHERE session_id = ?",
        (security.token_digest(token),)) is None


def test_logout_revokes_the_session():
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    assert service.validate_session(token) is not None
    service.logout(token)
    assert service.validate_session(token) is None


def test_password_change_kills_other_sessions_and_reissues_the_callers():
    assert _register().ok
    first = service.authenticate("analyst", TEST_PASSWORD).token
    second = service.authenticate("analyst", TEST_PASSWORD).token
    user = service.validate_session(first)
    new_password = "N3w!Passphrase-Long"
    result = service.change_password(
        user["user_id"], TEST_PASSWORD, new_password, new_password,
        current_token=first)
    assert result.ok, result.errors
    assert service.validate_session(first) is None
    assert service.validate_session(second) is None
    assert result.token and service.validate_session(result.token) is not None
    assert service.authenticate("analyst", new_password).ok
    assert not service.authenticate("analyst", TEST_PASSWORD).ok


def test_password_change_requires_the_current_password():
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    user = service.validate_session(token)
    result = service.change_password(
        user["user_id"], "Wr0ng!Current-1", "N3w!Passphrase-Long",
        "N3w!Passphrase-Long", current_token=token)
    assert not result.ok
    assert service.validate_session(token) is not None


def test_admin_can_list_and_revoke_sessions(admin_user):
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    sessions = service.list_sessions(admin)
    target = next(s for s in sessions if s["username"] == "analyst")
    assert service.revoke_session(admin, target["session_id"]).ok
    assert service.validate_session(token) is None


# ------------------------------------------------- idle session timeout
def _age_session(token: str, minutes: float) -> None:
    past = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(
        timespec="seconds")
    db.get_store().execute(
        "UPDATE sessions SET last_seen_at = ? WHERE session_id = ?",
        (past, security.token_digest(token)))


def test_idle_session_is_rejected_removed_and_audited(monkeypatch):
    monkeypatch.setattr(auth_settings, "SESSION_IDLE_TIMEOUT_MINUTES", 60)
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    _age_session(token, 61)
    assert service.validate_session(token) is None
    assert db.get_store().query_one(
        "SELECT * FROM sessions WHERE session_id = ?",
        (security.token_digest(token),)) is None
    events = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'session_idle_expired'")
    assert len(events) == 1
    assert events[0]["user_id"] == db.get_store().query_one(
        "SELECT user_id FROM users")["user_id"]


def test_activity_refreshes_the_idle_window(monkeypatch):
    monkeypatch.setattr(auth_settings, "SESSION_IDLE_TIMEOUT_MINUTES", 60)
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    _age_session(token, 59)
    assert service.validate_session(token) is not None  # activity refreshes
    _age_session(token, 59)
    assert service.validate_session(token) is not None  # still within window
    _age_session(token, 61)
    assert service.validate_session(token) is None


def test_idle_timeout_applies_to_remember_me_sessions(monkeypatch):
    monkeypatch.setattr(auth_settings, "SESSION_IDLE_TIMEOUT_MINUTES", 60)
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD, remember=True).token
    _age_session(token, 61)
    assert service.validate_session(token) is None


def test_zero_idle_timeout_disables_the_check(monkeypatch):
    monkeypatch.setattr(auth_settings, "SESSION_IDLE_TIMEOUT_MINUTES", 0)
    assert _register().ok
    token = service.authenticate("analyst", TEST_PASSWORD).token
    _age_session(token, 60 * 24)
    assert service.validate_session(token) is not None


# --------------------------------------------------------- password recovery
def test_reset_request_is_generic_for_known_and_unknown_addresses():
    assert _register().ok
    known = service.request_password_reset("analyst@example.test")
    unknown = service.request_password_reset("ghost@example.test")
    assert known.ok and unknown.ok
    prefix = "If an account exists for that address"
    assert known.info.startswith(prefix)
    assert unknown.info.startswith(prefix)
    assert db.get_store().query_one(
        "SELECT COUNT(*) AS n FROM reset_tokens")["n"] == 1


def test_reset_token_is_single_use_and_invalidates_sessions():
    assert _register().ok
    old_session = service.authenticate("analyst", TEST_PASSWORD).token
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    new_password = "R3set!Passphrase-New"
    result = service.reset_password(token, new_password, new_password)
    assert result.ok, result.errors
    assert service.validate_session(old_session) is None
    replay = service.reset_password(token, "An0ther!Passphrase-X",
                                    "An0ther!Passphrase-X")
    assert not replay.ok
    assert service.authenticate("analyst", new_password).ok
    assert not service.authenticate("analyst", TEST_PASSWORD).ok


def test_expired_reset_token_is_rejected():
    assert _register().ok
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    _expire("reset_tokens", token)
    result = service.reset_password(token, TEST_PASSWORD, TEST_PASSWORD)
    assert not result.ok
    assert service.authenticate("analyst", TEST_PASSWORD).ok


def test_reset_clears_an_attackers_lockout(monkeypatch):
    monkeypatch.setattr(auth_settings, "MAX_FAILED_ATTEMPTS", 2)
    assert _register().ok
    for _ in range(2):
        service.authenticate("analyst", "Wr0ng!Password-XX")
    assert not service.authenticate("analyst", TEST_PASSWORD).ok  # locked
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    new_password = "Own3r!Recovery-Pass"
    assert service.reset_password(token, new_password, new_password).ok
    assert service.authenticate("analyst", new_password).ok


def test_reset_rejects_weak_or_mismatched_password_without_consuming_token():
    assert _register().ok
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    assert not service.reset_password(token, "weak", "weak").ok
    assert not service.reset_password(token, TEST_PASSWORD,
                                      "Different-1!").ok
    assert service.reset_password(token, "Fr3sh!Recovery-Pass",
                                  "Fr3sh!Recovery-Pass").ok


def test_reset_request_window_cap(monkeypatch):
    monkeypatch.setattr(auth_settings, "RESET_REQUESTS_PER_WINDOW", 1)
    assert _register().ok
    first = service.request_password_reset("analyst@example.test")
    second = service.request_password_reset("analyst@example.test")
    assert first.ok and second.ok
    assert second.info.startswith("If an account exists")
    assert "outbox" not in second.info  # no new message was generated
    assert db.get_store().query_one(
        "SELECT COUNT(*) AS n FROM reset_tokens")["n"] == 1


def test_invalid_email_on_recovery_is_a_validation_error_only():
    result = service.request_password_reset("not-an-email")
    assert not result.ok
    assert result.errors == ["Enter a valid email address."]


# ------------------------------------------------------ password history
def test_registration_seeds_password_history():
    result = _register()
    rows = db.get_store().query(
        "SELECT * FROM password_history WHERE user_id = ?",
        (result.user["user_id"],))
    assert len(rows) == 1
    assert rows[0]["password_hash"].startswith("$argon2id$")
    assert security.verify_password(rows[0]["password_hash"], TEST_PASSWORD)
    assert TEST_PASSWORD not in rows[0]["password_hash"]


def test_provision_admin_seeds_password_history():
    result = service.provision_admin(
        full_name="Root Operator", email="root@example.test",
        username="rootop", password=ADMIN_PASSWORD)
    assert result.ok, result.errors
    rows = db.get_store().query(
        "SELECT * FROM password_history WHERE user_id = ?",
        (result.user["user_id"],))
    assert len(rows) == 1
    assert security.verify_password(rows[0]["password_hash"], ADMIN_PASSWORD)


def test_password_change_blocks_reuse_of_the_current_password():
    user = _register().user
    token = service.authenticate("analyst", TEST_PASSWORD).token
    result = service.change_password(
        user["user_id"], TEST_PASSWORD, TEST_PASSWORD, TEST_PASSWORD,
        current_token=token)
    assert not result.ok
    assert any("used recently" in e for e in result.errors)
    assert service.validate_session(token) is not None  # nothing invalidated
    assert service.authenticate("analyst", TEST_PASSWORD).ok


def test_password_history_is_trimmed_and_oldest_becomes_reusable(monkeypatch):
    monkeypatch.setattr(auth_settings, "PASSWORD_HISTORY_SIZE", 3)
    user = _register().user
    passwords = [TEST_PASSWORD, "S3cond!Passphrase-A", "Th1rd!Passphrase-B",
                 "F0urth!Passphrase-C"]
    for current, nxt in zip(passwords, passwords[1:]):
        result = service.change_password(user["user_id"], current, nxt, nxt)
        assert result.ok, result.errors
    rows = db.get_store().query(
        "SELECT * FROM password_history WHERE user_id = ? ORDER BY id",
        (user["user_id"],))
    assert len(rows) == 3
    current = passwords[-1]
    for blocked in passwords[1:]:
        assert not service.change_password(
            user["user_id"], current, blocked, blocked).ok
    # The trimmed oldest password may be used again:
    assert service.change_password(
        user["user_id"], current, passwords[0], passwords[0]).ok


def test_reset_blocks_reuse_without_consuming_the_token():
    user = _register().user
    changed = "Ch4nged!Passphrase-X"
    assert service.change_password(
        user["user_id"], TEST_PASSWORD, changed, changed).ok
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    for blocked in (TEST_PASSWORD, changed):
        result = service.reset_password(token, blocked, blocked)
        assert not result.ok
        assert any("used recently" in e for e in result.errors)
    fresh = "R3set!Passphrase-New"
    assert service.reset_password(token, fresh, fresh).ok
    assert service.authenticate("analyst", fresh).ok


def test_legacy_account_without_history_still_blocks_current_reuse():
    user = _register().user
    db.get_store().execute(
        "DELETE FROM password_history WHERE user_id = ?", (user["user_id"],))
    assert not service.change_password(
        user["user_id"], TEST_PASSWORD, TEST_PASSWORD, TEST_PASSWORD).ok
    assert service.change_password(
        user["user_id"], TEST_PASSWORD, "N3w!Passphrase-Long",
        "N3w!Passphrase-Long").ok


def test_reuse_block_is_audited_without_the_password():
    user = _register().user
    service.change_password(user["user_id"], TEST_PASSWORD, TEST_PASSWORD,
                            TEST_PASSWORD)
    rows = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'password_reuse_blocked'")
    assert len(rows) == 1
    assert rows[0]["user_id"] == user["user_id"]
    assert TEST_PASSWORD not in rows[0]["detail"]


# ------------------------------------------------------- email verification
def test_verification_is_single_use():
    assert _register().ok
    token = latest_mail_link_token()
    result = service.verify_email(token)
    assert result.ok, result.errors
    row = db.get_store().query_one("SELECT * FROM users")
    assert row["email_verified"] == 1
    replay = service.verify_email(token)
    assert not replay.ok


def test_expired_verification_token_is_rejected():
    assert _register().ok
    token = latest_mail_link_token()
    _expire("verify_tokens", token)
    assert not service.verify_email(token).ok
    assert db.get_store().query_one("SELECT * FROM users")["email_verified"] == 0


def test_resend_respects_cooldown_then_issues_a_new_single_use_link(
        monkeypatch):
    assert _register().ok
    first_token = latest_mail_link_token()
    immediate = service.resend_verification("analyst@example.test")
    assert immediate.ok
    assert db.get_store().query_one(
        "SELECT COUNT(*) AS n FROM verify_tokens")["n"] == 1
    monkeypatch.setattr(auth_settings, "VERIFY_RESEND_COOLDOWN_SECONDS", 0)
    resent = service.resend_verification("analyst@example.test")
    assert resent.ok
    second_token = latest_mail_link_token()
    assert second_token != first_token
    assert not service.verify_email(first_token).ok  # old link invalidated
    assert service.verify_email(second_token).ok


def test_resend_for_unknown_address_stays_generic():
    result = service.resend_verification("ghost@example.test")
    assert result.ok
    assert result.info.startswith("If an unverified account exists")


# ---------------------------------------------------------- RBAC + admin
def test_non_admins_cannot_reach_administrative_services():
    viewer = _register().user
    for actor in ({"role": "viewer"}, {"role": "analyst"}, viewer):
        with pytest.raises(rbac.PermissionDenied):
            service.list_users(actor)
        with pytest.raises(rbac.PermissionDenied):
            service.set_role(actor, "USR-x", "administrator")
        with pytest.raises(rbac.PermissionDenied):
            service.audit_events(actor)
        with pytest.raises(rbac.PermissionDenied):
            service.list_sessions(actor)


def test_admin_can_change_roles_but_not_their_own(admin_user):
    target = _register().user
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    assert service.set_role(admin, target["user_id"], "analyst").ok
    assert db.get_store().query_one(
        "SELECT role FROM users WHERE user_id = ?",
        (target["user_id"],))["role"] == "analyst"
    own = service.set_role(admin, admin["user_id"], "viewer")
    assert not own.ok
    assert service.set_role(admin, target["user_id"], "superuser").ok is False


def test_admin_cannot_disable_their_own_account(admin_user):
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    assert not service.set_account_status(
        admin, admin["user_id"], "disabled").ok


def test_disabling_revokes_sessions_and_blocking_login(admin_user):
    target = _register().user
    token = service.authenticate("analyst", TEST_PASSWORD).token
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    assert service.set_account_status(admin, target["user_id"], "disabled").ok
    assert service.validate_session(token) is None
    assert not service.authenticate("analyst", TEST_PASSWORD).ok
    assert service.set_account_status(admin, target["user_id"], "active").ok
    assert service.authenticate("analyst", TEST_PASSWORD).ok


def test_page_permission_matrix():
    assert rbac.page_allowed("administrator", "settings_page")
    assert rbac.page_allowed("administrator", "user_management")
    assert not rbac.page_allowed("analyst", "settings_page")
    assert not rbac.page_allowed("analyst", "user_management")
    assert rbac.page_allowed("analyst", "alert_management")
    assert not rbac.page_allowed("viewer", "settings_page")
    assert not rbac.page_allowed("viewer", "user_management")
    for role in rbac.ROLES:
        assert rbac.page_allowed(role, "overview")
        assert rbac.page_allowed(role, "reports")
    assert not rbac.can("viewer", "triage_alerts")
    assert rbac.can("analyst", "triage_alerts")
    assert not rbac.can("analyst", "manage_settings")


def test_navigation_is_filtered_by_role():
    from dashboard import nav
    admin_pages = set(nav.build(role="administrator"))
    analyst_pages = set(nav.build(role="analyst"))
    viewer_pages = set(nav.build(role="viewer"))
    assert "user_management" in admin_pages
    assert "settings_page" in admin_pages
    assert "user_management" not in analyst_pages
    assert "settings_page" not in analyst_pages
    assert "user_management" not in viewer_pages
    assert "settings_page" not in viewer_pages
    assert "overview" in viewer_pages
    # Analysts and viewers open the same pages; the analyst difference is
    # write access inside pages (triage), enforced in the service layer.
    assert viewer_pages == analyst_pages
    assert analyst_pages < admin_pages


def test_provision_admin_creates_a_verified_administrator():
    result = service.provision_admin(
        full_name="Root Operator", email="root@example.test",
        username="rootop", password=ADMIN_PASSWORD)
    assert result.ok, result.errors
    assert result.user["role"] == "administrator"
    assert result.user["email_verified"] is True
    duplicate = service.provision_admin(
        full_name="Root Operator", email="root@example.test",
        username="rootop", password=ADMIN_PASSWORD)
    assert not duplicate.ok
    weak = service.provision_admin(
        full_name="Weak", email="weak@example.test", username="weakop",
        password="short")
    assert not weak.ok


# ------------------------------------------------------------------- audit
def test_audit_records_events_but_never_passwords_or_tokens(admin_user):
    assert _register().ok
    verify_token = latest_mail_link_token()
    service.authenticate("analyst", TEST_PASSWORD)
    service.authenticate("analyst", "Wr0ng!Password-XX")
    service.request_password_reset("analyst@example.test")
    reset_token = latest_mail_link_token()
    service.reset_password(reset_token, "Aud1t!New-Passphrase",
                           "Aud1t!New-Passphrase")
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    events = service.audit_events(admin, limit=500)
    kinds = {e["event"] for e in events}
    assert {"user_registered", "login_success", "login_failed",
            "password_reset_requested", "password_reset_completed",
            "admin_provisioned"} <= kinds
    for event in events:
        blob = f"{event['detail']} {event['event']}"
        for secret in (TEST_PASSWORD, ADMIN_PASSWORD, verify_token,
                       reset_token, "Wr0ng!Password-XX",
                       "Aud1t!New-Passphrase"):
            assert secret not in blob


def test_admin_actions_are_attributed(admin_user):
    target = _register().user
    admin = service.validate_session(
        service.authenticate("testadmin", ADMIN_PASSWORD).token)
    service.set_role(admin, target["user_id"], "analyst")
    events = service.audit_events(admin)
    role_event = next(e for e in events if e["event"] == "role_changed")
    assert role_event["actor_id"] == admin["user_id"]
    assert role_event["user_id"] == target["user_id"]
    assert "viewer -> analyst" in role_event["detail"]


# ------------------------------------------------------ security primitives
def test_argon2id_hashes_are_salted_and_verifiable():
    a = security.hash_password(TEST_PASSWORD)
    b = security.hash_password(TEST_PASSWORD)
    assert a != b
    assert a.startswith("$argon2id$") and b.startswith("$argon2id$")
    assert security.verify_password(a, TEST_PASSWORD)
    assert security.verify_password(b, TEST_PASSWORD)
    assert not security.verify_password(a, "Wr0ng!Password-XX")


@pytest.mark.parametrize("password", [
    "short1!A",                       # too short
    "alllowercase123!extra",          # no uppercase
    "ALLUPPERCASE123!EXTRA",          # no lowercase
    "NoDigitsHere!Extra",             # no digit
    "NoSpecialChars123Extra",         # no special
])
def test_password_policy_rejects_each_weakness(password):
    assert security.password_policy_errors(password)


def test_long_passphrase_is_allowed():
    passphrase = "Correct-Horse-Battery-Staple-9!"
    assert security.password_policy_errors(passphrase) == []
    score, label = security.password_strength(passphrase)
    assert score >= 3


def test_password_strength_is_ordered():
    weak_score, _ = security.password_strength("abc")
    strong_score, strong_label = security.password_strength(TEST_PASSWORD)
    assert weak_score < strong_score
    assert strong_label in ("Good", "Strong")
    assert len(security.password_checks(TEST_PASSWORD)) == 5
    assert all(met for _, met in security.password_checks(TEST_PASSWORD))


def test_tokens_are_unique_and_only_digests_are_derivable():
    tokens = {security.new_token() for _ in range(50)}
    assert len(tokens) == 50
    token = security.new_token()
    digest = security.token_digest(token)
    assert digest == security.token_digest(token)
    assert digest != token and len(digest) == 64


@pytest.mark.parametrize("email,valid", [
    ("name@example.com", True), ("a.b+tag@sub.example.co.uk", True),
    ("no-at-sign", False), ("@example.com", False), ("a@b", False),
    ("spaces in@example.com", False),
])
def test_email_validation(email, valid):
    assert security.valid_email(email) is valid


@pytest.mark.parametrize("username,valid", [
    ("ada", True), ("analyst_01", True), ("soc.team-lead", True),
    ("ab", False), ("-nope", False), ("has space", False),
    ("x" * 33, False), ("inject';--", False),
])
def test_username_validation(username, valid):
    assert security.valid_username(username) is valid


def test_mail_links_carry_the_single_use_token():
    token = security.new_token()
    assert mail.reset_link(token).endswith(f"?auth=reset&token={token}")
    assert mail.verify_link(token).endswith(f"?auth=verify&token={token}")


def test_smtp_mode_without_host_reports_failure_honestly(monkeypatch):
    monkeypatch.setattr(auth_settings, "EMAIL_MODE", "smtp")
    monkeypatch.setattr(auth_settings, "SMTP_HOST", "")
    result = mail.send_mail("a@example.test", "Subject", "Body")
    assert result.mode == "smtp" and result.delivered is False
    assert "not configured" in result.note


@pytest.mark.parametrize("provider,host", [
    ("sendgrid", "smtp.sendgrid.net"),
    ("brevo", "smtp-relay.brevo.com"),
    ("mailgun", "smtp.mailgun.org"),
    ("postmark", "smtp.postmarkapp.com"),
    ("ses", "email-smtp.us-east-1.amazonaws.com"),
    ("gmail", "smtp.gmail.com"),
    ("SendGrid", "smtp.sendgrid.net"),  # case-insensitive
])
def test_smtp_provider_presets_resolve_host(provider, host):
    resolved_host, port, starttls = auth_settings.smtp_preset(provider)
    assert resolved_host == host
    assert port == 587 and starttls is True


def test_smtp_unknown_provider_has_no_host():
    assert auth_settings.smtp_preset("not-a-provider") == ("", 587, True)
    assert auth_settings.smtp_preset("") == ("", 587, True)