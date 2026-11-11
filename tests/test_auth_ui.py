"""UI-level authentication tests driven through Streamlit's AppTest harness.

These exercise the real app.py gate and the real auth pages (widget values,
form submits, button clicks, query-parameter deep links) against the
disposable per-test auth database. Dashboard-rendering tests are marked
because they need the trained artefacts, exactly like test_dashboard.py.
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pyotp
from streamlit.testing.v1 import AppTest

import config
from src.auth import db, security, service
from tests.conftest import TEST_PASSWORD, latest_mail_link_token

ROOT = Path(__file__).resolve().parents[1]
LOGIN_SUBMIT = "FormSubmitter:login_form-Sign In to AI-SOC"
PW_SUBMIT = "FormSubmitter:change_password_form-Update password"
MFA_SUBMIT = "FormSubmitter:mfa_form-Verify code"
NEW_PASSWORD = "N3w!UI-Passphrase"

requires_artefacts = pytest.mark.skipif(
    not all(Path(p).exists() for p in config.MODEL_FILES.values())
    or not (config.DATA_PROCESSED_DIR / "alerts.csv").exists(),
    reason="trained models / generated alert store not present",
)


def _app(timeout: float = 300.0) -> AppTest:
    return AppTest.from_file(ROOT / "app.py", default_timeout=timeout)


def _widget(elements, key):
    for element in elements:
        if element.key == key:
            return element
    raise AssertionError(f"widget '{key}' not rendered; keys="
                         f"{[e.key for e in elements]}")


def _page_app(module_name: str, token: str) -> AppTest:
    module = importlib.import_module(f"dashboard.pages.{module_name}")
    source = Path(module.__file__).read_text(encoding="utf-8")
    at = AppTest.from_string(source + "\nrender()\n", default_timeout=300)
    at.session_state["auth_token"] = token
    return at


def _register_viewer():
    result = service.register(
        full_name="Vera Viewer", email="viewer@example.test",
        username="viewer", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD)
    assert result.ok, result.errors
    return result.user


def _login(at: AppTest, identifier: str, password: str,
           remember: bool = False) -> AppTest:
    _widget(at.text_input, "login_identifier").set_value(identifier)
    _widget(at.text_input, "login_password").set_value(password)
    if remember:
        _widget(at.checkbox, "login_remember").check()
    return _widget(at.button, LOGIN_SUBMIT).click().run()


# ------------------------------------------------------------------- gate
def test_unauthenticated_visitor_sees_only_the_branded_login():
    at = _app().run()
    assert not at.exception, [e.message for e in at.exception]
    keys = [t.key for t in at.text_input]
    assert "login_identifier" in keys and "login_password" in keys
    assert any(b.key == LOGIN_SUBMIT for b in at.button)
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Secure access to your security operations workspace." in rendered
    assert "Intelligent Security Operations Platform" in rendered
    sidebar = " ".join(str(m.value) for m in at.sidebar.markdown)
    assert "AI-SOC" not in sidebar, "dashboard shell must not render pre-auth"
    assert not at.session_state.get("auth_token")


def test_auth_pages_are_reachable_from_the_login_links():
    at = _app().run()
    at = _widget(at.button, "auth_to_signup").click().run()
    assert at.query_params.get("auth") == "signup"
    assert "su_password" in [t.key for t in at.text_input]
    at = _widget(at.button, "auth_to_login").click().run()
    assert at.query_params.get("auth") == "login"
    at = _widget(at.button, "auth_to_forgot").click().run()
    assert at.query_params.get("auth") == "forgot"
    assert "fp_email" in [t.key for t in at.text_input]


def test_unknown_auth_route_falls_back_to_login():
    at = _app()
    at.query_params["auth"] = "not-a-page"
    at = at.run()
    assert not at.exception
    assert "login_identifier" in [t.key for t in at.text_input]


# ----------------------------------------------------------------- signup
def test_signup_creates_a_real_least_privileged_account():
    at = _app()
    at.query_params["auth"] = "signup"
    at = at.run()
    _widget(at.text_input, "su_full_name").set_value("Grace Hopper")
    _widget(at.text_input, "su_username").set_value("grace")
    _widget(at.text_input, "su_email").set_value("grace@example.test")
    _widget(at.text_input, "su_org").set_value("Naval Systems")
    _widget(at.text_input, "su_password").set_value(TEST_PASSWORD)
    _widget(at.text_input, "su_confirm").set_value(TEST_PASSWORD)
    at = _widget(at.button, "su_submit").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.query_params.get("auth") == "login"
    assert any("Account created" in s.value for s in at.success)
    row = db.get_store().query_one(
        "SELECT * FROM users WHERE username = 'grace'")
    assert row is not None
    assert row["role"] == "viewer"
    assert row["password_hash"].startswith("$argon2id$")
    assert row["organisation"] == "Naval Systems"


def test_signup_rejects_a_weak_password():
    at = _app()
    at.query_params["auth"] = "signup"
    at = at.run()
    _widget(at.text_input, "su_full_name").set_value("Weak User")
    _widget(at.text_input, "su_username").set_value("weakuser")
    _widget(at.text_input, "su_email").set_value("weak@example.test")
    _widget(at.text_input, "su_password").set_value("weak")
    _widget(at.text_input, "su_confirm").set_value("weak")
    at = _widget(at.button, "su_submit").click().run()
    assert not at.exception
    assert at.error, "policy errors must be shown"
    assert db.get_store().query("SELECT * FROM users") == []


def test_signup_rejects_mismatched_confirmation():
    at = _app()
    at.query_params["auth"] = "signup"
    at = at.run()
    _widget(at.text_input, "su_full_name").set_value("Mismatch User")
    _widget(at.text_input, "su_username").set_value("mismatch")
    _widget(at.text_input, "su_email").set_value("mismatch@example.test")
    _widget(at.text_input, "su_password").set_value(TEST_PASSWORD)
    _widget(at.text_input, "su_confirm").set_value("Different-1!Password")
    at = _widget(at.button, "su_submit").click().run()
    assert any("do not match" in e.value for e in at.error)
    assert db.get_store().query("SELECT * FROM users") == []


# ------------------------------------------------------------------ login
@requires_artefacts
def test_valid_login_opens_the_dashboard():
    _register_viewer()
    at = _login(_app().run(), "viewer@example.test", TEST_PASSWORD)
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state.get("auth_token")
    sidebar = " ".join(str(m.value) for m in at.sidebar.markdown)
    assert "AI-SOC" in sidebar
    assert "Vera Viewer" in sidebar
    assert "Researcher / Viewer" in sidebar
    assert len(at.main.children) > 0


@requires_artefacts
def test_remember_me_queues_the_opaque_session_cookie():
    _register_viewer()
    at = _login(_app().run(), "viewer", TEST_PASSWORD, remember=True)
    assert not at.exception
    token = at.session_state.get("auth_token")
    assert token
    cookie_state = at.session_state.get("_cookie_state")
    assert cookie_state and cookie_state.get("value") == token
    assert cookie_state.get("max_age", 0) > 0


def test_wrong_password_shows_the_generic_error_and_no_session():
    _register_viewer()
    at = _login(_app().run(), "viewer", "Wr0ng!Password-XX")
    assert not at.exception
    assert [e.value for e in at.error] == [
        "Invalid email/username or password."]
    assert not at.session_state.get("auth_token")
    assert "login_identifier" in [t.key for t in at.text_input]


def test_expired_session_returns_to_login_with_an_honest_notice():
    _register_viewer()
    token = service.authenticate("viewer", TEST_PASSWORD).token
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(
        timespec="seconds")
    db.get_store().execute(
        "UPDATE sessions SET expires_at = ? WHERE session_id = ?",
        (past, security.token_digest(token)))
    at = _app()
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception
    assert "login_identifier" in [t.key for t in at.text_input]
    assert any("session expired" in i.value for i in at.info)
    assert not at.session_state.get("auth_token")


@requires_artefacts
def test_logout_revokes_the_session_and_returns_to_login():
    _register_viewer()
    token = service.authenticate("viewer", TEST_PASSWORD).token
    at = _app()
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception
    at = _widget(at.button, "logout_btn").click().run()
    assert not at.exception
    assert "login_identifier" in [t.key for t in at.text_input]
    assert not at.session_state.get("auth_token")
    assert service.validate_session(token) is None


# --------------------------------------------------------------- recovery
def test_forgot_password_response_never_enumerates_accounts():
    at = _app()
    at.query_params["auth"] = "forgot"
    at = at.run()
    _widget(at.text_input, "fp_email").set_value("ghost@example.test")
    at = _widget(at.button, "fp_submit").click().run()
    assert not at.exception
    assert any(s.value.startswith("If an account exists") for s in at.success)
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "development outbox" in rendered  # honest delivery-mode note


def test_reset_flow_end_to_end_through_the_outbox_link():
    _register_viewer()
    assert service.request_password_reset("viewer@example.test").ok
    token = latest_mail_link_token()
    assert token
    at = _app()
    at.query_params["auth"] = "reset"
    at.query_params["token"] = token
    at = at.run()
    assert not at.exception
    _widget(at.text_input, "rp_new_password").set_value(NEW_PASSWORD)
    _widget(at.text_input, "rp_confirm").set_value(NEW_PASSWORD)
    at = _widget(at.button, "rp_submit").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Password updated" in s.value for s in at.success)
    at = _widget(at.button, "rp_continue").click().run()
    assert at.query_params.get("auth") == "login"
    assert service.authenticate("viewer", NEW_PASSWORD).ok
    assert not service.authenticate("viewer", TEST_PASSWORD).ok


def test_reset_page_without_a_token_points_back_to_recovery():
    at = _app()
    at.query_params["auth"] = "reset"
    at = at.run()
    assert any("No recovery token" in e.value for e in at.error)
    at = _widget(at.button, "auth_to_forgot").click().run()
    assert at.query_params.get("auth") == "forgot"


def test_verification_link_marks_the_account_verified_once():
    _register_viewer()
    token = latest_mail_link_token()
    at = _app()
    at.query_params["auth"] = "verify"
    at.query_params["token"] = token
    at = at.run()
    assert not at.exception
    assert any("verified" in s.value for s in at.success)
    assert db.get_store().query_one(
        "SELECT email_verified FROM users")["email_verified"] == 1
    at = _widget(at.button, "ve_continue").click().run()
    assert at.query_params.get("auth") == "login"
    replay = _app()
    replay.query_params["auth"] = "verify"
    replay.query_params["token"] = token
    replay = replay.run()
    assert any("invalid, expired or already used" in e.value
               for e in replay.error)


# ----------------------------------------------------- role enforcement
@requires_artefacts
def test_viewer_is_refused_admin_pages_server_side(viewer_token):
    for module in ("settings_page", "user_management"):
        at = _page_app(module, viewer_token).run()
        assert not at.exception, [e.message for e in at.exception]
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "Access restricted" in rendered
        assert not at.form


@requires_artefacts
def test_viewer_sees_read_only_alert_triage(viewer_token):
    at = _page_app("alert_management", viewer_token).run()
    assert not at.exception
    alert_id = _widget(at.selectbox, "am_pick").value.split("  |")[0]
    assert _widget(at.button, f"am_quick_inv_{alert_id}").disabled
    assert _widget(at.button, f"am_quick_res_{alert_id}").disabled
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "read-only" in rendered.lower()


@requires_artefacts
def test_admin_can_open_user_management(admin_token):
    at = _page_app("user_management", admin_token).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Access restricted" not in rendered
    assert any(t.key == "um_pick" for t in at.selectbox)


@requires_artefacts
def test_password_change_from_the_account_panel_reissues_the_session():
    _register_viewer()
    old_token = service.authenticate("viewer", TEST_PASSWORD).token
    at = _app()
    at.session_state["auth_token"] = old_token
    at = at.run()
    assert not at.exception
    _widget(at.text_input, "pw_current").set_value(TEST_PASSWORD)
    _widget(at.text_input, "pw_new").set_value(NEW_PASSWORD)
    _widget(at.text_input, "pw_confirm").set_value(NEW_PASSWORD)
    at = _widget(at.button, PW_SUBMIT).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Password changed" in s.value for s in at.success)
    new_token = at.session_state.get("auth_token")
    assert new_token and new_token != old_token
    assert service.validate_session(old_token) is None
    assert service.validate_session(new_token) is not None
    assert service.authenticate("viewer", NEW_PASSWORD).ok


# ------------------------------------------------------- two-factor (UI)
def _enrol_viewer() -> str:
    user = _register_viewer()
    result = service.begin_totp_enrollment(user["user_id"])
    assert result.ok, result.errors
    secret = result.payload["secret"]
    assert service.confirm_totp_enrollment(
        user["user_id"], pyotp.TOTP(secret).now()).ok
    return secret


def test_mfa_login_wrong_code_stays_on_the_code_step():
    secret = _enrol_viewer()
    at = _login(_app().run(), "viewer", TEST_PASSWORD)
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state.get("_mfa_token")
    assert not at.session_state.get("auth_token")
    assert "mfa_code" in [t.key for t in at.text_input]
    wrong = "999999" if pyotp.TOTP(secret).now() != "999999" else "888888"
    _widget(at.text_input, "mfa_code").set_value(wrong)
    at = _widget(at.button, MFA_SUBMIT).click().run()
    assert not at.exception
    assert any("Invalid authentication code" in e.value for e in at.error)
    assert not at.session_state.get("auth_token")
    assert at.session_state.get("_mfa_token")


@requires_artefacts
def test_mfa_login_completes_with_the_real_code():
    secret = _enrol_viewer()
    at = _login(_app().run(), "viewer", TEST_PASSWORD)
    assert at.session_state.get("_mfa_token")
    _widget(at.text_input, "mfa_code").set_value(pyotp.TOTP(secret).now())
    at = _widget(at.button, MFA_SUBMIT).click().run()
    assert not at.exception, [e.message for e in at.exception]
    token = at.session_state.get("auth_token")
    assert token and service.validate_session(token) is not None
    assert not at.session_state.get("_mfa_token")


@requires_artefacts
def test_account_panel_enrols_totp_end_to_end():
    user = _register_viewer()
    token = service.authenticate("viewer", TEST_PASSWORD).token
    at = _app()
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception
    at = _widget(at.button, "totp_begin").click().run()
    assert not at.exception, [e.message for e in at.exception]
    payload = at.session_state.get("_totp_enroll")
    assert payload and payload["secret"]
    _widget(at.text_input, "totp_confirm_code").set_value(
        pyotp.TOTP(payload["secret"]).now())
    at = _widget(at.button,
                 "FormSubmitter:totp_confirm_form-Verify and enable"
                 ).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert service.totp_enabled(user["user_id"])
    assert any("now enabled" in s.value for s in at.success)
    assert not at.session_state.get("_totp_enroll")


@requires_artefacts
def test_account_panel_disables_totp_with_a_current_code():
    secret = _enrol_viewer()
    user_id = db.get_store().query_one(
        "SELECT user_id FROM users WHERE username = 'viewer'")["user_id"]
    challenge = service.authenticate("viewer", TEST_PASSWORD).mfa_token
    assert challenge  # MFA is on: password alone yields a challenge
    token = service.complete_totp_login(
        challenge, pyotp.TOTP(secret).now()).token
    at = _app()
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception
    _widget(at.text_input, "totp_disable_code").set_value(
        pyotp.TOTP(secret).now())
    at = _widget(at.button,
                 "FormSubmitter:totp_disable_form-Disable two-factor"
                 ).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not service.totp_enabled(user_id)
    assert service.authenticate("viewer", TEST_PASSWORD).token


@requires_artefacts
def test_account_panel_reports_mfa_unavailable_without_key(monkeypatch):
    from src.auth import settings as auth_settings
    _register_viewer()
    token = service.authenticate("viewer", TEST_PASSWORD).token
    monkeypatch.setattr(auth_settings, "TOTP_KEY", "")
    at = _app()
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception
    captions = " ".join(str(c.value) for c in at.sidebar.caption)
    assert "Unavailable in this deployment" in captions
    assert "totp_begin" not in [b.key for b in at.button]


@requires_artefacts
def test_admin_removes_totp_from_user_management(admin_token):
    _enrol_viewer()
    user_id = db.get_store().query_one(
        "SELECT user_id FROM users WHERE username = 'viewer'")["user_id"]
    at = _page_app("user_management", admin_token).run()
    assert not at.exception, [e.message for e in at.exception]
    _widget(at.selectbox, "um_pick").set_value(
        "viewer  (viewer@example.test)")
    at = at.run()
    assert not _widget(at.button, "um_disable_totp").disabled
    at = _widget(at.button, "um_disable_totp").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not service.totp_enabled(user_id)
    assert service.authenticate("viewer", TEST_PASSWORD).token