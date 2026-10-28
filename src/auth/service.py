"""Authentication service: registration, login, sessions, recovery, RBAC
admin actions and the audit trail.

Security properties:
  * argon2id hashing; plaintext passwords are never stored or logged;
  * opaque 256-bit session/recovery tokens, persisted only as SHA-256
    digests; recovery and verification tokens are single-use and expiring;
  * generic credential-failure messages (no account enumeration);
  * per-identifier rate limiting with temporary lockout;
  * password changes bump an auth epoch, invalidating every old session;
  * password history blocks reuse of the last N passwords on change/reset;
  * sessions expire after an idle period as well as their absolute TTL;
  * optional TOTP second factor: Fernet-encrypted secrets, single-use
    short-lived login challenges, admin removal as the recovery path;
  * roles are assigned server-side; registration always yields `viewer`;
  * audit events record important actions without secrets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.auth import breach, mail, mfa, rbac, security
from src.auth import settings as auth_settings
from src.auth.db import get_store


@dataclass
class AuthResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    user: dict | None = None
    token: str | None = None
    info: str | None = None
    mfa_token: str | None = None
    payload: dict | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _public_user(row) -> dict:
    return {
        "user_id": row["user_id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "username": row["username"],
        "organisation": row["organisation"],
        "role": row["role"],
        "email_verified": bool(row["email_verified"]),
        "account_status": row["account_status"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def _audit(event: str, user_id: str | None = None, actor_id: str | None = None,
           detail: str = "") -> None:
    get_store().execute(
        "INSERT INTO audit_events (event, user_id, actor_id, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (event, user_id, actor_id, security.clean_text(detail, 300),
         _iso(_now())))


def _find_user(identifier: str):
    store = get_store()
    ident = identifier.strip().lower()
    return store.query_one(
        "SELECT * FROM users WHERE email = ? OR username = ?", (ident, ident))


def _record_attempt(identifier: str, success: bool) -> None:
    get_store().execute(
        "INSERT INTO login_attempts (identifier, success, attempted_at) "
        "VALUES (?, ?, ?)",
        (identifier.strip().lower(), int(success), _iso(_now())))


def _lockout_remaining(identifier: str) -> int:
    """Minutes remaining on lockout, 0 if not locked. Applies to any
    identifier (even unknown ones) so the response cannot enumerate
    accounts."""
    store = get_store()
    window_start = _iso(_now() - timedelta(
        minutes=auth_settings.ATTEMPT_WINDOW_MINUTES))
    rows = store.query(
        "SELECT success, attempted_at FROM login_attempts "
        "WHERE identifier = ? AND attempted_at >= ? "
        "ORDER BY attempted_at DESC, id DESC",
        (identifier.strip().lower(), window_start))
    failures = 0
    last_failure: datetime | None = None
    for row in rows:
        if row["success"]:
            break
        failures += 1
        if last_failure is None:
            last_failure = datetime.fromisoformat(row["attempted_at"])
    if failures < auth_settings.MAX_FAILED_ATTEMPTS or last_failure is None:
        return 0
    unlock_at = last_failure + timedelta(minutes=auth_settings.LOCKOUT_MINUTES)
    remaining = (unlock_at - _now()).total_seconds() / 60
    return max(0, int(remaining) + (1 if remaining % 1 else 0))


def _create_session(user_row, remember: bool) -> str:
    token = security.new_token()
    ttl = (timedelta(days=auth_settings.REMEMBER_ME_DAYS) if remember
           else timedelta(hours=auth_settings.SESSION_TTL_HOURS))
    now = _now()
    get_store().execute(
        "INSERT INTO sessions (session_id, user_id, auth_epoch, remember, "
        "created_at, expires_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (security.token_digest(token), user_row["user_id"],
         user_row["auth_epoch"], int(remember), _iso(now), _iso(now + ttl),
         _iso(now)))
    return token


# ---------------------------------------------------------- password history
def _password_reused(user_row, new_password: str) -> bool:
    """True if the candidate matches one of the last N passwords. Every
    password-setting path seeds/appends history, so the newest entry is the
    current hash; accounts created before history existed have no rows and
    fall back to checking the current hash only."""
    rows = get_store().query(
        "SELECT password_hash FROM password_history WHERE user_id = ? "
        "ORDER BY id DESC LIMIT ?",
        (user_row["user_id"], auth_settings.PASSWORD_HISTORY_SIZE))
    hashes = [r["password_hash"] for r in rows] or [user_row["password_hash"]]
    return any(security.verify_password(h, new_password) for h in hashes)


def _record_password_history(user_id: str, password_hash: str) -> None:
    store = get_store()
    store.execute(
        "INSERT INTO password_history (user_id, password_hash, created_at) "
        "VALUES (?, ?, ?)", (user_id, password_hash, _iso(_now())))
    store.execute(
        "DELETE FROM password_history WHERE user_id = ? AND id NOT IN "
        "(SELECT id FROM password_history WHERE user_id = ? "
        "ORDER BY id DESC LIMIT ?)",
        (user_id, user_id, auth_settings.PASSWORD_HISTORY_SIZE))


_REUSE_ERROR = ("This password was used recently. Choose a password you "
                "have not used before.")


def _breach_screen(password: str, user_id: str | None = None) -> str | None:
    """Optional HIBP k-anonymity check. Returns a user-facing error when the
    password is known-breached; an unreachable service fails open and is
    audited as such (never presented as a clean result)."""
    status, count = breach.check_password(password)
    if status == "breached":
        _audit("breach_password_blocked", user_id=user_id,
               detail=f"occurrences={count}")
        return (f"This password appears in known data breaches (seen "
                f"{count:,} times). Choose a different password.")
    if status == "unreachable":
        _audit("breach_check_unavailable", user_id=user_id,
               detail="fail_open")
    return None


# ------------------------------------------------------------- registration
def register(*, full_name: str, email: str, username: str, password: str,
             confirm_password: str, organisation: str = "") -> AuthResult:
    """Create a least-privileged account. Any caller-supplied role is
    impossible by construction: this function accepts no role argument."""
    errors: list[str] = []
    full_name = security.clean_text(full_name, 100)
    organisation = security.clean_text(organisation, 120)
    email = email.strip().lower()
    username = username.strip().lower()

    if not full_name:
        errors.append("Enter your full name.")
    if not email:
        errors.append("Enter your email address.")
    elif not security.valid_email(email):
        errors.append("Enter a valid email address (name@example.com).")
    if not username:
        errors.append("Choose a username.")
    elif not security.valid_username(username):
        errors.append("Username must be 3-32 characters: letters, numbers, "
                      "dots, underscores or hyphens, starting with a letter "
                      "or number.")
    if not password:
        errors.append("Enter a password.")
    else:
        errors.extend(security.password_policy_errors(password))
    if password != confirm_password:
        errors.append("The two passwords do not match.")
    if errors:
        return AuthResult(False, errors=errors)

    store = get_store()
    hour_ago = _iso(_now() - timedelta(hours=1))
    recent = store.query_one(
        "SELECT COUNT(*) AS n FROM users WHERE created_at >= ?", (hour_ago,))
    if recent and recent["n"] >= auth_settings.MAX_REGISTRATIONS_PER_HOUR:
        return AuthResult(False, errors=[
            "Too many accounts were created recently. Try again later."])
    if store.query_one("SELECT user_id FROM users WHERE email = ?", (email,)):
        return AuthResult(False, errors=[
            "That email address is already registered. Sign in instead, or "
            "use password recovery if you cannot access the account."])
    if store.query_one("SELECT user_id FROM users WHERE username = ?",
                       (username,)):
        return AuthResult(False, errors=["That username is already taken."])
    breach_error = _breach_screen(password)
    if breach_error:
        return AuthResult(False, errors=[breach_error])

    user_id = security.new_user_id()
    now = _iso(_now())
    password_hash = security.hash_password(password)
    store.execute(
        "INSERT INTO users (user_id, full_name, email, username, organisation, "
        "password_hash, role, email_verified, account_status, auth_epoch, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', 1, ?, ?)",
        (user_id, full_name, email, username, organisation,
         password_hash, rbac.DEFAULT_ROLE, now, now))
    _record_password_history(user_id, password_hash)
    _audit("user_registered", user_id=user_id,
           detail=f"role={rbac.DEFAULT_ROLE}")

    token = security.new_token()
    store.execute(
        "INSERT INTO verify_tokens (token_hash, user_id, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (security.token_digest(token), user_id, now,
         _iso(_now() + timedelta(hours=auth_settings.VERIFY_TOKEN_TTL_HOURS))))
    delivery = mail.send_verification(email, token)
    _audit("verification_requested", user_id=user_id,
           detail=f"mode={delivery.mode}")

    if delivery.mode == "off":
        info = ("Account created. Email delivery is not configured, so the "
                "verification message could not be sent. You can sign in "
                "now; the account stays marked unverified until an "
                "administrator enables email delivery.")
    elif delivery.mode == "outbox":
        info = ("Account created. Email delivery is not configured for real "
                "sending: the verification message was written to the "
                "development outbox (results/mail_outbox/). Open the .eml "
                "file there to find your verification link. You can also "
                "sign in now; the account stays unverified until the link "
                "is used.")
    else:
        info = ("Account created. Check your inbox for the verification "
                "link (it expires in "
                f"{auth_settings.VERIFY_TOKEN_TTL_HOURS} hours). You can "
                "sign in now; the account stays unverified until the link "
                "is used.")
    row = store.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
    return AuthResult(True, user=_public_user(row), info=info)


# -------------------------------------------------------------------- login
_GENERIC_LOGIN_ERROR = "Invalid email/username or password."


def authenticate(identifier: str, password: str,
                 remember: bool = False) -> AuthResult:
    identifier = identifier.strip()
    if not identifier or not password:
        return AuthResult(False, errors=[
            "Enter your email/username and password."])

    locked = _lockout_remaining(identifier)
    if locked > 0:
        _audit("login_locked", detail=f"identifier_hash="
               f"{security.token_digest(identifier.lower())[:12]}")
        return AuthResult(False, errors=[
            f"Too many failed attempts. Try again in {locked} minute(s)."])

    row = _find_user(identifier)
    if row is None:
        security.verify_password(security.DUMMY_HASH, password)  # timing
        _record_attempt(identifier, False)
        _audit("login_failed", detail="unknown_account")
        return AuthResult(False, errors=[_GENERIC_LOGIN_ERROR])
    if row["account_status"] != "active":
        security.verify_password(security.DUMMY_HASH, password)
        _record_attempt(identifier, False)
        _audit("login_failed", user_id=row["user_id"], detail="account_disabled")
        return AuthResult(False, errors=[_GENERIC_LOGIN_ERROR])
    if not security.verify_password(row["password_hash"], password):
        _record_attempt(identifier, False)
        _audit("login_failed", user_id=row["user_id"], detail="bad_password")
        return AuthResult(False, errors=[_GENERIC_LOGIN_ERROR])

    if security.needs_rehash(row["password_hash"]):
        get_store().execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?",
            (security.hash_password(password), _iso(_now()), row["user_id"]))
    if _totp_credential(row["user_id"]) is not None:
        # Password stage passed; no session until the TOTP code is verified.
        challenge = security.new_token()
        get_store().execute(
            "INSERT INTO totp_challenges (token_hash, user_id, auth_epoch, "
            "remember, attempts, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            (security.token_digest(challenge), row["user_id"],
             row["auth_epoch"], int(remember), _iso(_now()),
             _iso(_now() + timedelta(
                 minutes=auth_settings.TOTP_CHALLENGE_TTL_MINUTES))))
        _audit("totp_challenge_issued", user_id=row["user_id"])
        return AuthResult(True, user=_public_user(row), mfa_token=challenge)
    _record_attempt(identifier, True)
    token = _create_session(row, remember)
    get_store().execute(
        "UPDATE users SET last_login_at = ? WHERE user_id = ?",
        (_iso(_now()), row["user_id"]))
    _audit("login_success", user_id=row["user_id"],
           detail=f"remember={int(remember)}")
    return AuthResult(True, user=_public_user(row), token=token)


def validate_session(token: str | None) -> dict | None:
    """Return the public user for a live session, else None. Expired,
    revoked, disabled-account and post-password-change sessions fail."""
    if not token or not isinstance(token, str):
        return None
    store = get_store()
    digest = security.token_digest(token)
    session = store.query_one(
        "SELECT * FROM sessions WHERE session_id = ?", (digest,))
    if session is None:
        return None
    now = _iso(_now())
    if session["expires_at"] < now:
        store.execute("DELETE FROM sessions WHERE session_id = ?", (digest,))
        return None
    idle_minutes = auth_settings.SESSION_IDLE_TIMEOUT_MINUTES
    if idle_minutes > 0 and (_now() - datetime.fromisoformat(
            session["last_seen_at"])).total_seconds() > idle_minutes * 60:
        store.execute("DELETE FROM sessions WHERE session_id = ?", (digest,))
        _audit("session_idle_expired", user_id=session["user_id"])
        return None
    user = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (session["user_id"],))
    if user is None or user["account_status"] != "active":
        store.execute("DELETE FROM sessions WHERE session_id = ?", (digest,))
        return None
    if session["auth_epoch"] != user["auth_epoch"]:
        store.execute("DELETE FROM sessions WHERE session_id = ?", (digest,))
        return None
    store.execute(
        "UPDATE sessions SET last_seen_at = ? WHERE session_id = ?",
        (now, digest))
    result = _public_user(user)
    result["session_expires_at"] = session["expires_at"]
    return result


def logout(token: str | None) -> None:
    if not token:
        return
    store = get_store()
    digest = security.token_digest(token)
    session = store.query_one(
        "SELECT user_id FROM sessions WHERE session_id = ?", (digest,))
    store.execute("DELETE FROM sessions WHERE session_id = ?", (digest,))
    if session:
        _audit("logout", user_id=session["user_id"])


# --------------------------------------------------------- password recovery
def request_password_reset(email: str) -> AuthResult:
    """Always returns the same generic message - the response never reveals
    whether the address belongs to an account."""
    generic = (
        "If an account exists for that address, a recovery link has been "
        f"sent. It expires in {auth_settings.RESET_TOKEN_TTL_MINUTES} minutes "
        "and can be used once.")
    email = email.strip().lower()
    if not security.valid_email(email):
        return AuthResult(False, errors=["Enter a valid email address."])

    store = get_store()
    row = store.query_one("SELECT * FROM users WHERE email = ?", (email,))
    if row is not None and row["account_status"] == "active":
        window_start = _iso(_now() - timedelta(
            minutes=auth_settings.ATTEMPT_WINDOW_MINUTES))
        recent = store.query_one(
            "SELECT COUNT(*) AS n FROM reset_tokens WHERE user_id = ? "
            "AND created_at >= ?", (row["user_id"], window_start))
        if not recent or recent["n"] < auth_settings.RESET_REQUESTS_PER_WINDOW:
            store.execute(
                "UPDATE reset_tokens SET used_at = ? WHERE user_id = ? "
                "AND used_at IS NULL", (_iso(_now()), row["user_id"]))
            token = security.new_token()
            store.execute(
                "INSERT INTO reset_tokens (token_hash, user_id, created_at, "
                "expires_at) VALUES (?, ?, ?, ?)",
                (security.token_digest(token), row["user_id"], _iso(_now()),
                 _iso(_now() + timedelta(
                     minutes=auth_settings.RESET_TOKEN_TTL_MINUTES))))
            delivery = mail.send_password_reset(email, token)
            _audit("password_reset_requested", user_id=row["user_id"],
                   detail=f"mode={delivery.mode}")
            if delivery.mode == "outbox":
                return AuthResult(True, info=(
                    generic + " Email delivery is not configured for real "
                    "sending: the message was written to the development "
                    "outbox (results/mail_outbox/). Open the newest .eml "
                    "file there to find the recovery link."))
            if delivery.mode == "off":
                return AuthResult(True, info=(
                    generic + " Email delivery is currently disabled, so in "
                    "this deployment the message cannot actually be "
                    "delivered until SMTP is configured."))
            if not delivery.delivered:
                return AuthResult(True, info=generic + " " + delivery.note)
    return AuthResult(True, info=generic)


def reset_password(token: str, password: str,
                   confirm_password: str) -> AuthResult:
    errors = []
    if not password:
        errors.append("Enter a new password.")
    else:
        errors.extend(security.password_policy_errors(password))
    if password != confirm_password:
        errors.append("The two passwords do not match.")
    if errors:
        return AuthResult(False, errors=errors)

    store = get_store()
    row = store.query_one(
        "SELECT * FROM reset_tokens WHERE token_hash = ?",
        (security.token_digest(token),))
    invalid = AuthResult(False, errors=[
        "This recovery link is invalid, expired or already used. Request a "
        "new one from the sign-in page."])
    if row is None or row["used_at"] is not None:
        return invalid
    if row["expires_at"] < _iso(_now()):
        return invalid
    user = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (row["user_id"],))
    if user is None or user["account_status"] != "active":
        return invalid
    if _password_reused(user, password):
        _audit("password_reuse_blocked", user_id=user["user_id"],
               detail="recovery_reset")
        return AuthResult(False, errors=[_REUSE_ERROR])
    breach_error = _breach_screen(password, user_id=user["user_id"])
    if breach_error:
        return AuthResult(False, errors=[breach_error])

    now = _iso(_now())
    password_hash = security.hash_password(password)
    store.execute(
        "UPDATE users SET password_hash = ?, auth_epoch = auth_epoch + 1, "
        "updated_at = ? WHERE user_id = ?",
        (password_hash, now, user["user_id"]))
    _record_password_history(user["user_id"], password_hash)
    store.execute(
        "UPDATE reset_tokens SET used_at = ? WHERE token_hash = ?",
        (now, row["token_hash"]))
    store.execute("DELETE FROM sessions WHERE user_id = ?", (user["user_id"],))
    # The owner proved control of the account email, so an attacker's
    # failed-attempt lockout must not keep them out of their own account.
    store.execute("DELETE FROM login_attempts WHERE identifier IN (?, ?)",
                  (user["email"], user["username"]))
    _audit("password_reset_completed", user_id=user["user_id"],
           detail="all_sessions_invalidated")
    return AuthResult(True, info=(
        "Password updated. All existing sessions were signed out. Sign in "
        "with your new password."))


def change_password(user_id: str, current_password: str, new_password: str,
                    confirm_password: str,
                    current_token: str | None = None) -> AuthResult:
    """Self-service password change; invalidates every session and re-issues
    one for the caller when their current session was valid."""
    store = get_store()
    user = store.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
    if user is None:
        return AuthResult(False, errors=["Account not found."])
    if not security.verify_password(user["password_hash"], current_password):
        _audit("password_change_failed", user_id=user_id, actor_id=user_id)
        return AuthResult(False, errors=["Current password is incorrect."])
    errors = security.password_policy_errors(new_password)
    if new_password != confirm_password:
        errors.append("The two passwords do not match.")
    if errors:
        return AuthResult(False, errors=errors)
    if _password_reused(user, new_password):
        _audit("password_reuse_blocked", user_id=user_id, actor_id=user_id,
               detail="self_service_change")
        return AuthResult(False, errors=[_REUSE_ERROR])
    breach_error = _breach_screen(new_password, user_id=user_id)
    if breach_error:
        return AuthResult(False, errors=[breach_error])

    now = _iso(_now())
    password_hash = security.hash_password(new_password)
    store.execute(
        "UPDATE users SET password_hash = ?, auth_epoch = auth_epoch + 1, "
        "updated_at = ? WHERE user_id = ?",
        (password_hash, now, user_id))
    _record_password_history(user_id, password_hash)
    store.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    store.execute("DELETE FROM login_attempts WHERE identifier IN (?, ?)",
                  (user["email"], user["username"]))
    _audit("password_changed", user_id=user_id, actor_id=user_id,
           detail="all_sessions_invalidated")
    fresh = store.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
    token = _create_session(fresh, remember=False) if current_token else None
    return AuthResult(True, user=_public_user(fresh), token=token, info=(
        "Password changed. Other sessions were signed out."))


# ------------------------------------------------------- email verification
def verify_email(token: str) -> AuthResult:
    store = get_store()
    row = store.query_one(
        "SELECT * FROM verify_tokens WHERE token_hash = ?",
        (security.token_digest(token),))
    invalid = AuthResult(False, errors=[
        "This verification link is invalid, expired or already used. Sign "
        "in and request a new verification email from your account panel."])
    if row is None or row["used_at"] is not None:
        return invalid
    if row["expires_at"] < _iso(_now()):
        return invalid
    user = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (row["user_id"],))
    if user is None:
        return invalid
    now = _iso(_now())
    store.execute(
        "UPDATE verify_tokens SET used_at = ? WHERE token_hash = ?",
        (now, row["token_hash"]))
    if not user["email_verified"]:
        store.execute(
            "UPDATE users SET email_verified = 1, updated_at = ? "
            "WHERE user_id = ?", (now, user["user_id"]))
        _audit("email_verified", user_id=user["user_id"])
    return AuthResult(True, user=_public_user(user), info=(
        "Email address verified. You can sign in to your workspace."))


def resend_verification(email: str) -> AuthResult:
    """Rate-limited resend; the response does not reveal account existence."""
    generic = ("If an unverified account exists for that address, a new "
               "verification link has been sent.")
    email = email.strip().lower()
    if not security.valid_email(email):
        return AuthResult(False, errors=["Enter a valid email address."])
    store = get_store()
    user = store.query_one("SELECT * FROM users WHERE email = ?", (email,))
    if user is not None and not user["email_verified"] \
            and user["account_status"] == "active":
        latest = store.query_one(
            "SELECT created_at FROM verify_tokens WHERE user_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1", (user["user_id"],))
        cooldown_ok = True
        if latest is not None:
            age = (_now() - datetime.fromisoformat(
                latest["created_at"])).total_seconds()
            cooldown_ok = age >= auth_settings.VERIFY_RESEND_COOLDOWN_SECONDS
        if cooldown_ok:
            store.execute(
                "UPDATE verify_tokens SET used_at = ? WHERE user_id = ? "
                "AND used_at IS NULL", (_iso(_now()), user["user_id"]))
            token = security.new_token()
            store.execute(
                "INSERT INTO verify_tokens (token_hash, user_id, created_at, "
                "expires_at) VALUES (?, ?, ?, ?)",
                (security.token_digest(token), user["user_id"], _iso(_now()),
                 _iso(_now() + timedelta(
                     hours=auth_settings.VERIFY_TOKEN_TTL_HOURS))))
            delivery = mail.send_verification(email, token)
            _audit("verification_requested", user_id=user["user_id"],
                   detail=f"resend mode={delivery.mode}")
            if delivery.mode == "outbox":
                return AuthResult(True, info=(
                    generic + " Email delivery is not configured for real "
                    "sending: the message is in the development outbox "
                    "(results/mail_outbox/)."))
    return AuthResult(True, info=generic)


# ----------------------------------------------------- two-factor (TOTP)
def _totp_credential(user_id: str):
    return get_store().query_one(
        "SELECT * FROM totp_credentials WHERE user_id = ? AND enabled = 1",
        (user_id,))


def totp_enabled(user_id: str) -> bool:
    return _totp_credential(user_id) is not None


def complete_totp_login(mfa_token: str, code: str) -> AuthResult:
    """Second login factor. The challenge is single-use, short-lived and
    attempt-limited; wrong codes count towards the identifier lockout."""
    invalid = AuthResult(False, errors=[
        "This sign-in request expired or was already used. Start again "
        "from the sign-in page."])
    if not mfa_token or not isinstance(mfa_token, str):
        return invalid
    store = get_store()
    digest = security.token_digest(mfa_token)
    challenge = store.query_one(
        "SELECT * FROM totp_challenges WHERE token_hash = ?", (digest,))
    if challenge is None:
        return invalid
    if challenge["expires_at"] < _iso(_now()):
        store.execute("DELETE FROM totp_challenges WHERE token_hash = ?",
                      (digest,))
        return invalid
    user = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (challenge["user_id"],))
    if (user is None or user["account_status"] != "active"
            or user["auth_epoch"] != challenge["auth_epoch"]):
        store.execute("DELETE FROM totp_challenges WHERE token_hash = ?",
                      (digest,))
        return invalid
    cred = _totp_credential(user["user_id"])
    if cred is None:  # MFA removed between the two stages
        store.execute("DELETE FROM totp_challenges WHERE token_hash = ?",
                      (digest,))
        return invalid
    if challenge["attempts"] >= auth_settings.TOTP_MAX_ATTEMPTS:
        store.execute("DELETE FROM totp_challenges WHERE token_hash = ?",
                      (digest,))
        _audit("totp_challenge_exhausted", user_id=user["user_id"])
        return AuthResult(False, errors=[
            "Too many incorrect codes. Start again from the sign-in page."])
    locked = max(_lockout_remaining(user["email"]),
                 _lockout_remaining(user["username"]))
    if locked > 0:
        return AuthResult(False, errors=[
            f"Too many failed attempts. Try again in {locked} minute(s)."])
    secret = mfa.decrypt_secret(cred["secret_encrypted"])
    if secret is None or not mfa.verify_code(secret, code):
        store.execute(
            "UPDATE totp_challenges SET attempts = attempts + 1 "
            "WHERE token_hash = ?", (digest,))
        # Record under both identifiers so the lockout applies whichever
        # one the next sign-in attempt uses.
        for ident in (user["email"], user["username"]):
            _record_attempt(ident, False)
        _audit("totp_login_failed", user_id=user["user_id"])
        return AuthResult(False, errors=["Invalid authentication code."])
    store.execute("DELETE FROM totp_challenges WHERE user_id = ?",
                  (user["user_id"],))
    for ident in (user["email"], user["username"]):
        _record_attempt(ident, True)
    token = _create_session(user, bool(challenge["remember"]))
    store.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?",
                  (_iso(_now()), user["user_id"]))
    _audit("login_success", user_id=user["user_id"],
           detail=f"remember={int(challenge['remember'])} mfa=totp")
    return AuthResult(True, user=_public_user(user), token=token)


def begin_totp_enrollment(user_id: str) -> AuthResult:
    """Create a pending (not yet enabled) credential and return the setup
    material to the authenticated owner: QR image, otpauth URI and the
    manual-entry key. The secret is shown once, never logged."""
    if not mfa.available():
        return AuthResult(False, errors=[
            "Two-factor authentication is unavailable: the operator has "
            "not configured a valid AI_SOC_TOTP_KEY."])
    store = get_store()
    user = store.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
    if user is None:
        return AuthResult(False, errors=["Account not found."])
    if _totp_credential(user_id) is not None:
        return AuthResult(False, errors=[
            "Two-factor authentication is already enabled. Disable it "
            "first if you need to enrol a new device."])
    secret = mfa.new_secret()
    store.execute(
        "INSERT INTO totp_credentials (user_id, secret_encrypted, enabled, "
        "created_at) VALUES (?, ?, 0, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "secret_encrypted = excluded.secret_encrypted, enabled = 0, "
        "enrolled_at = NULL, created_at = excluded.created_at",
        (user_id, mfa.encrypt_secret(secret), _iso(_now())))
    uri = mfa.provisioning_uri(secret, user["email"])
    return AuthResult(True, payload={
        "secret": secret, "uri": uri, "qr_png": mfa.qr_png(uri)})


def confirm_totp_enrollment(user_id: str, code: str) -> AuthResult:
    if not mfa.available():
        return AuthResult(False, errors=[
            "Two-factor authentication is unavailable: the operator key "
            "(AI_SOC_TOTP_KEY) is not configured."])
    store = get_store()
    row = store.query_one(
        "SELECT * FROM totp_credentials WHERE user_id = ? AND enabled = 0",
        (user_id,))
    if row is None:
        return AuthResult(False, errors=[
            "No enrolment is in progress. Start again from the account "
            "panel."])
    secret = mfa.decrypt_secret(row["secret_encrypted"])
    if secret is None or not mfa.verify_code(secret, code):
        return AuthResult(False, errors=[
            "That code did not match. Check your authenticator app and "
            "enter its current code."])
    store.execute(
        "UPDATE totp_credentials SET enabled = 1, enrolled_at = ? "
        "WHERE user_id = ?", (_iso(_now()), user_id))
    _audit("totp_enrolled", user_id=user_id, actor_id=user_id)
    return AuthResult(True, info=(
        "Two-factor authentication is now enabled. Every sign-in will "
        "require a code from your authenticator app."))


def cancel_totp_enrollment(user_id: str) -> None:
    get_store().execute(
        "DELETE FROM totp_credentials WHERE user_id = ? AND enabled = 0",
        (user_id,))


def disable_totp(user_id: str, code: str) -> AuthResult:
    """Self-service removal requires a current code, so a hijacked session
    cannot silently strip the second factor."""
    cred = _totp_credential(user_id)
    if cred is None:
        return AuthResult(False, errors=[
            "Two-factor authentication is not enabled on this account."])
    if not mfa.available():
        return AuthResult(False, errors=[
            "The operator key (AI_SOC_TOTP_KEY) is not configured, so codes "
            "cannot be checked. Ask an administrator to remove 2FA from "
            "User Management."])
    secret = mfa.decrypt_secret(cred["secret_encrypted"])
    if secret is None or not mfa.verify_code(secret, code):
        _audit("totp_disable_failed", user_id=user_id, actor_id=user_id)
        return AuthResult(False, errors=["Invalid authentication code."])
    store = get_store()
    store.execute("DELETE FROM totp_credentials WHERE user_id = ?", (user_id,))
    store.execute("DELETE FROM totp_challenges WHERE user_id = ?", (user_id,))
    _audit("totp_disabled", user_id=user_id, actor_id=user_id)
    return AuthResult(True, info="Two-factor authentication was disabled.")


def admin_disable_totp(actor: dict, user_id: str) -> AuthResult:
    """Recovery path when a user loses their authenticator device."""
    rbac.require(actor["role"], "manage_users")
    store = get_store()
    target = store.query_one(
        "SELECT username FROM users WHERE user_id = ?", (user_id,))
    if target is None:
        return AuthResult(False, errors=["User not found."])
    if store.query_one(
            "SELECT user_id FROM totp_credentials WHERE user_id = ?",
            (user_id,)) is None:
        return AuthResult(True, info=(
            "Two-factor authentication is not enrolled for this account."))
    store.execute("DELETE FROM totp_credentials WHERE user_id = ?", (user_id,))
    store.execute("DELETE FROM totp_challenges WHERE user_id = ?", (user_id,))
    _audit("totp_disabled_by_admin", user_id=user_id,
           actor_id=actor["user_id"])
    return AuthResult(True, info=(
        f"Two-factor authentication was removed for {target['username']}. "
        "They can sign in with their password and re-enrol."))


# ------------------------------------------------- administration (audited)
def list_users(actor: dict) -> list[dict]:
    rbac.require(actor["role"], "manage_users")
    rows = get_store().query("SELECT * FROM users ORDER BY created_at DESC")
    return [_public_user(r) for r in rows]


def set_role(actor: dict, target_user_id: str, new_role: str) -> AuthResult:
    rbac.require(actor["role"], "manage_users")
    if new_role not in rbac.ROLES:
        return AuthResult(False, errors=["Unknown role."])
    if target_user_id == actor["user_id"]:
        return AuthResult(False, errors=[
            "You cannot change your own role. Ask another administrator."])
    store = get_store()
    target = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (target_user_id,))
    if target is None:
        return AuthResult(False, errors=["User not found."])
    if target["role"] == new_role:
        return AuthResult(True, info="Role unchanged.")
    store.execute(
        "UPDATE users SET role = ?, updated_at = ? WHERE user_id = ?",
        (new_role, _iso(_now()), target_user_id))
    _audit("role_changed", user_id=target_user_id, actor_id=actor["user_id"],
           detail=f"{target['role']} -> {new_role}")
    return AuthResult(True, info=(
        f"{target['username']} is now {rbac.ROLE_LABELS[new_role]}."))


def set_account_status(actor: dict, target_user_id: str,
                       status: str) -> AuthResult:
    rbac.require(actor["role"], "manage_users")
    if status not in ("active", "disabled"):
        return AuthResult(False, errors=["Unknown account status."])
    if target_user_id == actor["user_id"]:
        return AuthResult(False, errors=[
            "You cannot disable your own account."])
    store = get_store()
    target = store.query_one(
        "SELECT * FROM users WHERE user_id = ?", (target_user_id,))
    if target is None:
        return AuthResult(False, errors=["User not found."])
    store.execute(
        "UPDATE users SET account_status = ?, updated_at = ? WHERE user_id = ?",
        (status, _iso(_now()), target_user_id))
    if status == "disabled":
        store.execute("DELETE FROM sessions WHERE user_id = ?",
                      (target_user_id,))
    _audit("account_status_changed", user_id=target_user_id,
           actor_id=actor["user_id"],
           detail=f"{target['account_status']} -> {status}")
    return AuthResult(True, info=(
        f"{target['username']} is now {status}."))


def list_sessions(actor: dict, user_id: str | None = None) -> list[dict]:
    rbac.require(actor["role"], "manage_sessions")
    sql = ("SELECT s.session_id, s.user_id, u.username, s.remember, "
           "s.created_at, s.expires_at, s.last_seen_at FROM sessions s "
           "JOIN users u ON u.user_id = s.user_id")
    params: tuple = ()
    if user_id:
        sql += " WHERE s.user_id = ?"
        params = (user_id,)
    sql += " ORDER BY s.last_seen_at DESC"
    return [dict(r) for r in get_store().query(sql, params)]


def revoke_session(actor: dict, session_id: str) -> AuthResult:
    rbac.require(actor["role"], "manage_sessions")
    store = get_store()
    row = store.query_one(
        "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
    if row is None:
        return AuthResult(False, errors=["Session not found (already ended)."])
    store.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    _audit("session_revoked", user_id=row["user_id"],
           actor_id=actor["user_id"])
    return AuthResult(True, info="Session revoked.")


def audit_events(actor: dict, limit: int = 200) -> list[dict]:
    rbac.require(actor["role"], "view_audit")
    rows = get_store().query(
        "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (int(limit),))
    return [dict(r) for r in rows]


def provision_admin(*, full_name: str, email: str, username: str,
                    password: str) -> AuthResult:
    """Operator-run initial administrator creation (scripts/provision_admin.py).
    Never exposed through the web registration form."""
    errors = []
    full_name = security.clean_text(full_name, 100)
    email = email.strip().lower()
    username = username.strip().lower()
    if not full_name:
        errors.append("Enter the administrator's full name.")
    if not security.valid_email(email):
        errors.append("Enter a valid email address.")
    if not security.valid_username(username):
        errors.append("Username must be 3-32 characters (letters, numbers, "
                      "dots, underscores, hyphens).")
    errors.extend(security.password_policy_errors(password))
    if errors:
        return AuthResult(False, errors=errors)
    store = get_store()
    if store.query_one(
            "SELECT user_id FROM users WHERE email = ? OR username = ?",
            (email, username)):
        return AuthResult(False, errors=[
            "A user with that email or username already exists. Use the "
            "User Management page to change their role instead."])
    user_id = security.new_user_id()
    now = _iso(_now())
    password_hash = security.hash_password(password)
    store.execute(
        "INSERT INTO users (user_id, full_name, email, username, organisation, "
        "password_hash, role, email_verified, account_status, auth_epoch, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, '', ?, 'administrator', "
        "1, 'active', 1, ?, ?)",
        (user_id, full_name, email, username, password_hash, now, now))
    _record_password_history(user_id, password_hash)
    _audit("admin_provisioned", user_id=user_id, actor_id=user_id,
           detail="local operator provisioning")
    row = store.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
    return AuthResult(True, user=_public_user(row), info=(
        "Administrator account created (email marked verified because it "
        "was provisioned locally by the operator)."))