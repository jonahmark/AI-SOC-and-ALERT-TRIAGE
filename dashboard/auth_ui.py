"""Authentication experience for AI-SOC.

Branded sign-in / registration / recovery pages that share the dashboard's
design system, the server-side session gate, remember-me cookie bridge and
the signed-in account panel. All credential decisions are made by
src.auth.service (argon2id, rate limiting, single-use tokens); this module
only renders forms and their real results.
"""
from __future__ import annotations

import html

import streamlit as st

from src.auth import mfa, rbac, security, service
from src.auth import settings as auth_settings
from dashboard import components as ui

AUTH_PAGES = ("login", "signup", "forgot", "reset", "verify")

# ---------------------------------------------------------------------------
# Remember-me cookie bridge (CCv2). st.context.cookies is read-only, so a
# tiny inline component writes the opaque session cookie client-side. The
# token is a random server-side session id (revocable, expiring); it is not
# a credential and grants nothing without a matching live session row.
# SameSite=Strict; Secure when the app is served over HTTPS.
# ---------------------------------------------------------------------------
_COOKIE_BRIDGE = st.components.v2.component(
    "soc_session_cookie",
    html="<div></div>",
    js="""
export default function (component) {
  const { data } = component
  const state = (data && data.state) || null
  if (!state || !state.name) return
  if (state.delete) {
    document.cookie = state.name + "=; path=/; max-age=0; SameSite=Strict"
    return
  }
  let cookie = state.name + "=" + encodeURIComponent(state.value) +
    "; path=/; max-age=" + state.max_age + "; SameSite=Strict"
  if (state.secure) cookie += "; Secure"
  document.cookie = cookie
}
""",
)


def _cookie_secure() -> bool:
    try:
        return str(st.context.url).startswith("https")
    except Exception:
        return False


def render_cookie_bridge() -> None:
    """Mount once per run when a cookie write/delete is pending. Idempotent:
    the desired state stays in session state until replaced."""
    state = st.session_state.get("_cookie_state")
    if not state:
        return
    payload = {"name": auth_settings.SESSION_COOKIE, "secure": _cookie_secure()}
    payload.update(state)
    _COOKIE_BRIDGE(key="soc_cookie_bridge", data={"state": payload}, height=0)


def _set_remember_cookie(token: str) -> None:
    st.session_state["_cookie_state"] = {
        "value": token,
        "max_age": int(auth_settings.REMEMBER_ME_DAYS) * 86400,
    }


def _delete_remember_cookie() -> None:
    st.session_state["_cookie_state"] = {"delete": True}


# ---------------------------------------------------------------------------
# Session gate
# ---------------------------------------------------------------------------
def _request_token() -> str | None:
    token = st.session_state.get("auth_token")
    if isinstance(token, str) and token:
        return token
    try:
        cookie = st.context.cookies.get(auth_settings.SESSION_COOKIE)
    except Exception:
        return None
    return cookie if isinstance(cookie, str) and cookie else None


def current_user() -> dict | None:
    """The signed-in user for this request, or None. Validates the session
    against the database on every run (expiry, revocation, password-change
    epoch, account status)."""
    token = _request_token()
    user = service.validate_session(token)
    if user is not None and not st.session_state.get("auth_token"):
        st.session_state["auth_token"] = token  # adopt remember-me cookie
    if user is None:
        if st.session_state.get("auth_token"):
            st.session_state.pop("auth_token", None)
        if token:  # stale remember-me cookie: remove it client-side
            _delete_remember_cookie()
    return user


def require_user() -> dict:
    """Gate for app.py: returns the user or renders the auth pages and
    stops. The sidebar and dashboard never render before authentication."""
    # Capture before current_user() clears a stale token, so an expired or
    # revoked session still earns an honest notice on the login page.
    had_token = bool(_request_token())
    user = current_user()
    if user is not None:
        # The login widget is not rendered on authenticated runs, so its
        # residual value can be cleared safely here.
        st.session_state.pop("login_password", None)
        return user
    if had_token:
        st.session_state["auth_notice"] = (
            "Your session expired or was signed out. Sign in again to "
            "continue.", "info")
    render_auth_page()
    st.stop()


def logout() -> None:
    service.logout(_request_token())
    st.session_state.pop("auth_token", None)
    st.session_state.pop("_mfa_token", None)
    st.session_state.pop("_mfa_remember", None)
    _delete_remember_cookie()
    st.query_params.clear()
    st.rerun()


def _goto(page: str) -> None:
    st.query_params.clear()
    st.query_params["auth"] = page
    st.rerun()


def _notice() -> None:
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        text, kind = notice
        getattr(st, kind)(text)


# ---------------------------------------------------------------------------
# Auth-page styling (same palette and radii as the dashboard)
# ---------------------------------------------------------------------------
_AUTH_CSS = f"""
<style>
section[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

div[data-testid="stMainBlockContainer"] {{
  max-width: 1180px; padding-top: 4vh;
}}

.st-key-auth_brand {{ padding: 1.2rem 1.4rem 1rem 0.4rem; }}
.soc-auth-logo {{
  width: 52px; height: 52px; border-radius: 15px; display: flex;
  align-items: center; justify-content: center; margin-bottom: 18px;
  background: linear-gradient(135deg, {ui.ACCENT} 0%, {ui.ACCENT_2} 100%);
  box-shadow: 0 8px 28px {ui.ACCENT}33;
}}
.soc-auth-brand-name {{
  font-size: 1.9rem; font-weight: 800; letter-spacing: 0.02em;
  color: {ui.TEXT}; line-height: 1.1;
}}
.soc-auth-brand-sub {{
  color: {ui.ACCENT}; font-size: 0.92rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px;
}}
.soc-auth-welcome {{
  color: {ui.TEXT_MUTED}; font-size: 0.98rem; margin: 16px 0 22px 0;
  max-width: 42ch; line-height: 1.55;
}}
.soc-auth-feature {{
  display: flex; gap: 12px; align-items: flex-start; padding: 10px 0;
  border-top: 1px solid {ui.BORDER};
}}
.soc-auth-feature .fi {{
  flex: 0 0 auto; width: 32px; height: 32px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  background: {ui.CARD_ELEVATED}; border: 1px solid {ui.BORDER};
}}
.soc-auth-feature b {{ color: {ui.TEXT}; font-size: 0.9rem; display: block; }}
.soc-auth-feature span {{ color: {ui.TEXT_MUTED}; font-size: 0.8rem; }}
.soc-auth-visual {{ margin-top: 20px; opacity: 0.9; }}
.soc-auth-foot {{
  margin-top: 18px; color: {ui.TEXT_MUTED}; font-size: 0.74rem;
  border-top: 1px solid {ui.BORDER}; padding-top: 12px;
}}

.st-key-auth_card {{
  background: linear-gradient(180deg, {ui.CARD} 0%, #0E1930 100%);
  border: 1px solid {ui.BORDER}; border-radius: 18px;
  padding: 1.9rem 2rem 1.6rem 2rem;
  box-shadow: 0 18px 50px #00000059;
}}
.soc-auth-title {{ font-size: 1.35rem; font-weight: 750; color: {ui.TEXT}; }}
.soc-auth-sub {{ color: {ui.TEXT_MUTED}; font-size: 0.86rem; margin: 4px 0 18px 0; }}
.soc-auth-links {{ display: flex; justify-content: space-between; gap: 8px;
  margin-top: 14px; flex-wrap: wrap; }}
.soc-auth-alt {{ text-align: center; color: {ui.TEXT_MUTED};
  font-size: 0.85rem; margin-top: 16px; }}

.soc-strength {{ display: flex; gap: 5px; margin: 8px 0 4px 0; }}
.soc-strength i {{ height: 5px; border-radius: 3px; flex: 1;
  background: {ui.BORDER}; }}
.soc-strength-label {{ font-size: 0.76rem; color: {ui.TEXT_MUTED}; }}
.soc-req {{ display: flex; align-items: center; gap: 7px; font-size: 0.78rem;
  color: {ui.TEXT_MUTED}; padding: 2px 0; }}
.soc-req.met {{ color: #22C55E; }}

.sac {{ display: flex; gap: 10px; align-items: center; padding: 10px 12px;
  background: {ui.CARD}; border: 1px solid {ui.BORDER}; border-radius: 12px;
  margin: 10px 0 8px 0; }}
.sac-avatar {{ flex: 0 0 auto; width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, {ui.ACCENT}2E, {ui.ACCENT_2}2E);
  border: 1px solid {ui.BORDER}; color: {ui.ACCENT}; font-weight: 700; }}
.sac-name {{ color: {ui.TEXT}; font-weight: 650; font-size: 0.88rem;
  line-height: 1.2; overflow-wrap: anywhere; }}
.sac-role {{ font-size: 0.72rem; color: {ui.TEXT_MUTED}; }}

@media (max-width: 768px) {{
  div[data-testid="stMainBlockContainer"] {{ padding-top: 1.5vh; }}
  .st-key-auth_brand {{ padding: 0.4rem 0.2rem; }}
  .soc-auth-visual {{ display: none; }}
  .st-key-auth_card {{ padding: 1.3rem 1.1rem; }}
}}
</style>
"""

_VISUAL_SVG = f"""
<svg class="soc-auth-visual" viewBox="0 0 440 170" width="100%" height="170"
     aria-hidden="true" fill="none">
  <defs>
    <linearGradient id="soclg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{ui.ACCENT}"/>
      <stop offset="1" stop-color="{ui.ACCENT_2}"/>
    </linearGradient>
  </defs>
  <g stroke="{ui.BORDER}" stroke-width="1">
    <path d="M40 130 120 60 210 110 300 40 400 90"/>
    <path d="M120 60 180 140 210 110 260 150 300 40"/>
    <path d="M40 130 180 140 M300 40 340 130 400 90 M210 110 340 130"/>
  </g>
  <g fill="url(#soclg)">
    <circle cx="40" cy="130" r="4"/><circle cx="120" cy="60" r="6"/>
    <circle cx="210" cy="110" r="8"/><circle cx="300" cy="40" r="5"/>
    <circle cx="400" cy="90" r="4"/><circle cx="180" cy="140" r="3.5"/>
    <circle cx="260" cy="150" r="3"/><circle cx="340" cy="130" r="4.5"/>
  </g>
  <circle cx="210" cy="110" r="15" stroke="{ui.ACCENT}" stroke-opacity="0.35"/>
  <circle cx="120" cy="60" r="11" stroke="{ui.ACCENT_2}" stroke-opacity="0.35"/>
</svg>
"""


def _brand_panel() -> None:
    features = [
        ("activity", "Two-stage detection",
         "Isolation Forest anomaly scoring, then attack classification - "
         "kept as separate, labelled decisions."),
        ("bell", "Alert triage workflow",
         "Transparent severity mapping and risk scores with a full "
         "investigation trail."),
        ("brain", "Explainable AI",
         "Global feature importance and per-alert SHAP explanations, "
         "clearly distinguished."),
        ("shield-check", "Honest by design",
         "Every metric traces to a real model run on the static NSL-KDD "
         "benchmark - nothing simulated."),
    ]
    rows = "".join(
        f'<div class="soc-auth-feature"><div class="fi">{ui.icon(name, 16, ui.ACCENT)}</div>'
        f'<div><b>{title}</b><span>{body}</span></div></div>'
        for name, title, body in features)
    st.markdown(
        f'<div class="soc-auth-logo">{ui.icon("shield-check", 28, "#06121F", 2.0)}</div>'
        '<div class="soc-auth-brand-name">AI-SOC</div>'
        '<div class="soc-auth-brand-sub">Intelligent Security Operations Platform</div>'
        '<div class="soc-auth-welcome">Secure access to your security operations '
        'workspace.</div>'
        f'{rows}{_VISUAL_SVG}'
        '<div class="soc-auth-foot">Research prototype &middot; static NSL-KDD '
        'benchmark &middot; not a live enterprise IDS/SIEM</div>',
        unsafe_allow_html=True)


def _auth_shell():
    """Split layout: brand left, form card right."""
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    left, right = st.columns([1.02, 1], gap="large")
    with left, st.container(key="auth_brand"):
        _brand_panel()
    return right


def _strength_meter(password: str) -> None:
    score, label = security.password_strength(password)
    colors = ["#EF4444", "#F97316", "#EAB308", "#38BDF8", "#22C55E"]
    bars = "".join(
        f'<i style="{"background:" + colors[score] if i < score and password else ""}"></i>'
        for i in range(4))
    checks = "".join(
        f'<div class="soc-req{" met" if met else ""}">'
        f'{ui.icon("check-circle" if met else "x-circle", 13, "#22C55E" if met else ui.TEXT_MUTED)}'
        f'{text}</div>'
        for text, met in security.password_checks(password))
    st.markdown(
        f'<div class="soc-strength">{bars}</div>'
        f'<div class="soc-strength-label">Strength: <b style="color:'
        f'{colors[score] if password else ui.TEXT_MUTED}">{label}</b> '
        f'&middot; passphrases welcome</div>{checks}',
        unsafe_allow_html=True)


def _mail_mode_note() -> None:
    mode = auth_settings.EMAIL_MODE
    if mode == "outbox":
        ui.note("Email delivery is not configured for real sending. Messages "
                "are written to the development outbox "
                "<code>results/mail_outbox/</code> - open the newest "
                "<code>.eml</code> file to find your link.", "info", ui.ACCENT)
    elif mode == "off":
        ui.note("Email delivery is disabled in this deployment "
                "(<code>AI_SOC_EMAIL_MODE=off</code>). Recovery and "
                "verification messages cannot be delivered until SMTP is "
                "configured.", "alert", "#EAB308")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def _login_page() -> None:
    # Signup widgets are not rendered here; clear their residual secrets.
    st.session_state.pop("su_password", None)
    st.session_state.pop("su_confirm", None)
    if st.session_state.get("_mfa_token"):
        _mfa_login_step()
        return
    st.markdown('<div class="soc-auth-title">Sign in</div>'
                '<div class="soc-auth-sub">Use your account credentials to '
                'access the workspace.</div>', unsafe_allow_html=True)
    _notice()
    with st.form("login_form", border=False):
        st.text_input("Email or username", key="login_identifier",
                      placeholder="name@example.com", autocomplete="username")
        st.text_input("Password", type="password", key="login_password",
                      autocomplete="current-password",
                      help="Use the eye icon to show or hide what you typed.")
        st.checkbox(
            f"Remember me on this device for {auth_settings.REMEMBER_ME_DAYS} days",
            key="login_remember",
            help="Stores an opaque, revocable session cookie. Without it, "
                 "the session ends when you close or refresh the browser.")
        submitted = st.form_submit_button(
            "Sign In to AI-SOC", type="primary", icon=":material/login:",
            width="stretch")
    if submitted:
        with st.spinner("Authenticating..."):
            result = service.authenticate(
                st.session_state.get("login_identifier", ""),
                st.session_state.get("login_password", ""),
                remember=bool(st.session_state.get("login_remember")))
        if result.ok:
            if result.mfa_token:
                st.session_state["_mfa_token"] = result.mfa_token
                st.session_state["_mfa_remember"] = bool(
                    st.session_state.get("login_remember"))
                st.session_state.pop("login_password", None)
                st.rerun()
            st.session_state["auth_token"] = result.token
            if st.session_state.get("login_remember"):
                _set_remember_cookie(result.token)
            st.query_params.clear()
            st.rerun()
        for error in result.errors:
            st.error(error)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Forgot password?", key="auth_to_forgot", type="tertiary",
                     icon=":material/lock_open:", width="stretch"):
            _goto("forgot")
    with c2:
        if st.button("Create an account", key="auth_to_signup", type="tertiary",
                     icon=":material/person_add:", width="stretch"):
            _goto("signup")


def _mfa_login_step() -> None:
    st.markdown('<div class="soc-auth-title">Two-factor verification</div>'
                '<div class="soc-auth-sub">Your password was accepted. Enter '
                'the 6-digit code from your authenticator app to finish '
                'signing in.</div>', unsafe_allow_html=True)
    with st.form("mfa_form", border=False):
        st.text_input("Authentication code", key="mfa_code",
                      placeholder="123456", max_chars=6,
                      autocomplete="one-time-code")
        submitted = st.form_submit_button(
            "Verify code", type="primary", icon=":material/verified_user:",
            width="stretch")
    if submitted:
        result = service.complete_totp_login(
            st.session_state.get("_mfa_token", ""),
            st.session_state.get("mfa_code", ""))
        if result.ok:
            st.session_state["auth_token"] = result.token
            if st.session_state.pop("_mfa_remember", False):
                _set_remember_cookie(result.token)
            st.session_state.pop("_mfa_token", None)
            st.session_state.pop("mfa_code", None)
            st.query_params.clear()
            st.rerun()
        for error in result.errors:
            st.error(error)
    if st.button("Back to sign in", key="mfa_back", type="tertiary",
                 icon=":material/arrow_back:", width="stretch"):
        st.session_state.pop("_mfa_token", None)
        st.session_state.pop("_mfa_remember", None)
        st.session_state.pop("mfa_code", None)
        st.rerun()


def _signup_page() -> None:
    st.markdown('<div class="soc-auth-title">Create your account</div>'
                '<div class="soc-auth-sub">New accounts start with the '
                'Researcher / Viewer role (least privilege). An administrator '
                'can grant additional access.</div>', unsafe_allow_html=True)
    _notice()
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Full name", key="su_full_name", placeholder="Ada Lovelace")
    with c2:
        st.text_input("Username", key="su_username", placeholder="ada",
                      help="3-32 characters: letters, numbers, dots, "
                           "underscores or hyphens.")
    st.text_input("Email address", key="su_email",
                  placeholder="name@example.com", autocomplete="email")
    st.text_input("Organisation / institution (optional)", key="su_org",
                  placeholder="University or team")
    password = st.text_input("Password", type="password", key="su_password",
                             autocomplete="new-password")
    _strength_meter(password)
    confirm = st.text_input("Confirm password", type="password", key="su_confirm",
                            autocomplete="new-password")
    if confirm:
        if confirm == password:
            st.caption(":material/check_circle: Passwords match.")
        else:
            st.caption(":material/error: The two passwords do not match yet.")
    if st.button("Create account", key="su_submit", type="primary",
                 icon=":material/person_add:", width="stretch"):
        with st.spinner("Creating your account..."):
            result = service.register(
                full_name=st.session_state.get("su_full_name", ""),
                email=st.session_state.get("su_email", ""),
                username=st.session_state.get("su_username", ""),
                organisation=st.session_state.get("su_org", ""),
                password=password, confirm_password=confirm)
        if result.ok:
            st.session_state["auth_notice"] = (result.info, "success")
            _goto("login")
        for error in result.errors:
            st.error(error)
    st.markdown('<div class="soc-auth-alt">Already registered?</div>',
                unsafe_allow_html=True)
    if st.button("Back to sign in", key="auth_to_login", type="tertiary",
                 icon=":material/arrow_back:", width="stretch"):
        _goto("login")


def _forgot_page() -> None:
    st.markdown('<div class="soc-auth-title">Password recovery</div>'
                '<div class="soc-auth-sub">Enter your account email. If it '
                'matches an account, we send a single-use recovery link '
                f'valid for {auth_settings.RESET_TOKEN_TTL_MINUTES} minutes.'
                '</div>', unsafe_allow_html=True)
    _notice()
    email = st.text_input("Email address", key="fp_email",
                          placeholder="name@example.com", autocomplete="email")
    if st.button("Send recovery link", key="fp_submit", type="primary",
                 icon=":material/mail:", width="stretch"):
        with st.spinner("Processing recovery request..."):
            result = service.request_password_reset(email)
        if result.ok:
            st.success(result.info)
        for error in result.errors:
            st.error(error)
    _mail_mode_note()
    if st.button("Back to sign in", key="auth_to_login", type="tertiary",
                 icon=":material/arrow_back:", width="stretch"):
        _goto("login")


def _reset_page() -> None:
    token = st.query_params.get("token", "")
    st.markdown('<div class="soc-auth-title">Choose a new password</div>'
                '<div class="soc-auth-sub">The recovery link is single-use. '
                'Changing your password signs out all existing sessions.'
                '</div>', unsafe_allow_html=True)
    if st.session_state.get("rp_done"):
        # Persist the success state until the user actually navigates: the
        # Continue button must still be rendered on the run that processes
        # its click. The password widgets are not instantiated here, so
        # their residual secrets can be cleared safely on every such run.
        st.session_state.pop("rp_new_password", None)
        st.session_state.pop("rp_confirm", None)
        st.success(st.session_state.get("rp_done_info", ""))
        if st.button("Continue to sign in", key="rp_continue", type="primary",
                     icon=":material/login:", width="stretch"):
            st.session_state.pop("rp_done", None)
            st.session_state.pop("rp_done_info", None)
            _goto("login")
        return
    if not token:
        st.error("No recovery token was supplied. Request a recovery link "
                 "from the sign-in page.")
        if st.button("Request a recovery link", key="auth_to_forgot",
                     type="primary", icon=":material/lock_open:",
                     width="stretch"):
            _goto("forgot")
        return
    password = st.text_input("New password", type="password",
                             key="rp_new_password", autocomplete="new-password")
    _strength_meter(password)
    confirm = st.text_input("Confirm new password", type="password",
                            key="rp_confirm", autocomplete="new-password")
    if st.button("Reset password", key="rp_submit", type="primary",
                 icon=":material/lock_reset:", width="stretch"):
        with st.spinner("Updating your password..."):
            result = service.reset_password(token, password, confirm)
        if result.ok:
            st.session_state["rp_done"] = True
            st.session_state["rp_done_info"] = result.info
            st.rerun()
        for error in result.errors:
            st.error(error)
    if st.button("Back to sign in", key="auth_to_login", type="tertiary",
                 icon=":material/arrow_back:", width="stretch"):
        _goto("login")


def _verify_page() -> None:
    token = st.query_params.get("token", "")
    st.markdown('<div class="soc-auth-title">Email verification</div>'
                '<div class="soc-auth-sub">Confirm the address on your '
                'account.</div>', unsafe_allow_html=True)
    if not token:
        st.error("No verification token was supplied. Sign in and use "
                 "'Resend verification email' in the account panel.")
    else:
        result = service.verify_email(token)
        if result.ok:
            st.success(result.info)
        else:
            for error in result.errors:
                st.error(error)
    if st.button("Continue to sign in", key="ve_continue", type="primary",
                 icon=":material/login:", width="stretch"):
        _goto("login")


def render_auth_page() -> None:
    page = st.query_params.get("auth", "login")
    if page not in AUTH_PAGES:
        page = "login"
    right = _auth_shell()
    with right, st.container(key="auth_card"):
        {"login": _login_page, "signup": _signup_page, "forgot": _forgot_page,
         "reset": _reset_page, "verify": _verify_page}[page]()
    render_cookie_bridge()


# ---------------------------------------------------------------------------
# Signed-in account panel (sidebar)
# ---------------------------------------------------------------------------
def sidebar_account(user: dict) -> None:
    if st.session_state.pop("_pw_changed", None):
        # The form widgets below have not been instantiated yet this run.
        for key in ("pw_current", "pw_new", "pw_confirm"):
            st.session_state.pop(key, None)
    initials = "".join(w[0] for w in user["full_name"].split()[:2]).upper() \
        or user["username"][:2].upper()
    role_color = rbac.ROLE_COLORS.get(user["role"], ui.TEXT_MUTED)
    verified = (f'{ui.icon("check-circle", 12, "#22C55E")} verified'
                if user["email_verified"]
                else f'{ui.icon("alert", 12, "#EAB308")} unverified')
    st.sidebar.markdown(
        f'<div class="sac"><div class="sac-avatar">{html.escape(initials)}</div>'
        f'<div style="min-width:0"><div class="sac-name">'
        f'{html.escape(user["full_name"])}</div>'
        f'<div class="sac-role"><span style="color:{role_color}">'
        f'{rbac.ROLE_LABELS[user["role"]]}</span> &middot; {verified}</div>'
        f'</div></div>', unsafe_allow_html=True)

    with st.sidebar.popover("Account & security", icon=":material/manage_accounts:",
                            width="stretch"):
        st.markdown(f"**{html.escape(user['full_name'])}**")
        ui.kv_table([
            ("Username", user["username"]),
            ("Email", user["email"]),
            ("Role", rbac.ROLE_LABELS[user["role"]]),
            ("Email status", "verified" if user["email_verified"]
             else "unverified"),
            ("Member since", ui.timestamp_label(user["created_at"])),
            ("Session expires", ui.timestamp_label(user.get("session_expires_at"))),
        ])
        if auth_settings.SESSION_IDLE_TIMEOUT_MINUTES > 0:
            st.caption("Sessions also end after "
                       f"{auth_settings.SESSION_IDLE_TIMEOUT_MINUTES} minutes "
                       "of inactivity.")
        if not user["email_verified"]:
            if auth_settings.EMAIL_MODE == "off":
                st.caption("Email delivery is disabled; verification becomes "
                           "available once SMTP is configured.")
            elif st.button("Resend verification email", key="resend_verify_btn",
                           icon=":material/mail:", width="stretch"):
                result = service.resend_verification(user["email"])
                st.info(result.info)
        st.divider()
        st.markdown("**Change password**")
        with st.form("change_password_form", border=False):
            st.text_input("Current password", type="password", key="pw_current",
                          autocomplete="current-password")
            new_pw = st.text_input("New password", type="password", key="pw_new",
                                   autocomplete="new-password")
            _strength_meter(new_pw)
            st.text_input("Confirm new password", type="password",
                          key="pw_confirm", autocomplete="new-password")
            if st.form_submit_button("Update password", type="primary",
                                     icon=":material/lock_reset:",
                                     width="stretch"):
                result = service.change_password(
                    user["user_id"],
                    st.session_state.get("pw_current", ""),
                    new_pw, st.session_state.get("pw_confirm", ""),
                    current_token=st.session_state.get("auth_token"))
                if result.ok:
                    if result.token:
                        st.session_state["auth_token"] = result.token
                    st.session_state["_pw_changed"] = True
                    st.session_state["_pw_notice"] = result.info
                    st.rerun()
                for error in result.errors:
                    st.error(error)
        notice = st.session_state.pop("_pw_notice", None)
        if notice:
            st.success(notice)
        st.caption("Changing your password signs out every other session "
                   "immediately. Recently used passwords cannot be reused.")
        st.divider()
        st.markdown("**Two-factor authentication**")
        _totp_panel(user)

    if st.sidebar.button("Sign out", key="logout_btn", icon=":material/logout:",
                         width="stretch"):
        logout()


def _totp_panel(user: dict) -> None:
    if not mfa.available():
        st.caption("Unavailable in this deployment: the operator has not "
                   "configured AI_SOC_TOTP_KEY. See docs/AUTHENTICATION.md "
                   "(Two-factor authentication).")
        return
    notice = st.session_state.pop("_totp_notice", None)
    if notice:
        getattr(st, notice[1])(notice[0])
    if service.totp_enabled(user["user_id"]):
        st.caption("Enabled. Every sign-in requires a code from your "
                   "authenticator app. Lost your device? An administrator "
                   "can remove 2FA from User Management.")
        with st.form("totp_disable_form", border=False):
            st.text_input("Current code to disable", key="totp_disable_code",
                          max_chars=6, autocomplete="one-time-code")
            if st.form_submit_button("Disable two-factor",
                                     icon=":material/no_encryption:",
                                     width="stretch"):
                result = service.disable_totp(
                    user["user_id"],
                    st.session_state.get("totp_disable_code", ""))
                if result.ok:
                    st.session_state["_totp_notice"] = (result.info, "success")
                    st.session_state.pop("totp_disable_code", None)
                    st.rerun()
                for error in result.errors:
                    st.error(error)
        return
    enroll = st.session_state.get("_totp_enroll")
    if enroll is None:
        st.caption("Add an authenticator app as a second sign-in factor "
                   "(recommended for administrator accounts).")
        if st.button("Set up two-factor", key="totp_begin",
                     icon=":material/phonelink_lock:", width="stretch"):
            result = service.begin_totp_enrollment(user["user_id"])
            if result.ok:
                st.session_state["_totp_enroll"] = result.payload
                st.rerun()
            for error in result.errors:
                st.error(error)
        return
    st.image(enroll["qr_png"], caption="Scan with your authenticator app",
             width=180)
    st.caption("Or enter this setup key manually:")
    st.code(enroll["secret"], language=None)
    with st.form("totp_confirm_form", border=False):
        st.text_input("Code from the app", key="totp_confirm_code",
                      max_chars=6, autocomplete="one-time-code")
        if st.form_submit_button("Verify and enable", type="primary",
                                 icon=":material/verified_user:",
                                 width="stretch"):
            result = service.confirm_totp_enrollment(
                user["user_id"], st.session_state.get("totp_confirm_code", ""))
            if result.ok:
                st.session_state.pop("_totp_enroll", None)
                st.session_state.pop("totp_confirm_code", None)
                st.session_state["_totp_notice"] = (result.info, "success")
                st.rerun()
            for error in result.errors:
                st.error(error)
    if st.button("Cancel setup", key="totp_cancel", width="stretch"):
        service.cancel_totp_enrollment(user["user_id"])
        st.session_state.pop("_totp_enroll", None)
        st.rerun()


def enforce_page_access(user: dict, page_key: str) -> bool:
    """Server-side page gate (defence in depth behind role-filtered nav)."""
    if rbac.page_allowed(user["role"], page_key):
        return True
    ui.empty_state(
        "Access restricted",
        f"The <b>{rbac.ROLE_LABELS[user['role']]}</b> role does not include "
        "this area. Ask an administrator if you need additional access.",
        "lock")
    return False