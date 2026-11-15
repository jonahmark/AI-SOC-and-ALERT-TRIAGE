"""Page 11 - User Management (Administrator only).

Role assignments, account status, active sessions and the security audit
trail. Every mutation goes through src.auth.service, which re-checks the
actor's permission server-side and records an audit event. Passwords and
tokens are never displayed or logged.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.auth import rbac, service
from src.auth import settings as auth_settings
from dashboard import auth_ui
from dashboard import components as ui


def _users_frame(users: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "username": u["username"],
        "full_name": u["full_name"],
        "email": u["email"],
        "role": rbac.ROLE_LABELS[u["role"]],
        "email_status": "verified" if u["email_verified"] else "unverified",
        "account": u["account_status"],
        "2fa": "on" if service.totp_enabled(u["user_id"]) else "off",
        "created": (u["created_at"] or "")[:19].replace("T", " "),
        "last_login": (u["last_login_at"] or "never")[:19].replace("T", " "),
    } for u in users])


def render() -> None:
    actor = auth_ui.current_user()
    if actor is None or not rbac.can(actor["role"], "manage_users"):
        ui.empty_state(
            "Access restricted",
            "User Management requires the <b>Administrator</b> role. "
            "Permissions are enforced on the server, not only in the "
            "navigation.", "lock")
        return

    users = service.list_users(actor)
    sessions = service.list_sessions(actor)
    ui.topbar(
        ["AI-SOC", "User Management"],
        "User Management",
        "Manage accounts, roles and sessions; review the security audit "
        "trail. All actions are permission-checked and recorded.",
        chips=[
            ui.chip("Accounts", f"{len(users):,}", "user"),
            ui.chip("Active sessions", f"{len(sessions):,}", "lock"),
            ui.chip("Signed in as", actor["username"], "shield-check"),
        ],
        refresh_key="um_refresh",
        icon_name="user",
    )

    tab_users, tab_sessions, tab_audit, tab_roles = st.tabs([
        ":material/group: Users",
        ":material/key: Active sessions",
        ":material/history: Security audit",
        ":material/admin_panel_settings: Roles & provisioning",
    ])

    # ------------------------------------------------------------------ users
    with tab_users:
        if "um_notice" in st.session_state:
            text, kind = st.session_state.pop("um_notice")
            if kind == "success":
                st.success(text)
            else:
                st.error(text)
        with ui.card("Accounts", "user",
                     "Public signups start as Researcher / Viewer; "
                     "administrators can provision any role",
                     key="um_accounts"):
            ui.show_table(_users_frame(users), height=320, key="um_users_table")

        with ui.card("Add a user", "person_add",
                     "Administrator-provisioned account; available even when "
                     "public registration is closed", key="um_add_user"):
            with st.form("um_add_user_form", border=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.text_input("Full name", key="nu_full_name",
                                  placeholder="Ada Lovelace")
                with c2:
                    st.text_input("Username", key="nu_username",
                                  placeholder="ada",
                                  help="3-32 characters: letters, numbers, "
                                       "dots, underscores or hyphens.")
                c3, c4 = st.columns(2)
                with c3:
                    st.text_input("Email address", key="nu_email",
                                  placeholder="name@example.com")
                with c4:
                    st.text_input("Organisation / institution (optional)",
                                  key="nu_org", placeholder="University or team")
                st.selectbox("Role", rbac.ROLES,
                             format_func=lambda r: rbac.ROLE_LABELS[r],
                             index=rbac.ROLES.index(rbac.DEFAULT_ROLE),
                             key="nu_role")
                c5, c6 = st.columns(2)
                with c5:
                    st.text_input("Initial password", type="password",
                                  key="nu_password",
                                  autocomplete="new-password")
                with c6:
                    st.text_input("Confirm initial password", type="password",
                                  key="nu_confirm",
                                  autocomplete="new-password")
                submitted = st.form_submit_button(
                    "Create account", type="primary",
                    icon=":material/person_add:", width="stretch")
            if submitted:
                result = service.admin_create_user(
                    actor,
                    full_name=st.session_state.get("nu_full_name", ""),
                    email=st.session_state.get("nu_email", ""),
                    username=st.session_state.get("nu_username", ""),
                    organisation=st.session_state.get("nu_org", ""),
                    role=st.session_state.get("nu_role", rbac.DEFAULT_ROLE),
                    password=st.session_state.get("nu_password", ""),
                    confirm_password=st.session_state.get("nu_confirm", ""))
                if result.ok:
                    st.session_state.pop("nu_password", None)
                    st.session_state.pop("nu_confirm", None)
                    st.session_state["um_notice"] = (result.info, "success")
                    st.rerun()
                for error in result.errors:
                    st.error(error)
            st.caption("The account starts unverified: a verification link is "
                       "emailed when delivery is configured. Passwords are "
                       "hashed with argon2id and never logged - share the "
                       "initial password over a channel you trust and ask the "
                       "user to change it after signing in.")

        with ui.card("Manage an account", "settings",
                     "Role and status changes apply immediately and are "
                     "audited", key="um_manage"):
            options = {f"{u['username']}  ({u['email']})": u for u in users}
            picked = st.selectbox("Account", list(options), key="um_pick")
            target = options[picked]
            is_self = target["user_id"] == actor["user_id"]
            if is_self:
                ui.note("This is your own account. You cannot change your "
                        "own role or disable yourself - ask another "
                        "administrator.", "info", ui.ACCENT)
            c1, c2 = st.columns([1.4, 1], vertical_alignment="bottom")
            with c1:
                new_role = st.selectbox(
                    "Role", rbac.ROLES,
                    format_func=lambda r: rbac.ROLE_LABELS[r],
                    index=rbac.ROLES.index(target["role"]),
                    key="um_role", disabled=is_self)
            with c2:
                if st.button("Apply role", key="um_apply_role",
                             icon=":material/save:", width="stretch",
                             disabled=is_self):
                    result = service.set_role(actor, target["user_id"],
                                              new_role)
                    if result.ok:
                        st.success(result.info)
                        st.rerun()
                    for error in result.errors:
                        st.error(error)
            c3, c4 = st.columns([1.4, 1], vertical_alignment="bottom")
            with c3:
                disabled_account = target["account_status"] == "disabled"
                st.markdown(
                    f"Current status: **{target['account_status']}**"
                    + (" - all sessions were revoked when disabled."
                       if disabled_account else ""))
            with c4:
                label = ("Enable account" if disabled_account
                         else "Disable account")
                if st.button(label, key="um_toggle_status",
                             icon=":material/toggle_on:" if disabled_account
                             else ":material/block:",
                             width="stretch", disabled=is_self):
                    new_status = "active" if disabled_account else "disabled"
                    result = service.set_account_status(
                        actor, target["user_id"], new_status)
                    if result.ok:
                        st.success(result.info)
                        st.rerun()
                    for error in result.errors:
                        st.error(error)
            st.caption(
                "Disabling an account revokes its sessions immediately. "
                "Role changes take effect on the user's next page load "
                "(sessions are re-validated against the database every "
                "request).")
            c5, c6 = st.columns([1.4, 1], vertical_alignment="bottom")
            with c5:
                mfa_on = service.totp_enabled(target["user_id"])
                st.markdown("Two-factor authentication: **"
                            + ("enabled" if mfa_on else "not enabled") + "**")
            with c6:
                if st.button("Remove 2FA", key="um_disable_totp",
                             icon=":material/no_encryption:", width="stretch",
                             disabled=not mfa_on):
                    result = service.admin_disable_totp(actor,
                                                        target["user_id"])
                    if result.ok:
                        st.success(result.info)
                        st.rerun()
                    for error in result.errors:
                        st.error(error)
            st.caption("Removing 2FA is the recovery path for a lost "
                       "authenticator device. The user can then sign in "
                       "with their password and re-enrol; the action is "
                       "audited.")

    # --------------------------------------------------------------- sessions
    with tab_sessions:
        with ui.card("Active sessions", "lock",
                     "Server-side sessions; tokens are stored only as "
                     "SHA-256 digests", key="um_sessions"):
            if not sessions:
                ui.empty_state("No active sessions",
                               "Sessions appear here after sign-in.", "lock")
            else:
                frame = pd.DataFrame([{
                    "session": s["session_id"][:12] + "...",
                    "user": s["username"],
                    "remember_me": "yes" if s["remember"] else "no",
                    "created": s["created_at"][:19].replace("T", " "),
                    "last_seen": s["last_seen_at"][:19].replace("T", " "),
                    "expires": s["expires_at"][:19].replace("T", " "),
                } for s in sessions])
                ui.show_table(frame, height=320, key="um_sessions_table")
                pick = st.selectbox(
                    "Session to revoke",
                    [f"{s['session_id'][:12]}...  ({s['username']})"
                     for s in sessions], key="um_session_pick")
                session_id = sessions[[
                    f"{s['session_id'][:12]}...  ({s['username']})"
                    for s in sessions].index(pick)]["session_id"]
                if st.button("Revoke session", key="um_revoke",
                             icon=":material/link_off:", type="primary"):
                    result = service.revoke_session(actor, session_id)
                    if result.ok:
                        st.success(result.info)
                        st.rerun()
                    for error in result.errors:
                        st.error(error)
                st.caption("Revoking a session signs that device out on its "
                           "next request. Password changes revoke every "
                           "session automatically.")

    # ------------------------------------------------------------------ audit
    with tab_audit:
        with ui.card("Security audit trail", "list",
                     "Important authentication and administration events",
                     key="um_audit"):
            events = service.audit_events(actor, limit=300)
            if not events:
                ui.empty_state("No audit events yet",
                               "Events are recorded as users interact with "
                               "authentication.", "list")
            else:
                frame = pd.DataFrame([{
                    "time": e["created_at"][:19].replace("T", " "),
                    "event": e["event"],
                    "subject": e["user_id"] or "-",
                    "actor": e["actor_id"] or "-",
                    "detail": e["detail"],
                } for e in events])
                ui.show_table(frame, height=440, key="um_audit_table")
            st.caption("The audit trail never contains passwords, session "
                       "tokens or recovery links.")

    # ------------------------------------------------------------------ roles
    with tab_roles:
        with ui.card("Role model", "shield-check",
                     "Least privilege by default; permissions enforced "
                     "server-side", key="um_roles"):
            for role in rbac.ROLES:
                st.markdown(
                    f"**{rbac.ROLE_LABELS[role]}** (`{role}`) - "
                    f"{rbac.ROLE_DESCRIPTIONS[role]}")
                st.caption("Permissions: " + ", ".join(
                    sorted(rbac.PERMISSIONS[role])))
                st.divider()
            ui.note(
                "Public registration is <b>"
                + ("open" if auth_settings.ALLOW_REGISTRATION else "closed")
                + "</b> on this deployment "
                "(<code>AI_SOC_ALLOW_REGISTRATION</code>). When it is closed, "
                "the signup page says so and accounts are added with the "
                "<b>Add a user</b> card on the Users tab.", "info", ui.ACCENT)
            ui.note(
                "Initial administrator provisioning is operator-run: "
                "<code>python scripts/provision_admin.py</code> (hidden "
                "password input, no hardcoded credentials). The web "
                "registration form cannot create administrators.",
                "info", ui.ACCENT)