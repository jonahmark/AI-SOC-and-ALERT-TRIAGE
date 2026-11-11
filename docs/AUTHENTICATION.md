# Authentication, sessions and access control

AI-SOC gates the entire dashboard behind a real, server-side authentication
system. This document describes how it works, how to set it up, and what it
does and does not protect. It is a research prototype, not a hardened
production identity provider — the limitations section is part of the design.

All credential decisions live in `src/auth/`; the Streamlit layer
(`dashboard/auth_ui.py`, `app.py`) only renders forms and their real results.
No authentication response is fabricated, and no password, token or session
secret is ever written to the activity log, the audit log, or the UI.

## Contents

- [Quick start](#quick-start)
- [Roles and permissions](#roles-and-permissions)
- [Architecture](#architecture)
- [Password policy and hashing](#password-policy-and-hashing)
- [Sessions and remember-me](#sessions-and-remember-me)
- [Two-factor authentication (TOTP)](#two-factor-authentication-totp)
- [Breach-password screening](#breach-password-screening)
- [Password recovery](#password-recovery)
- [Email verification](#email-verification)
- [Email delivery modes](#email-delivery-modes)
- [Brute-force protection](#brute-force-protection)
- [Auditing](#auditing)
- [CSRF, cookies and transport](#csrf-cookies-and-transport)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Security posture and deferred items](#security-posture-and-deferred-items)
- [Limitations](#limitations)

## Quick start

```bash
cd ai_soc_prototype
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

cp .env.example .env          # edit values; never commit a real .env
python train.py               # produces the real model artefacts the pages read
python scripts/provision_admin.py   # interactive, hidden password input
streamlit run app.py
```

`provision_admin.py` prompts for full name, email, username and a password
(hidden input, confirmation required) and creates the first **administrator**.
No administrator credential is hardcoded, echoed, logged, or stored in any
template. Registration through the web UI always creates a least-privileged
**viewer**; only an administrator can grant more.

On first sign-in, change the provisioned password from the account panel
(Password change). Changing a password invalidates every other session.

## Roles and permissions

Three roles, enforced server-side in the service layer and re-checked at the
page gate in `app.py`. Hiding a frontend control is never the security
boundary — the same permission is required before any privileged action runs.

| Permission | Administrator | SOC Analyst | Researcher / Viewer |
|---|:---:|:---:|:---:|
| `view_dashboard` (analytical pages) | ✓ | ✓ | ✓ |
| `generate_reports` | ✓ | ✓ | ✓ |
| `triage_alerts` (write status / notes) | ✓ | ✓ | — |
| `manage_settings` (System Settings) | ✓ | — | — |
| `manage_users` (User Management) | ✓ | — | — |
| `view_audit` | ✓ | — | — |
| `manage_sessions` (revoke) | ✓ | — | — |

Analysts and viewers open the **same pages**; the difference is in-page write
access: a viewer sees Alert Management read-only (a clear notice, no status or
note controls), while an analyst can triage. Administrators additionally see
System Settings and User Management, which are not even registered in the
navigation for other roles (`dashboard/nav.py` filters by `rbac.page_allowed`).

Guards against self-escalation and lockout:

- `register()` has no role parameter — a new account is always `viewer`.
- An administrator cannot change their **own** role or disable their **own**
  account (refused with a user-safe message), preventing accidental lockout.
- Role and status changes require `manage_users`; session revocation requires
  `manage_sessions`; the audit feed requires `view_audit`. Each raises
  `rbac.PermissionDenied` for any other role.

## Architecture

| Module | Responsibility |
|---|---|
| `src/auth/settings.py` | Environment-driven configuration (`.env` loaded as defaults; real env vars win) |
| `src/auth/db.py` | SQLite (WAL) store, per-operation connections, parameterised queries, schema |
| `src/auth/security.py` | Argon2id hashing, token generation/digests, password policy + strength, text sanitisation |
| `src/auth/mfa.py` | TOTP primitives: pyotp codes, Fernet secret encryption, QR/provisioning URI |
| `src/auth/service.py` | All credential logic: register, authenticate, sessions, recovery, verification, RBAC admin actions, audit |
| `src/auth/rbac.py` | Roles, permission sets, page-permission map, `require()` / `page_allowed()` |
| `src/auth/mail.py` | Delivery modes (`off` / `outbox` / `smtp`), link building |
| `dashboard/auth_ui.py` | Branded sign-in / signup / recovery / reset / verification pages, session gate, remember-me cookie bridge, account panel |
| `dashboard/pages/user_management.py` | Admin-only user / role / session / audit management |
| `scripts/provision_admin.py` | Operator-run secure bootstrap of the first administrator |

Storage defaults to `data/processed/auth.db`. Tables: `users`, `sessions`,
`reset_tokens`, `verify_tokens`, `password_history`, `totp_credentials`,
`totp_challenges`, `login_attempts`, and `audit_events`. Every query is
parameterised; no SQL is built from user input.

## Password policy and hashing

- Policy: **≥ 12 characters** with at least one upper-case, one lower-case, one
  digit and one special character. Maximum 256. Long passphrases are accepted
  and encouraged; the live strength meter and requirement checklist update as
  you type, but the server re-validates independently — client-side checks are
  convenience, never the boundary.
- Hashing: **Argon2id** via `argon2-cffi` (`PasswordHasher` defaults). Each hash
  carries its own random salt; verification re-hashes on parameter changes.
  No custom cryptography is implemented.
- A dummy hash is verified on unknown identifiers so login timing does not
  reveal whether an account exists.
- Plaintext passwords are never stored, logged, or returned. Only the hash
  persists.
- **Password history:** the last `AI_SOC_PASSWORD_HISTORY_SIZE` passwords
  (default 5, including the current one) cannot be reused on password change
  or recovery reset. History stores argon2id hashes only, is seeded at
  registration/admin provisioning, appended on every successful change/reset,
  and trimmed to the configured size. A blocked attempt is audited
  (`password_reuse_blocked`) and, on reset, does not consume the token.
  Accounts created before this feature have no earlier history — their
  current password is still protected against immediate reuse.

## Sessions and remember-me

- A session token is `secrets.token_urlsafe(32)`. The database stores only its
  **SHA-256 digest**, so a database leak does not yield usable tokens.
- Standard sessions expire after `AI_SOC_SESSION_TTL_HOURS` (default 12);
  "remember me" sessions after `AI_SOC_REMEMBER_ME_DAYS` (default 14).
- **Idle timeout:** a session with no validated activity for
  `AI_SOC_SESSION_IDLE_MINUTES` (default 60) is deleted server-side on the
  next validation and the visitor returns to sign-in (`session_idle_expired`
  is audited). This applies to remember-me sessions too; set `0` to disable
  the idle check (the absolute TTL still applies). Activity refreshes the
  window, so a working session never expires mid-use.
- Every request re-validates the session against the database: expiry,
  revocation, account status, and the user's `auth_epoch`. Changing a password
  bumps `auth_epoch`, which invalidates **all** existing sessions; the caller
  is re-issued a fresh one so they are not signed out mid-change.
- Logout revokes the session row and clears the cookie. Administrators can
  revoke any session from User Management.
- An expired or revoked session returns the visitor to the branded sign-in
  page with a clear "Your session expired or was signed out" notice — never a
  partially rendered dashboard.

## Two-factor authentication (TOTP)

Optional per-account second factor using time-based one-time passwords
(RFC 6238, `pyotp`). No custom cryptography: codes come from pyotp, the
shared secret is encrypted at rest with **Fernet** (`cryptography`
library), and the setup QR is rendered by `qrcode`.

**Operator setup.** Generate a key once and put it in `.env`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# then set AI_SOC_TOTP_KEY=<that value> and restart the dashboard
```

Without a valid `AI_SOC_TOTP_KEY`, enrolment is **unavailable and the UI
says so** — secrets are never stored in plaintext as a fallback. Keep the
key with the same care as the SMTP credential: anyone with both the
database and the key can decrypt TOTP secrets. Rotating the key makes
existing secrets undecryptable; affected users then need admin removal
(below) and re-enrolment.

**Enrolment (account panel → Two-factor authentication).**

1. "Set up two-factor" creates a *pending* credential and shows a QR code
   (or a manual setup key) for any authenticator app.
2. The user enters the app's current 6-digit code; only a matching code
   enables the factor. Until then, login is unchanged.
3. Once enabled, every sign-in is two-stage: password first, then a code.
   The password stage issues a **single-use challenge token** (digest-only
   storage, 5-minute TTL, 5 wrong-code attempts). Wrong codes count towards
   the normal login lockout under both email and username. A password
   change or account disable invalidates pending challenges.

**Removal / recovery.**

- Self-service disable requires a current code (a hijacked session cannot
  silently strip the second factor).
- **Lost device:** an administrator uses User Management → "Remove 2FA"
  (`manage_users`, audited as `totp_disabled_by_admin`). The user signs in
  with their password and re-enrols.
- Password recovery does **not** remove TOTP — otherwise control of the
  email account alone would defeat the second factor.

Codes tolerate ± one 30-second step of clock drift. Audit events:
`totp_challenge_issued`, `totp_login_failed`, `totp_challenge_exhausted`,
`totp_enrolled`, `totp_disabled`, `totp_disable_failed`,
`totp_disabled_by_admin`; successful MFA logins are `login_success` with
`mfa=totp` in the detail. Secrets and codes never appear in the audit trail.

## Breach-password screening

Optional (`AI_SOC_BREACH_SCREENING=1`, off by default; enabled in this
deployment as of 2026-11-09). New passwords — at registration, self-service
change and recovery reset — are checked against the Have I Been Pwned
range API using the **k-anonymity** model:

- Only the first 5 hex characters of the password's SHA-1 digest are sent.
  The password and the full hash never leave the machine; the service
  returns all suffixes for that prefix and the match happens locally.
- A known-breached password is rejected with the occurrence count, and the
  block is audited (`breach_password_blocked`). On reset, the token is not
  consumed, so the user can retry the same link with a different password.
- **Honest fail-open:** if the service is unreachable (offline deployment,
  outage, timeout), the operation proceeds and `breach_check_unavailable`
  is audited with `fail_open`. An outage is never presented as a clean
  screening result.
- Verified live on 2026-11-09: a common password returned `breached`
  (295,389 occurrences); a random strong passphrase returned `clean`.
  Tests mock the HTTP layer — the suite never contacts the real service.

## Password recovery

- "Forgot password" always returns the **same generic message** whether or not
  the email matches an account (no enumeration). Requests are rate-limited
  (`RESET_REQUESTS_PER_WINDOW`, default 3 per window).
- The recovery token is single-use and expires after
  `AI_SOC_RESET_TOKEN_TTL_MINUTES` (default 30). Only its digest is stored.
- Completing a reset consumes the token, sets the new hash, bumps `auth_epoch`
  (signing out all old sessions), and clears any attacker-induced lockout on
  the account so the legitimate owner can sign in.
- A weak or mismatched new password is rejected **without** consuming the
  token, so the user can retry the same link. Reusing a recent password is
  rejected the same way (password history applies to recovery too).
- Recovery/verification tokens are never logged or shown in the UI.

## Email verification

- New accounts receive a single-use verification token valid for
  `AI_SOC_VERIFY_TOKEN_TTL_HOURS` (default 24); resend is cooldown-limited
  (120 s). Verification is enforced only where the deployment configures real
  delivery — see below.
- If email is not configured, the UI **states the limitation honestly** rather
  than displaying a fake "verification sent" success.

## Email delivery modes

Set `AI_SOC_EMAIL_MODE`:

| Mode | Behaviour |
|---|---|
| `off` | Nothing is sent. Recovery/verification UI explains that delivery is disabled until SMTP is configured. |
| `outbox` | **Development only.** Messages are written as `.eml` files to `results/mail_outbox/`; the UI labels this explicitly and tells you to open the newest file for your link. No real email is sent. This is the default so the prototype is usable offline. |
| `smtp` | Real delivery via `AI_SOC_SMTP_*`. Requires a host; missing configuration fails honestly rather than pretending to send. Use STARTTLS (default on) and an app password where the provider offers one. |

Links use `AI_SOC_APP_URL` as their public base. In `outbox` mode the `.eml`
files contain live single-use tokens — treat that directory as sensitive and
clear it after testing.

### Deployment & reachability (LAN-only posture)

As of 2026-11-09 this prototype is deployed **LAN-only**:

- `AI_SOC_APP_URL` is set to the server's LAN address
  (`http://192.168.3.47:8501`), so verification/recovery links open on any
  device on the same network. `localhost` would only work on the server
  machine itself.
- The server binds all interfaces (`--server.address 0.0.0.0`); a Windows
  Firewall inbound rule **"AI-SOC Streamlit (LAN)"** allows TCP 8501 on
  **Private profiles only, from the local subnet**, and the home Wi-Fi is
  categorised Private. `scripts/allow_lan_access.ps1` (run as administrator,
  idempotent) applies both settings and logs to `results/allow_lan_access.log`.
- If the machine's LAN IP changes (DHCP), update `AI_SOC_APP_URL` and restart
  the dashboard; old links keep the previous base URL.
- **No internet exposure.** Public access (HTTPS reverse proxy or tunnel) is
  deliberately deferred to the security-hardening phase; plain HTTP must not
  be exposed publicly because session cookies and credentials would travel
  unencrypted.

### Transactional email providers (recommended for real use)

For real delivery, use a transactional provider rather than a personal mailbox.
Set `AI_SOC_EMAIL_PROVIDER` to a preset and the host/port/STARTTLS fill in
automatically; you supply only the account credentials and a verified sender.

| Provider | Preset host | `AI_SOC_SMTP_USER` | `AI_SOC_SMTP_PASSWORD` |
|---|---|---|---|
| SendGrid | `smtp.sendgrid.net` | the literal word `apikey` | your SendGrid API key (Mail Send) |
| Brevo | `smtp-relay.brevo.com` | your Brevo account email | your Brevo SMTP key |
| Mailgun | `smtp.mailgun.org` | `postmaster@your-domain` | your Mailgun SMTP password |
| Postmark | `smtp.postmarkapp.com` | your Server API token | the same Server API token |
| Amazon SES | `email-smtp.<region>.amazonaws.com` | SES SMTP username | SES SMTP password |
| Gmail | `smtp.gmail.com` | your Gmail address | a Google App Password |

```bash
AI_SOC_EMAIL_MODE=smtp
AI_SOC_EMAIL_PROVIDER=sendgrid
AI_SOC_SMTP_USER=apikey
AI_SOC_SMTP_PASSWORD=<your SendGrid API key>
AI_SOC_SMTP_FROM=AI-SOC <no-reply@yourdomain.com>   # a verified sender
```

Notes:

- **This is one server-side sending credential, configured once by the
  operator.** Users who register choose their own AI-SOC password and receive
  mail at their own address; they never see, need, or share this credential.
- The sender (`AI_SOC_SMTP_FROM`) must be an address or domain you **verified**
  with the provider (e.g. SendGrid Sender Authentication), otherwise mail is
  rejected. A dedicated `no-reply@yourdomain.com` is preferable to a personal
  address.
- Amazon SES hosts are region-specific: the preset uses `us-east-1`; set
  `AI_SOC_SMTP_HOST` explicitly for another region. Any explicit
  `AI_SOC_SMTP_HOST` / `AI_SOC_SMTP_PORT` / `AI_SOC_SMTP_STARTTLS` overrides the
  preset.
- After editing `.env`, **restart the dashboard** — Streamlit caches settings
  at startup, so a running server keeps its old mode until restarted.
- Verify delivery with `python scripts/send_test_email.py recipient@example.com`;
  `Delivered: True` confirms the same path verification/recovery emails use.

## Brute-force protection

- A sliding window (`ATTEMPT_WINDOW_MINUTES`, default 15) counts consecutive
  failures per identifier; `MAX_FAILED_ATTEMPTS` (default 5) triggers a
  `LOCKOUT_MINUTES` (default 15) lockout.
- The same lockout applies to **unknown** identifiers, so the response cannot
  be used to enumerate accounts. Login failures always return an identical
  generic message; disabled accounts are indistinguishable from bad passwords.
- Registration is capped at `MAX_REGISTRATIONS_PER_HOUR` (default 20).

## Auditing

Security-relevant events are recorded in `audit_events` with the acting user,
the affected user, a sanitised detail string (≤ 300 chars) and a timestamp:
registration, login success/failure/lockout, logout, password reset
requested/completed, password change, password-reuse blocks, idle session
expiry, TOTP enrolment/challenges/disable (see above), breach screening
blocks and fail-open outages, email verification, role change, account
status change, session revocation, and admin provisioning.

The audit detail **never** contains a password or a token. Failed-login detail
records only a category (`unknown_account`, `account_disabled`, `bad_password`)
or a hash of the identifier for lockouts — not the credential. Only
administrators (`view_audit`) can read the feed.

## CSRF, cookies and transport

- The remember-me cookie (`soc_session`) holds only the opaque session token —
  it is not a credential and grants nothing without a matching live session
  row. It is written client-side by a small Streamlit components-v2 bridge
  (`st.context.cookies` is read-only) with `SameSite=Strict`, `path=/`, and
  `Secure` whenever the app is served over HTTPS. Without "remember me", no
  cookie is written and the session ends when the browser closes.
- Streamlit's built-in XSRF token protects the WebSocket connection; the
  `SameSite=Strict` cookie prevents cross-site request forgery on the
  session. The agent API (if enabled) refuses cross-origin browser requests
  and disallowed hosts.
- **Deploy behind HTTPS in any real use.** `Secure` cookies and token-bearing
  recovery links require transport encryption; the prototype's default
  `http://localhost:8501` is for local development only.

## Configuration reference

All values come from `.env.example` (placeholders and safe defaults only — no
real secrets). Real environment variables override `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `AI_SOC_SESSION_TTL_HOURS` | 12 | Standard session lifetime |
| `AI_SOC_REMEMBER_ME_DAYS` | 14 | Remember-me session lifetime |
| `AI_SOC_SESSION_IDLE_MINUTES` | 60 | Inactivity expiry (0 disables; applies to remember-me too) |
| `AI_SOC_PASSWORD_HISTORY_SIZE` | 5 | Recent passwords that cannot be reused |
| `AI_SOC_TOTP_KEY` | — | Fernet key for TOTP secret encryption; MFA unavailable without it |
| `AI_SOC_TOTP_CHALLENGE_TTL_MINUTES` | 5 | Lifetime of the password-stage login challenge |
| `AI_SOC_TOTP_MAX_ATTEMPTS` | 5 | Wrong codes allowed per challenge |
| `AI_SOC_BREACH_SCREENING` | 0 | HIBP k-anonymity screening of new passwords (fail-open on outage, audited) |
| `AI_SOC_BREACH_CHECK_TIMEOUT` | 5 | Seconds before the HIBP request fails open |
| `AI_SOC_RESET_TOKEN_TTL_MINUTES` | 30 | Recovery token lifetime |
| `AI_SOC_VERIFY_TOKEN_TTL_HOURS` | 24 | Verification token lifetime |
| `AI_SOC_MAX_FAILED_ATTEMPTS` | 5 | Failures before lockout |
| `AI_SOC_EMAIL_MODE` | outbox | `off` / `outbox` / `smtp` |
| `AI_SOC_EMAIL_PROVIDER` | — | Preset: `sendgrid` / `brevo` / `mailgun` / `postmark` / `ses` / `gmail` (auto-fills host/port) |
| `AI_SOC_APP_URL` | http://localhost:8501 | Public base for links |
| `AI_SOC_SMTP_HOST/PORT/USER/PASSWORD/FROM/STARTTLS` | — | SMTP delivery (smtp mode) |
| `AI_SOC_AUTH_DB` | data/processed/auth.db | Storage location |

## Testing

```bash
pytest tests/ -v
```

- `tests/test_auth.py` exercises the service layer: registration (least
  privilege, no role parameter, duplicate/invalid fields, rate limit,
  digest-only token storage, honest mail-mode messages), login (email/username,
  identical generic errors, lockout incl. unknown identifiers, remember-me
  expiry), sessions (invalid/expired/revoked, logout, password-change epoch
  invalidation, admin revoke, idle expiry incl. remember-me and refresh),
  password history (seeding, reuse blocks on change/reset, trimming, legacy
  accounts, secret-free audit), recovery (generic responses, single-use,
  expiry, lockout clearing, weak/mismatch non-consumption, window cap),
  verification (single-use, expiry, resend cooldown), RBAC/admin guards,
  audit (events recorded, no secrets, attribution) and the security primitives.
- `tests/test_mfa.py` covers the TOTP layer: encrypted secret storage,
  enrolment confirmation, two-stage login, challenge expiry/single-use/
  attempt caps, lockout integration, self-disable and admin recovery, and
  secret-free auditing — with real pyotp codes.
- `tests/test_breach.py` covers breach screening with mocked HTTP: the
  k-anonymity request shape (prefix only), breach blocks at registration/
  change/reset, token non-consumption, and audited fail-open on outages.
- `tests/test_auth_ui.py` drives the real Streamlit pages through `AppTest`:
  the authentication gate, page links, signup, valid/invalid login,
  remember-me cookie state, expired-session notice, logout, recovery via a
  real outbox link, single-use verification, viewer refusal of admin pages,
  viewer read-only triage, admin user management, password-change session
  re-issuance, the MFA code step at login, account-panel TOTP enrol/disable
  and admin 2FA removal.

The auth fixtures redirect the database and mail outbox to a temporary folder,
so the suite never touches your real accounts or analyst state. Full suite:
**358 passing tests**.

## Security posture and deferred items

Decisions recorded 2026-11-09 (security-hardening phase), with the reasoning
so a future operator can revisit them:

- **Exposure: LAN-only, plain HTTP.** The dashboard is reachable only from
  the local subnet (Private-profile firewall rule). Public exposure is
  deliberately deferred: it requires a TLS-terminating reverse proxy (or a
  tunnel) first, because session cookies and token-bearing email links must
  not travel unencrypted over the internet. Nothing in the app terminates
  TLS itself.
- **Secrets: `.env` + environment, not a secrets manager.** For a
  single-node research prototype this is proportionate: the SMTP key and
  the TOTP encryption key live in a gitignored `.env` with operator-only
  access. A vault/KMS integration would add operational weight without
  changing the threat model here; revisit before any multi-user production
  deployment.
- **MFA is optional per account, not policy-enforced.** Administrators are
  encouraged to enrol (the account panel says so); forcing it would need a
  grace-period and recovery workflow beyond this prototype's scope.
- **Idle timeout applies to remember-me too.** Convenience never overrides
  the inactivity expiry; operators can tune `AI_SOC_SESSION_IDLE_MINUTES`.
- **Password history starts at hardening.** Accounts created earlier have
  no pre-hardening history; their current password is still protected
  against immediate reuse.

## Limitations

- **Research prototype, not a production IdP.** No SSO/OAuth or
  account-lockout notification email. TOTP MFA is optional per account,
  not enforced by policy.
- **SQLite** is single-node; there is no horizontal scaling or distributed
  rate-limit store.
- **Outbox mode is not delivery.** It exists so the prototype is demonstrable
  offline; it must not be used to claim real verification in production.
- **HTTPS is the operator's responsibility.** The app does not terminate TLS;
  put it behind a reverse proxy that does before any non-local use.
- **No secrets manager integration.** SMTP credentials live in `.env`/env vars;
  protect that file and never commit it.
- Session and token lifetimes are configurable defaults, not a compliance
  guarantee. Review them against your own policy before deployment.