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
- [Brute-force and abuse protection](#brute-force-and-abuse-protection)
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

On a public deployment set `AI_SOC_ALLOW_REGISTRATION=0`. The signup page then
states that registration is closed, and administrators add people in
**User Management → Add a user** (any role, permission-checked and audited,
with a verification link emailed inside the deployment's sending budget).

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
| `src/auth/migration.py` | Read-only SQLite → empty PostgreSQL account transfer (validation, re-encryption, atomic verify) |
| `scripts/migrate_accounts.py` | Operator CLI: inspect local accounts, or `--apply` to import into an empty private database |

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

### Deployment & reachability (two deployments)

**Local instance — LAN-only, plain HTTP (since 2026-11-09):**

- `AI_SOC_APP_URL` is set to the server's LAN address
  (`http://192.168.3.47:8501`), so verification/recovery links open on any
  device on the same network. `localhost` would only work on the server
  machine itself.
- The server binds all interfaces (`--server.address 0.0.0.0`); a Windows
  Firewall inbound rule **"AI-SOC Streamlit (LAN)"** allows TCP 8501 on
  **Private profiles only, from the local subnet**, and the home Wi-Fi is
  categorised Private. `scripts/allow_lan_access.ps1` (run as administrator,
  idempotent) applies both settings and logs to `results/allow_lan_access.log`.
  The dashboard itself runs in WSL and is reached through a Windows portproxy.
- If the machine's LAN IP changes (DHCP), update `AI_SOC_APP_URL` and restart
  the dashboard; old links keep the previous base URL.
- **No internet exposure for this instance.** Plain HTTP must not be exposed
  publicly because session cookies and credentials would travel unencrypted.

**Hosted instance — Streamlit Cloud, HTTPS, viewer-restricted (since 2026-11-15):**

- **Viewing is restricted by the platform, not by the app.** Checked
  2026-11-15: an anonymous request to the app root returns `303` to
  Streamlit's sign-in gate (`share.streamlit.io/-/auth/app`), and the owner's
  Share panel offers a "Make this app public" action — public viewing is not
  enabled. Only the owner and invited viewers reach the app's sign-in page.
  The operator chose to keep it restricted. (The inner `/~/+/` path and its
  `/_stcore/health` do answer anonymously, so those alone do not prove the app
  is publicly viewable.)
- `AI_SOC_APP_URL` is the hosted app URL, so emailed links open from any
  device that can view the app; the platform terminates TLS, so `Secure`
  cookies apply.
- Accounts, sessions, tokens and audit events live in a private PostgreSQL
  database (`AI_SOC_AUTH_DATABASE_URL`, set in the platform's owner-only
  Secrets manager). The local instance keeps its own SQLite store unless the
  same URL is configured there too.
- `AI_SOC_ALLOW_REGISTRATION=0` is set in the app's Secrets (applied and
  verified live on 2026-11-15): the signup page renders the closed notice
  instead of a form, and administrators add people in **User Management →
  Add a user**.
- The email budget, login-work cap and retention pruning are defence in depth
  behind that viewer gate, and become the primary protection if public
  viewing is ever enabled: the container has a 1 GB RAM ceiling, so unbounded
  argon2id work or provider quota burn is a real denial-of-service risk.
- The container disk is ephemeral — triage state and analysis caches reset on
  redeploy. Authentication state survives because it is in PostgreSQL.

### Private PostgreSQL storage (optional, for cloud deployments)

Local installs keep using SQLite at `data/processed/auth.db` — nothing changes
for them. A hosted deployment (e.g. Streamlit Community Cloud) has **ephemeral
local disk**, so accounts written to SQLite there disappear on every restart.
For durable cloud accounts, point the app at a private PostgreSQL database:

- Set `AI_SOC_AUTH_DATABASE_URL` to a `postgresql://` (or `postgres://`) URI.
  A non-blank URL selects PostgreSQL **with no SQLite fallback**; if it is
  unreachable the app fails closed rather than silently writing to a throwaway
  local file.
- TLS is always `verify-full`. The CA defaults to `certifi`; an explicit
  `sslrootcert` in the URI is honoured. Unix-socket hosts are rejected (libpq
  does not apply TLS to sockets), and GSS encryption is disabled so it cannot
  take precedence over TLS.
- The URI is a **secret**. Supply it only through protected settings — on
  Streamlit Cloud, the app's *Secrets* manager. The app bridges `AI_SOC_*`
  secret keys into its configuration at startup; real environment variables
  win. Never commit a real URL, and never paste it into chat or logs. Storage
  errors are sanitised so a failure never echoes the URI or SQL values.
- Schema initialisation is serialised across processes with an advisory lock;
  all queries remain parameterised.

### Migrating local accounts to a private cloud database

`scripts/migrate_accounts.py` copies existing local accounts (with their
working passwords) into an **empty** private PostgreSQL database. It is
operator-run and deliberately conservative:

```bash
# 1. Inspect only — reads the local SQLite file, makes no cloud connection.
python scripts/migrate_accounts.py

# 2. Import into an empty destination (prompts for the URI via hidden input).
python scripts/migrate_accounts.py --apply
```

Safety properties:

- **Read-only source.** The SQLite file is opened read-only inside a
  transaction, passes `quick_check`, and its transferred tables pass a
  referential-integrity check before anything is read. The source is never
  modified. (Orphaned rows in discarded tables — e.g. single-use tokens left
  behind by a deleted account — do not block the transfer.)
- **Empty destination only.** All nine auth tables are locked and checked; if
  any already contains data the import refuses and overwrites nothing. Use a
  fresh database.
- **Atomic + verified.** Inserts run in one transaction; per-table row counts
  are verified and any mismatch rolls the whole import back.
- **Passwords preserved, never re-hashed.** Argon2id hashes are validated and
  copied as-is, so existing passwords keep working. Only Argon2id is accepted.
- **MFA re-encrypted.** Enabled TOTP enrolments are decrypted with the source
  `AI_SOC_TOTP_KEY` and re-encrypted with a destination key you supply; the
  destination key must then be set as `AI_SOC_TOTP_KEY` in cloud secrets.
  Missing/invalid keys fail before any write.
- **No live credentials copied.** Sessions, recovery/verification tokens,
  pending MFA challenges, login-attempt history and the audit log are *not*
  transferred — everyone signs in fresh. An `accounts_migrated` audit event is
  recorded in the destination.

After a successful import, set `AI_SOC_AUTH_DATABASE_URL` (and the destination
`AI_SOC_TOTP_KEY`, plus your email settings and a reachable `AI_SOC_APP_URL`)
in the cloud app's secrets, redeploy, and verify sign-in with a real account.

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

## Brute-force and abuse protection

Per-identifier controls:

- A sliding window (`ATTEMPT_WINDOW_MINUTES`, default 15) counts consecutive
  failures per identifier; `MAX_FAILED_ATTEMPTS` (default 5) triggers a
  `LOCKOUT_MINUTES` (default 15) lockout.
- The same lockout applies to **unknown** identifiers, so the response cannot
  be used to enumerate accounts. Login failures always return an identical
  generic message; disabled accounts are indistinguishable from bad passwords.

Deployment-wide controls (added 2026-11-15 for the hosted Streamlit Cloud
deployment). The hosted app is currently viewer-restricted by the platform
(see "Deployment & reachability"), so there these are defence in depth; they
become the primary protection if public viewing is enabled, and they apply to
any self-hosted instance that is reachable from the internet:

- **Registration policy.** `AI_SOC_ALLOW_REGISTRATION=0` closes public signup:
  the signup page renders an honest "registration is closed" notice instead of
  a form, and `service.register()` refuses before any validation, hashing,
  database write or email. Administrators add people in **User Management →
  Add a user** (`service.admin_create_user`: `manage_users` permission, role
  validated against the server-side allow-list, audited with the acting
  administrator, account starts unverified and a verification link is emailed
  within the budget). `scripts/provision_admin.py` still creates the first
  administrator, and the public signup cap does not apply to admin
  provisioning.
- **Signup cap.** Public registration is capped at
  `MAX_REGISTRATIONS_PER_HOUR` (default 20, counted deployment-wide). Tradeoff:
  while the cap is hit, legitimate self-service signup is refused too — one
  more reason to close registration on a public URL.
- **Email budget.** Real SMTP sends (verification + recovery) are capped
  deployment-wide at `AI_SOC_MAX_EMAILS_PER_HOUR` (20) and
  `AI_SOC_MAX_EMAILS_PER_DAY` (60); keep the daily cap below your provider's
  plan limit. Every real attempt is recorded in `email_sends` (account id,
  purpose, timestamp — never an address, link or token), including failures,
  because each one still contacts the provider. When the budget is spent the
  message is **not** sent and the UI says so honestly. Recovery and resend
  check the budget *before* rotating tokens, so a spent budget never
  invalidates a link already in someone's inbox. `outbox`/`off` modes are not
  budgeted because nothing leaves the machine.
- **Login work cap.** `AI_SOC_MAX_LOGIN_ATTEMPTS_PER_MINUTE` (default 30, 0
  disables) refuses sign-in before any argon2id work once the deployment-wide
  per-minute count is reached. Without it, rotating identifiers would buy
  unlimited memory-hard hashing (64 MiB per attempt) on a small host. Refusals
  are not recorded, so the window drains on its own and the refusal path stays
  cheap.
- **Retention.** `service.prune_expired()` runs once per server process
  (`app.py`, `st.cache_resource`) and deletes aged `login_attempts`, expired
  `sessions` and `totp_challenges`, `reset_tokens`/`verify_tokens` older than
  `AI_SOC_TOKEN_RETENTION_DAYS` (7), `email_sends` older than a day, and
  `audit_events` older than `AI_SOC_AUDIT_RETENTION_DAYS` (90). A storage
  failure while pruning never blocks sign-in.

**Why not per-IP limits?** Streamlit 1.67 exposes `st.context.ip_address` as
the raw TCP peer (the package does not parse `X-Forwarded-For`), so behind
Streamlit Cloud's proxy it is the proxy address or `None` — it would throttle
all visitors as one bucket. Reading `x-forwarded-for` from `st.context.headers`
is client-spoofable unless the proxy overwrites it. The controls above
therefore do not depend on a trustworthy client IP.

**Known tradeoff:** lockout is per identifier, so someone who knows an address
can still lock that account out for 15 minutes with 5 wrong guesses. The
login-work cap bounds how fast that can be done deployment-wide but does not
eliminate it.

## Auditing

Security-relevant events are recorded in `audit_events` with the acting user,
the affected user, a sanitised detail string (≤ 300 chars) and a timestamp:
registration, login success/failure/lockout, logout, password reset
requested/completed, password change, password-reuse blocks, idle session
expiry, TOTP enrolment/challenges/disable (see above), breach screening
blocks and fail-open outages, email verification, role change, account
status change, session revocation, admin provisioning (`user_provisioned`,
attributed to the acting administrator) and exhausted email budget
(`email_budget_exhausted`).

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
| `AI_SOC_MAX_LOGIN_ATTEMPTS_PER_MINUTE` | 30 | Deployment-wide sign-in cap before argon2id work (0 disables) |
| `AI_SOC_ALLOW_REGISTRATION` | 1 | 0 closes public signup; accounts are administrator-provisioned |
| `AI_SOC_MAX_EMAILS_PER_HOUR` | 20 | Deployment-wide real-SMTP sends per hour (0 disables) |
| `AI_SOC_MAX_EMAILS_PER_DAY` | 60 | Deployment-wide real-SMTP sends per day (0 disables) |
| `AI_SOC_AUDIT_RETENTION_DAYS` | 90 | Audit events kept before pruning |
| `AI_SOC_TOKEN_RETENTION_DAYS` | 7 | Recovery/verification token rows kept before pruning |
| `AI_SOC_EMAIL_MODE` | outbox | `off` / `outbox` / `smtp` |
| `AI_SOC_EMAIL_PROVIDER` | — | Preset: `sendgrid` / `brevo` / `mailgun` / `postmark` / `ses` / `gmail` (auto-fills host/port) |
| `AI_SOC_APP_URL` | http://localhost:8501 | Public base for links |
| `AI_SOC_SMTP_HOST/PORT/USER/PASSWORD/FROM/STARTTLS` | — | SMTP delivery (smtp mode) |
| `AI_SOC_AUTH_DB` | data/processed/auth.db | SQLite storage location (used when no PostgreSQL URL is set) |
| `AI_SOC_AUTH_DATABASE_URL` | — | Private PostgreSQL URI; selects PostgreSQL with verified TLS, no SQLite fallback |

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
  Public-exposure hardening is covered too: closed registration creates nothing
  and sends nothing, admin provisioning works while signup is closed (and is
  refused for non-admins / off-list roles), the hourly and daily email budgets
  stop sending and report honestly, a spent budget keeps an already-issued
  recovery/verification link usable, the development outbox is not budgeted,
  the login-work cap refuses before hashing, and retention pruning removes aged
  rows and expired sessions while keeping live ones.
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
  viewer read-only triage, admin user management (including provisioning a
  working account through the Add a user form while public registration is
  closed), the closed-signup notice and removed signup form, password-change
  session re-issuance, the MFA code step at login, account-panel TOTP
  enrol/disable and admin 2FA removal.

The auth fixtures redirect the database and mail outbox to a temporary folder,
so the suite never touches your real accounts or analyst state. Full suite:
**408 passing tests**.

## Security posture and deferred items

Decisions recorded 2026-11-09 (security-hardening phase), with the reasoning
so a future operator can revisit them:

- **Exposure: local instance LAN-only over plain HTTP; cloud instance over
  HTTPS, restricted to invited viewers.** The WSL dashboard is reachable only
  from the local subnet (Private-profile firewall rule plus a Windows portproxy
  to the WSL port). The Streamlit Cloud deployment is TLS-terminated by the
  platform and its viewer gate is left enabled, so session cookies and
  token-bearing email links are encrypted there and strangers cannot reach the
  sign-in page. Nothing in the app terminates TLS itself — never expose the
  local instance to the internet without a TLS-terminating reverse proxy.
- **Secrets: `.env` locally, platform Secrets manager on cloud.** For a
  single-node research prototype this is proportionate: the SMTP key and the
  TOTP encryption key live in a gitignored `.env` with operator-only access,
  and on Streamlit Cloud in the app's owner-only Secrets manager (never in
  Git). A vault/KMS integration would add operational weight without changing
  the threat model here; revisit before any multi-user production deployment.
- **MFA is optional per account, not policy-enforced.** Administrators are
  encouraged to enrol (the account panel says so); forcing it would need a
  grace-period and recovery workflow beyond this prototype's scope.
- **Idle timeout applies to remember-me too.** Convenience never overrides
  the inactivity expiry; operators can tune `AI_SOC_SESSION_IDLE_MINUTES`.
- **Password history starts at hardening.** Accounts created earlier have
  no pre-hardening history; their current password is still protected
  against immediate reuse.

Decisions recorded 2026-11-15 (public-exposure hardening), after an audit of
what a visitor can do on the hosted cloud URL:

- **The hosted app stays viewer-restricted.** Measured, not assumed: anonymous
  requests to the app root are redirected to Streamlit's sign-in gate and the
  owner's Share panel still offers "Make this app public". The operator chose
  to keep that gate enabled rather than open the app to anyone with the link.
  The app-level controls below are therefore defence in depth today, and the
  precondition for safely enabling public viewing later.
- **Cloud registration is administrator-provisioned.** Open signup would let
  any viewer create accounts, consume the SendGrid quota with verification mail
  to arbitrary addresses (a sender-reputation risk as well as a quota risk), and
  fill the database with junk. `AI_SOC_ALLOW_REGISTRATION=0` was added to the
  cloud Secrets and verified live on 2026-11-15 (closed notice, no signup form,
  no "Create an account" button on the login page); the local instance keeps
  open signup for convenience. The operator chose this over invite codes and
  over leaving signup open.
- **Caps that do not depend on client IP.** Per-IP limiting is not trustworthy
  here (see the abuse-protection section), so the budget, login-work cap and
  registration policy are deployment-wide and account-scoped instead.
- **Email failures stay visible.** A spent budget, a disabled provider or an
  SMTP error is reported in the UI as "not sent"; no path claims delivery that
  did not happen.
- **Retention is bounded, audit is not silent.** Pruning removes aged
  anonymous-traffic rows; audit events survive 90 days by default so an
  operator can still review abuse after the fact.

## Limitations

- **Research prototype, not a production IdP.** No SSO/OAuth or
  account-lockout notification email. TOTP MFA is optional per account,
  not enforced by policy.
- **Storage is single-node by default.** SQLite is the default; the optional
  private PostgreSQL URI (used by the cloud deployment) lets instances share
  accounts, but the app itself does not scale horizontally and rate limits are
  enforced in the database rather than a distributed cache.
- **The hosted viewer gate is a platform setting, not app code.** Nothing in
  this repository enforces it: anyone with owner access can enable public
  viewing from the Share panel. Re-check it after any change of hosting or
  ownership, and keep the app-level controls configured regardless.
- **No trustworthy client IP behind the platform proxy.** Abuse controls are
  deployment-wide and per-account, not per-IP, so a determined visitor can
  still consume a share of the email budget within the caps.
- **Per-identifier lockout can be used to annoy a known account.** Five wrong
  guesses lock that identifier for 15 minutes; the login-work cap bounds the
  rate but does not prevent it.
- **Outbox mode is not delivery.** It exists so the prototype is demonstrable
  offline; it must not be used to claim real verification in production.
- **HTTPS is the operator's responsibility for self-hosting.** The app does not
  terminate TLS; put it behind a reverse proxy that does before any non-local
  use (Streamlit Cloud already terminates TLS for the hosted deployment).
- **No vault/KMS integration.** Locally, SMTP credentials live in `.env`/env
  vars — protect that file and never commit it. On Streamlit Cloud they live in
  the platform's owner-only Secrets manager, which is not a general-purpose
  secrets management system.
- Session and token lifetimes are configurable defaults, not a compliance
  guarantee. Review them against your own policy before deployment.