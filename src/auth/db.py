"""SQLite persistence for authentication (schema + thin query layer).

A documented schema-initialisation process runs on first use (no external
migration tool needed for a single-file research prototype). All queries
are parameterised; connections are opened per operation so concurrent
Streamlit sessions are safe (WAL mode).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.auth import settings as auth_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    organisation    TEXT NOT NULL DEFAULT '',
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer'
                    CHECK (role IN ('administrator', 'analyst', 'viewer')),
    email_verified  INTEGER NOT NULL DEFAULT 0,
    account_status  TEXT NOT NULL DEFAULT 'active'
                    CHECK (account_status IN ('active', 'disabled')),
    auth_epoch      INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_login_at   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,           -- SHA-256 digest of the token
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    auth_epoch   INTEGER NOT NULL,
    remember     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS reset_tokens (
    token_hash TEXT PRIMARY KEY,             -- SHA-256 digest, never plaintext
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_reset_user ON reset_tokens(user_id);

CREATE TABLE IF NOT EXISTS verify_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_verify_user ON verify_tokens(user_id);

CREATE TABLE IF NOT EXISTS password_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,              -- argon2id, never plaintext
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_user ON password_history(user_id, id);

CREATE TABLE IF NOT EXISTS totp_credentials (
    user_id          TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    secret_encrypted TEXT NOT NULL,          -- Fernet ciphertext, never plaintext
    enabled          INTEGER NOT NULL DEFAULT 0,
    enrolled_at      TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS totp_challenges (
    token_hash TEXT PRIMARY KEY,             -- SHA-256 digest of the pending token
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    auth_epoch INTEGER NOT NULL,
    remember   INTEGER NOT NULL DEFAULT 0,
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_totp_challenges_user ON totp_challenges(user_id);

CREATE TABLE IF NOT EXISTS login_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier   TEXT NOT NULL,              -- normalised email/username only
    success      INTEGER NOT NULL,
    attempted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_identifier ON login_attempts(identifier);

CREATE TABLE IF NOT EXISTS audit_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event      TEXT NOT NULL,
    user_id    TEXT,                         -- subject of the event
    actor_id   TEXT,                         -- who performed it (if known)
    detail     TEXT NOT NULL DEFAULT '',    -- never passwords or tokens
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at);
"""


class AuthStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_pragmas(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode = WAL")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            return cur.rowcount


_STORES: dict[str, AuthStore] = {}


def get_store() -> AuthStore:
    """Store for the configured DB path (path is read at call time so tests
    can redirect it); one instance per distinct path per process."""
    path = Path(auth_settings.AUTH_DB_PATH)
    key = str(path.resolve())
    if key not in _STORES:
        store = AuthStore(path)
        store.init_pragmas()
        _STORES[key] = store
    return _STORES[key]


def reset_store_registry() -> None:
    _STORES.clear()