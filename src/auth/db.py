"""Authentication persistence: SQLite by default, optional private PostgreSQL.

Schema initialisation runs on first use. All queries are parameterised;
connections are opened per operation. The SQLite backend retains WAL mode;
PostgreSQL requires verified TLS and never falls back to a local database.
"""
from __future__ import annotations

import hashlib
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


# Lowercase indexes preserve SQLite's case-insensitive uniqueness.
_POSTGRES_SCHEMA = _SCHEMA.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT",
    "INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY",
).replace(" COLLATE NOCASE", "") + """
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower ON users (LOWER(email));
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON users (LOWER(username));
"""
_AUTH_SCHEMA_LOCK_ID = 0x4149534F43415554  # Fixed, process-independent auth DDL lock.


class AuthStorageError(RuntimeError):
    """Safe storage failure without connection credentials or SQL details."""


class PostgresAuthStore:
    """Private PostgreSQL storage; importing this module never imports psycopg.

    query/query_one/execute accept only trusted internal auth SQL templates:
    '?' denotes a parameter, never a literal or operator. Values must be passed
    separately. Use transaction() for native PostgreSQL SQL and placeholders.
    """

    def __init__(self, database_url: str):
        try:
            import certifi
            import psycopg
            from psycopg.conninfo import conninfo_to_dict, make_conninfo
            from psycopg.rows import dict_row
        except (ImportError, OSError):
            raise AuthStorageError(
                "Private PostgreSQL authentication dependencies are unavailable."
            ) from None

        self._psycopg = psycopg
        self._dict_row = dict_row
        # Stage validation so a misconfigured value yields actionable guidance
        # without ever echoing the value itself (it contains credentials).
        if not isinstance(database_url, str) or not database_url.startswith(
                ("postgresql://", "postgres://")):
            raise AuthStorageError(
                "The database URL must start with postgresql:// - copy only the "
                "URI itself, without psql commands, surrounding quotes, or labels.")
        try:
            options = conninfo_to_dict(database_url)
        except (psycopg.Error, ValueError, TypeError):
            raise AuthStorageError(
                "The database URL could not be parsed - check the configured "
                "value for stray spaces, quotes, or line breaks.") from None
        # Require an explicit network host: libpq ignores TLS on sockets.
        hosts = options.get("host", "").split(",")
        if any(not host or host.startswith(("/", "@")) for host in hosts):
            raise AuthStorageError(
                "The database URL must include a network host name.")
        options["sslmode"] = "verify-full"
        options["sslrootcert"] = options.get("sslrootcert") or certifi.where()
        # GSS encryption otherwise takes precedence over libpq's TLS mode.
        options["gssencmode"] = "disable"
        options.setdefault("connect_timeout", "30")
        try:
            self._conninfo = make_conninfo(**options)
        except (psycopg.Error, ValueError, TypeError, OSError):
            raise AuthStorageError(
                "The database URL could not be parsed - check the configured "
                "value for stray spaces, quotes, or line breaks.") from None

        # CREATE IF NOT EXISTS alone is not safe against concurrent catalog
        # creation. Serialize all DDL across processes until this txn commits.
        with self.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)",
                         (_AUTH_SCHEMA_LOCK_ID,))
            for statement in _POSTGRES_SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)

    @contextmanager
    def transaction(self):
        """Yield a native psycopg connection with dict rows and '%s' parameters.

        Commit on success, roll back on failure, and always close the connection.
        Driver/connection failures are sanitized, including commit/rollback errors.
        """
        try:
            conn = self._psycopg.connect(
                self._conninfo, row_factory=self._dict_row)
            try:
                yield conn
                conn.commit()
            except BaseException:
                # Handle rollback ourselves: the driver's connection context
                # manager can log raw diagnostics if rollback also fails.
                conn.rollback()
                raise
            finally:
                conn.close()
        except (self._psycopg.Error, OSError, ValueError, TypeError):
            raise AuthStorageError(
                "Private PostgreSQL authentication storage operation failed."
            ) from None

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.transaction() as conn:
            return conn.execute(sql.replace("?", "%s"), params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> dict | None:
        with self.transaction() as conn:
            return conn.execute(sql.replace("?", "%s"), params).fetchone()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.transaction() as conn:
            return conn.execute(sql.replace("?", "%s"), params).rowcount


_STORES: dict[str, AuthStore | PostgresAuthStore] = {}


def get_store() -> AuthStore | PostgresAuthStore:
    """Read configuration at call time; cache per backend/target per process.

    Configured PostgreSQL fails closed. Its URL is never a literal cache key.
    """
    database_url = auth_settings.AUTH_DATABASE_URL
    if database_url:
        key = "postgresql:" + hashlib.sha256(database_url.encode("utf-8")).hexdigest()
        if key not in _STORES:
            _STORES[key] = PostgresAuthStore(database_url)
        return _STORES[key]

    path = Path(auth_settings.AUTH_DB_PATH)
    key = str(path.resolve())
    if key not in _STORES:
        store = AuthStore(path)
        store.init_pragmas()
        _STORES[key] = store
    return _STORES[key]


def reset_store_registry() -> None:
    _STORES.clear()