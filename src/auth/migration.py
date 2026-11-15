"""Operator-only account transfer from local SQLite to empty PostgreSQL storage."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from argon2 import Type, extract_parameters
from argon2.exceptions import InvalidHashError
from cryptography.fernet import Fernet, InvalidToken

from src.auth.db import PostgresAuthStore


class MigrationError(ValueError):
    pass


_COLUMNS = {
    "users": (
        "user_id", "full_name", "email", "username", "organisation",
        "password_hash", "role", "email_verified", "account_status",
        "auth_epoch", "created_at", "updated_at", "last_login_at",
    ),
    "password_history": ("user_id", "password_hash", "created_at"),
    "totp_credentials": (
        "user_id", "secret_encrypted", "enabled", "enrolled_at", "created_at",
    ),
}
_AUTH_TABLES = (
    "users", "sessions", "reset_tokens", "verify_tokens", "password_history",
    "totp_credentials", "totp_challenges", "login_attempts", "audit_events",
)


def _read_source(source: Path) -> dict[str, list[dict]]:
    path = Path(source).resolve()
    if not path.is_file():
        raise MigrationError("The source account database does not exist.")
    conn = None
    try:
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise MigrationError("The source database failed its integrity check.")
        # Scope referential integrity to the tables being transferred. Orphaned
        # rows in discarded tables (e.g. single-use tokens left behind by a
        # deleted account) are not copied and must not block the migration.
        for table in _COLUMNS:
            if conn.execute(f"PRAGMA foreign_key_check({table})").fetchone() is not None:
                raise MigrationError("The source accounts have inconsistent references.")
        rows = {}
        for table, columns in _COLUMNS.items():
            suffix = " ORDER BY id" if table == "password_history" else ""
            if table == "totp_credentials":
                suffix = " WHERE enabled = 1"
            query = f"SELECT {', '.join(columns)} FROM {table}{suffix}"
            rows[table] = [dict(row) for row in conn.execute(query)]
    except sqlite3.Error:
        raise MigrationError("The source database could not be read; nothing was imported.") from None
    finally:
        if conn is not None:
            conn.close()
    if not rows["users"]:
        raise MigrationError("The source database contains no accounts.")
    identifiers = {"email": set(), "username": set()}
    for user in rows["users"]:
        for column, seen in identifiers.items():
            user[column] = user[column].strip().lower()
            if user[column] in seen:
                raise MigrationError("Source accounts contain conflicting login identifiers.")
            seen.add(user[column])
    for row in rows["users"] + rows["password_history"]:
        try:
            if extract_parameters(row["password_hash"]).type != Type.ID:
                raise InvalidHashError
        except (InvalidHashError, TypeError, ValueError):
            raise MigrationError("A source password hash is not valid Argon2id.") from None
    return rows


def inspect_accounts(source: Path) -> dict[str, int]:
    return {table: len(rows) for table, rows in _read_source(source).items()}


def migrate_accounts(source: Path, database_url: str, *,
                     source_totp_key: str = "", destination_totp_key: str = "") -> dict[str, int]:
    rows = _read_source(source)
    if rows["totp_credentials"]:
        try:
            source_cipher = Fernet(source_totp_key.encode("utf-8"))
            destination_cipher = Fernet(destination_totp_key.encode("utf-8"))
            for credential in rows["totp_credentials"]:
                secret = source_cipher.decrypt(credential["secret_encrypted"].encode("utf-8"))
                credential["secret_encrypted"] = destination_cipher.encrypt(secret).decode("utf-8")
        except (ValueError, TypeError, InvalidToken):
            raise MigrationError(
                "MFA keys are missing, invalid, or cannot decrypt the source enrolments. "
                "Configure the destination key in the cloud before importing."
            ) from None

    from psycopg import sql

    destination = PostgresAuthStore(database_url)
    counts = {table: len(entries) for table, entries in rows.items()}
    with destination.transaction() as conn:
        conn.execute("SET LOCAL lock_timeout = '5s'")
        conn.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
            sql.SQL(", ").join(sql.Identifier(table) for table in _AUTH_TABLES)))
        occupied = conn.execute(sql.SQL("SELECT EXISTS ({}) AS occupied").format(
            sql.SQL(" UNION ALL ").join(
                sql.SQL("SELECT 1 FROM {}").format(sql.Identifier(table))
                for table in _AUTH_TABLES
            ))).fetchone()["occupied"]
        if occupied:
            raise MigrationError(
                "The destination already contains authentication data. "
                "Nothing was overwritten; use a new empty database."
            )
        for table, columns in _COLUMNS.items():
            statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table),
                sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            )
            with conn.cursor() as cursor:
                cursor.executemany(statement, [tuple(row[c] for c in columns) for row in rows[table]])
            count = conn.execute(sql.SQL("SELECT COUNT(*) AS n FROM {}").format(
                sql.Identifier(table))).fetchone()["n"]
            if count != counts[table]:
                raise MigrationError("Destination verification failed; the import was rolled back.")
        conn.execute(
            "INSERT INTO audit_events (event, detail, created_at) VALUES (%s, %s, %s)",
            ("accounts_migrated", f"accounts={counts['users']} mfa={counts['totp_credentials']}",
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    return counts
