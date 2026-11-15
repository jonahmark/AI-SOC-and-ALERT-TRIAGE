"""Migration safety tests: real SQLite source, simulated PostgreSQL destination.

The destination is an in-memory stand-in for the psycopg connection API, so
these tests verify the migration's decisions (source validation, the
empty-destination rule, MFA re-encryption, count verification, rollback)
without a network database. Real-database behaviour is covered separately
by tests/test_auth_storage.py and the operator's verified-TLS connection.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from argon2 import PasswordHasher
from cryptography.fernet import Fernet, InvalidToken
from psycopg import sql

import config
from src.auth import migration
from src.auth.db import AuthStore

ADMIN_PW = "Adm1n!Str0ng-Pass"
VIEWER_PW = "V3ry!Strong-Pass"
SOURCE_TOTP_KEY = Fernet.generate_key().decode("utf-8")
DEST_TOTP_KEY = Fernet.generate_key().decode("utf-8")
TOTP_PLAINTEXT = b"JBSWY3DPEHPK3PXP"
DEST_URL = "postgresql://user:private@db.example.test/accounts"
NOW = "2026-11-01T00:00:00+00:00"

_USER_INSERT = (
    "INSERT INTO users (user_id, full_name, email, username, organisation, "
    "password_hash, role, email_verified, account_status, auth_epoch, "
    "created_at, updated_at, last_login_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")


@pytest.fixture
def source_db(tmp_path) -> "AuthStore":
    store = AuthStore(tmp_path / "source.db")
    hasher = PasswordHasher()
    store.execute(_USER_INSERT, (
        "u-admin", "Test Admin", "Admin@Example.Test", "Admin", "",
        hasher.hash(ADMIN_PW), "administrator", 1, "active", 1, NOW, NOW, None))
    store.execute(_USER_INSERT, (
        "u-viewer", "Vera Viewer", "viewer@example.test", "viewer", "",
        hasher.hash(VIEWER_PW), "viewer", 0, "active", 1, NOW, NOW, None))
    store.execute(
        "INSERT INTO password_history (user_id, password_hash, created_at) "
        "VALUES (?,?,?)", ("u-admin", hasher.hash(ADMIN_PW), NOW))
    store.execute(
        "INSERT INTO password_history (user_id, password_hash, created_at) "
        "VALUES (?,?,?)", ("u-admin", hasher.hash("Old1!Pass-Phrase"), NOW))
    store.execute(
        "INSERT INTO totp_credentials (user_id, secret_encrypted, enabled, "
        "enrolled_at, created_at) VALUES (?,?,?,?,?)",
        ("u-admin", Fernet(SOURCE_TOTP_KEY.encode()).encrypt(TOTP_PLAINTEXT).decode(),
         1, NOW, NOW))
    store.execute(
        "INSERT INTO totp_credentials (user_id, secret_encrypted, enabled, "
        "enrolled_at, created_at) VALUES (?,?,?,?,?)",
        ("u-viewer", Fernet(SOURCE_TOTP_KEY.encode()).encrypt(b"UNUSED").decode(),
         0, None, NOW))
    return store


class _FakeCursor:
    def __init__(self, destination):
        self._destination = destination

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, statement, rows):
        text = statement.as_string(None)
        table = text.split("INSERT INTO", 1)[1].split("(", 1)[0].strip().strip('"')
        self._destination.tables[table].extend(rows)


class FakeDestination:
    """Minimal psycopg-connection stand-in covering the migration's SQL."""

    def __init__(self, *, occupied_table=None, bad_counts=False):
        self.tables = {table: [] for table in migration._AUTH_TABLES}
        self.bad_counts = bad_counts
        self.committed = False
        self.rolled_back = False
        self.statements = []
        if occupied_table:
            self.tables[occupied_table].append(("pre-existing",))

    @contextmanager
    def transaction(self):
        try:
            yield self
            self.committed = True
        except BaseException:
            self.rolled_back = True
            raise

    def execute(self, statement, params=None):
        text = (statement.as_string(None)
                if isinstance(statement, sql.Composable) else str(statement))
        self.statements.append(text)
        result = MagicMock()
        if text.startswith("SELECT EXISTS"):
            result.fetchone.return_value = {"occupied": any(self.tables.values())}
        elif text.startswith("SELECT COUNT(*)"):
            table = text.split("FROM", 1)[1].strip().strip('"')
            count = len(self.tables[table])
            result.fetchone.return_value = {"n": count + 1 if self.bad_counts else count}
        elif text.startswith("INSERT INTO audit_events"):
            self.tables["audit_events"].append(params)
        return result

    def cursor(self):
        return _FakeCursor(self)


@pytest.fixture
def install_destination(monkeypatch):
    def install(destination):
        monkeypatch.setattr(migration, "PostgresAuthStore", lambda url: destination)
        return destination
    return install


def _migrate(source, destination, **keys):
    options = {"source_totp_key": SOURCE_TOTP_KEY,
               "destination_totp_key": DEST_TOTP_KEY}
    options.update(keys)
    return migration.migrate_accounts(source, DEST_URL, **options)


def test_inspection_is_read_only_and_counts(source_db):
    before = hashlib.sha256(source_db.path.read_bytes()).hexdigest()
    counts = migration.inspect_accounts(source_db.path)
    assert counts == {"users": 2, "password_history": 2, "totp_credentials": 1}
    assert hashlib.sha256(source_db.path.read_bytes()).hexdigest() == before


def test_migrate_imports_accounts_and_audits(source_db, install_destination):
    destination = install_destination(FakeDestination())
    counts = _migrate(source_db.path, destination)
    assert counts == {"users": 2, "password_history": 2, "totp_credentials": 1}
    assert destination.committed and not destination.rolled_back
    columns = migration._COLUMNS["users"]
    users = {row[0]: dict(zip(columns, row)) for row in destination.tables["users"]}
    assert users["u-admin"]["email"] == "admin@example.test"
    assert users["u-admin"]["username"] == "admin"
    assert users["u-admin"]["role"] == "administrator"
    hasher = PasswordHasher()
    assert hasher.verify(users["u-admin"]["password_hash"], ADMIN_PW)
    assert hasher.verify(users["u-viewer"]["password_hash"], VIEWER_PW)
    assert len(destination.tables["password_history"]) == 2
    for table in ("sessions", "reset_tokens", "verify_tokens",
                  "totp_challenges", "login_attempts"):
        assert destination.tables[table] == []
    audit = destination.tables["audit_events"]
    assert len(audit) == 1
    assert audit[0][0] == "accounts_migrated"
    assert "accounts=2" in audit[0][1] and "mfa=1" in audit[0][1]


def test_mfa_secrets_are_reencrypted_for_destination(source_db, install_destination):
    destination = install_destination(FakeDestination())
    _migrate(source_db.path, destination)
    rows = destination.tables["totp_credentials"]
    assert len(rows) == 1  # the disabled enrolment is not copied
    record = dict(zip(migration._COLUMNS["totp_credentials"], rows[0]))
    assert record["user_id"] == "u-admin" and record["enabled"] == 1
    assert Fernet(DEST_TOTP_KEY.encode()).decrypt(
        record["secret_encrypted"].encode()) == TOTP_PLAINTEXT
    with pytest.raises(InvalidToken):
        Fernet(SOURCE_TOTP_KEY.encode()).decrypt(record["secret_encrypted"].encode())


@pytest.mark.parametrize("occupied_table", ["users", "sessions", "audit_events"])
def test_occupied_destination_is_refused_without_overwrite(
        source_db, install_destination, occupied_table):
    destination = install_destination(FakeDestination(occupied_table=occupied_table))
    with pytest.raises(migration.MigrationError, match="already contains"):
        _migrate(source_db.path, destination)
    assert destination.rolled_back and not destination.committed
    assert destination.tables[occupied_table] == [("pre-existing",)]
    assert destination.tables["password_history"] == []


def test_count_mismatch_rolls_back(source_db, install_destination):
    destination = install_destination(FakeDestination(bad_counts=True))
    with pytest.raises(migration.MigrationError, match="rolled back"):
        _migrate(source_db.path, destination)
    assert destination.rolled_back and not destination.committed


@pytest.mark.parametrize("source_key,destination_key", [
    ("", DEST_TOTP_KEY),
    (Fernet.generate_key().decode("utf-8"), DEST_TOTP_KEY),
    (SOURCE_TOTP_KEY, ""),
])
def test_mfa_key_problems_fail_before_any_write(
        source_db, install_destination, source_key, destination_key):
    destination = install_destination(FakeDestination())
    with pytest.raises(migration.MigrationError, match="MFA keys"):
        _migrate(source_db.path, destination,
                 source_totp_key=source_key, destination_totp_key=destination_key)
    assert destination.statements == []
    assert not destination.committed and not destination.rolled_back


def test_orphaned_tokens_in_discarded_tables_do_not_block(source_db):
    # A deleted account can leave single-use tokens behind (FK enforcement is
    # per-connection in SQLite). Those tables are never copied, so they must
    # not block the transfer of the live accounts.
    import sqlite3
    conn = sqlite3.connect(source_db.path)  # FK enforcement off by default
    conn.execute(
        "INSERT INTO reset_tokens (token_hash, user_id, created_at, expires_at) "
        "VALUES (?,?,?,?)", ("orphan-reset", "USR-deleted", NOW, NOW))
    conn.execute(
        "INSERT INTO verify_tokens (token_hash, user_id, created_at, expires_at) "
        "VALUES (?,?,?,?)", ("orphan-verify", "USR-deleted", NOW, NOW))
    conn.commit()
    conn.close()
    counts = migration.inspect_accounts(source_db.path)
    assert counts == {"users": 2, "password_history": 2, "totp_credentials": 1}


def test_orphaned_password_history_blocks(tmp_path):
    import sqlite3
    store = AuthStore(tmp_path / "orphan.db")
    store.execute(_USER_INSERT, (
        "u1", "Real User", "real@example.test", "real", "",
        PasswordHasher().hash(ADMIN_PW), "viewer", 0, "active", 1, NOW, NOW, None))
    conn = sqlite3.connect(store.path)
    conn.execute(
        "INSERT INTO password_history (user_id, password_hash, created_at) "
        "VALUES (?,?,?)", ("USR-absent", PasswordHasher().hash(ADMIN_PW), NOW))
    conn.commit()
    conn.close()
    with pytest.raises(migration.MigrationError, match="inconsistent references"):
        migration.inspect_accounts(store.path)


def test_missing_source_rejected(tmp_path):
    with pytest.raises(migration.MigrationError, match="does not exist"):
        migration.inspect_accounts(tmp_path / "absent.db")


def test_empty_source_rejected(tmp_path):
    store = AuthStore(tmp_path / "empty.db")
    with pytest.raises(migration.MigrationError, match="no accounts"):
        migration.inspect_accounts(store.path)


def test_non_argon2id_hash_rejected(tmp_path):
    store = AuthStore(tmp_path / "bad.db")
    store.execute(
        "INSERT INTO users (user_id, full_name, email, username, password_hash, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("u1", "Bad Hash", "bad@example.test", "bad", "not-a-hash", NOW, NOW))
    with pytest.raises(migration.MigrationError, match="Argon2id"):
        migration.inspect_accounts(store.path)


def test_cli_default_is_inspection_only(source_db):
    script = config.PROJECT_ROOT / "scripts" / "migrate_accounts.py"
    result = subprocess.run(
        [sys.executable, str(script), "--source", str(source_db.path)],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "2 accounts" in result.stdout
    assert "Inspection only" in result.stdout


def test_cli_apply_without_credentials_fails_safely(source_db):
    script = config.PROJECT_ROOT / "scripts" / "migrate_accounts.py"
    result = subprocess.run(
        [sys.executable, str(script), "--source", str(source_db.path), "--apply"],
        capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
    assert result.returncode == 1
    assert "AI_SOC_MIGRATION_DATABASE_URL" in result.stderr
    assert "Import could not be confirmed" in result.stderr