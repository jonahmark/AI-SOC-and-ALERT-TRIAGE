from unittest.mock import MagicMock

import certifi
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from src.auth import db, settings


@pytest.fixture
def postgres_connection(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    return connection, connect


def test_local_storage_remains_sqlite():
    store = db.get_store()
    assert isinstance(store, db.AuthStore)
    assert store.path == settings.AUTH_DB_PATH
    assert db.get_store() is store


def test_cloud_storage_selected_without_creating_local_database(monkeypatch):
    stores = []

    def factory(url):
        store = object()
        stores.append(store)
        return store

    monkeypatch.setattr(db, "PostgresAuthStore", factory)
    monkeypatch.setattr(settings, "AUTH_DATABASE_URL", "postgresql://user:private@db.example.test/accounts")
    first = db.get_store()
    assert first is db.get_store()
    assert not settings.AUTH_DB_PATH.exists()
    assert all("private" not in str(key) for key in db._STORES)
    monkeypatch.setattr(settings, "AUTH_DATABASE_URL", "postgresql://user:other@db.example.test/accounts")
    assert db.get_store() is not first
    assert len(stores) == 2


def test_postgres_forces_verified_tls_and_dictionary_rows(postgres_connection):
    _, connect = postgres_connection
    db.PostgresAuthStore("postgresql://user:private@db.example.test/accounts?sslmode=disable")
    options = conninfo_to_dict(connect.call_args.args[0])
    assert options["sslmode"] == "verify-full"
    assert options["sslrootcert"] == certifi.where()
    assert options["gssencmode"] == "disable"
    assert connect.call_args.kwargs["row_factory"] is dict_row
    assert int(options["connect_timeout"]) > 0


def test_postgres_accepts_an_operator_ca_certificate(postgres_connection):
    _, connect = postgres_connection
    db.PostgresAuthStore("postgresql://user:private@db.example.test/accounts?sslrootcert=/tmp/test-ca.pem")
    options = conninfo_to_dict(connect.call_args.args[0])
    assert options["sslrootcert"] == "/tmp/test-ca.pem"
    assert options["sslmode"] == "verify-full"


def test_postgres_preserves_parameter_binding(postgres_connection):
    connection, _ = postgres_connection
    store = db.PostgresAuthStore("postgresql://user:private@db.example.test/accounts")
    connection.execute.return_value.fetchall.return_value = [{"n": 7}]
    assert store.query("SELECT ? AS n", (7,)) == [{"n": 7}]
    connection.execute.assert_called_with("SELECT %s AS n", (7,))
    connection.execute.return_value.fetchone.return_value = {"n": 8}
    assert store.query_one("SELECT ? AS n", (8,)) == {"n": 8}
    connection.execute.return_value.rowcount = 1
    assert store.execute("DELETE FROM sessions WHERE session_id = ?", ("test-token-digest",)) == 1
    connection.execute.assert_called_with("DELETE FROM sessions WHERE session_id = %s", ("test-token-digest",))


def test_postgres_connection_failure_is_redacted_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_DATABASE_URL", "postgresql://user:private@db.example.test/accounts")
    monkeypatch.setattr(psycopg, "connect", MagicMock(side_effect=psycopg.OperationalError("password=private")))
    with pytest.raises(db.AuthStorageError) as caught:
        db.get_store()
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert not settings.AUTH_DB_PATH.exists()
    assert not db._STORES


def test_postgres_query_failure_does_not_expose_values(postgres_connection):
    connection, _ = postgres_connection
    store = db.PostgresAuthStore("postgresql://user:private@db.example.test/accounts")
    connection.execute.side_effect = psycopg.errors.UniqueViolation("password_hash=private-hash")
    with pytest.raises(db.AuthStorageError) as caught:
        store.execute("INSERT INTO users (user_id) VALUES (?)", ("duplicate",))
    assert "private-hash" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("url", ["", "sqlite:///accounts.db", "not-a-database-url"])
def test_invalid_postgres_configuration_is_rejected_without_connection(url, postgres_connection):
    _, connect = postgres_connection
    with pytest.raises(db.AuthStorageError):
        db.PostgresAuthStore(url)
    connect.assert_not_called()
