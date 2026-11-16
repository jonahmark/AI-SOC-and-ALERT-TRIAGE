"""Shared PostgreSQL connectivity for private deployment storage.

Importing this module never imports psycopg: the driver is loaded when a
store is constructed, so SQLite-only installs need no PostgreSQL dependency.
Connections require verified TLS, every value is bound as a parameter, and
failures are sanitized so credentials and SQL never reach a log or the UI.
"""
from __future__ import annotations

from contextlib import contextmanager


class StorageError(RuntimeError):
    """Safe storage failure without connection credentials or SQL details."""


class PostgresStore:
    """Private PostgreSQL storage with verified TLS and parameter binding.

    query/query_one/execute accept only trusted internal SQL templates:
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
            raise StorageError(
                "Private PostgreSQL dependencies are unavailable."
            ) from None

        self._psycopg = psycopg
        self._dict_row = dict_row
        # Stage validation so a misconfigured value yields actionable guidance
        # without ever echoing the value itself (it contains credentials).
        if not isinstance(database_url, str) or not database_url.startswith(
                ("postgresql://", "postgres://")):
            raise StorageError(
                "The database URL must start with postgresql:// - copy only the "
                "URI itself, without psql commands, surrounding quotes, or labels.")
        try:
            options = conninfo_to_dict(database_url)
        except (psycopg.Error, ValueError, TypeError):
            raise StorageError(
                "The database URL could not be parsed - check the configured "
                "value for stray spaces, quotes, or line breaks.") from None
        # Require an explicit network host: libpq ignores TLS on sockets.
        hosts = options.get("host", "").split(",")
        if any(not host or host.startswith(("/", "@")) for host in hosts):
            raise StorageError(
                "The database URL must include a network host name.")
        options["sslmode"] = "verify-full"
        options["sslrootcert"] = options.get("sslrootcert") or certifi.where()
        # GSS encryption otherwise takes precedence over libpq's TLS mode.
        options["gssencmode"] = "disable"
        options.setdefault("connect_timeout", "30")
        try:
            self._conninfo = make_conninfo(**options)
        except (psycopg.Error, ValueError, TypeError, OSError):
            raise StorageError(
                "The database URL could not be parsed - check the configured "
                "value for stray spaces, quotes, or line breaks.") from None

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
            raise StorageError(
                "Private PostgreSQL storage operation failed."
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