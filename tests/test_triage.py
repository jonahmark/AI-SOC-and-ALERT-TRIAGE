"""Triage state backends: JSON files, shared PostgreSQL, and backend selection.

The database backend's SQL is executed for real against SQLite, which offers
the same query/query_one/execute contract and '?' placeholders as PostgresStore,
so upserts, note ordering and parameter binding are verified rather than
asserted on mocks. Selection, DDL bootstrap, verified TLS and failure redaction
are verified against a mocked psycopg connection. No test contacts a real
database, and the alert rows below are synthetic persistence fixtures, not
model output.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pandas as pd
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from src import triage
from src.alert_generator import ALERT_COLUMNS, AlertStore
from src.postgres import StorageError

URL = "postgresql://user:private@db.example.test/triage"
CREATED_AT = "2026-11-15T10:00:00+00:00"


def _alerts(*ids: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"alert_id": alert_id, "timestamp": CREATED_AT, "record_index": i,
         "detection_status": "Anomaly", "predicted_category": "DoS",
         "true_label": "", "true_category": "", "severity": "High",
         "risk_score": 80.0, "model_confidence": 0.9, "anomaly_score": -0.6,
         "explanation": "fixture row", "recommended_action": "fixture action",
         "investigation_status": "New"}
        for i, alert_id in enumerate(ids)], columns=ALERT_COLUMNS)


def _note(alert_id: str, text: str, author: str = "analyst-1") -> dict:
    return {"alert_id": alert_id, "author": author, "text": text,
            "created_at": CREATED_AT}


class SqliteTriageStore:
    """PostgresStore's query contract on SQLite, running the same SQL."""

    def __init__(self, path):
        # AppTest renders pages on a worker thread; production PostgreSQL opens
        # a connection per operation and has no such thread affinity.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(triage._TRIAGE_SCHEMA)

    def query(self, sql, params=()):
        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def execute(self, sql, params=()):
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return cursor.rowcount


@pytest.fixture
def sql_store(tmp_path):
    return SqliteTriageStore(tmp_path / "triage.db")


@pytest.fixture
def state(sql_store):
    return triage.DatabaseTriageState(sql_store)


@pytest.fixture
def db_alerts(tmp_path, state, monkeypatch):
    """An AlertStore whose workflow state lives in the database backend."""
    monkeypatch.setattr(triage, "get_state", lambda *a, **k: state)
    return AlertStore(alerts_file=tmp_path / "alerts.csv",
                      status_file=tmp_path / "statuses.json",
                      notes_file=tmp_path / "notes.json")


@pytest.fixture
def postgres_connection(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    return connection, connect


# ------------------------------------------------------------------ SQL layer
def test_status_upsert_keeps_one_row_per_alert(state):
    state.set_status("ALT-1", "Investigating")
    state.set_status("ALT-1", "Resolved")
    state.set_status("ALT-2", "New")
    assert state.statuses() == {"ALT-1": "Resolved", "ALT-2": "New"}


def test_notes_append_in_insertion_order_with_the_file_entry_shape(state):
    state.add_note("ALT-1", _note("ALT-1", "first"))
    state.add_note("ALT-1", _note("ALT-1", "second"))
    assert state.notes("ALT-1") == [_note("ALT-1", "first"),
                                    _note("ALT-1", "second")]
    assert state.note_counts() == {"ALT-1": 2}
    assert state.notes("ALT-UNKNOWN") == []
    assert state.note_counts() == {"ALT-1": 2}


def test_non_ascii_note_text_round_trips(state):
    text = "Traffic analysé – 确认 ✓"
    state.add_note("ALT-1", _note("ALT-1", text))
    assert state.notes("ALT-1")[0]["text"] == text


def test_values_are_bound_not_interpolated(state, sql_store):
    hostile = "ALT-'; DROP TABLE alert_notes;--"
    state.set_status(hostile, "Resolved")
    state.add_note(hostile, _note(hostile, "'); DELETE FROM alert_statuses;--",
                                  author="a'\""))
    assert state.statuses() == {hostile: "Resolved"}
    assert state.notes(hostile)[0]["text"] == "'); DELETE FROM alert_statuses;--"
    assert sql_store.query("SELECT COUNT(*) AS n FROM alert_notes")[0]["n"] == 1
    assert sql_store.query("SELECT COUNT(*) AS n FROM alert_statuses")[0]["n"] == 1


def test_shared_state_is_never_pruned_or_cleared(state):
    state.set_status("ALT-1", "Investigating")
    state.add_note("ALT-1", _note("ALT-1", "kept"))
    state.retain(set())
    state.clear()
    assert state.statuses() == {"ALT-1": "Investigating"}
    assert state.note_counts() == {"ALT-1": 1}


def test_two_instances_share_the_same_state(sql_store):
    first = triage.DatabaseTriageState(sql_store)
    second = triage.DatabaseTriageState(sql_store)
    first.set_status("ALT-1", "Resolved")
    first.add_note("ALT-1", _note("ALT-1", "shared"))
    assert second.statuses() == {"ALT-1": "Resolved"}
    assert second.notes("ALT-1") == first.notes("ALT-1")


# ------------------------------------------------------- AlertStore on the DB
def test_store_writes_to_the_database_and_creates_no_json(db_alerts, state):
    db_alerts.save(_alerts("ALT-1", "ALT-2"), merge=False)
    db_alerts.update_status("ALT-1", "Investigating")
    entry = db_alerts.add_note("ALT-1", "Checked the source bytes.",
                               author="analyst-1")
    assert not db_alerts.status_file.exists()
    assert not db_alerts.notes_file.exists()
    loaded = db_alerts.load().set_index("alert_id")
    assert loaded.loc["ALT-1", "investigation_status"] == "Investigating"
    assert loaded.loc["ALT-2", "investigation_status"] == "New"
    assert db_alerts.get_notes("ALT-1") == [entry]
    assert db_alerts.note_counts() == {"ALT-1": 1}
    assert state.statuses() == {"ALT-1": "Investigating"}


def test_regenerating_alerts_keeps_shared_triage_history(db_alerts, state):
    db_alerts.save(_alerts("ALT-OLD"), merge=False)
    db_alerts.update_status("ALT-OLD", "Resolved")
    db_alerts.add_note("ALT-OLD", "Closed after review.")
    db_alerts.clear()
    assert not db_alerts.alerts_file.exists()
    db_alerts.save(_alerts("ALT-NEW"), merge=False)
    assert state.statuses() == {"ALT-OLD": "Resolved"}
    assert state.note_counts() == {"ALT-OLD": 1}
    fresh = db_alerts.load()
    assert fresh["alert_id"].tolist() == ["ALT-NEW"]
    assert fresh["investigation_status"].tolist() == ["New"]
    assert db_alerts.get_notes("ALT-NEW") == []


def test_clearing_one_queue_leaves_another_instance_intact(tmp_path, state,
                                                          monkeypatch):
    monkeypatch.setattr(triage, "get_state", lambda *a, **k: state)
    local = AlertStore(alerts_file=tmp_path / "local.csv",
                       status_file=tmp_path / "local-status.json",
                       notes_file=tmp_path / "local-notes.json")
    hosted = AlertStore(alerts_file=tmp_path / "hosted.csv",
                        status_file=tmp_path / "hosted-status.json",
                        notes_file=tmp_path / "hosted-notes.json")
    local.save(_alerts("ALT-1"), merge=False)
    hosted.save(_alerts("ALT-1"), merge=False)
    hosted.update_status("ALT-1", "Investigating")
    hosted.add_note("ALT-1", "Hosted analyst note.")
    local.clear()
    local.save(_alerts("ALT-2"), merge=False)
    assert hosted.load().set_index("alert_id").loc[
        "ALT-1", "investigation_status"] == "Investigating"
    assert hosted.get_notes("ALT-1")[0]["text"] == "Hosted analyst note."
    assert local.load()["alert_id"].tolist() == ["ALT-2"]


@pytest.mark.parametrize("status", ["Deleted", ""])
def test_invalid_status_is_rejected(db_alerts, state, status):
    db_alerts.save(_alerts("ALT-1"), merge=False)
    with pytest.raises(ValueError):
        db_alerts.update_status("ALT-1", status)
    assert state.statuses() == {}


def test_unknown_alert_is_rejected(db_alerts, state):
    db_alerts.save(_alerts("ALT-1"), merge=False)
    with pytest.raises(KeyError):
        db_alerts.update_status("ALT-NOPE", "Resolved")
    with pytest.raises(KeyError):
        db_alerts.add_note("ALT-NOPE", "orphan note")
    assert state.statuses() == {}
    assert state.note_counts() == {}


@pytest.mark.parametrize("text", ["", "   ", "x" * 4001])
def test_invalid_note_is_rejected(db_alerts, state, text):
    db_alerts.save(_alerts("ALT-1"), merge=False)
    with pytest.raises(ValueError):
        db_alerts.add_note("ALT-1", text)
    assert state.note_counts() == {}


# ------------------------------------------------------------------ selection
def test_no_url_selects_json_files(tmp_path):
    selected = triage.get_state(tmp_path / "statuses.json",
                                tmp_path / "notes.json")
    assert isinstance(selected, triage.FileTriageState)
    assert not selected.shared
    assert selected.status_file == tmp_path / "statuses.json"
    assert selected.notes_file == tmp_path / "notes.json"


def test_url_resolution_prefers_triage_then_falls_back_to_auth(monkeypatch):
    monkeypatch.setenv("AI_SOC_AUTH_DATABASE_URL",
                       "postgresql://u:p@db.example.test/auth")
    monkeypatch.setenv("AI_SOC_TRIAGE_DATABASE_URL", "")
    assert triage._configured_url() == "postgresql://u:p@db.example.test/auth"
    monkeypatch.setenv("AI_SOC_TRIAGE_DATABASE_URL",
                       " postgresql://u:p@db.example.test/triage ")
    assert triage._configured_url() == "postgresql://u:p@db.example.test/triage"
    monkeypatch.setenv("AI_SOC_AUTH_DATABASE_URL", "")
    monkeypatch.setenv("AI_SOC_TRIAGE_DATABASE_URL", "")
    assert triage._configured_url() == ""


def test_database_backend_bootstraps_schema_over_verified_tls(
        monkeypatch, postgres_connection, tmp_path):
    connection, connect = postgres_connection
    monkeypatch.setattr(triage, "TRIAGE_DATABASE_URL", URL)
    selected = triage.get_state(tmp_path / "statuses.json",
                                tmp_path / "notes.json")
    assert isinstance(selected, triage.DatabaseTriageState)
    assert selected.shared
    options = conninfo_to_dict(connect.call_args.args[0])
    assert options["sslmode"] == "verify-full"
    assert options["gssencmode"] == "disable"
    executed = [call.args[0] for call in connection.execute.call_args_list]
    assert executed[0] == "SELECT pg_advisory_xact_lock(%s)"
    assert connection.execute.call_args_list[0].args[1] == (
        triage._TRIAGE_SCHEMA_LOCK_ID,)
    assert any("CREATE TABLE IF NOT EXISTS alert_statuses" in sql
               for sql in executed)
    assert any("CREATE TABLE IF NOT EXISTS alert_notes" in sql
               for sql in executed)
    # SQLite cannot check the PostgreSQL dialect, so assert the identity
    # column directly: AUTOINCREMENT would be rejected by PostgreSQL.
    ddl = "\n".join(executed)
    assert "GENERATED BY DEFAULT AS IDENTITY" in ddl
    assert "AUTOINCREMENT" not in ddl
    assert not (tmp_path / "statuses.json").exists()
    assert not (tmp_path / "notes.json").exists()


def test_database_state_is_cached_per_target_without_the_secret(
        monkeypatch, postgres_connection, tmp_path):
    _, connect = postgres_connection
    monkeypatch.setattr(triage, "TRIAGE_DATABASE_URL", URL)
    first = triage.get_state(tmp_path / "statuses.json",
                             tmp_path / "notes.json")
    connections = connect.call_count
    assert triage.get_state(tmp_path / "statuses.json",
                            tmp_path / "notes.json") is first
    assert connect.call_count == connections
    assert all("private" not in key for key in triage._STATES)
    monkeypatch.setattr(triage, "TRIAGE_DATABASE_URL",
                        URL.replace("/triage", "/other"))
    assert triage.get_state() is not first
    assert len(triage._STATES) == 2


def test_database_failure_is_redacted_and_never_falls_back_to_json(
        monkeypatch, tmp_path):
    monkeypatch.setattr(triage, "TRIAGE_DATABASE_URL", URL)
    monkeypatch.setattr(psycopg, "connect", MagicMock(
        side_effect=psycopg.OperationalError("password=private")))
    with pytest.raises(StorageError) as caught:
        triage.get_state(tmp_path / "statuses.json", tmp_path / "notes.json")
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert not (tmp_path / "statuses.json").exists()
    assert not (tmp_path / "notes.json").exists()
    assert not triage._STATES


def test_write_failure_does_not_expose_values(monkeypatch, postgres_connection):
    connection, _ = postgres_connection
    monkeypatch.setattr(triage, "TRIAGE_DATABASE_URL", URL)
    selected = triage.get_state()
    connection.execute.side_effect = psycopg.errors.UniqueViolation(
        "note_text=private-note")
    with pytest.raises(StorageError) as caught:
        selected.set_status("ALT-1", "Resolved")
    assert "private-note" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("url", [
    "sqlite:///triage.db",
    "not-a-database-url",
    "psql 'postgresql://u:p@db.example.test/triage'",
    "postgresql:///triage",
])
def test_invalid_url_is_rejected_without_connecting(url, postgres_connection):
    _, connect = postgres_connection
    with pytest.raises(StorageError):
        triage.PostgresTriageStore(url)
    connect.assert_not_called()