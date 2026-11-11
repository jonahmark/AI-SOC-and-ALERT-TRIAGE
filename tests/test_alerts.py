"""Tests for alert generation, the alert store, and triage workflow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.alert_generator import (
    ALERT_COLUMNS, AlertGenerator, AlertStore,
)
from src.anomaly_detection import AnomalyResult
from src.attack_classification import ClassificationResult
from src.severity_engine import SeverityEngine


def _results(n=10):
    """8 flagged records (mixed reasons) + 2 benign-and-normal records."""
    scores = np.array([-0.9, -0.7, -0.6, -0.5, -0.5, -0.4, -0.2, -0.1,
                       0.1, 0.2])          # threshold -0.45 => first 8 anom.
    is_anom = scores <= config.ANOMALY_THRESHOLD
    preds = np.array(["DoS", "U2R", "Probe", "R2L", "Normal", "DoS",
                      "Probe", "Normal", "Normal", "Normal"])
    probs = np.zeros((n, 5))
    classes = config.ATTACK_CATEGORIES
    for i, p in enumerate(preds):
        probs[i, classes.index(p)] = 0.9
        probs[i, (classes.index(p) + 1) % 5] = 0.1
    anomaly = AnomalyResult(scores=scores, is_anomaly=is_anom,
                            threshold=config.ANOMALY_THRESHOLD)
    classification = ClassificationResult(
        predictions=preds, probabilities=probs,
        confidences=probs.max(axis=1), classes=classes)
    return anomaly, classification


def test_alerts_generated_for_flagged_records_only():
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    # records 0-4: anomalous -> alerts (regardless of category)
    # records 5-6: not anomalous but predicted attack -> alerts
    # records 7-9: normal score AND predicted Normal -> no alert
    assert len(alerts) == 7
    assert set(alerts["record_index"]) == set(range(7))


def test_alert_fields_complete_and_unique():
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    assert list(alerts.columns) == ALERT_COLUMNS
    assert alerts["alert_id"].is_unique
    assert alerts["severity"].isin(config.SEVERITY_ORDER).all()
    assert (alerts["risk_score"] >= 0).all()
    assert (alerts["risk_score"] <= 100).all()
    assert alerts["investigation_status"].eq("New").all()
    assert alerts["explanation"].str.len().gt(20).all()
    assert alerts["recommended_action"].str.len().gt(20).all()


def test_alerts_sorted_by_risk_desc():
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    assert alerts["risk_score"].is_monotonic_decreasing


def test_u2r_alert_outranks_probe():
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    u2r = alerts[alerts["predicted_category"] == "U2R"].iloc[0]
    probe = alerts[alerts["predicted_category"] == "Probe"].iloc[0]
    assert u2r["risk_score"] > probe["risk_score"]
    assert u2r["severity"] == "Critical"


def test_true_labels_recorded_when_provided():
    anomaly, classification = _results()
    labels = np.array(["neptune"] * 10)
    alerts = AlertGenerator().generate(anomaly, classification,
                                       true_labels=labels)
    assert (alerts["true_label"] == "neptune").all()
    assert (alerts["true_category"] == "DoS").all()


def test_generate_length_mismatch_raises():
    anomaly, classification = _results()
    anomaly_short = AnomalyResult(scores=anomaly.scores[:5],
                                  is_anomaly=anomaly.is_anomaly[:5],
                                  threshold=anomaly.threshold)
    with pytest.raises(ValueError, match="align"):
        AlertGenerator().generate(anomaly_short, classification)


# ------------------------------------------------------------- store
@pytest.fixture
def store(tmp_path):
    return AlertStore(alerts_file=tmp_path / "alerts.csv",
                      status_file=tmp_path / "statuses.json",
                      notes_file=tmp_path / "notes.json")


def test_store_save_load_roundtrip(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    loaded = store.load()
    assert len(loaded) == len(alerts)
    assert set(loaded["alert_id"]) == set(alerts["alert_id"])


def test_store_merge_deduplicates(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    store.save(alerts, merge=True)
    assert len(store.load()) == len(alerts)


def test_store_load_empty(tmp_path):
    store = AlertStore(alerts_file=tmp_path / "nope.csv",
                       status_file=tmp_path / "nope.json")
    assert store.load().empty


def test_update_status_persists(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    alert_id = alerts.iloc[0]["alert_id"]
    store.update_status(alert_id, "Investigating")
    loaded = store.load()
    assert loaded[loaded["alert_id"] == alert_id
                  ].iloc[0]["investigation_status"] == "Investigating"
    store.update_status(alert_id, "False Positive")
    assert store.load()[store.load()["alert_id"] == alert_id
                        ].iloc[0]["investigation_status"] == "False Positive"


def test_update_status_invalid(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    with pytest.raises(ValueError):
        store.update_status(alerts.iloc[0]["alert_id"], "Deleted")
    with pytest.raises(KeyError):
        store.update_status("ALT-NOPE", "Resolved")


def test_filter_alerts(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    loaded = store.load()
    crit = AlertStore.filter_alerts(loaded, severity="Critical")
    assert len(crit) == 1 and crit.iloc[0]["predicted_category"] == "U2R"
    dos = AlertStore.filter_alerts(loaded, category="DoS")
    assert (dos["predicted_category"] == "DoS").all()
    found = AlertStore.filter_alerts(
        loaded, search=alerts.iloc[0]["alert_id"][:8])
    assert len(found) >= 1
    asc = AlertStore.filter_alerts(loaded, sort_by="risk_score",
                                   ascending=True)
    assert asc["risk_score"].is_monotonic_increasing


def test_clear(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    store.clear()
    assert store.load().empty


# ------------------------------------------------------------- analyst notes
@pytest.fixture
def stored_alerts(store):
    anomaly, classification = _results()
    alerts = AlertGenerator().generate(anomaly, classification)
    store.save(alerts, merge=False)
    return alerts


def test_add_note_persists_with_author_and_timestamp(store, stored_alerts):
    alert_id = stored_alerts.iloc[0]["alert_id"]
    entry = store.add_note(alert_id, "Confirmed scanning behaviour.",
                           author="analyst-1")
    assert entry["text"] == "Confirmed scanning behaviour."
    assert entry["author"] == "analyst-1"
    assert entry["created_at"].endswith("+00:00")
    notes = store.get_notes(alert_id)
    assert len(notes) == 1 and notes[0] == entry
    assert store.note_counts() == {alert_id: 1}
    # a fresh store instance reads the same persisted notes
    other = AlertStore(alerts_file=store.alerts_file,
                       status_file=store.status_file,
                       notes_file=store.notes_file)
    assert other.get_notes(alert_id) == notes


def test_notes_append_in_order(store, stored_alerts):
    alert_id = stored_alerts.iloc[0]["alert_id"]
    store.add_note(alert_id, "first")
    store.add_note(alert_id, "second")
    assert [n["text"] for n in store.get_notes(alert_id)] == ["first", "second"]
    assert store.note_counts()[alert_id] == 2


def test_notes_survive_non_ascii_text(store, stored_alerts):
    alert_id = stored_alerts.iloc[0]["alert_id"]
    store.add_note(alert_id, "Traffic analysé – 确认 \u2713")
    assert store.get_notes(alert_id)[0]["text"] == "Traffic analysé – 确认 \u2713"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_notes_are_rejected(store, stored_alerts, text):
    with pytest.raises(ValueError, match="must not be empty"):
        store.add_note(stored_alerts.iloc[0]["alert_id"], text)
    assert store.note_counts() == {}


def test_overlong_note_is_rejected(store, stored_alerts):
    with pytest.raises(ValueError, match="4000 characters"):
        store.add_note(stored_alerts.iloc[0]["alert_id"], "x" * 4001)


def test_note_for_unknown_alert_is_rejected(store, stored_alerts):
    with pytest.raises(KeyError):
        store.add_note("ALT-DOES-NOT-EXIST", "orphan note")
    assert store.note_counts() == {}


def test_notes_for_removed_alerts_are_pruned(store, stored_alerts):
    keep = stored_alerts.iloc[0]["alert_id"]
    drop = stored_alerts.iloc[1]["alert_id"]
    store.add_note(keep, "keep me")
    store.add_note(drop, "drop me")
    store.save(stored_alerts.iloc[[0]], merge=False)
    assert store.get_notes(keep)[0]["text"] == "keep me"
    assert store.get_notes(drop) == []
    assert store.note_counts() == {keep: 1}


def test_pruning_all_notes_removes_the_file(store, stored_alerts):
    store.add_note(stored_alerts.iloc[0]["alert_id"], "temporary")
    assert store.notes_file.exists()
    store.save(stored_alerts.iloc[:0], merge=False)
    assert not store.notes_file.exists()


def test_clear_removes_notes(store, stored_alerts):
    store.add_note(stored_alerts.iloc[0]["alert_id"], "investigated")
    store.clear()
    assert not store.notes_file.exists()
    assert store.note_counts() == {}


def test_corrupt_notes_file_does_not_crash(store, stored_alerts):
    store.notes_file.write_text("{ truncated", encoding="utf-8")
    assert store.get_notes(stored_alerts.iloc[0]["alert_id"]) == []
    assert store.note_counts() == {}
    store.add_note(stored_alerts.iloc[0]["alert_id"], "recovered")
    assert len(store.get_notes(stored_alerts.iloc[0]["alert_id"])) == 1


def test_notes_and_status_are_independent(store, stored_alerts):
    alert_id = stored_alerts.iloc[0]["alert_id"]
    store.add_note(alert_id, "escalating")
    store.update_status(alert_id, "Investigating")
    assert store.load().set_index("alert_id").loc[
        alert_id, "investigation_status"] == "Investigating"
    assert store.get_notes(alert_id)[0]["text"] == "escalating"
