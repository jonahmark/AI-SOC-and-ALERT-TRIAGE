"""Tests for the validated, persisted application settings."""
from __future__ import annotations

import json

import pytest

import config
from src import settings as S


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Never read or write the real data/processed files during tests."""
    monkeypatch.setattr(S, "SETTINGS_FILE", tmp_path / "app_settings.json")
    monkeypatch.setattr(S, "LOG_FILE", tmp_path / "activity.log")
    return tmp_path


# ------------------------------------------------------------------ defaults
def test_defaults_come_from_config():
    base = S.defaults()
    assert base["severity_map"] == config.SEVERITY_MAP
    assert base["risk_weights"] == config.RISK_WEIGHTS
    assert base["shap_top_k"] == config.SHAP_LOCAL_TOP_K
    assert base["anomaly_threshold"] is None      # model default, not a guess
    assert sum(base["risk_weights"].values()) == pytest.approx(1.0)


def test_defaults_are_not_shared_mutable_state():
    a, b = S.defaults(), S.defaults()
    a["severity_map"]["DoS"] = "Low"
    assert b["severity_map"]["DoS"] == config.SEVERITY_MAP["DoS"]


def test_validate_defaults_has_no_errors():
    clean, errors = S.validate(S.defaults())
    assert errors == []
    assert clean == S.defaults()


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("payload,fragment", [
    ({"dataset": "KDDCup99"}, "Dataset must be one of"),
    ({"classifier": "deep_learning"}, "Classifier must be one of"),
    ({"anomaly_threshold": "abc"}, "Anomaly threshold must be a number"),
    ({"anomaly_threshold": -99.0}, "Anomaly threshold must be between"),
    ({"severity_map": {"DoS": "Apocalyptic"}}, "Severity for 'DoS'"),
    ({"severity_map": []}, "Severity mapping must be a mapping"),
    ({"risk_weights": {"severity": 0.9, "confidence": 0.9, "anomaly": 0.9}},
     "must sum to exactly 1.0"),
    ({"risk_weights": {"severity": -0.5, "confidence": 1.0, "anomaly": 0.5}},
     "Risk weight 'severity' must be between"),
    ({"risk_weights": "heavy"}, "Risk weights must be a mapping"),
    ({"shap_top_k": 0}, "SHAP top-K must be between"),
    ({"shap_global_sample_size": 5}, "SHAP global sample size must be between"),
    ({"table_page_size": 2}, "Table page size must be between"),
    ({"shap_top_k": "many"}, "SHAP top-K must be a whole number"),
])
def test_invalid_values_are_rejected_with_a_message(payload, fragment):
    _, errors = S.validate(payload)
    assert errors and any(fragment in e for e in errors), errors


def test_non_mapping_payload_is_rejected():
    clean, errors = S.validate(["dataset"])
    assert errors and clean == S.defaults()


@pytest.mark.parametrize("value", [None, "", "auto"])
def test_threshold_auto_means_model_default(value):
    clean, errors = S.validate({"anomaly_threshold": value})
    assert errors == []
    assert clean["anomaly_threshold"] is None


def test_threshold_override_is_rounded_and_kept():
    clean, errors = S.validate({"anomaly_threshold": -0.40487654321})
    assert errors == []
    assert clean["anomaly_threshold"] == pytest.approx(-0.404877)


def test_partial_payload_fills_documented_defaults():
    clean, errors = S.validate({"classifier": "random_forest"})
    assert errors == []
    assert clean["classifier"] == "random_forest"
    assert clean["severity_map"] == config.SEVERITY_MAP


def test_alternative_weights_and_severity_map_are_accepted():
    payload = {
        "severity_map": {**config.SEVERITY_MAP, "Probe": "High"},
        "risk_weights": {"severity": 0.6, "confidence": 0.2, "anomaly": 0.2},
    }
    clean, errors = S.validate(payload)
    assert errors == []
    assert clean["severity_map"]["Probe"] == "High"
    assert clean["risk_weights"]["severity"] == 0.6


# --------------------------------------------------------------- persistence
def test_save_load_roundtrip(isolated_settings):
    payload = {**S.defaults(), "classifier": "random_forest", "table_page_size": 50}
    saved, errors = S.save(payload)
    assert errors == []
    assert saved["saved_at"]
    assert S.SETTINGS_FILE.exists()
    loaded = S.load()
    assert loaded["classifier"] == "random_forest"
    assert loaded["table_page_size"] == 50
    assert loaded["saved_at"] == saved["saved_at"]


def test_invalid_save_writes_nothing(isolated_settings):
    _, errors = S.save({"classifier": "not_a_model"})
    assert errors
    assert not S.SETTINGS_FILE.exists()
    assert S.load()["classifier"] == S.defaults()["classifier"]


def test_corrupt_file_falls_back_to_defaults(isolated_settings):
    S.SETTINGS_FILE.write_text("{not json", encoding="utf-8")
    assert S.load() == S.defaults()
    assert any("unreadable" in line for line in S.read_log())


def test_stored_invalid_values_fall_back_and_are_logged(isolated_settings):
    S.SETTINGS_FILE.write_text(json.dumps({"classifier": "bogus"}),
                               encoding="utf-8")
    loaded = S.load()
    assert loaded["classifier"] == S.defaults()["classifier"]
    assert any("invalid" in line for line in S.read_log())


def test_reset_removes_the_file(isolated_settings):
    S.save(S.defaults())
    assert S.SETTINGS_FILE.exists()
    assert S.reset() == S.defaults()
    assert not S.SETTINGS_FILE.exists()


# ------------------------------------------------------------------- helpers
def test_effective_threshold_prefers_the_override():
    assert S.effective_threshold({"anomaly_threshold": None}, -0.4) == -0.4
    assert S.effective_threshold({"anomaly_threshold": -0.9}, -0.4) == -0.9


def test_settings_drive_severity_and_risk():
    settings = {**S.defaults(),
                "severity_map": {**config.SEVERITY_MAP, "Probe": "Critical"}}
    engine = S.severity_engine(settings)
    assert engine.severity_for("Probe") == "Critical"
    assert engine.severity_for("DoS") == config.SEVERITY_MAP["DoS"]

    generator = S.alert_generator(settings)
    assert generator.risk_weights == config.RISK_WEIGHTS
    assert generator.severity_engine.severity_for("Probe") == "Critical"


def test_settings_diff_reports_only_real_changes():
    assert S.settings_diff(S.defaults()) == []
    diffs = S.settings_diff({**S.defaults(), "classifier": "random_forest"})
    assert len(diffs) == 1 and "classifier" in diffs[0]


def test_dataset_path_maps_to_the_real_files():
    assert S.dataset_path({"dataset": "KDDTest+"}) == config.TEST_FILE
    assert S.dataset_path({"dataset": "KDDTrain+"}) == config.TRAIN_FILE


# --------------------------------------------------------------------- log
def test_log_is_capped_and_timestamped(isolated_settings, monkeypatch):
    monkeypatch.setattr(S, "MAX_LOG_LINES", 5)
    for i in range(12):
        S.log(f"event {i}")
    lines = S.read_log(100)
    assert len(lines) == 5
    assert lines[-1].endswith("event 11")
    assert "T" in lines[0] and "+" in lines[0]      # ISO UTC stamp


def test_read_log_missing_file_is_empty(isolated_settings):
    assert S.read_log() == []