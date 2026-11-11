"""Tests for the disk persistence of a completed analysis.

The cache directory is sandboxed to a temporary folder by an autouse fixture,
so these tests never touch the project's real restart-survival cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import analysis_cache as ac
from src import settings as S
from src.pipeline import AnalysisResult, StageResult


def _result() -> AnalysisResult:
    records = pd.DataFrame({
        "record_index": np.arange(4),
        "detection_status": ["Normal", "Anomaly", "Anomaly", "Normal"],
        "anomaly_score": [-0.1, -0.8, -0.6, 0.2],
        "predicted_category": ["Normal", "DoS", "Probe", "Normal"],
        "model_confidence": [0.9, 0.8, 0.7, 0.95],
        "true_label": ["normal", "neptune", "satan", "normal"],
        "true_category": ["Normal", "DoS", "Probe", "Normal"],
        "labelled": [True, True, True, True],
        "severity": ["Informational", "High", "Medium", "Informational"],
        "risk_score": [10.0, 80.0, 55.0, 12.0],
    })
    alerts = pd.DataFrame({
        "alert_id": ["ALT-A", "ALT-B"],
        "record_index": [1, 2],
        "predicted_category": ["DoS", "Probe"],
        "severity": ["High", "Medium"],
        "risk_score": [80.0, 55.0],
        "investigation_status": ["New", "New"],
    })
    stages = [StageResult("Data ingestion", "done", 0.1, "4 records"),
              StageResult("Explanation generation", "skipped", None, "not requested")]
    meta = {"n_records": 4, "n_anomalous": 2, "classifier": "xgboost",
            "anomaly_threshold": -0.45, "total_seconds": 1.5,
            "ran_at": "2026-11-10T12:00:00+00:00"}
    return AnalysisResult(records=records, alerts=alerts, stages=stages,
                          meta=meta)


def test_roundtrip_preserves_frames_stages_and_meta():
    result = _result()
    key = ac.cache_key(S.defaults(), model_fingerprint="fp-1")
    ac.save(result, key)
    loaded = ac.load(key)
    assert loaded is not None and loaded.ok
    pd.testing.assert_frame_equal(loaded.records, result.records)
    pd.testing.assert_frame_equal(loaded.alerts, result.alerts)
    assert loaded.meta == result.meta
    assert [(s.name, s.status, s.seconds, s.detail) for s in loaded.stages] == \
        [(s.name, s.status, s.seconds, s.detail) for s in result.stages]
    assert loaded.explanations == {}


def test_load_with_wrong_key_returns_none():
    ac.save(_result(), ac.cache_key(S.defaults(), model_fingerprint="fp-1"))
    assert ac.load(ac.cache_key(S.defaults(), model_fingerprint="fp-2")) is None


def test_settings_change_invalidates_the_key():
    base = S.defaults()
    k0 = ac.cache_key(base)
    assert ac.cache_key({**base, "classifier": "random_forest"}) != k0
    assert ac.cache_key({**base, "anomaly_threshold": -0.9}) != k0
    assert ac.cache_key({**base, "dataset": "KDDTrain+"}) != k0
    assert ac.cache_key(base, max_records=1000) != k0
    assert ac.cache_key(base, explain_top_n=5) != k0


def test_model_fingerprint_change_invalidates_the_key():
    base = S.defaults()
    assert ac.cache_key(base, model_fingerprint="a") != \
        ac.cache_key(base, model_fingerprint="b")


def test_load_missing_returns_none():
    assert ac.load("no-such-key") is None


def test_corrupt_json_returns_none():
    ac.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (ac.CACHE_DIR / "result.json").write_text("{ truncated", encoding="utf-8")
    assert ac.load("any") is None


def test_missing_parquet_returns_none():
    ac.save(_result(), "k")
    (ac.CACHE_DIR / "records.parquet").unlink()
    assert ac.load("k") is None


def test_failed_result_is_not_persisted():
    bad = _result()
    bad.errors.append("preprocessing failed")
    ac.save(bad, "k")
    assert ac.load("k") is None


def test_clear_removes_the_cache():
    ac.save(_result(), "k")
    assert ac.load("k") is not None
    ac.clear()
    assert ac.load("k") is None
    assert not ac.CACHE_DIR.exists()


def test_model_fingerprint_reads_trained_at():
    fp = ac.model_fingerprint()
    if config.PIPELINE_STATE_FILE.exists():
        import json
        state = json.loads(config.PIPELINE_STATE_FILE.read_text())
        assert fp == str(state.get("trained_at") or "")
    else:
        assert fp == ""